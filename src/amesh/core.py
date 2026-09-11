import random
from collections.abc import Callable
from types import MethodType

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from pygmo import (
    crowding_distance,  # type: ignore
    fast_non_dominated_sorting,  # type: ignore
    select_best_N_mo,  # type: ignore
)
from tqdm import tqdm

from .auxiliar import PreAllocated, StoppingAlgorithm
from .operations.differential_crossover import get_differential_crossover
from .operations.differential_mutation import get_differential_mutation
from .operations.differential_mutation_pool import get_differential_mutation_pool
from .operations.global_guide_method import get_global_guide_method
from .parameters import AMESHParameters
from .particles import Memory, Population
from .validations.python_validations import assert_type, is_function, is_greater_in_type


class AMESH:
    '''A-MESH algorithm.
    
    Args:
        params (:class:`~amesh.parameters.AMESHParameters`): A-MESH parameters.
        fitness_function (:type:`Callable[[NDArray[np.number]], NDArray[np.number]]`): A fitness function that returns a numpy vector with each objective value in the respective component.
        log_memory (:type:`str | None`): Path used to log the memory. If ``None``, memory positions and fitness values are not written to a file.
        num_proc (:type:`int | None`): Number of processes used to evaluate the fitness function. If ``None``, evaluations are performed sequentially.
    
    Raises:
        TypeError: If the input is not the expected type.
        ValueError: If the input is not the allowed value.
    
    Note:
        Using parallel evaluations is only advantageous if :attr:`fitness_function` is sufficiently computationally expensive.
    '''

    def __init__(self,
                params: AMESHParameters,
                fitness_function: Callable[[NDArray[np.number]], NDArray[np.number]],
                log_memory: str | None = None,
                num_proc: int | None = None):
        
        self.params: AMESHParameters
        '''A-MESH parameters.'''
        self.global_guide_method: Callable[[], None]
        ''' Function to find the global guides for the particles. '''
        self.differential_mutation_pool: Callable[[], tuple[NDArray[np.number], list[NDArray[np.intp]]]]
        '''Function that builds the pool from which solutions are sampled for differential mutation.'''
        self.differential_mutation: Callable[[tuple[NDArray[np.number], list[NDArray[np.intp]]]], tuple[NDArray[np.number], NDArray[np.intp]]]
        ''' Function to do the Differential Mutation operation. '''
        self.differential_crossover: Callable[[NDArray[np.number], NDArray[np.number], NDArray[np.number]], NDArray[np.number]]
        ''' Function to do the Differential Crossover operation. '''
        self.population: Population = Population(params)
        ''' Population of particles. '''
        self.memory: Memory = Memory(params)
        ''' Memory of particles. '''
        self.fitness_function: Callable[[NDArray[np.number]], NDArray[np.number]]
        ''' Fitness function. '''
        self.generation_counter: int
        '''Number of generations initialized or completed during the current run.'''
        self.fitness_eval_counter: int
        '''Number of objective-function evaluations performed during the current run.'''
        self.pre_allocated: PreAllocated
        ''' Pre-allocated data for the algorithm. '''
        self.log_memory: str | None
        '''Path used to log memory positions and fitness values, or ``None`` to disable logging.'''
        self.num_proc: int | None
        '''Number of processes used for fitness evaluations, or ``None`` for sequential execution.'''
        self.evaluation_way: Callable[[NDArray[np.number]], NDArray[np.number]]
        '''Fitness-evaluation strategy. Evaluations use :attr:`num_proc` processes when that attribute is not ``None``; otherwise, they are sequential.'''
        self.evaluate: Callable[[NDArray[np.number]], NDArray[np.number]]
        '''Fitness-evaluation function. Evaluations are counted when :attr:`~amesh.parameters.AMESHParameters.max_fit_eval` is not ``None``.'''
        self.count_generation: Callable[[], None]
        ''' Function to count generations. Only used if :attr:`~amesh.parameters.AMESHParameters.max_gen` is not None. '''
        self.algorithm_progress: int = 0
        ''' Current algorithm progress counter. '''
        self.max_algorithm_progress: int
        ''' Maximum algorithm progress value. '''
        self.update_algorithm_progress: Callable[[tqdm, int], int]
        ''' Function to update the algorithm progress. '''

        # Receive the algorithm parameters
        assert_type(params, 'params', AMESHParameters)
        self.params = params
        # Chosing the operations just one time
        self.global_guide_method = MethodType(get_global_guide_method(params.global_guide_method), self)
        self.differential_mutation_pool = MethodType(get_differential_mutation_pool(params.dm_pool_type), self)
        self.differential_mutation = MethodType(get_differential_mutation(params.dm_operation_type), self)
        self.differential_crossover = MethodType(get_differential_crossover('binomial'), self)
        # Use a random seed if there is
        np.random.seed(params.random_state)
        random.seed(params.random_state)
        # Estabilish the fitness function
        is_function(fitness_function, 'fitness_function')
        self.fitness_function = fitness_function
        # Start the generation counter (considering the initial generation)
        self.generation_counter = 1
        # Start the fitness evaluation counter
        self.fitness_eval_counter = 0
        # Store some pre-calculated data
        self.pre_allocated = PreAllocated(params)
        # Variable for logging memory
        assert_type(log_memory, 'log_memory', str, is_optional=True)
        self.log_memory = log_memory
        # Check if the fitness evaluation will be sequential or parallel
        is_greater_in_type(num_proc, 'num_proc', int, 0, is_optional=True)
        self.num_proc = num_proc
        if self.num_proc is not None:
            self.evaluation_way = self.parallel_fitness_evaluation
        else:
            self.evaluation_way = self.sequential_fitness_evaluation
        # Check if generation is a stopping criterion
        if params.max_gen > 0:
            self.count_generation = self.stopping_by_generation
        else:
            self.count_generation = lambda : None
        # Check if the fitness evaluation is a stopping criterion
        if params.max_fit_eval > 0:
            self.evaluate = self.stopping_by_fitness_evaluation
        else:
            self.evaluate = self.evaluation_way
        # Choose the way to update the algorithm progress
        if params.max_gen == 0:
            self.max_algorithm_progress = params.max_fit_eval
            self.update_algorithm_progress = self.update_progress_by_fitness_evaluation
        elif params.max_fit_eval == 0:
            self.max_algorithm_progress = params.max_gen
            self.update_algorithm_progress = self.update_progress_by_generation
        else:
            self.update_algorithm_progress = self.update_progress_by_fitness_evaluation
            self.max_algorithm_progress = min(params.population_size*(2*params.max_gen+1), params.max_fit_eval)

    def initialize(self):
        '''Initializes A-MESH by creating the population, memory, and personal-guide fitness values, evaluating the initial population, and calculating the non-dominated fronts.'''

        # Evaluate the initial population
        self.population.fitness[:] = self.evaluate(self.population.position)
        # Update A-MESH memory
        self.update_amesh_memory()
        # Repeat the population fitness for all personal guide input
        self.population.personal_guide_fit[:, :, :] = np.repeat(self.population.fitness[:, np.newaxis, :], self.params.max_personal_guides, axis=1)

    def sequential_fitness_evaluation(self, X: NDArray[np.number]) -> NDArray[np.number]:
        '''Evaluates the fitness of a particle-position matrix sequentially.
        
        Args:
            X (:type:`NDArray[np.number]`): A numpy matrix with the particle positions.

        Returns:
            :type:`NDArray[np.number]`: The fitness matrix associated with the particle positions.
        '''

        decision_varibles = X[:, :self.params.decision_dim]
        return np.array([self.fitness_function(x) for x in decision_varibles])

    def parallel_fitness_evaluation(self, X: NDArray[np.number]) -> NDArray[np.number]:
        '''Evaluates the fitness of a particle-position matrix in parallel.
        
        Args:
            X (:type:`NDArray[np.number]`): A numpy matrix with the particle positions.

        Returns:
            :type:`NDArray[np.number]`: The fitness matrix associated with the particle positions.
        '''
        
        decision_varibles = X[:, :self.params.decision_dim]
        # Create a pool of processes to execute the fitness function parallelly
        fitness_values = Parallel(n_jobs=self.num_proc)(delayed(self.fitness_function)(x) for x in decision_varibles)
        return np.array(fitness_values)
    
    def dominates(self, Fx: NDArray[np.number], Fy: NDArray[np.number], axis: int = 0) -> NDArray[np.bool]:
        r''' Checks if the domination condition for the numpy arrays ``Fx`` and ``Fy`` with fitness values are satisfied on the respective ``axis``.
        
        Note:
            Given two decision vectors :math:`x,\ y \in \mathbb{R}^m` and :math:`F(x) = [f_1(x),\ \ldots,\ f_n(x)]^T` as the fitness function of the multi-objective optimization problem, :math:`x` dominates :math:`y` if and only if the following condition are satisfied:

            .. math::
            
                F(x) \neq F(y)\ \land\ F(x) \preceq F(y),
            
            where:

            .. math::

                F(x) \neq F(y) &\iff \exists i \in \{1,\ \ldots,\ n\}\ (\ f_i(x) \neq f_i(y)), \\
                F(x) \preceq F(y) &\iff \forall i \in \{1,\ \ldots,\ n\}\ (f_i(x) \leq f_i(y)).
        
        Args:
            Fx (:type:`NDArray[np.number]`): A n-dimensional numpy array with fitness values.
            Fy (:type:`NDArray[np.number]`): A n-dimensional numpy array with fitness values.
            axis (:type:`int`): The axis to compare the arrays. Default is 0.
        
        Returns:
            :type:`NDArray[np.bool]`: A n-dimensional numpy array with the result of the comparison.
        '''

        return np.all(Fx <= Fy, axis=axis) & np.any(Fx < Fy, axis=axis)

    def get_non_domination_fronts(self, fitness_matrix: NDArray[np.number]) -> list[NDArray[np.intp]]:
        ''' Get the non-domination fronts of the particles given their fitness values. The fronts are calculated by the Fast Non-dominated Sorting algorithm from Pygmo.
        
        Note:
            The fronts are a list of numpy arrays. Each numpy array in the list represents a front, starting with the Pareto front. Each particle has its own index.

        Args:
            fitness_matrix (:type:`NDArray[np.number]`): A numpy matrix with the fitness values of the particles.

        Returns:
            :type:`list[NDArray[np.intp]]`: A list of numpy arrays, each representing a non-dominated front.
        '''

        # If there is only one particle in the particle list, then it is the Pareto front by itself
        if(len(fitness_matrix) == 1):
            return [np.array([0])]
        # Do the Fast Non-dominated Sorting from Pygmo
        non_dominated_fronts, _, _, _ = fast_non_dominated_sorting(points=fitness_matrix)
        return non_dominated_fronts

    def update_de_memory(self,
                         pop_promising_position: NDArray[np.number],
                         survival_position: NDArray[np.number],
                         pop_promising_idxs: NDArray[np.intp]) -> None:
        r''' Updates the DE historical means using successful offspring. The scaling factor ``F`` is updated using a weighted Lehmer mean, while the crossover rate ``CR`` is updated using a weighted arithmetic mean.

        Args:
            pop_promising_position (:type:`NDArray[np.number]`): Matrix containing the original decision vectors associated with the successful offspring. Its shape must be ``(n_success, decision_dim)``.
            survival_position (:type:`NDArray[np.number]`): Matrix containing the surviving offspring decision vectors corresponding to ``position``. Its shape must be ``(n_success, decision_dim)``.
            pop_promising_idxs (:type:`NDArray[np.intp]`): Indices of the particles from population that generated successful offspring. These indices are used to retrieve the successful ``F`` and ``CR`` values.
        '''

        if len(pop_promising_idxs) == 0:
            return
        # Get successful values of F and CR
        successful_F = self.params.DE_F[pop_promising_idxs]
        successful_CR = self.params.DE_CR[pop_promising_idxs]
        # Calculate the weight based on Euclidean distance between the position and the survival position
        normalized_step = (pop_promising_position - survival_position) / (self.params.decision_upper_bounds - self.params.decision_lower_bounds)
        euclidean_distance = np.linalg.norm(normalized_step, axis=1)
        distance_sum = np.sum(euclidean_distance)
        if distance_sum > 0:
            weight = euclidean_distance / distance_sum
        else:
            weight = np.full(len(euclidean_distance), 1.0 / len(euclidean_distance))
        # Weighted SHADE-style Lehmer mean for F
        mean_F = np.sum((successful_F ** 2) * weight) / np.sum(successful_F * weight)
        # Weighted arithmetic mean for CR
        mean_CR = np.sum(successful_CR * weight)
        # Update the DE memory
        k = self.params.hyperparameter_last_index
        self.params.DE_memory[k, 0] = mean_F
        self.params.DE_memory[k, 1] = mean_CR

    def sample_from_cauchy(self,
                           parameter: NDArray[np.floating],
                           loc: NDArray[np.floating],
                           scale: float,
                           bounds: tuple[float,float]) -> None:
        ''' Generate a array by sampling number from Cauchy distribution.
        
        Args:
            parameter (:type:`NDArray[np.floating]`): Output array where the sampled values will be stored (in-place).
            loc (:type:`NDArray[np.floating]`): Location values of the Cauchy distribution. Must be broadcastable to ``parameter.shape``.
            scale (:type:`float`): Scale parameter of the Cauchy distribution.
            bounds (:type:`tuple[float, float]`): Lower and upper bounds allowed for the sampled values.
        '''

        # Calculate adaptative parameter by sampling from Cauchy
        parameter[:] = loc + scale * np.random.standard_cauchy(size=parameter.shape)
        lower = bounds[0]
        invalid_mask = parameter <= lower
        while np.any(invalid_mask):
            parameter[invalid_mask] = loc[invalid_mask] + scale * np.random.standard_cauchy(np.count_nonzero(invalid_mask))
            invalid_mask = parameter <= lower
        # Safety fallback for rare pathological cases
        parameter[invalid_mask] = lower
        # SHADE-style truncation above the upper bound
        upper = bounds[1]
        parameter[parameter > upper] = upper

    def differential_evolution(self) -> None:
        r'''Generates solutions with Differential Evolution according to the mutation strategy selected by :attr:`~amesh.parameters.AMESHParameters.dm_operation_type`, using the sampling pool selected by :attr:`~amesh.parameters.AMESHParameters.dm_pool_type`. Elitist selection then replaces less promising members of the current population with successful generated solutions.
        
        Note:
            The criteria for the best elitism solutions are the same as those for the method :meth:`elitism`.
        '''

        # Calculate the DE parameter F
        random_index = np.random.randint(self.params.hyperparameter_memory_length, size=self.params.population_size)
        self.sample_from_cauchy(self.params.DE_F,
                                self.params.DE_memory[random_index, 0],
                                self.params.shade_scale,
                                (0., 1.))
        # Apply a differential mutation strategy
        Xst, applied_st_pop_idxs = self.differential_mutation(self.differential_mutation_pool())
        if len(Xst):
            # Calcualte the DE parameter CR
            self.params.DE_CR[:] = np.random.normal(loc=self.params.DE_memory[random_index, 1], scale=self.params.shade_scale)
            np.clip(self.params.DE_CR, 0, 1, out=self.params.DE_CR)
            # Apply the differential crossover
            population_size = self.params.population_size
            population_positions = self.population.position
            Xst_rec = self.differential_crossover(
                population_positions[applied_st_pop_idxs],
                Xst,
                self.params.DE_CR[applied_st_pop_idxs, np.newaxis]
            )
            # Update the current particle if the new particle from the strategy is better
            Fst_rec = self.evaluate(Xst_rec)
            # Concatenate the population fitness array with the strategy fitness array
            fitness_elitism = np.concatenate((self.population.fitness, Fst_rec), axis=0)
            # Find the best N indices
            best_N_idxs = select_best_N_mo(fitness_elitism, population_size)
            # Get the indices of the best particles in the strategy array
            mask_best = best_N_idxs >= population_size
            best_st_indices = best_N_idxs[mask_best] - population_size
            # Put the best particles from strategy array in the current population
            np.logical_not(mask_best, out=mask_best)
            mask_pop_worst = np.ones(population_size, dtype=bool)
            mask_pop_worst[best_N_idxs[mask_best]] = False
            worst_pop_idxs = np.flatnonzero(mask_pop_worst)
            population_positions[worst_pop_idxs] = Xst_rec[best_st_indices]
            self.population.fitness[worst_pop_idxs] = Fst_rec[best_st_indices]
            # Update the DE memory
            pop_promising_idxs = applied_st_pop_idxs[best_st_indices]
            self.update_de_memory(self.population.position[pop_promising_idxs],
                                  Xst[best_st_indices],
                                  pop_promising_idxs)

    def mutation(self) -> None:
        r''' Calculates the mutation of the global guides are done by the following equation:

        .. math::
                
            \tilde{x}_{gb} = x_{gb} + \tau_{mut} \cdot \vec{r}, \quad \vec{r} \sim \mathcal{N}(\vec{0}, I),
            
        where :math:`I` is the identity matrix.
        '''
        
        pop_size = self.params.population_size
        decision_dim = self.params.decision_dim
        # Calculate adaptative mutation rate
        random_index = np.random.randint(self.params.hyperparameter_memory_length, size=pop_size)
        self.sample_from_cauchy(
            self.params.SWARM_mutation_scale,
            self.params.SWARM_memory[random_index, 4],
            self.params.shade_scale,
            (0., 1.)
        )
        # Mutate the global guides
        bound_scale = self.params.decision_upper_bounds - self.params.decision_lower_bounds
        random_noise = np.random.normal(0, 1, (pop_size, decision_dim))
        np.clip(
            self.population.global_guide + self.params.SWARM_mutation_scale[:, np.newaxis] * random_noise * bound_scale,
            self.params.decision_lower_bounds,
            self.params.decision_upper_bounds,
            out=self.pre_allocated.global_guide_mutated
        )

    def update_swarm_memory(self,
                            pop_promising_position: NDArray[np.number],
                            survival_position: NDArray[np.number],
                            pop_promising_idxs: NDArray[np.intp]) -> None:
        r'''Updates the swarm historical means using successful offspring. The inertia, assimilation, and communication weights and the mutation rate are updated using a weighted Lehmer mean, while the communication probability is updated using a weighted arithmetic mean.
        
        Args:
            pop_promising_position (:type:`NDArray[np.number]`): Matrix containing the original decision vectors associated with the successful offspring. Its shape must be ``(n_success, decision_dim)``.
            survival_position (:type:`NDArray[np.number]`): Matrix containing the surviving offspring decision vectors corresponding to ``position``. Its shape must be ``(n_success, decision_dim)``.
            pop_promising_idxs (:type:`NDArray[np.intp]`): Indices of the particles from population that generated successful offspring. These indices are used to retrieve the successful ``F`` and ``CR`` values.
        '''

        if len(pop_promising_idxs) == 0:
            return
        # Get successful values of swarm hyperparameters
        successful_W = self.params.SWARM_W[pop_promising_idxs]
        successful_Pcom = self.params.SWARM_Pcom[pop_promising_idxs]
        successful_mutation_rate = self.params.SWARM_mutation_scale[pop_promising_idxs]
        # Calculate the weight based on Euclidean distance between the position and the survival position
        normalized_step = (pop_promising_position - survival_position) / (self.params.decision_upper_bounds - self.params.decision_lower_bounds)
        euclidean_distance = np.linalg.norm(normalized_step, axis=1)
        distance_sum = np.sum(euclidean_distance)
        if distance_sum > 0:
            weight = euclidean_distance / distance_sum
        else:
            weight = np.full(len(euclidean_distance), 1.0 / len(euclidean_distance))
        # SHADE-style Lehmer mean
        mean_W = np.sum((successful_W ** 2) * weight[:, np.newaxis], axis=0) / np.sum(successful_W * weight[:, np.newaxis], axis=0)
        mean_mutation_rate = np.sum((successful_mutation_rate ** 2) * weight) / np.sum(successful_mutation_rate * weight)
        # Arithmetic mean
        mean_Pcom = np.sum(successful_Pcom * weight)
        # Update the swarm memory
        k = self.params.hyperparameter_last_index
        self.params.SWARM_memory[k, 0:3] = mean_W
        self.params.SWARM_memory[k, 3] = mean_Pcom
        self.params.SWARM_memory[k, 4] = mean_mutation_rate

    def move_population(self) -> None:
        r'''Applies the A-MESH equation of motion to the copied particles:
        
        .. math::

            \begin{cases}
                v'^{(t)} = \tilde{w}_Iv^{(t)} + \tilde{w}_A(x_{pb} - x^{(t)}) + \tilde{w}_C C^{(t)} \times (\tilde{x}_{gb} - x^{(t)}), \\
                x''^{(t)} = x'^{(t)} + v'^{(t)},
            \end{cases}
        
        where:

        - :math:`v^{(t)}` is the velocity vector at time t;
        - :math:`x'^{(t)}` is the position vector at time t, after the differential mutation phase;
        - :math:`w_I` is the inertia weight;
        - :math:`w_A` is the assimilation weight;
        - :math:`w_C` is the cooperation weight;
        - :math:`C` is a binary diagonal matrix, called communication matrix. Given :math:`r_i \sim \mathcal{U}(0,\ 1)` a number sampled under a Uniform Distribution between 0 and 1 for each line of :math:`C` and :math:`\tau_{com}` calculated as a decision variable, :math:`C` is calculated by:

        .. math::

            C_{[i,\ j]} = \begin{cases} 1, & \text{if } (i = j) \land (r_i \leq \tau_{com}); \\ 0, & \text{otherwise}. \end{cases}

        - :math:`x_{pb}` is the personal guide vector of the particle;
        - :math:`x_{gb}` is the global guide vector of the particle.
        
        Note:
            In this implementation, the weights are calculated every generation by :meth:`mutation` and each particle has its own weight. Every particle has its communication matrix too.
        '''

        # Get the parameters
        params = self.params
        # Get the population size and the position dimension
        population_size = params.population_size
        # Generating random indices for each subarray
        pb_indices = np.random.randint(0, self.params.max_personal_guides, size=population_size)
        # Get matrix position of personal guides from personal guide list positions
        Xpb = self.population.personal_guide_pos[np.arange(population_size), pb_indices, :]
        # Get the global guide positions
        Xgb_mut = self.pre_allocated.global_guide_mutated
        # Get the positions, velocities and fitness of the population
        X_copy = self.pre_allocated.position_copy
        V_copy = self.pre_allocated.velocity_copy
        F_copy = self.pre_allocated.fitness_copy
        # Get the equation of motion parameters
        random_index = np.random.randint(self.params.hyperparameter_memory_length, size=self.params.population_size)
        self.sample_from_cauchy(
            self.params.SWARM_W,
            self.params.SWARM_memory[random_index, 0:3],
            self.params.shade_scale,
            (0., 1.)
        )
        W = self.params.SWARM_W
        self.params.SWARM_Pcom[:] = np.random.normal(loc=self.params.SWARM_memory[random_index, 3], scale=self.params.shade_scale)
        np.clip(self.params.SWARM_Pcom, 0, 1, out=self.params.SWARM_Pcom)
        C = np.random.rand(population_size, params.decision_dim) <= self.params.SWARM_Pcom[:, np.newaxis]
        # Calculate the new velocity
        V_copy[:] = W[:, 0:1] * V_copy + W[:, 1:2] * (Xpb - X_copy) + W[:, 2:3] * C * (Xgb_mut - X_copy)
        # Calculate the clipped velocity
        np.clip(V_copy, params.velocity_lower_bounds, params.velocity_upper_bounds, out=V_copy)
        # Calculate the clipped position
        X_copy += V_copy
        np.clip(X_copy, params.decision_lower_bounds, params.decision_upper_bounds, out=X_copy)
        # Evaluate the positions with the fitness function
        F_copy[:] = self.evaluate(X_copy)
    
    def elitism(self) -> NDArray[np.intp]:
        ''' Selects the best particles from the previous (before applying the equation of motion) and current populations (after applying the equation of motion). The top :attr:`~amesh.parameters.AMESHParameters.population_size` particles, i.e., those with the lowest domination rank, are chosen. In case of a tie, particles with the largest crowding distance are selected.
        
        Note:
            The domination ranks are ordered from the lowest to the highest, starting at the Pareto front with rank zero.
        
        Returns:
            :type:`NDArray[np.intp]`: A numpy array with the indices of the copy population that were selected.
        '''

        population_size = self.params.population_size
        pre_allocated = self.pre_allocated
        # Get the fitness matrix with the previous and the current population
        pre_allocated.fitness_elitism[:population_size] = self.population.fitness
        pre_allocated.fitness_elitism[population_size:] = pre_allocated.fitness_copy
        # Find the best N indices
        best_N_idxs = select_best_N_mo(pre_allocated.fitness_elitism, population_size)
        # Get the copy population indices
        mask_best_idxs = best_N_idxs >= population_size
        best_copy_idxs = best_N_idxs[mask_best_idxs] - population_size
        # Get the population indices that will be replaced by the best copy particles
        np.logical_not(mask_best_idxs, out=mask_best_idxs)
        best_pop_idxs = best_N_idxs[mask_best_idxs]
        mask_worst_pop = np.ones(population_size, dtype=bool)
        mask_worst_pop[best_pop_idxs] = False
        worst_pop_idxs = np.flatnonzero(mask_worst_pop)
        # Put the best copy decision variables in the population
        self.population.position[worst_pop_idxs] = pre_allocated.position_copy[best_copy_idxs]
        self.population.velocity[worst_pop_idxs] = pre_allocated.velocity_copy[best_copy_idxs]
        self.population.fitness[worst_pop_idxs] = pre_allocated.fitness_copy[best_copy_idxs]
        self.population.personal_guide_pos[worst_pop_idxs] = self.population.personal_guide_pos[best_copy_idxs]
        self.population.personal_guide_fit[worst_pop_idxs] = self.population.personal_guide_fit[best_copy_idxs]
        return best_copy_idxs

    def update_personal_guides(self) -> None:
        ''' Updates the personal-guide memories of the population particles.

            Notes:
                Each particle maintains a fixed-size ordered memory of personal guides, where index 0 corresponds to the most recently accepted guide The memory is updated according to the following rules:

                1. If the current particle is Pareto-dominated by at least one personal guide, the memory remains unchanged.

                2. Otherwise, the current particle is inserted at the beginning of the memory. The existing guides are shifted one position toward the end of the list, and the oldest guide is discarded.

                3. After insertion, every retained personal guide that is Pareto-dominated by the current particle is replaced by a copy of the current particle.

                Therefore, a promising current particle may occupy multiple positions in the personal-guide memory. Since personal guides are sampled uniformly during the movement operation, these repeated entries implicitly reinforce particles that dominate previously stored guides without introducing an additional selection parameter.
        '''

        # Get the population fitness as a tensor
        fitness_tensor = self.population.fitness[:, np.newaxis]
        # Get the personal guide fitness
        pb_fitness = self.population.personal_guide_fit
        # Get the mask to update the personal guide
        update_mask = ~np.any(self.dominates(pb_fitness, fitness_tensor, axis=2), axis=1)
        update_idxs = np.flatnonzero(update_mask)
        # Delete the last personal guide and include the current particle as a new personal guide
        self.population.personal_guide_fit[update_idxs, 1:, :] = self.population.personal_guide_fit[update_idxs, :-1, :]
        self.population.personal_guide_pos[update_idxs, 1:, :] = self.population.personal_guide_pos[update_idxs, :-1, :]
        # Update the personal guide list by adding the current particle as a new personal guide
        self.population.personal_guide_fit[update_idxs, 0, :] = self.population.fitness[update_idxs, :]
        self.population.personal_guide_pos[update_idxs, 0, :] = self.population.position[update_idxs, :]
        # Get the mask to replace the personal guide dominated by the current particle
        replace_mask = self.dominates(fitness_tensor[update_mask], pb_fitness[update_mask, 1:, :], axis=2)
        # Replace the dominated personal guide by the current particle
        replace_row, replace_col = np.nonzero(replace_mask)
        particle_to_replace_pb = update_idxs[replace_row]
        pb_to_replace = replace_col + 1
        self.population.personal_guide_fit[particle_to_replace_pb, pb_to_replace, :] = self.population.fitness[particle_to_replace_pb, :]
        self.population.personal_guide_pos[particle_to_replace_pb, pb_to_replace, :] = self.population.position[particle_to_replace_pb, :]

    def update_amesh_memory(self) -> None:
        ''' Updates the memory position and fitness faster using position and fitness numpy matrices from population. '''
        
        # Get the unique positions from the population positions and the memory
        unique_pop_positions, unique_idxs = np.unique(self.population.position, axis=0, return_index=True)
        unique_pop_fitnesses = self.population.fitness[unique_idxs]
        # Get the pareto front indices from population
        memory_pareto_idxs = self.get_non_domination_fronts(unique_pop_fitnesses)[0]
        # If the new memory Pareto front has size less or equal than the memory size, then set the new memory
        memory_size = self.params.population_size
        if(len(memory_pareto_idxs) <= memory_size):
            self.memory.position = unique_pop_positions[memory_pareto_idxs]
            self.memory.fitness = unique_pop_fitnesses[memory_pareto_idxs]
        # Else get the particles with the highest crowd distance in the new memory Pareto front
        else:
            # Select the particles with the highest crowd distance
            selected_fitness = unique_pop_fitnesses[memory_pareto_idxs]
            # Calculate the crowding distance
            crowd_distances = crowding_distance(selected_fitness)
            # Get the indices of the particles with the highest crowd distance
            idxs = np.argpartition(crowd_distances, -memory_size)[-memory_size:]
            # Update the memory
            self.memory.position = unique_pop_positions[memory_pareto_idxs[idxs]]
            self.memory.fitness = selected_fitness[idxs]

    def generic_update_amesh_memory(self, position_matrix: NDArray[np.number], fitness_matrix: NDArray[np.number]) -> None:
        ''' Updates the memory position and fitness using a position and fitness numpy matrices.
        
        Args:
            position_matrix (:type:`NDArray[np.number]`): A numpy matrix with the position of the particles.
            fitness_matrix (:type:`NDArray[np.number]`): A numpy matrix with the fitness of the particles.
        '''

        # Get the unique positions from the position matrix and the memory
        unique_positions, unique_idxs = np.unique(np.concatenate((self.memory.position, position_matrix), axis=0), axis=0, return_index=True)
        # Get the unique fitnesses from the position matrix and the memory
        unique_fitnesses = np.concatenate((self.memory.fitness, fitness_matrix), axis=0)[unique_idxs]
        # Get the Pareto front indices from the memory candidates
        memory_pareto_idxs = self.get_non_domination_fronts(unique_fitnesses)[0]
        # If the new memory Pareto front has size less or equal than the memory size, then set the new memory
        memory_size = self.params.population_size
        if(len(memory_pareto_idxs) <= memory_size):
            self.memory.position = unique_positions[memory_pareto_idxs]
            self.memory.fitness = unique_fitnesses[memory_pareto_idxs]
        # Else get the particles with the highest crowd distance in the new memory Pareto front
        else:
            # Select the particles with the highest crowd distance
            selected_fitness = unique_fitnesses[memory_pareto_idxs]
            # Calculate the crowding distance
            crowd_distances = crowding_distance(selected_fitness)
            # Get the indices of the particles with the highest crowd distance
            idxs = np.argpartition(crowd_distances, -memory_size)[-memory_size:]
            # Update the memory
            self.memory.position = unique_positions[memory_pareto_idxs[idxs]]
            self.memory.fitness = selected_fitness[idxs]

    def run(self):
        '''Runs A-MESH until the maximum number of generations or fitness evaluations is reached.'''

        # Start the progress bars
        with tqdm(total=self.max_algorithm_progress, leave=False) as pbar:
            try:    
                # Initialize the algorithm with initial operations
                self.initialize()
                # Main loop
                while True:
                    # Count generations if it is a stopping criterion
                    self.count_generation()
                    # Calculate Xst for each particle
                    self.differential_evolution()
                    # Update the memory
                    self.update_amesh_memory()
                    # Update the personal guides
                    self.update_personal_guides()
                    # Update global guides
                    self.global_guide_method()
                    # Mutate the global guides
                    self.mutation()
                    # Store some data of the population before the movement
                    self.pre_allocated.position_copy[:] = self.population.position.copy()
                    self.pre_allocated.velocity_copy[:] = self.population.velocity.copy()
                    self.pre_allocated.fitness_copy[:] = self.population.fitness.copy()
                    # Apply the movviment to the particles
                    self.move_population()
                    # Select the best particles from those before and after the moviment
                    best_copy_idxs = self.elitism()
                    # Update the swarm hyperparameter memory
                    self.update_swarm_memory(self.population.position[best_copy_idxs],
                                             self.pre_allocated.position_copy[best_copy_idxs],
                                             best_copy_idxs)
                    # Update A-MESH memory
                    self.update_amesh_memory()
                    # Update the algorithm progress
                    self.algorithm_progress = self.update_algorithm_progress(pbar, self.algorithm_progress)
                    # Update hyperparameter last index
                    self.params.hyperparameter_last_index += 1
                    if self.params.hyperparameter_last_index >= self.params.hyperparameter_memory_length:
                        self.params.hyperparameter_last_index = 0
            # The end of the algorithm
            except StoppingAlgorithm as stop:
                # Update the memory
                self.generic_update_amesh_memory(stop.position, stop.fitness)
                # Log the memory if it is necessary
                self.logging()

    def update_progress_by_fitness_evaluation(self, pbar: tqdm, prev_progress_value: int) -> int:
        ''' Updates the algorithm progress by fitness evaluations. It is used when the stopping criterion is fitness evaluation or both generation and fitness evaluation.
        
        Args:
            pbar (:type:`tqdm`): A :type:`tqdm` object.
            prev_progress_value (:type:`int`): The previous value of the algorithm progress.

        Returns:
            :type:`int`: The current value of the algorithm progress.
        '''

        pbar.update(self.fitness_eval_counter - prev_progress_value)
        return self.fitness_eval_counter
    
    def update_progress_by_generation(self, pbar: tqdm, prev_progress_value: int) -> int:
        ''' Updates the algorithm progress by generations. It is used when the stopping criterion is generation counter.
        
        Args:
            pbar (:type:`tqdm`): A :type:`tqdm` object.
            prev_progress_value (:type:`int`): The previous value of the algorithm progress.

        Returns:
            :type:`int`: The current value of the algorithm progress.
        '''

        pbar.update(self.generation_counter - prev_progress_value)
        return self.generation_counter

    def stopping_by_generation(self) -> None:
        ''' Counts generations if it is a stopping criterion.
        
        Raises:
            :class:`~amesh.utils.auxiliar.StoppingAlgorithm`: If the number of generations is greater than the maximum number of generations.    
        '''

        self.generation_counter += 1
        if self.generation_counter > self.params.max_gen:
            self.generation_counter -= 1
            raise StoppingAlgorithm(np.empty((0, self.params.decision_dim)), np.empty((0, self.params.objective_dim)))
    
    def stopping_by_fitness_evaluation(self, X: NDArray[np.number]) -> NDArray[np.number]:
        ''' Evaluates the position matrix ``X`` and counts the fitness evaluations. This method is used when the stopping criterion is by fitness evaluations.
        
        Args:
            X (:type:`NDArray[np.number]`): A numpy matrix with the particle positions.
            
        Returns:
            :type:`NDArray[np.number]`: The fitness matrix.
        
        Raises:
            :class:`~amesh.utils.auxiliar.StoppingAlgorithm`: If the number of fitness evaluations is greater than the maximum number of fitness evaluations.    
        '''

        # Get the size of the position matrix
        X_size = len(X)
        # Calculate the minimum number of fitness evaluations
        min_evaluations = min(self.params.max_fit_eval - self.fitness_eval_counter, X_size)
        # Update the fitness counter
        self.fitness_eval_counter += min_evaluations
        # Evaluate the fitness function
        if(self.fitness_eval_counter < self.params.max_fit_eval):
            return self.evaluation_way(X)
        else:
            # Evaluate the sliced particle positions and stop the algorithm
            X_sliced = X[:min_evaluations]
            raise StoppingAlgorithm(X_sliced, self.evaluation_way(X_sliced))

    def get_results(self) -> tuple[NDArray[np.number], NDArray[np.number]]:
        ''' Returns a tuple with the memory position and fitness, respectively.
        
        Note:
            This method must be used at the end of the algorithm.
        
        Returns:
            :type:`tuple[NDArray[np.number], NDArray[np.number]]`: A tuple with the memory position and fitness, respectively.
        '''

        return self.memory.position, self.memory.fitness

    def logging(self) -> None:
        ''' Logs memory position and fitness at the end of the algorithm in two .txt files if :attr:`log_memory` is a string. Then this method uses the string value :attr:`log_memory` at the beginning of both files as the name of the fitness and position logs.
        '''

        if self.log_memory is not None:
            # Log the fitness
            with open(self.log_memory+"-fit.txt","a+") as file:
                memory_fitness = ""
                for fit in self.memory.fitness:
                    string = ""
                    for i in range(self.params.objective_dim):
                        string += str(fit[i]) + " "
                    string = string[:-1]
                    memory_fitness += string + ", "
                memory_fitness = memory_fitness[:-2]
                memory_fitness += "\n"
                file.write(memory_fitness)
            # Log the position
            with open(self.log_memory + "-pos.txt", "a+") as file:
                memory_position = ""
                for pos in self.memory.position:
                    string = ""
                    for i in range(self.params.decision_dim):
                        string += str(pos[i])+" "
                    string = string[:-1]
                    memory_position += string + ", "
                memory_position = memory_position[:-2]
                memory_position += "\n"
                file.write(memory_position)
