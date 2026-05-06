from .operations.differential_crossover import get_differential_crossover
from .operations.differential_mutation import get_differential_mutation
from .operations.differential_mutation_pool import get_differential_mutation_pool
from .operations.global_guide_method import get_global_guide_method
from .parameters import MeshParameters
from .particles import Population, Memory
from .auxiliar import PreAllocated, StoppingAlgorithm
from .validations.python_validations import assert_type, is_greater_in_type, is_function

from joblib import Parallel, delayed
from numpy.typing import NDArray
from pygmo import fast_non_dominated_sorting, select_best_N_mo, crowding_distance # type: ignore
from tqdm import tqdm
from types import MethodType
from typing import Any, Callable, Optional

import numpy as np
import random

class Mesh():
    ''' MESH algorithm.
    
    Args:
        params (:class:`~mesh.parameters.MeshParameters`): MESH parameters.
        fitness_function (:type:`Callable[[NDArray[np.number]], NDArray[np.number]]`): A fitness function that returns a numpy vector with each objective value in the respective component.
        log_memory (:type:`str | None`): A string to log the memory. If its value is ``None``, then the memory location and fitness will not be logged in a file.
        num_proc (:type:`int | None`): Number of processes to execute the fitness function in parallel. If it is ``None``, so the fitness function will execute sequentially.
    
    Raises:
        TypeError: If the input is not the expected type.
        ValueError: If the input is not the allowed value.
    
    Note:
        Using parallel evaluations is only advantageous if :attr:`fitness_function` is sufficiently computationally expensive.
    '''

    def __init__(self,
                params: MeshParameters,
                fitness_function: Callable[[NDArray[np.number]], NDArray[np.number]],
                log_memory: Optional[str] = None,
                num_proc: Optional[int] = None):
        
        self.params: MeshParameters
        ''' Mesh parameters. '''
        self.global_guide_method: Callable[[], None]
        ''' Function to find the global guides for the particles. '''
        self.differential_mutation_pool: Callable[[], tuple[NDArray[np.number], list[NDArray[np.intp]]]]
        ''' Function to make the Differential Mutation pool where the solutions are samppled. '''
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
        ''' Generation counter. Used to stop the algorithm if its value is greater than 0. '''
        self.fitness_eval_counter: int
        ''' Fitness evaluation counter. Used to stop the algorithm if its value is greater than 0. '''
        self.pre_allocated: PreAllocated
        ''' Pre-allocated data for the algorithm. '''
        self.log_memory: Optional[str]
        ''' A string to log the memory. If its value is ``None``, then the memory location and fitness will not be logged in a file. '''
        self.num_proc: Optional[int]
        ''' Number of processes to execute the fitness function in parallel. If it is ``None``, so the fitness function will execute sequentially. '''
        self.evaluation_way: Callable[[NDArray[np.number]], NDArray[np.number]]
        ''' The way to evaluate the fitness function. It can be sequentially or parallelly. If :attr:`num_proc` is not None, so the fitness evaluations will be parallel with :attr:`num_proc` processes. '''
        self.evaluate: Callable[[NDArray[np.number]], NDArray[np.number]]
        ''' Function for fitness evaluations. If :attr:`~mesh.parameters.MeshParameters.max_fit_eval` is not None, so the fitness evaluations will be counted. '''
        self.count_generation: Callable[[], None]
        ''' Function to count generations. Only used if :attr:`~mesh.parameters.MeshParameters.max_gen` is not None. '''
        self.update_memory: Callable[[NDArray[np.number], NDArray[np.number]]]
        ''' Function to update the memory matrix. When the :attr:`~mesh.parameters.MeshParameters.memory_size` is less or equal :attr:`~mesh.parameters.MeshParameters.population_size`, the update can be faster. '''
        self.update_progress_bar: Callable[[tqdm, int], int]
        ''' Function to update the progress bar. '''
        self.total_bar: int
        ''' Total value of the progress bar. '''

        # Receive the algorithm parameters
        assert_type(params, 'params', MeshParameters)
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
        # Choose the memory update function according to memory size
        if params.memory_size <= params.population_size:
            self.update_memory = self.fast_update_memory
        else:
            self.update_memory = self.generic_update_memory
        # Choose the way to update the algorithm progress bar
        if params.max_gen == 0:
            self.total_bar = params.max_fit_eval
            self.update_progress_bar = self.update_progress_bar_by_fitness_evaluation
        elif params.max_fit_eval == 0:
            self.total_bar = params.max_gen
            self.update_progress_bar = self.update_progress_bar_by_generation
        else:
            self.update_progress_bar = self.update_progress_bar_by_fitness_evaluation
            self.total_bar = min(params.population_size*(2*params.max_gen+1), params.max_fit_eval)
    
    def initialize(self):
        ''' Initializes the MESH with some initial operations. It initializes the population, memory and personal guide fitness, does initial fitness evaluations and calculates the domination fronts. '''

        # Evaluate the initial population
        self.population.fitness[:] = self.evaluate(self.population.position)
        # Update memory
        self.update_memory(self.population.position, self.population.fitness)
        # Repeat the population fitness for all personal guide input
        self.population.personal_guide_fit[:, :, :] = np.repeat(self.population.fitness[:, np.newaxis, :], self.params.max_personal_guides, axis=1)

    def sequential_fitness_evaluation(self, X: NDArray[np.number]) -> NDArray[np.number]:
        ''' Evaluates the fitness given a particle position matrix sequentially.
        
        Args:
            X (:type:`NDArray[np.number]`): A numpy matrix with the particle positions.

        Returns:
            :type:`NDArray[np.number]`: The fitness matrix associated with the particle positions.
        '''

        decision_varibles = X[:, :self.params.decision_dim]
        return np.array([self.fitness_function(x) for x in decision_varibles])

    def parallel_fitness_evaluation(self, X: NDArray[np.number]) -> NDArray[np.number]:
        ''' Evaluates the fitness given a particle position matrix parallelly.
        
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

    def differential_evolution(self) -> None:
        r''' Generates solutions by Differential Evolution algorithm according to a differential mutation strategy decided by :attr:`~mesh.parameters.MeshParameters.dm_operation_type`, with solutions sampled in a pool decided by :attr:`~mesh.parameters.MeshParameters.dm_pool_type`. When new solutions are generated, an elitism is performed to update the position of the current population's less promising solutions.
        
        Note:
            The criteria for the best elitism solutions are the same as those for the method :meth:`elitism`.
        '''
        
        # Apply a differential mutation strategy
        Xst, pop_idxs = self.differential_mutation(self.differential_mutation_pool())
        if len(Xst):
            population_size = self.params.population_size
            # Apply the differential crossover
            Xst_rec = self.differential_crossover(
                self.population.position[pop_idxs],
                Xst,
                self.population.position[pop_idxs, self.params.decision_dim+1:self.params.decision_dim+2]
            )
            # Update the current particle if the new particle from the strategy is better
            Fst_rec = self.evaluate(Xst_rec)
            # Concatenate the arrays with the population position and fitness with the strategy arrays
            update_memory_pos = np.concatenate((self.population.position, Xst_rec), axis=0)
            update_memory_fit = np.concatenate((self.population.fitness, Fst_rec), axis=0)
            # Find the best N indices
            best_N_idxs = select_best_N_mo(update_memory_fit, population_size)
            # Get the indices of the best particles in the strategy array
            mask = best_N_idxs >= population_size
            best_st_indices = best_N_idxs[mask] - population_size
            # Put the best strategy particles in the current population
            np.logical_not(mask, out=mask)
            worst_pop_idxs = np.setdiff1d(np.arange(population_size), best_N_idxs[mask], assume_unique=True)
            self.population.position[worst_pop_idxs, :self.params.decision_dim+2] = Xst_rec[best_st_indices, :self.params.decision_dim+2]
            self.population.fitness[worst_pop_idxs] = Fst_rec[best_st_indices]
            # Update the memory with the new particles from the strategy
            self.update_memory(update_memory_pos, update_memory_fit)

    def mutation(self) -> None:
        r''' Calculates the mutation of the global guides are done by the following equation:

        .. math::
                
            \tilde{x}_{gb} = x_{gb} + \tau_{mut} \cdot \vec{r},
            
        where :math:`\vec{r}` is calculated as a decision variable.
        '''
        
        # Mutate the global guides with a vector sampled from the Standard Gaussian Distribution
        np.clip(
            self.population.global_guide + np.random.normal(0, 1, (self.params.population_size, self.params.position_dim)) * self.population.position[:, self.params.decision_dim+6:],
            self.params.position_lower_bounds,
            self.params.position_upper_bounds,
            out=self.pre_allocated.global_guide_mutated
        )

    def move_population(self) -> None:
        r''' Applies the equation of motion to the particles. The MESH equation of motion is given by:
        
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
        # Get the positions
        X = self.population.position
        # Get the equation weights
        W = X[:, params.decision_dim+2:params.decision_dim+6]
        # Calculate the new velocity
        C = np.random.rand(population_size, params.position_dim) <= W[:, 0][:, np.newaxis]
        self.population.velocity[:] = W[:, 1][:, np.newaxis] * self.population.velocity + W[:, 2][:, np.newaxis] * (Xpb - X) + W[:, 3][:, np.newaxis] * C * (Xgb_mut - X)
        # Calculate the clipped velocity
        np.clip(self.population.velocity, params.velocity_lower_bounds, params.velocity_upper_bounds, out=self.population.velocity)
        # Calculate the clipped position
        self.population.position += self.population.velocity
        np.clip(self.population.position, params.position_lower_bounds, params.position_upper_bounds, out=self.population.position)
        # Evaluate the positions with the fitness function
        self.population.fitness[:] = self.evaluate(self.population.position)
    
    def elitism(self) -> None:
        ''' Selects the best particles from the previous (before applying the equation of motion) and current populations (after applying the equation of motion). The top :attr:`~mesh.parameters.MeshParameters.population_size` particles, i.e., those with the lowest domination rank, are chosen. In case of a tie, particles with the largest crowding distance are selected.
        
        Note:
            The domination ranks are ordered from the lowest to the highest, starting at the Pareto front with rank zero.
        
        Returns:
            :type:`NDArray[np.integer]`: A numpy array with the indices of the current population that were selected.
        '''

        population_size = self.params.population_size
        pre_allocated = self.pre_allocated
        # Get the fitness matrix with the previous and the current population
        pre_allocated.fitness_elitism[:population_size] = pre_allocated.fitness_copy
        pre_allocated.fitness_elitism[population_size:] = self.population.fitness
        # Find the best N indices
        best_N_idxs = select_best_N_mo(pre_allocated.fitness_elitism, population_size)
        # Get the previous population indices
        mask = best_N_idxs < population_size
        prev_idxs = best_N_idxs[mask]
        # Get the current population indices
        np.logical_not(mask, out=mask)
        current_idxs = best_N_idxs[mask] - population_size
        # Put the best previous particles in the current population
        worst_current_idxs = np.setdiff1d(np.arange(population_size), current_idxs, assume_unique=True)
        decision_dim = self.params.decision_dim
        self.population.position[worst_current_idxs, :decision_dim] = pre_allocated.position_copy[prev_idxs, :decision_dim]
        self.population.position[worst_current_idxs, decision_dim+2:] = pre_allocated.position_copy[prev_idxs, decision_dim+2:]
        self.population.velocity[worst_current_idxs, :decision_dim] = pre_allocated.velocity_copy[prev_idxs, :decision_dim]
        self.population.velocity[worst_current_idxs, decision_dim+2:] = pre_allocated.velocity_copy[prev_idxs, decision_dim+2:]
        self.population.fitness[worst_current_idxs] = pre_allocated.fitness_copy[prev_idxs]
        self.population.personal_guide_pos[worst_current_idxs, :decision_dim] = self.population.personal_guide_pos[prev_idxs, :decision_dim]
        self.population.personal_guide_pos[worst_current_idxs, decision_dim+2:] = self.population.personal_guide_pos[prev_idxs, decision_dim+2:]
        self.population.personal_guide_fit[worst_current_idxs] = self.population.personal_guide_fit[prev_idxs]

    def update_personal_guides(self) -> None:
        ''' Updates the personal guides of the population particles.

        Note:
            There is three cases to update the personal guides:

            - When the current particle is dominated by any of its personal guide, the current particle is ignored;
            - When the current particle dominates a personal guide, the current particle replaces the dominated personal guide. This replacement is done for all dominated personal guides, so the more the current particle dominates its personal guides, the more chance it has of being sampled in :meth:`move_population`;
            - When the current particle don't dominate and is not dominated by any personal guide, the current particle is added to the personal guide matrix. The oldest personal guide is removed when the current particle is only added.
        '''

        # Get the population fitness as a tensor
        fitness_tensor = self.population.fitness[:, np.newaxis]
        # Get the personal guide fitness
        pb_fitness = self.population.personal_guide_fit
        # Get the mask to update the personal guide
        update_mask = ~np.all(self.dominates(pb_fitness, fitness_tensor, axis=2), axis=1)
        update_idxs = np.flatnonzero(update_mask)
        # Get the mask to replace the personal guide dominated by the current particle
        replace_mask = self.dominates(fitness_tensor[update_mask], pb_fitness[update_mask], axis=2)
        # Replace the dominated personal guide by the current particle
        replace_row, replace_col = np.nonzero(replace_mask)
        particle_to_replace_pb = update_idxs[replace_row]
        self.population.personal_guide_fit[particle_to_replace_pb, replace_col, :] = self.population.fitness[particle_to_replace_pb, :]
        self.population.personal_guide_pos[particle_to_replace_pb, replace_col, :] = self.population.position[particle_to_replace_pb, :]
        # Get the mask to add the current to the personal guide list
        add_idxs = update_idxs[~np.any(replace_mask, axis=1)]
        # Delete the oldest personal guide and include the current particle as a new personal guide
        self.population.personal_guide_fit[add_idxs, 1:, :] = self.population.personal_guide_fit[add_idxs, :-1, :]
        self.population.personal_guide_pos[add_idxs, 1:, :] = self.population.personal_guide_pos[add_idxs, :-1, :]
        # Update the personal guide list by adding the current particle as a new personal guide
        self.population.personal_guide_fit[add_idxs, 0, :] = self.population.fitness[add_idxs, :]
        self.population.personal_guide_pos[add_idxs, 0, :] = self.population.position[add_idxs, :]

    def fast_update_memory(self, _: Any, __: Any) -> None:
        ''' Updates the memory position and fitness faster using position and fitness numpy matrices from population.

        Note:
            Function arguments are for compatibility with the method :meth:`generic_update_memory`.
        '''
        
        # Get the unique positions from the population positions and the memory
        unique_pop_positions, unique_idxs = np.unique(self.population.position, axis=0, return_index=True)
        unique_pop_fitnesses = self.population.fitness[unique_idxs]
        # Get the pareto front indices from population
        memory_pareto_idxs = self.get_non_domination_fronts(unique_pop_fitnesses)[0]
        # If the new memory Pareto front has size less or equal than the memory size, then set the new memory
        memory_size = self.params.memory_size
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

    def generic_update_memory(self, position_matrix: NDArray[np.number], fitness_matrix: NDArray[np.number]) -> None:
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
        memory_size = self.params.memory_size
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
        ''' This method runs the MESH algorithm. It stops when the maximum number of generations and/or fitness evaluations is reached. '''

        # Start the progress bars
        with tqdm(total=self.total_bar, leave=False) as pbar:
            try:    
                # A variable to update the tqdm bar
                prev_bar_value = 0
                # Initialize the algorithm with initial operations
                self.initialize()
                # Main loop
                while True:
                    # Count generations if it is a stopping criterion
                    self.count_generation()
                    # Calculate Xst for each particle
                    self.differential_evolution()
                    # Update global guides
                    self.global_guide_method()
                    # Mutate the weights and the global guides
                    self.mutation()
                    # Update the personal guides
                    self.update_personal_guides()
                    # Store some data of the population before the movement
                    self.pre_allocated.position_copy[:] = self.population.position.copy()
                    self.pre_allocated.velocity_copy[:] = self.population.velocity.copy()
                    self.pre_allocated.fitness_copy[:] = self.population.fitness.copy()
                    # Apply the movviment to the particles
                    self.move_population()
                    # Select the best particles from those before and after the moviment
                    self.elitism()
                    # Update memory
                    self.update_memory(self.population.position, self.population.fitness)
                    # Update the progress bar
                    prev_bar_value = self.update_progress_bar(pbar, prev_bar_value)
            # The end of the algorithm
            except StoppingAlgorithm as stop:
                # Updated the memory
                self.generic_update_memory(stop.position, stop.fitness)
                # Log the memory if it is necessary
                self.logging()

    def update_progress_bar_by_fitness_evaluation(self, pbar: tqdm, prev_bar_value: int) -> int:
        ''' Updates the progress bar by fitness evaluations. It is used when the stopping criterion is fitness evaluation or both generation and fitness evaluation.
        
        Args:
            pbar (:type:`tqdm`): A :type:`tqdm` object.
            prev_bar_value (:type:`int`): The previous value of the progress bar.

        Returns:
            :type:`int`: The current value of the progress bar.
        '''

        pbar.update(self.fitness_eval_counter - prev_bar_value)
        return self.fitness_eval_counter
    
    def update_progress_bar_by_generation(self, pbar: tqdm, prev_bar_value: int) -> int:
        ''' Updates the progress bar by generations. It is used when the stopping criterion is generation counter.
        
        Args:
            pbar (:type:`tqdm`): A :type:`tqdm` object.
            prev_bar_value (:type:`int`): The previous value of the progress bar.

        Returns:
            :type:`int`: The current value of the progress bar.
        '''

        pbar.update(self.generation_counter - prev_bar_value)
        return self.generation_counter

    def stopping_by_generation(self) -> None:
        ''' Counts generations if it is a stopping criterion.
        
        Raises:
            :class:`~mesh.utils.auxiliar.StoppingAlgorithm`: If the number of generations is greater than the maximum number of generations.    
        '''

        self.generation_counter += 1
        if self.generation_counter > self.params.max_gen:
            self.generation_counter -= 1
            raise StoppingAlgorithm(np.empty((0, self.params.position_dim)), np.empty((0, self.params.objective_dim)))
    
    def stopping_by_fitness_evaluation(self, X: NDArray[np.number]) -> NDArray[np.number]:
        ''' Evaluates the position matrix ``X`` and counts the fitness evaluations. This method is used when the stopping criterion is by fitness evaluations.
        
        Args:
            X (:type:`NDArray[np.number]`): A numpy matrix with the particle positions.
            
        Returns:
            :type:`NDArray[np.number]`: The fitness matrix.
        
        Raises:
            :class:`~mesh.utils.auxiliar.StoppingAlgorithm`: If the number of fitness evaluations is greater than the maximum number of fitness evaluations.    
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

        return self.memory.position[:, :self.params.decision_dim], self.memory.fitness

    def logging(self) -> None:
        ''' Logs memory position and fitness at the end of the algorithm in two .txt files if :attr:`log_memory` is a string. Then this method uses the string value :attr:`log_memory` at the beginning of both files as the name of the fitness and position logs.
        '''

        if self.log_memory is not None:
            # Log the fitness
            file = open(self.log_memory+"-fit.txt","a+")
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
            file.close()
            # Log the position
            file2 = open(self.log_memory + "-pos.txt", "a+")
            memory_position = ""
            for pos in self.memory.position[self.params.decision_dim]:
                string = ""
                for i in range(self.params.decision_dim):
                    string += str(pos[i])+" "
                string = string[:-1]
                memory_position += string + ", "
            memory_position = memory_position[:-2]
            memory_position += "\n"
            file2.write(memory_position)
            file2.close()
