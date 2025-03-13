###########################################################################
# Lucas Braga, MS.c. (email: lucas.braga.deo@gmail.com )
# Gabriel Matos Leite, PhD candidate (email: gmatos@cos.ufrj.br)
# Carolina Marcelino, PhD (email: carolimarc@ic.ufrj.br)
# June 16, 2021
###########################################################################
# Copyright (c) 2021, Lucas Braga, Gabriel Matos Leite, Carolina Marcelino
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in
#      the documentation and/or other materials provided with the
#      distribution
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS USING 
# THE CREATIVE COMMONS LICENSE: CC BY-NC-ND "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


from parameters import MeshParameters
from utils.particles import Population, Memory
from utils.auxiliar import PreAllocated, StoppingAlgorithm
from operations.global_best_attribution import get_global_best_attribution
from operations.differential_mutation_pool import get_differential_mutation_pool
from operations.differential_mutation_operation import get_differential_mutation_operation
from validations.python import assert_type, assert_type_or_falsy
from validations.numpy import is_fitness_function

from scipy.stats import truncnorm
from tqdm import tqdm
from pygmo import fast_non_dominated_sorting, select_best_N_mo, crowding_distance
from types import MethodType
from typing import Callable

import numpy as np

# import tracemalloc
# tracemalloc.start()
# current, peak = tracemalloc.get_traced_memory()
# print(f"Memória atual: {current / 10**6:.2f} MB; Pico de memória: {peak / 10**6:.2f} MB")
# tracemalloc.stop()

class Mesh():
    ''' MESH algorithm.
    
    Args:
        params (:class:`~mesh.parameters.MeshParameters`): MESH parameters.
        fitness_function (:type:`Callable[[np.ndarray[np.number]], np.ndarray[np.number]]`): A fitness function that returns a numpy array with each objective value in the respective component.
        log_memory (:type:`str | False`): A string to log the memory. The file name will use this string. It must be a string or a falsy value. Default is False.
    
    Raises:
        TypeError: If the input is not the expected type.
        ValueError: If the input is not the allowed value.
    '''

    def __init__(self,
                params: MeshParameters, # MESH parameters
                fitness_function: Callable[[np.ndarray[np.number]], np.ndarray[np.number]], # A fitness function that returns a numpy array with each objective value in the respective component
                log_memory=False): # A string to log the memory (the file name will use this string)
        
        self.params: MeshParameters
        ''' Mesh parameters. '''
        self.global_best_attribution: MethodType[Callable[[Mesh], None]]
        ''' Function to attribute the global best to the particles. '''
        self.differential_mutation_pool: MethodType[Callable[[Mesh], list[np.ndarray[np.float64, 2]]]]
        ''' Function to make the differential mutation pool. '''
        self.differential_mutation_operation: MethodType[Callable[[Mesh, list[np.ndarray[np.float64, 2]]], tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]]]
        ''' Function to do the differential mutation operation. '''
        self.population: Population
        ''' Population of particles. '''
        self.memory: Memory
        ''' Memory of particles. '''
        self.fronts: list[np.ndarray[np.integer]]
        ''' List of numpy arrays with index of each particle in the respective front. '''
        self.fitness_function: Callable[..., np.ndarray[np.number]]
        ''' Fitness function. '''
        self.generation_counter: int
        ''' Generation counter. Used to stop the algorithm if its value is greater than 0. '''
        self.fitness_eval_counter: int
        ''' Fitness evaluation counter. Used to stop the algorithm if its value is greater than 0. '''
        self.weights: np.ndarray[np.float64, 2]
        ''' Weights for the algorithm operations moving the population. '''
        self.pre_allocated: PreAllocated
        ''' Pre-allocated data for the algorithm. '''
        self.log_memory: str | False
        ''' A string to log the memory. '''
        self.fitness_eval: Callable[[np.ndarray[np.float64, 2]], tuple[np.ndarray[np.float64, 1], int]]
        ''' Function for fitness evaluations. If :attr:`~mesh.parameters.MeshParameters.max_fit_eval` is greater than 0, so the fitness evaluations will be counted. '''
        self.count_generation: Callable[[], None]
        ''' Function to count generations. Only used if :attr:`~mesh.parameters.MeshParameters.max_fit_eval` is 0. '''
        self.update_progress_bar: Callable[[tqdm, int], int]
        ''' Function to update the progress bar. '''
        self.total_bar: int
        ''' Total value of the progress bar. '''

        # Receive the algorithm parameters
        assert_type(params, 'params', MeshParameters)
        self.params = params
        # Chosing the operations just one time
        self.global_best_attribution = MethodType(get_global_best_attribution(params.global_best_attribution_type), self)
        self.differential_mutation_pool = MethodType(get_differential_mutation_pool(params.dm_pool_type), self)
        self.differential_mutation_operation = MethodType(get_differential_mutation_operation(params.dm_operation_type), self)
        # Use a random seed if there is
        np.random.seed(params.random_state)
        # Particles
        self.population = Population(params)
        # Memory particles (and the final result after run MESH)
        self.memory = None
        # Fronts (a list of numpy arrays with index of each particle in the respective front)
        self.fronts = []
        # Estabilish the fitness function
        is_fitness_function(fitness_function, 'fitness_function', params.position_dim, params.objective_dim)
        self.fitness_function = fitness_function
        # Start the generation counter
        self.generation_counter = 0
        # Start the fitness evaluation counter
        self.fitness_eval_counter = 0
        # Create a random matrix (4 x population_size) with weights for the algorithm operations
        self.weights = np.random.uniform(0.0, 1.0, [4, params.population_size])
        # Store some pre-calculated data
        self.pre_allocated = PreAllocated(params)
        # Variable for logging memory
        assert_type_or_falsy(log_memory, 'log_memory', str)
        self.log_memory = log_memory
        # Check if generation is a stopping criterion
        if params.max_gen > 0:
            self.count_generation = self.stopping_by_generation
        else:
            self.count_generation = lambda : None
        # Check if the fitness evaluation is a stopping criterion
        if params.max_fit_eval > 0:
            self.fitness_eval = self.stopping_by_fitness_eval
        else:
            self.fitness_eval = self.fitness_evaluations
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
    
    def fitness_evaluations(self, X: np.ndarray[np.number, 2]) -> tuple[np.ndarray[np.number, 2], int]:
        ''' Evaluates the fitness given a particle position matrix.
        
        Args:
            X (:type:`np.ndarray[np.number, 2]`): A numpy matrix with the particle positions.

        Returns:
            :type:`tuple[np.ndarray[np.number, 2], int]`: A tuple with the fitness matrix and the number of evaluations.
        '''

        return np.array([self.fitness_function(x) for x in X]), len(X)
    
    def dominates(self, x: np.ndarray[np.number, ], y: np.ndarray[np.number, ], axis: int | np.integer = 0) -> np.ndarray[np.bool, ]:
        r''' Checks if an numpy array x dominates an numpy array y on the respective axis. Given two arrays :math:`x \in \mathbb{R}^n` and :math:`y \in \mathbb{R}^n`, :math:`x` dominates :math:`y` if and only if the following condition are satisfied:

        .. math::
        
            x \neq y\ \land\ x \preceq y,
        
        where:

        .. math::

            x \neq y &\iff \exists i \in \{1,\ \ldots,\ n\}\ (\ x_i \neq y_i), \\
            x \preceq y &\iff \forall i \in \{1,\ \ldots,\ n\}\ (x_i \leq y_i).
            
        Args:
            x (:type:`np.ndarray[np.number, n]`): A n-dimensional numpy array.
            y (:type:`np.ndarray[np.number, n]`): A n-dimensional numpy array.
            axis (:type:`int | np.integer`): The axis to compare the arrays. Default is 0.
        
        Returns:
            :type:`np.ndarray[np.bool, n-1]`: A (n-1)-dimensional numpy array with the result of the comparison.
        '''

        return np.all(x <= y, axis=axis) & np.any(x < y, axis=axis)

    def get_domination_fronts(self, fitness_matrix: np.ndarray[np.number, 2]) -> tuple[list[np.ndarray[np.integer]], np.ndarray[np.integer]]:
        ''' Gets the fronts and the domination ranks of the particles given a fitness matrix.
        
        Note:
            The fronts are a list of numpy arrays. Each numpy array in the list represents a front, starting with the Pareto front. Each particle has its own index.

        Args:
            fitness_matrix (:type:`np.ndarray[np.number, 2]`): A numpy matrix with the fitness values of the particles.

        Returns:
            :type:`tuple[list[np.ndarray[np.integer]], np.ndarray[np.integer]]`: A tuple with the fronts and the domination ranks of the particles, respectively.
        '''

        # If there is only one particle in the particle list, then it is the Pareto front by itself
        if(len(fitness_matrix) == 1):
            return np.array([np.array([0])]), np.array([0])
        # Do the Fast Non-dominated Sorting from Pygmo
        non_dominated_fronts, _, _, ranks = fast_non_dominated_sorting(points=fitness_matrix)
        return non_dominated_fronts, ranks
    
    ''' ################################################################################################################################################## '''
    def reflect_velocity_at_bounds(self, velocity_input: np.ndarray[np.number, 2], position_input: np.ndarray[np.number, 2]) -> np.ndarray[np.number, 2]:
        ''' Reverses the direction of each component of the velocity that took the particle out of its respective boundaries.
        
        Args:
            velocity_input (:type:`np.ndarray[np.number, 2]`): A numpy matrix with the particle velocities.
            position_input (:type:`np.ndarray[np.number, 2]`): A numpy matrix with the particle positions.
        
        Returns:
            :type:`np.ndarray[np.number, 2]`: A numpy matrix with the velocities reflected at the boundaries.
        '''

        neg_velocity = (velocity_input < 0)
        return np.where(((position_input == self.params.position_min_value) & neg_velocity) |
                        ((position_input == self.params.position_max_value) & (~ neg_velocity)),
                        -velocity_input,
                        velocity_input)
    ''' ################################################################################################################################################## '''

    def differential_mutation(self) -> None:
        ''' Applies a differential mutation operation decided by :attr:`~mesh.parameters.MeshParameters.dm_operation_type` in a pool decided by :attr:`~mesh.parameters.MeshParameters.dm_pool_type`. '''
        
        # A array of a matrix pool in each row
        xr_pool_list = self.differential_mutation_pool()
        # Apply a strategy
        xst, valid_idxs = self.differential_mutation_operation(xr_pool_list)
        if len(xst):
            # Update the current particle if the new particle from the strategy is better
            st_fitnesses, min_evaluations = self.fitness_eval(xst)
            min_valid_idxs = valid_idxs[:min_evaluations]
            valid_pop_fitnesses = self.population.fitness[min_valid_idxs]
            domination_mask = self.dominates(st_fitnesses, valid_pop_fitnesses, axis=1)
            update_idxs = min_valid_idxs[domination_mask]
            # Update the positions and the fitnesses
            self.population.position[update_idxs] = xst[:min_evaluations][domination_mask]
            self.population.fitness[update_idxs] = st_fitnesses[domination_mask]
            # If a particle was replaced for a particle from a strategy update some information
            if len(update_idxs):
                self.update_personal_best(update_idxs)
                self.fronts, self.population.rank = self.get_domination_fronts(self.population.fitness)
                self.update_memory()

    def mutate_weights(self) -> None:
        ''' Calculates the weights by a truncated normal distribution with mean 1 and standard deviation 0 between 0 and 1, and then multiplies by :attr:`~mesh.parameters.MeshParameters.mutation_rate`.
        
        Warning:
            The weights don't follow exactly the equation in the article.
        '''
        
        
        # Get the values from truncated normal distribution
        self.weights[:, :] = truncnorm.rvs(0, 1, size=(4, self.params.population_size)) * self.params.mutation_rate
    
    def move_population(self) -> None:
        r''' Applies the equation of motion to the particles. The MESH equation of motion is given by:
        
        .. math::

            \begin{cases}
                v^{(t+1)} = w^*_Iv^{(t)} + w^*_A(x_{pb} - x^{(t)}) + w^*_CC \times (x^*_{gb} - x^{(t)}) \\
                x^{(t+1)} = x^{(t)} + v^{(t+1)}
            \end{cases},
        
        where:

        - :math:`v^{(t)}` is the velocity vector at time t;
        - :math:`x^{(t)}` is the position vector at time t;
        - :math:`w_I` is the inertia weight;
        - :math:`w_A` is the assimilation weight;
        - :math:`w_C` is the cooperation weight;
        - :math:`C` is a binary diagonal matrix, called communication matrix. Given :math:`U(0,\ 1)` a number sampled under a uniform distribution between 0 and 1 and :math:`\tau_{com}` the :attr:`~mesh.parameters.MeshParameters.communication_probability`, :math:`C` is calculated by:

        .. math::

            C_{ij} = \begin{cases} 1,\ \text{if } (i = j) \land (U(0,\ 1) \leq \tau_{com}) \\ 0,\ \text{otherwise} \end{cases}.

        - :math:`x_{pb}` is the personal best vector of the particle;
        - :math:`x_{gb}` is the global best vector o the particle.
        
        Note:
            In this implementation, the weights are calculated every generation by :meth:`mutate_weights`. The mutation of :math:`x_{gb}` is done by:
            
            .. math::
                
                x^*_{gb} = x_{gb}(1 + \tau_{mut} \cdot \mathcal{N}(0, 1)).
            
            where :math:`\tau_{mut}` is the :attr:`~mesh.parameters.MeshParameters.mutation_rate` and :math:`\mathcal{N}(0, 1)` is a number sampled from the standard Gaussian Distribution.
        
        Warning:
            :math:`\tau_{mut}` is not used in the equation of motion directly. It is used together with a weight.
        '''

        # Get the parameters
        params = self.params
        # Get the population size and the position dimension
        population_size = params.population_size
        # Generating random indices for each sublist
        random_indices = np.random.randint(0, self.params.max_personal_guides, size=population_size)
        # Get matrix of personal best list positions
        pb_positions = self.population.personal_best_pos[np.arange(population_size), random_indices, :]
        # Get the global best positions
        gb_positions = self.population.global_best
        # Get the positions
        positions = self.population.position
        # Get the velocities
        velocities = self.population.velocity
        # Get the weights
        weights = self.weights
        # Calculate the inertia term and accumulate it in the velocities
        np.multiply(velocities, weights[0][:, np.newaxis], out=velocities)
        # Calculate the memory term and accumulate it in the velocities
        matrix_for_operations = self.pre_allocated.matrix_for_operations
        np.subtract(pb_positions, positions, out=matrix_for_operations)
        np.multiply(matrix_for_operations, weights[1][:, np.newaxis], out=matrix_for_operations)
        np.add(velocities, matrix_for_operations, out=velocities)
        # Calculate the cooperation term and accumulate it in the velocities
        vector_for_operations = self.pre_allocated.vector_for_operations
        vector_for_operations[:] = np.random.normal(0, 1, population_size)
        np.multiply(vector_for_operations, weights[3], out=vector_for_operations)
        np.add(vector_for_operations, 1, out=vector_for_operations)
        np.multiply(vector_for_operations[:, np.newaxis], gb_positions, out=matrix_for_operations)
        np.subtract(matrix_for_operations, positions, out=matrix_for_operations)
        np.multiply(matrix_for_operations, weights[2][:, np.newaxis], out=matrix_for_operations)
        np.multiply(matrix_for_operations, np.random.uniform(0.0, 1.0, (population_size, params.position_dim)) < params.communication_probability, out=matrix_for_operations)
        np.add(velocities, matrix_for_operations, out=velocities)
        # Calculate the new velocity (clipped)
        np.clip(velocities, params.velocity_min_value, params.velocity_max_value, out=velocities)
        # Calculate the new position (clipped)
        np.add(positions, velocities, out=positions)
        np.clip(positions, params.position_min_value, params.position_max_value, out=positions)
        ''' ################################################################################################################################################## '''
        # self.population.velocity[:, :] = self.reflect_velocity_at_bounds(velocities, positions)
        ''' ################################################################################################################################################## '''
        # Evaluate the fitness function
        fitnesses, min_evaluations = self.fitness_eval(self.population.position)
        self.population.fitness[:min_evaluations] = fitnesses
    
    def population_selection(self) -> None:
        ''' Selects the best particles from the previous (before applying the equation of motion) and current populations. The top :attr:~mesh.parameters.MeshParameters.population_size particles, i.e., those with the lowest domination rank, are chosen. In case of a tie, particles with the largest crowding distance are selected.
        
        Note:
            The domination ranks are ordered from the lowest to the highest, starting at the Pareto front with zero.
        '''

        population_size = self.params.population_size
        pre_allocated = self.pre_allocated
        # Get the fitness matrix with the previous and the current population
        pre_allocated.fitness_selection[:population_size] = pre_allocated.fitness_copy
        pre_allocated.fitness_selection[population_size:] = self.population.fitness
        # Find the best N indices
        best_N_idxs = select_best_N_mo(pre_allocated.fitness_selection, population_size)
        # Separate the previous and current population indices from best_N_idxs
        mask = best_N_idxs < population_size
        prev_idxs = best_N_idxs[mask]
        # Get the current indices
        np.logical_not(mask, out=mask)
        current_idxs = best_N_idxs[mask] - population_size
        # Get the previous and the current size of indices
        prev_idx_size = len(prev_idxs)
        # Select the best previous particles
        self.population.position[:prev_idx_size] = pre_allocated.position_copy[prev_idxs]
        self.population.velocity[:prev_idx_size] = pre_allocated.velocity_copy[prev_idxs]
        self.population.fitness[:prev_idx_size] = pre_allocated.fitness_copy[prev_idxs]
        # Select the best current particles
        self.population.position[prev_idx_size:] = self.population.position[current_idxs]
        self.population.velocity[prev_idx_size:] = self.population.velocity[current_idxs]
        self.population.fitness[prev_idx_size:] = self.population.fitness[current_idxs]
        # Select the best N personal best
        pb_idxs = np.concatenate((prev_idxs, current_idxs), axis=0)
        self.population.personal_best_fit[:] = self.population.personal_best_fit[pb_idxs]
        self.population.personal_best_pos[:] = self.population.personal_best_pos[pb_idxs]

    def update_personal_best(self, pop_indices: np.ndarray[np.integer]) -> None:
        ''' Updates the personal guides of the particles by the population index.

        Note:
            There is three cases to update the personal guides:

            - When the current particle is dominated by any of its personal guide, the current particle is ignored;
            - When the current particle dominates a personal guide, the current particle replaces the dominated personal guide. This replacement is done for all dominated personal guides, so the more the current particle dominates its personal guides, the more chance it has of being sampled in :meth:`move_population`;
            - When the current particle don't dominate and is not dominated by any personal guide, the current particle is added to the personal guide matrix. The oldest personal guide is removed when the current particle is only added.
        
        Args:
            pop_indices (:type:`np.ndarray[np.integer]`): A numpy array with the indices of the population particles to update the personal guides.
        '''

        # Get the population fitness as a tensor
        fitness_tensor = self.population.fitness[pop_indices, np.newaxis]
        # Get the personal best fitness
        pb_fitness = self.population.personal_best_fit[pop_indices]
        # Get the mask to update the personal best
        update_mask = ~np.any(self.dominates(pb_fitness, fitness_tensor, axis=2), axis=1)
        update_idxs = pop_indices[update_mask]
        # Get the mask to replace the personal best dominated by the current particle
        replace_mask = self.dominates(fitness_tensor[update_mask], pb_fitness[update_mask], axis=2)
        # Replace the dominated personal best by the current particle
        replace_row, replace_col = np.nonzero(replace_mask)
        particle_to_replace_pb = update_idxs[replace_row]
        self.population.personal_best_fit[particle_to_replace_pb, replace_col, :] =  self.population.fitness[particle_to_replace_pb, :]
        self.population.personal_best_pos[particle_to_replace_pb, replace_col, :] =  self.population.position[particle_to_replace_pb, :]
        # Get the mask to add the current to the personal best list
        add_idxs = update_idxs[~np.any(replace_mask, axis=1)]
        # Delete the oldest personal best and include the current particle as a new personal best
        self.population.personal_best_fit[add_idxs, 1:, :] = self.population.personal_best_fit[add_idxs, :-1, :]
        self.population.personal_best_pos[add_idxs, 1:, :] = self.population.personal_best_pos[add_idxs, :-1, :]
        # Update the personal best list by adding the current particle as a new personal best
        self.population.personal_best_fit[add_idxs, 0, :] = self.population.fitness[add_idxs, :]
        self.population.personal_best_pos[add_idxs, 0, :] = self.population.position[add_idxs, :]

    def update_memory(self):
        ''' Updates the memory position and fitness using the Pareto front formed by the particles from previous memory particles and the current Pareto front. '''
        
        # Get the indices of the Pareto front
        pareto_idxs = self.fronts[0]
        # Get the unique positions from the Pareto front and the memory
        position_matrix = np.concatenate((self.population.position[pareto_idxs], self.memory.position), axis=0)
        position_matrix, unique_idxs = np.unique(position_matrix, axis=0, return_index=True)
        # Get the unique fitnesses from the Pareto front and the memory
        fitness_matrix = np.concatenate((self.population.fitness[pareto_idxs], self.memory.fitness), axis=0)[unique_idxs]
        # Get the Pareto front indices from the memory candidates
        memory_pareto_front_idxs = self.get_domination_fronts(fitness_matrix)[0][0]
        # If the new memory Pareto front has size less or equal than the memory size, then set the new memory
        memory_size = self.params.memory_size
        if(len(memory_pareto_front_idxs) <= memory_size):
            self.memory.position = position_matrix[memory_pareto_front_idxs]
            self.memory.fitness = fitness_matrix[memory_pareto_front_idxs]
        # Else get the particles with the highest crowd distance in the new memory Pareto front
        else:
            # Select the particles with the highest crowd distance
            selected_fitness = fitness_matrix[memory_pareto_front_idxs]
            # Calculate the crowding distance
            crowd_distances = crowding_distance(selected_fitness)
            # Get the indices of the particles with the highest crowd distance
            idxs = np.argpartition(crowd_distances, -memory_size)[-memory_size:]
            # Update the memory
            self.memory.position = position_matrix[memory_pareto_front_idxs[idxs]]
            self.memory.fitness = selected_fitness[idxs]

    def run(self):
        ''' This method runs the MESH algorithm. It stops when the maximum number of generations and/or fitness evaluations is reached. '''

        try:
            # Start the progress bars
            with tqdm(total=self.total_bar, leave=False) as pbar:
                # A variable to update the tqdm bar
                prev_bar_value = 0
                # Evaluate the initial population
                fitnesses, min_evaluations = self.fitness_eval(self.population.position)
                self.population.fitness[:min_evaluations] = fitnesses
                # Repeat the population fitness for all personal best input
                self.population.personal_best_fit[:, :, :] = np.repeat(fitnesses[:, np.newaxis, :], self.params.max_personal_guides, axis=1)
                # Get the population fronts and domination ranks
                self.fronts, self.population.rank = self.get_domination_fronts(self.population.fitness)
                # Initialize the memory
                self.memory = Memory(self.population, self.fronts[0], self.params)
                # Main loop
                while True:
                    # Count generations if it is a stopping criterion
                    self.count_generation()
                    # Calculate Xst for each particle
                    self.differential_mutation()
                    # Mutate the weights
                    self.mutate_weights()
                    # Update global best
                    self.global_best_attribution()
                    # Store some data of the population before the movement
                    self.pre_allocated.position_copy[:] = self.population.position.copy()
                    self.pre_allocated.velocity_copy[:] = self.population.velocity.copy()
                    self.pre_allocated.fitness_copy[:] = self.population.fitness.copy()
                    # Apply the movviment to the particles
                    self.move_population()
                    # Select the best particles from those before and after movement
                    self.population_selection()
                    # Update the personal best
                    self.update_personal_best(np.arange(self.params.population_size))
                    # Get the fronts
                    self.fronts, self.population.rank = self.get_domination_fronts(self.population.fitness)
                    # Update memory
                    self.update_memory()
                    # Update the progress bar
                    prev_bar_value = self.update_progress_bar(pbar, prev_bar_value)
        # The end of the algorithm
        except StoppingAlgorithm:
            # Get the fronts
            self.fronts, self.population.rank = self.get_domination_fronts(self.population.fitness)
            # Update memory
            self.update_memory()
            # Log the memory
            if self.log_memory:
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
    
    def update_progress_bar_by_generation(self, pbar, prev_bar_value):
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
            raise StoppingAlgorithm()
    
    def stopping_by_fitness_eval(self, X: np.ndarray[np.float64, 2]) -> tuple[np.ndarray[np.float64, 2], int]:
        ''' Evaluates the position matrix ``X`` and counts the fitness evaluations. This method is used when the stopping criterion is by fitness evaluations.
        
        Args:
            X (:type:`np.ndarray[np.float64, 2]`): A numpy matrix with the particle positions.
            
        Returns:
            :type:`tuple[np.ndarray[np.float64, 2], int]`: A tuple with the fitness matrix and the minimum number of evaluations that doesn't stop the algorithm.
        
        Raises:
            :class:`~mesh.utils.auxiliar.StoppingAlgorithm`: If the number of fitness evaluations is greater than the maximum number of fitness evaluations.    
        '''

        # Check if the stopping criterion reached
        if self.fitness_eval_counter >= self.params.max_fit_eval:
            raise StoppingAlgorithm()
        # Calculate the minimum number of fitness evaluations
        min_evaluations = min(self.params.max_fit_eval - self.fitness_eval_counter, len(X))
        # Update the fitness counter
        self.fitness_eval_counter += min_evaluations
        # Slice the particle positions for the minimum evaluations
        X_min = X[:min_evaluations]
        return self.fitness_evaluations(X_min)

    def get_results(self) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.float64, 2]]:
        ''' Returns a tuple with the memory position and fitness, respectively.
        
        Note:
            This method must be used at the end of the algorithm.
        
        Returns:
            :type:`tuple[np.ndarray[np.float64, 2], np.ndarray[np.float64, 2]]`: A tuple with the memory position and fitness, respectively.
        '''

        return self.memory.position, self.memory.fitness

    def logging(self) -> None:
        ''' Logs memory position and fitness at the end of the algorithm in two .txt files if :attr:`log_memory` is a string. Then this method uses the string value :attr:`log_memory` at the beginning of both files as the name of the fitness and position logs.
        '''

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
        for pos in self.memory.position:
            string = ""
            for i in range(self.params.position_dim):
                string += str(pos[i])+" "
            string = string[:-1]
            memory_position += string + ", "
        memory_position = memory_position[:-2]
        memory_position += "\n"
        file2.write(memory_position)
        file2.close()