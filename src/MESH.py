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


from utils.Particles import *
from utils.Auxiliar import *

from scipy.stats import truncnorm
from tqdm import tqdm
from pygmo import fast_non_dominated_sorting, select_best_N_mo

# import tracemalloc
# tracemalloc.start()
# current, peak = tracemalloc.get_traced_memory()
# print(f"Memória atual: {current / 10**6:.2f} MB; Pico de memória: {peak / 10**6:.2f} MB")
# tracemalloc.stop()

''' MESH parameters '''
class MeshParameters:
    ''' Initialize the instance '''
    def __init__(self,
                 objective_dim, # Number of objectives
                 position_dim, # Design space dimension
                 position_max_value, # A array with each upper bound of problem
                 position_min_value, # A array with each lower bound of problem
                 population_size, # Population size
                 memory_size=None, # Number of particles in memory
                 global_best_attribution_type=0, # 0 -> E1 | 1 -> E2 | 2 -> E3 | 3 -> E4 (E3 and E4 with problem)
                 dm_pool_type=0, # Sampling vectors 0 -> swarm (V1) | 1 -> memory (V2) | 2 -> both swarm and memory (V3)
                 de_mutation_type=0, # 0 -> DE\rand\1\Bin (D1) | 1 -> DE\rand\2\Bin (D2) | 2 -> DE/Best/1/Bin (D3) | 3 -> DE/Current-to-best/1/Bin (D4) | 4 -> DE/Current-to-rand/1/Bin (D5)
                 communication_probability=0.7, # Communication probability
                 mutation_rate=0.9, # Mutation rate
                 max_gen=0, # Maximum number of generations (not used if it less than one)
                 max_fit_eval=0, # Maximum number of fitness evaluations (not used if it is less than one)
                 max_personal_guides=3, # Maximum number of personal guides (greater than zero)
                 random_state = None): # Numpy random seed to generate random numbers
        
        # Set the number of objectives
        self.objective_dim = objective_dim
        if objective_dim < 1:
            raise ValueError('The number of objectives must be greater than 0!')
        elif type(objective_dim) != int:
            raise TypeError('The input objective_dim must be an integer!')
        # Set the maximum number of fitness evaluations
        self.max_fit_eval = max(max_fit_eval, 0)
        # Set the maximum number o generations
        self.max_gen = max(max_gen, 0)
        if max_gen < 1 and max_fit_eval < 1:
            raise ValueError('Either maximum generation or maximum fitness eval must be greater than 0!')
        # Set the position dimension and the position boundaries
        self.position_dim = position_dim
        if position_dim < 1:
            raise ValueError('The number of decision variables must be greater than 0!')
        elif type(objective_dim) != int:
            raise TypeError('The input position_dim must be an integer!')
        self.position_max_value = position_max_value
        self.position_min_value = position_min_value
        # Set the maximum and minimum velocities
        self.velocity_max_value = self.position_max_value - self.position_min_value
        self.velocity_min_value = -self.velocity_max_value
        # Set the population size
        self.population_size = population_size
        if population_size < 1:
            raise ValueError('The number of particles must be greater than 0!')
        elif type(objective_dim) != int:
            raise TypeError('The input population_size must be an integer!')
        # Set the memory size
        self.memory_size = memory_size
        if memory_size == None:
            self.memory_size = population_size
        elif memory_size < 1:
            raise ValueError('Memory size must be greater than 0!')
        elif type(memory_size) != int:
            raise TypeError('The input memory_size must be an integer!')
        # Set the global attribution type
        self.global_best_attribution_type = global_best_attribution_type
        if global_best_attribution_type not in {0,1,2,3}:
            raise ValueError(f'The input global best attribution type must be one of these options: {(0,1,2,3)}!')
        # Set the differential mutation type
        self.de_mutation_type = de_mutation_type
        if de_mutation_type not in {0,1,2,3,4}:
            raise ValueError(f'The input differential mutation type must be one of these options: {(0,1,2,3,4)}!')
        # Set the differential mutation pool type
        self.dm_pool_type = dm_pool_type
        if dm_pool_type not in {0,1,2}:
            raise ValueError(f'The input cdifferential mutation pool type type must be one of these options: {(0,1,2)}!')
        # Set the communication probability
        if 0 <= communication_probability <= 1:
            self.communication_probability = communication_probability
        else:
            raise ValueError('Communication probability must be a number between 0 and 1!')
        # Set the mutation rate
        if 0 <= mutation_rate <= 1:
            self.mutation_rate = mutation_rate
        else:
            raise ValueError('Mutation rate must be a number between 0 and 1!')
        # Set the number of personal guides
        self.max_personal_guides = max_personal_guides
        if max_personal_guides < 1:
            raise ValueError('Maximum personal guides must be greater than 0!')
        elif type(max_personal_guides) != int:
            TypeError('The input max_personal_guides must be an integer!')
        # Set the random state (if different from None)
        self.random_state = random_state
        
''' Algoritmo MESH inheriting operations from the Operation class '''
class MESH(Operation):
    ''' Initialize the instance '''
    def __init__(self,
                 params, # MESH parameters
                 fitness_function, # A fitness function that returns a numpy array with each value in the respective component
                 log_memory=False): # A string to log the memory (the name of the files will use this string)
        
        # Receive the algorithm parameters
        self.params = params
        # Initizaling Operation class with some operations of MESH
        super().__init__()
        # Chosing the operations just one time
        self.global_best_attribution = super().get_global_best_attribution(params.global_best_attribution_type)
        self.differential_mutation_pool = super().get_differential_mutation_pool(params.dm_pool_type)
        self.differential_mutation_strategy = super().get_differential_mutation_strategy(params.de_mutation_type)
        # Use a random seed if there is
        if(params.random_state):
            np.random.seed(params.random_state)
        # Particles
        self.population = Particles(params.population_size,
                                    params.max_personal_guides,
                                    params.objective_dim,
                                    params.position_dim,
                                    (params.position_min_value, params.position_max_value),
                                    (params.velocity_min_value, params.velocity_max_value),
                                    params.global_best_attribution_type)
        # Memory particles (and the final result after run MESH)
        self.memory = Memory(params.global_best_attribution_type)
        # Frontiers (a list of numpy arrays with index of each particle in the respective frontier)
        self.fronts = []
        # Estabilish the fitness function and start the fitness counter
        self.fitness_function = fitness_function
        self.fitness_eval_counter = 0
        # Start the generation counter
        self.generation_counter = 0
        # Create a random matrix (4 x population_size) for w1 and two random vectors for w2 and w3
        self.weights = np.random.uniform(0.0, 1.0, [4, params.population_size])
        # Store some pre-calculated data
        self.pre_alocated = PreAlocated(params.objective_dim, params.position_dim, params.population_size)
        # Variable for logging memory
        self.log_memory = log_memory
        # Check if generation is a stopping criterion
        if self.params.max_gen > 0:
            self.count_generation = self.stopping_by_generation
        else:
            self.count_generation = lambda : None
        # Check if the fitness evaluation is a stopping criterion
        if self.params.max_fit_eval > 0:
            self.count_fitness_eval = self.stopping_by_fitness_eval
        else:
            self.count_fitness_eval = self.fitness_evaluations
        # Choose the way to update the algorithm progress bar
        if self.params.max_gen == 0:
            self.total_bar = params.max_fit_eval
            self.update_progress_bar = self.update_progress_bar_by_fitness_evaluation
        elif self.params.max_fit_eval == 0:
            self.total_bar = params.max_gen
            self.update_progress_bar = self.update_progress_bar_by_generation
        else:
            self.update_progress_bar = self.update_progress_bar_by_fitness_evaluation
            self.total_bar = min(params.population_size*(2*params.max_gen+1), params.max_fit_eval)

    ''' Initialize the population randomly '''
    def init_population_randomly(self):
        # Evaluate the initial population
        fitnesses, min_evaluations = self.count_fitness_eval(self.population.position)
        self.population.fitness[:min_evaluations] = fitnesses
        # Repeat the population fitness for all personal best input
        self.population.personal_best_list_fit[:, :, :] = np.repeat(fitnesses[:, np.newaxis, :], self.params.max_personal_guides, axis=1)
    
    ''' Evaluate the fitness given a particle position matrix '''
    def fitness_evaluations(self, X):
        return np.array([self.fitness_function(x) for x in X]), len(X)
    
    ''' Check if an array x dominates an array or matrix (axis=1) y (vectorized) '''
    def np_dominate(self, x, y, axis=0):
        return np.all(x <= y, axis=axis) & np.any(x < y, axis=axis)
    
    ''' Check if an array x dominates an array y '''
    def dominates(self, x, y):
        dominates = False
        for xi, yi in zip(x, y):
            if xi > yi:
                return False
            elif xi < yi:
                dominates = True
        return dominates

    ''' Update the population frontiers '''
    def get_domination_fronts(self, fitness_matrix):
        # If there is only one particle in the particle list, then it is the Pareto frontier by itself
        if(len(fitness_matrix) == 1):
            return np.array([np.array([0])]), np.array([0])
        # Do the Fast Non-dominated Sorting from Pygmo
        non_dominated_fronts, _, _, ranks = fast_non_dominated_sorting(points=fitness_matrix)
        return non_dominated_fronts, ranks
    
    ''' ################################################################################################################################################## '''
    ''' Reverses the direction of each component of the velocity that took the particle out of its respective limits (applied after the movement) '''
    def reflect_velocity_at_bounds(self, velocity_input, position_input):
        neg_velocity = (velocity_input < 0)
        return np.where(((position_input == self.params.position_min_value) & neg_velocity) |
                        ((position_input == self.params.position_max_value) & (~ neg_velocity)),
                        -velocity_input,
                        velocity_input)
    ''' ################################################################################################################################################## '''

    ''' Apply the equation of motion to the particles '''
    def move_population(self):
        # Get the parameters
        params = self.params
        # Get the population size and the position dimension
        population_size = params.population_size
        # Generating random indices for each sublist
        random_indices = np.random.randint(0, self.params.max_personal_guides, size=population_size)
        # Get matrix of personal best list positions
        pb_positions = self.population.personal_best_list_pos[np.arange(population_size), random_indices, :]
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
        matrix_for_operations = self.pre_alocated.matrix_for_operations
        np.subtract(pb_positions, positions, out=matrix_for_operations)
        np.multiply(matrix_for_operations, weights[1][:, np.newaxis], out=matrix_for_operations)
        np.add(velocities, matrix_for_operations, out=velocities)
        # Calculate the cooperation term
        vector_for_operations = self.pre_alocated.vector_for_operations
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
        fitnesses, min_evaluations = self.count_fitness_eval(self.population.position)
        self.population.fitness[:min_evaluations] = fitnesses

    ''' Make the selection of the population between the previous and current population '''
    def population_selection(self):
        population_size = self.params.population_size
        pre_alocated = self.pre_alocated
        # Get the fitness matrix with the previous and the current population
        pre_alocated.fitness_selection[:population_size] = pre_alocated.fitness_copy
        pre_alocated.fitness_selection[population_size:] = self.population.fitness
        # Find the best N indices
        best_N_idxs = select_best_N_mo(pre_alocated.fitness_selection, population_size)
        # Separate the previous and current population indices from best_N_idxs
        mask = best_N_idxs < population_size
        prev_idxs = best_N_idxs[mask]
        # Get the current indices
        np.logical_not(mask, out=mask)
        current_idxs = best_N_idxs[mask] - population_size
        # Get the previous and the current size of indices
        prev_idx_size = len(prev_idxs)
        # Select the best previous particles
        self.population.position[:prev_idx_size] = pre_alocated.position_copy[prev_idxs]
        self.population.velocity[:prev_idx_size] = pre_alocated.velocity_copy[prev_idxs]
        self.population.fitness[:prev_idx_size] = pre_alocated.fitness_copy[prev_idxs]
        # Select the best current particles
        self.population.position[prev_idx_size:] = self.population.position[current_idxs]
        self.population.velocity[prev_idx_size:] = self.population.velocity[current_idxs]
        self.population.fitness[prev_idx_size:] = self.population.fitness[current_idxs]
        # Select the best N personal best
        pb_idxs = np.concatenate((prev_idxs, current_idxs), axis=0)
        self.population.personal_best_list_fit[:] = self.population.personal_best_list_fit[pb_idxs]
        self.population.personal_best_list_pos[:] = self.population.personal_best_list_pos[pb_idxs]

    ''' Mutate the weights by a truncated normal distribution '''
    def mutate_weights(self):
        # Get the values from truncated normal distribution
        self.weights[:, :] = truncnorm.rvs(0, 1, size=(4, self.params.population_size)) * self.params.mutation_rate
    
    ''' Apply a strategy from differential evolution '''
    def differential_mutation(self):
        # A array of a matrix pool in each row
        xr_pool_tensor = self.differential_mutation_pool()
        # Apply a strategy
        xst, valid_idxs = self.differential_mutation_strategy(xr_pool_tensor)
        if len(xst):
            # Update the current particle if the new particle from the strategy is better
            st_fitnesses, min_evaluations = self.count_fitness_eval(xst)
            min_valid_idxs = valid_idxs[:min_evaluations]
            valid_pop_fitnesses = self.population.fitness[min_valid_idxs]
            domination_mask = self.np_dominate(st_fitnesses, valid_pop_fitnesses, axis=1)
            update_idxs = min_valid_idxs[domination_mask]
            # Update the positions and the fitnesses
            self.population.position[update_idxs] = xst[:min_evaluations][domination_mask]
            self.population.fitness[update_idxs] = st_fitnesses[domination_mask]
            # If a particle was replaced for a particle from a strategy update some information
            if len(update_idxs):
                self.update_personal_best(update_idxs)
                self.fronts, self.population.rank = self.get_domination_fronts(self.population.fitness)
                self.update_memory()

    ''' Update the memory '''
    def update_memory(self):
        # Get the indices of the Pareto frontier
        Pareto_idxs = self.fronts[0]
        # Get the unique positions from the Pareto frontier and the memory
        position_matrix = np.concatenate((self.population.position[Pareto_idxs], self.memory.position), axis=0)
        position_matrix, unique_idxs = np.unique(position_matrix, axis=0, return_index=True)
        # Get the unique fitnesses from the Pareto frontier and the memory
        fitness_matrix = np.concatenate((self.population.fitness[Pareto_idxs], self.memory.fitness), axis=0)[unique_idxs]
        # Get the Pareto frontier indices from the memory candidates
        memory_pareto_front_idxs = self.get_domination_fronts(fitness_matrix)[0][0]
        # If the new memory Pareto frontier has size less or equal than the memory size, then set the new memory
        memory_size = self.params.memory_size
        if(len(memory_pareto_front_idxs) <= memory_size):
            self.memory.position = position_matrix[memory_pareto_front_idxs]
            self.memory.fitness = fitness_matrix[memory_pareto_front_idxs]
        # Else get the particles with the highest crowd distance in the new memory Pareto frontier
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

    ''' Update the list of particle's personal best '''
    def update_personal_best(self, indices):
        # Get the population fitness as a tensor
        fitness_tensor = self.population.fitness[indices, np.newaxis]
        # Get the personal best fitness and position
        pb_fitness = self.population.personal_best_list_fit[indices]
        # Get the mask to update the personal best
        update_mask = ~np.any(self.np_dominate(pb_fitness, fitness_tensor, axis=2), axis=1)
        update_idxs = indices[update_mask]
        # Get the mask to replace the personal best dominated by the current particle
        replace_mask = self.np_dominate(fitness_tensor[update_mask], pb_fitness[update_mask], axis=2)
        # Replace the dominated personal best by the current particle
        replace_row, replace_col = np.nonzero(replace_mask)
        particle_to_replace_pb = update_idxs[replace_row]
        self.population.personal_best_list_fit[particle_to_replace_pb, replace_col, :] =  self.population.fitness[particle_to_replace_pb, :]
        self.population.personal_best_list_pos[particle_to_replace_pb, replace_col, :] =  self.population.position[particle_to_replace_pb, :]
        # Get the mask to add the current to the personal best list
        add_idxs = update_idxs[~np.any(replace_mask, axis=1)]
        # Delete the oldest personal best and include the current particle as a new personal best
        self.population.personal_best_list_fit[add_idxs, 1:, :] = self.population.personal_best_list_fit[add_idxs, :-1, :]
        self.population.personal_best_list_pos[add_idxs, 1:, :] = self.population.personal_best_list_pos[add_idxs, :-1, :]
        # Update the personal best list by adding the current particle as a new personal best
        self.population.personal_best_list_fit[add_idxs, 0, :] = self.population.fitness[add_idxs, :]
        self.population.personal_best_list_pos[add_idxs, 0, :] = self.population.position[add_idxs, :]

    ''' Run the MESH '''
    def run(self):
        try:
            # Start the progress bars
            with tqdm(total=self.total_bar, leave=False) as pbar:
                # A variable to update the tqdm bar
                prev_bar_value = 0
                # Initialize population
                self.init_population_randomly()
                # get the population frontiers and ranks
                self.fronts, self.population.rank = self.get_domination_fronts(self.population.fitness)
                # Initialize the memory
                self.memory.init(self.population, self.fronts[0], self.params.memory_size)
                # Main loop
                while True:
                    # Calculate Xst for each particle
                    self.differential_mutation()
                    # Mutate the weights
                    self.mutate_weights()
                    # Update global best
                    self.global_best_attribution()
                    # Store some data of the population before the movement
                    self.pre_alocated.position_copy[:] = self.population.position.copy()
                    self.pre_alocated.velocity_copy[:] = self.population.velocity.copy()
                    self.pre_alocated.fitness_copy[:] = self.population.fitness.copy()
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
                    # Count generations if it is a stopping criterion
                    self.count_generation()
        # The end of the algorithm
        except StoppingAlgorithm:
            ''' ############################################################################### '''
            # Population_selection() here?
            ''' ############################################################################### '''
            # Get the fronts
            self.fronts, self.population.rank = self.get_domination_fronts(self.population.fitness)
            # Update memory
            self.update_memory()
            # Log the memory
            if self.log_memory:
                self.logging()

    ''' Update the progress bar by fitness evaluations '''
    def update_progress_bar_by_fitness_evaluation(self, pbar, prev_bar_value):
        pbar.update(self.fitness_eval_counter - prev_bar_value)
        return self.fitness_eval_counter
    
    ''' Update the progress bar by generations '''
    def update_progress_bar_by_generation(self, pbar, prev_bar_value):
        pbar.update(self.generation_counter - prev_bar_value)
        return self.generation_counter

    ''' Count generations if it is a stopping criterion '''
    def stopping_by_generation(self):
        self.generation_counter += 1
        if self.generation_counter >= self.params.max_gen:
            raise StoppingAlgorithm()
    
    ''' Count fitness evaluations if it is a stopping criterion '''
    def stopping_by_fitness_eval(self, X):
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

    ''' Return the algorithm results '''
    def get_results(self):
        return self.memory.position, self.memory.fitness

    ''' Log the memory '''
    def logging(self):
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