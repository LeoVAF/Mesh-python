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


from Particles import *
from Auxiliar import *

from scipy.stats import truncnorm
from tqdm import tqdm
from pygmo import fast_non_dominated_sorting, select_best_N_mo
from copy import deepcopy

# import tracemalloc
# tracemalloc.start()
# current, peak = tracemalloc.get_traced_memory()
# print(f"Memória atual: {current / 10**6:.2f} MB; Pico de memória: {peak / 10**6:.2f} MB")
# tracemalloc.stop()

''' MESH parameters '''
class MESH_Params:
    ''' Initialize the instance '''
    def __init__(self,
                 objective_dim, # Number of objectives
                 position_dim, # Design space dimension
                 position_max_value, # A array with each upper bound of problem
                 position_min_value, # A array with each lower bound of problem
                 population_size, # Population size
                 memory_size, # Number of particles in memory
                 global_best_attribution_type, # 0 -> E1 | 1 -> E2 | 2 -> E3 | 3 -> E4 (E3 and E4 with problem)
                 de_mutation_type, # 0 -> DE\rand\1\Bin (D1) | 1 -> DE\rand\2\Bin (D2) | 2 -> DE/Best/1/Bin (D3) | 3 -> DE/Current-to-best/1/Bin (D4) | 4 -> DE/Current-to-rand/1/Bin (D5)
                 dm_pool_type, # Sampling vectors 0 -> swarm (V1) | 1 -> memory (V2) | 2 -> both swarm and memory (V3)
                 crowding_distance_type, # 0 -> Crowding Distance Tradicional (C1)
                 communication_probability, # Communication probability
                 mutation_rate, # Mutation rate
                 max_gen=0, # Maximum number of generations (not used if it less than one)
                 max_fit_eval=0, # Maximum number of fitness evaluations (not used if it is less than one)
                 max_personal_guides=3, # Maximum number of personal guides (greater than zero)
                 random_state = None): # Numpy random seed to generate random numbers
        
        # Set the number of objectives
        self.objective_dim = objective_dim
        # Set the maximum number of fitness evaluations
        if(max_gen == 0):
            self.max_fit_eval = max_fit_eval
        else:
            self.max_fit_eval = max(population_size*(2*max_gen+1), max_fit_eval)
        # Set the position dimension and the position boundaries
        self.position_dim = position_dim
        self.position_max_value = position_max_value
        self.position_min_value = position_min_value
        # Set the maximum and minimum velocities
        self.velocity_max_value = self.position_max_value - self.position_min_value
        self.velocity_min_value = -self.velocity_max_value
        # Set the population and the memory sizes
        self.population_size = population_size
        self.memory_size = memory_size
        # Set the strategies and operators
        self.global_best_attribution_type = global_best_attribution_type
        self.de_mutation_type = de_mutation_type
        self.dm_pool_type = dm_pool_type
        self.crowding_distance_type = crowding_distance_type
        # Set the communication and the mutation rates
        self.communication_probability = communication_probability
        self.mutation_rate = mutation_rate
        # Set the number of personal guides
        self.max_personal_guides = max_personal_guides
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
        self.fitness_eval_count = 0
        # Create a random matrix (4 x population_size) for w1 and two random vectors for w2 and w3
        self.weights = np.random.uniform(0.0, 1.0, [4, params.population_size])
        # Store some pre-calculated data
        self.pre_calculated = PreCalculated(params.objective_dim, params.position_dim, params.population_size)
        # Variable for logging memory
        self.log_memory = log_memory

    ''' Initialize the population randomly '''
    def init_population_randomly(self):
        # Evaluate the initial population
        fitnesses, min_evaluations = self.fitness_evaluations(self.population.position)
        self.population.fitness[:min_evaluations] = fitnesses
    
    ''' Evaluate the fitness given a particle position matrix '''
    def fitness_evaluations(self, X):
        # Calculate remanining evaluations
        remaining_eval = self.params.max_fit_eval - self.fitness_eval_count
        # Check if the stopping criterion reached
        if remaining_eval == 0:
            raise StoppingAlgorithm()
        # Calculate the minimum number of fitness evaluations
        min_evaluations = min(remaining_eval, len(X))
        # Update the fitness counter
        self.fitness_eval_count += min_evaluations
        # Slice the particle positions for the minimum evaluations
        X_min = X[:min_evaluations]
        return np.array([self.fitness_function(x) for x in X_min], copy=False), min_evaluations
    
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
        # Generating random indexes for each sublist
        personal_guides = self.population.personal_best_list
        random_indices = np.random.randint(0, [len(pb_list) for pb_list in personal_guides])
        # Get matrix of personal best list positions
        pb_positions = np.array([pb_list[idx].position for pb_list, idx in zip(personal_guides, random_indices)], copy=False)
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
        matrix_for_operations = self.pre_calculated.matrix_for_operations
        np.subtract(pb_positions, positions, out=matrix_for_operations)
        np.multiply(matrix_for_operations, weights[1][:, np.newaxis], out=matrix_for_operations)
        np.add(velocities, matrix_for_operations, out=velocities)
        # Calculate the cooperation term
        vector_for_operations = self.pre_calculated.vector_for_operations
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
        fitnesses, min_evaluations = self.fitness_evaluations(self.population.position)
        self.population.fitness[:min_evaluations] = fitnesses

    ''' Make the selection of the population between the previous and current population '''
    def population_selection(self):
        population_size = self.params.population_size
        population_copy = self.pre_calculated
        # Get the fitness matrix with the previous and the current population
        fitness_matrix = np.concatenate((population_copy.fitness_copy, self.population.fitness), axis=0)
        # Find the best N indexes
        best_N_idxs = select_best_N_mo(fitness_matrix, population_size)
        # Separate the previous and current population indexes from best_N_idxs
        prev_mask = best_N_idxs < population_size
        prev_idxs = best_N_idxs[prev_mask]
        current_idxs = best_N_idxs[~prev_mask] - population_size
        # Select the best N particles
        self.population.position[:, :] = np.concatenate((population_copy.position_copy[prev_idxs], self.population.position[current_idxs]), axis=0)
        self.population.velocity[:, :] = np.concatenate((population_copy.velocity_copy[prev_idxs], self.population.velocity[current_idxs]), axis=0)
        self.population.fitness[:, :] = np.concatenate((population_copy.fitness_copy[prev_idxs], self.population.fitness[current_idxs]), axis=0)
        self.population.personal_best_list[:] = deepcopy(self.population.personal_best_list[np.concatenate((prev_idxs, current_idxs), axis=0)])

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
            fitnesses, min_evaluations = self.fitness_evaluations(xst)
            min_valid_idxs = valid_idxs[:min_evaluations]
            pop_fitnesses = self.population.fitness[min_valid_idxs]
            domination_mask = self.np_dominate(fitnesses, pop_fitnesses, axis=1)
            update_idxs = min_valid_idxs[domination_mask]
            # Update the positions and the fitnesses
            self.population.position[update_idxs] = xst[:min_evaluations][domination_mask]
            self.population.fitness[update_idxs] = fitnesses[domination_mask]
            # If a particle was replaced for a particle from a strategy update some information
            if len(update_idxs):
                self.update_personal_best(update_idxs)
                self.fronts, self.population.rank = self.get_domination_fronts(self.population.fitness)
                self.memory_update()

    ''' Update the memory '''
    def memory_update(self):
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
            # Get the indexes of the particles with the highest crowd distance
            idxs = np.argpartition(crowd_distances, -memory_size)[-memory_size:]
            # Update the memory
            self.memory.position = position_matrix[memory_pareto_front_idxs[idxs]]
            self.memory.fitness = selected_fitness[idxs]

    ''' Update the list of particle's personal best '''
    def update_personal_best(self, indices):
        for idx in indices:
            # Get the current personal best list
            pb_list = self.population.personal_best_list[idx]
            # Get the current particle fitness
            fitness = self.population.fitness[idx]
            # Particles that will be removed from the list of personal best
            removal_particles = set()
            # Flag to indicate that a particle from personal best list will be removed
            is_changed = False
            # Check if the current particle is better than the previous particles from the list of personal best
            for pb in pb_list:
                # If the current particle is dominated by at least one personal best, then ignore the current particle
                if(self.dominates(pb.fitness, fitness)):
                    return
                # If the current particle dominates this personal best, then add this particle in a removal list
                elif(self.dominates(fitness, pb.fitness)):
                    is_changed = True
                    removal_particles.add(pb)
            # Filter the personal bests if there are changes
            if is_changed:
                # Update the personal best list
                pb_list = deque([pb for pb in pb_list if pb not in removal_particles], maxlen=self.params.max_personal_guides)
            # Create a new personal best and include it to the list
            pb_list.appendleft(PBest(self.population.position[idx], fitness))
            # Update the personal best list
            self.population.personal_best_list[idx] = pb_list

    ''' Run the MESH '''
    def run(self):
        try:
            # Start the progress bars
            with tqdm(total=self.params.max_fit_eval, leave=False) as pbar:
                # A variable to update the tqdm bar
                prev_fitness_eval = 0
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
                    self.pre_calculated.position_copy[:] = self.population.position.copy()
                    self.pre_calculated.velocity_copy[:] = self.population.velocity.copy()
                    self.pre_calculated.fitness_copy[:] = self.population.fitness.copy()
                    # Apply the movviment to the particles
                    self.move_population()
                    # Select the best particles from those before and after movement
                    self.population_selection()
                    # Update the personal best
                    self.update_personal_best(np.arange(self.params.population_size))
                    # Get the fronts
                    self.fronts, self.population.rank = self.get_domination_fronts(self.population.fitness)
                    # Update memory
                    self.memory_update()
                    # Update the progress bar
                    delta_evals = self.fitness_eval_count - prev_fitness_eval
                    pbar.update(delta_evals)
                    prev_fitness_eval = self.fitness_eval_count
        # The end of the algorithm
        except StoppingAlgorithm:
            ''' ############################################################################### '''
            # Population_selection() here?
            ''' ############################################################################### '''
            # Get the fronts
            self.fronts, self.population.rank = self.get_domination_fronts(self.population.fitness)
            # Update memory
            self.memory_update()
            # Log the memory
            if self.log_memory:
                self.logging()

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