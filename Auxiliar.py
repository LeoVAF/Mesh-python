import numpy as np
from sklearn.neighbors import NearestNeighbors

''' MESH operations (used in MESH) '''
class Operation():
    ''' Initialize the instance '''
    def __init__(self):
        # The options of global guide attribution
        self.global_best_attribution_type = {
            0: self.sigma_method_in_memory,
            1: self.sigma_method_in_fronts,
            2: self.random_in_memory,
            3: self.random_in_fronts
        }
        # The options of Differential Mutation pool
        self.dm_pool_type = {
            0: self.differential_mutation_pool_from_population,
            1: self.differential_mutation_pool_from_memory,
            2: self.differential_mutation_pool_from_population_and_memory
        }
        # The options of Differential Evolution strategy
        self.de_mutation_type = {
            0: self.rand_1_bin,
            1: self.rand_2_bin,
            2: self.best_1_bin,
            3: self.current_to_best_1_bin,
            4: self.current_to_rand_1_bin
        }

    ''' Choose the global best attribution type'''
    def get_global_best_attribution(self, type):
        return self.global_best_attribution_type[type]

    """ Calculate the sigma value for the particle set """
    def sigma_evaluation(self, fitness_matrix):
        # Get the squared fitness matrix
        squared_fitnesses = np.square(fitness_matrix)
        # Get the sum of each line in the fitness matrix
        sum_squared_fitnesses = np.sum(squared_fitnesses, axis=1, keepdims=True)
        # Take the indexes to make the combination of differences (simulate a lower triangular matrix per vector to make the differences efficiently)
        row_indexes, col_indexes = self.pre_calculated.np_tril_indices
        # Get the fitness differences
        differences = squared_fitnesses[:, row_indexes] - squared_fitnesses[:, col_indexes]
        # Calculate the sigma values for each particle
        return differences / sum_squared_fitnesses

    ''' Find the nearest particle by the sigma value from memory '''
    def sigma_nearest_by_memory(self, particle_idxs):
        memory_sigma = self.memory.sigma
        num_particles = len(particle_idxs)
        # If there is just one particle in the memory, it is the global best of all indexed particles
        if len(memory_sigma) == 1:
            return np.zeros(num_particles, dtype=int)
        else:
            # Get the nearest neighbor distances and indices
            distances, indices = self.pre_calculated.nearest_neighbors.fit(memory_sigma).kneighbors(self.population.sigma[particle_idxs])
            # The nearest neighbor must be different from itself
            zero_distances_mask = distances[:, 0] == 0
            first_valid_idxs = np.where(zero_distances_mask, 1, 0)
            # Return the nearest indices
            return indices[np.arange(num_particles), first_valid_idxs]
    
    ''' Find the nearest particle by the sigma value from the previous frontier '''
    def sigma_nearest_by_fronts(self, particle_idxs, search_idxs):
        population_sigma = self.population.sigma
        num_particles = len(particle_idxs)
        # If there is just one particle in the memory, it is the global best of all indexed particles
        if len(search_idxs) == 1:
            return np.zeros(num_particles, dtype=int)
        else:
            # Get the nearest neighbor distances and indices
            distances, indices = self.pre_calculated.nearest_neighbors.fit(population_sigma[search_idxs]).kneighbors(population_sigma[particle_idxs])
            # The nearest neighbor must be different from itself
            non_zero_distances_mask = distances[:, 0] == 0
            first_valid_idxs = np.where(non_zero_distances_mask, 1, 0)
            return indices[np.arange(num_particles), first_valid_idxs]

    ''' Global best attribution with sigma in memory '''
    def sigma_method_in_memory(self):
        # Evaluate sigma
        self.memory.sigma = self.sigma_evaluation(self.memory.fitness)
        self.population.sigma[:, :] = self.sigma_evaluation(self.population.fitness)
        # Choose the global best for the population by the nearest neighbors using sigma value
        nearest_idxs = self.sigma_nearest_by_memory(np.arange(self.params.population_size))
        self.population.global_best[:, :] = self.memory.position[nearest_idxs]
    
    ''' Global best attribution with sigma in fronts '''
    def sigma_method_in_fronts(self):
        # Get the fronts and its length
        fronts = self.fronts
        num_fronts = len(fronts)
        # Evaluate sigma
        self.memory.sigma = self.sigma_evaluation(self.memory.fitness)
        self.population.sigma[:, :] = self.sigma_evaluation(self.population.fitness)
        # Choose the global best for the Pareto frontier by the nearest neighbors using sigma value
        pareto_idxs = fronts[0]
        nearest_idxs = self.sigma_nearest_by_memory(pareto_idxs)
        self.population.global_best[pareto_idxs] = self.memory.position[nearest_idxs]
        # Choose the global best for others frontiers by the nearest neighbors from the previous frontier using sigma value
        prev_front = fronts[0]
        for i in range(1, num_fronts):
            current_front = fronts[i]
            nearest_idxs = self.sigma_nearest_by_fronts(current_front, prev_front)
            prev_front = current_front
            self.population.global_best[nearest_idxs] = self.population.position[nearest_idxs]
    
    ''' Global best attribution with choosing randomly in memory '''
    def random_in_memory(self):
        # Get the random indexes for the particles from memory
        random_indices = np.random.randint(0, len(self.memory.position), size=self.params.population_size)
        # Choose the global best
        self.population.global_best[:, :] = self.memory.position[random_indices]
    
    ''' Global best attribution with choosing randomly in memory '''
    def random_in_fronts(self):
        # Get the masks for the respective rank positions
        rank_zero_mask = self.population.rank == 0
        rank_non_zero_mask = ~rank_zero_mask
        # Set the global best from memory
        num_rank_zero = np.sum(rank_zero_mask)
        self.population.global_best[rank_zero_mask] = self.memory.position[np.random.randint(0, len(self.memory.position), size=num_rank_zero)]
        # Get the particles indices which have rank greater than zero
        prev_front_idxs = self.population.rank[rank_non_zero_mask] - 1
        if(len(prev_front_idxs)):
            # Get the fronts and the front sizes
            fronts = self.fronts
            # Generate the random indices for ranks greater than zero
            random_indices = np.random.randint(0, [len(fronts[r]) for r in prev_front_idxs])
            rank_non_zero_idxs = np.array([fronts[idx][random_indices[i]] for i, idx in enumerate(prev_front_idxs)])
            # Set the global best from previous front
            self.population.global_best[rank_non_zero_mask] = self.population.position[rank_non_zero_idxs]

    ''' Choose the differential mutation pool type'''
    def get_differential_mutation_pool(self, type):
        return self.dm_pool_type[type]
    
    ''' Return a pool tensor of particles from population '''
    def differential_mutation_pool_from_population(self, pb_positions):
        # Get the positions
        positions = self.population.position
        # A array with each position as a matrix with just one row vector
        position_tensor = np.expand_dims(positions, axis=1)
        pb_position_tensor = np.expand_dims(pb_positions, axis=1)
        # Get the pool masks
        pool_masks = np.any((pb_position_tensor != positions) | (position_tensor != positions), axis=2) & (~self.np_dominate(position_tensor, positions, axis=2))
        # Get the indices to generate the pool with subarrays
        split_indices = np.cumsum(np.sum(pool_masks, axis=1)[:-1])
        # Get the indices of the positions for each row of pool masks
        _, col_indices = np.where(pool_masks)
        # Return the pool
        return np.split(positions[col_indices], split_indices)

    ''' Return a pool of particles from memory '''
    def differential_mutation_pool_from_memory(self, pb_positions):
        # Get the positions
        positions = self.population.position
        # A array with each position as a matrix with just one row vector
        position_tensor = np.expand_dims(positions, axis=1)
        pb_position_tensor = np.expand_dims(pb_positions, axis=1)
        # Get the memory positions
        mem_positions = self.memory.position
        # Get the pool masks
        pool_masks = np.any((pb_position_tensor != mem_positions) | (position_tensor != mem_positions), axis=2) & (~self.np_dominate(position_tensor, mem_positions, axis=2))
        # Get the indices to generate the pool with subarrays
        split_indices = np.cumsum(np.sum(pool_masks, axis=1)[:-1])
        # Get the indices of the positions for each row of pool masks
        _, col_indices = np.where(pool_masks)
        # Return the pool
        return np.split(mem_positions[col_indices], split_indices)

    ''' Return a pool of particles from population and memory '''
    def differential_mutation_pool_from_population_and_memory(self, pb_positions):
        # Get the positions
        positions = self.population.position
        # Get the memory positions
        mem_positions = self.memory.position
        # A array with each position as a matrix with just one row vector
        position_tensor = np.expand_dims(positions, axis=1)
        pb_positions_tensor = np.expand_dims(pb_positions, axis=1)
        # Concatenate the population position and the memory position
        pop_and_mem_positions = np.concatenate((positions, mem_positions), axis=0)
        # Get the pool masks
        pool_masks = np.any((pb_positions_tensor != pop_and_mem_positions) | (position_tensor != pop_and_mem_positions), axis=2) & (~self.np_dominate(position_tensor, pop_and_mem_positions, axis=2))
        # Get the indices to generate the pool with subarrays
        split_indices = np.cumsum(np.sum(pool_masks, axis=1)[:-1])
        # Get the indices of the positions for each row of pool masks
        _, col_indices = np.where(pool_masks)
        # Return the pool
        return np.split(pop_and_mem_positions[col_indices], split_indices)
    
    ''' Choose the Differential Mutation strategy '''
    def get_differential_mutation_strategy(self, type):
        return self.de_mutation_type[type]

    ''' Applies the DE/rand/1/bin '''
    def rand_1_bin(self, idx, personal_best_position, xr_pool):
        if len(xr_pool) >= 3:    
            # Get three particle positions from pool randomly
            xr_idx = np.random.choice(np.arange(xr_pool.shape[0]), 3, replace=False)
            xr = xr_pool[xr_idx]
            # Apply the DE\rand\1\Bin strategy
            xst = xr[0] + self.weights[5][idx] * (xr[1] - xr[2])
            np.clip(xst, self.params.position_min_value, self.params.position_max_value, out=xst)
            # Apply the mutation operation
            mutation_index = np.random.randint(0, self.params.position_dim)
            mutation_chance = np.random.uniform(0.0, 1.0, self.params.position_dim)
            mutation_mask = (mutation_chance < self.weights[4][idx]) | idx == mutation_index
            xst[mutation_mask] = personal_best_position[mutation_mask]
            return xst, True
        else:
            return None, False
    
    ''' Applies the DE/rand/2/bin '''
    def rand_2_bin(self, idx, personal_best_position, xr_pool):
        if len(xr_pool) >= 5:
            # Get the five particle positions from pool randomly
            xr_idx = np.random.choice(np.arange(xr_pool.shape[0]), 5, replace=False)
            xr = xr_pool[xr_idx]
            # Apply the DE/rand/2/bin
            xst = xr[0] + self.weights[5][idx] * ((xr[1] - xr[2]) + (xr[3] - xr[4]))
            np.clip(xst, self.params.position_min_value, self.params.position_max_value, out=xst)
            # Apply the mutation operation
            mutation_index = np.random.randint(0, self.params.position_dim)
            mutation_chance = np.random.uniform(0.0, 1.0, self.params.position_dim)
            mutation_mask = (mutation_chance < self.weights[4][idx]) | idx == mutation_index
            xst[mutation_mask] = personal_best_position[mutation_mask]
            return xst, True
        else:
            return None, False
    
    ''' Applies the DE/best/1/bin '''
    def best_1_bin(self, idx, personal_best_position, xr_pool):
        if len(xr_pool) >= 2:
            # Get the two particle positions from pool randomly
            xr_idx = np.random.choice(np.arange(xr_pool.shape[0]), 2, replace=False)
            xr = xr_pool[xr_idx]
            # Apply the DE/best/1/bin
            xst = self.population.global_best[idx] + self.weights[5][idx] * (xr[0] - xr[1])
            np.clip(xst, self.params.position_min_value, self.params.position_max_value, out=xst)
            # Apply the mutation operation
            mutation_index = np.random.randint(0, self.params.position_dim)
            mutation_chance = np.random.uniform(0.0, 1.0, self.params.position_dim)
            mutation_mask = (mutation_chance < self.weights[4][idx]) | idx == mutation_index
            xst[mutation_mask] = personal_best_position[mutation_mask]
            return xst, True
        else:
            return None, False
    
    ''' Applies the DE/current-to-best/1/bin '''
    def current_to_best_1_bin(self, idx, personal_best_position, xr_pool):
        if len(xr_pool) >= 2:
            # Get the two particle positions from pool randomly
            xr_idx = np.random.choice(np.arange(xr_pool.shape[0]), 2, replace=False)
            xr = xr_pool[xr_idx]
            # Apply the DE/current-to-best/1/bin
            xst = personal_best_position + self.weights[5][idx] * ((xr[0] - xr[1]) + (self.population.global_best[idx] - personal_best_position))
            np.clip(xst, self.params.position_min_value, self.params.position_max_value, out=xst)
            # Apply the mutation operation
            mutation_index = np.random.randint(0, self.params.position_dim)
            mutation_chance = np.random.uniform(0.0, 1.0, self.params.position_dim)
            mutation_mask = (mutation_chance < self.weights[4][idx]) | idx == mutation_index
            xst[mutation_mask] = personal_best_position[mutation_mask]
            return xst, True
        else:
            return None, False
    
    ''' Applies the DE/current-to-rand/1/bin '''
    def current_to_rand_1_bin(self, idx, personal_best_position, xr_pool):
        if len(xr_pool) >= 3:    
            # Get the two particle positions from pool randomly
            xr_idx = np.random.choice(np.arange(xr_pool.shape[0]), 3, replace=False)
            xr = xr_pool[xr_idx]
            # Apply the DE/current-to-rand/1/bin
            xst = personal_best_position + self.weights[5][idx] * ((xr[0] - xr[1]) + (xr[2] - personal_best_position))
            np.clip(xst, self.params.position_min_value, self.params.position_max_value, out=xst)
            # Apply the mutation operation
            mutation_index = np.random.randint(0, self.params.position_dim)
            mutation_chance = np.random.uniform(0.0, 1.0, self.params.position_dim)
            mutation_mask = (mutation_chance < self.weights[4][idx]) | idx == mutation_index
            xst[mutation_mask] = personal_best_position[mutation_mask]
            return xst, True
        else:
            return None, False

''' Algorithm stop '''
class StoppingAlgorithm(Exception):
    ''' Initialize the instance '''
    def __init__(self):
        pass

''' Pre-calculated data '''
class PreCalculated():
    ''' Initialize the instance '''
    def __init__(self,
                 objective_dim,
                 position_dim,
                 population_size):
        self.np_tril_indices = np.tril_indices(objective_dim, k=-1)
        self.nearest_neighbors = NearestNeighbors(n_neighbors=2, algorithm='auto', metric='euclidean')
        self.matrix_for_operations = np.empty((population_size, position_dim))
        self.vector_for_operations = np.empty(population_size)