import numpy as np

from sklearn.neighbors import NearestNeighbors
from scipy.stats import truncnorm

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
        row_indexes, col_indexes = self.pre_alocated.np_tril_indices
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
            distances, indices = self.pre_alocated.nearest_neighbors.fit(memory_sigma).kneighbors(self.population.sigma[particle_idxs])
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
            distances, indices = self.pre_alocated.nearest_neighbors.fit(population_sigma[search_idxs]).kneighbors(population_sigma[particle_idxs])
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
    def differential_mutation_pool_from_population(self):
        # Get the positions
        positions = self.population.position
        # A array with each position as a matrix with just one row vector
        position_tensor = positions[:, np.newaxis]
        # Get the pool masks
        pool_masks = np.any(position_tensor != positions, axis=2) & (~self.np_dominate(position_tensor, positions, axis=2))
        # Get the indices to generate the pool with subarrays
        split_indices = np.cumsum(np.sum(pool_masks, axis=1)[:-1])
        # Get the indices of the positions for each row of pool masks
        _, col_indices = np.where(pool_masks)
        # Return the pool
        return np.split(positions[col_indices], split_indices)

    ''' Return a pool of particles from memory '''
    def differential_mutation_pool_from_memory(self):
        # Get the positions
        positions = self.population.position
        # A array with each position as a matrix with just one row vector
        position_tensor = positions[:, np.newaxis]
        # Get the memory positions
        mem_positions = self.memory.position
        # Get the pool masks
        pool_masks = np.any(position_tensor != mem_positions, axis=2) & (~self.np_dominate(position_tensor, mem_positions, axis=2))
        # Get the indices to generate the pool with subarrays
        split_indices = np.cumsum(np.sum(pool_masks, axis=1)[:-1])
        # Get the indices of the positions for each row of pool masks
        _, col_indices = np.where(pool_masks)
        # Return the pool
        return np.split(mem_positions[col_indices], split_indices)

    ''' Return a pool of particles from population and memory '''
    def differential_mutation_pool_from_population_and_memory(self):
        # Get the positions
        positions = self.population.position
        # A array with each position as a matrix with just one row vector
        position_tensor = positions[:, np.newaxis]
        # Concatenate the population position and the memory position
        pop_and_mem_positions = np.concatenate((positions, self.memory.position), axis=0)
        # Get the pool masks
        pool_masks = np.any(position_tensor != pop_and_mem_positions, axis=2) & (~self.np_dominate(position_tensor, pop_and_mem_positions, axis=2))
        # Get the indices to generate the pool with subarrays
        split_indices = np.cumsum(np.sum(pool_masks, axis=1)[:-1])
        # Get the indices of the positions for each row of pool masks
        _, col_indices = np.where(pool_masks)
        # Return the pool
        return np.split(pop_and_mem_positions[col_indices], split_indices)
    
    ''' Choose the Differential Mutation strategy '''
    def get_differential_mutation_strategy(self, type):
        return self.de_mutation_type[type]

    ''' Make the mutation mask to apply the binomial mutation '''
    def mutation_operator_bin(self, idx_size):
        # Get the mutation weight
        mutation_weight = truncnorm.rvs(0, 0.5, size=(idx_size, 1)) * self.params.mutation_rate
        # Make the mutation index for each particle
        mutation_index = np.random.randint(0, self.params.position_dim, size=idx_size)
        # Calculate the mutation chance to apply the binomial mutation
        mutation_chance = np.random.uniform(0.0, 1.0, size=(idx_size, self.params.position_dim))
        # Get the mutation mask
        mutation_mask = mutation_chance < mutation_weight
        mutation_mask[np.arange(idx_size), mutation_index] = True
        return mutation_mask

    ''' Applies the DE/rand/1/bin '''
    def rand_1_bin(self, xr_pool_tensor):
        # Set the valid size of each pool
        valid_size = 3
        # Get the mask for the pools with valid length
        valid_mask = [len(x) >= valid_size for x in xr_pool_tensor]
        valid_idxs = np.flatnonzero(valid_mask)
        idx_size = len(valid_idxs)
        if idx_size:
            # Get three random indices for particle positions from pool
            xr = np.array([xr_pool_tensor[idx][np.random.permutation(len(xr_pool_tensor[idx]))[:valid_size]] for idx in valid_idxs], order='F')
            # Get the operation weight
            operation_weight = truncnorm.rvs(0, 2, size=(idx_size, 1)) * self.params.mutation_rate
            # Apply the DE\rand\1\bin strategy
            xst = xr[:, 1] - xr[:, 2]
            xst *= operation_weight
            xst += xr[:, 0]
            # Clip the positions to the boundaries
            np.clip(xst, self.params.position_min_value, self.params.position_max_value, out=xst)
            # Apply the mutation operator
            mutation_mask = self.mutation_operator_bin(idx_size)
            xst[mutation_mask] = self.population.position[valid_mask][mutation_mask]
            return xst, valid_idxs
        else:
            return np.array([]), np.array([])
    
    ''' Applies the DE/rand/2/bin '''
    def rand_2_bin(self, xr_pool_tensor):
        # Set the valid size of each pool
        valid_size = 5
        # Get the mask for the pools with valid length
        valid_mask = [len(x) >= valid_size for x in xr_pool_tensor]
        valid_idxs = np.flatnonzero(valid_mask)
        idx_size = len(valid_idxs)
        if idx_size:
            # Get three random indices for particle positions from pool
            xr = np.array([xr_pool_tensor[idx][np.random.permutation(len(xr_pool_tensor[idx]))[:valid_size]] for idx in valid_idxs], order='F')
            # Get the operation weight
            operation_weight = truncnorm.rvs(0, 2, size=(idx_size, 1)) * self.params.mutation_rate
            # Apply the DE\rand\2\bin strategy
            xst = xr[:, 3] - xr[:, 4]
            xst += xr[:, 1]
            xst -= xr[:, 2]
            xst *= operation_weight
            xst += xr[:, 0]
            # Clip the positions to the boundaries
            np.clip(xst, self.params.position_min_value, self.params.position_max_value, out=xst)
            # Apply the mutation operator
            mutation_mask = self.mutation_operator_bin(idx_size)
            xst[mutation_mask] = self.population.position[valid_mask][mutation_mask]
            return xst, valid_idxs
        else:
            return np.array([]), np.array([])
    
    ''' Applies the DE/best/1/bin '''
    def best_1_bin(self, xr_pool_tensor):
        # Update the global best
        self.global_best_attribution()
        # Set the valid size of each pool
        valid_size = 2
        # Get the mask for the pools with valid length
        valid_mask = [len(x) >= valid_size for x in xr_pool_tensor]
        valid_idxs = np.flatnonzero(valid_mask)
        idx_size = len(valid_idxs)
        if idx_size:
            # Get three random indices for particle positions from pool
            xr = np.array([xr_pool_tensor[idx][np.random.permutation(len(xr_pool_tensor[idx]))[:valid_size]] for idx in valid_idxs], order='F')
            # Get the operation weight
            operation_weight = truncnorm.rvs(0, 2, size=(idx_size, 1)) * self.params.mutation_rate
            # Apply the DE\rand\1\bin strategy
            xst = xr[:, 0] - xr[:, 1]
            xst *= operation_weight
            xst += self.population.global_best[valid_idxs]
            # Clip the positions to the boundaries
            np.clip(xst, self.params.position_min_value, self.params.position_max_value, out=xst)
            # Apply the mutation operator
            mutation_mask = self.mutation_operator_bin(idx_size)
            xst[mutation_mask] = self.population.position[valid_mask][mutation_mask]
            return xst, valid_idxs
        else:
            return np.array([]), np.array([])
    
    ''' Applies the DE/current-to-best/1/bin '''
    def current_to_best_1_bin(self, xr_pool_tensor):
        # Update the global best
        self.global_best_attribution()
        # Set the valid size of each pool
        valid_size = 2
        # Get the mask for the pools with valid length
        valid_mask = [len(x) >= valid_size for x in xr_pool_tensor]
        valid_idxs = np.flatnonzero(valid_mask)
        idx_size = len(valid_idxs)
        if idx_size:
            # Get three random indices for particle positions from pool
            xr = np.array([xr_pool_tensor[idx][np.random.permutation(len(xr_pool_tensor[idx]))[:valid_size]] for idx in valid_idxs], order='F')
            # Get the operation weight
            operation_weight = truncnorm.rvs(0, 2, size=(idx_size, 1)) * self.params.mutation_rate
            # Apply the DE\rand\1\bin strategy
            positions = self.population.position
            xst = xr[:, 0] - xr[:, 1]
            xst += self.population.global_best[valid_idxs]
            xst -= positions[valid_idxs]
            xst *= operation_weight
            xst += positions[valid_idxs]
            # Clip the positions to the boundaries
            np.clip(xst, self.params.position_min_value, self.params.position_max_value, out=xst)
            # Apply the mutation operator
            mutation_mask = self.mutation_operator_bin(idx_size)
            xst[mutation_mask] = self.population.position[valid_mask][mutation_mask]
            return xst, valid_idxs
        else:
            return np.array([]), np.array([])
    
    ''' Applies the DE/current-to-rand/1/bin '''
    def current_to_rand_1_bin(self, xr_pool_tensor):
        # Set the valid size of each pool
        valid_size = 4
        # Get the mask for the pools with valid length
        valid_mask = [len(x) >= valid_size for x in xr_pool_tensor]
        valid_idxs = np.flatnonzero(valid_mask)
        idx_size = len(valid_idxs)
        if idx_size:
            # Get three random indices for particle positions from pool
            xr = np.array([xr_pool_tensor[idx][np.random.permutation(len(xr_pool_tensor[idx]))[:valid_size]] for idx in valid_idxs], order='F')
            # Get the operation weight
            operation_weight = truncnorm.rvs(0, 2, size=(idx_size, 1)) * self.params.mutation_rate
            # Apply the DE\rand\2\bin strategy
            xst = xr[:, 2] - xr[:, 3]
            xst += xr[:, 0]
            xst -= xr[:, 1]
            xst *= operation_weight
            xst += self.population.position[valid_idxs]
            # Clip the positions to the boundaries
            np.clip(xst, self.params.position_min_value, self.params.position_max_value, out=xst)
            # Apply the mutation operator
            mutation_mask = self.mutation_operator_bin(idx_size)
            xst[mutation_mask] = self.population.position[valid_mask][mutation_mask]
            return xst, valid_idxs
        else:
            return np.array([]), np.array([])
            

''' Algorithm stop '''
class StoppingAlgorithm(Exception):
    ''' Initialize the instance '''
    def __init__(self):
        pass

''' Pre-alocated data '''
class PreAlocated():
    ''' Initialize the instance '''
    def __init__(self,
                 objective_dim,
                 position_dim,
                 population_size):
        # Used to calculate the sigma
        self.np_tril_indices = np.tril_indices(objective_dim, k=-1)
        # The object to get the nearest neighbors
        self.nearest_neighbors = NearestNeighbors(n_neighbors=2, algorithm='auto', metric='euclidean')
        # Structures used to calculate repetitive operations
        self.matrix_for_operations = np.empty((population_size, position_dim))
        self.vector_for_operations = np.empty(population_size)
        # Fitness matrix for the population selection
        self.fitness_selection = np.empty((2*population_size, objective_dim))
        # Copies for the population
        self.position_copy = np.empty((population_size, position_dim))
        self.velocity_copy = np.empty((population_size, position_dim))
        self.fitness_copy = np.empty((population_size, objective_dim))