import numpy as np

''' Return a pool tensor of particles from population '''
def differential_mutation_pool_from_population(self: object):
  # Get the positions
  positions = self.population.position
  # A array with each position as a matrix with just one row vector
  position_tensor = positions[:, np.newaxis]
  # Get the pool masks
  pool_masks = np.any(position_tensor != positions, axis=2)
  # Get the indices to generate the pool with subarrays
  split_indices = np.cumsum(np.sum(pool_masks, axis=1)[:-1])
  # Get the indices of the positions for each row of pool masks
  _, col_indices = np.where(pool_masks)
  # Return the pool
  return np.split(positions[col_indices], split_indices)

''' Return a pool of particles from memory '''
def differential_mutation_pool_from_memory(self: object):
  # Get the positions
  positions = self.population.position
  # A array with each position as a matrix with just one row vector
  position_tensor = positions[:, np.newaxis]
  # Get the memory positions
  mem_positions = self.memory.position
  # Get the pool masks
  pool_masks = np.any(position_tensor != mem_positions, axis=2)
  # Get the indices to generate the pool with subarrays
  split_indices = np.cumsum(np.sum(pool_masks, axis=1)[:-1])
  # Get the indices of the positions for each row of pool masks
  _, col_indices = np.where(pool_masks)
  # Return the pool
  return np.split(mem_positions[col_indices], split_indices)

''' Return a pool of particles from population and memory '''
def differential_mutation_pool_from_population_and_memory(self: object):
  # Get the positions
  positions = self.population.position
  # A array with each position as a matrix with just one row vector
  position_tensor = positions[:, np.newaxis]
  # Concatenate the population position and the memory position
  pop_and_mem_positions = np.concatenate((positions, self.memory.position), axis=0)
  # Get the pool masks
  pool_masks = np.any(position_tensor != pop_and_mem_positions, axis=2)
  # Get the indices to generate the pool with subarrays
  split_indices = np.cumsum(np.sum(pool_masks, axis=1)[:-1])
  # Get the indices of the positions for each row of pool masks
  _, col_indices = np.where(pool_masks)
  # Return the pool
  return np.split(pop_and_mem_positions[col_indices], split_indices)

# The options of Differential Mutation pool
differential_mutation_pool_options = {
    0: differential_mutation_pool_from_population,
    1: differential_mutation_pool_from_memory,
    2: differential_mutation_pool_from_population_and_memory
}

''' Choose the differential mutation pool type'''
def get_differential_mutation_pool(type: {0, 1, 2}):
  return differential_mutation_pool_options[type]