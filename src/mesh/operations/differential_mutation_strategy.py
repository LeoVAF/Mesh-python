import numpy as np
from scipy.stats import truncnorm

''' Make the mutation mask to apply the binomial mutation '''
def binomial_mutation_operator(self, idx_size):
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
    mutation_mask = binomial_mutation_operator(self, idx_size)
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
    mutation_mask = binomial_mutation_operator(self, idx_size)
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
    mutation_mask = binomial_mutation_operator(self, idx_size)
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
    mutation_mask = binomial_mutation_operator(self, idx_size)
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
    mutation_mask = binomial_mutation_operator(self, idx_size)
    xst[mutation_mask] = self.population.position[valid_mask][mutation_mask]
    return xst, valid_idxs
  else:
    return np.array([]), np.array([])


# The options of Differential Evolution strategy
differential_mutation_strategy_options = {
  0: rand_1_bin,
  1: rand_2_bin,
  2: best_1_bin,
  3: current_to_best_1_bin,
  4: current_to_rand_1_bin
}

''' Choose the Differential Evolution strategy'''
def get_differential_mutation_strategy(type: {0, 1, 2, 3, 4}):
  return differential_mutation_strategy_options[type]