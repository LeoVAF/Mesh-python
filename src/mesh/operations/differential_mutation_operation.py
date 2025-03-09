from __future__ import annotations
from typing import TYPE_CHECKING
from scipy.stats import truncnorm

import numpy as np

if TYPE_CHECKING:
    from MESH import Mesh
    from parameters import MeshParameters

def binomial_mutation_mask(params: MeshParameters, idx_size: int) -> np.ndarray[bool, 2]:
  ''' Makes a mask numpy matrix to apply the binomial mutation. Each value represents if the mutation will be applied or not.
  
  Args:
    params (:class:`~mesh.parameters.MeshParameters`): The parameters :attr:`~mesh.parameters.MeshParameters.position_dim` and :attr:`~mesh.parameters.MeshParameters.mutation_rate` are used to make the mask.
    idx_size (:type:`int`): The size of the indexes to apply the mutation.
    
  Returns:
    :type:`np.ndarray[bool, 2]`: The mutation mask.
  '''

  # Get the mutation weight
  mutation_weight = truncnorm.rvs(0, 0.5, size=(idx_size, 1)) * params.mutation_rate
  # Make the mutation index for each particle
  mutation_index = np.random.randint(0, params.position_dim, size=idx_size)
  # Calculate the mutation chance to apply the binomial mutation
  mutation_chance = np.random.uniform(0.0, 1.0, size=(idx_size, params.position_dim))
  # Get the mutation mask
  mutation_mask = mutation_chance < mutation_weight
  mutation_mask[np.arange(idx_size), mutation_index] = True
  return mutation_mask

''' Applies the DE/rand/1/bin '''
def rand_1_bin(self: Mesh, xr_pool_list: list[np.ndarray[np.float64, 2]]) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]:
  r''' Applies the DE/rand/1/bin strategy. The strategy is defined as follows:
  
  .. math::
    x_{st} = x_{r1} + \alpha \cdot (x_{r2} - x_{r3}),
  
  where:

    - :math:`x_{r1}`, :math:`x_{r2}` and :math:`x_{r3}` are three random particle positions chosen from the pool under uniform distribution;
    - :math:`\alpha` is the scaling factor.

  Note:
    In this implementation, the scaling factor :math:`\alpha` is calculated by a truncated normal distribution with mean 1 and standard deviation 0 between 0 and 2, and then multiplied by :attr:`~mesh.parameters.MeshParameters.mutation_rate`.

  Args:
    self (:class:`~mesh.MESH.Mesh`): An instance of Mesh.
    xr_pool_list (:type:`list[np.ndarray[np.float64, 2]]`): A pool list of particle position.
    
  Returns:
    :type:`tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]`: The new particle position matrix and the indices of the particles that underwent differential mutation.
  '''

  # Set the valid size of each pool
  valid_size = 3
  # Get the mask for the pools with valid length
  valid_mask = [len(x) >= valid_size for x in xr_pool_list]
  valid_idxs = np.flatnonzero(valid_mask)
  idx_size = len(valid_idxs)
  if idx_size:
    # Get three random indices for particle positions from pool
    xr = np.array([xr_pool_list[idx][np.random.permutation(len(xr_pool_list[idx]))[:valid_size]] for idx in valid_idxs], order='F')
    # Get the operation weight
    operation_weight = truncnorm.rvs(0, 2, size=(idx_size, 1)) * self.params.mutation_rate
    # Apply the DE\rand\1\bin strategy
    xst = xr[:, 1] - xr[:, 2]
    xst *= operation_weight
    xst += xr[:, 0]
    # Clip the positions to the boundaries
    np.clip(xst, self.params.position_min_value, self.params.position_max_value, out=xst)
    # Apply the mutation operator
    mutation_mask = binomial_mutation_mask(self.params, idx_size)
    xst[mutation_mask] = self.population.position[valid_mask][mutation_mask]
    return xst, valid_idxs
  else:
    return np.array([]), np.array([])

def rand_2_bin(self: Mesh, xr_pool_list: list[np.ndarray[np.float64, 2]]) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]:
  r''' Applies the DE/rand/2/bin strategy. The strategy is defined as follows:

  .. math::
    x_{st} = x_{r1} + \alpha \cdot (x_{r2} - x_{r3}) + \alpha \cdot (x_{r4} - x_{r5}),

  where:
  
    - :math:`x_{r1}`, :math:`x_{r2}`, :math:`x_{r3}`, :math:`x_{r4}` and :math:`x_{r5}` are five random particle positions chosen from the pool under uniform distribution;
    - :math:`\alpha` is the scaling factor.

  Note:
    In this implementation, the scaling factor :math:`\alpha` is calculated by a truncated normal distribution with mean 1 and standard deviation 0 between 0 and 2, and then multiplied by :attr:`~mesh.parameters.MeshParameters.mutation_rate`.

  Args:
    self (:class:`~mesh.MESH.Mesh`): An instance of Mesh.
    xr_pool_list (:type:`list[np.ndarray[np.float64, 2]]`): A pool list of particle position.

  Returns:
    :type:`tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]`: The new particle position matrix and the indices of the particles that underwent differential mutation.
  '''

  # Set the valid size of each pool
  valid_size = 5
  # Get the mask for the pools with valid length
  valid_mask = [len(x) >= valid_size for x in xr_pool_list]
  valid_idxs = np.flatnonzero(valid_mask)
  idx_size = len(valid_idxs)
  if idx_size:
    # Get three random indices for particle positions from pool
    xr = np.array([xr_pool_list[idx][np.random.permutation(len(xr_pool_list[idx]))[:valid_size]] for idx in valid_idxs], order='F')
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
    mutation_mask = binomial_mutation_mask(self.params, idx_size)
    xst[mutation_mask] = self.population.position[valid_mask][mutation_mask]
    return xst, valid_idxs
  else:
    return np.array([]), np.array([])

def best_1_bin(self: Mesh, xr_pool_list: list[np.ndarray[np.float64, 2]]) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]:
  r''' Applies the DE/best/1/bin. The strategy is defined as follows:
  
  .. math::
    x_{st} = x_{gb} + \alpha \cdot (x_{r1} - x_{r2}),
  
  where:

    - :math:`x_{gb}` is the global best position;
    - :math:`x_{r1}` and :math:`x_{r2}` are two random particle positions chosen from the pool under uniform distribution;
    - :math:`\alpha` is the scaling factor.

  Note:
    In this implementation, the scaling factor :math:`\alpha` is calculated by a truncated normal distribution with mean 1 and standard deviation 0 between 0 and 2, and then multiplied by :attr:`~mesh.parameters.MeshParameters.mutation_rate`.

  Args:
    self (:class:`~mesh.MESH.Mesh`): An instance of Mesh.
    xr_pool_list (:type:`list[np.ndarray[np.float64, 2]]`): A pool list of particle position.

  Returns:
    :type:`tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]`: The new particle position matrix and the indices of the particles that underwent differential mutation.
  '''

  # Update the global best
  self.global_best_attribution()
  # Set the valid size of each pool
  valid_size = 2
  # Get the mask for the pools with valid length
  valid_mask = [len(x) >= valid_size for x in xr_pool_list]
  valid_idxs = np.flatnonzero(valid_mask)
  idx_size = len(valid_idxs)
  if idx_size:
    # Get three random indices for particle positions from pool
    xr = np.array([xr_pool_list[idx][np.random.permutation(len(xr_pool_list[idx]))[:valid_size]] for idx in valid_idxs], order='F')
    # Get the operation weight
    operation_weight = truncnorm.rvs(0, 2, size=(idx_size, 1)) * self.params.mutation_rate
    # Apply the DE\rand\1\bin strategy
    xst = xr[:, 0] - xr[:, 1]
    xst *= operation_weight
    xst += self.population.global_best[valid_idxs]
    # Clip the positions to the boundaries
    np.clip(xst, self.params.position_min_value, self.params.position_max_value, out=xst)
    # Apply the mutation operator
    mutation_mask = binomial_mutation_mask(self.params, idx_size)
    xst[mutation_mask] = self.population.position[valid_mask][mutation_mask]
    return xst, valid_idxs
  else:
    return np.array([]), np.array([])

def current_to_best_1_bin(self, xr_pool_list):
  r''' Applies the DE/current-to-best/1/bin. The strategy is defined as follows:
  
  .. math::
    x_{st} = x + \alpha \cdot (x_{gb} - x) + \alpha \cdot (x_{r1} - x_{r2}),
  
  where:

    - :math:`x` is the current particle position;
    - :math:`x_{r1}` and :math:`x_{r2}` are two random particle positions chosen from the pool under uniform distribution;
    - :math:`x_{gb}` is the global best position;
    - :math:`\alpha` is the scaling factor.

  Note:
    In this implementation, the scaling factor :math:`\alpha` is calculated by a truncated normal distribution with mean 1 and standard deviation 0 between 0 and 2, and then multiplied by :attr:`~mesh.parameters.MeshParameters.mutation_rate`.

  Args:
    self (:class:`~mesh.MESH.Mesh`): An instance of Mesh.
    xr_pool_list (:type:`list[np.ndarray[np.float64, 2]]`): A pool list of particle position.

  Returns:
    :type:`tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]`: The new particle position matrix and the indices of the particles that underwent differential mutation.
  '''

  # Update the global best
  self.global_best_attribution()
  # Set the valid size of each pool
  valid_size = 2
  # Get the mask for the pools with valid length
  valid_mask = [len(x) >= valid_size for x in xr_pool_list]
  valid_idxs = np.flatnonzero(valid_mask)
  idx_size = len(valid_idxs)
  if idx_size:
    # Get three random indices for particle positions from pool
    xr = np.array([xr_pool_list[idx][np.random.permutation(len(xr_pool_list[idx]))[:valid_size]] for idx in valid_idxs], order='F')
    # Get the operation weight
    operation_weight = truncnorm.rvs(0, 2, size=(idx_size, 1)) * self.params.mutation_rate
    # Apply the DE\rand\1\bin strategy
    x = self.population.position[valid_idxs]
    xst = xr[:, 0] - xr[:, 1]
    xst += self.population.global_best[valid_idxs]
    xst -= x
    xst *= operation_weight
    xst += x
    # Clip the positions to the boundaries
    np.clip(xst, self.params.position_min_value, self.params.position_max_value, out=xst)
    # Apply the mutation operator
    mutation_mask = binomial_mutation_mask(self.params, idx_size)
    xst[mutation_mask] = self.population.position[valid_mask][mutation_mask]
    return xst, valid_idxs
  else:
    return np.array([]), np.array([])

''' Applies the DE/current-to-rand/1/bin '''
def current_to_rand_1_bin(self, xr_pool_list):
  r''' Applies the DE/current-to-rand/1/bin. The strategy is defined as follows:

  .. math::

    x_{st} = x + \alpha \cdot (x_{r1} - x) + \alpha \cdot (x_{r2} - x_{r3}),

  where:

    - :math:`x` is the current particle position;
    - :math:`x_{r1}`, :math:`x_{r2}` and :math:`x_{r3}` are four random particle positions chosen from the pool under uniform distribution;
    - :math:`\alpha` is the scaling factor.

  Note:
    In this implementation, the scaling factor :math:`\alpha` is calculated by a truncated normal distribution with mean 1 and standard deviation 0 between 0 and 2, and then multiplied by :attr:`~mesh.parameters.MeshParameters.mutation_rate`.

  Args:
    self (:class:`~mesh.MESH.Mesh`): An instance of Mesh.
    xr_pool_list (:type:`list[np.ndarray[np.float64, 2]]`): A pool list of particle position.

  Returns:
    :type:`tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]`: The new particle position matrix and the indices of the particles that underwent differential mutation.
  '''

  # Set the valid size of each pool
  valid_size = 3
  # Get the mask for the pools with valid length
  valid_mask = [len(x) >= valid_size for x in xr_pool_list]
  valid_idxs = np.flatnonzero(valid_mask)
  idx_size = len(valid_idxs)
  if idx_size:
    # Get three random indices for particle positions from pool
    xr = np.array([xr_pool_list[idx][np.random.permutation(len(xr_pool_list[idx]))[:valid_size]] for idx in valid_idxs], order='F')
    # Get the operation weight
    operation_weight = truncnorm.rvs(0, 2, size=(idx_size, 1)) * self.params.mutation_rate
    # Apply the DE\rand\2\bin strategy
    x = self.population.position[valid_idxs]
    xst = xr[:, 1] - xr[:, 2]
    xst += xr[:, 0]
    xst -= x
    xst *= operation_weight
    xst += x
    # Clip the positions to the boundaries
    np.clip(xst, self.params.position_min_value, self.params.position_max_value, out=xst)
    # Apply the mutation operator
    mutation_mask = binomial_mutation_mask(self.params, idx_size)
    xst[mutation_mask] = self.population.position[valid_mask][mutation_mask]
    return xst, valid_idxs
  else:
    return np.array([]), np.array([])

differential_mutation_operation_options = {
  0: rand_1_bin,
  1: rand_2_bin,
  2: best_1_bin,
  3: current_to_best_1_bin,
  4: current_to_rand_1_bin
}
''' The options of Differential Mutation operation. They are:

  - :type:`0`: Applies the DE/rand/1/bin strategy.
  - :type:`1`: Applies the DE/rand/2/bin strategy.
  - :type:`2`: Applies the DE/best/1/bin strategy.
  - :type:`3`: Applies the DE/current-to-best/1/bin strategy.
  - :type:`4`: Applies the DE/current-to-rand/1/bin strategy.
'''

def get_differential_mutation_operation(type: {0, 1, 2, 3, 4}) -> function:
  ''' Chooses the Differential Mutation operation. 
  
  Args:
    type (:type:`{0, 1, 2, 3, 4}`): The type of Differential Mutation operation.

  Returns:
    :type:`function`: The Differential Mutation operation function.
  '''

  return differential_mutation_operation_options[type]