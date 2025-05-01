from __future__ import annotations
from typing import TYPE_CHECKING, Callable
from scipy.stats import truncnorm

import numpy as np

if TYPE_CHECKING:
    from mesh.core import Mesh
    from mesh.parameters import MeshParameters

def binomial_crossover_mask(params: MeshParameters, idx_size: int) -> np.ndarray[np.bool, 2]:
  ''' Makes a mask numpy matrix to apply the binomial crossover. Each value represents if the crossover will be applied (True values) or not (False values).
  
  Args:
    params (:class:`~mesh.parameters.MeshParameters`): The parameters :attr:`~mesh.parameters.MeshParameters.position_dim` and :attr:`~mesh.parameters.MeshParameters.mutation_rate` are used to make the mask.
    idx_size (:type:`int`): The size of the indexes to apply the crossover.
    
  Returns:
    :type:`np.ndarray[np.bool, 2]`: The crossover mask.
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

def rand_1_bin(self: Mesh, Xr_pool_list: list[np.ndarray[np.float64, 2]]) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]:
  r''' Applies the DE/rand/1/bin strategy. The strategy is defined as follows:
  
  .. math::
    x_{st} = x_{r1} + \alpha \cdot (x_{r2} - x_{r3}),
  
  where:

  - :math:`x_{r1}`, :math:`x_{r2}` and :math:`x_{r3}` are three random particle positions chosen from the pool under uniform distribution;
  - :math:`\alpha` is the scaling factor.

  Note:
    In this implementation, the scaling factor :math:`\alpha` is calculated by a truncated normal distribution with mean 1 and standard deviation 0 between 0 and 2, and then multiplied by :attr:`~mesh.parameters.MeshParameters.mutation_rate`.

  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
    Xr_pool_list (:type:`list[np.ndarray[np.float64, 2]]`): A pool list of particle position.
    
  Returns:
    :type:`tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]`: The new particle position matrix and the indices of the particles that underwent differential mutation.
  '''

  # Set the valid size of each pool
  valid_size = 3
  # Get the mask for the pools with valid length
  valid_mask = [len(x) >= valid_size for x in Xr_pool_list]
  valid_idxs = np.flatnonzero(valid_mask)
  idx_size = len(valid_idxs)
  if idx_size:
    # Get three random indices for particle positions from pool
    Xr = np.array([Xr_pool_list[idx][np.random.permutation(len(Xr_pool_list[idx]))[:valid_size]] for idx in valid_idxs], order='F')
    # Get the operation weight
    operation_weight = truncnorm.rvs(0, 2, size=(idx_size, 1)) * self.params.mutation_rate
    # Apply the DE\rand\1\bin strategy
    Xst = Xr[:, 1] - Xr[:, 2]
    Xst *= operation_weight
    Xst += Xr[:, 0]
    # Clip the positions to the boundaries
    np.clip(Xst, self.params.position_min_value, self.params.position_max_value, out=Xst)
    # Apply the crossover operator in the personal best position
    crossover_mask = binomial_crossover_mask(self.params, idx_size)
    random_indices = np.random.randint(0, self.params.max_personal_guides, size=idx_size)
    valid_pb_positions = self.population.personal_best_pos[valid_idxs, random_indices, :]
    Xst[crossover_mask] = valid_pb_positions[crossover_mask]
    return Xst, valid_idxs
  else:
    return np.array([]), np.array([])

def rand_2_bin(self: Mesh, Xr_pool_list: list[np.ndarray[np.float64, 2]]) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]:
  r''' Applies the DE/rand/2/bin strategy. The strategy is defined as follows:

  .. math::
    x_{st} = x_{r1} + \alpha \cdot (x_{r2} - x_{r3}) + \alpha \cdot (x_{r4} - x_{r5}),

  where:
  
  - :math:`x_{r1}`, :math:`x_{r2}`, :math:`x_{r3}`, :math:`x_{r4}` and :math:`x_{r5}` are five random particle positions chosen from the pool under uniform distribution;
  - :math:`\alpha` is the scaling factor.

  Note:
    In this implementation, the scaling factor :math:`\alpha` is calculated by a truncated normal distribution with mean 1 and standard deviation 0 between 0 and 2, and then multiplied by :attr:`~mesh.parameters.MeshParameters.mutation_rate`.

  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
    Xr_pool_list (:type:`list[np.ndarray[np.float64, 2]]`): A pool list of particle position.

  Returns:
    :type:`tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]`: The new particle position matrix and the indices of the particles that underwent differential mutation.
  '''

  # Set the valid size of each pool
  valid_size = 5
  # Get the mask for the pools with valid length
  valid_mask = [len(x) >= valid_size for x in Xr_pool_list]
  valid_idxs = np.flatnonzero(valid_mask)
  idx_size = len(valid_idxs)
  if idx_size:
    # Get five random indices for particle positions from pool
    Xr = np.array([Xr_pool_list[idx][np.random.permutation(len(Xr_pool_list[idx]))[:valid_size]] for idx in valid_idxs], order='F')
    # Get the operation weight
    operation_weight = truncnorm.rvs(0, 2, size=(idx_size, 1)) * self.params.mutation_rate
    # Apply the DE\rand\2\bin strategy
    Xst = Xr[:, 3] - Xr[:, 4]
    Xst += Xr[:, 1]
    Xst -= Xr[:, 2]
    Xst *= operation_weight
    Xst += Xr[:, 0]
    # Clip the positions to the boundaries
    np.clip(Xst, self.params.position_min_value, self.params.position_max_value, out=Xst)
    # Apply the crossover operator in the personal best position
    crossover_mask = binomial_crossover_mask(self.params, idx_size)
    random_indices = np.random.randint(0, self.params.max_personal_guides, size=idx_size)
    valid_pb_positions = self.population.personal_best_pos[valid_idxs, random_indices, :]
    Xst[crossover_mask] = valid_pb_positions[crossover_mask]
    return Xst, valid_idxs
  else:
    return np.array([]), np.array([])

def best_1_bin(self: Mesh, Xr_pool_list: list[np.ndarray[np.float64, 2]]) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]:
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
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
    Xr_pool_list (:type:`list[np.ndarray[np.float64, 2]]`): A pool list of particle position.

  Returns:
    :type:`tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]`: The new particle position matrix and the indices of the particles that underwent differential mutation.
  '''

  # Update the global best
  self.global_best_attribution()
  # Set the valid size of each pool
  valid_size = 2
  # Get the mask for the pools with valid length
  valid_mask = [len(x) >= valid_size for x in Xr_pool_list]
  valid_idxs = np.flatnonzero(valid_mask)
  idx_size = len(valid_idxs)
  if idx_size:
    # Get two random indices for particle positions from pool
    Xr = np.array([Xr_pool_list[idx][np.random.permutation(len(Xr_pool_list[idx]))[:valid_size]] for idx in valid_idxs], order='F')
    # Get the operation weight
    operation_weight = truncnorm.rvs(0, 2, size=(idx_size, 1)) * self.params.mutation_rate
    # Apply the DE\rand\1\bin strategy
    Xst = Xr[:, 0] - Xr[:, 1]
    Xst *= operation_weight
    Xst += self.population.global_best[valid_idxs]
    # Clip the positions to the boundaries
    np.clip(Xst, self.params.position_min_value, self.params.position_max_value, out=Xst)
    # Apply the crossover operator in the personal best position
    crossover_mask = binomial_crossover_mask(self.params, idx_size)
    random_indices = np.random.randint(0, self.params.max_personal_guides, size=idx_size)
    valid_pb_positions = self.population.personal_best_pos[valid_idxs, random_indices, :]
    Xst[crossover_mask] = valid_pb_positions[crossover_mask]
    return Xst, valid_idxs
  else:
    return np.array([]), np.array([])

def current_to_best_1_bin(self: Mesh, Xr_pool_list: list[np.ndarray[np.float64, 2]]) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]:
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
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
    Xr_pool_list (:type:`list[np.ndarray[np.float64, 2]]`): A pool list of particle position.

  Returns:
    :type:`tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]`: The new particle position matrix and the indices of the particles that underwent differential mutation.
  '''

  # Update the global best
  self.global_best_attribution()
  # Set the valid size of each pool
  valid_size = 2
  # Get the mask for the pools with valid length
  valid_mask = [len(x) >= valid_size for x in Xr_pool_list]
  valid_idxs = np.flatnonzero(valid_mask)
  idx_size = len(valid_idxs)
  if idx_size:
    # Get two random indices for particle positions from pool
    Xr = np.array([Xr_pool_list[idx][np.random.permutation(len(Xr_pool_list[idx]))[:valid_size]] for idx in valid_idxs], order='F')
    # Get the operation weight
    operation_weight = truncnorm.rvs(0, 2, size=(idx_size, 1)) * self.params.mutation_rate
    # Apply the DE\rand\1\bin strategy
    X = self.population.position[valid_idxs]
    Xst = Xr[:, 0] - Xr[:, 1]
    Xst += self.population.global_best[valid_idxs]
    Xst -= X
    Xst *= operation_weight
    Xst += X
    # Clip the positions to the boundaries
    np.clip(Xst, self.params.position_min_value, self.params.position_max_value, out=Xst)
    # Apply the crossover operator in the personal best position
    crossover_mask = binomial_crossover_mask(self.params, idx_size)
    random_indices = np.random.randint(0, self.params.max_personal_guides, size=idx_size)
    valid_pb_positions = self.population.personal_best_pos[valid_idxs, random_indices, :]
    Xst[crossover_mask] = valid_pb_positions[crossover_mask]
    return Xst, valid_idxs
  else:
    return np.array([]), np.array([])

def current_to_rand_1_bin(self: Mesh, Xr_pool_list: list[np.ndarray[np.float64, 2]]) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]:
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
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
    Xr_pool_list (:type:`list[np.ndarray[np.float64, 2]]`): A pool list of particle position.

  Returns:
    :type:`tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]`: The new particle position matrix and the indices of the particles that underwent differential mutation.
  '''

  # Set the valid size of each pool
  valid_size = 3
  # Get the mask for the pools with valid length
  valid_mask = [len(x) >= valid_size for x in Xr_pool_list]
  valid_idxs = np.flatnonzero(valid_mask)
  idx_size = len(valid_idxs)
  if idx_size:
    # Get three random indices for particle positions from pool
    Xr = np.array([Xr_pool_list[idx][np.random.permutation(len(Xr_pool_list[idx]))[:valid_size]] for idx in valid_idxs], order='F')
    # Get the operation weight
    operation_weight = truncnorm.rvs(0, 2, size=(idx_size, 1)) * self.params.mutation_rate
    # Apply the DE\rand\2\bin strategy
    X = self.population.position[valid_idxs]
    Xst = Xr[:, 1] - Xr[:, 2]
    Xst += Xr[:, 0]
    Xst -= X
    Xst *= operation_weight
    Xst += X
    # Clip the positions to the boundaries
    np.clip(Xst, self.params.position_min_value, self.params.position_max_value, out=Xst)
    # Apply the crossover operator in the personal best position
    crossover_mask = binomial_crossover_mask(self.params, idx_size)
    random_indices = np.random.randint(0, self.params.max_personal_guides, size=idx_size)
    valid_pb_positions = self.population.personal_best_pos[valid_idxs, random_indices, :]
    Xst[crossover_mask] = valid_pb_positions[crossover_mask]
    return Xst, valid_idxs
  else:
    return np.array([]), np.array([])

differential_mutation_strategy_options = {
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

def get_differential_mutation_strategy(type: {0, 1, 2, 3, 4}) -> Callable[[Mesh, list[np.ndarray[np.float64, 2]]], tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]]:
  ''' Chooses the Differential Mutation operation. 
  
  Args:
    type (:type:`{0, 1, 2, 3, 4}`): The type of Differential Mutation operation.

  Returns:
    :type:`Callable[[:class:`~mesh.core.Mesh`, list[np.ndarray[np.float64, 2]]], tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]]`: The Differential Mutation operation function.
  '''

  return differential_mutation_strategy_options[type]