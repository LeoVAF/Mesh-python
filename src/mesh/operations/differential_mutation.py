from __future__ import annotations

from random import sample
from scipy.stats import truncnorm
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from mesh.core import Mesh

def rand_1_bin(self: Mesh, Xr_pool_list: list[np.ndarray[np.float64, 2]]) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]:
  r''' Applies the DE/rand/1 strategy. The strategy is defined as follows:
  
  .. math::
    x_{st} = x_{r1} + \alpha \cdot (x_{r2} - x_{r3}),
  
  where:

  - :math:`x_{r1}`, :math:`x_{r2}` and :math:`x_{r3}` are three random particle positions chosen from the pool under uniform distribution;
  - :math:`\alpha` is the scaling factor.

  Note:
    In this implementation, the scaling factor :math:`\alpha` is calculated by a truncated normal distribution between 0 and 2 with mean 0 and standard deviation 1, and then multiplied by :attr:`~mesh.parameters.MeshParameters.mutation_rate`.

  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
    Xr_pool_list (:type:`list[np.ndarray[np.float64, 2]]`): A pool list of particle position.
    
  Returns:
    :type:`tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]`: The new particle position matrix and the indices of the particles that underwent Differential Mutation.
  '''

  # Set the valid size of each pool
  valid_size = 3
  # Get the mask for the pools with valid length
  valid_idxs = np.flatnonzero([len(x) >= valid_size for x in Xr_pool_list])
  valid_idx_size = len(valid_idxs)
  if valid_idx_size:
    # Get three random indices for particle positions from pool
    Xr = np.array([Xr_pool_list[idx][sample(range(len(Xr_pool_list[idx])), k=valid_size)] for idx in valid_idxs], order='F')
    # Get the operation weight
    operation_weight = truncnorm.rvs(0, 2, size=(valid_idx_size, 1)) * self.params.mutation_rate
    # Apply the DE\rand\1\bin strategy
    Xst = Xr[:, 1] - Xr[:, 2]
    Xst *= operation_weight
    Xst += Xr[:, 0]
    # Clip the positions to the boundaries
    np.clip(Xst, self.params.lower_bound_array, self.params.upper_bound_array, out=Xst)
    return Xst, valid_idxs
  else:
    return np.array([]), np.array([])

def rand_2_bin(self: Mesh, Xr_pool_list: list[np.ndarray[np.float64, 2]]) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]:
  r''' Applies the DE/rand/2 strategy. The strategy is defined as follows:

  .. math::
    x_{st} = x_{r1} + \alpha \cdot (x_{r2} - x_{r3}) + \alpha \cdot (x_{r4} - x_{r5}),

  where:
  
  - :math:`x_{r1}`, :math:`x_{r2}`, :math:`x_{r3}`, :math:`x_{r4}` and :math:`x_{r5}` are five random particle positions chosen from the pool under uniform distribution;
  - :math:`\alpha` is the scaling factor.

  Note:
    In this implementation, the scaling factor :math:`\alpha` is calculated by a truncated normal distribution between 0 and 2 with mean 0 and standard deviation 1, and then multiplied by :attr:`~mesh.parameters.MeshParameters.mutation_rate`.

  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
    Xr_pool_list (:type:`list[np.ndarray[np.float64, 2]]`): A pool list of particle position.

  Returns:
    :type:`tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]`: The new particle position matrix and the indices of the particles that underwent Differential Mutation.
  '''

  # Set the valid size of each pool
  valid_size = 5
  # Get the mask for the pools with valid length
  valid_idxs = np.flatnonzero([len(x) >= valid_size for x in Xr_pool_list])
  valid_idx_size = len(valid_idxs)
  if valid_idx_size:
    # Get five random indices for particle positions from pool
    Xr = np.array([Xr_pool_list[idx][sample(range(len(Xr_pool_list[idx])), k=valid_size)] for idx in valid_idxs], order='F')
    # Get the operation weight
    operation_weight = truncnorm.rvs(0, 2, size=(valid_idx_size, 1)) * self.params.mutation_rate
    # Apply the DE\rand\2\bin strategy
    Xst = Xr[:, 3] - Xr[:, 4]
    Xst += Xr[:, 1]
    Xst -= Xr[:, 2]
    Xst *= operation_weight
    Xst += Xr[:, 0]
    # Clip the positions to the boundaries
    np.clip(Xst, self.params.lower_bound_array, self.params.upper_bound_array, out=Xst)
    return Xst, valid_idxs
  else:
    return np.array([]), np.array([])

def best_1_bin(self: Mesh, Xr_pool_list: list[np.ndarray[np.float64, 2]]) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]:
  r''' Applies the DE/best/1. The strategy is defined as follows:
  
  .. math::
    x_{st} = x_{gb} + \alpha \cdot (x_{r1} - x_{r2}),
  
  where:

  - :math:`x_{gb}` is the global best position;
  - :math:`x_{r1}` and :math:`x_{r2}` are two random particle positions chosen from the pool under uniform distribution;
  - :math:`\alpha` is the scaling factor.

  Note:
    In this implementation, the scaling factor :math:`\alpha` is calculated by a truncated normal distribution between 0 and 2 with mean 0 and standard deviation 1, and then multiplied by :attr:`~mesh.parameters.MeshParameters.mutation_rate`.

  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
    Xr_pool_list (:type:`list[np.ndarray[np.float64, 2]]`): A pool list of particle position.

  Returns:
    :type:`tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]`: The new particle position matrix and the indices of the particles that underwent Differential Mutation.
  '''

  # Update the global best
  self.global_best_attribution()
  # Set the valid size of each pool
  valid_size = 2
  # Get the mask for the pools with valid length
  valid_idxs = np.flatnonzero([len(x) >= valid_size for x in Xr_pool_list])
  valid_idx_size = len(valid_idxs)
  if valid_idx_size:
    # Get two random indices for particle positions from pool
    Xr = np.array([Xr_pool_list[idx][sample(range(len(Xr_pool_list[idx])), k=valid_size)] for idx in valid_idxs], order='F')
    # Get the operation weight
    operation_weight = truncnorm.rvs(0, 2, size=(valid_idx_size, 1)) * self.params.mutation_rate
    # Apply the DE\rand\1\bin strategy
    Xst = Xr[:, 0] - Xr[:, 1]
    Xst *= operation_weight
    Xst += self.population.global_best[valid_idxs]
    # Clip the positions to the boundaries
    np.clip(Xst, self.params.lower_bound_array, self.params.upper_bound_array, out=Xst)
    return Xst, valid_idxs
  else:
    return np.array([]), np.array([])

def current_to_best_1_bin(self: Mesh, Xr_pool_list: list[np.ndarray[np.float64, 2]]) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]:
  r''' Applies the DE/current-to-best/1. The strategy is defined as follows:
  
  .. math::
    x_{st} = x + \alpha \cdot (x_{gb} - x) + \alpha \cdot (x_{r1} - x_{r2}),
  
  where:

  - :math:`x` is the current particle position;
  - :math:`x_{r1}` and :math:`x_{r2}` are two random particle positions chosen from the pool under uniform distribution;
  - :math:`x_{gb}` is the global best position;
  - :math:`\alpha` is the scaling factor.

  Note:
    In this implementation, the scaling factor :math:`\alpha` is calculated by a truncated normal distribution between 0 and 2 with mean 0 and standard deviation 1, and then multiplied by :attr:`~mesh.parameters.MeshParameters.mutation_rate`.

  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
    Xr_pool_list (:type:`list[np.ndarray[np.float64, 2]]`): A pool list of particle position.

  Returns:
    :type:`tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]`: The new particle position matrix and the indices of the particles that underwent Differential Mutation.
  '''

  # Update the global best
  self.global_best_attribution()
  # Set the valid size of each pool
  valid_size = 2
  # Get the mask for the pools with valid length
  valid_idxs = np.flatnonzero([len(x) >= valid_size for x in Xr_pool_list])
  valid_idx_size = len(valid_idxs)
  if valid_idx_size:
    # Get two random indices for particle positions from pool
    Xr = np.array([Xr_pool_list[idx][sample(range(len(Xr_pool_list[idx])), k=valid_size)] for idx in valid_idxs], order='F')
    # Get the operation weight
    operation_weight = truncnorm.rvs(0, 2, size=(valid_idx_size, 1)) * self.params.mutation_rate
    # Apply the DE\rand\1\bin strategy
    X = self.population.position[valid_idxs]
    Xst = Xr[:, 0] - Xr[:, 1]
    Xst += self.population.global_best[valid_idxs]
    Xst -= X
    Xst *= operation_weight
    Xst += X
    # Clip the positions to the boundaries
    np.clip(Xst, self.params.lower_bound_array, self.params.upper_bound_array, out=Xst)
    return Xst, valid_idxs
  else:
    return np.array([]), np.array([])

def current_to_rand_1_bin(self: Mesh, Xr_pool_list: list[np.ndarray[np.float64, 2]]) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]:
  r''' Applies the DE/current-to-rand/1. The strategy is defined as follows:

  .. math::

    x_{st} = x + \alpha \cdot (x_{r1} - x) + \alpha \cdot (x_{r2} - x_{r3}),

  where:

  - :math:`x` is the current particle position;
  - :math:`x_{r1}`, :math:`x_{r2}` and :math:`x_{r3}` are four random particle positions chosen from the pool under uniform distribution;
  - :math:`\alpha` is the scaling factor.

  Note:
    In this implementation, the scaling factor :math:`\alpha` is calculated by a truncated normal distribution between 0 and 2 with mean 0 and standard deviation 1, and then multiplied by :attr:`~mesh.parameters.MeshParameters.mutation_rate`.

  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
    Xr_pool_list (:type:`list[np.ndarray[np.float64, 2]]`): A pool list of particle position.

  Returns:
    :type:`tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]`: The new particle position matrix and the indices of the particles that underwent Differential Mutation.
  '''

  # Set the valid size of each pool
  valid_size = 3
  # Get the mask for the pools with valid length
  valid_idxs = np.flatnonzero([len(x) >= valid_size for x in Xr_pool_list])
  valid_idx_size = len(valid_idxs)
  if valid_idx_size:
    # Get three random indices for particle positions from pool
    Xr = np.array([Xr_pool_list[idx][sample(range(len(Xr_pool_list[idx])), k=valid_size)] for idx in valid_idxs], order='F')
    # Get the operation weight
    operation_weight = truncnorm.rvs(0, 2, size=(valid_idx_size, 1)) * self.params.mutation_rate
    # Apply the DE\rand\2\bin strategy
    X = self.population.position[valid_idxs]
    Xst = Xr[:, 1] - Xr[:, 2]
    Xst += Xr[:, 0]
    Xst -= X
    Xst *= operation_weight
    Xst += X
    # Clip the positions to the boundaries
    np.clip(Xst, self.params.lower_bound_array, self.params.upper_bound_array, out=Xst)
    return Xst, valid_idxs
  else:
    return np.array([]), np.array([])

# The options of Differential Mutation operation
differential_mutation_options = {
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

def get_differential_mutation(option: {0, 1, 2, 3, 4}) -> Callable[[Mesh, list[np.ndarray[np.float64, 2]]], tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]]:
  ''' Sets the Differential Mutation strategy from Differential Evolution according to :attr:`~mesh.operations.differential_mutation.differential_mutation_options`.
  
  Args:
    option (:type:`{0, 1, 2, 3, 4}`): Defines the Differential Mutation strategy.

  Returns:
    :type:`Callable[[Mesh, list[np.ndarray[np.float64, 2]]], tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]]`: The Differential Mutation strategy function.
  '''

  return differential_mutation_options[option]