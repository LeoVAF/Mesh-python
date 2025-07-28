from __future__ import annotations

from random import sample
from scipy.stats import truncnorm
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from mesh.core import Mesh

def rand_1(self: Mesh, pool_tuple: tuple[np.ndarray[np.float64, 2], list[np.ndarray[np.uint64, 2]]]) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]:
  r''' Applies the DE/rand/1 strategy. The strategy is defined as follows:
  
  .. math::
    x_{st} = x_{r1} + \alpha \cdot (x_{r2} - x_{r3}),
  
  where:

  - :math:`x_{r1}`, :math:`x_{r2}` and :math:`x_{r3}` are three random particle positions chosen from the pool under Uniform Distribution;
  - :math:`\alpha` is the scaling factor.

  Note:
    In this implementation, the scaling factor :math:`\alpha` is calculated by a Truncated Normal Distribution between 0 and 2 with mean 0 and standard deviation 1, and then multiplied by :attr:`~mesh.parameters.MeshParameters.mutation_rate`.

  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
    pool_tuple (:type:`tuple[np.ndarray[np.float64, 2], list[np.ndarray[np.uint64, 2]]]`): A particle position pool (first item) and a list of indices for the allowed positions for each particle (second item).
    
  Returns:
    :type:`tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]`: The new particle position matrix and the indices of the particles that underwent Differential Mutation.
  '''

  # Get the particle position pool and the index list for the particles
  pool, pool_idxs = pool_tuple
  # Set the valid size of each pool
  valid_size = 3
  # Get the mask for the pools with valid length
  valid_idxs = np.flatnonzero(np.fromiter((len(x) for x in pool_idxs), dtype=int) >= valid_size)
  valid_idx_size = len(valid_idxs)
  if valid_idx_size:
    # Get three random indices for particle positions from pool
    Xr = np.array([pool[sample(pool_idxs[idx].tolist(), k=valid_size)] for idx in valid_idxs], order='F')
    # Get the operation weight
    operation_weight = truncnorm.rvs(0, 2, size=(valid_idx_size, 1))
    # Apply the DE\rand\1 strategy
    Xst = Xr[:, 0, :] + operation_weight * (Xr[:, 1, :] - Xr[:, 2, :])
    # Clip the positions to the boundaries
    np.clip(Xst, self.params.position_lower_bounds, self.params.position_upper_bounds, out=Xst)
    return Xst, valid_idxs
  else:
    return np.array([]), np.array([])

def rand_2(self: Mesh, pool_tuple: tuple[np.ndarray[np.float64, 2], list[np.ndarray[np.uint64, 2]]]) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]:
  r''' Applies the DE/rand/2 strategy. The strategy is defined as follows:

  .. math::
    x_{st} = x_{r1} + \alpha \cdot (x_{r2} - x_{r3}) + \alpha \cdot (x_{r4} - x_{r5}),

  where:
  
  - :math:`x_{r1}`, :math:`x_{r2}`, :math:`x_{r3}`, :math:`x_{r4}` and :math:`x_{r5}` are five random particle positions chosen from the pool under Uniform Distribution;
  - :math:`\alpha` is the scaling factor.

  Note:
    In this implementation, the scaling factor :math:`\alpha` is calculated by a Truncated Normal Distribution between 0 and 2 with mean 0 and standard deviation 1, and then multiplied by :attr:`~mesh.parameters.MeshParameters.mutation_rate`.

  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
    pool_tuple (:type:`tuple[np.ndarray[np.float64, 2], list[np.ndarray[np.uint64, 2]]]`): A particle position pool (first item) and a list of indices for the allowed positions for each particle (second item).

  Returns:
    :type:`tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]`: The new particle position matrix and the indices of the particles that underwent Differential Mutation.
  '''

  # Get the particle position pool and the index list for the particles
  pool, pool_idxs = pool_tuple
  # Set the valid size of each pool
  valid_size = 5
  # Get the mask for the pools with valid length
  valid_idxs = np.flatnonzero(np.fromiter((len(x) for x in pool_idxs), dtype=int) >= valid_size)
  valid_idx_size = len(valid_idxs)
  if valid_idx_size:
    # Get five random indices for particle positions from pool
    Xr = np.array([pool[sample(pool_idxs[idx].tolist(), k=valid_size)] for idx in valid_idxs], order='F')
    # Get the operation weight
    operation_weight = truncnorm.rvs(0, 2, size=(valid_idx_size, 1))
    # Apply the DE\rand\2 strategy
    Xst = Xr[:, 0, :] + operation_weight * (Xr[:, 1, :] - Xr[:, 2, :]  + Xr[:, 3, :] - Xr[:, 4, :])
    # Clip the positions to the boundaries
    np.clip(Xst, self.params.position_lower_bounds, self.params.position_upper_bounds, out=Xst)
    return Xst, valid_idxs
  else:
    return np.array([]), np.array([])

def best_1(self: Mesh, pool_tuple: tuple[np.ndarray[np.float64, 2], list[np.ndarray[np.uint64, 2]]]) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]:
  r''' Applies the DE/best/1. The strategy is defined as follows:
  
  .. math::
    x_{st} = x_{gb} + \alpha \cdot (x_{r1} - x_{r2}),
  
  where:

  - :math:`x_{gb}` is the global best position;
  - :math:`x_{r1}` and :math:`x_{r2}` are two random particle positions chosen from the pool under Uniform Distribution;
  - :math:`\alpha` is the scaling factor.

  Note:
    In this implementation, the scaling factor :math:`\alpha` is calculated by a Truncated Normal Distribution between 0 and 2 with mean 0 and standard deviation 1, and then multiplied by :attr:`~mesh.parameters.MeshParameters.mutation_rate`.

  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
    pool_tuple (:type:`tuple[np.ndarray[np.float64, 2], list[np.ndarray[np.uint64, 2]]]`): A particle position pool (first item) and a list of indices for the allowed positions for each particle (second item).

  Returns:
    :type:`tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]`: The new particle position matrix and the indices of the particles that underwent Differential Mutation.
  '''

  # Update the global best
  self.global_guide_method()
  # Get the particle position pool and the index list for the particles
  pool, pool_idxs = pool_tuple
  # Set the valid size of each pool
  valid_size = 2
  # Get the mask for the pools with valid length
  valid_idxs = np.flatnonzero(np.fromiter((len(x) for x in pool_idxs), dtype=int) >= valid_size)
  valid_idx_size = len(valid_idxs)
  if valid_idx_size:
    # Get two random indices for particle positions from pool
    Xr = np.array([pool[sample(pool_idxs[idx].tolist(), k=valid_size)] for idx in valid_idxs], order='F')
    # Get the operation weight
    operation_weight = truncnorm.rvs(0, 2, size=(valid_idx_size, 1))
    # Apply the DE\rand\1 strategy
    Xst = self.population.global_guide[valid_idxs] + operation_weight * (Xr[:, 0, :] - Xr[:, 1, :])
    # Clip the positions to the boundaries
    np.clip(Xst, self.params.position_lower_bounds, self.params.position_upper_bounds, out=Xst)
    return Xst, valid_idxs
  else:
    return np.array([]), np.array([])

def current_to_best_1(self: Mesh, pool_tuple: tuple[np.ndarray[np.float64, 2], list[np.ndarray[np.uint64, 2]]]) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]:
  r''' Applies the DE/current-to-best/1. The strategy is defined as follows:
  
  .. math::
    x_{st} = x + \alpha \cdot (x_{gb} - x) + \alpha \cdot (x_{r1} - x_{r2}),
  
  where:

  - :math:`x` is the current particle position;
  - :math:`x_{r1}` and :math:`x_{r2}` are two random particle positions chosen from the pool under Uniform Distribution;
  - :math:`x_{gb}` is the global best position;
  - :math:`\alpha` is the scaling factor.

  Note:
    In this implementation, the scaling factor :math:`\alpha` is calculated by a Truncated Normal Distribution between 0 and 2 with mean 0 and standard deviation 1, and then multiplied by :attr:`~mesh.parameters.MeshParameters.mutation_rate`.

  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
    pool_tuple (:type:`tuple[np.ndarray[np.float64, 2], list[np.ndarray[np.uint64, 2]]]`): A particle position pool (first item) and a list of indices for the allowed positions for each particle (second item).

  Returns:
    :type:`tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]`: The new particle position matrix and the indices of the particles that underwent Differential Mutation.
  '''

  # Update the global best
  self.global_guide_method()
  # Get the particle position pool and the index list for the particles
  pool, pool_idxs = pool_tuple
  # Set the valid size of each pool
  valid_size = 2
  # Get the mask for the pools with valid length
  valid_idxs = np.flatnonzero(np.fromiter((len(x) for x in pool_idxs), dtype=int) >= valid_size)
  valid_idx_size = len(valid_idxs)
  if valid_idx_size:
    # Get two random indices for particle positions from pool
    Xr = np.array([pool[sample(pool_idxs[idx].tolist(), k=valid_size)] for idx in valid_idxs], order='F')
    # Get the operation weight
    operation_weight = truncnorm.rvs(0, 2, size=(valid_idx_size, 1))
    # Apply the DE\rand\1 strategy
    X = self.population.position[valid_idxs]
    Xst = X + operation_weight * (self.population.global_guide[valid_idxs] - X + Xr[:, 0, :] - Xr[:, 1, :])
    # Clip the positions to the boundaries
    np.clip(Xst, self.params.position_lower_bounds, self.params.position_upper_bounds, out=Xst)
    return Xst, valid_idxs
  else:
    return np.array([]), np.array([])

def current_to_rand_1(self: Mesh, pool_tuple: tuple[np.ndarray[np.float64, 2], list[np.ndarray[np.uint64, 2]]]) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]:
  r''' Applies the DE/current-to-rand/1. The strategy is defined as follows:

  .. math::

    x_{st} = x + \alpha \cdot (x_{r1} - x) + \alpha \cdot (x_{r2} - x_{r3}),

  where:

  - :math:`x` is the current particle position;
  - :math:`x_{r1}`, :math:`x_{r2}` and :math:`x_{r3}` are four random particle positions chosen from the pool under Uniform Distribution;
  - :math:`\alpha` is the scaling factor.

  Note:
    In this implementation, the scaling factor :math:`\alpha` is calculated by a Truncated Normal Distribution between 0 and 2 with mean 0 and standard deviation 1, and then multiplied by :attr:`~mesh.parameters.MeshParameters.mutation_rate`.

  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
    pool_tuple (:type:`tuple[np.ndarray[np.float64, 2], list[np.ndarray[np.uint64, 2]]]`): A particle position pool (first item) and a list of indices for the allowed positions for each particle (second item).

  Returns:
    :type:`tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]`: The new particle position matrix and the indices of the particles that underwent Differential Mutation.
  '''

  # Get the particle position pool and the index list for the particles
  pool, pool_idxs = pool_tuple
  # Set the valid size of each pool
  valid_size = 3
  # Get the mask for the pools with valid length
  valid_idxs = np.flatnonzero(np.fromiter((len(x) for x in pool_idxs), dtype=int) >= valid_size)
  valid_idx_size = len(valid_idxs)
  if valid_idx_size:
    # Get three random indices for particle positions from pool
    Xr = np.array([pool[sample(pool_idxs[idx].tolist(), k=valid_size)] for idx in valid_idxs], order='F')
    # Get the operation weight
    operation_weight = truncnorm.rvs(0, 2, size=(valid_idx_size, 1))
    # Apply the DE\rand\2 strategy
    X = self.population.position[valid_idxs]
    Xst = X + operation_weight * (Xr[:, 0, :] - X + Xr[:, 1, :] - Xr[:, 2, :])
    # Clip the positions to the boundaries
    np.clip(Xst, self.params.position_lower_bounds, self.params.position_upper_bounds, out=Xst)
    return Xst, valid_idxs
  else:
    return np.array([]), np.array([])

# The options of Differential Mutation operation
differential_mutation_options = {
  0: rand_1,
  1: rand_2,
  2: best_1,
  3: current_to_best_1,
  4: current_to_rand_1
}
''' The options of Differential Mutation operation. They are:

  - :type:`0`: Applies the DE/rand/1 strategy.
  - :type:`1`: Applies the DE/rand/2 strategy.
  - :type:`2`: Applies the DE/best/1 strategy.
  - :type:`3`: Applies the DE/current-to-best/1 strategy.
  - :type:`4`: Applies the DE/current-to-rand/1 strategy.
'''

def get_differential_mutation(option: {0, 1, 2, 3, 4}) -> Callable[[Mesh, list[np.ndarray[np.float64, 2]]], tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]]:
  ''' Sets the Differential Mutation strategy from Differential Evolution according to :attr:`~mesh.operations.differential_mutation.differential_mutation_options`.
  
  Args:
    option (:type:`{0, 1, 2, 3, 4}`): Defines the Differential Mutation strategy.

  Returns:
    :type:`Callable[[Mesh, list[np.ndarray[np.float64, 2]]], tuple[np.ndarray[np.float64, 2], np.ndarray[np.integer]]]`: The Differential Mutation strategy function.
  '''

  return differential_mutation_options[option]