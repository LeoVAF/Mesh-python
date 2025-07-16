from __future__ import annotations
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from mesh.core import Mesh

def pool_from_memory(self: Mesh) -> list[np.ndarray[np.float64, 2]]:
  ''' Returns a pool list of particle position from memory according to Differential Mutation strategies. The pool list of particle position is a list of matrices with the respective pool for each particle.
  
  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.

  Returns:
    :type:`list[np.ndarray[np.float64, 2]]`: The pool list of particles from memory.
  '''

  # Get the positions
  positions = self.population.position
  pool_positions = self.memory.position
  # Compare with the current population positions
  pool_mask = np.any(positions[:, np.newaxis, :] != pool_positions, axis=2)
  # Indices to generate the pool with subarrays
  split_indices = np.cumsum(np.sum(pool_mask, axis=1)[:-1])
  # Indices of the positions for each row of final pool masks
  _, col_indices = np.where(pool_mask)
  # Generate the pool list of positions
  return np.split(pool_positions[col_indices], split_indices)

def pool_from_population(self: Mesh) -> list[np.ndarray[np.float64, 2]]:
  ''' Makes a pool list of particle position from population according to Differential Mutation strategies. The pool list of particle position is a list of matrices with the respective pool for each particle.
  
  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.

  Returns:
    :type:`list[np.ndarray[np.float64, 2]]`: The pool list of particles from population.
  '''

  # Get the positions
  positions = self.population.position
  pool_positions = np.unique(self.population.position, axis=0)
  # Compare with the current population positions
  pool_mask = np.any(positions[:, np.newaxis, :] != pool_positions, axis=2)
  # Indices to generate the pool with subarrays
  split_indices = np.cumsum(np.sum(pool_mask, axis=1)[:-1])
  # Indices of the positions for each row of final pool masks
  _, col_indices = np.where(pool_mask)
  # Generate the pool list of positions
  return np.split(pool_positions[col_indices], split_indices)

def pool_from_population_and_memory(self: Mesh) -> list[np.ndarray[np.float64, 2]]:
  ''' Makes a pool list of particle position from population and memory according to Differential Mutation strategies. The pool list of particle position is a list of matrices with the respective pool for each particle.
  
  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.

  Returns:
    :type:`list[np.ndarray[np.float64, 2]]`: The pool list of particles from population and memory.
  '''

  # Get the positions
  positions = self.population.position
  pool_positions = np.unique(np.concatenate((positions, self.memory.position), axis=0), axis=0)
  # Compare with the current population positions
  pool_mask = np.any(positions[:, np.newaxis, :] != pool_positions, axis=2)
  # Indices to generate the pool with subarrays
  split_indices = np.cumsum(np.sum(pool_mask, axis=1)[:-1])
  # Indices of the positions for each row of final pool masks
  _, col_indices = np.where(pool_mask)
  # Generate the pool list of positions
  return np.split(pool_positions[col_indices], split_indices)

# The options of Differential Mutation pool
differential_mutation_pool_options = {
    0: pool_from_memory,
    1: pool_from_population,
    2: pool_from_population_and_memory
}
''' The options of Differential Mutation pool. They are:

  - :type:`0`: Pool from memory.
  - :type:`1`: Pool from population.
  - :type:`2`: Pool from population and memory.
'''

def get_differential_mutation_pool(option: {0, 1, 2}) -> Callable[[Mesh], list[np.ndarray[np.float64, 2]]]:
  ''' Sets the Differential Mutation pool according to :attr:`~mesh.operations.differential_mutation_pool.differential_mutation_pool_options`.
  
  Args:
    option (:type:`{0, 1, 2}`): Defines the Differential Mutation pool.
  
  Returns:
    :type:`Callable[[Mesh], list[np.ndarray[np.float64, 2]]]`: The respective function to make the Differential Mutation pool.
  '''

  return differential_mutation_pool_options[option]