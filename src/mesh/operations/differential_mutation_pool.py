from __future__ import annotations
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from mesh.core import Mesh

def pool_from_memory(self: Mesh) -> tuple[np.ndarray[np.float64, 2], list[np.ndarray[np.integer, 2]]]:
  ''' Makes a tuple containing memory positions where positions will be sampled and a list of indices for the allowed positions for each particles according to Differential Mutation strategies.
  
  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.

  Returns:
    :type:`tuple[np.ndarray[np.float64, 2], list[np.ndarray[np.integer, 2]]]`: Memory positions (first item) and list of indices for the allowed positions for each particle (second item).
  '''

  # Get the positions
  positions = self.population.position
  pool = self.memory.position
  # Compare with the current population positions
  pool_mask = np.any(positions[:, np.newaxis, :] != pool, axis=2)
  # Indices to generate the pool with subarrays
  split_indices = np.cumsum(np.sum(pool_mask, axis=1)[:-1])
  # Indices of the positions for each row of final pool masks
  _, col_indices = np.where(pool_mask)
  # Generate the pool list of positions
  return pool, np.split(col_indices, split_indices)

def pool_from_population(self: Mesh) -> tuple[np.ndarray[np.float64, 2], list[np.ndarray[np.integer, 2]]]:
  ''' Makes a tuple containing population positions where positions will be sampled and a list of indices for the allowed positions for each particles according to Differential Mutation strategies.
  
  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.

  Returns:
    :type:`tuple[np.ndarray[np.float64, 2], list[np.ndarray[np.integer, 2]]]`: Population positions (first item) and list of indices for the allowed positions for each particle (second item).
  '''

  # Get the positions
  positions = self.population.position
  pool = np.unique(self.population.position, axis=0)
  # Compare with the current population positions
  pool_mask = np.any(positions[:, np.newaxis, :] != pool, axis=2)
  # Indices to generate the pool with subarrays
  split_indices = np.cumsum(np.sum(pool_mask, axis=1)[:-1])
  # Indices of the positions for each row of final pool masks
  _, col_indices = np.where(pool_mask)
  # Generate the pool list of positions
  return pool, np.split(col_indices, split_indices)

def pool_from_population_and_memory(self: Mesh) -> tuple[np.ndarray[np.float64, 2], list[np.ndarray[np.integer, 2]]]:
  ''' Makes a tuple containing population and memory positions where positions will be sampled and a list of indices for the allowed positions for each particles according to Differential Mutation strategies.
  
  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.

  Returns:
    :type:`tuple[np.ndarray[np.float64, 2], list[np.ndarray[np.integer, 2]]]`: Population and memory positions (first item) and list of indices for the allowed positions for each particle (second item).
  '''

  # Get the positions
  positions = self.population.position
  pool = np.unique(np.concatenate((positions, self.memory.position), axis=0), axis=0)
  # Compare with the current population positions
  pool_mask = np.any(positions[:, np.newaxis, :] != pool, axis=2)
  # Indices to generate the pool with subarrays
  split_indices = np.cumsum(np.sum(pool_mask, axis=1)[:-1])
  # Indices of the positions for each row of final pool masks
  _, col_indices = np.where(pool_mask)
  # Generate the pool list of positions
  return pool, np.split(col_indices, split_indices)

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

def get_differential_mutation_pool(option: {0, 1, 2}) -> Callable[[Mesh], tuple[np.ndarray[np.float64, 2], list[np.ndarray[np.integer, 2]]]]:
  ''' Sets the Differential Mutation pool according to :attr:`~mesh.operations.differential_mutation_pool.differential_mutation_pool_options`.
  
  Args:
    option (:type:`{0, 1, 2}`): Defines the Differential Mutation pool.
  
  Returns:
    :type:`Callable[[Mesh], tuple[np.ndarray[np.float64, 2], list[np.ndarray[np.integer, 2]]]]`: The respective function to make the Differential Mutation pool.
  '''

  return differential_mutation_pool_options[option]