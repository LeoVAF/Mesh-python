from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from MESH import Mesh

def pool_from_population(self: Mesh) -> np.ndarray[np.float64, 3]:
  ''' Return a pool tensor of particles from population according to differential mutation operations. The pool tensor is a list of matrices with the respective pool for each particle.
  
  Args:
    self (:class:`~mesh.MESH.Mesh`): An instance of Mesh.

  Returns:
    :type:`np.ndarray[np.float64, 3]`: The pool tensor of particles from population.
  '''

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

def pool_from_memory(self: Mesh) -> np.ndarray[np.float64, 3]:
  ''' Return a pool tensor of particles from memory according to differential mutation operations. The pool tensor is a list of matrices with the respective pool for each particle.
  
  Args:
    self (:class:`~mesh.MESH.Mesh`): An instance of Mesh.

  Returns:
    :type:`np.ndarray[np.float64, 3]`: The pool tensor of particles from memory.
  '''

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

def pool_from_population_and_memory(self: Mesh) -> np.ndarray[np.float64, 3]:
  ''' Return a pool tensor of particles from population and memory according to differential mutation operations. The pool tensor is a list of matrices with the respective pool for each particle.
  
  Args:
    self (:class:`~mesh.MESH.Mesh`): An instance of Mesh.

  Returns:
    :type:`np.ndarray[np.float64, 3]`: The pool tensor of particles from population and memory.
  '''

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
    0: pool_from_population,
    1: pool_from_memory,
    2: pool_from_population_and_memory
}
''' The options of Differential Mutation pool. They are:

  - :data:`0`: Pool from population.
  - :data:`1`: Pool from memory.
  - :data:`2`: Pool from population and memory.
'''

def get_differential_mutation_pool(type: {0, 1, 2}) -> function:
  ''' Choose the Differential Mutation pool type.
  
  Args:
    type (:type:`{0, 1, 2}`): The type of Differential Mutation pool.
  
  Returns:
    :type:`function`: The respective function to get the differential mutation pool.
  '''
  return differential_mutation_pool_options[type]