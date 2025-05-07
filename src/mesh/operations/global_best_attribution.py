from __future__ import annotations
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from mesh.core import Mesh

def sigma_evaluation(self: Mesh, fitness_matrix: np.ndarray[np.number, 2]) -> np.ndarray[np.number, 2]:
  r''' Calculates the sigma value for each particle in the population. The sigma value is the fitness difference of all the dimensions. The sigma value is a :math:`C^{n_{obj}}_2`-dimensional vector calculated as follows:

  .. math::
    \sigma = \frac{1}{\sum^{n_{obj}}_{i=1} f^2_i} \left[f^2_1 - f^2_2,\ f^2_1 - f^2_3,\ \ldots,\ f^2_2 - f^2_3,\ f^2_2 - f^2_4,\ \ldots,\ f^2_{n_{obj}-1} - f^2_{n_{obj}} \right]^T,
  
  where :math:`f_i` is the value of the i-th objective, :math:`\forall i \in \{1, 2, \ldots, n_{obj} \}`.

  Note:
    :math:`C^{n_{obj}}_2` is the combination of :math:`n_{obj}` elements taken 2 by 2.

  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
    fitness_matrix (:type:`np.ndarray[np.number, 2]`): The fitness matrix of the population.

  Returns:
    :type:`np.ndarray[np.number, 2]`: The sigma value for each particle in the population.
  '''

  # Get the squared fitness matrix
  squared_fitnesses = np.square(fitness_matrix)
  # Get the sum of each line in the fitness matrix
  sum_squared_fitnesses = np.sum(squared_fitnesses, axis=1, keepdims=True)
  # Take the indices to make the combination of differences (simulate a lower triangular matrix per vector to make the differences efficiently)
  row_indices, col_indices = self.pre_allocated.np_tril_indices
  # Get the fitness differences
  differences = squared_fitnesses[:, row_indices] - squared_fitnesses[:, col_indices]
  # Calculate the sigma values for each particle
  return differences / sum_squared_fitnesses

def nearest_sigma_in_memory(self: Mesh, particle_idxs: np.ndarray[np.integer]) -> np.ndarray[np.integer, 2]:
  ''' Finds the nearest particle index in memory by sigma value, For each population particle index. The nearest particle will be different from itself (some particles in population can be in memory).
  
  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
    particle_idxs (:type:`np.ndarray[np.integer]`): The indices of the population particles to find the nearest particle.

  Returns:
    :type:`np.ndarray[np.integer, 2]`: The indices of the nearest particles in the memory. Each row has the index of the nearest particle for the respective particle in the input. If the nearest particle is itself, the index of the second particle is returned.
  '''

  memory_sigma = self.memory.sigma
  num_particles = len(particle_idxs)
  # If there is just one particle in the memory, it is the global best of all indexed particles
  if len(memory_sigma) == 1:
    return np.zeros(num_particles, dtype=np.uint64)
  else:
    # Get the nearest neighbor distances and indices
    distances, indices = self.pre_allocated.nearest_neighbors.fit(memory_sigma).kneighbors(self.population.sigma[particle_idxs])
    # The nearest neighbor must be different from itself
    zero_distances_mask = distances[:, 0] == 0
    first_valid_idxs = np.where(zero_distances_mask, 1, 0)
    # Return the nearest indices
    return indices[np.arange(num_particles), first_valid_idxs]

def nearest_sigma_in_fronts(self: Mesh, particle_idxs: np.ndarray[np.integer], search_idxs: np.ndarray[np.integer]) -> np.ndarray[np.integer, 2]:
  ''' Finds the nearest particle index in the search indices by sigma value, for each population particle index. The nearest particle will be different from itself.

  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
    particle_idxs (:type:`np.ndarray[np.integer]`): The indices of the population particles to find the nearest particle.
    search_idxs (:type:`np.ndarray[np.integer]`): The indices of the particles to search for the nearest neighbors.

  Returns:
    :type:`np.ndarray[np.integer, 2]`: The indices of the nearest particles in the search indices. Each row has the index of the nearest particle for the respective particle in the input. If the nearest particle is itself, the index of the second particle is returned.
  '''

  population_sigma = self.population.sigma
  num_particles = len(particle_idxs)
  # If there is just one particle in the memory, it is the global best of all indexed particles
  if len(search_idxs) == 1:
    return np.zeros(num_particles, dtype=np.uint64)
  else:
    # Get the nearest neighbor distances and indices
    distances, indices = self.pre_allocated.nearest_neighbors.fit(population_sigma[search_idxs]).kneighbors(population_sigma[particle_idxs])
    # The nearest neighbor must be different from itself
    non_zero_distances_mask = distances[:, 0] == 0
    first_valid_idxs = np.where(non_zero_distances_mask, 1, 0)
    return search_idxs[indices[np.arange(num_particles), first_valid_idxs]]

def sigma_method_in_memory(self: Mesh) -> None:
  ''' Global best attribution with sigma method in memory. The global best for each particle in the population will be the nearest particle different from itself in memory using the sigma value.
  
  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
  '''

  # Evaluate sigma
  self.memory.sigma = sigma_evaluation(self, self.memory.fitness)
  self.population.sigma[:, :] = sigma_evaluation(self, self.population.fitness)
  # Choose the global best for the population by the nearest neighbors using sigma value
  nearest_idxs = nearest_sigma_in_memory(self, np.arange(self.params.population_size))
  self.population.global_best[:, :] = self.memory.position[nearest_idxs]

def sigma_method_in_fronts(self: Mesh) -> None:
  ''' Global best attribution with sigma method in fronts. The global best for each particle in the population will be the nearest particle different from itself in the previous front using the sigma value. Particles in the Pareto front will choose the global best from memory.
  
  Note:
    The previous front is the front with domination rank immediately lower than the domination rank of the current front. The domination ranks are ordered from the lowest to the highest, starting at the Pareto front with zero.

  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
  '''

  # Get the fronts and its length
  fronts = self.fronts
  num_fronts = len(fronts)
  # Evaluate sigma
  self.memory.sigma = sigma_evaluation(self, self.memory.fitness)
  self.population.sigma[:, :] = sigma_evaluation(self, self.population.fitness)
  # Choose the global best for the Pareto front by the nearest neighbors using sigma value
  pareto_idxs = fronts[0]
  nearest_idxs = nearest_sigma_in_memory(self, pareto_idxs)
  self.population.global_best[pareto_idxs] = self.memory.position[nearest_idxs]
  # Choose the global best for others fronts by the nearest neighbors from the front using sigma value
  search_front = fronts[0]
  for i in range(1, num_fronts):
    current_front = fronts[i]
    nearest_idxs = nearest_sigma_in_fronts(self, current_front, search_front)
    search_front = current_front
    self.population.global_best[current_front] = self.population.position[nearest_idxs]

def random_in_memory(self: Mesh) -> None:
  ''' Global best attribution with random choice in memory under an uniform distribution. The global best for each particle in the population will be a random particle in memory.
  
  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
  '''
  # Get the random indices for the particles from memory
  random_indices = np.random.randint(0, len(self.memory.position), size=self.params.population_size)
  # Choose the global best
  self.population.global_best[:, :] = self.memory.position[random_indices]

def random_in_fronts(self: Mesh) -> None:
  ''' Global best attribution with random choice in fronts under an uniform distribution. The global best for each particle in the population will be a random particle in the previous front. Particles in the Pareto front will choose the global best from memory.

  Note:
    The previous front is the front with domination rank immediately lower than the domination rank of the current front. The domination ranks are ordered from the lowest to the highest, starting at the Pareto front with zero.
  
  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
  '''

  # Get the masks for the respective rank positions
  rank_zero_mask = self.population.rank == 0
  rank_non_zero_mask = ~rank_zero_mask
  # Set the global best from memory
  num_rank_zero = np.sum(rank_zero_mask)
  self.population.global_best[rank_zero_mask] = self.memory.position[np.random.randint(0, len(self.memory.position), size=num_rank_zero)]
  # Get the particles indices which have domination rank greater than zero
  search_front_idxs = self.population.rank[rank_non_zero_mask] - 1
  if(len(search_front_idxs)):
    # Get the fronts and the front sizes
    fronts = self.fronts
    # Generate the random indices for domination ranks greater than zero
    random_indices = np.random.randint(0, [len(fronts[r]) for r in search_front_idxs])
    rank_non_zero_idxs = np.array([fronts[idx][random_indices[i]] for i, idx in enumerate(search_front_idxs)])
    # Set the global best from previous front
    self.population.global_best[rank_non_zero_mask] = self.population.position[rank_non_zero_idxs]

global_best_attribution_options = {
  0: sigma_method_in_memory,
  1: sigma_method_in_fronts,
  2: random_in_memory,
  3: random_in_fronts
}
''' The options of global guide attribution operation. They are:

  - :type:`0`: Applies Sigma method in memory to select the global best.
  - :type:`1`: Applies Sigma method in fronts to select the global best. Each particle will select its global best from the next front. Particles in Pareto front will select the global best from memory.
  - :type:`2`: Chooses randomly under uniform distribution a particle from memory.
  - :type:`3`: Chooses randomly under uniform distribution a particle from fronts. Each particle will select its global best from the next front. Particles in Pareto front will select the global best from memory.
'''

def get_global_best_attribution(type: {0, 1, 2, 3}) -> Callable[[Mesh], None]:
  ''' Chooses the global best attribution operation.
  
  Args:
    type (:type:`{0, 1, 2, 3}`): The type of global best attribution operation.

  Returns:
    :type:`Callable[[:class:`~mesh.core.Mesh`], None]`: The respective function to select the global best.
  '''

  return global_best_attribution_options[type]