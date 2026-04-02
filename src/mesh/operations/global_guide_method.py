from __future__ import annotations

from numpy.typing import NDArray
from sklearn.neighbors import KDTree
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from mesh.core import Mesh

def sigma_evaluation(self: Mesh, fitness_matrix: NDArray[np.number]) -> NDArray[np.number]:
  r''' Calculates the sigma value for each particle in the population. The sigma value is the fitness difference of all the dimensions. The sigma value is a :math:`C^{n_{obj}}_2`-dimensional vector calculated as follows:

  .. math::
    \sigma = \frac{1}{\sum^{n_{obj}}_{i=1} f^2_i} \left[f^2_1 - f^2_2,\ f^2_1 - f^2_3,\ \ldots,\ f^2_2 - f^2_3,\ f^2_2 - f^2_4,\ \ldots,\ f^2_{n_{obj}-1} - f^2_{n_{obj}} \right]^T,
  
  where :math:`f_i` is the value of the i-th objective, :math:`\forall i \in \{1, 2, \ldots, n_{obj} \}`.

  Note:
    :math:`C^{n_{obj}}_2` is the combination of :math:`n_{obj}` elements taken 2 by 2.

  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
    fitness_matrix (:type:`NDArray[np.number]`): The fitness matrix of the population.

  Returns:
    :type:`NDArray[np.number]`: The sigma matrix with sigma value for each particle in the population.
  '''

  # Get the squared fitness matrix
  squared_fitnesses = np.square(fitness_matrix)
  # Get the sum of each line in the fitness matrix (Treat the case when fitness sum is equal to zero)
  sum_squared_fitnesses = np.sum(squared_fitnesses, axis=1, keepdims=True)
  sum_squared_fitnesses[sum_squared_fitnesses == 0] = 1
  # Take the indices to make the combination of differences (simulate a lower triangular matrix per vector to make the differences efficiently)
  row_indices, col_indices = self.pre_allocated.np_tril_indices
  # Get the fitness differences
  differences = squared_fitnesses[:, row_indices] - squared_fitnesses[:, col_indices]
  # Calculate the sigma values for each particle
  return differences / sum_squared_fitnesses

def nearest_sigma_in_memory(self: Mesh, particle_idxs: NDArray[np.intp]) -> NDArray[np.intp]:
  ''' Finds the index of the nearest particle on the memory by the sigma value for each index particle from population. The nearest particle will be different from itself (some particles in population can be in memory).

  Note:
    Because the nearest particle in Sigma space will be different from itself, the memory must have two or more particles when calling this function.
  
  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
    particle_idxs (:type:`NDArray[np.intp]`): The indices of the population particles to find the nearest particle.

  Returns:
    :type:`NDArray[np.intp]`: The indices of the nearest particles in the memory. Each row has the index of the nearest particle for the respective particle in the input. If the nearest particle is itself, the index of the second particle is returned.
  '''

  # Get the nearest neighbor distances and indices
  distances, indices = KDTree(self.memory.sigma).query(self.population.sigma[particle_idxs], k=2)
  # The nearest neighbor must be different from itself
  zero_distances_mask = distances[:, 0] == 0
  first_valid_idxs = np.where(zero_distances_mask, 1, 0)
  # Return the nearest indices
  return indices[np.arange(len(particle_idxs)), first_valid_idxs]

def nearest_sigma_in_fronts(self: Mesh, particle_idxs: NDArray[np.intp], search_idxs: NDArray[np.intp]) -> NDArray[np.intp]:
  ''' Finds the index of the nearest particle on the search front by the sigma value for each index particle from population. Each row has the index of the nearest particle for the respective particle in the input.

  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
    particle_idxs (:type:`NDArray[np.intp]`): The indices of the population particles to find the nearest particle.
    search_idxs (:type:`NDArray[np.intp]`): The indices of the particles to search for the nearest neighbors.

  Returns:
    :type:`NDArray[np.intp]`: The indices of the nearest particles in the search indices. Each row has the index of the nearest particle for the respective particle in the input. If the nearest particle is itself, the index of the second particle is returned.
  '''

  population_sigma = self.population.sigma
  num_particles = len(particle_idxs)
  # If there is just one particle in the search front, it is the global guide of all indexed particles
  if len(search_idxs) == 1:
    return np.full(num_particles, search_idxs[0])
  else:
    # Get the nearest neighbor distances and indices
    _, indices = KDTree(population_sigma[search_idxs]).query(population_sigma[particle_idxs], k=1)
    return search_idxs[indices[np.arange(num_particles), 0]]

def sigma_method_in_memory(self: Mesh) -> None:
  ''' Global guide attribution by sigma method in memory. The global guide for each particle in the population will be the nearest particle different from itself in memory, by sigma value.
  
  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
  '''

  if len(self.memory.position) == 1:
    self.population.global_guide[:, :] = np.repeat(self.memory.position, self.params.population_size, axis=0)
  else:
    # Evaluate sigma
    self.memory.sigma = sigma_evaluation(self, self.memory.fitness)
    self.population.sigma[:, :] = sigma_evaluation(self, self.population.fitness)
    # Choose the global guide for the population by the nearest neighbors using sigma value
    nearest_idxs = nearest_sigma_in_memory(self, np.arange(self.params.population_size))
    self.population.global_guide[:, :] = self.memory.position[nearest_idxs]

def sigma_method_in_fronts(self: Mesh) -> None:
  ''' Global guide search by sigma method in fronts. The global guide for each particle in the population will be the nearest particle different from itself in the previous front, by the sigma value. Particles in the Pareto front will choose the global guide from memory.
  
  Note:
    The previous front is the front with domination rank immediately lower than the domination rank of the current front. The domination ranks are ordered from the lowest to the highest, starting at the Pareto front with zero.

  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
  '''

  # Get the fronts and its length
  fronts = self.get_non_domination_fronts(self.population.fitness)
  num_fronts = len(fronts)
  pareto_idxs = fronts[0]
  # Evaluate population sigma
  self.population.sigma[:, :] = sigma_evaluation(self, self.population.fitness)
  if len(self.memory.position) == 1:
    self.population.global_guide[pareto_idxs] = np.repeat(self.memory.position, len(pareto_idxs), axis=0)
  else:
    # Evaluate memory sigma
    self.memory.sigma = sigma_evaluation(self, self.memory.fitness)
    # Choose the global guide for the Pareto front by the nearest neighbors using sigma value
    nearest_idxs = nearest_sigma_in_memory(self, pareto_idxs)
    self.population.global_guide[pareto_idxs] = self.memory.position[nearest_idxs]
  # Choose the global guide for others fronts by the nearest neighbors from the front using sigma value (This part is inefficient in Python)
  search_front = pareto_idxs
  for i in range(1, num_fronts):
    current_front = fronts[i]
    nearest_idxs = nearest_sigma_in_fronts(self, current_front, search_front)
    search_front = current_front
    self.population.global_guide[current_front] = self.population.position[nearest_idxs]

# The options of global guide attribution operation
global_guide_method_options = {
  0: sigma_method_in_memory,
  1: sigma_method_in_fronts,
}
''' The options of global guide attribution operation. They are:

  - :type:`0`: Applies Sigma method in memory to select the global guides.
  - :type:`1`: Applies Sigma method in fronts to select the global guides. Each particle will select its global guide from the next front. Particles in Pareto front will select the global guide from memory.
  - :type:`2`: Chooses randomly under Uniform Distribution a particle from memory.
  - :type:`3`: Chooses randomly under Uniform Distribution a particle from fronts. Each particle will select its global guide from the next front. Particles in Pareto front will select the global guide from memory.
'''

def get_global_guide_method(option: int) -> Callable[[Mesh], None]:
  ''' Sets the global guide method according to :attr:`~mesh.operations.global_guide_method.global_guide_method_options`.
  
  Args:
    option (:type:`int`): Defines the global guide method.

  Returns:
    :type:`Callable[[Mesh], None]`: The respective function to select the global guide.
  '''

  return global_guide_method_options[option]