from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from MESH import Mesh

""" Calculate the sigma value for the particle set """
def sigma_evaluation(self, fitness_matrix):
  # Get the squared fitness matrix
  squared_fitnesses = np.square(fitness_matrix)
  # Get the sum of each line in the fitness matrix
  sum_squared_fitnesses = np.sum(squared_fitnesses, axis=1, keepdims=True)
  # Take the indexes to make the combination of differences (simulate a lower triangular matrix per vector to make the differences efficiently)
  row_indexes, col_indexes = self.pre_allocated.np_tril_indices
  # Get the fitness differences
  differences = squared_fitnesses[:, row_indexes] - squared_fitnesses[:, col_indexes]
  # Calculate the sigma values for each particle
  return differences / sum_squared_fitnesses

''' Find the nearest particle by the sigma value from memory '''
def sigma_nearest_by_memory(self, particle_idxs):
  memory_sigma = self.memory.sigma
  num_particles = len(particle_idxs)
  # If there is just one particle in the memory, it is the global best of all indexed particles
  if len(memory_sigma) == 1:
    return np.zeros(num_particles, dtype=int)
  else:
    # Get the nearest neighbor distances and indices
    distances, indices = self.pre_allocated.nearest_neighbors.fit(memory_sigma).kneighbors(self.population.sigma[particle_idxs])
    # The nearest neighbor must be different from itself
    zero_distances_mask = distances[:, 0] == 0
    first_valid_idxs = np.where(zero_distances_mask, 1, 0)
    # Return the nearest indices
    return indices[np.arange(num_particles), first_valid_idxs]

''' Find the nearest particle by the sigma value from the previous frontier '''
def sigma_nearest_by_fronts(self, particle_idxs, search_idxs):
  population_sigma = self.population.sigma
  num_particles = len(particle_idxs)
  # If there is just one particle in the memory, it is the global best of all indexed particles
  if len(search_idxs) == 1:
    return np.zeros(num_particles, dtype=int)
  else:
    # Get the nearest neighbor distances and indices
    distances, indices = self.pre_allocated.nearest_neighbors.fit(population_sigma[search_idxs]).kneighbors(population_sigma[particle_idxs])
    # The nearest neighbor must be different from itself
    non_zero_distances_mask = distances[:, 0] == 0
    first_valid_idxs = np.where(non_zero_distances_mask, 1, 0)
    return indices[np.arange(num_particles), first_valid_idxs]

''' Global best attribution with sigma in memory '''
def sigma_method_in_memory(self):
  # Evaluate sigma
  self.memory.sigma = sigma_evaluation(self, self.memory.fitness)
  self.population.sigma[:, :] = sigma_evaluation(self, self.population.fitness)
  # Choose the global best for the population by the nearest neighbors using sigma value
  nearest_idxs = sigma_nearest_by_memory(self, np.arange(self.params.population_size))
  self.population.global_best[:, :] = self.memory.position[nearest_idxs]

''' Global best attribution with sigma in fronts '''
def sigma_method_in_fronts(self):
  # Get the fronts and its length
  fronts = self.fronts
  num_fronts = len(fronts)
  # Evaluate sigma
  self.memory.sigma = sigma_evaluation(self, self.memory.fitness)
  self.population.sigma[:, :] = sigma_evaluation(self, self.population.fitness)
  # Choose the global best for the Pareto frontier by the nearest neighbors using sigma value
  pareto_idxs = fronts[0]
  nearest_idxs = sigma_nearest_by_memory(self, pareto_idxs)
  self.population.global_best[pareto_idxs] = self.memory.position[nearest_idxs]
  # Choose the global best for others frontiers by the nearest neighbors from the previous frontier using sigma value
  prev_front = fronts[0]
  for i in range(1, num_fronts):
    current_front = fronts[i]
    nearest_idxs = sigma_nearest_by_fronts(self, current_front, prev_front)
    prev_front = current_front
    self.population.global_best[nearest_idxs] = self.population.position[nearest_idxs]

''' Global best attribution with choosing randomly in memory '''
def random_in_memory(self):
  # Get the random indexes for the particles from memory
  random_indices = np.random.randint(0, len(self.memory.position), size=self.params.population_size)
  # Choose the global best
  self.population.global_best[:, :] = self.memory.position[random_indices]

''' Global best attribution with choosing randomly in memory '''
def random_in_fronts(self):
  # Get the masks for the respective rank positions
  rank_zero_mask = self.population.rank == 0
  rank_non_zero_mask = ~rank_zero_mask
  # Set the global best from memory
  num_rank_zero = np.sum(rank_zero_mask)
  self.population.global_best[rank_zero_mask] = self.memory.position[np.random.randint(0, len(self.memory.position), size=num_rank_zero)]
  # Get the particles indices which have rank greater than zero
  prev_front_idxs = self.population.rank[rank_non_zero_mask] - 1
  if(len(prev_front_idxs)):
    # Get the fronts and the front sizes
    fronts = self.fronts
    # Generate the random indices for ranks greater than zero
    random_indices = np.random.randint(0, [len(fronts[r]) for r in prev_front_idxs])
    rank_non_zero_idxs = np.array([fronts[idx][random_indices[i]] for i, idx in enumerate(prev_front_idxs)])
    # Set the global best from previous front
    self.population.global_best[rank_non_zero_mask] = self.population.position[rank_non_zero_idxs]


# The options of global guide attribution
global_best_attribution_options = {
  0: sigma_method_in_memory,
  1: sigma_method_in_fronts,
  2: random_in_memory,
  3: random_in_fronts
}

''' Choose the global best attribution type'''
def get_global_best_attribution(type: {0, 1, 2, 3}):
  return global_best_attribution_options[type]