from mesh.core import Mesh
from mesh.operations import differential_mutation_pool as dmp
from mesh.parameters import MeshParameters

import numpy as np

toy_function = lambda x: (np.sum(x**2), -np.sum(x**2))

def test_pool_from_memory():
  # Initialize a random Mesh instance
  params = MeshParameters(objective_dim=2,
                          position_dim=5, lower_bound_array=np.array([0]*5), upper_bound_array=np.array([1]*5), 
                          population_size=20, memory_size=None,
                          global_best_attribution_type=0,
                          dm_pool_type=0,
                          dm_operation_type=0,
                          communication_probability=0.5, mutation_rate=0.5,
                          max_gen=0, max_fit_eval=200,
                          max_personal_guides=3,
                          random_state=None)
  mesh = Mesh(params, toy_function, log_memory=None)

  # Initialize the algorithm
  mesh.initialize()

  # Get the pool list
  pool_list = dmp.pool_from_memory(mesh)

  # Check if each particle or personal best position in the respective particle pool is not in the pool
  for i, pool in enumerate(pool_list):
    assert all([not np.array_equal(mesh.population.position[i], pos) for pos in pool])

def test_pool_from_population():
  # Initialize a random Mesh instance
  params = MeshParameters(objective_dim=2,
                          position_dim=5, lower_bound_array=np.array([0]*5), upper_bound_array=np.array([1]*5), 
                          population_size=20, memory_size=None,
                          global_best_attribution_type=0,
                          dm_pool_type=1,
                          dm_operation_type=0,
                          communication_probability=0.5, mutation_rate=0.5,
                          max_gen=0, max_fit_eval=200,
                          max_personal_guides=3,
                          random_state=None)
  mesh = Mesh(params, toy_function, log_memory=None)

  # Initialize the algorithm
  mesh.initialize()

  # Get the pool list
  pool_list = dmp.pool_from_population(mesh)

  # Check if each particle or personal best position in the respective particle pool is not in the pool
  for i, pool in enumerate(pool_list):
    assert all([not np.array_equal(mesh.population.position[i], pos) for pos in pool])

def test_pool_from_population_and_memory():
  # Initialize a random Mesh instance
  params = MeshParameters(objective_dim=2,
                          position_dim=5, lower_bound_array=np.array([0]*5), upper_bound_array=np.array([1]*5), 
                          population_size=3, memory_size=None,
                          global_best_attribution_type=0,
                          dm_pool_type=2,
                          dm_operation_type=0,
                          communication_probability=0.5, mutation_rate=0.5,
                          max_gen=0, max_fit_eval=200,
                          max_personal_guides=3,
                          random_state=None)
  mesh = Mesh(params, toy_function, log_memory=None)

  # Initialize the algorithm
  mesh.initialize()

  # Get the pool list
  pool_list = dmp.pool_from_population_and_memory(mesh)

  # Check if each particle or personal best position in the respective particle pool is not in the pool
  for i, pool in enumerate(pool_list):
    assert all([not np.array_equal(mesh.population.position[i], pos) for pos in pool])