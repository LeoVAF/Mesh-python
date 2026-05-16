from mesh import Mesh
from mesh.operations import differential_mutation_pool as dmp
from mesh.parameters import MeshParameters

import numpy as np

# ---------- Fixed parameters for test setup ----------
objective_dim = 5
decision_dim = 5
population_size = 20
lower_bound = np.array([0] * decision_dim)
upper_bound = np.array([1] * decision_dim)
mutation_rate = 0.5
communication_probability = 0.8
max_gen = None
max_fit_eval = 200
max_personal_guides = 3
random_state = None

def toy_function(x):
  return np.random.rand(objective_dim)

def test_pool_from_population():
  # Initialize a random Mesh instance
  test_params = MeshParameters(objective_dim=objective_dim,
                          decision_dim=decision_dim, decision_lower_bounds=lower_bound, decision_upper_bounds=upper_bound, 
                          population_size=population_size, memory_size=None,
                          dm_pool_type=1,
                          max_gen=max_gen, max_fit_eval=max_fit_eval,
                          max_personal_guides=max_personal_guides,
                          random_state=random_state)
  mesh = Mesh(test_params, toy_function, log_memory=None)

  # Initialize the algorithm
  mesh.initialize()

  # Get the pool list
  pool, pool_idxs = dmp.pool_from_population(mesh)

  # Check if each particle or personal best position in the respective particle pool is not in the pool
  for i, idxs in enumerate(pool_idxs):
    assert all([not np.array_equal(mesh.population.position[i], pool[idx]) for idx in idxs])

def test_pool_from_memory():
  # Initialize a random Mesh instance
  test_params = MeshParameters(objective_dim=objective_dim,
                          decision_dim=decision_dim, decision_lower_bounds=lower_bound, decision_upper_bounds=upper_bound, 
                          population_size=population_size, memory_size=None,
                          dm_pool_type=0,
                          max_gen=max_gen, max_fit_eval=max_fit_eval,
                          max_personal_guides=max_personal_guides,
                          random_state=random_state)
  mesh = Mesh(test_params, toy_function, log_memory=None)

  # Initialize the algorithm
  mesh.initialize()

  # Get the pool list
  pool, pool_idxs = dmp.pool_from_memory(mesh)

  # Check if each particle or personal best position in the respective particle pool is not in the pool
  for i, idxs in enumerate(pool_idxs):
    assert all([not np.array_equal(mesh.population.position[i], pool[idx]) for idx in idxs])

def test_pool_from_population_and_memory():
  # Initialize a random Mesh instance
  test_params = MeshParameters(objective_dim=objective_dim,
                          decision_dim=decision_dim, decision_lower_bounds=lower_bound, decision_upper_bounds=upper_bound, 
                          population_size=population_size, memory_size=None,
                          dm_pool_type=2,
                          max_gen=max_gen, max_fit_eval=max_fit_eval,
                          max_personal_guides=max_personal_guides,
                          random_state=random_state)
  mesh = Mesh(test_params, toy_function, log_memory=None)

  # Initialize the algorithm
  mesh.initialize()

  # Get the pool list
  pool, pool_idxs = dmp.pool_from_population_and_memory(mesh)

  # Check if each particle or personal best position in the respective particle pool is not in the pool
  for i, idxs in enumerate(pool_idxs):
    assert all([not np.array_equal(mesh.population.position[i], pool[idx]) for idx in idxs])