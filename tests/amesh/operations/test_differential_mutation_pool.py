import numpy as np

from amesh import AMESH
from amesh.operations import differential_mutation_pool as dmp
from amesh.parameters import AMESHParameters

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
  # Initialize a random AMESH instance
  test_params = AMESHParameters(objective_dim=objective_dim,
                          decision_dim=decision_dim, decision_lower_bounds=lower_bound, decision_upper_bounds=upper_bound, 
                          population_size=population_size,
                          dm_pool_type=1,
                          max_gen=max_gen, max_fit_eval=max_fit_eval,
                          max_personal_guides=max_personal_guides,
                          random_state=random_state)
  amesh = AMESH(test_params, toy_function, log_memory=None)

  # Initialize the algorithm
  amesh.initialize()

  # Get the pool list
  pool, pool_idxs = dmp.pool_from_population(amesh)

  # Check if each particle or personal best position in the respective particle pool is not in the pool
  for i, idxs in enumerate(pool_idxs):
    assert all(not np.array_equal(amesh.population.position[i], pool[idx]) for idx in idxs)

def test_pool_from_memory():
  # Initialize a random AMESH instance
  test_params = AMESHParameters(objective_dim=objective_dim,
                          decision_dim=decision_dim, decision_lower_bounds=lower_bound, decision_upper_bounds=upper_bound, 
                          population_size=population_size,
                          dm_pool_type=0,
                          max_gen=max_gen, max_fit_eval=max_fit_eval,
                          max_personal_guides=max_personal_guides,
                          random_state=random_state)
  amesh = AMESH(test_params, toy_function, log_memory=None)

  # Initialize the algorithm
  amesh.initialize()

  # Get the pool list
  pool, pool_idxs = dmp.pool_from_memory(amesh)

  # Check if each particle or personal best position in the respective particle pool is not in the pool
  for i, idxs in enumerate(pool_idxs):
    assert all(not np.array_equal(amesh.population.position[i], pool[idx]) for idx in idxs)