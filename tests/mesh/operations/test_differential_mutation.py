from mesh.core import Mesh
from mesh.operations import differential_mutation as dm
from mesh.parameters import MeshParameters

from random import sample
from scipy.stats import truncnorm
from unittest.mock import patch

import numpy as np

# ---------- Fixed parameters for test setup ----------
objective_dim = 5
position_dim = 5
population_size = 20
lower_bound = np.array([0] * position_dim)
upper_bound = np.array([1] * position_dim)
mutation_rate = 0.5
communication_probability = 0.8
max_gen = None
max_fit_eval = 200
max_personal_guides = 3
random_state = None

toy_function = lambda x: np.random.rand(objective_dim)

equal_tolerance_for_array = 1e-15

def test_rand_1():
  # Create a list of arrays to sample from
  pool = np.random.rand(population_size, position_dim)
  pool_idxs = [np.arange(np.random.randint(len(pool) + 1)) for _ in range(population_size)]

  # Get the valid indices where the arrays have at least the minimum number of elements
  valid_size = 3
  valid_idxs = [i for i in range(len(pool_idxs)) if len(pool_idxs[i]) >= valid_size]
  
  # Mock the random functions to return predetermined values
  operation_weight = truncnorm.rvs(0, 2, size=(len(valid_idxs), 1))
  random_idxs = [sample(pool_idxs[i].tolist(), k=valid_size) for i in valid_idxs]
  # The name sample is copied directly into the module's local namespace. The patch in "random.sample" does not override this.
  with patch("scipy.stats.truncnorm.rvs", return_value=operation_weight), patch("mesh.operations.differential_mutation.sample", side_effect=random_idxs):
    # Create a Mesh instance with a toy function
    test_params = MeshParameters(
      objective_dim=objective_dim,
      position_dim=position_dim,
      position_lower_bounds=lower_bound,
      position_upper_bounds=upper_bound,
      population_size=population_size,
      memory_size=population_size,
      dm_operation_type=0,
      mutation_rate=mutation_rate,
      communication_probability=communication_probability,
      max_gen=max_gen,
      max_fit_eval=max_fit_eval,
      max_personal_guides=max_personal_guides,
      random_state=random_state
    )
    mesh = Mesh(test_params, toy_function)
    
    # Initialize the algorithm
    mesh.initialize()

    # Call the operation function
    Xst, _ = dm.rand_1(mesh, (pool, pool_idxs))

    # Check the output
    for i, r_idxs in enumerate(random_idxs):
      x0 = pool[r_idxs[0]]
      x1 = pool[r_idxs[1]]
      x2 = pool[r_idxs[2]]
      xst = np.clip(x0 + operation_weight[i] * (x1 - x2), test_params.position_lower_bounds, test_params.position_upper_bounds)
      # Treating numeric errors
      assert np.linalg.norm(Xst[i] - xst) < equal_tolerance_for_array

    # Testing the case that the pool has less than minimum number of elements
    pool_idxs = [np.arange(np.random.randint(valid_size)) for _ in range(population_size)]
    Xst, idxs = dm.rand_1(mesh, (pool, pool_idxs))
    assert np.array_equal(Xst, np.array([])) and np.array_equal(idxs, np.array([]))

def test_rand_2():
  # Create a list of arrays to sample from
  pool = np.random.rand(population_size, position_dim)
  pool_idxs = [np.arange(np.random.randint(len(pool) + 1)) for _ in range(population_size)]

  # Get the valid indices where the arrays have at least the minimum number of elements
  valid_size = 5
  valid_idxs = [i for i in range(len(pool_idxs)) if len(pool_idxs[i]) >= valid_size]
  
  # Mock the random functions to return predetermined values
  operation_weight = truncnorm.rvs(0, 2, size=(len(valid_idxs), 1))
  random_idxs = [sample(pool_idxs[i].tolist(), k=valid_size) for i in valid_idxs]
  # The name sample is copied directly into the module's local namespace. The patch in "random.sample" does not override this.
  with patch("scipy.stats.truncnorm.rvs", return_value=operation_weight), patch("mesh.operations.differential_mutation.sample", side_effect=random_idxs):
    # Create a Mesh instance with a toy function
    test_params = MeshParameters(
      objective_dim=objective_dim,
      position_dim=position_dim,
      position_lower_bounds=lower_bound,
      position_upper_bounds=upper_bound,
      population_size=population_size,
      memory_size=population_size,
      dm_operation_type=1,
      mutation_rate=mutation_rate,
      communication_probability=communication_probability,
      max_gen=max_gen,
      max_fit_eval=max_fit_eval,
      max_personal_guides=max_personal_guides,
      random_state=random_state
    )
    mesh = Mesh(test_params, toy_function)

    # Initialize the algorithm
    mesh.initialize()

    # Call the operation function
    Xst, _ = dm.rand_2(mesh, (pool, pool_idxs))

    # Check the output
    for i, r_idxs in enumerate(random_idxs):
      x0 = pool[r_idxs[0]]
      x1 = pool[r_idxs[1]]
      x2 = pool[r_idxs[2]]
      x3 = pool[r_idxs[3]]
      x4 = pool[r_idxs[4]]
      xst = np.clip(x0 + operation_weight[i] * (x1 - x2 + x3 - x4), test_params.position_lower_bounds, test_params.position_upper_bounds)
      # Treating numeric errors
      assert np.linalg.norm(Xst[i] - xst) < equal_tolerance_for_array

    # Testing the case that the pool has less than minimum number of elements
    pool_idxs = [np.arange(np.random.randint(valid_size)) for _ in range(population_size)]
    Xst, idxs = dm.rand_2(mesh, (pool, pool_idxs))
    assert np.array_equal(Xst, np.array([])) and np.array_equal(idxs, np.array([]))

def test_best_1():
  # Create a list of arrays to sample from
  pool = np.random.rand(population_size, position_dim)
  pool_idxs = [np.arange(np.random.randint(len(pool) + 1)) for _ in range(population_size)]

  # Get the valid indices where the arrays have at least the minimum number of elements
  valid_size = 2
  valid_idxs = [i for i in range(len(pool_idxs)) if len(pool_idxs[i]) >= valid_size]
  
  # Mock the random functions to return predetermined values
  operation_weight = truncnorm.rvs(0, 2, size=(len(valid_idxs), 1))
  random_idxs = [sample(pool_idxs[i].tolist(), k=valid_size) for i in valid_idxs]
  # The name sample is copied directly into the module's local namespace. The patch in "random.sample" does not override this.
  with patch("scipy.stats.truncnorm.rvs", return_value=operation_weight), patch("mesh.operations.differential_mutation.sample", side_effect=random_idxs):
    # Create a Mesh instance with a toy function
    test_params = MeshParameters(
      objective_dim=objective_dim,
      position_dim=position_dim,
      position_lower_bounds=lower_bound,
      position_upper_bounds=upper_bound,
      population_size=population_size,
      memory_size=population_size,
      dm_operation_type=2,
      mutation_rate=mutation_rate,
      communication_probability=communication_probability,
      max_gen=max_gen,
      max_fit_eval=max_fit_eval,
      max_personal_guides=max_personal_guides,
      random_state=random_state
    )
    mesh = Mesh(test_params, toy_function)

    # Initialize the algorithm
    mesh.initialize()

    # Call the operation function
    Xst, idxs = dm.best_1(mesh, (pool, pool_idxs))

    # Check the output
    for i, r_idxs in enumerate(random_idxs):
      x0 = pool[r_idxs[0]]
      x1 = pool[r_idxs[1]]
      xgb = mesh.population.global_guide[idxs[i]]
      xst = np.clip(xgb + operation_weight[i] * (x0 - x1), test_params.position_lower_bounds, test_params.position_upper_bounds)
      # Treating numeric errors
      assert np.linalg.norm(Xst[i] - xst) < equal_tolerance_for_array

    # Testing the case that the pool has less than minimum number of elements
    pool_idxs = [np.arange(np.random.randint(valid_size)) for _ in range(population_size)]
    Xst, idxs = dm.best_1(mesh, (pool, pool_idxs))
    assert np.array_equal(Xst, np.array([])) and np.array_equal(idxs, np.array([]))

def test_current_to_best_1():
  # Create a list of arrays to sample from
  pool = np.random.rand(population_size, position_dim)
  pool_idxs = [np.arange(np.random.randint(len(pool) + 1)) for _ in range(population_size)]

  # Get the valid indices where the arrays have at least the minimum number of elements
  valid_size = 2
  valid_idxs = [i for i in range(len(pool_idxs)) if len(pool_idxs[i]) >= valid_size]
  
  # Mock the random functions to return predetermined values
  operation_weight = truncnorm.rvs(0, 2, size=(len(valid_idxs), 1))
  random_idxs = [sample(pool_idxs[i].tolist(), k=valid_size) for i in valid_idxs]
  # The name sample is copied directly into the module's local namespace. The patch in "random.sample" does not override this.
  with patch("scipy.stats.truncnorm.rvs", return_value=operation_weight), patch("mesh.operations.differential_mutation.sample", side_effect=random_idxs):
    # Create a Mesh instance with a toy function
    test_params = MeshParameters(
      objective_dim=objective_dim,
      position_dim=position_dim,
      position_lower_bounds=lower_bound,
      position_upper_bounds=upper_bound,
      population_size=population_size,
      memory_size=population_size,
      dm_operation_type=3,
      mutation_rate=mutation_rate,
      communication_probability=communication_probability,
      max_gen=max_gen,
      max_fit_eval=max_fit_eval,
      max_personal_guides=max_personal_guides,
      random_state=random_state
    )
    mesh = Mesh(test_params, toy_function)

    # Initialize the algorithm
    mesh.initialize()

    # Call the operation function
    Xst, idxs = dm.current_to_best_1(mesh, (pool, pool_idxs))

    # Check the output
    for i, r_idxs in enumerate(random_idxs):
      x0 = pool[r_idxs[0]]
      x1 = pool[r_idxs[1]]
      x = mesh.population.position[idxs[i]]
      xgb = mesh.population.global_guide[idxs[i]]
      xst = np.clip(x + operation_weight[i] * (xgb - x + x0 - x1), test_params.position_lower_bounds, test_params.position_upper_bounds)
      # Treating numeric errors
      assert np.linalg.norm(Xst[i] - xst) < equal_tolerance_for_array

    # Testing the case that the pool has less than minimum number of elements
    pool_idxs = [np.arange(np.random.randint(valid_size)) for _ in range(population_size)]
    Xst, idxs = dm.current_to_best_1(mesh, (pool, pool_idxs))
    assert np.array_equal(Xst, np.array([])) and np.array_equal(idxs, np.array([]))

def test_current_to_rand_1():
  # Create a list of arrays to sample from
  pool = np.random.rand(population_size, position_dim)
  pool_idxs = [np.arange(np.random.randint(len(pool) + 1)) for _ in range(population_size)]

  # Get the valid indices where the arrays have at least the minimum number of elements
  valid_size = 3
  valid_idxs = [i for i in range(len(pool_idxs)) if len(pool_idxs[i]) >= valid_size]
  
  # Mock the random functions to return predetermined values
  operation_weight = truncnorm.rvs(0, 2, size=(len(valid_idxs), 1))
  random_idxs = [sample(pool_idxs[i].tolist(), k=valid_size) for i in valid_idxs]
  # The name sample is copied directly into the module's local namespace. The patch in "random.sample" does not override this.
  with patch("scipy.stats.truncnorm.rvs", return_value=operation_weight), patch("mesh.operations.differential_mutation.sample", side_effect=random_idxs):
    # Create a Mesh instance with a toy function
    test_params = MeshParameters(
      objective_dim=objective_dim,
      position_dim=position_dim,
      position_lower_bounds=lower_bound,
      position_upper_bounds=upper_bound,
      population_size=population_size,
      memory_size=population_size,
      dm_operation_type=4,
      mutation_rate=mutation_rate,
      communication_probability=communication_probability,
      max_gen=max_gen,
      max_fit_eval=max_fit_eval,
      max_personal_guides=max_personal_guides,
      random_state=random_state
    )
    mesh = Mesh(test_params, toy_function)

    # Initialize the algorithm
    mesh.initialize()

    # Call the operation function
    Xst, idxs = dm.current_to_rand_1(mesh, (pool, pool_idxs))

    # Check the output
    for i, r_idxs in enumerate(random_idxs):
      x0 = pool[r_idxs[0]]
      x1 = pool[r_idxs[1]]
      x2 = pool[r_idxs[2]]
      x = mesh.population.position[idxs[i]]
      xst = np.clip(x + operation_weight[i] * (x0 - x + x1 - x2), test_params.position_lower_bounds, test_params.position_upper_bounds)
      # Treating numeric errors
      assert np.linalg.norm(Xst[i] - xst) < equal_tolerance_for_array

    # Testing the case that the pool has less than minimum number of elements
    pool_idxs = [np.arange(np.random.randint(valid_size)) for _ in range(population_size)]
    Xst, idxs = dm.current_to_rand_1(mesh, (pool, pool_idxs))
    assert np.array_equal(Xst, np.array([])) and np.array_equal(idxs, np.array([]))