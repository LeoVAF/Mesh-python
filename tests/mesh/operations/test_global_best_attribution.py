from mesh.core import Mesh
from mesh.operations import global_best_attribution as gba
from mesh.parameters import MeshParameters

import numpy as np

# ---------- Fixed parameters for test setup ----------
objective_dim = np.random.randint(2, 101)  # Randomly choose objective dimension
position_dim = 5
population_size = 100
lower_bound = np.array([0] * position_dim)
upper_bound = np.array([1] * position_dim)
mutation_rate = 0.5
communication_probability = 0.8
max_gen = None
max_fit_eval = 500
max_personal_guides = 3
random_state = 45

equal_tolerance_for_array = 1e-15

# Function for testing
toy_function = lambda x: np.array([x[0], 1 - x[0]] + [x[0] for _ in range(objective_dim-2)])

def test_sigma_evaluation():
  # Create a Mesh instance with a toy function
  params = MeshParameters(
    objective_dim=objective_dim,
    position_dim=position_dim,
    lower_bound_array=lower_bound,
    upper_bound_array=upper_bound,
    population_size=population_size,
    memory_size=population_size,
    mutation_rate=mutation_rate,
    communication_probability=communication_probability,
    max_gen=None,
    max_fit_eval=max_fit_eval,
    max_personal_guides=max_personal_guides,
    random_state=random_state
  )
  mesh = Mesh(params, toy_function)
  
  # Initialize the algorithm
  mesh.initialize()

  # Run the global best attribution operation
  sigma_arrays = gba.sigma_evaluation(mesh, mesh.population.fitness)

  # Check if the operation is correctly applied
  for idx, fitness in enumerate(mesh.population.fitness):
    fitness_squared_sum = np.sum(fitness ** 2)
    sigma_array = []
    for i in range(1, objective_dim):
      for j in range(0, i):
        sigma_array.append(fitness[i] ** 2 - fitness[j] ** 2)
    sigma_array = np.array(sigma_array) / fitness_squared_sum
    # Treating numeric errors
    assert np.linalg.norm(sigma_arrays[idx] - sigma_array) < equal_tolerance_for_array

def test_sigma_method_in_memory():
  # Create a Mesh instance with a toy function
  params = MeshParameters(
    objective_dim=objective_dim,
    position_dim=position_dim,
    lower_bound_array=lower_bound,
    upper_bound_array=upper_bound,
    population_size=population_size,
    memory_size=population_size,
    global_best_attribution_type=0,
    mutation_rate=mutation_rate,
    communication_probability=communication_probability,
    max_gen=None,
    max_fit_eval=max_fit_eval,
    max_personal_guides=max_personal_guides,
    random_state=random_state
  )
  mesh = Mesh(params, toy_function)
  
  # Initialize the algorithm
  mesh.initialize()

  # Find the global best for each particle
  mesh.global_best_attribution()

  # Check the global best attribution
  for idx in range(population_size):
    min_dist = np.inf
    particle_sigma = mesh.population.sigma[idx]
    nearest_idx = None
    for mem_idx, memory_sigma in enumerate(mesh.memory.sigma):
      dist = np.linalg.norm(particle_sigma - memory_sigma)
      if dist < min_dist and dist != 0:
        nearest_idx = mem_idx
        min_dist = dist
    assert np.array_equal(mesh.population.global_best[idx], mesh.memory.position[nearest_idx])
  
  # Testing when memory has only one particle
  params = MeshParameters(
    objective_dim=objective_dim,
    position_dim=position_dim,
    lower_bound_array=lower_bound,
    upper_bound_array=upper_bound,
    population_size=population_size,
    memory_size=1,
    global_best_attribution_type=0,
    mutation_rate=mutation_rate,
    communication_probability=communication_probability,
    max_gen=None,
    max_fit_eval=max_fit_eval,
    max_personal_guides=max_personal_guides,
    random_state=random_state
  )
  mesh = Mesh(params, toy_function)
  mesh.initialize()
  mesh.global_best_attribution()

  # Check the best global attribution in the case where the memory has only one particle
  for idx in range(population_size):
    assert np.array_equal(mesh.population.global_best[idx], mesh.memory.position[0])

def test_sigma_method_in_fronts():
  # Create a Mesh instance with a toy function
  params = MeshParameters(
    objective_dim=objective_dim,
    position_dim=position_dim,
    lower_bound_array=lower_bound,
    upper_bound_array=upper_bound,
    population_size=population_size,
    memory_size=population_size,
    global_best_attribution_type=1,
    mutation_rate=mutation_rate,
    communication_probability=communication_probability,
    max_gen=None,
    max_fit_eval=max_fit_eval,
    max_personal_guides=max_personal_guides,
    random_state=random_state
  )
  mesh = Mesh(params, toy_function)
  
  # Initialize the algorithm and set some informations
  mesh.initialize()

  # Find the global best for each particle
  mesh.global_best_attribution()

  # Check the global best from memory
  for idx in mesh.fronts[0]:
    min_dist = np.inf
    particle_sigma = mesh.population.sigma[idx]
    nearest_idx = None
    for mem_idx, memory_sigma in enumerate(mesh.memory.sigma):
      dist = np.linalg.norm(particle_sigma - memory_sigma)
      if dist < min_dist and dist != 0:
        nearest_idx = mem_idx
        min_dist = dist
    assert np.array_equal(mesh.population.global_best[idx], mesh.memory.position[nearest_idx])

  # Check the global best from fronts
  for rank in range(1, len(mesh.fronts)):
    for idx in mesh.fronts[rank]:
      min_dist = np.inf
      particle_sigma = mesh.population.sigma[idx]
      nearest_idx = None
      search_front = mesh.fronts[rank-1]
      for search_idx, search_sigma in enumerate(mesh.population.sigma[search_front]):
        dist = np.linalg.norm(particle_sigma - search_sigma)
        if dist < min_dist and dist != 0:
          nearest_idx = search_front[search_idx]
          min_dist = dist
      assert np.array_equal(mesh.population.global_best[idx], mesh.population.position[nearest_idx])

  # Create a Mesh instance with a rank function (one particle in each front)
  params = MeshParameters(
    objective_dim=objective_dim,
    position_dim=position_dim,
    lower_bound_array=lower_bound,
    upper_bound_array=upper_bound,
    population_size=population_size,
    memory_size=population_size,
    global_best_attribution_type=1,
    mutation_rate=mutation_rate,
    communication_probability=communication_probability,
    max_gen=None,
    max_fit_eval=max_fit_eval,
    max_personal_guides=max_personal_guides,
    random_state=random_state
  )
  mesh = Mesh(params, toy_function)
  # Initialize the algorithm
  mesh.initialize()
  # Find the global best for each particle
  mesh.global_best_attribution()

  # Check the global best from memory
  for idx in mesh.fronts[0]:
    assert np.array_equal(mesh.population.global_best[idx], mesh.memory.position[0])

  # Check the global best from fronts
  for rank in range(1, len(mesh.fronts)):
    for idx in mesh.fronts[rank]:
      min_dist = np.inf
      particle_sigma = mesh.population.sigma[idx]
      nearest_idx = None
      search_front = mesh.fronts[rank-1]
      for search_idx, search_sigma in enumerate(mesh.population.sigma[search_front]):
        dist = np.linalg.norm(particle_sigma - search_sigma)
        if dist < min_dist and dist != 0:
          nearest_idx = search_front[search_idx]
          min_dist = dist
      assert np.array_equal(mesh.population.global_best[idx], mesh.population.position[nearest_idx])