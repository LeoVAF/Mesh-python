from mesh.core import Mesh
from mesh.operations import global_best_attribution as gba
from mesh.parameters import MeshParameters

import numpy as np

# ---------- Fixed parameters for test setup ----------
objective_dim = np.random.randint(2, 101)  # Randomly choose objective dimension
position_dim = 5
population_size = 50
lower_bound = np.array([0] * position_dim)
upper_bound = np.array([1] * position_dim)
mutation_rate = 0.5
communication_probability = 0.8
max_gen = None
max_fit_eval = 200
max_personal_guides = 3
random_state = 45

def generate_objective_function(objective_dim):
  def objective_function(position):
    position = np.array(position)
    objectives = []
    for i in range(objective_dim):
      # Switch between different patterns to create trade-offs
      if i % 3 == 0:
        # Sphere-type function with displacement
        obj = np.sum((position - (i + 1))**2)
      elif i % 3 == 1:
        # Wave-based function (nonlinear oscillation)
        obj = np.sum(np.sin(position * (i + 1))**2)
      else:
        # Mixed function: product between position and target index
        obj = np.prod(1 + position / (i + 1))
      # Normalizes each objective against the index
      obj /= (i + 1) * 10
      objectives.append(obj)
    return np.array(objectives)
  return objective_function
toy_function = generate_objective_function(objective_dim)

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
    assert np.linalg.norm(sigma_arrays[idx] - sigma_array) < 1e-15

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
  for i in range(population_size):
    min_dist = np.inf
    particle_sigma = mesh.population.sigma[i]
    nearest_idx = 0
    for j, memory_sigma in enumerate(mesh.memory.sigma):
      dist = np.linalg.norm(particle_sigma - memory_sigma)
      if dist < min_dist and dist != 0:
        nearest_idx = j
        min_dist = dist
    assert np.array_equal(mesh.population.global_best[i], mesh.memory.position[nearest_idx])
  
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
  for i in range(population_size):
    assert np.array_equal(mesh.population.global_best[i], mesh.memory.position[0])

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
  
  # Initialize the algorithm
  mesh.initialize()

  # Find the global best for each particle
  mesh.global_best_attribution()

  # Check the global best attribution
  for i in range(population_size):
    min_dist = np.inf
    particle_sigma = mesh.population.sigma[i]
    nearest_idx = 0
    for j, memory_sigma in enumerate(mesh.memory.sigma):
      dist = np.linalg.norm(particle_sigma - memory_sigma)
      if dist < min_dist and dist != 0:
        nearest_idx = j
        min_dist = dist
    assert np.array_equal(mesh.population.global_best[i], mesh.memory.position[nearest_idx])
  
  # Testing when memory has only one particle
  params = MeshParameters(
    objective_dim=objective_dim,
    position_dim=position_dim,
    lower_bound_array=lower_bound,
    upper_bound_array=upper_bound,
    population_size=population_size,
    memory_size=1,
    global_best_attribution_type=1,
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
  for i in range(population_size):
    assert np.array_equal(mesh.population.global_best[i], mesh.memory.position[0])