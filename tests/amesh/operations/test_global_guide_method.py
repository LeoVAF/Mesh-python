import numpy as np

from amesh import AMESH
from amesh.operations import global_guide_method as gba
from amesh.parameters import AMESHParameters

# ---------- Fixed parameters for test setup ----------
objective_dim = np.random.randint(2, 101) # Randomly choose objective dimension
decision_dim = np.random.randint(2, 101) # Randomly choose position dimension
population_size = np.random.randint(4, 101) # Randomly choose population size
lower_bound = np.array([0] * decision_dim)
upper_bound = np.array([5] * decision_dim)
mutation_rate = 0.5
communication_probability = 0.8
max_gen = None
max_fit_eval = 500
max_personal_guides = 3
random_state = None

equal_tolerance_for_array = 1e-15

def toy_function(x):
  return np.array([x[0], 1 - x[0]] + [x[0] for _ in range(objective_dim-2)])
def rank_function(x):
  return np.array([x[0] + x[1], x[0] + 1 - x[1]] + [x[0] for _ in range(objective_dim-2)]) # x[0] controls the particle rank


def test_sigma_evaluation():
  # Create an AMESH instance with a toy function
  test_params = AMESHParameters(
    objective_dim=objective_dim,
    decision_dim=decision_dim,
    decision_lower_bounds=lower_bound,
    decision_upper_bounds=upper_bound,
    population_size=population_size,
    max_gen=None,
    max_fit_eval=max_fit_eval,
    max_personal_guides=max_personal_guides,
    random_state=random_state
  )
  amesh = AMESH(test_params, toy_function)
  
  # Initialize the algorithm
  amesh.initialize()

  # Run the global guide search method
  sigma_arrays = gba.sigma_evaluation(amesh, amesh.population.fitness)

  # Check if the operation is correctly applied
  for idx, fitness in enumerate(amesh.population.fitness):
    fitness_squared_sum = np.sum(fitness ** 2)
    sigma_array = []
    for i in range(1, objective_dim):
      for j in range(i):
        sigma_array.append(fitness[i] ** 2 - fitness[j] ** 2)
    sigma_array = np.array(sigma_array) / fitness_squared_sum
    # Treating numeric errors
    assert np.linalg.norm(sigma_arrays[idx] - sigma_array) < equal_tolerance_for_array
  
  # Check the case with fitnesses equal to zero
  sigma_arrays = gba.sigma_evaluation(amesh, np.zeros((population_size, objective_dim)))
  for idx in range(population_size):
    sigma_array = []
    for i in range(1, objective_dim):
      for j in range(i):
        sigma_array.append(0)
    # Treating numeric errors
    assert np.array_equal(sigma_arrays[idx], sigma_array)


def test_sigma_method_in_memory():
  # Create an AMESH instance with a toy function
  steps = np.linspace(0, 1, population_size)
  initial_points = np.hstack((np.array([[steps[i]] for i in range(population_size)]), np.random.rand(population_size, decision_dim-1)))
  test_params = AMESHParameters(
    objective_dim=objective_dim,
    decision_dim=decision_dim,
    decision_lower_bounds=lower_bound,
    decision_upper_bounds=upper_bound,
    population_size=population_size,
    global_guide_method=0,
    max_gen=None,
    max_fit_eval=max_fit_eval,
    max_personal_guides=max_personal_guides,
    initial_points=initial_points,
    random_state=random_state
  )
  amesh = AMESH(test_params, toy_function)
  
  # Initialize the algorithm
  amesh.initialize()

  # Find the global guide for each particle
  amesh.global_guide_method()

  # Check the global guide search
  for idx in range(population_size):
    min_dist = np.inf
    particle_sigma = amesh.population.sigma[idx]
    nearest_idx = None
    for mem_idx, memory_sigma in enumerate(amesh.memory.sigma):
      dist = np.linalg.norm(particle_sigma - memory_sigma)
      if dist < min_dist and dist != 0:
        nearest_idx = mem_idx
        min_dist = dist
    assert np.array_equal(amesh.population.global_guide[idx], amesh.memory.position[nearest_idx])

def test_sigma_method_in_fronts():
  # Create an AMESH instance with a rank function
  steps = np.linspace(0, 1, population_size)
  ranks = [0, 4]
  initial_points = np.hstack((np.array([[ranks[i % len(ranks)]] for i in range(population_size - 1)] + [[2]]),
                              np.array([[steps[i]] for i in range(population_size)]),
                              np.random.rand(population_size, decision_dim-2)))
  test_params = AMESHParameters(
    objective_dim=objective_dim,
    decision_dim=decision_dim,
    decision_lower_bounds=lower_bound,
    decision_upper_bounds=upper_bound,
    population_size=population_size,
    global_guide_method=1,
    max_gen=None,
    max_fit_eval=max_fit_eval,
    max_personal_guides=max_personal_guides,
    initial_points=initial_points,
    random_state=random_state
  )
  amesh = AMESH(test_params, rank_function)
  
  # Initialize the algorithm
  amesh.initialize()

  # Find the global guide for each particle
  amesh.global_guide_method()

  # Check the global guide from memory
  amesh_fronts = amesh.get_non_domination_fronts(amesh.population.fitness)
  for idx in amesh_fronts[0]:
    min_dist = np.inf
    particle_sigma = amesh.population.sigma[idx]
    nearest_idx = None
    for mem_idx, memory_sigma in enumerate(amesh.memory.sigma):
      dist = np.linalg.norm(particle_sigma - memory_sigma)
      if dist < min_dist and dist != 0:
        nearest_idx = mem_idx
        min_dist = dist
    assert np.array_equal(amesh.population.global_guide[idx], amesh.memory.position[nearest_idx])

  # Check the global guide from fronts
  for rank in range(1, len(amesh_fronts)):
    for idx in amesh_fronts[rank]:
      min_dist = np.inf
      particle_sigma = amesh.population.sigma[idx]
      nearest_idx = None
      search_front = amesh_fronts[rank-1]
      for search_idx, search_sigma in enumerate(amesh.population.sigma[search_front]):
        dist = np.linalg.norm(particle_sigma - search_sigma)
        if dist < min_dist and dist != 0:
          nearest_idx = search_front[search_idx]
          min_dist = dist
      assert np.array_equal(amesh.population.global_guide[idx], amesh.population.position[nearest_idx])
