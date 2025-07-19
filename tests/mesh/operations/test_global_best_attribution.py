from mesh.core import Mesh
from mesh.operations import global_best_attribution as gba
from mesh.parameters import MeshParameters

from math import comb
import numpy as np

# ---------- Fixed parameters for test setup ----------
objective_dim = np.random.randint(1, 101)  # Randomly choose objective dimension
position_dim = 20
population_size = 5
lower_bound = np.array([0] * position_dim)
upper_bound = np.array([1] * position_dim)
mutation_rate = 0.5
communication_probability = 0.8
max_gen = None
max_fit_eval = 200
max_personal_guides = 3
random_state = None

toy_function = lambda x: np.array([np.random.choice([-1, 1]) * np.random.choice(x) for _ in range(objective_dim)])

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
    assert np.array_equal(sigma_arrays[idx], sigma_array)