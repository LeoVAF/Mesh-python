from mesh.core import Mesh
from mesh.parameters import MeshParameters

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

# Fixed the parameters for some tests
test_params = MeshParameters(
    objective_dim=objective_dim,
    position_dim=position_dim,
    position_lower_bounds=lower_bound,
    position_upper_bounds=upper_bound,
    population_size=population_size,
    memory_size=population_size,
    mutation_rate=mutation_rate,
    communication_probability=communication_probability,
    max_gen=max_gen,
    max_fit_eval=max_fit_eval,
    max_personal_guides=max_personal_guides,
    random_state=random_state
  )
toy_function = lambda x: np.array([(1 - 2 * (i % 2)) * x[i % position_dim] for i in range(objective_dim)])

''' ######################################################################## '''
def test_Mesh():
  pass

''' ######################################################################## '''
def test_initialize():
  # Initialize the algortihm
  mesh = Mesh(test_params, toy_function)
  mesh.initialize()

def test_sequential_fitness_evaluation():
  # Initialize the algortihm
  mesh = Mesh(test_params, toy_function)
  mesh.initialize()

  # Test the fitness evaluation
  positions = np.random.rand(population_size, position_dim)
  fitnesses = mesh.sequential_fitness_evaluation(positions)
  for i, p in enumerate(positions):
    assert np.array_equal(toy_function(p), fitnesses[i])

def test_parallel_fitness_evaluation():
  # Initialize the algortihm
  mesh = Mesh(test_params, toy_function, num_proc=4)
  mesh.initialize()

  # Test the fitness evaluation
  positions = np.random.rand(population_size, position_dim)
  fitnesses = mesh.parallel_fitness_evaluation(positions)
  for i, p in enumerate(positions):
    assert np.array_equal(toy_function(p), fitnesses[i])

''' ######################################################################## '''
def test_differential_evolution():
  pass

''' ######################################################################## '''
def test_mutate_weights():
  pass

''' ######################################################################## '''
def test_reflect_velocity_at_bounds():
  pass

''' ######################################################################## '''
def test_move_population():
  pass

''' ######################################################################## '''
def test_population_selection():
  pass

''' ######################################################################## '''
def test_update_personal_guides():
  pass

''' ######################################################################## '''
def test_update_memory():
  pass

def test_stopping_by_generation():
  # Initialize the algoritm
  maximum_generations = np.random.randint(1, 10)
  params = MeshParameters(
    objective_dim=objective_dim,
    position_dim=position_dim,
    position_lower_bounds=lower_bound,
    position_upper_bounds=upper_bound,
    population_size=population_size,
    max_gen=maximum_generations
  )
  mesh = Mesh(params, toy_function)

  # Run the algorithm and check if the maximum generations was counted correctly 
  mesh.run()

  assert mesh.generation_counter == maximum_generations

def test_stopping_by_fitness_evalution():
  # Initialize the algoritm
  maximum_fitnes_evaluations = np.random.randint(1, population_size * 2)
  params = MeshParameters(
    objective_dim=objective_dim,
    position_dim=position_dim,
    position_lower_bounds=lower_bound,
    position_upper_bounds=upper_bound,
    population_size=population_size,
    max_fit_eval=maximum_fitnes_evaluations
  )
  mesh = Mesh(params, toy_function)

  # Run the algorithm and check if the maximum fitness evaluations was counted correctly
  mesh.run()

  assert mesh.fitness_eval_counter == maximum_fitnes_evaluations