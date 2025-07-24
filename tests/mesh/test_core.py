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
  # Initialize the algorithm with initial positions 
  initial_positions = np.array([[i % 2] * position_dim for i in range(population_size)])
  params = MeshParameters(
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
    initial_positions=initial_positions,
    random_state=random_state
  )
  mesh = Mesh(params, lambda x: [x[0] for _ in range(objective_dim)])
  mesh.initialize()

  # Set the velocity
  mesh.population.velocity = np.array([[i % 2] * position_dim for i in range(population_size)])

  # Copy the particles
  mesh.pre_allocated.position_copy = mesh.population.position.copy()
  mesh.pre_allocated.velocity_copy = mesh.population.velocity.copy()
  mesh.pre_allocated.fitness_copy = mesh.population.fitness.copy()

  # Select only the particles with the fitness equals to zero
  mesh.population_selection()

  # Check if the particles were selected correctly
  for i in range(population_size):
    assert np.array_equal(mesh.population.position[i], np.zeros(position_dim))
    # assert np.array_equal()
    # assert np.array_equal()

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
  # Initialize the algoritm with fitness evaluations less or equal than the number of particles
  maximum_fitnes_evaluations = np.random.randint(1, population_size + 1)
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

  # Initialize the algorithm with fitness evaluations greater than the number of particles
  maximum_fitnes_evaluations = np.random.randint(population_size + 1, 2 * population_size)
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