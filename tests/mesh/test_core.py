from mesh.core import Mesh
from mesh.parameters import MeshParameters

import numpy as np

# ---------- Fixed parameters for test setup ----------
objective_dim = 5
position_dim = 5
population_size = 5
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

equal_tolerance_for_array = 1e-15

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

def test_mutation(mocker):
  # Initialize the algortihm
  mesh = Mesh(test_params, toy_function)
  mesh.initialize()

  # Copy the weights
  weights = mesh.weights.copy()

  # Mock the random function to return predetermined values
  weight_noise = np.random.normal(0.0, 1.0, size=(3, mesh.params.population_size))
  global_guide_noise = np.random.normal(0.0, 1.0, size=(mesh.params.population_size, mesh.params.position_dim))
  mocker.patch('numpy.random.normal', side_effect=[weight_noise, global_guide_noise])

  # Find the global guides
  mesh.global_guide_method()

  # Mutate the variables
  mesh.mutation()

  # Check if the mutation operation was applied correctly
  for i in range(3):
    assert np.linalg.norm(mesh.weights[i] - np.clip((weights[i] + weight_noise[i] * mesh.params.mutation_rate), 0, 1)) < equal_tolerance_for_array
  for i, gb_mut in enumerate(mesh.pre_allocated.global_guide_mutated):
    gb_expected = np.clip((mesh.population.global_guide[i] + global_guide_noise[i] * mesh.params.mutation_rate), mesh.params.position_lower_bounds, mesh.params.position_upper_bounds)
    assert np.linalg.norm(gb_mut - gb_expected) < equal_tolerance_for_array

''' ######################################################################## '''
def test_reflect_velocity_at_bounds():
  pass

def test_move_population(mocker):
  # Initialize the algortihm and prepare the population for the equation of motion
  mesh = Mesh(test_params, toy_function)
  mesh.initialize()
  mesh.population.personal_guide_pos = np.random.uniform(mesh.params.position_lower_bounds,
                                                         mesh.params.position_upper_bounds,
                                                         size=(mesh.params.population_size, mesh.params.max_personal_guides, mesh.params.position_dim))
  mesh.global_guide_method()
  mesh.mutation()

  # Mock the random function to return predetermined values
  pb_indices = np.random.randint(0, mesh.params.max_personal_guides, size=mesh.params.population_size)
  mocker.patch('numpy.random.randint', return_value=pb_indices)

  communication_probs = np.random.rand(mesh.params.population_size, mesh.params.position_dim)
  mocker.patch('numpy.random.rand', return_value=communication_probs)

  # Copy the population position and velocity
  positions = mesh.population.position.copy()
  velocities = mesh.population.velocity.copy()

  # Move the particles
  mesh.move_population()

  # Check if the particles were correctly moved
  W = mesh.weights
  C = communication_probs < mesh.params.communication_probability
  for i, x in enumerate(positions):
    # Check the velocity
    x_pb = mesh.population.personal_guide_pos[i, pb_indices[i], :]
    x_gb_mut = mesh.pre_allocated.global_guide_mutated[i]
    v = W[0, i] * velocities[i] + W[1, i] * (x_pb - x) + W[2, i] * C[i] * (x_gb_mut - x)
    np.clip(v, mesh.params.velocity_lower_bounds, mesh.params.velocity_upper_bounds, out=v)
    assert np.linalg.norm(mesh.population.velocity[i] - v) < equal_tolerance_for_array
    # Check the position
    x_clipped = np.clip(x + v, mesh.params.position_lower_bounds, mesh.params.position_upper_bounds)
    assert np.linalg.norm(mesh.population.position[i] - x_clipped) < equal_tolerance_for_array
    # Check the fitness
    assert np.linalg.norm(mesh.population.fitness[i] - mesh.fitness_function(x_clipped)) < equal_tolerance_for_array

# test_move_population(Mocker())

def test_elitism():
  test_population_size = 2 * population_size
  # Initialize the algorithm with initial positions
  initial_positions = np.array([[i % 2] * position_dim for i in range(test_population_size)])
  params = MeshParameters(
    objective_dim=objective_dim,
    position_dim=position_dim,
    position_lower_bounds=lower_bound,
    position_upper_bounds=upper_bound,
    population_size=test_population_size,
    memory_size=None,
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
  mesh.population.velocity = np.array([[i % 2] * position_dim for i in range(test_population_size)])

  # Copy the particles
  mesh.pre_allocated.position_copy = mesh.population.position.copy()
  mesh.pre_allocated.velocity_copy = mesh.population.velocity.copy()
  mesh.pre_allocated.fitness_copy = mesh.population.fitness.copy()

  # Select only the particles with the fitness equals to zero
  mesh.elitism()

  # Check if the particles were selected correctly
  for i in range(test_population_size):
    assert np.array_equal(mesh.population.position[i], np.zeros(position_dim))
    assert np.array_equal(mesh.population.velocity[i], np.zeros(position_dim))
    assert np.array_equal(mesh.population.fitness[i], np.zeros(objective_dim))
    for j in range(max_personal_guides):
      assert np.array_equal(mesh.population.personal_guide_pos[i, j], np.zeros(position_dim))
      assert np.array_equal(mesh.population.personal_guide_fit[i, j], np.zeros(objective_dim))
  
  # Initialize the algorithm with initial positions with one arrays in random indices
  one_idxs = np.random.choice(test_population_size, size=population_size, replace=False)
  initial_positions = np.zeros((test_population_size, position_dim))
  initial_positions[one_idxs] = np.ones((population_size, position_dim))
  params = MeshParameters(
    objective_dim=objective_dim,
    position_dim=position_dim,
    position_lower_bounds=lower_bound,
    position_upper_bounds=upper_bound,
    population_size=test_population_size,
    memory_size=None,
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
  mesh.population.velocity = np.zeros((test_population_size, position_dim))
  mesh.population.velocity[one_idxs] = np.ones((population_size, position_dim))

  # Copy the particles
  mesh.pre_allocated.position_copy = mesh.population.position.copy()
  mesh.pre_allocated.velocity_copy = mesh.population.velocity.copy()
  mesh.pre_allocated.fitness_copy = mesh.population.fitness.copy()

  # Select only the particles with the fitness equals to zero
  mesh.elitism()

  # Check if the particles were selected correctly
  for i in range(test_population_size):
    assert np.array_equal(mesh.population.position[i], np.zeros(position_dim))
    assert np.array_equal(mesh.population.velocity[i], np.zeros(position_dim))
    assert np.array_equal(mesh.population.fitness[i], np.zeros(objective_dim))
    for j in range(max_personal_guides):
      assert np.array_equal(mesh.population.personal_guide_pos[i, j], np.zeros(position_dim))
      assert np.array_equal(mesh.population.personal_guide_fit[i, j], np.zeros(objective_dim))

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