from mesh import Mesh
from mesh.parameters import MeshParameters
from mesh.auxiliar import StoppingAlgorithm

from unittest.mock import patch

import numpy as np
import pytest

# ---------- Fixed parameters for test setup ----------
objective_dim = 5
decision_dim = 5
population_size = 20
lower_bound = np.array([0] * decision_dim)
upper_bound = np.array([1] * decision_dim)
max_gen = None
max_fit_eval = 200
max_personal_guides = 3
random_state = None

params = MeshParameters(
    objective_dim=objective_dim,
    decision_dim=decision_dim,
    decision_lower_bounds=lower_bound,
    decision_upper_bounds=upper_bound,
    population_size=population_size,
    memory_size=population_size,
    max_gen=max_gen,
    max_fit_eval=max_fit_eval,
    max_personal_guides=max_personal_guides,
    random_state=random_state
  )
def toy_function(x):
  return np.array([x[0], 1 - x[0]] + [x[0] for _ in range(objective_dim-2)])
def rank_function(x):
  return np.array([x[0] + x[1], x[0] + 1 - x[1]] + [x[0] for _ in range(objective_dim-2)]) # x[0] controls the particle rank

equal_tolerance_for_array = 1e-15

def test_initialize():
  # Initialize the algortihm with none max_fit_eval
  test_params = MeshParameters(
    objective_dim=objective_dim,
    decision_dim=decision_dim,
    decision_lower_bounds=lower_bound,
    decision_upper_bounds=upper_bound,
    population_size=population_size,
    max_gen=1,
    max_fit_eval=None,
    random_state=random_state
  )
  mesh = Mesh(test_params, toy_function)
  mesh.initialize()

  # Initialize the algortihm with less fitness evaluations than population size
  test_params = MeshParameters(
    objective_dim=objective_dim,
    decision_dim=decision_dim,
    decision_lower_bounds=lower_bound,
    decision_upper_bounds=upper_bound,
    population_size=population_size,
    max_gen=None,
    max_fit_eval=population_size-1,
    random_state=random_state
  )
  mesh = Mesh(test_params, toy_function)
  with pytest.raises(StoppingAlgorithm, match=''):
    mesh.initialize()

  # Initialize the algortihm
  test_params = MeshParameters(
    objective_dim=objective_dim,
    decision_dim=decision_dim,
    decision_lower_bounds=lower_bound,
    decision_upper_bounds=upper_bound,
    population_size=population_size,
    max_gen=1,
    max_fit_eval=population_size+1,
    random_state=random_state
  )
  mesh = Mesh(test_params, toy_function)
  mesh.initialize()



def test_sequential_fitness_evaluation():
  # Initialize the algortihm
  mesh = Mesh(params, toy_function)
  mesh.initialize()

  # Test the fitness evaluation
  positions = np.random.rand(population_size, decision_dim)
  fitnesses = mesh.sequential_fitness_evaluation(positions)
  for i, p in enumerate(positions):
    assert np.array_equal(toy_function(p), fitnesses[i])

def test_parallel_fitness_evaluation():
  # Initialize the algortihm
  mesh = Mesh(params, toy_function, num_proc=4)
  mesh.initialize()

  # Test the fitness evaluation
  positions = np.random.rand(population_size, mesh.params.position_dim)
  fitnesses = mesh.parallel_fitness_evaluation(positions)
  for i, p in enumerate(positions):
    assert np.array_equal(toy_function(p), fitnesses[i])

def test_differential_evolution():
  test_population_size = 2 * population_size
  # Create a Mesh instance with a rank function
  steps = np.linspace(0, 1, test_population_size)
  ranks = [0, 4]
  initial_points = np.hstack((np.array([[ranks[i % len(ranks)]] for i in range(test_population_size)]),
                                 np.array([[steps[i]] for i in range(test_population_size)]),
                                 np.random.rand(test_population_size, decision_dim-2)))
  test_params = MeshParameters(
    objective_dim=objective_dim,
    decision_dim=decision_dim,
    decision_lower_bounds=lower_bound,
    decision_upper_bounds=upper_bound,
    population_size=test_population_size,
    memory_size=test_population_size,
    max_gen=max_gen,
    max_fit_eval=max_fit_eval,
    max_personal_guides=max_personal_guides,
    initial_points=initial_points,
    random_state=random_state
  )
  mesh = Mesh(test_params, rank_function)
  mesh.initialize()

  # Set the Xst and pop_idxs
  Xst = np.hstack((np.array([[0] for _ in range(population_size)]),
                   np.array([[steps[i]] for i in range(population_size)]),
                   np.random.rand(population_size, mesh.params.position_dim-2)))
  pop_idxs = np.array([i for i in range(population_size)])
  with patch.object(mesh, 'differential_mutation', return_value=(Xst, pop_idxs)), patch.object(mesh, 'differential_crossover', return_value=Xst):
    # Run the Differential Evolution phase
    mesh.differential_evolution()
    # Check if the strategy particles are in the population
    st_idxs = np.arange(1, test_population_size, 2)
    for i, idx in enumerate(st_idxs):
      assert np.array_equal(mesh.population.position[idx], Xst[i])

def test_mutation():
  # Initialize the algortihm
  mesh = Mesh(params, toy_function)
  mesh.initialize()

  mutation_rate = mesh.population.position[:, decision_dim+2:decision_dim+3]

  # Mock the random function to return predetermined values
  global_guide_noise = np.random.normal(0.0, 1.0, size=(mesh.params.population_size, mesh.params.position_dim))
  with patch('numpy.random.normal', return_value=global_guide_noise):

    # Find the global guides
    mesh.global_guide_method()

    # Mutate the variables
    mesh.mutation()

    # Check if the mutation operation was applied correctly
    for i, gb_mut in enumerate(mesh.pre_allocated.global_guide_mutated):
      gb_expected = np.clip(mesh.population.global_guide[i] + global_guide_noise[i] * mutation_rate[i],
                            mesh.params.position_lower_bounds,
                            mesh.params.position_upper_bounds)
      assert np.linalg.norm(gb_mut - gb_expected) < equal_tolerance_for_array

def test_move_population():
  # Initialize the algortihm and prepare the population for the equation of motion
  mesh = Mesh(params, toy_function)
  mesh.initialize()
  mesh.population.personal_guide_pos = np.random.uniform(mesh.params.position_lower_bounds,
                                                         mesh.params.position_upper_bounds,
                                                         size=(mesh.params.population_size, mesh.params.max_personal_guides, mesh.params.position_dim))
  mesh.global_guide_method()
  mesh.mutation()

  # Mock the random function to return predetermined values
  pb_indices = np.random.randint(0, mesh.params.max_personal_guides, size=mesh.params.population_size)
  communication_probs = np.random.rand(mesh.params.population_size, mesh.params.position_dim)
  with patch('numpy.random.randint', return_value=pb_indices), patch('numpy.random.rand', return_value=communication_probs):
    # Copy the population position, velocity and fitness
    mesh.pre_allocated.position_copy[:] = mesh.population.position.copy()
    mesh.pre_allocated.velocity_copy[:] = mesh.population.velocity.copy()
    mesh.pre_allocated.fitness_copy[:] = mesh.population.fitness.copy()

    # Get the parameter for the equation of motion
    W = mesh.population.position[:, decision_dim+4:]
    C = communication_probs <= mesh.population.position[:, decision_dim+3:decision_dim+4]

    # Move the particles
    mesh.move_population()

    for i, x in enumerate(mesh.population.position):
      # Check the velocity
      x_pb = mesh.population.personal_guide_pos[i, pb_indices[i], :]
      x_gb_mut = mesh.pre_allocated.global_guide_mutated[i]
      v = W[i, 0] * mesh.population.velocity[i] + W[i, 1] * (x_pb - x) + W[i, 2] * C[i] * (x_gb_mut - x)
      np.clip(v, mesh.params.velocity_lower_bounds, mesh.params.velocity_upper_bounds, out=v)
      assert np.linalg.norm(mesh.pre_allocated.velocity_copy[i] - v) < equal_tolerance_for_array
      # Check the position
      x_clipped = np.clip(x + v, mesh.params.position_lower_bounds, mesh.params.position_upper_bounds)
      assert np.linalg.norm(mesh.pre_allocated.position_copy[i] - x_clipped) < equal_tolerance_for_array
      # Check the fitness
      assert np.linalg.norm(mesh.pre_allocated.fitness_copy[i] - mesh.fitness_function(x_clipped)) < equal_tolerance_for_array

test_move_population()

def test_elitism():
  test_population_size = 2 * population_size
  # Initialize the algorithm with initial positions
  initial_points = np.array([[i % 2] * decision_dim for i in range(test_population_size)])
  test_params = MeshParameters(
    objective_dim=objective_dim,
    decision_dim=decision_dim,
    decision_lower_bounds=lower_bound,
    decision_upper_bounds=upper_bound,
    population_size=test_population_size,
    memory_size=None,
    max_gen=max_gen,
    max_fit_eval=max_fit_eval,
    max_personal_guides=max_personal_guides,
    initial_points=initial_points,
    random_state=random_state
  )
  def f1(x):
    return np.array([x[0] for _ in range(objective_dim)])
  mesh = Mesh(test_params, f1)
  mesh.initialize()

  # Set the velocity
  mesh.population.velocity = np.array([[i % 2] * mesh.params.position_dim for i in range(test_population_size)])

  # Copy the particles
  mesh.pre_allocated.position_copy = mesh.population.position.copy()
  mesh.pre_allocated.velocity_copy = mesh.population.velocity.copy()
  mesh.pre_allocated.fitness_copy = mesh.population.fitness.copy()

  # Select only the particles with the fitness equals to zero
  mesh.elitism()

  # Check if the particles were selected correctly
  for i in range(test_population_size):
    assert np.array_equal(mesh.population.position[i, :decision_dim], np.zeros(decision_dim))
    assert np.array_equal(mesh.population.velocity[i], np.zeros(mesh.params.position_dim))
    assert np.array_equal(mesh.population.fitness[i], np.zeros(objective_dim))
    for j in range(max_personal_guides):
      assert np.array_equal(mesh.population.personal_guide_pos[i, j, :decision_dim], np.zeros(decision_dim))
      assert np.array_equal(mesh.population.personal_guide_fit[i, j], np.zeros(objective_dim))
  
  # Initialize the algorithm with initial positions with one arrays in random indices
  one_idxs = np.random.choice(test_population_size, size=population_size, replace=False)
  initial_points = np.zeros((test_population_size, decision_dim))
  initial_points[one_idxs] = np.ones((population_size, decision_dim))
  test_params = MeshParameters(
    objective_dim=objective_dim,
    decision_dim=decision_dim,
    decision_lower_bounds=lower_bound,
    decision_upper_bounds=upper_bound,
    population_size=test_population_size,
    memory_size=None,
    max_gen=max_gen,
    max_fit_eval=max_fit_eval,
    max_personal_guides=max_personal_guides,
    initial_points=initial_points,
    random_state=random_state
  )
  def f2(x):
    return np.array([x[0] for _ in range(objective_dim)])
  mesh = Mesh(test_params, f2)
  mesh.initialize()

  # Set the velocity
  mesh.population.velocity = np.zeros((test_population_size, mesh.params.position_dim))
  mesh.population.velocity[one_idxs] = np.ones((population_size, mesh.params.position_dim))

  # Copy the particles
  mesh.pre_allocated.position_copy = mesh.population.position.copy()
  mesh.pre_allocated.velocity_copy = mesh.population.velocity.copy()
  mesh.pre_allocated.fitness_copy = mesh.population.fitness.copy()

  # Select only the particles with the fitness equals to zero
  mesh.elitism()

  # Check if the particles were selected correctly
  for i in range(test_population_size):
    assert np.array_equal(mesh.population.position[i, :decision_dim], np.zeros(decision_dim))
    assert np.array_equal(mesh.population.velocity[i], np.zeros(mesh.params.position_dim))
    assert np.array_equal(mesh.population.fitness[i], np.zeros(objective_dim))
    for j in range(max_personal_guides):
      assert np.array_equal(mesh.population.personal_guide_pos[i, j, :decision_dim], np.zeros(decision_dim))
      assert np.array_equal(mesh.population.personal_guide_fit[i, j], np.zeros(objective_dim))

def test_update_personal_guides():
  # Set some test parameters
  test_population_size = 3 * population_size
  test_max_personal_guides = 3
  # initial_positions = np.random.rand(test_population_size, position_dim)
  initial_points = np.array([[i % 3] * decision_dim for i in range(test_population_size)])
  # Initialize the algorithm with initial positions
  test_params = MeshParameters(
    objective_dim=objective_dim,
    decision_dim=decision_dim,
    decision_lower_bounds=lower_bound,
    decision_upper_bounds=upper_bound,
    population_size=test_population_size,
    memory_size=population_size,
    max_gen=max_gen,
    max_fit_eval=max_fit_eval,
    max_personal_guides=test_max_personal_guides,
    initial_points=initial_points,
    random_state=random_state
  )
  mesh = Mesh(test_params, toy_function)
  
  # Set fitness values to check the personal guide update
  mesh.population.fitness = np.array([[i % 3] * objective_dim for i in range(test_population_size)])
  personal_guide_fit_options = [np.full((test_max_personal_guides, objective_dim), -1), # Check when the particle is dominated by one of the personal guides
                                np.array([[2 * (i % 2)] * objective_dim for i in range(test_max_personal_guides)]), # Check when the current particle dominates some personal guides
                                np.full((test_max_personal_guides, objective_dim), 2)] # Check when there is no domination between current particle and the personal guides
  mesh.population.personal_guide_fit = np.array([personal_guide_fit_options[i % 3].copy() for i in range(test_population_size)])
  # Set personal guide positions randomly
  pb_positions = np.random.rand(test_population_size, test_max_personal_guides, mesh.params.position_dim)
  mesh.population.personal_guide_pos = pb_positions.copy()

  # Update the personal guides
  mesh.update_personal_guides()

  # Check if the personal guides were updated correctly
  for i in range(test_population_size):
    # The current particle is discarded
    if i % 3 == 0:
      for j in range(test_max_personal_guides):
        assert np.array_equal(mesh.population.personal_guide_pos[i, j, :], pb_positions[i, j, :])
    # Particles in odd positions are updated
    elif i % 3 == 1:
      for j in range(test_max_personal_guides):
        if j % 2 == 1:
          assert np.array_equal(mesh.population.personal_guide_pos[i, j, :], mesh.population.position[i, :])
        else:
          assert np.array_equal(mesh.population.personal_guide_pos[i, j, :], pb_positions[i, j, :])
    # The oldest particle is discarded
    else:
      assert np.array_equal(mesh.population.personal_guide_pos[i, 0, :], mesh.population.position[i, :])
      for j in range(1, test_max_personal_guides):
        assert np.array_equal(mesh.population.personal_guide_pos[i, j, :], pb_positions[i, j-1, :])

def test_stopping_by_generation():
  # Initialize the algoritm
  maximum_generations = np.random.randint(1, 10)
  test_params = MeshParameters(
    objective_dim=objective_dim,
    decision_dim=decision_dim,
    decision_lower_bounds=lower_bound,
    decision_upper_bounds=upper_bound,
    population_size=population_size,
    max_gen=maximum_generations
  )
  mesh = Mesh(test_params, toy_function)

  # Run the algorithm and check if the maximum generations was counted correctly 
  mesh.run()

  assert mesh.generation_counter == maximum_generations

def test_stopping_by_fitness_evalution():
  # Initialize the algoritm with fitness evaluations less or equal than the number of particles
  maximum_fitnes_evaluations = np.random.randint(1, population_size + 1)
  test_params = MeshParameters(
    objective_dim=objective_dim,
    decision_dim=decision_dim,
    decision_lower_bounds=lower_bound,
    decision_upper_bounds=upper_bound,
    population_size=population_size,
    max_fit_eval=maximum_fitnes_evaluations
  )
  mesh = Mesh(test_params, toy_function)

  # Run the algorithm and check if the maximum fitness evaluations was counted correctly
  mesh.run()

  assert mesh.fitness_eval_counter == maximum_fitnes_evaluations

  # Initialize the algorithm with fitness evaluations greater than the number of particles
  maximum_fitnes_evaluations = np.random.randint(3 * population_size + 1, 5 * population_size)
  test_params = MeshParameters(
    objective_dim=objective_dim,
    decision_dim=decision_dim,
    decision_lower_bounds=lower_bound,
    decision_upper_bounds=upper_bound,
    population_size=population_size,
    max_fit_eval=maximum_fitnes_evaluations
  )
  mesh = Mesh(test_params, toy_function)

  # Run the algorithm and check if the maximum fitness evaluations was counted correctly
  mesh.run()

  assert mesh.fitness_eval_counter == maximum_fitnes_evaluations