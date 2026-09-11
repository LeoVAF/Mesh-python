from unittest.mock import patch

import numpy as np
import pytest

from amesh import AMESH
from amesh.auxiliar import StoppingAlgorithm
from amesh.parameters import AMESHParameters

# ---------- Fixed parameters for test setup ----------
objective_dim = 5
decision_dim = 5
population_size = 20
lower_bound = np.array([0] * decision_dim)
upper_bound = np.array([5] * decision_dim)
max_gen = None
max_fit_eval = 200
max_personal_guides = 3
random_state = None

params = AMESHParameters(
    objective_dim=objective_dim,
    decision_dim=decision_dim,
    decision_lower_bounds=lower_bound,
    decision_upper_bounds=upper_bound,
    population_size=population_size,
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
  test_params = AMESHParameters(
    objective_dim=objective_dim,
    decision_dim=decision_dim,
    decision_lower_bounds=lower_bound,
    decision_upper_bounds=upper_bound,
    population_size=population_size,
    max_gen=1,
    max_fit_eval=None,
    random_state=random_state
  )
  amesh = AMESH(test_params, toy_function)
  amesh.initialize()

  # Initialize the algortihm with less fitness evaluations than population size
  test_params = AMESHParameters(
    objective_dim=objective_dim,
    decision_dim=decision_dim,
    decision_lower_bounds=lower_bound,
    decision_upper_bounds=upper_bound,
    population_size=population_size,
    max_gen=None,
    max_fit_eval=population_size-1,
    random_state=random_state
  )
  amesh = AMESH(test_params, toy_function)
  with pytest.raises(StoppingAlgorithm, match=''):
    amesh.initialize()

  # Initialize the algortihm
  test_params = AMESHParameters(
    objective_dim=objective_dim,
    decision_dim=decision_dim,
    decision_lower_bounds=lower_bound,
    decision_upper_bounds=upper_bound,
    population_size=population_size,
    max_gen=1,
    max_fit_eval=population_size+1,
    random_state=random_state
  )
  amesh = AMESH(test_params, toy_function)
  amesh.initialize()



def test_sequential_fitness_evaluation():
  # Initialize the algortihm
  amesh = AMESH(params, toy_function)
  amesh.initialize()

  # Test the fitness evaluation
  positions = np.random.rand(population_size, decision_dim)
  fitnesses = amesh.sequential_fitness_evaluation(positions)
  for i, p in enumerate(positions):
    assert np.array_equal(toy_function(p), fitnesses[i])

def test_parallel_fitness_evaluation():
  # Initialize the algortihm
  amesh = AMESH(params, toy_function, num_proc=4)
  amesh.initialize()

  # Test the fitness evaluation
  positions = np.random.rand(population_size, amesh.params.decision_dim)
  fitnesses = amesh.parallel_fitness_evaluation(positions)
  for i, p in enumerate(positions):
    assert np.array_equal(toy_function(p), fitnesses[i])

def test_differential_evolution():
  test_population_size = 2 * population_size
  # Create an AMESH instance with a rank function
  steps = np.linspace(0, 1, test_population_size)
  ranks = [0, 4]
  initial_points = np.hstack((np.array([[ranks[i % len(ranks)]] for i in range(test_population_size)]),
                                 np.array([[steps[i]] for i in range(test_population_size)]),
                                 np.random.rand(test_population_size, decision_dim-2)))
  test_params = AMESHParameters(
    objective_dim=objective_dim,
    decision_dim=decision_dim,
    decision_lower_bounds=lower_bound,
    decision_upper_bounds=upper_bound,
    population_size=test_population_size,
    max_gen=max_gen,
    max_fit_eval=max_fit_eval,
    max_personal_guides=max_personal_guides,
    initial_points=initial_points,
    random_state=random_state
  )
  amesh = AMESH(test_params, rank_function)
  amesh.initialize()

  # Set the Xst and pop_idxs
  Xst = np.hstack((np.array([[0] for _ in range(population_size)]),
                   np.array([[steps[i]] for i in range(population_size)]),
                   np.random.rand(population_size, amesh.params.decision_dim-2)))
  pop_idxs = np.array([i for i in range(population_size)])
  with patch.object(amesh, 'differential_mutation', return_value=(Xst, pop_idxs)), patch.object(amesh, 'differential_crossover', return_value=Xst):
    # Run the Differential Evolution phase
    amesh.differential_evolution()
    # Check if the strategy particles are in the population
    st_idxs = np.arange(1, test_population_size, 2)
    for i, idx in enumerate(st_idxs):
      assert np.array_equal(amesh.population.position[idx], Xst[i])

def test_mutation():
  # Initialize the algortihm
  amesh = AMESH(params, toy_function)
  amesh.initialize()

  mutation_rate = amesh.params.SWARM_mutation_scale

  # Mock the random function to return predetermined values
  global_guide_noise = np.random.normal(0.0, 1.0, size=(amesh.params.population_size, amesh.params.decision_dim))
  with patch('numpy.random.normal', return_value=global_guide_noise):

    # Find the global guides
    amesh.global_guide_method()

    # Mutate the variables
    amesh.mutation()

    # Check if the mutation operation was applied correctly
    bound_scale = amesh.params.decision_upper_bounds - amesh.params.decision_lower_bounds
    for i, gb_mut in enumerate(amesh.pre_allocated.global_guide_mutated):
      gb_expected = np.clip(amesh.population.global_guide[i] + mutation_rate[i] * global_guide_noise[i] * bound_scale,
                            amesh.params.decision_lower_bounds,
                            amesh.params.decision_upper_bounds)
      assert np.linalg.norm(gb_mut - gb_expected) < equal_tolerance_for_array

def test_move_population():
  amesh = AMESH(params, toy_function)
  amesh.initialize()

  amesh.population.personal_guide_pos = np.random.uniform(
      amesh.params.decision_lower_bounds,
      amesh.params.decision_upper_bounds,
      size=(
          amesh.params.population_size,
          amesh.params.max_personal_guides,
          amesh.params.decision_dim,
      ),
  )
  
  amesh.global_guide_method()
  amesh.mutation()
  
  pop_size = amesh.params.population_size
  dim = amesh.params.decision_dim
  # Mock of the personal guide index
  pb_indices = np.random.randint(0, amesh.params.max_personal_guides, size=pop_size,)
  # Mock of the SHADE memory index used for W and Pcom
  random_index = np.random.randint(0, amesh.params.hyperparameter_memory_length, size=pop_size)
  # Deterministic E weights: shape (population size, 3)
  W_mock = np.random.uniform(0.0, 1.0, size=(pop_size, 3))
  # Probability of communication per particle: shape (population_size,)
  Pcom_mock = np.random.uniform(0.0, 1.0, size=pop_size)
  # Random values ​​used to form C: shape (population_size, decision_dim)
  communication_probs = np.random.rand(pop_size, dim)
  def mock_sample_from_cauchy(out, loc, scale, bounds):
      out[:] = W_mock
  with (
      patch("numpy.random.randint", side_effect=[pb_indices, random_index]),
      patch.object(amesh, "sample_from_cauchy", side_effect=mock_sample_from_cauchy),
      patch("numpy.random.normal", return_value=Pcom_mock.copy()),
      patch("numpy.random.rand", return_value=communication_probs),
  ):
      # Original state before the move
      amesh.pre_allocated.position_copy[:] = amesh.population.position.copy()
      amesh.pre_allocated.velocity_copy[:] = amesh.population.velocity.copy()
      amesh.pre_allocated.fitness_copy[:] = amesh.population.fitness.copy()

      X0 = amesh.pre_allocated.position_copy.copy()
      V0 = amesh.pre_allocated.velocity_copy.copy()
      Xgb_mut = amesh.pre_allocated.global_guide_mutated.copy()
      # Expected by the same rule as the actual function.
      W = W_mock
      Pcom = np.clip(Pcom_mock, 0.0, 1.0)
      C = communication_probs <= Pcom[:, np.newaxis]
      # Move the population and then test if the particles were correctly moved
      amesh.move_population()
      for i in range(pop_size):
          x = X0[i]
          v_old = V0[i]

          x_pb = amesh.population.personal_guide_pos[i, pb_indices[i], :]
          x_gb_mut = Xgb_mut[i]

          v_expected = W[i, 0] * v_old + W[i, 1] * (x_pb - x) + W[i, 2] * C[i] * (x_gb_mut - x)
          np.clip(v_expected, amesh.params.velocity_lower_bounds, amesh.params.velocity_upper_bounds, out=v_expected)
          x_expected = np.clip(x + v_expected, amesh.params.decision_lower_bounds, amesh.params.decision_upper_bounds)

          np.testing.assert_allclose(amesh.pre_allocated.velocity_copy[i], v_expected, atol=equal_tolerance_for_array)
          np.testing.assert_allclose(amesh.pre_allocated.position_copy[i], x_expected, atol=equal_tolerance_for_array)
          np.testing.assert_allclose(amesh.pre_allocated.fitness_copy[i], amesh.fitness_function(x_expected), atol=equal_tolerance_for_array)

def test_elitism():
  test_population_size = 2 * population_size
  # Initialize the algorithm with initial positions
  initial_points = np.array([[i % 2] * decision_dim for i in range(test_population_size)])
  test_params = AMESHParameters(
    objective_dim=objective_dim,
    decision_dim=decision_dim,
    decision_lower_bounds=lower_bound,
    decision_upper_bounds=upper_bound,
    population_size=test_population_size,
    max_gen=max_gen,
    max_fit_eval=max_fit_eval,
    max_personal_guides=max_personal_guides,
    initial_points=initial_points,
    random_state=random_state
  )
  def f1(x):
    return np.array([x[0] for _ in range(objective_dim)])
  amesh = AMESH(test_params, f1)
  amesh.initialize()

  # Set the velocity
  amesh.population.velocity = np.array([[i % 2] * amesh.params.decision_dim for i in range(test_population_size)])

  # Copy the particles
  amesh.pre_allocated.position_copy = amesh.population.position.copy()
  amesh.pre_allocated.velocity_copy = amesh.population.velocity.copy()
  amesh.pre_allocated.fitness_copy = amesh.population.fitness.copy()

  # Select only the particles with the fitness equals to zero
  amesh.elitism()

  # Check if the particles were selected correctly
  for i in range(test_population_size):
    assert np.array_equal(amesh.population.position[i, :decision_dim], np.zeros(decision_dim))
    assert np.array_equal(amesh.population.velocity[i], np.zeros(amesh.params.decision_dim))
    assert np.array_equal(amesh.population.fitness[i], np.zeros(objective_dim))
    for j in range(max_personal_guides):
      assert np.array_equal(amesh.population.personal_guide_pos[i, j, :decision_dim], np.zeros(decision_dim))
      assert np.array_equal(amesh.population.personal_guide_fit[i, j], np.zeros(objective_dim))
  
  # Initialize the algorithm with initial positions with one arrays in random indices
  one_idxs = np.random.choice(test_population_size, size=population_size, replace=False)
  initial_points = np.zeros((test_population_size, decision_dim))
  initial_points[one_idxs] = np.ones((population_size, decision_dim))
  test_params = AMESHParameters(
    objective_dim=objective_dim,
    decision_dim=decision_dim,
    decision_lower_bounds=lower_bound,
    decision_upper_bounds=upper_bound,
    population_size=test_population_size,
    max_gen=max_gen,
    max_fit_eval=max_fit_eval,
    max_personal_guides=max_personal_guides,
    initial_points=initial_points,
    random_state=random_state
  )
  def f2(x):
    return np.array([x[0] for _ in range(objective_dim)])
  amesh = AMESH(test_params, f2)
  amesh.initialize()

  # Set the velocity
  amesh.population.velocity = np.zeros((test_population_size, amesh.params.decision_dim))
  amesh.population.velocity[one_idxs] = np.ones((population_size, amesh.params.decision_dim))

  # Copy the particles
  amesh.pre_allocated.position_copy = amesh.population.position.copy()
  amesh.pre_allocated.velocity_copy = amesh.population.velocity.copy()
  amesh.pre_allocated.fitness_copy = amesh.population.fitness.copy()

  # Select only the particles with the fitness equals to zero
  amesh.elitism()

  # Check if the particles were selected correctly
  for i in range(test_population_size):
    assert np.array_equal(amesh.population.position[i, :decision_dim], np.zeros(decision_dim))
    assert np.array_equal(amesh.population.velocity[i], np.zeros(amesh.params.decision_dim))
    assert np.array_equal(amesh.population.fitness[i], np.zeros(objective_dim))
    for j in range(max_personal_guides):
      assert np.array_equal(amesh.population.personal_guide_pos[i, j, :decision_dim], np.zeros(decision_dim))
      assert np.array_equal(amesh.population.personal_guide_fit[i, j], np.zeros(objective_dim))

def test_update_personal_guides():
  # Set some test parameters
  test_population_size = 3 * population_size
  test_max_personal_guides = 3
  initial_points = np.array([[i % 3] * decision_dim for i in range(test_population_size)])
  # Initialize the algorithm with initial positions
  test_params = AMESHParameters(
    objective_dim=objective_dim,
    decision_dim=decision_dim,
    decision_lower_bounds=lower_bound,
    decision_upper_bounds=upper_bound,
    population_size=test_population_size,
    max_gen=max_gen,
    max_fit_eval=max_fit_eval,
    max_personal_guides=test_max_personal_guides,
    initial_points=initial_points,
    random_state=random_state
  )
  amesh = AMESH(test_params, toy_function)
  
  # Set fitness values to check the personal guide update
  amesh.population.fitness = np.full((test_population_size, objective_dim), 1)
  personal_guide_fit_options = [
    # Check when the particle is dominated by one of the personal guides
    np.full((test_max_personal_guides, objective_dim), 0),
    # Check when the current particle dominates some personal guides
    np.array([[(2 - (i % 2))] * objective_dim for i in range(test_max_personal_guides)]),
    # Check when there is no domination between current particle and the personal guides
    np.full((test_max_personal_guides, objective_dim), 1)
  ]
  amesh.population.personal_guide_fit = np.array([personal_guide_fit_options[i % 3].copy() for i in range(test_population_size)])
  # Set personal guide positions randomly
  pb_positions = np.random.rand(test_population_size, test_max_personal_guides, amesh.params.decision_dim)
  amesh.population.personal_guide_pos = pb_positions.copy()

  # Update the personal guides
  amesh.update_personal_guides()

  # Check if the personal guides were updated correctly
  for i in range(test_population_size):
    # The current particle is discarded
    if i % 3 == 0:
      for j in range(test_max_personal_guides):
        assert np.array_equal(amesh.population.personal_guide_pos[i, j, :], pb_positions[i, j, :])
    # Particles in odd positions are updated
    elif i % 3 == 1:
      assert np.array_equal(amesh.population.personal_guide_pos[i, 0, :], amesh.population.position[i, :])
      for j in range(1, test_max_personal_guides):
        if j % 2 == 1:
          assert np.array_equal(amesh.population.personal_guide_pos[i, j, :], amesh.population.position[i, :])
        else:
          assert np.array_equal(amesh.population.personal_guide_pos[i, j, :], pb_positions[i, j-1, :])
    # The rightmost position is discarded
    else:
      assert np.array_equal(amesh.population.personal_guide_pos[i, 0, :], amesh.population.position[i, :])
      for j in range(1, test_max_personal_guides):
        assert np.array_equal(amesh.population.personal_guide_pos[i, j, :], pb_positions[i, j-1, :])

def test_stopping_by_generation():
  # Initialize the algoritm
  maximum_generations = np.random.randint(1, 10)
  test_params = AMESHParameters(
    objective_dim=objective_dim,
    decision_dim=decision_dim,
    decision_lower_bounds=lower_bound,
    decision_upper_bounds=upper_bound,
    population_size=population_size,
    max_gen=maximum_generations
  )
  amesh = AMESH(test_params, toy_function)

  # Run the algorithm and check if the maximum generations was counted correctly 
  amesh.run()

  assert amesh.generation_counter == maximum_generations

def test_stopping_by_fitness_evalution():
  # Initialize the algoritm with fitness evaluations less or equal than the number of particles
  maximum_fitnes_evaluations = np.random.randint(1, population_size + 1)
  test_params = AMESHParameters(
    objective_dim=objective_dim,
    decision_dim=decision_dim,
    decision_lower_bounds=lower_bound,
    decision_upper_bounds=upper_bound,
    population_size=population_size,
    max_fit_eval=maximum_fitnes_evaluations
  )
  amesh = AMESH(test_params, toy_function)

  # Run the algorithm and check if the maximum fitness evaluations was counted correctly
  amesh.run()

  assert amesh.fitness_eval_counter == maximum_fitnes_evaluations

  # Initialize the algorithm with fitness evaluations greater than the number of particles
  maximum_fitnes_evaluations = np.random.randint(3 * population_size + 1, 5 * population_size)
  test_params = AMESHParameters(
    objective_dim=objective_dim,
    decision_dim=decision_dim,
    decision_lower_bounds=lower_bound,
    decision_upper_bounds=upper_bound,
    population_size=population_size,
    max_fit_eval=maximum_fitnes_evaluations
  )
  amesh = AMESH(test_params, toy_function)

  # Run the algorithm and check if the maximum fitness evaluations was counted correctly
  amesh.run()

  assert amesh.fitness_eval_counter == maximum_fitnes_evaluations