from mesh import Mesh
from mesh.operations import differential_crossover as dc
from mesh.parameters import MeshParameters

from unittest.mock import patch

import numpy as np

# ---------- Fixed parameters for test setup ----------
objective_dim = 5
decision_dim = 5
population_size = 20
lower_bound = np.array([0] * decision_dim)
upper_bound = np.array([1] * decision_dim)
mutation_rate = 0.5
communication_probability = 0.8
max_gen = None
max_fit_eval = 200
max_personal_guides = 3
random_state = None

def toy_function(x):
  return np.random.rand(objective_dim)

test_size = 10

def test_binomial_crossover():
  # Create a Mesh instance with a toy function
  test_params = MeshParameters(
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
  mesh = Mesh(test_params, toy_function)

  # Initialize the algorithm
  mesh.initialize()

  # Generate two random arrays for crossover
  X1 = np.random.rand(test_size, mesh.params.position_dim)
  X2 = np.random.rand(test_size, mesh.params.position_dim)

  # Mock the random functions to return predetermined values
  crossover_idxs = np.random.randint(0, mesh.params.position_dim, size=test_size)
  crossover_chances = np.random.uniform(0.0, 1.0, size=(test_size, mesh.params.position_dim))
  crossover_probability = np.random.rand(test_size, 1)
  with patch("numpy.random.randint", return_value=crossover_idxs),\
       patch("numpy.random.uniform", return_value=crossover_chances):

    # Apply the binomial crossover operation
    Xcross = dc.binomial_crossover(mesh, X1, X2, crossover_probability)

    # Check if the crossover was applied correctly
    for i, x in enumerate(Xcross):
      for j, c in enumerate(x):
        if (crossover_chances[i, j] <= crossover_probability[i]) or (j == crossover_idxs[i]):
          assert c == X2[i][j]
        else:
          assert c == X1[i][j]