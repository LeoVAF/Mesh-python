from mesh.core import Mesh
from mesh.parameters import MeshParameters
from mesh.operations import differential_crossover as dc

from scipy.stats import truncnorm

import numpy as np
import pytest

# ---------- Fixed parameters for test setup ----------
population_size = 20
objective_dim = 2
position_dim = 5
lower_bound = np.array([0] * position_dim)
upper_bound = np.array([1] * position_dim)
test_size = 10

params = MeshParameters(
    objective_dim=objective_dim,
    position_dim=position_dim,
    lower_bound_array=lower_bound,
    upper_bound_array=upper_bound,
    population_size=population_size,
    memory_size=population_size,
    mutation_rate=0.5,
    communication_probability= 0.5,
    max_gen=None,
    max_fit_eval=200,
    max_personal_guides=3,
    random_state=None
)

def test_binomial_crossover(mocker):
  # Mock the random functions to return predetermined values
  crossover_rates = truncnorm.rvs(0, 1, size=(test_size, 1))
  mocker.patch("scipy.stats.truncnorm.rvs", return_value=crossover_rates)

  crossover_idxs = np.random.randint(0, params.position_dim, size=test_size)
  mocker.patch("numpy.random.randint", return_value=crossover_idxs)

  crossover_chances = np.random.uniform(0.0, 1.0, size=(test_size, params.position_dim))
  mocker.patch("numpy.random.uniform", return_value=crossover_chances)

  mesh = Mesh(params, lambda x: (np.sum(x), np.prod(x)))

  X1 = np.random.rand(test_size, position_dim)
  X2 = np.random.rand(test_size, position_dim)

  X = dc.binomial_crossover(mesh, X1, X2)

  for i, x in enumerate(X):
    for j, c in enumerate(x):
      if (crossover_chances[i, j] <= crossover_rates[i]) or (j == crossover_idxs[i]):
        assert c == X2[i][j]
      else:
        assert c == X1[i][j]