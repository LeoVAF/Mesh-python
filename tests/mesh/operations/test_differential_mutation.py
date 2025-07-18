from mesh.core import Mesh
from mesh.parameters import MeshParameters
from mesh.operations import differential_mutation as dm

from scipy.stats import truncnorm

import numpy as np

# ---------- Fixed parameters for test setup ----------
population_size = 20
objective_dim = 2
position_dim = 5
lower_bound = np.array([0] * position_dim)
upper_bound = np.array([1] * position_dim)

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

def test_rand_1(mocker):
  # Create a list of arrays to sample from
  Xr_pool_list = [np.random.rand(np.random.choice(population_size) + 1, position_dim) for _ in range(population_size)]

  # Get the valid indices where the arrays have at least 3 elements
  valid_size = 3
  valid_idxs = [i for i in range(len(Xr_pool_list)) if len(Xr_pool_list[i]) >= valid_size]
  
  # Mock the random functions to return predetermined values
  operation_weight = truncnorm.rvs(0, 2, size=(len(valid_idxs), 1))
  mocker.patch("scipy.stats.truncnorm.rvs", return_value=operation_weight)

  # The name sample is copied directly into the module's local namespace. The patch in "random.sample" does not override this.
  mocker.patch("mesh.operations.differential_mutation.sample", return_value=np.array([0,1,2]))

  # Create a Mesh instance with a dummy function
  mesh = Mesh(params, lambda x: np.array([np.sum(x), np.prod(x)]))

  # Call the rand_1 function
  Xst, idxs = dm.rand_1(mesh, Xr_pool_list)

  # Check the output
  for i, idx in enumerate(idxs):
    Xr = Xr_pool_list[idx]
    x0 = Xr[0]
    x1 = Xr[1]
    x2 = Xr[2]
    xst = np.clip(x0 + operation_weight[i] * (x1 - x2), params.lower_bound_array, params.upper_bound_array)
    assert np.array_equal(Xst[i], xst)

  # Testing the case that the pool has less than minimum number of elements
  Xr_pool_list = [np.random.rand(np.random.choice(valid_size), position_dim) for _ in range(population_size)]
  Xst, idxs = dm.rand_1(mesh, [np.random.rand(2, position_dim)])
  assert np.array_equal(Xst, np.array([])) and np.array_equal(idxs, np.array([]))
