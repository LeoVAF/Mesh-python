from mesh.parameters import MeshParameters
from mesh.utils import auxiliar as aux

import numpy as np
import pytest

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

def test_PreAllocated_success():
  # Create a PreAllocated instance
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
  pre_allocated_instance = aux.PreAllocated(test_params)

  # Check if the class has the correct attributes
  assert hasattr(pre_allocated_instance, 'np_tril_indices')
  assert hasattr(pre_allocated_instance, 'global_guide_mutated')
  assert hasattr(pre_allocated_instance, 'matrix_for_operations')
  assert hasattr(pre_allocated_instance, 'vector_for_operations')
  assert hasattr(pre_allocated_instance, 'fitness_elitism')
  assert hasattr(pre_allocated_instance, 'position_copy')
  assert hasattr(pre_allocated_instance, 'velocity_copy')
  assert hasattr(pre_allocated_instance, 'fitness_copy')

def test_PreAllocated_failure():
  # Create a PreAllocated instance with a non MeshParameters instance
  test_params = object()
  with pytest.raises(TypeError, match=r'The input "params" has type <class \'object\'>, but expected <class \'mesh.parameters.MeshParameters\'>.'):
    aux.PreAllocated(test_params) # type: ignore