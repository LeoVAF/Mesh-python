from mesh.parameters import MeshParameters
from mesh.utils.particles import Memory, Population

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

def test_Population():
  # Create a Population instance with initial positions
  initial_positions = np.random.rand(population_size, position_dim)
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
  mesh_population = Population(params)
  
  # Check if the class has the correct attributes
  assert hasattr(mesh_population, 'position')
  assert hasattr(mesh_population, 'velocity')
  assert hasattr(mesh_population, 'fitness')
  assert hasattr(mesh_population, 'rank')
  assert hasattr(mesh_population, 'sigma')
  assert hasattr(mesh_population, 'global_guide')
  assert hasattr(mesh_population, 'personal_guide_pos')
  assert hasattr(mesh_population, 'personal_guide_fit')

  # Check if the initial positions was initialized correctly
  assert np.all(initial_positions == mesh_population.position)

  # Check if the positions and velocities were initialized correctly
  for p in mesh_population.position:
    assert np.all(p <= params.position_upper_bounds)
    assert np.all(p >= params.position_lower_bounds)
  for v in mesh_population.velocity:
    assert np.all(v <= params.velocity_upper_bounds)
    assert np.all(v >= params.velocity_lower_bounds)

def test_Memory():
  # Create a Memory instance
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
    random_state=random_state
  )
  mesh_memory = Memory(params)

  # Check if the class has the correct attributes
  assert hasattr(mesh_memory, 'position')
  assert hasattr(mesh_memory, 'fitness')
  assert hasattr(mesh_memory, 'sigma')