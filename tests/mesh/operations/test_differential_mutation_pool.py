from mesh.operations import differential_mutation_pool as dmp
from mesh.core import Mesh
from mesh.parameters import MeshParameters

from pymoo.problems import get_problem
from unittest import TestCase, main

import numpy as np

class TestDifferentialMutationPool(TestCase):
  def test_pool_from_population():
    # Initialize a random Mesh instance
    params = MeshParameters(objective_dim=2,
                            position_dim=5, position_min_value=np.array([0]*5), position_max_value=np.array([1]*5), 
                            population_size=20, memory_size=None,
                            global_best_attribution_type=0,
                            dm_pool_type=0,
                            dm_operation_type=0,
                            communication_probability=0.5, mutation_rate=0.5,
                            max_gen=0, max_fit_eval=200,
                            max_personal_guides=3,
                            random_state=None)
    func = get_problem('zdt1', n_var=5).evaluate
    mesh = Mesh(params, func, log_memory=False)
    # Initialize the algorithm
    mesh.initialize()

    # Get the pool list
    pool_list = dmp.pool_from_population(mesh)

    # Check if each position in in the respective particle pool is different by the particle position
    for i, pool in enumerate(pool_list):
      assert all([np.all(mesh.population.position[i] != pos) for pos in pool])

  def test_pool_from_memory():
    # Initialize a random Mesh instance
    params = MeshParameters(objective_dim=2,
                            position_dim=5, position_min_value=np.array([0]*5), position_max_value=np.array([1]*5), 
                            population_size=20, memory_size=None,
                            global_best_attribution_type=0,
                            dm_pool_type=0,
                            dm_operation_type=0,
                            communication_probability=0.5, mutation_rate=0.5,
                            max_gen=0, max_fit_eval=200,
                            max_personal_guides=3,
                            random_state=None)
    func = get_problem('zdt1', n_var=5).evaluate
    mesh = Mesh(params, func, log_memory=False)
    # Initialize the algorithm
    mesh.initialize()

    # Get the pool list
    pool_list = dmp.pool_from_memory(mesh)

    # Check if each position in in the respective particle pool is different by the particle position
    for i, pool in enumerate(pool_list):
      assert all([np.all(mesh.population.position[i] != pos) for pos in pool])

  def test_pool_from_population_and_memory():
    # Initialize a random Mesh instance
    params = MeshParameters(objective_dim=2,
                            position_dim=5, position_min_value=np.array([0]*5), position_max_value=np.array([1]*5), 
                            population_size=20, memory_size=None,
                            global_best_attribution_type=0,
                            dm_pool_type=0,
                            dm_operation_type=0,
                            communication_probability=0.5, mutation_rate=0.5,
                            max_gen=0, max_fit_eval=200,
                            max_personal_guides=3,
                            random_state=None)
    func = get_problem('zdt1', n_var=5).evaluate
    mesh = Mesh(params, func, log_memory=False)
    # Initialize the algorithm
    mesh.initialize()

    # Get the pool list
    pool_list = dmp.pool_from_population_and_memory(mesh)

    # Check if each position in in the respective particle pool is different by the particle position
    for i, pool in enumerate(pool_list):
      assert all([np.all(mesh.population.position[i] != pos) for pos in pool])


if __name__ == '__main__':
  main()