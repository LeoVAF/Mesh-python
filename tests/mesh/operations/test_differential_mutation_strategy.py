from mesh.operations import differential_mutation_strategy as dms
from mesh.core import Mesh
from mesh.parameters import MeshParameters

from pymoo.problems import get_problem
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np

class TestDifferentialMutationStrategy(TestCase):
  ######################## Constants to initialize the tests ########################
  test_params = MeshParameters(objective_dim=2,
                          position_dim=5, position_min_value=np.array([0]*5), position_max_value=np.array([1]*5), 
                          population_size=20, memory_size=None,
                          global_best_attribution_type=0,
                          dm_pool_type=0,
                          dm_operation_type=0,
                          communication_probability=0.5, mutation_rate=0.5,
                          max_gen=0, max_fit_eval=200,
                          max_personal_guides=3,
                          random_state=None)
  test_idx_size = 10
  ###################################################################################

  @patch('numpy.random.randint', return_value=np.random.randint(0, test_params.position_dim, size=test_idx_size))
  @patch('numpy.random.uniform', return_value=np.random.uniform(0.0, 1.0, size=(test_idx_size, test_params.position_dim)))
  def test_binomial_mutation_mask(self, mock_uniform, mock_randint):
    dms.binomial_mutation_mask(self.test_params, self.test_idx_size)
    
    # Check the call numbers
    self.assertEqual(mock_randint.call_count, 1)
    self.assertEqual(mock_uniform.call_count, 1)
    # Check the parameters of the calls
    mock_randint.assert_any_call(0, self.test_params.position_dim, size=self.test_idx_size)
    mock_uniform.assert_any_call(0., 1., size=(self.test_idx_size, self.test_params.position_dim))

  def test_rand_1_bin(self):
    pass    

if __name__ == '__main__':
  main()