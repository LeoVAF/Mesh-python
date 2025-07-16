from __future__ import annotations

from scipy.stats import truncnorm
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from mesh.parameters import MeshParameters

def binomial_crossover(X1: np.ndarray[np.any, 2], X2: np.ndarray[np.any, 2], params: MeshParameters) -> np.ndarray[np.bool, 2]:
  ''' Apply the Binomial Crossover in ``X1`` in-place from information in ``X2``.

  Note:
    The crossover is applied in-place in ``X1``. The crossover probability is calculated by a truncated normal between 0 and 1 distribution with mean 0 and standard deviation 1, and then multiplied by :attr:`~mesh.parameters.MeshParameters.mutation_rate`.
  
  Args:
    X1 (:type:`np.ndarray[np.any, 2]`): The numpy matrix to apply the crossover.
    X2 (:type:`np.ndarray[np.any, 2]`): The second numpy matrix that will share information in the crossover.
    params (:class:`~mesh.parameters.MeshParameters`): The parameters :attr:`~mesh.parameters.MeshParameters.position_dim` and :attr:`~mesh.parameters.MeshParameters.mutation_rate` are used to apply the crossover.
    
  Returns:
    :type:`np.ndarray[np.any, 2]`: ``X1`` after applying the Binomial Crossover.
  '''

  # Get the size of the X1 to apply the crossover
  size = X1.shape[0]
  # Get the crossover rate
  crossover_rate = truncnorm.rvs(0, 1, size=(size, 1)) * params.mutation_rate
  # Make the crossover index for each particle
  crossover_index = np.random.randint(0, params.position_dim, size=size)
  # Calculate the crossover chance to apply the Binomial Crossover
  crossover_chance = np.random.uniform(0.0, 1.0, size=(size, params.position_dim))
  # Get the crossover mask
  crossover_mask = crossover_chance < crossover_rate
  crossover_mask[np.arange(size), crossover_index] = True
  # Apply the crossover
  X1[crossover_mask] = X2[crossover_mask]
  return X1

# The options of Differential Crossover operation
differential_crossover_options = {
  'binomial': binomial_crossover
}
''' The options of Differential Mutation operation. They are:

  - :type:`binomial`: Applies the Binomial Crossover from Differential Evolution.
'''

def get_differential_crossover(option: {0}) -> Callable[[np.ndarray[np.float64, 2], np.ndarray[np.float64, 2], MeshParameters], np.ndarray[np.float64, 2]]:
  ''' Sets the Differential Crossover from Differential Evolution according to :attr:`~mesh.operations.differential_crossover.differential_crossover_options`. 
  
  Args:
    option (:type:`{0, 1, 2, 3, 4}`): Defines the Differential Crossover option.

  Returns:
    :type:`Callable[[np.ndarray[np.float64, 2], np.ndarray[np.float64, 2], MeshParameters], np.ndarray[np.float64, 2]]`: The Differential Crossover function.
  '''

  return differential_crossover_options[option]