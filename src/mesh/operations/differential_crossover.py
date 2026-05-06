from __future__ import annotations

from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from mesh import Mesh

def binomial_crossover(self: Mesh,
                       X1: NDArray[np.number],
                       X2: NDArray[np.number],
                       crossover_probability: NDArray[np.number]) -> NDArray[np.number]:
  r''' Apply the Binomial Crossover in ``X1`` in-place from information in ``X2`` according to:

  .. math::
      X_{1{[i,\ j]}} = \begin{cases}
                          X_{1{[i,\ j]}}, & \text{if } (r_i \leq p_{cross}) \lor (j = j_{rand}), \text{ with } r_i \sim \mathcal{U}(0,\ 1); \\
                          X_{2{[i,\ j]}}, & \text{otherwise};
                       \end{cases}

  where :math:`p_{cross}` is the crossover probability and :math:`j_{rand}` \in \{1,\ \ldots,\ m\} is a random index sampled under a Uniform Distribution.
  
  Note:
    The crossover probability is calculated as a decision variable.
  
  Args:
    self (:class:`~mesh.core.Mesh`): An instance of :class:`~mesh.core.Mesh`.
    X1 (:type:`NDArray[np.number]`): The numpy matrix to apply the crossover.
    X2 (:type:`NDArray[np.number]`): The second numpy matrix that will share information in the crossover.
    crossover_probability (:type:`NDArray[np.number]`): The crossover probability for each point.
    
    
  Returns:
    :type:`NDArray[np.number]`: ``X1`` after applying the Binomial Crossover.
  '''

  # Get the size of the X1 to apply the crossover
  size = X1.shape[0]
  # Make the crossover index for each particle
  crossover_index = np.random.randint(0, self.params.position_dim, size=size)
  # Calculate the crossover chance to apply the Binomial Crossover
  crossover_chance = np.random.uniform(0.0, 1.0, size=(size, self.params.position_dim))
  # Get the crossover mask
  crossover_mask = crossover_chance <= crossover_probability
  crossover_mask[np.arange(size), crossover_index] = True
  # Apply the crossover
  X1[crossover_mask] = X2[crossover_mask]
  return X1

# The options of Differential Crossover operation
differential_crossover_options: dict[str, Callable[[Mesh, NDArray[np.number], NDArray[np.number], NDArray[np.number]], NDArray[np.number]]] = {
  'binomial': binomial_crossover
}
''' The options of Differential Mutation operation. They are:

  - :type:`binomial`: Applies the Binomial Crossover from Differential Evolution.
'''

def get_differential_crossover(option: str) -> Callable[[Mesh, NDArray[np.number], NDArray[np.number], NDArray[np.number]], NDArray[np.number]]:
  ''' Sets the Differential Crossover from Differential Evolution according to :attr:`~mesh.operations.differential_crossover.differential_crossover_options`. 
  
  Args:
    option (:type:`str`): Differential Crossover option.

  Returns:
    :type:`Callable[[`~mesh.core.Mesh`, NDArray[np.number], NDArray[np.number], NDArray[np.number]], NDArray[np.number]]`: The Differential Crossover function.
  '''

  return differential_crossover_options[option]