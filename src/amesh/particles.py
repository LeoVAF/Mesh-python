from math import comb

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

from .parameters import AMESHParameters
from .validations.python_validations import assert_type


class Population:
    """Represents the A-MESH population.

    Args:
        params (:class:`~amesh.parameters.AMESHParameters`): Parameters defining the objective and decision dimensions, decision and velocity bounds, population size, guide strategy, maximum number of personal guides, and optional initial positions.
    
    Raises:
        TypeError: If the input is not an instance of :class:`~amesh.parameters.AMESHParameters`.
    """

    def __init__(self, params: AMESHParameters):
        assert_type(params, 'params', AMESHParameters)

        self.position: NDArray[np.number]
        ''' Numpy matrix with the particle's positions initialized randomly under Uniform Distribution. '''
        self.velocity: NDArray[np.number]
        ''' Numpy matrix with the particle's velocities initialized randomly under Uniform Distribution. '''
        self.fitness: NDArray[np.number]
        ''' Numpy matrix with the particle's fitnesses initialized with ``np.inf`` values. '''
        self.sigma: NDArray[np.number]
        ''' Numpy matrix for the sigma values. Initialized with ``np.inf`` values. Used only if the Sigma method is used. '''
        self.global_guide: NDArray[np.number]
        ''' Numpy matrix with the global guide position for each particle. '''
        self.personal_guide_pos: NDArray[np.number]
        ''' 3-dimensional numpy array with a matrix of personal guide positions for each particle. Each matrix has :attr:`~amesh.parameters.AMESHParameters.max_personal_guides` positions. Initialized with the respective particle's position repeated for all matrix entries. '''
        self.personal_guide_fit: NDArray[np.number]
        ''' 3-dimensional numpy array with a matrix of personal guide fitnesses for each particle. Each matrix has :attr:`~amesh.parameters.AMESHParameters.max_personal_guides` fitnesses. '''

        if params.initial_points is None:
            sampler = qmc.LatinHypercube(d=params.decision_dim, scramble=True)
            sample = sampler.random(n=params.population_size)
            self.position = qmc.scale(sample, params.decision_lower_bounds, params.decision_upper_bounds)
        else:
            self.position = params.initial_points
        self.velocity = np.random.uniform(params.velocity_lower_bounds, params.velocity_upper_bounds, (params.population_size, params.decision_dim))
        self.fitness = np.full((params.population_size, params.objective_dim), np.inf)
        if params.global_guide_method in {0, 1}:
            self.sigma = np.full((params.population_size, comb(params.objective_dim, 2)), np.nan)
        else:
            self.sigma = np.empty((0, comb(params.objective_dim, 2)))
        self.global_guide = np.full((params.population_size, params.decision_dim), np.nan)
        self.personal_guide_pos = np.repeat(self.position[:, np.newaxis, :], params.max_personal_guides, axis=1)
        self.personal_guide_fit = np.full((params.population_size, params.max_personal_guides, params.objective_dim), np.inf)

class Memory:
    """Represents the A-MESH external memory.

    Args:
        params (:class:`~amesh.parameters.AMESHParameters`): Parameters that define the objective and decision dimensions of the empty memory arrays.

    Raises:
        TypeError: If the input is not of the expected type.
    """
    
    def __init__(self, params: AMESHParameters) -> None:
        assert_type(params, 'params', AMESHParameters)

        # Set the class attributes
        self.position: NDArray[np.number] = np.empty((0, params.decision_dim))
        """ Numpy matrix with the memory position. """
        self.fitness: NDArray[np.number] = np.empty((0, params.objective_dim))
        """ Numpy matrix with the memory fitness. """
        self.sigma: NDArray[np.number] = np.empty((0, 0))
        """ Numpy matrix with the memory sigma values. This attribute is only used when the Sigma method is used. """
