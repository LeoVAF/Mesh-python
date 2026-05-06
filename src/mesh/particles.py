from .parameters import MeshParameters
from .validations.python_validations import assert_type

from math import comb
from numpy.typing import NDArray
from scipy.stats import qmc

import numpy as np

class Population:
    """ Represents the MESH population.

    Args:
        params (:class:`~mesh.parameters.MeshParameters`): The attributes :attr:`~mesh.parameters.MeshParameters.objective_dim`, :attr:`~mesh.parameters.MeshParameters.position_dim`, :attr:`~mesh.parameters.MeshParameters.lower_bound_array`, :attr:`~mesh.parameters.MeshParameters.upper_bound_array`, :attr:`~mesh.parameters.MeshParameters.velocity_min_value`, :attr:`~mesh.parameters.MeshParameters.velocity_max_value`, :attr:`~mesh.parameters.MeshParameters.population_size`, :attr:`~mesh.parameters.MeshParameters.global_guide_method` and :attr:`~mesh.parameters.MeshParameters.max_personal_guides` are used to initialize the population.
    
    Raises:
        TypeError: If the input is not an instance of :class:`~mesh.parameters.MeshParameters`.
    """

    def __init__(self, params: MeshParameters):
        assert_type(params, 'params', MeshParameters)

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
        ''' 3-dimensional numpy array with a matrix of personal guide positions for each particle. Each matrix has :attr:`~mesh.parameters.MeshParameters.max_personal_guides` positions. Initialized with the respective particle's position repeated for all matrix entries. '''
        self.personal_guide_fit: NDArray[np.number]
        ''' 3-dimensional numpy array with a matrix of personal guide fitnesses for each particle. Each matrix has :attr:`~mesh.parameters.MeshParameters.max_personal_guides` fitnesses. '''

        if params.initial_points is None:
            sampler = qmc.LatinHypercube(d=params.decision_dim, scramble=True)
            sample = sampler.random(n=params.population_size)
            self.position = qmc.scale(sample, params.position_lower_bounds[:params.decision_dim], params.position_upper_bounds[:params.decision_dim])
        else:
            self.position = params.initial_points
        hiperparameter_dim = params.position_dim - params.decision_dim
        # Position = decision varibles plus (DE scaling factor, crossover probability, communication probability, three weights and mutation rate)
        self.position = np.hstack((self.position, np.random.rand(params.population_size, hiperparameter_dim)))
        self.velocity = np.random.uniform(params.velocity_lower_bounds, params.velocity_upper_bounds, (params.population_size, params.position_dim))
        self.fitness = np.full((params.population_size, params.objective_dim), np.inf)
        if params.global_guide_method in {0, 1}:
            self.sigma = np.full((params.population_size, comb(params.objective_dim, 2)), np.nan)
        else:
            self.sigma = np.empty((0, comb(params.objective_dim, 2)))
        self.global_guide = np.full((params.population_size, params.position_dim), np.nan)
        self.personal_guide_pos = np.repeat(self.position[:, np.newaxis, :], params.max_personal_guides, axis=1)
        self.personal_guide_fit = np.full((params.population_size, params.max_personal_guides, params.objective_dim), np.inf)

class Memory:
    """ Represents the MESH memory.

    Args:
        population (:class:`Population`): The attributes :attr:`~Population.position` and :attr:`~Population.fitness` are used to set the memory position and fitness.
        pareto_front (:type:`NDArray[np.integer]`): A numpy array of the particle indices for the population position and fitness matrices.
        params (:class:`~mesh.parameters.MeshParameters`): The attribute :attr:`~mesh.parameters.MeshParameters.memory_size` is used to limit the memory size.

    Raises:
        TypeError: If the input is not of the expected type.
    """
    
    def __init__(self, params: MeshParameters) -> None:
        assert_type(params, 'params', MeshParameters)

        # Set the class attributes
        self.position: NDArray[np.number] = np.empty((0, params.position_dim))
        """ Numpy matrix with the memory position. """
        self.fitness: NDArray[np.number] = np.empty((0, params.objective_dim))
        """ Numpy matrix with the memory fitness. """
        self.sigma: NDArray[np.number] = np.empty((0, 0))
        """ Numpy matrix with the memory sigma values. This attribute is only used when the Sigma method is used. """
