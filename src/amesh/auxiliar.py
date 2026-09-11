import numpy as np

from .parameters import AMESHParameters
from .validations.python_validations import assert_type


class StoppingAlgorithm(Exception):
    ''' Class used to stop the algorithm with an exception.
    
    Args:
        position (:type:`np.typing.NDArray[np.number]`): The position matrix of the particles when the algorithm was stopped.
        fitness (:type:`np.typing.NDArray[np.number]`): The fitness matrix of the particles when the algorithm was stopped.
    '''

    def __init__(self, position: np.typing.NDArray[np.number], fitness: np.typing.NDArray[np.number]):
        self.position: np.typing.NDArray[np.number]
        self.fitness: np.typing.NDArray[np.number]

        self.position = position
        self.fitness = fitness

class PreAllocated:
    ''' Used for data allocation. It stores some data structures to avoid new allocations.
    
    Args:
        params (:class:`~amesh.parameters.AMESHParameters`): The attributes :attr:`~amesh.parameters.AMESHParameters.objective_dim`, :attr:`~amesh.parameters.AMESHParameters.position_dim`, :attr:`~amesh.parameters.AMESHParameters.population_size` and :attr:`~amesh.parameters.AMESHParameters.global_guide_method` are used to initialize the pre-allocations.

    Raises:
        TypeError: If the ``params`` is not an instance of :class:`~amesh.parameters.AMESHParameters`.
    '''

    def __init__(self, params: AMESHParameters):
        assert_type(params, 'params', AMESHParameters)

        self.np_tril_indices: tuple[np.typing.NDArray[np.intp], np.typing.NDArray[np.intp]]
        ''' The row and column indices for the lower-triangle of a matrix, respectively. The row indices are sorted in non-decreasing order, and the correspdonding column indices are strictly increasing for each row. Used only if the Sigma method is used. '''
        self.global_guide_mutated: np.typing.NDArray[np.number]
        ''' Numpy matrix for store the global guides after the mutation operation. '''
        self.fitness_elitism: np.typing.NDArray[np.number]
        '''NumPy matrix used in :meth:`~amesh.core.AMESH.elitism` to store population fitness before and after particle movement.'''
        self.position_copy: np.typing.NDArray[np.number]
        '''NumPy matrix that stores particle positions before particle movement.'''
        self.velocity_copy: np.typing.NDArray[np.number]
        '''NumPy matrix that stores particle velocities before particle movement.'''
        self.fitness_copy: np.typing.NDArray[np.number]
        '''NumPy matrix that stores particle fitness before particle movement.'''

        # Used to calculate the sigma
        if params.global_guide_method in {0, 1}:
            self.np_tril_indices = np.tril_indices(params.objective_dim, k=-1)
        else:
            self.np_tril_indices = np.empty((0, 0), dtype=int), np.empty((0, 0), dtype=int)
        # Matrix for store the global guides after the mutation
        self.global_guide_mutated = np.empty((params.population_size, params.decision_dim))
        # Fitness matrix for the elitism
        self.fitness_elitism = np.empty((2*params.population_size, params.objective_dim))
        # Copies for the population
        self.position_copy = np.empty((params.population_size, params.decision_dim))
        self.velocity_copy = np.empty((params.population_size, params.decision_dim))
        self.fitness_copy = np.empty((params.population_size, params.objective_dim))
