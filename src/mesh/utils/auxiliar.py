import numpy as np

from parameters import MeshParameters
from validations.python import assert_type

from sklearn.neighbors import NearestNeighbors

class StoppingAlgorithm(Exception):
    ''' Class used to stop the algorithm with an exception. '''

    def __init__(self):
        pass

class PreAllocated():
    ''' Used for data allocation. It stores some data structures to avoid new allocations.
    
    Args:
        params (:class:`~mesh.parameters.MeshParameters`): The attributes :attr:`~mesh.parameters.MeshParameters.objective_dim`, :attr:`~mesh.parameters.MeshParameters.position_dim`, :attr:`~mesh.parameters.MeshParameters.population_size` and :attr:`~mesh.parameters.MeshParameters.global_best_attribution_type` are used to initialize the pre-allocations.

    Raises:
        TypeError: If the ```params`` is not an instance of :class:`~mesh.parameters.MeshParameters`.
    '''

    def __init__(self, params: MeshParameters) -> None:
        assert_type(params, 'params', MeshParameters)

        self.np_tril_indices: tuple[np.array[np.uint64], np.array[np.uint64]]
        ''' The row and column indices for the lower-triangle of a matrix, respectively. The row indices are sorted in non-decreasing order, and the correspdonding column indices are strictly increasing for each row. Used only if the Sigma method is used. '''
        self.nearest_neighbors: NearestNeighbors
        ''' Instance of :type:`~sklearn.neighbors.NearestNeighbors` with two neighbors and Euclidean as distance metric. '''
        self.matrix_for_operations: np.ndarray[np.float64, 2]
        ''' Numpy matrix for operations. '''
        self.vector_for_operations: np.ndarray[np.float64]
        ''' Numpy array for operations. '''
        self.fitness_selection: np.ndarray[np.float64, 2]
        ''' Numpy matrix used in :meth:`~mesh.MESH.MESH.population_selection` to store the fitness of the population before and after the particle moviment. '''
        self.position_copy: np.ndarray[np.float64, 2]
        ''' Numpy matrix to store the position of the particles before the particle moviment. '''
        self.velocity_copy: np.ndarray[np.float64, 2]
        ''' Numpy matrix to store the velocity of the particles before the particle moviment. '''
        self.fitness_copy: np.ndarray[np.float64, 2]
        ''' Numpy matrix to store the fitness of the particles before the particle moviment. '''

        # Used to calculate the sigma
        if params.global_best_attribution_type < 2:
            self.np_tril_indices = np.tril_indices(params.objective_dim, k=-1)
        # The object to get the nearest neighbors
        self.nearest_neighbors = NearestNeighbors(n_neighbors=2, algorithm='auto', metric='euclidean')
        # Structures used to calculate repetitive operations
        self.matrix_for_operations = np.empty((params.population_size, params.position_dim))
        self.vector_for_operations = np.empty(params.population_size)
        # Fitness matrix for the population selection
        self.fitness_selection = np.empty((2*params.population_size, params.objective_dim))
        # Copies for the population
        self.position_copy = np.empty((params.population_size, params.position_dim))
        self.velocity_copy = np.empty((params.population_size, params.position_dim))
        self.fitness_copy = np.empty((params.population_size, params.objective_dim))
