from .operations.differential_mutation_pool import differential_mutation_pool_options
from .operations.differential_mutation import differential_mutation_options
from .operations.global_guide_method import global_guide_method_options
from .validations.numpy_validations import assert_np_array_for_operations, assert_np_vectors_for_boundary
from .validations.python_validations import assert_type, is_greater_in_type, is_between_inclusive, is_in_options

from numpy.typing import NDArray
from typing import Optional

import numpy as np

class MeshParameters:
    ''' MESH parameters.
    
    Args:
        objective_dim (:type:`int`): Number of problem objectives. Must be a positive integer (> 0).

        decision_dim (:type:`int`): Number of problem variables. Must be a positive integer (> 0).

        decision_lower_bounds (:type:`numpy.ndarray[np.floating]`): A array with each lower bounds of the decision variables. Must be a numpy array of numbers (without NaN values) and size equals to ``decision_dim``. Each element must be less than the respective element from ``decision_upper_bounds``.

        decision_upper_bounds (:type:`numpy.ndarray[np.floating]`): A array with each upper bounds of the decision variables. Must be a numpy array of numbers (without NaN values) and size equals to ``decision_dim``. Each element must be greater than the respective element from ``decision_lower_bounds``.
            
        population_size (:type:`int`): Population size. Must be a positive integer (> 0).
        
        memory_size (:type:`typing.Optional[int]`): Number of particles in memory. Default is None. Must be a positive integer (> 0) or ``None``. If it is ``None``, the memory size will be equals to :attr:`population_size`.
        
        global_guide_method (:type:`int`): Method to select the global guide of the particles. See :attr:`~mesh.operations.global_guide_method.global_guide_method_options`.
        
        dm_pool_type (:type:`int`): Differential mutation pool where the particles will be sampled for the differential mutation operation. See :attr:`~mesh.operations.differential_mutation_pool.differential_mutation_pool_options`.
    
        dm_operation_type (:type:`int`): Differential mutation operation type. See :attr:`~mesh.operations.differential_mutation.differential_mutation_options`.
    
        communication_probability (:type:`int | float`): Communication/cooperation probability. Must be a number between 0 and 1, inclusive.
        
        mutation_rate (:type:`int | float`): Mutation rate. Must be a positive number (> 0).
        
        max_gen (:type:`typing.Optional[int]`): Maximum number of generations. Must be a positive integer (> 0) or ``None``.
        
        max_fit_eval (:type:`typing.Optional[int]`): Maximum number of fitness evaluations. Must be a positive integer (> 0) or ``None``.
        
        max_personal_guides (:type:`int`): Maximum number of personal guides. Must be a positive integer (> 0).

        initial_poinitial_pointsitions (:type:`typing.Optional[NDArray[np.number]]`): The initial particle points. If it is None, the initial points are sampled.

        random_state (:type:`typing.Optional[int]`): Numpy random seed to generate random numbers. Default is None. Must be an integer (> 0) or ``None``.

    Raises:
        TypeError: If the input is not the expected type.
        ValueError: If the input is not the allowed value.
    '''

    def __init__(self,
                 objective_dim: int,
                 decision_dim: int,
                 decision_lower_bounds: NDArray[np.floating],
                 decision_upper_bounds: NDArray[np.floating],
                 population_size: int,
                 memory_size: Optional[int] = None,
                 global_guide_method: int = 0,
                 dm_pool_type: int = 0,
                 dm_operation_type: int = 0,
                 communication_probability: int | float = 0.7,
                 mutation_rate: int | float = 0.9,
                 max_gen: Optional[int] = None,
                 max_fit_eval: Optional[int] = None,
                 max_personal_guides: int = 1,
                 initial_points: Optional[NDArray[np.number]] = None,
                 random_state: Optional[int] = None):
        
        self.objective_dim: int
        ''' Number of problem objectives. '''
        self.decision_dim: int
        ''' Number of problem variables. '''
        self.position_dim: int
        ''' Number of decision variables plus hiperparameters to auto-optimize. '''
        self.position_lower_bounds: NDArray[np.floating]
        ''' Numpy array with the lower bounds of the problem for each variable and hiperparameters. '''
        self.position_upper_bounds: NDArray[np.floating]
        ''' Numpy array with the upper bounds of the problem for each variable and hiperparameters. '''
        self.velocity_upper_bounds: NDArray[np.floating]
        ''' Numpy array with the upper bounds of the velocity calculated by:

        .. math::
            V_{max} = X_{max} - X_{min}.
        '''
        self.velocity_lower_bounds: NDArray[np.floating]
        ''' Numpy array with the upper bounds of the velocity calculated by:

        .. math::
            V_{min} = X_{min} - X_{max}.
        '''
        self.population_size: int
        ''' Number of particles. '''
        self.memory_size: int
        ''' Maximum size of MESH memory. '''
        self.global_guide_method: int
        ''' Global best selection method. See :attr:`~mesh.operations.global_guide_method.global_guide_method_options` '''
        self.dm_pool_type: int
        ''' Differential mutation pool where the particles will be sampled for the differential mutation operation. See :attr:`~mesh.operations.differential_mutation_pool.differential_mutation_pool_options` '''
        self.dm_operation_type: int
        ''' Differential mutation operation. See :attr:`~mesh.operations.differential_mutation.differential_mutation_options`. '''
        self.communication_probability: int | float
        ''' Communication/cooperation probability. It must be a number between 0 and 1. '''
        self.mutation_rate: int | float
        ''' Mutation rate. '''
        self.max_gen: int
        ''' Maximum number of generations. It won't be used if it's ``None``. '''
        self.max_fit_eval: int
        ''' Maximum number of fitness evaluations. It won't be used if it's ``None``. '''
        self.max_personal_guides: int
        ''' Maximum number of personal guides. '''
        self.initial_points: Optional[NDArray[np.number]]
        ''' The initial points of the particles. '''
        self.random_state: int | None
        ''' Seed to generate random numbers. '''

        # Set the number of objectives
        is_greater_in_type(objective_dim, 'objective_dim', int, 1)
        self.objective_dim = objective_dim
        # Set the position dimension
        is_greater_in_type(decision_dim, 'decision_dim', int, 0)
        self.decision_dim = decision_dim
        # Set the problem position dimension plus hiperparameter dimension
        self.position_dim = decision_dim + 5
        # Set the maximum and the minimum boundaries for positions
        assert_np_vectors_for_boundary(decision_lower_bounds, 'decision_lower_bounds', decision_upper_bounds, 'decision_upper_bounds', decision_dim)
        self.position_lower_bounds = np.hstack((decision_lower_bounds, np.array([0., 0., 0., 0., 0.])))
        self.position_upper_bounds = np.hstack((decision_upper_bounds, np.array([2., 2., 2., 2., 1.])))
        # Set the maximum and minimum boundaries for velocities
        self.velocity_lower_bounds =  self.position_lower_bounds - self.position_upper_bounds
        self.velocity_upper_bounds = -self.velocity_lower_bounds
        # Set the population size
        is_greater_in_type(population_size, 'population_size', int, 0)
        self.population_size = population_size
        # Set the memory size
        is_greater_in_type(memory_size, 'memory_size', int, 0, is_optional=True)
        if memory_size is None:
            self.memory_size = population_size
        else:
            self.memory_size = memory_size
        # Set the global attribution type
        is_in_options(global_guide_method, 'global_guide_method', global_guide_method_options.keys())
        self.global_guide_method = global_guide_method
        # Set the differential mutation type
        is_in_options(dm_operation_type, 'dm_operation_type', differential_mutation_options.keys())
        self.dm_operation_type = dm_operation_type
        # Set the differential mutation pool type
        is_in_options(dm_pool_type, 'dm_pool_type', differential_mutation_pool_options.keys())
        self.dm_pool_type = dm_pool_type
        # Set the communication probability
        is_between_inclusive(communication_probability, 'communication_probability', 0, 1)
        self.communication_probability = communication_probability
        # Set the mutation rate
        is_greater_in_type(mutation_rate, 'mutation_rate', (int, float), 0)
        self.mutation_rate = mutation_rate
        # Check if at least one of the stopping criteria is not None
        if max_gen is None and max_fit_eval is None:
            raise ValueError('At least one of the parameters "max_gen" and "max_fit_eval" must be not None.')
        # Set the maximum number of generations
        is_greater_in_type(max_gen, 'max_gen', int, 0, is_optional=True)
        self.max_gen = max_gen if max_gen else 0
        # Set the maximum number of fitness evaluations
        is_greater_in_type(max_fit_eval, 'max_fit_eval', int, 0, is_optional=True)
        self.max_fit_eval = max_fit_eval if max_fit_eval else 0
        # Set the number of personal guides
        is_greater_in_type(max_personal_guides, 'max_personal_guides', int, 0)
        self.max_personal_guides = max_personal_guides
        # Set the initial points of the particles
        if initial_points is not None:
            assert_np_array_for_operations(initial_points, 'initial_points', (population_size, decision_dim))
            if np.any(initial_points > decision_upper_bounds) or np.any(initial_points < decision_lower_bounds):
                ValueError('The parameter "initial_points" is the bounds of the bounding arrays.')
            self.initial_points = initial_points.copy()
        else:
            self.initial_points = None
        # Set the random state
        assert_type(random_state, 'random_state', int, is_optional=True)
        self.random_state = random_state