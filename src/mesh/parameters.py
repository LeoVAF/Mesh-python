from mesh.operations.differential_mutation_pool import differential_mutation_pool_options
from mesh.operations.differential_mutation import differential_mutation_options
from mesh.operations.global_guide_method import global_guide_method_options
from mesh.validations.numpy_validations import assert_np_array_for_operations, assert_np_vectors_for_boundary
from mesh.validations.python_validations import assert_type, is_greater_in_type, is_between_inclusive, is_in_options

from typing import Optional

import numpy as np

class MeshParameters:
    ''' MESH parameters.
    
    Args:
        objective_dim (:type:`int | np.integer`): Number of problem objectives. Must be a positive integer (> 0).

        position_dim (:type:`int | np.integer`): Number of problem variables. Must be a positive integer (> 0).

        position_lower_bounds (:type:`numpy.ndarray[np.number]`): A array with each lower bounds of the problem. Must be a numpy array of numbers (without NaN values) and size equals to ``position_dim``. Each element must be less than the respective element from ``position_upper_bounds``.

        position_upper_bounds (:type:`numpy.ndarray[np.number]`): A array with each upper bounds of the problem. Must be a numpy array of numbers (without NaN values) and size equals to ``position_dim``. Each element must be greater than the respective element from ``position_lower_bounds``.
            
        population_size (:type:`int | np.integer`): Population size. Must be a positive integer (> 0).
        
        memory_size (:type:`int | np.integer | None`): Number of particles in memory. Default is None. Must be a positive integer (> 0) or ``None``. If it is ``None``, the memory size will be equals to population size.
        
        global_guide_method (:type:`{0, 1, 2, 3}`): Method to select the global guide of the particles. See :attr:`~mesh.operations.global_guide_method.global_guide_method_options`.
        
        dm_pool_type (:type:`{0, 1, 2}`): Differential mutation pool where the particles will be sampled for the differential mutation operation. See :attr:`~mesh.operations.differential_mutation_pool.differential_mutation_pool_options`.
    
        dm_operation_type (:type:`{0, 1, 2, 3, 4}`): Differential mutation operation type. See :attr:`~mesh.operations.differential_mutation.differential_mutation_options`.
    
        communication_probability (:type:`int | float | np.number`): Communication/cooperation probability. Must be a number between 0 and 1, inclusive.
        
        mutation_rate (:type:`int | float | np.number`): Mutation rate. Must be a number between 0 and 1, inclusive.
        
        max_gen (:type:`int | np.integer | None`): Maximum number of generations. Must be a non-negative integer or ``None``.
        
        max_fit_eval (:type:`int | np.integer | None`): Maximum number of fitness evaluations. Must be a non-negative integer or ``None``.
        
        max_personal_guides (:type:`int | np.integer`): Maximum number of personal guides. Must be a positive integer (> 0).

        initial_positions (:type:`np.ndarray[np.number, 2] | None`): The initial particle positions. If it is None, the initial positions are initialized randomly under the uniform distribution.

        random_state (:type:`int | np.integer | None`): Numpy random seed to generate random numbers. Default is None. Must be an integer or ``None``.

    Raises:
        TypeError: If the input is not the expected type.
        ValueError: If the input is not the allowed value.
    '''

    def __init__(self,
                 objective_dim: int | np.integer,
                 position_dim: int | np.integer,
                 position_lower_bounds: np.ndarray[np.number],
                 position_upper_bounds: np.ndarray[np.number],
                 population_size: int | np.integer,
                 memory_size: int | np.integer | None = None,
                 global_guide_method: {0,1,2,3} = 0,
                 dm_pool_type: {0,1,2} = 0,
                 dm_operation_type: {0,1,2,3,4} = 0,
                 communication_probability: int | float | np.number = 0.7,
                 mutation_rate: int | float | np.number = 0.9,
                 max_gen: int | np.integer = None,
                 max_fit_eval: int | np.integer = None,
                 max_personal_guides: int | np.integer = 3,
                 initial_positions: Optional[np.ndarray[np.number, 2]] = None,
                 random_state: int | np.integer | None = None):
        
        self.objective_dim: int | np.integer
        ''' Number of problem objectives. '''
        self.position_dim: int | np.integer
        ''' Number of problem variables. '''
        self.position_lower_bounds: np.ndarray[np.number]
        ''' Numpy array with the lower bounds of the problem for each variable. '''
        self.position_upper_bounds: np.ndarray[np.number]
        ''' Numpy array with the upper bounds of the problem for each variable. '''
        self.velocity_upper_bounds: np.ndarray[np.float64]
        ''' Numpy array with the upper bounds of the velocity calculated by:

        .. math::
            V_{max} = X_{max} - X_{min}.
        '''
        self.velocity_lower_bounds: np.ndarray[np.float64]
        ''' Numpy array with the upper bounds of the velocity calculated by:

        .. math::
            V_{min} = X_{min} - X_{max}.
        '''
        self.population_size: int | np.integer
        ''' Number of particles. '''
        self.memory_size: int | Optional[np.integer]
        ''' Maximum size of MESH memory. If it is ``None``, so the memory size will be equal to :attr:`population_size`. '''
        self.global_guide_method: {0,1,2,3}
        ''' Global best selection method. See :attr:`~mesh.operations.global_guide_method.global_guide_method_options` '''
        self.dm_pool_type: {0,1,2}
        ''' Differential mutation pool where the particles will be sampled for the differential mutation operation. See :attr:`~mesh.operations.differential_mutation_pool.differential_mutation_pool_options` '''
        self.dm_operation_type: {0,1,2,3,4}
        ''' Differential mutation operation. See :attr:`~mesh.operations.differential_mutation.differential_mutation_options`. '''
        self.communication_probability: int | float
        ''' Communication/cooperation probability. It must be a number between 0 and 1. '''
        self.mutation_rate: int | float
        ''' Mutation rate. '''
        self.max_gen: int | np.integer | None
        ''' Maximum number of generations. It won't be used if it's ``None``. '''
        self.max_fit_eval: int | np.integer | None
        ''' Maximum number of fitness evaluations. It won't be used if it's ``None``. '''
        self.max_personal_guides: int | np.integer
        ''' Maximum number of personal guides. '''
        self.initial_positions: Optional[np.ndarray[np.number, 2]]
        ''' The initial positions of the particles. '''
        self.random_state: int | np.integer | None
        ''' Seed to generate random numbers. '''

        # Set the number of objectives
        is_greater_in_type(objective_dim, 'objective_dim', (int, np.integer), 1)
        self.objective_dim = objective_dim
        # Set the position dimension
        is_greater_in_type(position_dim, 'position_dim', (int, np.integer), 0)
        self.position_dim = position_dim
        # Set the maximum and the minimum boundaries for positions
        assert_np_vectors_for_boundary(position_lower_bounds, 'position_lower_bounds', position_upper_bounds, 'position_upper_bounds', position_dim)
        self.position_lower_bounds = position_lower_bounds
        self.position_upper_bounds = position_upper_bounds
        # Set the maximum and minimum boundaries for velocities
        self.velocity_lower_bounds =  self.position_lower_bounds - self.position_upper_bounds
        self.velocity_upper_bounds = -self.velocity_lower_bounds
        # Set the population size
        is_greater_in_type(population_size, 'population_size', (int, np.integer), 0)
        self.population_size = population_size
        # Set the memory size
        is_greater_in_type(memory_size, 'memory_size', (int, np.integer), 0, is_optional=True)
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
        is_between_inclusive(mutation_rate, 'mutation_rate', 0, 1)
        self.mutation_rate = mutation_rate
        # Set the maximum number of generations
        is_greater_in_type(max_gen, 'max_gen', (int, np.integer), 0, is_optional=True)
        self.max_gen = max_gen
        # Set the maximum number of fitness evaluations
        is_greater_in_type(max_fit_eval, 'max_fit_eval', (int, np.integer), 0, is_optional=True)
        self.max_fit_eval = max_fit_eval
        if max_gen is None and max_fit_eval is None:
            raise ValueError('At least one of the parameters "max_gen" and "max_fit_eval" must be not None.')
        # Set the number of personal guides
        is_greater_in_type(max_personal_guides, 'max_personal_guides', (int, np.integer), 0)
        self.max_personal_guides = max_personal_guides
        # Set the initial positions of the particles
        if initial_positions is not None:
            assert_np_array_for_operations(initial_positions, 'initial_positions', (population_size, position_dim))
            if np.any(initial_positions > position_upper_bounds) or np.any(initial_positions < position_lower_bounds):
                ValueError('The parameter "initial_positions" is the bounds of the bounding arrays.')
        self.initial_positions = initial_positions
        # Set the random state
        assert_type(random_state, 'random_state', (int, np.integer), is_optional=True)
        self.random_state = random_state