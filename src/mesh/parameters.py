from mesh.operations.differential_mutation_pool import differential_mutation_pool_options
from mesh.operations.differential_mutation import differential_mutation_options
from mesh.operations.global_best_attribution import global_best_attribution_options
from mesh.validations.numpy_validations import assert_np_array_for_operations, assert_np_vectors_for_boundary
from mesh.validations.python_validations import assert_type, is_greater_in_type, is_between_inclusive, is_in_options

from typing import Optional

import numpy as np

class MeshParameters:
    ''' MESH parameters.
    
    Args:
        objective_dim (:type:`int | np.integer`): Number of problem objectives. Must be a positive integer (> 0).

        position_dim (:type:`int | np.integer`): Number of problem variables. Must be a positive integer (> 0).

        lower_bound_array (:type:`numpy.ndarray[np.number]`): A array with each lower bound of problem. Must be a numpy array of numbers (without NaN values) and size equal to ``position_dim``. Each element must be less than the respective element from ``upper_bound_array``.

        upper_bound_array (:type:`numpy.ndarray[np.number]`): A array with each upper bound of problem. Must be a numpy array of numbers (without NaN values) and size equal to ``position_dim``. Each element must be greater than the respective element from ``lower_bound_array``.
            
        population_size (:type:`int | np.integer`): Population size. Must be a positive integer (> 0).
        
        memory_size (:type:`int | np.integer | None`): Number of particles in memory. Default is None. Must be a positive integer (> 0) or ``None``. If it is ``None``, the memory size will be equal to population size.
        
        global_best_attribution_type (:type:`{0, 1, 2, 3}`): Global best attribution operation type. See :attr:`~mesh.operations.global_best_attribution.global_best_attribution_options`.
        
        dm_pool_type (:type:`{0, 1, 2}`): Differential mutation pool where the particles will be sampled for the differential mutation operation. See :attr:`~mesh.operations.differential_mutation_pool.differential_mutation_pool_options`.
    
        dm_operation_type (:type:`{0, 1, 2, 3, 4}`): Differential mutation operation type. See :attr:`~mesh.operations.differential_mutation.differential_mutation_options`.
    
        communication_probability (:type:`int | float | np.number`): Communication/cooperation probability. Must be a number between 0 and 1, inclusive.
        
        mutation_rate (:type:`int | float | np.number`): Mutation rate. Must be a number between 0 and 1, inclusive.
        
        max_gen (:type:`int | np.integer`): Maximum number of generations. Must be a non-negative integer or ``None``.
        
        max_fit_eval (:type:`int | np.integer`): Maximum number of fitness evaluations. Must be a non-negative integer or ``None``.
        
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
                 lower_bound_array: np.ndarray[np.number],
                 upper_bound_array: np.ndarray[np.number],
                 population_size: int | np.integer,
                 memory_size: int | Optional[np.integer] = None,
                 global_best_attribution_type: {0,1,2,3} = 0,
                 dm_pool_type: {0,1,2} = 0,
                 dm_operation_type: {0,1,2,3,4} = 0,
                 communication_probability: int | float | np.number = 0.7,
                 mutation_rate: int | float | np.number = 0.9,
                 max_gen: int | np.integer = 0,
                 max_fit_eval: int | np.integer = 0,
                 max_personal_guides: int | np.integer = 3,
                 initial_positions: np.ndarray[np.number, 2] = None,
                 random_state: int | Optional[np.integer] = None):
        
        self.objective_dim: int | np.integer
        ''' Number of problem objectives. '''
        self.position_dim: int | np.integer
        ''' Number of problem variables. '''
        self.lower_bound_array: np.ndarray[np.number]
        ''' Numpy array with the lower bound of the problem for each variable. '''
        self.upper_bound_array: np.ndarray[np.number]
        ''' Numpy array with the upper bound of the problem for each variable. '''
        self.velocity_max_value: np.ndarray[np.float64]
        ''' Numpy array with the upper bound of the velocity calculated by:

        .. math::
            V_{max} = X_{max} - X_{min}.
        '''
        self.velocity_min_value: np.ndarray[np.float64]
        ''' Numpy array with the upper bound of the velocity calculated by:

        .. math::
            V_{min} = X_{min} - X_{max}.
        '''
        self.population_size: int | np.integer
        ''' Number of particles. '''
        self.memory_size: int | Optional[np.integer]
        ''' Maximum size of MESH memory. If it is ``None``, so the memory size will be equal to :attr:`population_size`. '''
        self.global_best_attribution_type: {0,1,2,3}
        ''' Global best selection method. See :attr:`~mesh.operations.global_best_attribution.global_best_attribution_options` '''
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
        assert_np_vectors_for_boundary(lower_bound_array, 'lower_bound_array', upper_bound_array, 'upper_bound_array', position_dim)
        self.lower_bound_array = lower_bound_array
        self.upper_bound_array = upper_bound_array
        # Set the maximum and minimum boundaries for velocities
        self.velocity_min_value =  self.lower_bound_array - self.upper_bound_array
        self.velocity_max_value = - self.velocity_min_value
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
        is_in_options(global_best_attribution_type, 'global_best_attribution_type', global_best_attribution_options.keys())
        self.global_best_attribution_type = global_best_attribution_type
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
        assert_type(max_gen, 'max_gen', (int, np.integer), is_optional=True)
        self.max_gen = max_gen
        # Set the maximum number of fitness evaluations
        assert_type(max_fit_eval, 'max_fit_eval', (int, np.integer), is_optional=True)
        self.max_fit_eval = max_fit_eval
        if self.max_gen == None and self.max_fit_eval == None:
            raise ValueError('At least one of the parameters max_gen and max_fit_eval must be not None.')
        # Set the number of personal guides
        is_greater_in_type(max_personal_guides, 'max_personal_guides', (int, np.integer), 0)
        self.max_personal_guides = max_personal_guides
        # Set the initial positions of the particles
        if initial_positions != None:
            assert_np_array_for_operations(initial_positions, 'initial_positions', (population_size, position_dim))
        self.initial_positions = initial_positions
        # Set the random state
        assert_type(random_state, 'random_state', (int, np.integer), is_optional=True)
        self.random_state = random_state