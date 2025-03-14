from mesh.validations.python import assert_type, is_greater_in_type, is_between_inclusive, is_in_options
from mesh.validations.numpy import assert_np_vectors_for_boundary
from mesh.operations.global_best_attribution import global_best_attribution_options
from mesh.operations.differential_mutation_pool import differential_mutation_pool_options
from mesh.operations.differential_mutation_operation import differential_mutation_operation_options

from typing import Optional

import numpy as np

class MeshParameters:
    ''' MESH parameters.
    
    Args:
        objective_dim (:type:`int | np.integer`): Number of problem objectives. Must be a positive integer (> 0).

        position_dim (:type:`int | np.integer`): Number of problem variables. Must be a positive integer (> 0).

        position_max_value (:type:`numpy.ndarray[np.number]`): A array with each upper bound of problem. Must be a numpy array of numbers (without NaN values) and size equal to ``position_dim``. Each element must be greater than the respective element from ``position_min_value``.
            
        position_min_value (:type:`numpy.ndarray[np.number]`): A array with each lower bound of problem. Must be a numpy array of numbers (without NaN values) and size equal to ``position_dim``. Each element must be less than the respective element from ``position_max_value``.
            
        population_size (:type:`int | np.integer`): Population size. Must be a positive integer (> 0).
        
        memory_size (:type:`int | np.integer`, optional): Number of particles in memory. Default is None. Must be a positive integer (> 0) or ``None``.
        
        global_best_attribution_type (:type:`{0, 1, 2, 3}`): Global best attribution operation type. See :attr:`~mesh.operations.global_best_attribution.global_best_attribution_options`.
        
        dm_pool_type (:type:`{0, 1, 2}`): Differential mutation pool where the particles will be sampled for the differential mutation operation. See :attr:`~mesh.operations.differential_mutation_pool.differential_mutation_pool_options`.
    
        dm_operation_type (:type:`{0, 1, 2, 3, 4}`): Differential mutation operation type. See :attr:`~mesh.operations.differential_mutation_operation.differential_mutation_operation_options`.
    
        communication_probability (:type:`int | float | np.number`): Communication probability. Must be a number between 0 and 1, inclusive.
        
        mutation_rate (:type:`int | float | np.number`): Mutation rate. Must be a number between 0 and 1, inclusive.
        
        max_gen (:type:`int | np.integer`): Maximum number of generations. Must be an integer.
        
        max_fit_eval (:type:`int | np.integer`): Maximum number of fitness evaluations. Must be an integer.
        
        max_personal_guides (:type:`int | np.integer`): Maximum number of personal guides. Must be a positive integer (> 0).

        random_state (:type:`int | np.integer`, optional): Numpy random seed to generate random numbers. Default is None. Must be an integer or ``None``.

    Raises:
        TypeError: If the input is not the expected type.
        ValueError: If the input is not the allowed value.
    '''

    def __init__(self,
                 objective_dim: int | np.integer,
                 position_dim: int | np.integer,
                 position_min_value: np.ndarray[np.number],
                 position_max_value: np.ndarray[np.number],
                 population_size: int | np.integer,
                 memory_size: Optional[int] | np.integer = None,
                 global_best_attribution_type: {0,1,2,3} = 0,
                 dm_pool_type: {0,1,2} = 0,
                 dm_operation_type: {0,1,2,3,4} = 0,
                 communication_probability: int | float | np.number = 0.7,
                 mutation_rate: int | float | np.number = 0.9,
                 max_gen: int | np.integer = 0,
                 max_fit_eval: int | np.integer = 0,
                 max_personal_guides: int | np.integer = 3,
                 random_state: Optional[int] | np.integer = None):
        
        self.objective_dim: int | np.integer
        ''' Number of problem objectives. '''
        self.position_dim: int | np.integer
        ''' Number of problem variables. '''
        self.position_min_value: np.ndarray[np.number]
        ''' Numpy array with the lower bound of the problem for each variable. '''
        self.position_max_value: np.ndarray[np.number]
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
        self.memory_size: Optional[int] | np.integer
        ''' Maximum size of MESH memory. If it is ``None``, so the memory size will be equal to :attr:`population_size`. '''
        self.global_best_attribution_type: {0,1,2,3}
        ''' Global best selection method. '''
        self.dm_pool_type: {0,1,2}
        ''' Differential mutation pool where the particles will be sampled for the differential mutation operation. '''
        self.dm_operation_type: {0,1,2,3,4}
        ''' Differential mutation operation. '''
        self.communication_probability: int | float
        ''' Communication probability. It must be a number between 0 and 1. '''
        self.mutation_rate: int | float
        ''' Mutation rate. '''
        self.max_gen: int | np.integer
        ''' Maximum number of generations. If the parameter ``max_gen`` is negative, so it will be equal to 0. '''
        self.max_fit_eval: int | np.integer
        ''' Maximum number of fitness evaluations. If the parameter ``max_fit_eval`` is negative, so it will be equal to 0. '''
        self.max_personal_guides: int | np.integer
        ''' Maximum number of personal guides. '''
        self.random_state: Optional[int] | np.integer
        ''' Seed to generate random numbers. '''

        # Set the number of objectives
        is_greater_in_type(objective_dim, 'objective_dim', (int, np.integer), 0)
        self.objective_dim = objective_dim
        # Set the position dimension
        is_greater_in_type(position_dim, 'position_dim', (int, np.integer), 0)
        self.position_dim = position_dim
        # Set the maximum and the minimum boundaries for positions
        assert_np_vectors_for_boundary(position_min_value, 'position_min_value', position_max_value, 'position_max_value', position_dim)
        self.position_min_value = position_min_value
        self.position_max_value = position_max_value
        # Set the maximum and minimum boundaries for velocities
        self.velocity_min_value =  self.position_min_value - self.position_max_value
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
        is_in_options(dm_operation_type, 'dm_operation_type', differential_mutation_operation_options.keys())
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
        assert_type(max_gen, 'max_gen', (int, np.integer))
        self.max_gen = max(max_gen, 0)
        # Set the maximum number of fitness evaluations
        assert_type(max_fit_eval, 'max_fit_eval', (int, np.integer))
        self.max_fit_eval = max(max_fit_eval, 0)
        if self.max_gen == 0 and self.max_fit_eval == 0:
            raise ValueError('At least one of the parameters max_gen and max_fit_eval must be greater than zero.')
        # Set the number of personal guides
        is_greater_in_type(max_personal_guides, 'max_personal_guides', (int, np.integer), 0)
        self.max_personal_guides = max_personal_guides
        # Set the random state
        assert_type(random_state, 'random_state', (int, np.integer), is_optional=True)
        self.random_state = random_state