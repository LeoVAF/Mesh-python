import numpy as np

from typing import Optional
from utils.validations import is_greater

class MeshParameters:
    ''' MESH parameters.
    
    Args:
        objective_dim (:type:`int`): Number of problem objectives.

            Raises:
                TypeError: If its type is not :type:`int`.
                ValueError: If its value is less than 1.

        position_dim (:type:`int`): Number of problem variables.
            Raises:
                TypeError: If its type is not :type:`int`.
                ValueError: If its value is less than 1.

        position_max_value (:type:`numpy.ndarray`): A array with each upper bound of problem.
            
            Raises:
                TypeError: If it is not a numpy array of numbers (without NaN values).
                ValueError:
                    - If it is not an one-dimensional array.
                    - If it doesn't have the size equal to :attr:`position_dim`.
                    - If it has any element greater than the respective element from :attr:`position_min_value`.
            
        position_min_value (:type:`numpy.ndarray`): A array with each lower bound of problem.

            Raises:
                TypeError: If it is not a numpy array of numbers (without NaN values).
                ValueError:
                    - If it is not an one-dimensional array.
                    - If it doesn't have the size equal to :attr:`position_dim`.
                    - If it has any element greater than the respective element from :attr:`position_max_value`.
            
        population_size (:type:`int`): Population size.

            Raises:
                TypeError: If its type is not :type:`int`.
                ValueError: If its value is less than 1.
        
        memory_size (:type:`int`, optional): Number of particles in memory. Default is None.
            
            Raises:
                TypeError: If its type is not :type:`int` or :data:`None`.
                ValueError: If its value is less than 1, when its type is :type:`int`.
        
        global_best_attribution_type (:type:`{0,1,2,3}`): Global best selection method. The options are:

            - :data:`0`: Applies Sigma method in memory to select the global best.
            - :data:`1`: Applies Sigma method in fronts to select the global best. Each particle will select its global best from the next front. Particles in Pareto front will select the global best from memory.
            - :data:`2`: Chooses randomly under uniform distribution a particle from memory.
            - :data:`3`: Chooses randomly under uniform distribution a particle from fronts. Each particle will select its global best from the next front. Particles in Pareto front will select the global best from memory.

            Raises:
                ValueError: If its value is not one of the options.
        
        dm_pool_type (:type:`{0,1,2}`): Differential mutation pool where the particles will be sampled for the differential mutation operation. The options are:
        
            - :data:`0`: Population.
            - :data:`1`: Memory.
            - :data:`2`: Both population and memory.

        Raises:
            ValueError: If its value is not one of the options.
    
        de_mutation_type (:type:`{0,1,2,3,4}`): Differential mutation operation. The options are:

            - :data:`0`: DE/rand/1/Bin.
            - :data:`1`: DE/rand/2/Bin.
            - :data:`2`: DE/best/1/Bin.
            - :data:`3`: DE/current-to-best/1/Bin.
            - :data:`4`: DE/current-to-rand/1/Bin.

        Raises:
            ValueError: If its value is not one of the options.
    
        communication_probability (:type:`float` | :type:`int`): Communication probability.

            Raises:
                TypeError: If its type is not :type:`float` or :type:`int`.
                ValueError: If it is not a number between 0 and 1, inclusive.
        
        mutation_rate (:type:`float` | :type:`int`): Mutation rate.

            Raises:
                TypeError: If its type is not :type:`float` or :type:`int`.
                ValueError: If it is not a number between 0 and 1, inclusive.
        
        max_gen (:type:`int`): Maximum number of generations.

            Raises:
                TypeError: If its type is not :type:`int`.
        
        max_fit_eval (:type:`int`): Maximum number of fitness evaluations.

            Raises:
               TypeError: If its type is not :type:`int`.
        
        max_personal_guides (:type:`int`): Maximum number of personal guides.

            Raises:
                TypeError: If its type is not :type:`int`.
                ValueError: If its value is less than 1.
        
        random_state (:type:`int`, optional): Numpy random seed to generate random numbers.

            Raises:
                TypeError: If its type is not :type:`int` or :data:`None`.
    '''

    def __init__(self,
                 objective_dim: int,
                 position_dim: int,
                 position_max_value: np.ndarray[np.float64],
                 position_min_value: np.ndarray[np.float64],
                 population_size: int,
                 memory_size: Optional[int] = None,
                 global_best_attribution_type: {0,1,2,3} = 0,
                 dm_pool_type: {0,1,2} = 0,
                 de_mutation_type: {0,1,2,3,4} = 0,
                 communication_probability: float | int = 0.7,
                 mutation_rate: float | int = 0.9,
                 max_gen: int = 0,
                 max_fit_eval: int = 0,
                 max_personal_guides: int = 3,
                 random_state: Optional[int] = None):
        
        self.objective_dim: int
        ''' Number of problem objectives. '''
        self.position_dim: int
        ''' Number of problem variables. '''
        self.position_max_value: np.ndarray[np.float64]
        ''' Numpy array with the upper bound of the problem for each variable. '''
        self.position_min_value: np.ndarray[np.float64]
        ''' Numpy array with the lower bound of the problem for each variable. '''
        self.velocity_max_value: np.ndarray[np.float64]
        ''' Numpy array with the upper bound of the velocity calculated by:

        .. math::
            V_{max} = X_{max} - X_{min}. '''
        self.velocity_min_value: np.ndarray[np.float64]
        ''' Numpy array with the upper bound of the velocity calculated by:

        .. math::
            V_{min} = X_{min} - X_{max}. '''
        self.population_size: int
        ''' Number of particles. '''
        self.memory_size: Optional[int]
        ''' Maximum size of MESH memory. If it is :data:`None`, so the memory size will be equal to :param:`population_size`. '''
        self.global_best_attribution_type: {0,1,2,3}
        ''' Global best selection method. '''
        self.dm_pool_type: {0,1,2}
        ''' Differential mutation pool where the particles will be sampled for the differential mutation operation. '''
        self.de_mutation_type: {0,1,2,3,4}
        ''' Differential mutation operation. '''
        self.communication_probability: float | int
        ''' Communication probability. It must be a number between 0 and 1. '''
        self.mutation_rate: float | int
        ''' Mutation rate. '''
        self.max_gen: int
        ''' Maximum number of generations. If `max_gen` it is negative, so it will be equal to 0. '''
        self.max_fit_eval: int
        ''' Maximum number of fitness evaluations. If `max_fit_eval` is negative, so it will be equal to 0. '''
        self.max_personal_guides: int
        ''' Maximum number of personal guides. '''
        self.random_state: Optional[int]
        ''' Seed to generate random numbers. '''

        # Set the number of objectives
        if not isinstance(objective_dim, int):
            raise TypeError('The input "objective_dim" must be a integer.')
        if (objective_dim < 1):
            raise ValueError('The input "objective_dim" must be greater than 0.')
        self.objective_dim = objective_dim
        # Set the position dimension
        if not isinstance(position_dim, int):
            raise TypeError('The input "position_dim" must be a integer.')
        if (position_dim < 1):
            raise ValueError('The input "position_dim" must be greater than 0.')
        self.position_dim = position_dim
        # Set the maximum and minimum boundaries for positions
        if not isinstance(position_max_value, np.ndarray):
            raise TypeError('The input "position_max_value" must be a numpy.ndarray.')
        if not isinstance(position_min_value, np.ndarray):
            raise TypeError('The input "position_min_value" must be a numpy.ndarray.')
        if not np.issubdtype(position_max_value.dtype, np.number) or np.any(np.isnan(position_max_value)):
            raise TypeError('The input "position_max_value" must be an array of numbers (without NaN values).')
        if not np.issubdtype(position_min_value.dtype, np.number) or np.any(np.isnan(position_min_value)):
            raise TypeError('The input "position_min_value" must be an array of numbers (without NaN values).')
        if position_max_value.ndim != 1:
            raise ValueError('The input "position_max_value" must be a one-dimensional array.')
        if position_min_value.ndim != 1:
            raise ValueError('The input "position_min_value" must be a one-dimensional array.')
        if len(position_max_value) != position_dim:
            raise ValueError('The input "position_max_value" must be size equal to "position_dim".')
        if len(position_min_value) != position_dim:
            raise ValueError('The input "position_min_value" must be size equal to "position_dim".')
        if np.any(position_max_value < position_min_value):
            raise ValueError('Each element of "position_max_value" must be greater than or equal to "position_min_value".')
        self.position_max_value = position_max_value
        self.position_min_value = position_min_value
        # Set the maximum and minimum boundaries for velocities
        self.velocity_max_value = self.position_max_value - self.position_min_value
        self.velocity_min_value = -self.velocity_max_value
        # Set the population size
        if not isinstance(population_size, int):
            raise TypeError('The input "population_size" must be an integer.')
        if population_size < 1:
            raise ValueError('The input "population_size" must be greater than 0.')
        self.population_size = population_size
        # Set the memory size
        if memory_size is None:
            self.memory_size = population_size
        if not isinstance(memory_size, int):
            raise TypeError('The input "memory_size" must be an integer.')
        elif memory_size < 1:
            raise ValueError('The input "memory_size" must be greater than 0.')
        else:
            self.memory_size = memory_size
        # Validate the options of the MESH
        valid_options = {
            "global_best_attribution_type": (0, 1, 2, 3),
            "de_mutation_type": (0, 1, 2, 3, 4),
            "dm_pool_type": (0, 1, 2),
        }
        for param, valid_set in valid_options.items():
            if locals()[param] not in valid_set:
                raise ValueError(f'The input "{param}" must be one of these options: {valid_set}.')
        # Set the global attribution type
        self.global_best_attribution_type = global_best_attribution_type
        # Set the differential mutation type
        self.de_mutation_type = de_mutation_type
        # Set the differential mutation pool type
        self.dm_pool_type = dm_pool_type
        # Set the communication probability
        if not isinstance(communication_probability, (float, int)):
           raise TypeError('The input "communication_probability" must be a float or int.') 
        if not (0 <= communication_probability <= 1):
            raise ValueError('Communication probability must be between 0 and 1.')
        self.communication_probability = communication_probability
        # Set the mutation rate
        if not isinstance(mutation_rate, (float, int)):
            raise TypeError('The input "mutation_rate" must be a float or int.')
        if not (0 <= mutation_rate <= 1):
            raise ValueError('Mutation rate must be between 0 and 1.')
        self.mutation_rate = mutation_rate
        # Set the maximum number of generations
        if not isinstance(max_gen, int):
            raise TypeError('The input "max_gen" must be an integer.')
        self.max_gen = max(max_gen, 0)
        # Set the maximum number of fitness evaluations
        if not isinstance(max_fit_eval, int):
            raise TypeError('The input "max_fit_eval" must be an integer.')
        self.max_fit_eval = max(max_fit_eval, 0)
        # Set the number of personal guides
        if not isinstance(max_personal_guides, int):
            raise TypeError('The input "max_personal_guides" must be a integer.')
        if max_personal_guides < 1:
            raise ValueError('The input "max_personal_guides" must be greater than 0.')
        self.max_personal_guides = max_personal_guides
        # Set the random state (if different from None)
        if random_state is not None and not isinstance(random_state, int):
            raise TypeError('The input "random_state" must be a integer or None.')
        self.random_state = random_state