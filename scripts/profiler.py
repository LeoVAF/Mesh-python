from mesh.core import Mesh, MeshParameters

from problems.benchmark_problems import get_problem
from problems.microgrid_function import microgrid_function

import numpy as np
import cProfile
import pstats


objective_dim = 10
decision_dim = 30
max_iterations = None
max_fitness_eval = 50000
population_size = 2048

random_state = 42

position_min_value = np.array([0]*decision_dim)
position_max_value = np.array([1]*decision_dim)
num_final_solutions = population_size
memory_size = population_size
communication_probability = 0.7
mutation_rate = 0.9
personal_guide_array_size = 3
global_best_attribution_type = 0
dm_pool_type = 0
dm_operation_type = 0
crowding_distance_type = 0
optimization_type = [False]*objective_dim

# def generate_objective_function(objective_dim):
#     def objective_function(position):
#         position = np.array(position)
#         objectives = []
#         for i in range(objective_dim):
#             if i % 3 == 0:
#                 obj = np.sum((position - (i + 1))**2)
#             elif i % 3 == 1:
#                 obj = np.sum(np.sin(position * (i + 1))**2)
#             else:
#                 obj = np.prod(1 + position / (i + 1))
#             obj /= (i + 1) * 10
#             objectives.append(obj)
#         return np.array(objectives)
#     return objective_function
# func = generate_objective_function(objective_dim)

func, position_min_value, position_max_value = get_problem('dtlz1', n_obj=objective_dim, n_var=decision_dim)

# select_bat = 0 # Lead_Acid(0) Li-ion(1) ZEBRA(2) NaS(3) NiCd(4) NiMH(5) RFV(6) ZnBr(7)
# position_min_value = np.array([10, 10, 10]) # Lower bound of problem [max PV generation, max WT generation , battery capacity]
# position_max_value = np.array([450, 450, 500]) # Upper bound of problem [max PV generation, max WT generation, battery capacity]
# load = np.genfromtxt('scripts/seasonal_data/load.txt')
# temperature = np.genfromtxt('scripts/seasonal_data/temperature.txt')
# solar_data = np.genfromtxt('scripts/seasonal_data/irradiance.txt')
# wind_data = np.genfromtxt('scripts/seasonal_data/wind.txt')
# bat_name = ['Lead_Acid', 'Li-ion', 'ZEBRA', 'NaS', 'NiCd', 'NiMH', 'RFV', 'ZnBr']
# experiment_name = bat_name[select_bat]
# def func(args):
#     return microgrid_function(args[0], args[1], args[2], select_bat, load, temperature, solar_data, wind_data)

def run_new():
    params = MeshParameters(objective_dim,
                             decision_dim, position_min_value, position_max_value, 
                             population_size, memory_size,
                             global_guide_method=global_best_attribution_type, dm_pool_type=dm_pool_type, dm_operation_type=dm_operation_type,
                             max_gen=max_iterations, max_fit_eval=max_fitness_eval,
                             max_personal_guides=personal_guide_array_size,
                             random_state=random_state)

    new_mesh = Mesh(params, func)
    new_mesh.run()

cProfile.run('run_new()', sort='time', filename="profile.prof")
stats = pstats.Stats('profile.prof')
stats.strip_dirs()
stats.sort_stats('time')
stats.print_stats(10)
