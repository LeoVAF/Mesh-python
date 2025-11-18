from mesh.core import *
from mesh.MESH_old import *

from problems.benchmark_problems import get_problem

import numpy as np
import cProfile
import pstats


objective_dim = 2
position_dim = 10
max_iterations = None
max_fitness_eval = 15000
population_size = 100

random_state = 42

position_min_value = np.array([0]*position_dim)
position_max_value = np.array([1]*position_dim)
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

def generate_objective_function(objective_dim):
    def objective_function(position):
        position = np.array(position)
        objectives = []
        for i in range(objective_dim):
            if i % 3 == 0:
                obj = np.sum((position - (i + 1))**2)
            elif i % 3 == 1:
                obj = np.sum(np.sin(position * (i + 1))**2)
            else:
                obj = np.prod(1 + position / (i + 1))
            obj /= (i + 1) * 10
            objectives.append(obj)
        return np.array(objectives)
    return objective_function
func = generate_objective_function(objective_dim)

func, position_min_value, position_max_value = get_problem('zdt2', n_obj=objective_dim, n_var=position_dim)

def run_new():
    params = MeshParameters(objective_dim,
                             position_dim, position_min_value, position_max_value, 
                             population_size, memory_size,
                             global_best_attribution_type, dm_pool_type, dm_operation_type,
                             communication_probability, mutation_rate,
                             max_gen=max_iterations, max_fit_eval=max_fitness_eval,
                             max_personal_guides=personal_guide_array_size,
                             random_state=random_state)

    new_mesh = Mesh(params, func)
    new_mesh.run()

def run_old():
    params_old = MESH_Params_old(objective_dim,
                                [False]*objective_dim,
                                max_iterations,
                                max_fitness_eval,
                                position_dim,
                                position_max_value, position_min_value,
                                population_size,memory_size,
                                0,
                                global_best_attribution_type,
                                dm_operation_type,
                                dm_pool_type,
                                crowding_distance_type,
                                communication_probability,
                                mutation_rate,
                                personal_guide_array_size)
    old_mesh = MESH_old(params_old, func)
    old_mesh.run()

cProfile.run('run_new()', sort='time', filename="profile.prof")
stats = pstats.Stats('profile.prof')
stats.strip_dirs()
stats.sort_stats('time')
stats.print_stats(10)