import numpy as np
import timeit
import statistics
import cProfile
import pstats

from MESH import *
from MESH_old import *

from pymoo.problems import get_problem


objective_dim = 5
position_dim = 10
max_iterations = 0
max_fitness_eval = 20000
population_size = 500

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
de_mutation_type = 0
crowding_distance_type = 0
optimization_type = [False]*objective_dim
def generate_objective_function(objective_dim):
    def objective_function(position):
        position = np.array(position)
        objectives = []
        for i in range(objective_dim):
            # Alterna entre diferentes padrões para criar trade-offs
            if i % 3 == 0:
                # Função do tipo esfera com deslocamento
                obj = np.sum((position - (i + 1))**2)
            elif i % 3 == 1:
                # Função baseada em ondas (oscilação não-linear)
                obj = np.sum(np.sin(position * (i + 1))**2)
            else:
                # Função mista: produto entre posição e índice do objetivo
                obj = np.prod(1 + position / (i + 1))
            # Normaliza cada objetivo em relação ao índice
            obj /= (i + 1) * 10
            objectives.append(obj)
        return np.array(objectives)
    return objective_function
func = generate_objective_function(objective_dim)

def run_new():
    params = MeshParameters(objective_dim,
                             position_dim, position_max_value, position_min_value, 
                             population_size, memory_size,
                             global_best_attribution_type, dm_pool_type, de_mutation_type,
                             communication_probability, mutation_rate,
                             max_gen=max_iterations, max_fit_eval=max_fitness_eval,
                             max_personal_guides=personal_guide_array_size,
                             random_state=random_state)
    ##########################################################
    #experiment_name = 'zdt2'
    #func = get_problem(experiment_name, n_var=position_dim).evaluate
    ##########################################################
    new_mesh = MESH(params, func)
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
                                de_mutation_type,
                                dm_pool_type,
                                crowding_distance_type,
                                communication_probability,
                                mutation_rate,
                                personal_guide_array_size)
    old_mesh = MESH_old(params_old, func)
    old_mesh.run()


# print("Estatísticas para algoritmo original:")
# times = timeit.repeat("run_old()", globals=globals(), repeat=30, number=1)
# # Estatísticas
# mean_time = statistics.mean(times)
# std_dev_time = statistics.stdev(times)
# min_time = min(times)
# max_time = max(times)
# print(f"Tempo médio: {mean_time:.6f} segundos")
# print(f"Desvio padrão: {std_dev_time:.6f} segundos")
# print(f"Tempo mínimo: {min_time:.6f} segundos")
# print(f"Tempo máximo: {max_time:.6f} segundos\n\n")


print("Estatísticas para algoritmo otimizado:")
times = timeit.repeat("run_new()", globals=globals(), repeat=30, number=1)
# Estatísticas
mean_time = statistics.mean(times)
std_dev_time = statistics.stdev(times)
min_time = min(times)
max_time = max(times)
print(f"Tempo médio: {mean_time:.6f} segundos")
print(f"Desvio padrão: {std_dev_time:.6f} segundos")
print(f"Tempo mínimo: {min_time:.6f} segundos")
print(f"Tempo máximo: {max_time:.6f} segundos")

# cProfile.run('run_new()', sort='time', filename="profile.prof")
# stats = pstats.Stats('profile.prof')
# stats.strip_dirs()
# stats.sort_stats('time')
# stats.print_stats(10)