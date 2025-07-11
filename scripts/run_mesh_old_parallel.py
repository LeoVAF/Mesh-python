from mesh.MESH_old import MESH_old, MESH_Params_old

from pymoo.problems import get_problem
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from pickle import dump
from tqdm import tqdm
from pygmo import fast_non_dominated_sorting, select_best_N_mo

import numpy as np

Path("result").mkdir(parents=False, exist_ok=True)

def run_mesh_old(experiment_name,
						num_runs, # Number of executions
						position_dim, objective_dim, # Number of variables and dimension of objective space
						func, # Fitness Function
						global_best_attribution_type,
						dm_pool_type,
						dm_operation_type):

  objective_dim = 5
  position_dim = 10
  max_iterations = 0
  max_fitness_eval = 15000
  population_size = 50

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

  config = f"E{global_best_attribution_type+1}V{dm_pool_type+1}D{dm_operation_type+1}_{experiment_name}"
  print(f"Running E{global_best_attribution_type+1}V{dm_pool_type+1}D{dm_operation_type+1}-{experiment_name} on MG")

  result = {}
  combined_F = None
  combined_P = None
  for i in tqdm(range(num_runs)):
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
    old_mesh.log_memory = f"result/{config}-MG"
    old_mesh.run()
    # Read the results from the log files
    with open(old_mesh.log_memory+"-fit.txt", 'r') as file:
        fl = file.read().split("\n")[-2]
        Fit = np.array([v.split() for v in fl.split(",")], dtype=np.float64)
    with open(old_mesh.log_memory+"-pos.txt", 'r') as file:
        fl = file.read().split("\n")[-2]
        Pos = np.array([v.split() for v in fl.split(",")], dtype=np.float64)

    result[i+1] = {"F":Fit, "P":Pos}
    # Accumulates the results of all executions
    if combined_F is None:
        combined_F = Fit
        combined_P = Pos
    else:
        combined_F = np.vstack((combined_F, Fit))
        combined_P = np.vstack((combined_P, Pos))
  # Sorting the vector Fit
  # Return: (non dominated front, domination list, domination counter, non domination ranks)
  if len(combined_F) == 1:
    ndf = [[0]]
  else:
    ndf, _, _, _ = fast_non_dominated_sorting(points=combined_F)
  n = min(num_final_solutions, len(ndf[0]))
  # Get the best indexes based on number of final solutions
  best_idx = select_best_N_mo(combined_F, n)
  result['combined'] = (combined_P[best_idx], combined_F[best_idx])
  ########################### Possible critical section ###########################
  with open(f'result/{config}.pkl', 'wb') as file:
    dump(result, file)
	#################################################################################
  return None

def execute_with_parallelism(func, params_list, max_workers=4):
	results = []
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		# Send tasks for execution (a dict)
		future_to_params = {executor.submit(func, *params): params for params in params_list}
		# Collect results as tasks are completed
		for future in as_completed(future_to_params):
			params = future_to_params[future]
			try:
				result = future.result()
				print(f"Result for {params}: {result}")
				results.append((params, result))
			except Exception as e:
				print(f"Error when execute with {params}: {e}")
				results.append((params, None))
	return results

def list_of_funcs(func_name, position_dim, objective_dim):
	set1 = {'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6'}
	set2 = {'dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7'}
	if func_name.lower() in set1:
		return lambda x: get_problem(func_name.lower(), n_var=position_dim).evaluate(np.array(x))
	elif func_name.lower() in set2:
		return lambda x: get_problem(func_name.lower(), n_var=position_dim, n_obj=objective_dim).evaluate(np.array(x))
	else:
		raise ValueError

if __name__ == "__main__":
	# Parameter list
	mesh_exp = ['dtlz1'] #, 'dtlz2', 'dtlz4', 'dtlz7'] # ['zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6']
	mesh_runs = [1]
	mesh_pos_dim = [10]
	mesh_obj_dim = [2]
	mesh_global_best_type = [1] # 0 -> E1 | 1 -> E2 | 2 -> E3 | 3 -> E4
	mesh_dm_pool_type = [1] # 0 -> V1 | 1 -> V2 | 2 -> V3
	mesh_differential_evolution_type = [0] # 0 -> DE\rand\1\Bin (D1) | 1 -> DE\rand\2\Bin (D2) | 2 -> DE/Best/1/Bin (D3) | 3 -> DE/Current-to-best/1/Bin (D4) | 4 -> DE/Current-to-rand/1/Bin (D5)
	params_list = [
		[mf, runs, p_dim, obj_dim, list_of_funcs(mf, p_dim, obj_dim), gb_type, pool_type, de_type]
		for mf in mesh_exp
		for runs in mesh_runs
		for p_dim in mesh_pos_dim
		for obj_dim in mesh_obj_dim
		for gb_type in mesh_global_best_type
		for pool_type in mesh_dm_pool_type
		for de_type in mesh_differential_evolution_type
	]

	# Execute in parallel
	workers = 1
	resultados = execute_with_parallelism(run_mesh_old, params_list, max_workers=workers)