from mesh.core import *
from mesh.parameters import MeshParameters
from microgrid_old.techno_ka import techno_ka
from problems import get_problem

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from pickle import dump
from functools import partial

import numpy as np

Path("result").mkdir(parents=False, exist_ok=True)

def run_mesh(experiment_name,
						num_runs, # Number of executions
						position_dim, objective_dim, # Number of variables and dimension of objective space
						problem, # Tuple with function to be optimized and the limits of the design space
						global_guide_method,
						dm_pool_type,
						dm_operation_type):

	num_final_solutions = 300
	func, position_min_value, position_max_value = problem
	max_iterations = None # Maximum number of iterations (not used if it less than one)
	max_fitness_eval = 15000 # Maximum fitness evaluations (not used if it is less than one)
	population_size = 100 # Population size
	memory_size = population_size # Maximum number of particles in memory
	communication_probability = 0.7 # Communication probability
	mutation_rate = 0.4 # Mutation rate
	personal_guide_array_size = 3 # Number of personal guides
	random_state = None # Defines a seed for random numbers (not used if it is None)

	config = f"E{global_guide_method+1}V{dm_pool_type+1}D{dm_operation_type+1}_{experiment_name}"
	print(f"Running E{global_guide_method+1}V{dm_pool_type+1}D{dm_operation_type+1}-{experiment_name} on MG")

	result = {}
	combined_F = None
	combined_P = None
	for i in tqdm(range(num_runs)):
		params = MeshParameters(objective_dim,
							position_dim, position_min_value, position_max_value, 
							population_size, memory_size=memory_size,
							global_guide_method=global_guide_method,
							dm_pool_type=dm_pool_type,
							dm_operation_type=dm_operation_type,
							communication_probability=communication_probability, mutation_rate=mutation_rate,
							max_gen=max_iterations, max_fit_eval=max_fitness_eval,
							max_personal_guides=personal_guide_array_size,
							random_state=random_state)
		log = None
		mesh = Mesh(params, func, log_memory=log)
		mesh.run()
		Pos, Fit = mesh.get_results()
		result[i+1] = {"F":Fit, "P":Pos}
		# Accumulates the results of all executions
		if combined_F is None:
			combined_P = Pos
			combined_F = Fit
		else:
			combined_P = np.vstack((combined_P, Pos))
			combined_F = np.vstack((combined_F, Fit))
		# Getting the unique points
		unique_combined_P, unique_idxs = np.unique(combined_P, axis=0, return_index=True)
		unique_combined_F = combined_F[unique_idxs]
		# Sorting the vector Fit
		# Return: (non dominated front, domination list, domination counter, non domination ranks)
		if len(unique_combined_F) == 1:
				ndf = [[0]]
		else:
				ndf, _, _, _ = fast_non_dominated_sorting(points=unique_combined_F)
		n = min(len(ndf[0]), num_final_solutions)
		# Get the best indexes based on number of final solutions
		pareto_front = unique_combined_F[ndf[0]]
		best_idx = select_best_N_mo(pareto_front, n)
		result['combined'] = (unique_combined_P[ndf[0]][best_idx], pareto_front[best_idx])
		with open(f'./scripts/results/{config}.pkl', 'wb') as file:
				dump(result, file)
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

#################### Microgrid function #######################
Path("result").mkdir(parents=False, exist_ok=True)
solar_data = np.genfromtxt('scripts/microgrid_old/seasonal_data/solreal.txt')
wind_data = np.genfromtxt('scripts/microgrid_old/seasonal_data/wind_data.txt')
load_ind = np.genfromtxt('scripts/microgrid_old/seasonal_data/loadind.txt')
load_res = np.genfromtxt('scripts/microgrid_old/seasonal_data/loadres.txt')
def microgrid_func(args, bat_number, objective_dim):
	r = techno_ka(args[0], args[1], 0.8, args[2], bat_number, solar_data, wind_data, load_ind)[:objective_dim]
	#r = techno_ka(args[0], args[1], 0.8, args[2], select_bat, solar_data, wind_data, load_ind)[1:3]
	r[-1] = -r[-1] # Maximizing renewable factor
	return r
################################################################

def list_of_problems(func_name, position_dim, objective_dim):
	microgrid_dict = {'LAG':0, 'LTO':1, 'LCO':2, 'LFP':3, 'LMO':4, 'LNCMO':5, 'LNCAO':6, 'LPoly':7, 'NNC':8, 'NaS':9, 'NiC':10, 'NMH':11, 'RFV':12, 'ZnBr':13}
	if func_name in microgrid_dict:
		return partial(microgrid_func, bat_number=microgrid_dict[func_name], objective_dim=objective_dim), np.array([10, 1, 50]), np.array([500, 5, 500])
	else:
		func, position_min_value, position_max_value = get_problem(func_name, n_var=position_dim, n_obj=objective_dim)
		return func, position_min_value, position_max_value

if __name__ == "__main__":
	# Parameter list
	# mesh_exp = ['LAG', 'LTO', 'LCO', 'LFP', 'LMO', 'LNCMO', 'LNCAO', 'LPoly', 'NNC', 'NaS', 'NiC', 'NMH', 'RFV', 'ZnBr']
	mesh_exp = ['zdt1'] # ['dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7', 'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6']
	mesh_runs = [30]
	mesh_obj_dim = [2]
	mesh_pos_dim = [30]
	mesh_global_best_type = [0,1] # 0 -> E1 | 1 -> E2 | 2 -> E3 | 3 -> E4
	mesh_dm_pool_type = [0,1,2] # 0 -> V1 | 1 -> V2 | 2 -> V3
	mesh_differential_evolution_type = [0,1,2,3,4] # 0 -> DE\rand\1\Bin (D1) | 1 -> DE\rand\2\Bin (D2) | 2 -> DE/Best/1/Bin (D3) | 3 -> DE/Current-to-best/1/Bin (D4) | 4 -> DE/Current-to-rand/1/Bin (D5)
	params_list = [
		[mf, runs, p_dim, obj_dim, list_of_problems(mf, p_dim, obj_dim), gb_type, pool_type, de_type]
		for mf in mesh_exp
		for runs in mesh_runs
		for p_dim in mesh_pos_dim
		for obj_dim in mesh_obj_dim
		for gb_type in mesh_global_best_type
		for pool_type in mesh_dm_pool_type
		for de_type in mesh_differential_evolution_type
	]

	# Execute in parallel
	workers = 22
	resultados = execute_with_parallelism(run_mesh, params_list, max_workers=workers)