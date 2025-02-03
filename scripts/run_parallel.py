import numpy as np
import pickle

from MESH import *

from pymoo.problems import get_problem
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

Path("result").mkdir(parents=False, exist_ok=True)

def run_mesh(experiment_name,
						num_runs, # Number of executions
						position_dim, objective_dim, # Number of variables and dimension of objective space
						func, # Fitness Function
						global_best_attribution_type,
						Xr_pool_type,
						DE_mutation_type):

	position_min_value = np.array([0]*position_dim) # Lower bound of problem [max PV generation, number of wind turbines, battery capacity]
	position_max_value = np.array([1]*position_dim) # Upper bound of problem [max PV generation, number of wind turbines, battery capacity]
	max_iterations = 0 # Maximum number of iterations (not used if it less than one)
	max_fitness_eval = 5000 # Maximum fitness evaluations (not used if it is less than one)
	population_size = 100 # Population size
	num_final_solutions = population_size # Number of final solutions
	memory_size = population_size # Maximum number of particles in memory
	communication_probability = 0.7 # Communication probability
	mutation_rate = 0.9 # Mutation rate
	personal_guide_array_size = 3 # Number of personal guides
	random_state = None # Defines a seed for random numbers (not used if it is None)

	config = f"E{global_best_attribution_type+1}V{Xr_pool_type+1}D{DE_mutation_type+1}_{experiment_name}"
	print(f"Running E{global_best_attribution_type+1}V{Xr_pool_type+1}D{DE_mutation_type+1}-{experiment_name} on MG")

	result = {}
	combined_F = None
	combined_P = None
	for i in tqdm(range(num_runs)):
		params = MESH_Params(objective_dim,
							position_dim, position_max_value, position_min_value, 
							population_size, memory_size=memory_size,
							global_best_attribution_type=global_best_attribution_type,
							de_mutation_type=DE_mutation_type,
							dm_pool_type=Xr_pool_type,
							communication_probability=communication_probability, mutation_rate=mutation_rate,
							max_gen=max_iterations, max_fit_eval=max_fitness_eval,
							max_personal_guides=personal_guide_array_size,
							random_state=random_state)
		log = False
		MCDEEPSO = MESH(params, func, log_memory=log)
		MCDEEPSO.run()
		Pos, Fit = MCDEEPSO.get_results()
		result[i+1] = {"F":Fit, "P":Pos}
		# Accumulates the results of all executions
		if combined_F is None:
			combined_P = Pos
			combined_F = Fit
		else:
			combined_P = np.vstack((combined_P, Pos))
			combined_F = np.vstack((combined_F, Fit))
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
		pickle.dump(result, file)
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
	set2 = {'dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6'}
	if func_name.lower() in set1:
		return get_problem(func_name.lower(), n_var=position_dim).evaluate
	elif func_name.lower() in set2:
		return get_problem(func_name.lower(), n_var=position_dim, n_obj=objective_dim).evaluate
	else:
		raise ValueError

if __name__ == "__main__":
	# Parameters list
	mesh_exp = ['zdt2'] # ['zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6']
	mesh_runs = [1]
	mesh_pos_dim = [10]
	mesh_obj_dim = [2]
	mesh_global_best_type = [0,1,2,3] # 0 -> E1 | 1 -> E2 | 2 -> E3 | 3 -> E4
	mesh_xr_pool_type = [0,1,2] # 0 -> V1 | 1 -> V2 | 2 -> V3
	mesh_differential_evolution_type = [0,1,2,3,4] # 0 -> DE\rand\1\Bin (D1) | 1 -> DE\rand\2\Bin (D2) | 2 -> DE/Best/1/Bin (D3) | 3 -> DE/Current-to-best/1/Bin (D4) | 4 -> DE/Current-to-rand/1/Bin (D5)
	params_list = [
		[mf, runs, p_dim, obj_dim, list_of_funcs(mf, p_dim, obj_dim), gb_type, pool_type, de_type]
		for mf in mesh_exp
		for runs in mesh_runs
		for p_dim in mesh_pos_dim
		for obj_dim in mesh_obj_dim
		for gb_type in mesh_global_best_type
		for pool_type in mesh_xr_pool_type
		for de_type in mesh_differential_evolution_type
	]

	# Execute in parallel
	workers = 16
	resultados = execute_with_parallelism(run_mesh, params_list, max_workers=workers)