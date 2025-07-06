from mesh.core import Mesh
from mesh.parameters import MeshParameters

from tqdm import tqdm
from pathlib import Path
from pickle import dump
from pygmo import fast_non_dominated_sorting, select_best_N_mo

import numpy as np
import os

def get_tuned_parameters(file_name: str, file_folder: str) -> dict:
	dictionary = {}
	file_path = f'{file_folder}/{file_name}.txt'
	if os.path.exists(file_path):	
		with open(file_path, 'r') as f:
			for line in f:
				if line.strip():
					key, value = line.strip().split(':')
					dictionary[key.strip()] = float(value.strip())
	return dictionary

def dump_results(file_name: str, file_folder: str, results: dict, combined_pos: np.ndarray[np.float64, 2], combined_fit: np.ndarray[np.float64, 2], num_solutions: int) -> None:
	# Getting the unique solutions from all executions
	unique_combined_pos, unique_idxs = np.unique(combined_pos, axis=0, return_index=True)
	unique_combined_fit = combined_fit[unique_idxs]

	# Sorting the matrix Fit
	# Return: (non dominated front, domination list, domination counter, non domination ranks)
	if len(unique_combined_fit) == 1:
			ndf = [[0]]
	else:
			ndf, _, _, _ = fast_non_dominated_sorting(points=unique_combined_fit)
	n = min(len(ndf[0]), num_solutions)
	
	# Get the best indexes based on number of final solutions
	pareto_front = unique_combined_fit[ndf[0]]
	best_idx = select_best_N_mo(pareto_front, n)
	results['combined'] = (unique_combined_pos[ndf[0]][best_idx], pareto_front[best_idx])

	# Store the results in a file
	file_path = f'{file_folder}'
	Path(file_path).mkdir(parents=False, exist_ok=True)
	with open(f'{file_path}/{file_name}.pkl', 'wb') as file:
		dump(results, file)

def run_mesh(experiment: tuple, # Information to run the experiments (experiment name, experiment folder, fine tuning folder, number of runs, maximum fitness evaluations, population size, random seed)
			       problem: tuple, # Problem setup (fitness function, number of objectives, number of decision variables, lower bound array, upper bound array)
			       fixed_parameters: tuple, # MESH fixed parameters
			       tunable_parameters: tuple # MESH tunable parameters
						 ) -> str:

	# Get the experiment name and folder to store results
	experiment_configuration, experiment_folder, fine_tuning_folder, num_runs, max_fitness_eval, population_size, random_state = experiment

  # Get the problem
	fit_function, objective_dim, position_dim, lower_bound_array, upper_bound_array = problem

	# Get the fixed parameters
	memory_size, global_best_attribution_type, dm_pool_type, dm_operation_type = fixed_parameters

	# Get tunable parameters (check if the parameters was tuned)
	tuned_parameters_dict = get_tuned_parameters(experiment_configuration, fine_tuning_folder)
	communication_probability = tuned_parameters_dict['communication_probability'] if ('communication_probability' in tuned_parameters_dict) else tunable_parameters[0]
	mutation_rate = tuned_parameters_dict['mutation_rate'] if ('mutation_rate' in tuned_parameters_dict) else tunable_parameters[1]
	personal_guide_array_size = tuned_parameters_dict['personal_guide_array_size'] if ('personal_guide_array_size' in tuned_parameters_dict) else tunable_parameters[2]

	# Execute MESH
	results = {}
	combined_F = None
	combined_P = None
	for i in tqdm(range(num_runs)):
		params = MeshParameters(objective_dim,
								position_dim, lower_bound_array, upper_bound_array, 
								population_size,
								memory_size=memory_size,
								global_best_attribution_type=global_best_attribution_type,
								dm_pool_type=dm_pool_type,
								dm_operation_type=dm_operation_type,
								communication_probability=communication_probability,
								mutation_rate=mutation_rate,
								max_gen=None, max_fit_eval=max_fitness_eval,
								max_personal_guides=personal_guide_array_size,
								random_state=random_state)
		mesh = Mesh(params, fit_function)
		mesh.run()

		# Accumulates the results at each step
		Pos, Fit = mesh.get_results()
		results[i+1] = {"F":Fit, "P":Pos}
		if combined_F is None:
			combined_P = Pos
			combined_F = Fit
		else:
			combined_P = np.vstack((combined_P, Pos))
			combined_F = np.vstack((combined_F, Fit))
	
	# Store the results
	dump_results(experiment_configuration, experiment_folder, results, combined_P, combined_F, population_size)
	return f'{experiment_configuration} was successfully executed!'