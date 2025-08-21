from mesh.core import Mesh
from mesh.parameters import MeshParameters
from mesh.MESH_old import MESH_old, MESH_Params_old

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize

from pathlib import Path
from pickle import dump
from pygmo import fast_non_dominated_sorting, select_best_N_mo
from tqdm import tqdm

import numpy as np
import os

def get_tuned_parameters(file_name: str, file_folder: str) -> dict:
	dictionary = {}
	file_path = f"{file_folder}/{file_name}.txt"
	if os.path.exists(file_path):
		with open(file_path, 'r', encoding='utf-8') as f:
			for line in f:
				if line.strip():
					# Divide into "key" and "value (type)"
					key, value_and_type = line.strip().split(':', 1)
					key = key.strip()
					value_and_type = value_and_type.strip()
					# Separates value and type — expects type to be enclosed in parentheses at the end
					value_part, type_part = value_and_type.rsplit('(', 1)
					value = value_part.strip()
					value_type = type_part.strip(') ').lower()
					# Convert the value to the correct type
					if value_type == 'float':
						value = float(value)
					elif value_type == 'int':
						value = int(value)
					elif value_type == 'bool':
						value = value == 'True'
					dictionary[key] = value
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

def run_mesh(experiment: tuple, # Information to run the experiments
						 										# (experiment name, experiment folder, fine tuning folder, number of runs, maximum fitness evaluations, population size, random seed)
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
	for i in range(num_runs):
		params = MeshParameters(objective_dim = objective_dim,
								position_dim = position_dim,
								position_lower_bounds = lower_bound_array,
								position_upper_bounds = upper_bound_array, 
								population_size = population_size,
								memory_size = memory_size,
								global_guide_method = global_best_attribution_type,
								dm_pool_type = dm_pool_type,
								dm_operation_type = dm_operation_type,
								communication_probability = communication_probability,
								mutation_rate = mutation_rate,
								max_gen = None,
								max_fit_eval = max_fitness_eval,
								max_personal_guides = personal_guide_array_size,
								random_state = random_state)
		mesh = Mesh(params = params, fitness_function = fit_function)
		mesh.run()

		# Accumulate the results at each step
		Pos, Fit = mesh.get_results()
		results[i+1] = {"P":Pos, "F":Fit,}
		if combined_F is None:
			combined_P = Pos
			combined_F = Fit
		else:
			combined_P = np.vstack((combined_P, Pos))
			combined_F = np.vstack((combined_F, Fit))
	
	# Store the results
	dump_results(experiment_configuration, experiment_folder, results, combined_P, combined_F, population_size)
	return f'{experiment_configuration} with tunable parameters ({communication_probability}, {mutation_rate}, {personal_guide_array_size}) was successfully executed!'

def run_mesh_old(experiment: tuple, # Information to run the experiments
								 										# (experiment name, experiment folder, fine tuning folder, number of runs, maximum fitness evaluations, population size, random seed)
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

	results = {}
	combined_F = None
	combined_P = None
	for i in range(num_runs):
		params_old = MESH_Params_old(objectives_dim = objective_dim,
																 optimizations_type = [False]*objective_dim,
																 max_iterations = 0,
																 max_fitness_eval = max_fitness_eval,
																 position_dim = position_dim,
																 position_max_value = upper_bound_array,
																 position_min_value = lower_bound_array,
																 population_size = population_size,
																 memory_size = memory_size,
																 memory_update_type = 0,
																 global_best_attribution_type = global_best_attribution_type,
																 DE_mutation_type = dm_operation_type,
																 Xr_pool_type = dm_pool_type,
																 crowd_distance_type = 0,
																 communication_probability = communication_probability,
																 mutation_rate = mutation_rate,
																 personal_guide_array_size = personal_guide_array_size,
																 random_state=random_state)
		old_mesh = MESH_old(params_old, fit_function)
		old_mesh.log_memory = None
		Pos, Fit = old_mesh.run()

		# Accumulate the results at each step
		results[i+1] = {"P":Pos, "F":Fit,}
		if combined_F is None:
			combined_P = Pos
			combined_F = Fit
		else:
			combined_P = np.vstack((combined_P, Pos))
			combined_F = np.vstack((combined_F, Fit))

	# Store the results
	dump_results(experiment_configuration, experiment_folder, results, combined_P, combined_F, population_size)
	return f'{experiment_configuration} with tunable parameters ({communication_probability}, {mutation_rate}, {personal_guide_array_size}) was successfully executed!'

def run_nsga2(experiment: tuple, # Information to run the experiments
																 # (experiment name, experiment folder, fine tuning folder, number of runs, maximum fitness evaluations, population size, random seed)
			       problem: tuple, # Problem setup (fitness function, number of objectives, number of decision variables, lower bound array, upper bound array)
			       fixed_parameters: tuple, # MESH fixed parameters
			       tunable_parameters: tuple # MESH tunable parameters
						) -> str:

	# Get the experiment name and folder to store results
	experiment_configuration, experiment_folder, fine_tuning_folder, num_runs, max_fitness_eval, population_size, random_state = experiment

  # Get the problem
	fit_function, objective_dim, position_dim, lower_bound_array, upper_bound_array = problem
	class MyProblem(Problem):
		def __init__(self, n_var, n_obj, xl, xu):
			super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu)
		def _evaluate(self, X, out, *args, **kwargs):
			out["F"] = np.array([fit_function(x) for x in X])

	nsga2_fit_function = MyProblem(n_obj=objective_dim, n_var=position_dim, xl=lower_bound_array, xu=upper_bound_array)

	# Get tunable parameters (check if the parameters was tuned)
	tuned_parameters_dict = get_tuned_parameters(experiment_configuration, fine_tuning_folder)
	recombination_probability = tuned_parameters_dict['recombination_probability'] if ('recombination_probability' in tuned_parameters_dict) else tunable_parameters[0]
	eta_recombination = tuned_parameters_dict['eta_recombination'] if ('eta_recombination' in tuned_parameters_dict) else tunable_parameters[1]
	mutation_probability = tuned_parameters_dict['mutation_probability'] if ('mutation_probability' in tuned_parameters_dict) else tunable_parameters[2]
	eta_mutation = tuned_parameters_dict['eta_mutation'] if ('eta_mutation' in tuned_parameters_dict) else tunable_parameters[3]

	# Instantiate NSGA2
	crossover = SBX(prob=recombination_probability, prob_var=1.0, eta=eta_recombination)
	mutation = PolynomialMutation(prob=mutation_probability, eta=eta_mutation)
	nsga2 = NSGA2(pop_size=population_size,
				crossover=crossover,
				mutation=mutation,
				eliminate_duplicates=True)

	# Execute NSGA2
	results = {}
	combined_F = None
	combined_P = None
	for i in range(num_runs):
		res = minimize(nsga2_fit_function,
                	 nsga2,
                	 ('n_eval', max_fitness_eval),
									 seed=random_state,
                	 verbose=False)

		# Accumulate the results at each step
		Pos, Fit = res.X, res.F
		results[i+1] = {"P":Pos, "F":Fit,}
		if combined_F is None:
			combined_P = Pos
			combined_F = Fit
		else:
			combined_P = np.vstack((combined_P, Pos))
			combined_F = np.vstack((combined_F, Fit))
	
	# Store the results
	dump_results(experiment_configuration, experiment_folder, results, combined_P, combined_F, population_size)
	return f'{experiment_configuration} with tunable parameters ({recombination_probability}, {eta_recombination}, {mutation_probability}, {eta_mutation}) was successfully executed!'