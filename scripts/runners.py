from mesh import Mesh, MeshParameters

from pymoo.algorithms.moo.cmopso import CMOPSO
from pymoo.algorithms.moo.mopso_cd import MOPSO_CD
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize

from numpy.typing import NDArray
from pathlib import Path
from pickle import dump
from pygmo import fast_non_dominated_sorting, select_best_N_mo # type: ignore
from typing import Any

import numpy as np
import os
import pygmo as pg


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
					elif value_type == 'str':
						value = str(value)
					dictionary[key] = value
	return dictionary


def dump_results(file_name: str,
				 file_folder: str,
				 results: dict,
				 combined_pos: NDArray[np.number],
				 combined_fit: NDArray[np.number],
				 num_solutions: int) -> None:
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


def run_cmopso(experiment: dict[str, Any],
			  problem: dict[str, Any],
			  parameters: dict[str, Any]) -> str:
	# Get the experiment configuration
	experiment_name = experiment['name']
	results_folder = experiment['results_folder']
	fine_tuning_folder = experiment['fine_tuning_folder']
	num_runs = experiment['num_runs']
	max_fitness_eval = experiment['max_fitness_eval']
	population_size = experiment['population_size']
	random_state = experiment['random_state']
    # Get the problem configuration
	fitness = problem['fitness']
	objective_dim = problem['objective_dim']
	position_dim = problem['position_dim']
	lower_bound_array = problem['lower_bound_array']
	upper_bound_array = problem['upper_bound_array']
	class MyProblem(Problem):
		def __init__(self, n_var, n_obj, xl, xu):
			super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu)
		def _evaluate(self, X, out, *args, **kwargs):
			out["F"] = np.array([fitness(x) for x in X])
	pymoo_fitness = MyProblem(n_obj=objective_dim, n_var=position_dim, xl=lower_bound_array, xu=upper_bound_array)

	# Get tunable parameters (check if the parameters were tuned)
	tuned_parameters_dict = get_tuned_parameters(experiment_name, fine_tuning_folder)
	max_velocity_rate = tuned_parameters_dict['max_velocity_rate'] if ('max_velocity_rate' in tuned_parameters_dict) else parameters['max_velocity_rate']
	elite_size = tuned_parameters_dict['elite_size'] if ('elite_size' in tuned_parameters_dict) else parameters['elite_size']
	initial_velocity = tuned_parameters_dict['initial_velocity'] if ('initial_velocity' in tuned_parameters_dict) else parameters['initial_velocity']
	mutate_rate = tuned_parameters_dict['mutate_rate'] if ('mutate_rate' in tuned_parameters_dict) else parameters['mutate_rate']
	# Instantiate CMOPSO
	cmopso = CMOPSO(pop_size=population_size,
					max_velocity_rate=max_velocity_rate,
					elite_size=elite_size,
					initial_velocity=initial_velocity,
					mutate_rate=mutate_rate,
					sampling=LHS(), # type: ignore
					eliminate_duplicates=True,
					seed=random_state)
	# Execute CMOPSO
	results = {}
	combined_F = np.empty((0, objective_dim))
	combined_P = np.empty((0, position_dim))
	for i in range(num_runs):
		res = minimize(pymoo_fitness,
                	   cmopso,
                	   ('n_eval', max_fitness_eval),
					   seed=random_state,
                	   verbose=False)
		# Accumulate the results at each step
		Pos, Fit = np.array(res.X), np.array(res.F)
		results[i+1] = {"P":Pos, "F":Fit,}
		combined_P = np.vstack((combined_P, Pos))
		combined_F = np.vstack((combined_F, Fit))
	# Store the results
	dump_results(experiment_name, results_folder, results, combined_P, combined_F, population_size)
	return f'{experiment_name} with tunable parameters ({max_velocity_rate}, {elite_size}, {initial_velocity}, {mutate_rate}) was successfully executed!'


def run_maco(experiment: dict[str, Any],
			  problem: dict[str, Any],
			  parameters: dict[str, Any]) -> str:
	# Get the experiment configuration
	experiment_name = experiment['name']
	results_folder = experiment['results_folder']
	fine_tuning_folder = experiment['fine_tuning_folder']
	num_runs = experiment['num_runs']
	max_fitness_eval = experiment['max_fitness_eval']
	population_size = experiment['population_size']
	random_state = experiment['random_state']
    # Get the problem configuration
	fitness_function = problem['fitness']
	objective_dim = problem['objective_dim']
	position_dim = problem['position_dim']
	lower_bound_array = problem['lower_bound_array']
	upper_bound_array = problem['upper_bound_array']
	class PygmoProblem:
		def fitness(self, x):
			return fitness_function(x)
		def get_bounds(self):
			return (lower_bound_array, upper_bound_array)
		def get_nobj(self):
			return objective_dim
	pygmo_problem = PygmoProblem()

	# Get tunable parameters (check if the parameters were tuned)
	tuned_parameters_dict = get_tuned_parameters(experiment_name, fine_tuning_folder)
	ker = tuned_parameters_dict['ker'] if ('ker' in tuned_parameters_dict) else parameters['ker']
	q = tuned_parameters_dict['q'] if ('q' in tuned_parameters_dict) else parameters['q']
	threshold = tuned_parameters_dict['threshold'] if ('threshold' in tuned_parameters_dict) else parameters['threshold']
	n_gen_mark = tuned_parameters_dict['n_gen_mark'] if ('n_gen_mark' in tuned_parameters_dict) else parameters['n_gen_mark']
	focus = tuned_parameters_dict['focus'] if ('focus' in tuned_parameters_dict) else parameters['focus']
	
	# Execute NSPSO
	results = {}
	combined_F = np.empty((0, objective_dim))
	combined_P = np.empty((0, position_dim))
	for i in range(num_runs):
		random_state = experiment['random_state'] if experiment['random_state'] is not None else np.random.randint(0, 2147483647)
		initial_population = pg.population(pygmo_problem, size=population_size, seed=random_state) # type: ignore
		maco = pg.algorithm(pg.maco(gen=max_fitness_eval // population_size, # type: ignore
									ker=ker,
									q=q,
									threshold=threshold,
									n_gen_mark=n_gen_mark,
									focus=focus,
									seed=random_state))
		evolved_population = maco.evolve(initial_population)
		# Accumulate the results at each step
		Pos, Fit = np.array(evolved_population.get_x()), np.array(evolved_population.get_f())
		results[i+1] = {"P":Pos, "F":Fit,}
		combined_P = np.vstack((combined_P, Pos))
		combined_F = np.vstack((combined_F, Fit))
	# Store the results
	dump_results(experiment_name, results_folder, results, combined_P, combined_F, population_size)
	return f'{experiment_name} with tunable parameters ({ker}, {q}, {threshold}, {n_gen_mark}, {focus}) was successfully executed!'


def run_mesh(experiment: dict[str, Any],
			 problem: dict[str, Any],
			 parameters: dict[str, Any]) -> str:
	# Get the experiment configuration
	experiment_name = experiment['name']
	results_folder = experiment['results_folder']
	fine_tuning_folder = experiment['fine_tuning_folder']
	num_runs = experiment['num_runs']
	max_fitness_eval = experiment['max_fitness_eval']
	population_size = experiment['population_size']
	random_state = experiment['random_state']
    # Get the problem configuration
	fitness = problem['fitness']
	objective_dim = problem['objective_dim']
	position_dim = problem['position_dim']
	lower_bound_array = problem['lower_bound_array']
	upper_bound_array = problem['upper_bound_array']

	# Get the fixed parameters
	memory_size = parameters['memory_size']
	# Get tunable parameters (check if the parameters were tuned)
	tuned_parameters = get_tuned_parameters(experiment_name, fine_tuning_folder)
	global_best_attribution_type = tuned_parameters['global_best_attribution_type'] if ('global_best_attribution_type' in tuned_parameters) else parameters['global_best_attribution_type']
	dm_pool_type = tuned_parameters['differential_mutation_pool_type'] if ('differential_mutation_pool_type' in tuned_parameters) else parameters['differential_mutation_pool_type']
	dm_operation_type = tuned_parameters['differential_mutation_type'] if ('differential_mutation_type' in tuned_parameters) else parameters['differential_mutation_type']
	communication_probability = tuned_parameters['communication_probability'] if ('communication_probability' in tuned_parameters) else parameters['communication_probability']
	mutation_rate = tuned_parameters['mutation_rate'] if ('mutation_rate' in tuned_parameters) else parameters['mutation_rate']
	personal_guide_array_size = tuned_parameters['personal_guide_array_size'] if ('personal_guide_array_size' in tuned_parameters) else parameters['personal_guide_array_size']

	# Execute MESH
	results = {}
	combined_F = np.empty((0, objective_dim))
	combined_P = np.empty((0, position_dim))
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
		mesh = Mesh(params = params, fitness_function = fitness)
		mesh.run()

		# Accumulate the results at each step
		Pos, Fit = mesh.get_results()
		results[i+1] = {"P":Pos, "F":Fit,}
		combined_P = np.vstack((combined_P, Pos))
		combined_F = np.vstack((combined_F, Fit))
	
	# Store the results
	dump_results(experiment_name, results_folder, results, combined_P, combined_F, population_size)
	return f'{experiment_name} with tunable parameters ({communication_probability}, {mutation_rate}, {personal_guide_array_size}) was successfully executed!'


def run_mopso_cd(experiment: dict[str, Any],
				 problem: dict[str, Any],
				 parameters: dict[str, Any]) -> str:
	# Get the experiment configuration
	experiment_name = experiment['name']
	results_folder = experiment['results_folder']
	fine_tuning_folder = experiment['fine_tuning_folder']
	num_runs = experiment['num_runs']
	max_fitness_eval = experiment['max_fitness_eval']
	population_size = experiment['population_size']
	random_state = experiment['random_state']
    # Get the problem configuration
	fitness = problem['fitness']
	objective_dim = problem['objective_dim']
	position_dim = problem['position_dim']
	lower_bound_array = problem['lower_bound_array']
	upper_bound_array = problem['upper_bound_array']
	class MyProblem(Problem):
		def __init__(self, n_var, n_obj, xl, xu):
			super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu)
		def _evaluate(self, X, out, *args, **kwargs):
			out["F"] = np.array([fitness(x) for x in X])

	pymoo_fitness = MyProblem(n_obj=objective_dim, n_var=position_dim, xl=lower_bound_array, xu=upper_bound_array)
	# Get tunable parameters (check if the parameters were tuned)
	tuned_parameters_dict = get_tuned_parameters(experiment_name, fine_tuning_folder)
	w = tuned_parameters_dict['w'] if ('w' in tuned_parameters_dict) else parameters['w']
	c1 = tuned_parameters_dict['c1'] if ('c1' in tuned_parameters_dict) else parameters['c1']
	c2 = tuned_parameters_dict['c2'] if ('c2' in tuned_parameters_dict) else parameters['c2']
	max_velocity_rate = tuned_parameters_dict['max_velocity_rate'] if ('max_velocity_rate' in tuned_parameters_dict) else parameters['max_velocity_rate']
	archive_size = tuned_parameters_dict['archive_size'] if ('archive_size' in tuned_parameters_dict) else parameters['archive_size']
	# Instantiate MOPSO-CD
	mopso_cd = MOPSO_CD(pop_size=population_size,
						w=w,
						c1=c1,
						c2=c2,
						max_velocity_rate=max_velocity_rate,
						archive_size=archive_size,
						sampling=LHS(), # type: ignore
						eliminate_duplicates=True,
						seed=random_state)
	# Execute MOPSO-CD
	results = {}
	combined_F = np.empty((0, objective_dim))
	combined_P = np.empty((0, position_dim))
	for i in range(num_runs):
		res = minimize(pymoo_fitness,
                	   mopso_cd,
                	   ('n_eval', max_fitness_eval),
					   seed=random_state,
                	   verbose=False)
		# Accumulate the results at each step
		Pos, Fit = np.array(res.X), np.array(res.F)
		results[i+1] = {"P":Pos, "F":Fit,}
		combined_P = np.vstack((combined_P, Pos))
		combined_F = np.vstack((combined_F, Fit))
	# Store the results
	dump_results(experiment_name, results_folder, results, combined_P, combined_F, population_size)
	return f'{experiment_name} with tunable parameters ({w}, {c1}, {c2}, {max_velocity_rate}, {archive_size}) was successfully executed!'


def run_nsga2(experiment: dict[str, Any],
			  problem: dict[str, Any],
			  parameters: dict[str, Any]) -> str:
	# Get the experiment configuration
	experiment_name = experiment['name']
	results_folder = experiment['results_folder']
	fine_tuning_folder = experiment['fine_tuning_folder']
	num_runs = experiment['num_runs']
	max_fitness_eval = experiment['max_fitness_eval']
	population_size = experiment['population_size']
	random_state = experiment['random_state']
    # Get the problem configuration
	fitness_function = problem['fitness']
	objective_dim = problem['objective_dim']
	position_dim = problem['position_dim']
	lower_bound_array = problem['lower_bound_array']
	upper_bound_array = problem['upper_bound_array']
	class PygmoProblem:
		def fitness(self, x):
			return fitness_function(x)
		def get_bounds(self):
			return (lower_bound_array, upper_bound_array)
		def get_nobj(self):
			return objective_dim
	pygmo_problem = PygmoProblem()

	# Get tunable parameters (check if the parameters were tuned)
	tuned_parameters_dict = get_tuned_parameters(experiment_name, fine_tuning_folder)
	recombination_probability = tuned_parameters_dict['recombination_probability'] if ('recombination_probability' in tuned_parameters_dict) else parameters['recombination_probability']
	eta_recombination = tuned_parameters_dict['eta_recombination'] if ('eta_recombination' in tuned_parameters_dict) else parameters['eta_recombination']
	mutation_probability = tuned_parameters_dict['mutation_probability'] if ('mutation_probability' in tuned_parameters_dict) else parameters['mutation_probability']
	eta_mutation = tuned_parameters_dict['eta_mutation'] if ('eta_mutation' in tuned_parameters_dict) else parameters['eta_mutation']
	
	# Execute NSPSO
	results = {}
	combined_F = np.empty((0, objective_dim))
	combined_P = np.empty((0, position_dim))
	for i in range(num_runs):
		random_state = experiment['random_state'] if experiment['random_state'] is not None else np.random.randint(0, 2147483647)
		initial_population = pg.population(pygmo_problem, size=population_size, seed=random_state) # type: ignore
		nsga2 = pg.algorithm(pg.nsga2(gen=max_fitness_eval // population_size, # type: ignore
									  cr=recombination_probability,
									  eta_c=eta_recombination,
									  m=mutation_probability,
									  eta_m=eta_mutation,
									  seed=random_state))
		evolved_population = nsga2.evolve(initial_population)
		# Accumulate the results at each step
		Pos, Fit = np.array(evolved_population.get_x()), np.array(evolved_population.get_f())
		results[i+1] = {"P":Pos, "F":Fit,}
		combined_P = np.vstack((combined_P, Pos))
		combined_F = np.vstack((combined_F, Fit))
	# Store the results
	dump_results(experiment_name, results_folder, results, combined_P, combined_F, population_size)
	return f'{experiment_name} with tunable parameters ({recombination_probability}, {eta_recombination}, {mutation_probability}, {eta_mutation}) was successfully executed!'


def run_nspso(experiment: dict[str, Any],
			  problem: dict[str, Any],
			  parameters: dict[str, Any]) -> str:
	# Get the experiment configuration
	experiment_name = experiment['name']
	results_folder = experiment['results_folder']
	fine_tuning_folder = experiment['fine_tuning_folder']
	num_runs = experiment['num_runs']
	max_fitness_eval = experiment['max_fitness_eval']
	population_size = experiment['population_size']
	random_state = experiment['random_state']
    # Get the problem configuration
	fitness_function = problem['fitness']
	objective_dim = problem['objective_dim']
	position_dim = problem['position_dim']
	lower_bound_array = problem['lower_bound_array']
	upper_bound_array = problem['upper_bound_array']
	class PygmoProblem:
		def fitness(self, x):
			return fitness_function(x)
		def get_bounds(self):
			return (lower_bound_array, upper_bound_array)
		def get_nobj(self):
			return objective_dim
	pygmo_problem = PygmoProblem()

	# Get tunable parameters (check if the parameters were tuned)
	tuned_parameters_dict = get_tuned_parameters(experiment_name, fine_tuning_folder)
	omega = tuned_parameters_dict['omega'] if ('omega' in tuned_parameters_dict) else parameters['omega']
	c1 = tuned_parameters_dict['c1'] if ('c1' in tuned_parameters_dict) else parameters['c1']
	c2 = tuned_parameters_dict['c2'] if ('c2' in tuned_parameters_dict) else parameters['c2']
	chi = tuned_parameters_dict['chi'] if ('chi' in tuned_parameters_dict) else parameters['chi']
	velocity_coefficient = tuned_parameters_dict['velocity_coefficient'] if ('velocity_coefficient' in tuned_parameters_dict) else parameters['velocity_coefficient']
	leader_selection_range = tuned_parameters_dict['leader_selection_range'] if ('leader_selection_range' in tuned_parameters_dict) else parameters['leader_selection_range']
	diversity_mechanism = tuned_parameters_dict['diversity_mechanism'] if ('diversity_mechanism' in tuned_parameters_dict) else parameters['diversity_mechanism']
	
	# Execute NSPSO
	results = {}
	combined_F = np.empty((0, objective_dim))
	combined_P = np.empty((0, position_dim))
	for i in range(num_runs):
		random_state = experiment['random_state'] if experiment['random_state'] is not None else np.random.randint(0, 2147483647)
		initial_population = pg.population(pygmo_problem, size=population_size, seed=random_state) # type: ignore
		nspso = pg.algorithm(pg.nspso(gen=max_fitness_eval // population_size, # type: ignore
							 omega=omega,
							 c1=c1,
							 c2=c2,
							 chi=chi,
							 v_coeff=velocity_coefficient,
							 leader_selection_range=leader_selection_range,
							 diversity_mechanism=diversity_mechanism,
							 seed=random_state))
		evolved_population = nspso.evolve(initial_population)
		# Accumulate the results at each step
		Pos, Fit = np.array(evolved_population.get_x()), np.array(evolved_population.get_f())
		results[i+1] = {"P":Pos, "F":Fit,}
		combined_P = np.vstack((combined_P, Pos))
		combined_F = np.vstack((combined_F, Fit))
	# Store the results
	dump_results(experiment_name, results_folder, results, combined_P, combined_F, population_size)
	return f'{experiment_name} with tunable parameters ({omega}, {c1}, {c2}, {chi}, {velocity_coefficient}, {leader_selection_range}, {diversity_mechanism}) was successfully executed!'


def run_spea2(experiment: dict[str, Any],
			  problem: dict[str, Any],
			  parameters: dict[str, Any]) -> str:
	# Get the experiment configuration
	experiment_name = experiment['name']
	results_folder = experiment['results_folder']
	fine_tuning_folder = experiment['fine_tuning_folder']
	num_runs = experiment['num_runs']
	max_fitness_eval = experiment['max_fitness_eval']
	population_size = experiment['population_size']
	random_state = experiment['random_state']
    # Get the problem configuration
	fitness = problem['fitness']
	objective_dim = problem['objective_dim']
	position_dim = problem['position_dim']
	lower_bound_array = problem['lower_bound_array']
	upper_bound_array = problem['upper_bound_array']
	class MyProblem(Problem):
		def __init__(self, n_var, n_obj, xl, xu):
			super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu)
		def _evaluate(self, X, out, *args, **kwargs):
			out["F"] = np.array([fitness(x) for x in X])
	pymoo_fitness = MyProblem(n_obj=objective_dim, n_var=position_dim, xl=lower_bound_array, xu=upper_bound_array)

	# Get tunable parameters (check if the parameters were tuned)
	tuned_parameters_dict = get_tuned_parameters(experiment_name, fine_tuning_folder)
	recombination_probability = tuned_parameters_dict['recombination_probability'] if ('recombination_probability' in tuned_parameters_dict) else parameters['recombination_probability']
	eta_recombination = tuned_parameters_dict['eta_recombination'] if ('eta_recombination' in tuned_parameters_dict) else parameters['eta_recombination']
	mutation_probability = tuned_parameters_dict['mutation_probability'] if ('mutation_probability' in tuned_parameters_dict) else parameters['mutation_probability']
	eta_mutation = tuned_parameters_dict['eta_mutation'] if ('eta_mutation' in tuned_parameters_dict) else parameters['eta_mutation']
	# Instantiate SPEA-II
	crossover = SBX(prob=recombination_probability, prob_var=1.0, eta=eta_recombination)
	mutation = PM(prob=mutation_probability, eta=eta_mutation)
	spea2 = SPEA2(pop_size=population_size,
				  sampling=LHS(), # type: ignore
				  crossover=crossover,
				  mutation=mutation,
				  eliminate_duplicates=True,
				  seed=random_state)
	# Execute SPEA-II
	results = {}
	combined_F = np.empty((0, objective_dim))
	combined_P = np.empty((0, position_dim))
	for i in range(num_runs):
		res = minimize(pymoo_fitness,
                	   spea2,
                	   ('n_eval', max_fitness_eval),
					   seed=random_state,
                	   verbose=False)
		# Accumulate the results at each step
		Pos, Fit = np.array(res.X), np.array(res.F)
		results[i+1] = {"P":Pos, "F":Fit,}
		combined_P = np.vstack((combined_P, Pos))
		combined_F = np.vstack((combined_F, Fit))
	# Store the results
	dump_results(experiment_name, results_folder, results, combined_P, combined_F, population_size)
	return f'{experiment_name} with tunable parameters ({recombination_probability}, {eta_recombination}, {mutation_probability}, {eta_mutation}) was successfully executed!'
