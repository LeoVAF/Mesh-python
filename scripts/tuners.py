from mesh.core import Mesh, MeshParameters

from pathlib import Path
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from typing import Any, Callable

import numpy as np
import optuna
import statistics

optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress Optuna logs


def tuning_fitness(value_list: list) -> float:
	return -statistics.median(value_list)

def dump_results(file_name: str, file_folder: str, results: dict) -> None:
    # Creates the folder if it does not exist
    Path(file_folder).mkdir(parents=False, exist_ok=True)
    # Sets the full path of the file
    file_path = f"{file_folder}/{file_name}.txt"
    # Open the file in text mode and write the results
    with open(file_path, 'w', encoding='utf-8') as file:
        for key, value in results.items():
            file.write(f"{key}: {value} ({type(value).__name__})\n")

def fine_tune_mesh(experiment: dict[str, Any],
				   problem: dict[str, Any],
                   tuning_configuration: dict[str, Any],
                   fixed_parameters: dict[str, Any],
                   performance_indicator: Callable) -> str:
	# Get the experiment configuration
	experiment_name = experiment['name']
	fine_tuning_folder = experiment['fine_tuning_folder']
	max_fitness_eval = experiment['max_fitness_eval']
	population_size = experiment['population_size']
	random_state = experiment['random_state']
    # Get the problem configuration
	fitness = problem['fitness']
	objective_dim = problem['objective_dim']
	position_dim = problem['position_dim']
	lower_bound_array = problem['lower_bound_array']
	upper_bound_array = problem['upper_bound_array']
	# Get the fine tuning configuration
	n_trials = tuning_configuration['n_trials']
	n_steps = tuning_configuration['n_steps']
	pruner = tuning_configuration['pruner']

	# Get the fixed parameters
	memory_size = fixed_parameters['memory_size']
	global_best_attribution_type = fixed_parameters['global_best_attribution_type']
	dm_pool_type = fixed_parameters['differential_mutation_pool_type']
	dm_operation_type = fixed_parameters['differential_mutation_type']

	def tuning(trial: optuna.Trial):
		# Get tunable parameters (check if the parameters was tuned)
		communication_probability = trial.suggest_float('communication_probability', 0, 1)
		mutation_rate = trial.suggest_float('mutation_rate', 0, 1)
		personal_guide_array_size = trial.suggest_int('personal_guide_array_size', 1, 3)

		# Execute MESH
		loss_values = []
		for step in range(n_steps):
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

			# Get the result and calculate the loss value
			_, Fit = mesh.get_results()
			loss = performance_indicator(Fit)
			trial.report(loss, step)

			# If the prune criterion is satisfied, so prune this trial
			if trial.should_prune():
				raise optuna.exceptions.TrialPruned()

			# Accumulate the loss value at each step
			loss_values.append(loss)
		
		# Calculate the value to optimize
		fitness_value = tuning_fitness(loss_values)
		return fitness_value

	# Apply the fine tuning
	study = optuna.create_study(pruner=pruner)
	study.optimize(tuning, n_trials=n_trials)

	# Store the best parameters
	dump_results(experiment_name, fine_tuning_folder, study.best_params)
	return f'{experiment_name} was successfully executed!'

def fine_tune_nsga2(experiment: dict[str, Any],
				    problem: dict[str, Any],
                    tuning_configuration: dict[str, Any],
                    fixed_parameters: dict[str, Any],
                    performance_indicator: Callable) -> str:
	# Get the experiment configuration
	experiment_name = experiment['name']
	fine_tuning_folder = experiment['fine_tuning_folder']
	max_fitness_eval = experiment['max_fitness_eval']
	population_size = experiment['population_size']
	random_state = experiment['random_state']
    # Get the problem configuration
	fitness = problem['fitness']
	objective_dim = problem['objective_dim']
	position_dim = problem['position_dim']
	lower_bound_array = problem['lower_bound_array']
	upper_bound_array = problem['upper_bound_array']
	# Get the fine tuning configuration
	n_trials = tuning_configuration['n_trials']
	n_steps = tuning_configuration['n_steps']
	pruner = tuning_configuration['pruner']
	class MyProblem(Problem):
		def __init__(self, n_var, n_obj, xl, xu):
			super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu)
		def _evaluate(self, X, out, *args, **kwargs):
			out["F"] = np.array([fitness(x) for x in X])

	nsga2_fit_function = MyProblem(n_obj=objective_dim, n_var=position_dim, xl=lower_bound_array, xu=upper_bound_array)

	def tuning(trial: optuna.Trial):
		# Get tunable parameters (check if the parameters was tuned)
		recombination_probability = trial.suggest_float('recombination_probability', 0, 1)
		eta_recombination = trial.suggest_int('eta_recombination', 0, 20)
		mutation_probability = trial.suggest_float('mutation_probability', 0, 1)
		eta_mutation = trial.suggest_int('eta_mutation', 0, 20)

		# Instantiate NSGA2
		crossover = SBX(prob=recombination_probability, prob_var=1.0, eta=eta_recombination)
		mutation = PM(prob=mutation_probability, eta=eta_mutation)
		nsga2 = NSGA2(pop_size=population_size,
					  crossover=crossover,
					  mutation=mutation,
					  eliminate_duplicates=True)

		# Execute NSGA-II
		loss_values = []
		for step in range(n_steps):
			res = minimize(nsga2_fit_function,
						   nsga2,
						   ('n_eval', max_fitness_eval),
						   seed=random_state,
						   verbose=False)

			# Get the result and calculate the loss value
			_, Fit = res.X, res.F
			loss = performance_indicator(Fit)
			trial.report(loss, step)

			# If the prune criterion is satisfied, so prune this trial
			if trial.should_prune():
				raise optuna.exceptions.TrialPruned()

			# Accumulate the loss value at each step
			loss_values.append(loss)
		
		# Calculate the value to optimize
		fitness_value = tuning_fitness(loss_values)
		return fitness_value

	# Apply the fine tuning
	study = optuna.create_study(pruner=pruner)
	study.optimize(tuning, n_trials=n_trials)

	# Store the best parameters
	dump_results(experiment_name, fine_tuning_folder, study.best_params)
	return f'{experiment_name} was successfully executed!'