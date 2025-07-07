from mesh.core import *

from pathlib import Path

from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation

from typing import Callable

import optuna
import statistics


def tuning_fitness(value_list):
	return -statistics.mean(value_list)

def dump_results(file_name, file_folder, results) -> None:
    # Creates the folder if it does not exist
    Path(file_folder).mkdir(parents=False, exist_ok=True)
    # Sets the full path of the file
    file_path = f"{file_folder}/{file_name}.txt"
    # Open the file in text mode and write the results
    with open(file_path, 'w', encoding='utf-8') as file:
        for key, value in results.items():
            file.write(f"{key}: {value} ({type(value).__name__})\n")

def fine_tune_mesh(experiment: tuple, # Information to run the experiments
                                      # (experiment name, experiment folder, fine tuning folder, maximum fitness evaluations, population size, random seed)
                   tuning_configuration: tuple, # Fine tuning configuration (n_trials, n_steps, pruner)
                   problem: tuple, # Problem setup (fitness function, number of objectives, number of decision variables, lower bound array, upper bound array)
                   fixed_parameters: tuple, # MESH fixed parameters
                   indicator: Callable # Performance indicator
              ) -> str:
	# Get the experiment name and folder to store results
	experiment_configuration, fine_tuning_folder, max_fitness_eval, population_size, random_state = experiment

	# Get the fine tune configuration
	n_trials, n_steps, pruner = tuning_configuration

  	# Get the problem
	fit_function, objective_dim, position_dim, lower_bound_array, upper_bound_array = problem

	# Get the fixed parameters
	memory_size, global_best_attribution_type, dm_pool_type, dm_operation_type = fixed_parameters

	def tuning(trial):
		# Get tunable parameters (check if the parameters was tuned)
		communication_probability = trial.suggest_float('communication_probability', 0, 1)
		mutation_rate = trial.suggest_float('mutation_rate', 0, 1)
		personal_guide_array_size = trial.suggest_int('personal_guide_array_size', 1, 3)

		# Execute MESH
		loss_values = []
		for step in range(n_steps):
			params = MeshParameters(objective_dim = objective_dim,
									position_dim = position_dim,
									lower_bound_array = lower_bound_array,
									upper_bound_array = upper_bound_array, 
									population_size = population_size,
									memory_size = memory_size,
									global_best_attribution_type = global_best_attribution_type,
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

			# Get the result and calculate the loss value
			_, Fit = mesh.get_results()
			loss = indicator(Fit)
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
	dump_results(experiment_configuration, fine_tuning_folder, study.best_params)
	return f'{experiment_configuration} was successfully executed!'