from mesh.core import Mesh, MeshParameters

from pathlib import Path
from pymoo.algorithms.moo.cmopso import CMOPSO
from pymoo.algorithms.moo.mopso_cd import MOPSO_CD
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from typing import Any, Callable

import numpy as np
import optuna
import pygmo as pg
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


def fine_tune_cmopso(experiment: dict[str, Any],
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
	class PymooProblem(Problem):
		def __init__(self, n_var, n_obj, xl, xu):
			super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu)
		def _evaluate(self, X, out, *args, **kwargs):
			out["F"] = np.array([fitness(x) for x in X])
	pymoo_fitness = PymooProblem(n_obj=objective_dim, n_var=position_dim, xl=lower_bound_array, xu=upper_bound_array)

	def tuning(trial: optuna.Trial):
		# Get tunable parameters (check if the parameters was tuned)
		max_velocity_rate = trial.suggest_float('max_velocity_rate', 0.0, 1.0)
		elite_size = trial.suggest_int('max_elite_size', 1, population_size)
		initial_velocity = trial.suggest_categorical('initial_velocity', ['random', 'zero'])
		mutate_rate = trial.suggest_float('mutate_rate', 0.0, 1.0) # Probability
		# Execute CMOPSO
		loss_values = []
		for step in range(n_steps):
			cmopso = CMOPSO(pop_size=population_size,
							max_velocity_rate=max_velocity_rate,
							elite_size=elite_size,
							initial_velocity=initial_velocity,
							mutate_rate=mutate_rate,
							sampling=LHS(), # type: ignore
							eliminate_duplicates=True,
							seed=random_state)
			res = minimize(pymoo_fitness,
						   cmopso,
						   ('n_eval', max_fitness_eval),
						   seed=random_state,
						   verbose=False)
			# Get the result and calculate the loss value
			Fit = np.array(res.F)
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


def fine_tune_maco(experiment: dict[str, Any],
				    problem: dict[str, Any],
                    tuning_configuration: dict[str, Any],
                    fixed_parameters: dict[str, Any],
                    performance_indicator: Callable) -> str:
	# Get the experiment configuration
	experiment_name = experiment['name']
	fine_tuning_folder = experiment['fine_tuning_folder']
	max_fitness_eval = experiment['max_fitness_eval']
	population_size = experiment['population_size']
    # Get the problem configuration
	fitness_function = problem['fitness']
	objective_dim = problem['objective_dim']
	#position_dim = problem['position_dim']
	lower_bound_array = problem['lower_bound_array']
	upper_bound_array = problem['upper_bound_array']
	# Get the fine tuning configuration
	n_trials = tuning_configuration['n_trials']
	n_steps = tuning_configuration['n_steps']
	pruner = tuning_configuration['pruner']
	class PygmoProblem:
		def fitness(self, x):
			return fitness_function(x)
		def get_bounds(self):
			return (lower_bound_array, upper_bound_array)
		def get_nobj(self):
			return objective_dim
	pygmo_problem = PygmoProblem()

	def tuning(trial: optuna.Trial):
		# Get tunable parameters (check if the parameters was tuned)
		ker = trial.suggest_int('ker', 2, population_size)
		q = trial.suggest_float('q', 0.0, 1.0)
		focus = trial.suggest_float('focus', 0.0, 3.0)
		# Execute MACO
		loss_values = []
		for step in range(n_steps):
			random_state = experiment['random_state'] if experiment['random_state'] is not None else np.random.randint(0, 2147483647)
			initial_population = pg.population(pygmo_problem, size=population_size, seed=random_state) # type: ignore
			maco = pg.algorithm(pg.maco(gen=max_fitness_eval // population_size, # type: ignore
										ker=ker,
										q=q,
										focus=focus,
										seed=random_state))
			evolved_population = maco.evolve(initial_population)
			# Get the result and calculate the loss value
			Fit = evolved_population.get_f()
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

	def tuning(trial: optuna.Trial):
		global_best_attribution_type = trial.suggest_categorical('global_best_attribution_type', [0, 1])
		dm_pool_type = trial.suggest_categorical('differential_mutation_pool_type', [0, 1, 2])
		dm_operation_type = trial.suggest_categorical('differential_mutation_type', [0, 1, 2, 3, 4])
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


def fine_tune_mopso_cd(experiment: dict[str, Any],
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
	class PymooProblem(Problem):
		def __init__(self, n_var, n_obj, xl, xu):
			super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu)
		def _evaluate(self, X, out, *args, **kwargs):
			out["F"] = np.array([fitness(x) for x in X])
	pymoo_fitness = PymooProblem(n_obj=objective_dim, n_var=position_dim, xl=lower_bound_array, xu=upper_bound_array)

	def tuning(trial: optuna.Trial):
		# Get tunable parameters (check if the parameters was tuned)
		w = trial.suggest_float('w', 0.0, 1.0)
		c1 = trial.suggest_float('c1', 0.0, 3.0)
		c2 = trial.suggest_float('c2', 0.0, 3.0)
		max_velocity_rate = trial.suggest_float('max_velocity_rate', 0.0, 1.0)
		archive_size = trial.suggest_int('archive_size', population_size, 3*population_size)
		# Execute MOPSO-CD
		loss_values = []
		for step in range(n_steps):
			mopso_cd = MOPSO_CD(pop_size=population_size,
								w=w,
								c1=c1,
								c2=c2,
								max_velocity_rate=max_velocity_rate,
								archive_size=archive_size,
								sampling=LHS(), # type: ignore
								eliminate_duplicates=True,
								seed=random_state)
			res = minimize(pymoo_fitness,
						   mopso_cd,
						   ('n_eval', max_fitness_eval),
						   seed=random_state,
						   verbose=False)
			# Get the result and calculate the loss value
			Fit = res.F
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
    # Get the problem configuration
	fitness_function = problem['fitness']
	objective_dim = problem['objective_dim']
	#position_dim = problem['position_dim']
	lower_bound_array = problem['lower_bound_array']
	upper_bound_array = problem['upper_bound_array']
	# Get the fine tuning configuration
	n_trials = tuning_configuration['n_trials']
	n_steps = tuning_configuration['n_steps']
	pruner = tuning_configuration['pruner']
	class PygmoProblem:
		def fitness(self, x):
			return fitness_function(x)
		def get_bounds(self):
			return (lower_bound_array, upper_bound_array)
		def get_nobj(self):
			return objective_dim
	pygmo_problem = PygmoProblem()

	def tuning(trial: optuna.Trial):
		# Get tunable parameters (check if the parameters was tuned)
		recombination_probability = trial.suggest_float('recombination_probability', 0.0, 1.0)
		eta_recombination = trial.suggest_int('eta_recombination', 1, 99)
		mutation_probability = trial.suggest_float('mutation_probability', 0.0, 1.0)
		eta_mutation = trial.suggest_int('eta_mutation', 1, 99)
		# Execute NSGA-II
		loss_values = []
		for step in range(n_steps):
			random_state = experiment['random_state'] if experiment['random_state'] is not None else np.random.randint(0, 2147483647)
			initial_population = pg.population(pygmo_problem, size=population_size, seed=random_state) # type: ignore
			nsga2 = pg.algorithm(pg.nsga2(gen=max_fitness_eval // population_size, # type: ignore
								 		  cr=recombination_probability,
										  eta_c=eta_recombination,
										  m=mutation_probability,
										  eta_m=eta_mutation,
										  seed=random_state))
			evolved_population = nsga2.evolve(initial_population)
			# Get the result and calculate the loss value
			Fit = evolved_population.get_f()
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


def fine_tune_nspso(experiment: dict[str, Any],
				    problem: dict[str, Any],
                    tuning_configuration: dict[str, Any],
                    fixed_parameters: dict[str, Any],
                    performance_indicator: Callable) -> str:
	# Get the experiment configuration
	experiment_name = experiment['name']
	fine_tuning_folder = experiment['fine_tuning_folder']
	max_fitness_eval = experiment['max_fitness_eval']
	population_size = experiment['population_size']
    # Get the problem configuration
	fitness_function = problem['fitness']
	objective_dim = problem['objective_dim']
	#position_dim = problem['position_dim']
	lower_bound_array = problem['lower_bound_array']
	upper_bound_array = problem['upper_bound_array']
	# Get the fine tuning configuration
	n_trials = tuning_configuration['n_trials']
	n_steps = tuning_configuration['n_steps']
	pruner = tuning_configuration['pruner']
	class PygmoProblem:
		def fitness(self, x):
			return fitness_function(x)
		def get_bounds(self):
			return (lower_bound_array, upper_bound_array)
		def get_nobj(self):
			return objective_dim
	pygmo_problem = PygmoProblem()

	def tuning(trial: optuna.Trial):
		# Get tunable parameters (check if the parameters was tuned)
		omega = trial.suggest_float('omega', 0.0, 1.0)
		c1 = trial.suggest_float('c1', 0.0, 3)
		c2 = trial.suggest_float('c2', 0.0, 3)
		chi = trial.suggest_float('chi', 0.0, 3)
		velocity_coefficient = trial.suggest_float('velocity_coefficient', 0.0, 1.0)
		leader_selection_range = trial.suggest_int('leader_selection_range', 1, 100)
		diversity_mechanism = trial.suggest_categorical('diversity_mechanism', ['crowding distance', 'niche count', 'max min'])
		# Execute NSPSO
		loss_values = []
		for step in range(n_steps):
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
			# Get the result and calculate the loss value
			Fit = evolved_population.get_f()
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


def fine_tune_spea2(experiment: dict[str, Any],
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
	class PymooProblem(Problem):
		def __init__(self, n_var, n_obj, xl, xu):
			super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu)
		def _evaluate(self, X, out, *args, **kwargs):
			out["F"] = np.array([fitness(x) for x in X])
	pymoo_fitness = PymooProblem(n_obj=objective_dim, n_var=position_dim, xl=lower_bound_array, xu=upper_bound_array)

	def tuning(trial: optuna.Trial):
		# Get tunable parameters (check if the parameters was tuned)
		recombination_probability = trial.suggest_float('recombination_probability', 0.0, 1.0)
		eta_recombination = trial.suggest_int('eta_recombination', 1, 99)
		mutation_probability = trial.suggest_float('mutation_probability', 0.0, 1.0)
		eta_mutation = trial.suggest_int('eta_mutation', 1, 99)
		crossover = SBX(prob=recombination_probability, prob_var=1.0, eta=eta_recombination)
		mutation = PM(prob=mutation_probability, eta=eta_mutation)
		# Execute SPEA-II
		loss_values = []
		for step in range(n_steps):
			spea2 = SPEA2(pop_size=population_size,
						  sampling=LHS(), # type: ignore
						  crossover=crossover,
						  mutation=mutation,
						  eliminate_duplicates=True,
						  seed=random_state)
			res = minimize(pymoo_fitness,
						   spea2,
						   ('n_eval', max_fitness_eval),
						   seed=random_state,
						   verbose=False)
			# Get the result and calculate the loss value
			Fit = res.F
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
