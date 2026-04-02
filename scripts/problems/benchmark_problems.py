from problems.DTLZ import dtlz1_pareto, dtlz2_pareto, dtlz3_pareto, dtlz4_pareto, dtlz5_pareto, dtlz6_pareto, dtlz7_pareto

from numpy.typing import NDArray
from pygmo import problem, dtlz, zdt, fast_non_dominated_sorting, select_best_N_mo # type: ignore
from pymoo.problems.many.wfg import WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9
from optproblems import zdt as opt_zdt, wfg as opt_wfg
from typing import Callable

import numpy as np

def get_problem(name: str, n_var: int, n_obj: int, wfg_k: int | None = None) -> tuple[Callable, NDArray[np.floating], NDArray[np.floating]]:
  # Validation of inputs
  if name in {'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6'} and (n_obj != 2 or n_var < 2):
    raise ValueError(f'Problem {name} only supports 2 objectives.')
  if name in {'dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7'}:
    if (n_var < n_obj):
      raise ValueError(f'Problem {name} requires at least {n_obj} variables.')
    if n_obj < 2:
      ValueError(f'Problem {name} requires at least 2 objectives.')
  if name in {'wfg1', 'wfg2', 'wfg3', 'wfg4', 'wfg5', 'wfg6', 'wfg7', 'wfg8', 'wfg9'}:
    if wfg_k is None:
      raise ValueError('For WFG problems, the parameter "k" is required.')
    if (wfg_k % (n_obj - 1) != 0):
      raise ValueError('For WFG problems, the parameter "k" must be a multiple of "n_obj" minus one.')
    if name in {'wfg2', 'wfg3'} and ((n_var - wfg_k) % 2) != 0:
      raise ValueError('For WFG problems, the number of distance-related variables (n_var - wfg_k) must be divisible by two.')

  # Problem selection
  if name in {'zdt1', 'zdt2', 'zdt3', 'zdt6'}:
    func = {'zdt1': problem(zdt(prob_id=1, param=n_var)).fitness, 'zdt2': problem(zdt(prob_id=2, param=n_var)).fitness,
            'zdt3': problem(zdt(prob_id=3, param=n_var)).fitness, 'zdt6': problem(zdt(prob_id=6, param=n_var)).fitness}
    return func[name], np.zeros((n_var)), np.ones((n_var))
  elif name == 'zdt4':
    return problem(zdt(prob_id=4, param=n_var)).fitness, np.array([0.0] + [-5.0]*(n_var-1)), np.array([1] + [5.0]*(n_var-1))

  elif name in {'dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7'}:
    func = {'dtlz1':problem(dtlz(prob_id=1, dim=n_var, fdim=n_obj)).fitness, 'dtlz2':problem(dtlz(prob_id=2, dim=n_var, fdim=n_obj)).fitness,
            'dtlz3':problem(dtlz(prob_id=3, dim=n_var, fdim=n_obj)).fitness, 'dtlz4':problem(dtlz(prob_id=4, dim=n_var, fdim=n_obj)).fitness,
            'dtlz5':problem(dtlz(prob_id=5, dim=n_var, fdim=n_obj)).fitness, 'dtlz6':problem(dtlz(prob_id=6, dim=n_var, fdim=n_obj)).fitness,
            'dtlz7':problem(dtlz(prob_id=7, dim=n_var, fdim=n_obj)).fitness}
    return func[name], np.zeros((n_var)), np.ones((n_var))

  elif name in {'wfg1', 'wfg4', 'wfg5', 'wfg6', 'wfg7', 'wfg8', 'wfg9'}:
    func = {'wfg1': WFG1(n_var=n_var, n_obj=n_obj, k=wfg_k).evaluate, 'wfg4': WFG4(n_var=n_var, n_obj=n_obj, k=wfg_k).evaluate,
            'wfg5': WFG5(n_var=n_var, n_obj=n_obj, k=wfg_k).evaluate, 'wfg6': WFG6(n_var=n_var, n_obj=n_obj, k=wfg_k).evaluate,
            'wfg7': WFG7(n_var=n_var, n_obj=n_obj, k=wfg_k).evaluate, 'wfg8': WFG8(n_var=n_var, n_obj=n_obj, k=wfg_k).evaluate,
            'wfg9': WFG9(n_var=n_var, n_obj=n_obj, k=wfg_k).evaluate}
    return func[name], np.zeros((n_var)), np.arange(1, n_var+1, dtype=np.float64) * 2
  elif name in {'wfg2', 'wfg3'}:
    func = {'wfg2': WFG2(n_var=n_var, n_obj=n_obj, k=wfg_k).evaluate, 'wfg3': WFG3(n_var=n_var, n_obj=n_obj, k=wfg_k).evaluate}
    return func[name], np.zeros((n_var)), np.arange(1, n_var+1, dtype=np.float64) * 2

  else:
    raise ValueError(f"Problem {name} not found.")

######################################################## Pareto front #######################################################

def get_pareto(name: str, N: int, n_var: int, n_obj: int, wfg_k: int | None = None) -> NDArray[np.float64]:
  # Validation of inputs
  if name not in {'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6', 'dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7', 'wfg1', 'wfg2', 'wfg3', 'wfg4', 'wfg5', 'wfg6', 'wfg7', 'wfg8', 'wfg9'}:
    raise ValueError(f"Pareto front for {name} not found.")
  if name in {'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6'} and (n_obj != 2 or n_var < 2):
    raise ValueError(f'Problem {name} only supports 2 objectives.')
  if name in {'dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7'}:
    if (n_obj > 3):
      raise ValueError(f'Problem {name} only has 2D and 3D implementations.')
    if (n_var < n_obj):
      raise ValueError(f'Problem {name} requires at least {n_obj} variables.')
    if n_obj < 2:
      ValueError(f'Problem {name} requires at least 2 objectives.')
  if name in {'wfg1', 'wfg2', 'wfg3', 'wfg4', 'wfg5', 'wfg6', 'wfg7', 'wfg8', 'wfg9'}:
    if wfg_k is None:
      raise ValueError('For WFG problems, the parameter "k" is required.')
    if (wfg_k % (n_obj - 1) != 0):
      raise ValueError('For WFG problems, the parameter "k" must be a multiple of "n_obj" minus one.')
    if name in {'wfg2', 'wfg3'} and ((n_var - wfg_k) % 2) != 0:
      raise ValueError('For WFG problems, the number of distance-related variables (n_var - wfg_k) must be divisible by two.')

  # Pareto function selection
  if name in {'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6'}:
    prob_classes = {'zdt1': opt_zdt.ZDT1(n_var), 'zdt2': opt_zdt.ZDT2(n_var), 'zdt3': opt_zdt.ZDT3(n_var), 'zdt4': opt_zdt.ZDT4(n_var), 'zdt6': opt_zdt.ZDT6(n_var)}
    prob_class = prob_classes[name]
    optimal_solutions = prob_class.get_optimal_solutions(N)
    for individual in optimal_solutions:
      prob_class.evaluate(individual)
    objective_values = np.array([individual.objective_values for individual in optimal_solutions])

  elif name in {'dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7'}:
    pareto_solutions = {'dtlz1': dtlz1_pareto(N, n_obj), 'dtlz2': dtlz2_pareto(N, n_obj), 'dtlz3': dtlz3_pareto(N, n_obj), 'dtlz4': dtlz4_pareto(N, n_obj), 
                    'dtlz5': dtlz5_pareto(N, n_obj), 'dtlz6': dtlz6_pareto(N, n_obj), 'dtlz7': dtlz7_pareto(N, n_obj)}
    return pareto_solutions[name]
  
  elif name in {'wfg1', 'wfg4', 'wfg5', 'wfg6', 'wfg7', 'wfg8', 'wfg9'}:
    # Pymoo Pareto front generation
    pareto_classes = {'wfg1': np.array(WFG1(n_var=n_var, n_obj=n_obj, k=wfg_k).pareto_front()), 'wfg4': np.array(WFG4(n_var=n_var, n_obj=n_obj, k=wfg_k).pareto_front()),
                      'wfg5': np.array(WFG5(n_var=n_var, n_obj=n_obj, k=wfg_k).pareto_front()), 'wfg6': np.array(WFG6(n_var=n_var, n_obj=n_obj, k=wfg_k).pareto_front()),
                      'wfg7': np.array(WFG7(n_var=n_var, n_obj=n_obj, k=wfg_k).pareto_front()), 'wfg8': np.array(WFG8(n_var=n_var, n_obj=n_obj, k=wfg_k).pareto_front()),
                      'wfg9': np.array(WFG9(n_var=n_var, n_obj=n_obj, k=wfg_k).pareto_front())}
    objective_values = pareto_classes[name]
    # Optproblems Pareto front generation
    prob_classes = {'wfg1': opt_wfg.WFG1(n_obj, n_var, wfg_k), 'wfg4': opt_wfg.WFG4(n_obj, n_var, wfg_k), 'wfg5': opt_wfg.WFG5(n_obj, n_var, wfg_k),
                    'wfg6': opt_wfg.WFG6(n_obj, n_var, wfg_k), 'wfg7': opt_wfg.WFG7(n_obj, n_var, wfg_k), 'wfg8': opt_wfg.WFG8(n_obj, n_var, wfg_k),
                    'wfg9': opt_wfg.WFG9(n_obj, n_var, wfg_k)}
    prob_class = prob_classes[name]
    optimal_solutions = prob_class.get_optimal_solutions(N)
    for individual in optimal_solutions:
      prob_class.evaluate(individual)
    objective_values = np.vstack((objective_values, np.array([individual.objective_values for individual in optimal_solutions])))
  elif name in {'wfg2', 'wfg3'}:
    # Pymoo Pareto front generation
    pareto_dict = {'wfg2': np.array(WFG2(n_var=n_var, n_obj=n_obj, k=wfg_k).pareto_front()),
                   'wfg3': np.array(WFG3(n_var=n_var, n_obj=n_obj, k=wfg_k).pareto_front())}
    objective_values = pareto_dict[name]
    # Optproblems Pareto front generation
    prob_classes = {'wfg2': opt_wfg.WFG2(n_obj, n_var, wfg_k), 'wfg3': opt_wfg.WFG3(n_obj, n_var, wfg_k)}
    prob_class = prob_classes[name]
    optimal_solutions = prob_class.get_optimal_solutions(N)
    for individual in optimal_solutions:
      prob_class.evaluate(individual)
    objective_values = np.vstack((objective_values, np.array([individual.objective_values for individual in optimal_solutions])))
  else:
    raise ValueError(f"Problem {name} not found.")

  # Get the non dominated objective values
  best_idxs = select_best_N_mo(objective_values, N)
  best_objective_values = objective_values[best_idxs]
  pareto_solutions = best_objective_values[fast_non_dominated_sorting(points=best_objective_values)[0][0]]
  return pareto_solutions