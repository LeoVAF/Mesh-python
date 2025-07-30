from .DTLZ import dtlz1, dtlz2, dtlz3, dtlz4, dtlz5, dtlz6, dtlz7,\
                 dtlz1_pareto, dtlz2_pareto, dtlz3_pareto, dtlz4_pareto, dtlz5_pareto, dtlz6_pareto, dtlz7_pareto
# from WFG import 
from .ZDT import zdt1, zdt2, zdt3, zdt4, zdt6,\
                zdt1_pareto, zdt2_pareto, zdt3_pareto, zdt4_pareto, zdt6_pareto

from functools import partial
from pymoo.problems.many.wfg import WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9
from pygmo import problem, wfg
from typing import Callable

import numpy as np

def get_problem(name: str, n_var: int, n_obj: int, wfg_k: int | None = None):
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
      raise ValueError(f'For WFG problems, the parameter "k" is required.')
    if (wfg_k % (n_obj - 1) != 0):
      raise ValueError(f'For WFG problems, the parameter "k" must be a multiple of "n_obj" minus one.')
    if name in {'wfg2', 'wfg3'} and ((n_var - wfg_k) % 2) != 0:
      raise ValueError(f'For WFG problems, the number of distance-related variables (n_var - wfg_k) must be divisible by two.')

  # Problem selection
  if name in {'zdt1', 'zdt2', 'zdt3', 'zdt6'}:
    func = {'zdt1':zdt1, 'zdt2':zdt2, 'zdt3':zdt3, 'zdt6':zdt6}
    return func[name], np.array([0.0]*n_var), np.array([1.0]*n_var)

  elif name == 'zdt4':
    return zdt4, np.array([0.0] + [-5.0]*(n_var-1)), np.array([1] + [5.0]*(n_var-1))

  elif name in {'dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7'}:
    func = {'dtlz1':dtlz1, 'dtlz2':dtlz2, 'dtlz3':dtlz3, 'dtlz4':dtlz4, 'dtlz5':dtlz5, 'dtlz6':dtlz6, 'dtlz7':dtlz7}
    return partial(func[name], n_obj=n_obj), np.array([0.0]*n_var), np.array([1.0]*n_var)

  elif name in {'wfg1', 'wfg4', 'wfg5', 'wfg6', 'wfg7', 'wfg8', 'wfg9'}:
    func = {'wfg1': problem(wfg(prob_id=1, dim_dvs=n_var, dim_obj=n_obj, dim_k=wfg_k)).fitness, 'wfg4': problem(wfg(prob_id=4, dim_dvs=n_var, dim_obj=n_obj, dim_k=wfg_k)).fitness,
            'wfg5': problem(wfg(prob_id=5, dim_dvs=n_var, dim_obj=n_obj, dim_k=wfg_k)).fitness, 'wfg6': problem(wfg(prob_id=6, dim_dvs=n_var, dim_obj=n_obj, dim_k=wfg_k)).fitness,
            'wfg7': problem(wfg(prob_id=7, dim_dvs=n_var, dim_obj=n_obj, dim_k=wfg_k)).fitness, 'wfg8': problem(wfg(prob_id=8, dim_dvs=n_var, dim_obj=n_obj, dim_k=wfg_k)).fitness,
            'wfg9': problem(wfg(prob_id=9, dim_dvs=n_var, dim_obj=n_obj, dim_k=wfg_k)).fitness}
    return func[name], np.array([0.0]*n_var), np.arange(1, n_var+1)

  elif name in {'wfg2', 'wfg3'}:
    func = {'wfg2': problem(wfg(prob_id=2, dim_dvs=n_var, dim_obj=n_obj, dim_k=wfg_k)).fitness, 'wfg3': problem(wfg(prob_id=3, dim_dvs=n_var, dim_obj=n_obj, dim_k=wfg_k)).fitness}
    return func[name], np.array([0.0]*n_var), np.array(1, n_var+1)

  else:
    raise ValueError(f"Problem {name} not found.")

######################################################## Pareto front #######################################################

def get_pareto(name: str, N: int, n_var: int, n_obj: int) -> Callable:
  # Validation of inputs
  if name in {'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6'} and (n_obj != 2 or n_var < 2):
    raise ValueError(f"Problem {name} only supports 2 objectives.")
  if name in {'dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7'} and n_var < n_obj:
    raise ValueError(f"Problem {name} requires at least {n_obj} variables.")

  # Pareto function selection
  pareto_set = {'zdt1':zdt1_pareto(N), 'zdt2':zdt2_pareto(N), 'zdt3':zdt3_pareto(N), 'zdt4':zdt4_pareto(N), 'zdt6':zdt6_pareto(N),
                'dtlz1':dtlz1_pareto(N, n_obj), 'dtlz2':dtlz2_pareto(N, n_obj), 'dtlz3':dtlz3_pareto(N, n_obj), 'dtlz4':dtlz4_pareto(N, n_obj), 'dtlz5':dtlz5_pareto(N, n_obj),
                'dtlz6':dtlz6_pareto(N, n_obj), 'dtlz7':dtlz7_pareto(N, n_obj)}
  if name in pareto_set:
    return pareto_set[name]
  elif name == 'wfg1':
    return WFG1(n_var=n_var, n_obj=n_obj).pareto_front(n_pareto_points=N)
  elif name == 'wfg2':
    return WFG2(n_var=n_var, n_obj=n_obj).pareto_front(n_pareto_points=N)
  elif name == 'wfg3':
    return WFG3(n_var=n_var, n_obj=n_obj).pareto_front(n_pareto_points=N)
  elif name == 'wfg4':
    return WFG4(n_var=n_var, n_obj=n_obj).pareto_front(n_pareto_points=N)
  elif name == 'wfg5':
    return WFG5(n_var=n_var, n_obj=n_obj).pareto_front(n_pareto_points=N)
  elif name == 'wfg6':
    return WFG6(n_var=n_var, n_obj=n_obj).pareto_front(n_pareto_points=N)
  elif name == 'wfg7':
    return WFG7(n_var=n_var, n_obj=n_obj).pareto_front(n_pareto_points=N)
  elif name == 'wfg8':
    return WFG8(n_var=n_var, n_obj=n_obj).pareto_front(n_pareto_points=N)
  elif name == 'wfg9':
    return WFG9(n_var=n_var, n_obj=n_obj).pareto_front()
  else:
    raise ValueError(f"Pareto front for {name} not found.")