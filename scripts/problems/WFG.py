from pygmo import fast_non_dominated_sorting, select_best_N_mo
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.problems.many.wfg import WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize

from joblib import Parallel, delayed

import numpy as np

from pygmo import problem, wfg

################################ Auxiliar Functions ################################
def normalize(z, n_var):
  return z / (np.arange(1, n_var + 1) * 2)

def calculate_x(t, A):
  x = np.empty((t.shape[0]))
  x[:-1] = np.maximum(t[-1], A) * (t[:-1] - 0.5) + 0.5
  x[-1] = t[-1]
  return x
####################################################################################

################################# Shape Functions #################################
def concave(x):
  pi_div_2 = np.pi/2
  n_obj = x.shape[0] + 1
  arr = np.empty(n_obj)
  arr[0] = np.prod(np.sin(x * pi_div_2))
  arr[-1] = np.cos(x[0] * pi_div_2)
  for m in range(1, n_obj-1):
    arr[m] = np.prod(np.sin(x[:n_obj-m-1] * pi_div_2)) * np.cos(x[n_obj - 1 - m] * pi_div_2)
  return arr
###################################################################################

############################# Transforamtion Functions #############################
def s_multi(y, A, B, C):
  const = np.abs(y - C)/(2 * (np.floor(C - y) + C))
  return (1 + np.cos((4*A + 2)*np.pi*(0.5 - const)) + 4*B*(const**2)) / (B + 2)

def r_sum(y, w):
  return np.sum(y * w) / np.sum(w)
####################################################################################

def wfg4(z, n_obj, k):
  n_var = z.shape[0]
  z_normalized = normalize(z, n_var)
  # Transformations
  t1 = s_multi(z_normalized, 30, 10, 0.35)
  var_step = k // (n_obj-1)
  w = np.ones(var_step)
  t2 = np.empty((n_obj))
  for i in range(n_obj-1):
    t2[i] = r_sum(t1[i*var_step:(i+1)*var_step], w)
  t2[-1] = r_sum(t1[k:], np.ones(n_var-k))
  # Objective
  x = calculate_x(t2, np.ones(n_obj-1))
  S = np.arange(1, n_obj+1) * 2
  f = x[-1] + S * concave(x[:-1])
  return f



############################# Testing the functions #############################
n_var = 10
n_obj = 3
k = 2

X = np.random.rand(5, 10)

wfg4(np.array([0,0,0,0,0,0,0,0,0,0]), n_obj, k)

pygmo_wfg = problem(wfg(prob_id=4, dim_dvs=n_var, dim_obj=n_obj, dim_k=k)).fitness

for z in X:
  F1 = wfg4(z, 3, 2)
  F2 = pygmo_wfg(z)
  assert np.linalg.norm(F1 - F2) < 1e-10
##################################################################################

import time

def wfg_pareto_generation_by_algorithms(name, N, n_var, n_obj, wfg_k):
  ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
  nsga2 = NSGA2(pop_size=N, eliminate_duplicates=True)
  nsga3 = NSGA3(ref_dirs=ref_dirs, pop_size=N)
  ctaea = CTAEA(ref_dirs=ref_dirs)
  smsemoa = SMSEMOA(pop_size=N)
  moead = MOEAD(ref_dirs=ref_dirs, n_neighbors=N)
  if name in {'wfg1', 'wfg4', 'wfg5', 'wfg6', 'wfg7', 'wfg8', 'wfg9'}:
    problem_class = {'wfg1': WFG1(n_var=n_var, n_obj=n_obj, k=wfg_k), 'wfg4': WFG4(n_var=n_var, n_obj=n_obj, k=wfg_k),
                      'wfg5': WFG5(n_var=n_var, n_obj=n_obj, k=wfg_k), 'wfg6': WFG6(n_var=n_var, n_obj=n_obj, k=wfg_k),
                      'wfg7': WFG7(n_var=n_var, n_obj=n_obj, k=wfg_k), 'wfg8': WFG8(n_var=n_var, n_obj=n_obj, k=wfg_k),
                      'wfg9': WFG9(n_var=n_var, n_obj=n_obj, k=wfg_k)}
  elif name in {'wfg2', 'wfg3'}:
    problem_class = {'wfg2': WFG2(n_var=n_var, n_obj=n_obj, k=wfg_k), 'wfg3': WFG3(n_var=n_var, n_obj=n_obj, k=wfg_k)}

  alg_list = [nsga2, nsga3, ctaea, smsemoa, moead]

  algorithm_results_parallel = Parallel(n_jobs=len(alg_list))(delayed(minimize)(problem_class[name], algorithm=alg, termination=('n_gen', 50)) for alg in alg_list)
  algorithm_results = np.empty((0, n_obj))
  for res in algorithm_results_parallel:
    algorithm_results = np.vstack((algorithm_results, res.F))

  return algorithm_results[fast_non_dominated_sorting(algorithm_results)[0][0]]