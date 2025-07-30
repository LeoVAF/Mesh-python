from pygmo import fast_non_dominated_sorting

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

n_var = 10
n_obj = 3

X = np.random.rand(5, 10)

wfg4(np.array([0,0,0,0,0,0,0,0,0,0]), 3, 2)

pygmo_wfg = problem(wfg(prob_id=4, dim_dvs=n_var, dim_obj=n_obj, dim_k=n_obj-1)).fitness

for z in X:
  F1 = wfg4(z, 3, 2)
  F2 = pygmo_wfg(z)
  assert np.linalg.norm(F1 - F2) < 1e-10