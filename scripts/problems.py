import numpy as np
from functools import partial
import pymoo.problems

def zdt1(x):
  f1 = x[0]
  g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
  h = 1 - np.sqrt(f1 / g)
  f2 = g * h
  return np.array([f1, f2])

def zdt2(x):
  f1 = x[0]
  g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
  h = 1 - (f1 / g) ** 2
  f2 = g * h
  return np.array([f1, f2])

def zdt3(x):
  f1 = x[0]
  g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
  f1_div_g = f1/g
  h = 1 - np.sqrt(f1_div_g) - (f1_div_g) * np.sin(10 * np.pi * f1)
  f2 = g * h
  return np.array([f1, f2])

def zdt4(x):
  f1 = x[0]
  g = 1 + 10*(len(x) - 1) + np.sum(x[1:]**2 - 10*np.cos(4 * np.pi * x[1:]))
  h = 1 - np.sqrt(f1 / g)
  f2 = g * h
  return np.array([f1, f2])

def zdt6(x):
  f1 = 1 - np.exp(-4 * x[0]) * np.sin(6 * np.pi * x[0]) ** 6
  g = 1 + 9 * (np.sum(x[1:]) / (len(x) - 1)) ** 0.25
  h = 1 - (f1 / g) ** 2
  f2 = g * h
  return np.array([f1, f2])

def dtlz1(x, n_obj=3):
  k = len(x) - n_obj + 1
  g_plus_1 = 1 + 100 * (k + np.sum((x[n_obj-1:] - 0.5)**2 - np.cos(20 * np.pi * (x[n_obj-1:] - 0.5))))
  F = np.zeros(n_obj)
  F[0] = 0.5 * np.prod(x[:n_obj-1]) * g_plus_1
  F[-1] = 0.5 * (1 - x[0]) * g_plus_1
  for i in range(1, n_obj-1):
    F[i] = 0.5 * np.prod(x[:n_obj-i-1]) * (1 - x[n_obj-i-1]) * g_plus_1
  return F

def dtlz2(x, n_obj=3):
  g_plus_1 = 1 + (np.sum((x[n_obj-1:] - 0.5)**2))
  pi_div_2 = np.pi / 2
  F = np.zeros(n_obj)
  F[0] = np.prod(np.cos(x[:n_obj-1]*pi_div_2)) * g_plus_1
  F[-1] = np.sin(x[0]*pi_div_2) * g_plus_1
  for i in range(1, n_obj-1):
    F[i] = np.prod(np.cos(x[:n_obj-i-1]*pi_div_2)) * np.sin(x[n_obj-i-1]*pi_div_2) * g_plus_1
  return F

def dtlz3(x, n_obj=3):
  k = len(x) - n_obj + 1
  g_plus_1 = 1 + 100 * (k + np.sum((x[n_obj-1:] - 0.5)**2 - np.cos(20 * np.pi * (x[n_obj-1:] - 0.5))))
  pi_div_2 = np.pi / 2
  F = np.zeros(n_obj)
  F[0] = np.prod(np.cos(x[:n_obj-1]*pi_div_2)) * g_plus_1
  F[-1] = np.sin(x[0]*pi_div_2) * g_plus_1
  for i in range(1, n_obj-1):
    F[i] = np.prod(np.cos(x[:n_obj-i-1]*pi_div_2)) * np.sin(x[n_obj-i-1]*pi_div_2) * g_plus_1
  return F

def dtlz4(x, n_obj=3):
  alpha = 100
  g_plus_1 = 1 + (np.sum((x[n_obj-1:] - 0.5)**2))
  pi_div_2 = np.pi / 2
  F = np.zeros(n_obj)
  F[0] = np.prod(np.cos((x[:n_obj-1]**alpha)*pi_div_2)) * g_plus_1
  F[-1] = np.sin((x[0]**alpha)*pi_div_2) * g_plus_1
  for i in range(1, n_obj-1):
    F[i] = np.prod(np.cos((x[:n_obj-i-1]**alpha)*pi_div_2)) * np.sin((x[n_obj-i-1]**alpha)*pi_div_2) * g_plus_1
  return F

def dtlz5(x, n_obj=3):
  g_plus_1 = 1 + (np.sum((x[n_obj-1:] - 0.5)**2))
  theta = np.zeros(n_obj-1)
  theta[0] = x[0]*(np.pi / 2)
  theta[1:] = np.pi/(4*g_plus_1)*(1 + 2*(g_plus_1-1)*x[1:n_obj-1])
  F = np.zeros(n_obj)
  F[0] = np.prod(np.cos(theta)) * g_plus_1
  F[-1] = np.sin(theta[0]) * g_plus_1
  for i in range(1, n_obj-1):
    F[i] = np.prod(np.cos(theta[:n_obj-i-1])) * np.sin(theta[n_obj-i-1]) * g_plus_1
  return F

x = np.random.uniform(size=(5,))
print(dtlz5(x, n_obj=3) - pymoo.problems.get_problem("dtlz5", n_var=5, n_obj=3).evaluate(x, return_values_of=["F"]))

def get_problem(name, n_var, n_obj):
  # Validation of inputs
  if name in {'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6'} and n_obj != 2:
    raise ValueError(f"Problem {name} only supports 2 objectives.")
  if name in {'dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7'} and n_var < n_obj:
    raise ValueError(f"Problem {name} requires at least {n_obj} variables.")
  # Problem selection
  if name == 'zdt1':
    return zdt1, np.array([0.0]*n_var), np.array([1.0]*n_var)
  elif name == 'zdt2':
    return zdt2, np.array([0.0]*n_var), np.array([1.0]*n_var)
  elif name == 'zdt3':
    return zdt3, np.array([0.0]*n_var), np.array([1.0]*n_var)
  elif name == 'zdt4':
    return zdt4, np.array([0.0] + [-5.0]*(n_var-1)), np.array([1] + [5.0]*(n_var-1))
  elif name == 'zdt6':
    return zdt6, np.array([0.0]*n_var), np.array([1.0]*n_var)
  elif name == 'dtlz1':
    return partial(dtlz1, n_obj=n_obj), np.array([0.0]*n_var), np.array([1.0]*n_var)
  else:
    raise ValueError(f"Problem {name} not found.")

######################################################## Pareto front #######################################################
def zdt1_pareto(N, n_obj=None):
  f1 = np.linspace(0, 1, N)
  f2 = 1 - np.sqrt(f1)
  return np.column_stack((f1, f2))

def zdt2_pareto(N, n_obj=None):
  f1 = np.linspace(0, 1, N)
  f2 = 1 - f1 ** 2
  return np.column_stack((f1, f2))

def zdt3_pareto(N, n_obj=None):
  intervals = [
      (0.0, 0.0830015349),
      (0.1822287280, 0.2577623634),
      (0.4093136748, 0.4538821041),
      (0.6183967944, 0.6525117038),
      (0.8233317983, 0.8518328654)
    ]
  f1, f2 = [], []
  for a, b in intervals:
    x = np.linspace(a, b, N // 5)
    y = 1 - np.sqrt(x) - x * np.sin(10 * np.pi * x)
    f1.extend(x)
    f1.append(None)
    f2.extend(y)
    f2.append(None)
  return np.column_stack((f1, f2))

def zdt4_pareto(N, n_obj=None):
  f1 = np.linspace(0, 1, N)
  f2 = 1 - np.sqrt(f1)
  return np.column_stack((f1, f2))

def zdt6_pareto(N, n_obj=None):
  x = np.linspace(0, 1, N)
  f1 = 1 - np.exp(-4 * x) * np.sin(6 * np.pi * x) ** 6
  f2 = 1 - f1 ** 2
  return np.column_stack((f1, f2))

def dtlz1_pareto(N, n_obj=2):
  if n_obj == 2:
    f1 = np.linspace(0, 0.5, N)
    f2 = 0.5 - f1
    return np.column_stack((f1, f2))
  elif n_obj == 3:
    f = np.linspace(0, 0.5, N)
    f1, f2 = np.meshgrid(f, f)
    f3 = 0.5 - f1 - f2
    mask = f3 >= 0
    points = np.column_stack((f1[mask], f2[mask], f3[mask]))
    return np.array(points)

def get_pareto(name, N, n_var, n_obj):
  if name == 'zdt1':
    return zdt1_pareto(N, n_obj)
  elif name == 'zdt2':
    return zdt2_pareto(N, n_obj)
  elif name == 'zdt3':
    return zdt3_pareto(N, n_obj)
  elif name == 'zdt4':
    return zdt4_pareto(N, n_obj)
  elif name == 'zdt6':
    return zdt6_pareto(N, n_obj)
  elif name == 'dtlz1':
    return dtlz1_pareto(N, n_obj)
  else:
    raise ValueError(f"Pareto front for {name} not found.")