import numpy as np

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

def get_problem(name, n_var=10, n_obj=2):
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
  else:
    raise ValueError(f"Problem {name} not found.")

######################################################## Pareto front #######################################################
def zdt1_pareto(N, n_var=None, n_obj=None):
  f1 = np.linspace(0, 1, N)
  f2 = 1 - np.sqrt(f1)
  return np.column_stack((f1, f2))

def zdt2_pareto(N, n_var=None, n_obj=None):
  f1 = np.linspace(0, 1, N)
  f2 = 1 - f1 ** 2
  return np.column_stack((f1, f2))

def zdt3_pareto(N, n_var=None, n_obj=None):
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

def zdt4_pareto(N, n_var=None, n_obj=None):
  f1 = np.linspace(0, 1, N)
  f2 = 1 - np.sqrt(f1)
  return np.column_stack((f1, f2))

def zdt6_pareto(N, n_var=None, n_obj=None):
  x = np.linspace(0, 1, N)
  f1 = 1 - np.exp(-4 * x) * np.sin(6 * np.pi * x) ** 6
  f2 = 1 - f1 ** 2
  return np.column_stack((f1, f2))

def get_pareto(name, N, n_var=None, n_obj=None):
  if name == 'zdt1':
    return zdt1_pareto(N, n_var, n_obj)
  if name == 'zdt2':
    return zdt2_pareto(N, n_var, n_obj)
  if name == 'zdt3':
    return zdt3_pareto(N, n_var, n_obj)
  if name == 'zdt4':
    return zdt4_pareto(N, n_var, n_obj)
  if name == 'zdt6':
    return zdt6_pareto(N, n_var, n_obj)
  else:
    raise ValueError(f"Pareto front for {name} not found.")