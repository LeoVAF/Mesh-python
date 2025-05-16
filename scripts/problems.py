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
  g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
  h = 1 - (f1 / g) ** 2
  f2 = g * h
  return np.array([f1, f2])

def zdt6(x):
  f1 = 1 - np.exp(-4 * x[0]) * np.sin(6 * np.pi * x[0]) ** 6
  g = 1 + 9 * (np.sum(x[1:]) / (len(x) - 1)) ** 0.25
  h = 1 - (f1 / g) ** 2
  f2 = g * h
  return np.array([f1, f2])

def get_problem(name, n_var=None, n_obj=None):
  if name == 'zdt1':
    return zdt1, np.array([0.0]*n_var), np.array([1.0]*n_var)
  elif name == 'zdt2':
    return zdt2, np.array([0.0]*n_var), np.array([1.0]*n_var)
  elif name == 'zdt3':
    return zdt3, np.array([0.0] + [-10.0]*(n_var-1)), np.array([1] + [10.0]*(n_var-1))
  elif name == 'zdt4':
    return zdt4, np.array([0.0]*n_var), np.array([1.0]*n_var)
  elif name == 'zdt6':
    return zdt6, np.array([0.0]*n_var), np.array([1.0]*n_var)
  else:
    raise ValueError(f"Problem {name} not found.")

######################################################## Pareto front #######################################################
def zdt1_pareto(N, n_var=None, n_obj=None):
  x = np.linspace(0, 1, N)
  y = 1 - np.sqrt(x)
  return [np.column_stack((x, y))]

def get_pareto(name, N, n_var=None, n_obj=None):
  if name == 'zdt1':
    return zdt1_pareto(N, n_var, n_obj)
  else:
    raise ValueError(f"Pareto front for {name} not found.")