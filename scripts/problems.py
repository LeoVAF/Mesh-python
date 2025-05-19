import numpy as np
from functools import partial
from pygmo import fast_non_dominated_sorting
from pymoo.problems.many.wfg import WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9

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

def dtlz6(x, n_obj=3):
  g_plus_1 = 1 + (np.sum(x[n_obj-1:]**0.1))
  theta = np.zeros(n_obj-1)
  theta[0] = x[0]*(np.pi / 2)
  theta[1:] = np.pi/(4*g_plus_1)*(1 + 2*(g_plus_1-1)*x[1:n_obj-1])
  F = np.zeros(n_obj)
  F[0] = np.prod(np.cos(theta)) * g_plus_1
  F[-1] = np.sin(theta[0]) * g_plus_1
  for i in range(1, n_obj-1):
    F[i] = np.prod(np.cos(theta[:n_obj-i-1])) * np.sin(theta[n_obj-i-1]) * g_plus_1
  return F

def dtlz7(x, n_obj=3):
  g = 1 + 9 * (np.sum(x[n_obj-1:]) / (len(x) - n_obj + 1))
  F = np.zeros(n_obj)
  F[:n_obj-1] = x[:n_obj-1]
  h = n_obj - np.sum(F[:n_obj-1] * (1 + np.sin(3*np.pi*F[:n_obj-1])) / (1 + g))
  F[-1] = (1 + g) * h
  return F

def get_problem(name, n_var, n_obj):
  # Validation of inputs
  if name in {'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6'} and n_obj != 2:
    raise ValueError(f"Problem {name} only supports 2 objectives.")
  if name in {'dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7'} and n_var < n_obj:
    raise ValueError(f"Problem {name} requires at least {n_obj} variables.")
  
  # Problem selection
  if name == {'zdt1', 'zdt2', 'zdt3', 'zdt6'}:
    func = {'zdt1':zdt1, 'zdt2':zdt2, 'zdt3':zdt3, 'zdt6':zdt6}
    return func[name], np.array([0.0]*n_var), np.array([1.0]*n_var)
  elif name == 'zdt4':
    return zdt4, np.array([0.0] + [-5.0]*(n_var-1)), np.array([1] + [5.0]*(n_var-1))
  elif name in {'dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7'}:
    func = {'dtlz1':dtlz1, 'dtlz2':dtlz2, 'dtlz3':dtlz3, 'dtlz4':dtlz4, 'dtlz5':dtlz5, 'dtlz6':dtlz6, 'dtlz7':dtlz7}
    return partial(func[name], n_obj=n_obj), np.array([0.0]*n_var), np.array([1.0]*n_var)
  elif name in {'wfg1', 'wfg4', 'wfg5', 'wfg6', 'wfg7', 'wfg8', 'wfg9'}:
    func = {'wfg1':WFG1(n_var=n_var, n_obj=n_obj).evaluate, 'wfg4':WFG4(n_var=n_var, n_obj=n_obj).evaluate, 'wfg5':WFG5(n_var=n_var, n_obj=n_obj).evaluate,
            'wfg6':WFG6(n_var=n_var, n_obj=n_obj).evaluate, 'wfg7':WFG7(n_var=n_var, n_obj=n_obj).evaluate, 'wfg8':WFG8(n_var=n_var, n_obj=n_obj).evaluate,
            'wfg9':WFG9(n_var=n_var, n_obj=n_obj).evaluate}
    return func[name], np.array([0.0]*n_var), np.array([2.0*i for i in range(1, n_var+1)])
  elif name in {'wfg2', 'wfg3'}:
    func = {'wfg2':WFG2(n_var=n_var, n_obj=n_obj).evaluate, 'wfg3':WFG3(n_var=n_var, n_obj=n_obj).evaluate}
    return func[name], np.array([0.0]*n_var), np.array([2.0*i for i in range(1, n_var+1)])
  else:
    raise ValueError(f"Problem {name} not found.")

######################################################## Pareto front #######################################################
def zdt1_pareto(N):
  f1 = np.linspace(0, 1, N)
  f2 = 1 - np.sqrt(f1)
  return np.column_stack((f1, f2))

def zdt2_pareto(N):
  f1 = np.linspace(0, 1, N)
  f2 = 1 - f1 ** 2
  return np.column_stack((f1, f2))

def zdt3_pareto(N):
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

def zdt4_pareto(N):
  f1 = np.linspace(0, 1, N)
  f2 = 1 - np.sqrt(f1)
  return np.column_stack((f1, f2))

def zdt6_pareto(N):
  x = np.linspace(0, 1, N)
  f1 = 1 - np.exp(-4 * x) * np.sin(6 * np.pi * x) ** 6
  f2 = 1 - f1 ** 2
  return np.column_stack((f1, f2))

def dtlz1_pareto(N, n_obj=3):
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

def dtlz2_pareto(N, n_obj=3):
  if n_obj == 2:
    theta = np.linspace(0, np.pi/2, N)
    f1 = np.cos(theta)
    f2 = np.sin(theta)
    return np.column_stack((f1, f2))
  elif n_obj == 3:
    angle = np.linspace(0, np.pi/2, N)
    theta, phi = np.meshgrid(angle, angle)
    f1 = np.cos(phi) * np.cos(theta)
    f2 = np.cos(phi) * np.sin(theta)
    f3 = np.sin(phi)
    return np.column_stack((f1.flatten(), f2.flatten(), f3.flatten()))

def dtlz3_pareto(N, n_obj=3):
  return dtlz2_pareto(N, n_obj)

def dtlz4_pareto(N, n_obj=3):
  return dtlz2_pareto(N, n_obj)

def dtlz5_pareto(N, n_obj=3):
  if n_obj == 2:
    theta = np.linspace(0, np.pi/2, N)
    f1 = np.cos(theta)
    f2 = np.sin(theta)
    return np.column_stack((f1, f2))
  elif n_obj == 3:
    angle = np.linspace(0, np.pi/2, N)
    f1 = np.cos(angle) * np.cos(np.pi/4)
    f2 = np.cos(angle) * np.sin(np.pi/4)
    f3 = np.sin(angle)
    return np.column_stack((f1.flatten(), f2.flatten(), f3.flatten()))

def dtlz6_pareto(N, n_obj=3):
  return dtlz5_pareto(N, n_obj)

def dtlz7_pareto(N, n_obj=3):
  if n_obj == 2:
    f1 = np.linspace(0, 1, N)
    f2 = lambda x: 2 * (2 - x/2*(1 + np.sin(3 * np.pi * x)))
    all_points = np.column_stack((f1, f2(f1)))
    ndf, _, _, _ = fast_non_dominated_sorting(all_points)
    non_dominated_points = all_points[ndf[0]]
    non_dominated_points = non_dominated_points[np.argsort(non_dominated_points[:, 0])]
    points = []
    step = 1/(N-1)
    for i in range(len(non_dominated_points) - 1):
      points.append(non_dominated_points[i])
      if np.linalg.norm(non_dominated_points[i] - non_dominated_points[i + 1]) > np.linalg.norm(non_dominated_points[i] - np.array([(i+1)*step, f2((i+1)*step)])):
        points.append(np.array([None]*n_obj))
    points.append(non_dominated_points[-1])
    return np.array(points)
  elif n_obj == 3:
    f = np.linspace(0, 1, N)
    f1, f2 = np.meshgrid(f, f)
    f3 = 2 * (3 - f1/2*(1 + np.sin(3 * np.pi * f1)) - f2/2*(1 + np.sin(3 * np.pi * f2)))
    all_points = np.column_stack((f1.flatten(), f2.flatten(), f3.flatten()))
    ndf, _, _, _ = fast_non_dominated_sorting(all_points)
    non_dominated_points = all_points[ndf[0]]
    return non_dominated_points[np.argsort(non_dominated_points[:, 0])]

def get_pareto(name, N, n_var, n_obj):
  pareto_set = {'zdt1':zdt1_pareto(N), 'zdt2':zdt2_pareto(N), 'zdt3':zdt3_pareto(N), 'zdt4':zdt4_pareto(N), 'zdt6':zdt6_pareto(N),
                'dtlz1':dtlz1_pareto(N, n_obj), 'dtlz2':dtlz2_pareto(N, n_obj), 'dtlz3':dtlz3_pareto(N, n_obj), 'dtlz4':dtlz4_pareto(N, n_obj), 'dtlz5':dtlz5_pareto(N, n_obj),
                'dtlz6':dtlz6_pareto(N, n_obj), 'dtlz7':dtlz7_pareto(N, n_obj),
                'wfg1':WFG1(n_var=n_var, n_obj=n_obj).pareto_front(n_pareto_points=N), 'wfg4':WFG4(n_var=n_var, n_obj=n_obj).pareto_front(n_pareto_points=N),
                'wfg5':WFG5(n_var=n_var, n_obj=n_obj).pareto_front(n_pareto_points=N), 'wfg6':WFG6(n_var=n_var, n_obj=n_obj).pareto_front(n_pareto_points=N),
                'wfg7':WFG7(n_var=n_var, n_obj=n_obj).pareto_front(n_pareto_points=N), 'wfg8':WFG8(n_var=n_var, n_obj=n_obj).pareto_front(n_pareto_points=N),
                'wfg9':WFG9(n_var=n_var, n_obj=n_obj).pareto_front()}
  if name in pareto_set:
    return pareto_set[name]
  elif name in {'wfg2', 'wfg3'}:
    pareto_set = {'wfg2':WFG2(n_var=n_var, n_obj=n_obj).pareto_front(n_pareto_points=N), 'wfg3':WFG3(n_var=n_var, n_obj=n_obj).pareto_front(n_pareto_points=N)}
    return pareto_set[name]
  else:
    raise ValueError(f"Pareto front for {name} not found.")