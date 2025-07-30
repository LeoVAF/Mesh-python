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