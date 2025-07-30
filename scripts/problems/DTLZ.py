from pygmo import fast_non_dominated_sorting

import numpy as np

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
    return non_dominated_points
  elif n_obj == 3:
    f = np.linspace(0, 1, N)
    f1, f2 = np.meshgrid(f, f)
    f3 = 2 * (3 - f1/2*(1 + np.sin(3 * np.pi * f1)) - f2/2*(1 + np.sin(3 * np.pi * f2)))
    all_points = np.column_stack((f1.flatten(), f2.flatten(), f3.flatten()))
    ndf, _, _, _ = fast_non_dominated_sorting(all_points)
    non_dominated_points = all_points[ndf[0]]
    return non_dominated_points[np.argsort(non_dominated_points[:, 0])]