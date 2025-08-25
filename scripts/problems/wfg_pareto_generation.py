from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.problems.many.wfg import WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize

import numpy as np

def wfg_pareto_generation_by_algorithms(name, N, n_var, n_obj, wfg_k):
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
    nsga2 = NSGA2(pop_size=N, eliminate_duplicates=True)
    nsga3 = NSGA3(ref_dirs=ref_dirs, pop_size=N)
    ctaea = CTAEA(ref_dirs=ref_dirs)
    agemo = SMSEMOA(pop_size=N)
    moead = MOEAD(ref_dirs=ref_dirs, n_neighbors=N, decomposition='pbi')
    if name in {'wfg1', 'wfg4', 'wfg5', 'wfg6', 'wfg7', 'wfg8', 'wfg9'}:
        problem_class = {'wfg1': WFG1(n_var=n_var, n_obj=n_obj, k=wfg_k), 'wfg4': WFG4(n_var=n_var, n_obj=n_obj, k=wfg_k),
                         'wfg5': WFG5(n_var=n_var, n_obj=n_obj, k=wfg_k), 'wfg6': WFG6(n_var=n_var, n_obj=n_obj, k=wfg_k),
                         'wfg7': WFG7(n_var=n_var, n_obj=n_obj, k=wfg_k), 'wfg8': WFG8(n_var=n_var, n_obj=n_obj, k=wfg_k),
                         'wfg9': WFG9(n_var=n_var, n_obj=n_obj, k=wfg_k)}
    elif name in {'wfg2', 'wfg3'}:
        problem_class = {'wfg2': WFG2(n_var=n_var, n_obj=n_obj, k=wfg_k), 'wfg3': WFG3(n_var=n_var, n_obj=n_obj, k=wfg_k)}

    alg_list = [nsga2, nsga3, ctaea, agemo, moead]
    pareto_points = np.empty((0, n_obj))
    for i in range(5):
        res = minimize(problem_class[name],
                        algorithm=alg_list[i],
                        termination=('n_gen', 50))
        pareto_points = np.vstack((pareto_points, res.F))

    # return pareto_points
    return np.empty((0, n_obj))

# print(wfg_pareto_generation_by_algorithms('wfg1', 100, 10, 3, 6))