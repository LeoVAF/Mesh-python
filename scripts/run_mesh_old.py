from microgrid_old.techno_ka import techno_ka
from problems import get_problem
from mesh.MESH_old import *

from tqdm import tqdm
from pathlib import Path

from pathlib import Path
from tqdm import tqdm
from pygmo import fast_non_dominated_sorting, select_best_N_mo
from pickle import dump

import numpy as np

def main():
    Path("result").mkdir(parents=False, exist_ok=True)
    solar_data = np.genfromtxt('scripts/microgrid_old/seasonal_data/solreal.txt')
    wind_data = np.genfromtxt('scripts/microgrid_old/seasonal_data/wind_data.txt')
    load_ind = np.genfromtxt('scripts/microgrid_old/seasonal_data/loadind.txt')
    load_res = np.genfromtxt('scripts/microgrid_old/seasonal_data/loadres.txt')
    
    num_runs = 1 # Number of runs
    num_final_solutions = 300

    # LAG AGM(0) Li4Ti5O12(1) LiCoO2(2) LiFePO4(3) LiMnO2(4) LiNiCoMnO2(5) LiNiCoAlO2(6) LiPoly(7) NaNiCl(8) NaS(9) NiCd(10) NiMH(11) RFV(12) Zn/Br Redox(13)
    select_bat = 1
    bat_name = ['LAG', 'LTO', 'LCO', 'LFP', 'LMO', 'LNCMO', 'LNCAO', 'LPoly', 'NNC', 'NaS', 'NiC', 'NMH', 'RFV', 'ZnBr']
    experiment_name = 'zdt4'

    objectives_dim = 2 # Number of objectives
    optimizations_type = [False]*objectives_dim # Maximization (True) | Minimization (False) [LOLP, Price, RF]
    position_dim = 10 # Design space dimension

    func, position_min_value, position_max_value = get_problem(experiment_name, n_var=position_dim, n_obj=objectives_dim)

    # position_min_value = np.array([0]*position_dim) #[10, 1, 100] #, 0.2, 100] # Lower bound of problem [max PV generation, number of wind turbines, DoD, battery capacity]
    # position_max_value = np.array([1]*position_dim) #[450, 5, 500] #, 0.8, 500] # Upper bound of problem [max PV generation, number of wind turbines, DoD, battery capacity]
    # def func(args):
    #     #r = techno_ka(args[0], args[1], 0.8, args[2], select_bat, solar_data, wind_data, load_ind)[:objectives_dim]
    #     r = techno_ka(args[0], args[1], 0.8, args[2], select_bat, solar_data, wind_data, load_ind)[1:3]
    #     r[-1] = -r[-1] # Maximizando o fator renovável
    #     return r

    max_iterations = 0 #0 # Maximum number of iterations (not used if it is zero)
    max_fitness_eval = 15000 #100 # Maximum fitness evaluations
    population_size = 100 #10 # Population size
    num_final_solutions = population_size # Number of final solutions
    memory_size = population_size # Number of particles in memory
    memory_update_type = 1 # Not used yet

    communication_probability = 0.7 # Communication probability
    mutation_rate = 0.9 # Mutation rate
    personal_guide_array_size = 3 # Number of personal guides

    random_state = None # Defines a seed for random numbers (not used if it is None)

    # global_best_attribution_type = 2 and 3 is wrong
    global_best_attribution_type = 1 # 0 -> E1 | 1 -> E2 | 2 -> E3 | 3 -> E4
    Xr_pool_type = 1 # 0 -> V1 | 1 -> V2 | 2 -> V3
    DE_mutation_type = 0 # 0 -> DE\rand\1\Bin (D1) | 1 -> DE\rand\2\Bin (D2) | 2 -> DE/Best/1/Bin (D3) | 3 -> DE/Current-to-best/1/Bin (D4) | 4 -> DE/Current-to-rand/1/Bin (D5)
    crowd_distance_type = 0 # 0 -> Crowding Distance Tradicional (C1) | 1 -> Crowding Distance Suganthan (C2)

    config = f"E{global_best_attribution_type+1}V{Xr_pool_type+1}D{DE_mutation_type+1}_{experiment_name}"
    
    print(f"Running E{global_best_attribution_type+1}V{Xr_pool_type+1}D{DE_mutation_type+1}-{experiment_name} on MG")

    result = {}
    combined_F = None
    combined_P = None
    for i in tqdm(range(num_runs)):

        params = MESH_Params_old(objectives_dim,optimizations_type,max_iterations,max_fitness_eval,position_dim,position_max_value,position_min_value,population_size,memory_size,memory_update_type,global_best_attribution_type,DE_mutation_type,Xr_pool_type,crowd_distance_type,communication_probability,mutation_rate,personal_guide_array_size, random_state=random_state)
        MCDEEPSO = MESH_old(params,func)
        MCDEEPSO.log_memory = f"result/{config}_run{i+1}-MG"
        MCDEEPSO.run()
        
        ########################### Possible critical section ###########################
        with open(MCDEEPSO.log_memory+"-fit.txt", 'r') as file:
            fl = file.read().split("\n")[-2]
            Fit = np.array([v.split() for v in fl.split(",")], dtype=np.float64)
        with open(MCDEEPSO.log_memory+"-pos.txt", 'r') as file:
            fl = file.read().split("\n")[-2]
            Pos = np.array([v.split() for v in fl.split(",")], dtype=np.float64)
        #################################################################################

        result[i+1] = {"F":Fit, "P":Pos}
        # Accumulates the results of all executions
        if combined_F is None:
            combined_F = Fit
            combined_P = Pos
        else:
            combined_F = np.vstack((combined_F, Fit))
            combined_P = np.vstack((combined_P, Pos))

    # Getting the unique points
    unique_combined_P, unique_idxs = np.unique(combined_P, axis=0, return_index=True)
    unique_combined_F = combined_F[unique_idxs]
    # Sorting the vector Fit
    # Return: (non dominated front, domination list, domination counter, non domination ranks)
    if len(unique_combined_F) == 1:
        ndf = [[0]]
    else:
        ndf, _, _, _ = fast_non_dominated_sorting(points=unique_combined_F)
    n = min(len(ndf[0]), num_final_solutions)
    # Get the best indexes based on number of final solutions
    pareto_front = unique_combined_F[ndf[0]]
    best_idx = select_best_N_mo(pareto_front, n)
    result['combined'] = (unique_combined_P[ndf[0]][best_idx], pareto_front[best_idx])
    with open(f'result/{config}.pkl', 'wb') as file:
        dump(result, file)

if __name__ == '__main__':
    main()