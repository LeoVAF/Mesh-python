from mesh.core import Mesh
from mesh.parameters import MeshParameters
from problems.microgrid_function import microgrid_function
from problems.benchmark_problems import get_problem

from pathlib import Path
from tqdm import tqdm
from pygmo import fast_non_dominated_sorting, select_best_N_mo # type: ignore
from pickle import dump

import numpy as np

def main():
    Path("./scripts/results/").mkdir(parents=False, exist_ok=True)

    num_runs = 1 # Number of runs
    num_proc = None # Number of processes to execute the fitness function in parallel

    objective_dim = 3 # Number of objectives
    position_dim = 4 # Design space dimension
    
    # Benchmark problems
    experiment_name = 'dtlz1'
    func, position_min_value, position_max_value = get_problem(experiment_name, n_var=position_dim, n_obj=objective_dim, wfg_k=objective_dim-1)
    
    ############### Microgrid function ###############
    # select_bat = 0 # Lead_Acid(0) Li-ion(1) ZEBRA(2) NaS(3) NiCd(4) NiMH(5) RFV(6) ZnBr(7)
    # position_min_value = np.array([10, 10, 10]) # Lower bound of problem [max PV generation, max WT generation , battery capacity]
    # position_max_value = np.array([450, 450, 500]) # Upper bound of problem [max PV generation, max WT generation, battery capacity]
    ''' ###### '''
    # select_bat = 0 # LAG AGM(0) Li4Ti5O12(1) LiCoO2(2) LiFePO4(3) LiMnO2(4) LiNiCoMnO2(5) LiNiCoAlO2(6) LiPoly(7) NaNiCl(8) NaS(9) NiCd(10) NiMH(11) RFV(12) Zn/Br Redox(13)
    # bat_name = ['LAG', 'LTO', 'LCO', 'LFP', 'LMO', 'LNCMO', 'LNCAO', 'LPoly', 'NNC', 'NaS', 'NiC', 'NMH', 'RFV', 'ZnBr']
    ''' ###### '''
    # load = np.genfromtxt('scripts/seasonal_data/load.txt')
    # temperature = np.genfromtxt('scripts/seasonal_data/temperature.txt')
    # solar_data = np.genfromtxt('scripts/seasonal_data/irradiance.txt')
    # wind_data = np.genfromtxt('scripts/seasonal_data/wind.txt')
    # bat_name = ['Lead_Acid', 'Li-ion', 'ZEBRA', 'NaS', 'NiCd', 'NiMH', 'RFV', 'ZnBr']
    # experiment_name = bat_name[select_bat]
    # func = lambda args: microgrid_function(args[0], args[1], args[2], select_bat, load, temperature, solar_data, wind_data)
    ################ Microgrid function ###############

    max_iterations = None # Maximum number of iterations
    max_fitness_eval = 10000 # Maximum fitness evaluations
    population_size = 100 # Population size
    memory_size = population_size # Maximum number of particles in memory
    communication_probability = 0.2 # Communication probability
    mutation_rate = 0.8 # Mutation rate
    personal_guide_array_size = 1 # Number of personal guides
    random_state = None # Defines a seed for random numbers (not used if it is None)

    global_guide_method = 0 # 0 -> Sigma method (G1) | 1 -> Sigma Method in fronts (G2)
    dm_pool_type = 0 # 0 -> Sampling from memory (S1) | 1 -> Sampling from population (S2) | 2 -> Sampling from memory and population (S3)
    dm_operation_type = 0 # 0 -> DE\rand\1\Bin (D1) | 1 -> DE\rand\2\Bin (D2) | 2 -> DE/Best/1/Bin (D3) | 3 -> DE/Current-to-best/1/Bin (D4) | 4 -> DE/Current-to-rand/1/Bin (D5)

    config = f"MESH_G{global_guide_method+1}S{dm_pool_type+1}D{dm_operation_type+1}_{experiment_name}"
    print(f"Running MESH G{global_guide_method+1}S{dm_pool_type+1}D{dm_operation_type+1}-{experiment_name}")
    result = {}
    combined_F = None
    combined_P = None
    for i in tqdm(range(num_runs)):
        params = MeshParameters(objective_dim,
                                position_dim, position_min_value, position_max_value,
                                population_size, memory_size=memory_size,
                                global_guide_method=global_guide_method,
                                dm_pool_type=dm_pool_type,
                                dm_operation_type=dm_operation_type,
                                communication_probability=communication_probability, mutation_rate=mutation_rate,
                                max_gen=max_iterations, max_fit_eval=max_fitness_eval,
                                max_personal_guides=personal_guide_array_size,
                                random_state=random_state)
        
        log = None # f"./scripts/results/{config}_run{i+1}"
        mesh = Mesh(params, func, log_memory=log, num_proc=num_proc)
        mesh.run()
        Pos, Fit = mesh.get_results()
        result[i+1] = {"F":Fit, "P":Pos}
        # Accumulates the results of all executions
        if combined_F is None:
            combined_P = Pos
            combined_F = Fit
        else:
            combined_P = np.vstack((combined_P, Pos))
            combined_F = np.vstack((combined_F, Fit))
    # Getting the unique points
    unique_combined_P, unique_idxs = np.unique(combined_P, axis=0, return_index=True)
    unique_combined_F = combined_F[unique_idxs]
    # Sorting the vector Fit with unique values
    # Return: (non dominated front, domination list, domination counter, non domination ranks)
    if len(unique_combined_F) == 1:
        best_idxs = np.array([0])
        ndf = [np.array([0])]
    else:
        best_idxs = select_best_N_mo(unique_combined_F, population_size)
        ndf, _, _, _ = fast_non_dominated_sorting(points=unique_combined_F[best_idxs])
    # Get the best indexes based on number of final solutions
    ndf_idxs = ndf[0]
    pareto_front = unique_combined_F[best_idxs][ndf_idxs]
    result['combined'] = (unique_combined_P[best_idxs][ndf_idxs], pareto_front)
    with open(f'./scripts/results/{config}_{objective_dim}_{position_dim}.pkl', 'wb') as file:
        dump(result, file)

if __name__ == '__main__':
    main()