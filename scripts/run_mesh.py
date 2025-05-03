###########################################################################
# Lucas Braga, MS.c. (email: lucas.braga.deo@gmail.com )
# Gabriel Matos Leite, PhD candidate (email: gmatos@cos.ufrj.br)
# Carolina Marcelino, PhD (email: carolimarc@ic.ufrj.br)
# June 16, 2021
###########################################################################
# Copyright (c) 2021, Lucas Braga, Gabriel Matos Leite, Carolina Marcelino
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in
#      the documentation and/or other materials provided with the
#      distribution
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS USING 
# THE CREATIVE COMMONS LICENSE: CC BY-NC-ND "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from mesh.core import Mesh
from mesh.parameters import MeshParameters
from microgrid.techno_ka import techno_ka

from pathlib import Path
from pymoo.problems import get_problem
from tqdm import tqdm
from pygmo import fast_non_dominated_sorting, select_best_N_mo
from pickle import dump

import numpy as np

def main():
    Path("result").mkdir(parents=False, exist_ok=True)
    solar_data = np.genfromtxt('scripts/microgrid/seasonal_data/solreal.txt')
    wind_data = np.genfromtxt('scripts/microgrid/seasonal_data/wind_data.txt')
    load_ind = np.genfromtxt('scripts/microgrid/seasonal_data/loadind.txt')
    load_res = np.genfromtxt('scripts/microgrid/seasonal_data/loadres.txt')
    
    num_runs = 1 # Number of runs

    # LAG AGM(0) Li4Ti5O12(1) LiCoO2(2) LiFePO4(3) LiMnO2(4) LiNiCoMnO2(5) LiNiCoAlO2(6) LiPoly(7) NaNiCl(8) NaS(9) NiCd(10) NiMH(11) RFV(12) Zn/Br Redox(13)
    select_bat = 0
    bat_name = ['LAG', 'LTO', 'LCO', 'LFP', 'LMO', 'LNCMO', 'LNCAO', 'LPoly', 'NNC', 'NaS', 'NiC', 'NMH', 'RFV', 'ZnBr']
    # experiment_name = bat_name[select_bat]
    experiment_name = 'dtlz1'

    objective_dim = 2 # Number of objectives
    position_dim = 10 # Design space dimension
    position_min_value = np.array([0]*position_dim) # Lower bound of problem
    position_max_value = np.array([1]*position_dim) # Upper bound of problem
    # position_min_value = np.array([10, 1, 100]) # Lower bound of problem [max PV generation, number of wind turbines, battery capacity]
    # position_max_value = np.array([450, 5, 500]) # Upper bound of problem [max PV generation, number of wind turbines, battery capacity]
    
    # def func(args):
    #     r = techno_ka(args[0], args[1], 0.8, args[2], select_bat, solar_data, wind_data, load_ind)[:objective_dim]
    #     #r = techno_ka(args[0], args[1], 0.8, args[2], select_bat, solar_data, wind_data, load_ind)[1:3]
    #     r[-1] = -r[-1] # Maximizing renewable factor
    #     return r
    
    if experiment_name in {'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6'}:
        func = get_problem(experiment_name, n_var=position_dim).evaluate
    else:
        func = get_problem(experiment_name, n_var=position_dim, n_obj=objective_dim).evaluate

    num_proc = None # Number of processes to execute the fitness function in parallel

    max_iterations = 0 # Maximum number of iterations (not used if it less than one)
    max_fitness_eval = 15000 # Maximum fitness evaluations (not used if it is less than one)
    population_size = 50 # Population size
    num_final_solutions = population_size # Number of final solutions
    memory_size = population_size # Maximum number of particles in memory

    communication_probability =  0.2 # Communication probability
    mutation_rate = 0.5 # Mutation rate
    personal_guide_array_size = 3 # Number of personal guides

    random_state = None # Defines a seed for random numbers (not used if it is None)

    global_best_attribution_type = 0 # 0 -> E1 | 1 -> E2 | 2 -> E3 | 3 -> E4
    dm_pool_type = 0 # 0 -> V1 | 1 -> V2 | 2 -> V3
    dm_operation_type = 0 # 0 -> DE\rand\1\Bin (D1) | 1 -> DE\rand\2\Bin (D2) | 2 -> DE/Best/1/Bin (D3) | 3 -> DE/Current-to-best/1/Bin (D4) | 4 -> DE/Current-to-rand/1/Bin (D5)

    config = f"E{global_best_attribution_type+1}V{dm_pool_type+1}D{dm_operation_type+1}_{experiment_name}"
    print(f"Running E{global_best_attribution_type+1}V{dm_pool_type+1}D{dm_operation_type+1}-{experiment_name} on MG")
    result = {}
    combined_F = None
    combined_P = None
    for i in tqdm(range(num_runs)):
        params = MeshParameters(objective_dim,
                                position_dim, position_min_value, position_max_value, 
                                population_size, memory_size=memory_size,
                                global_best_attribution_type=global_best_attribution_type,
                                dm_pool_type=dm_pool_type,
                                dm_operation_type=dm_operation_type,
                                communication_probability=communication_probability, mutation_rate=mutation_rate,
                                max_gen=max_iterations, max_fit_eval=max_fitness_eval,
                                max_personal_guides=personal_guide_array_size,
                                random_state=random_state)
        
        log = None # f"result/{config}_run{i+1}"
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
    # Sorting the vector Fit
    # Return: (non dominated front, domination list, domination counter, non domination ranks)
    if len(combined_F) == 1:
        ndf = [[0]]
    else:
        ndf, _, _, _ = fast_non_dominated_sorting(points=combined_F)
    n = min(num_final_solutions, len(ndf[0]))
    # Get the best indexes based on number of final solutions
    best_idx = select_best_N_mo(combined_F, n)
    result['combined'] = (combined_P[best_idx], combined_F[best_idx])
    with open(f'result/{config}.pkl', 'wb') as file:
        dump(result, file)

if __name__ == '__main__':
    main()