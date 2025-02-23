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

import numpy as np

from pygmo import crowding_distance
from math import comb
from typing import Optional

# class PBest:
#     """
#     Represents the algorithm's particle personal best.
    
#     Attributes:
#         position (np.ndarray): An numpy matrix with the personal best position.
#         fitness (np.ndarray): An numpy matrix with the personal best fitness.
#     """
#     def __init__(self, position: np.ndarray, fitness: np.ndarray):
#         self.position = position.copy()
#         self.fitness = fitness.copy()

''' Represents the algorithm's population (an input for the respective particle) '''
class Population:
    ''' Initialize the instance'''
    def __init__(self,
                 population_size: int, # Number of particles
                 max_personal_guides: int, # Maximum personal guides
                 objective_dim: int, # Number of objectives
                 position_dim: int, # Number of variables
                 position_bounds, # Tuple with the lower and upper bounds of the position
                 velocity_bounds, # Tuple with the lower and upper bounds of the velocity
                 global_best_attribution_type): # Global best choice type
        
        # Particle's position initialized randomly
        self.position = np.empty((population_size, position_dim))
        self.position[:] = np.random.uniform(position_bounds[0], position_bounds[1], (population_size, position_dim))
        # Particle's velocity initialized randomly
        self.velocity = np.empty((population_size, position_dim))
        self.velocity[:] = np.random.uniform(velocity_bounds[0], velocity_bounds[1], (population_size, position_dim))
        self.fitness = np.full((population_size, objective_dim), np.inf) # Particle's fitness
        # self.crowd_distance = np.zeros(population_size) # Particle's crowd distance
        self.rank = np.empty(population_size, dtype=int) # Particle's rank
        if global_best_attribution_type < 2:
            self.sigma = np.full((population_size, comb(objective_dim, 2)), np.inf) # Sigma value for the sigma method
        self.global_best = np.zeros((population_size, position_dim)) # The global best position
        # Create a tensor of personal best information
        self.personal_best_list_pos = np.empty((population_size, max_personal_guides, position_dim))
        self.personal_best_list_fit = np.empty((population_size, max_personal_guides, objective_dim))
        # Repeat the population position for all personal best input
        self.personal_best_list_pos[:, :, :] = np.repeat(self.position[:, np.newaxis, :], max_personal_guides, axis=1)

class Memory:
    """
    Represents the algorithm's memory.

    Args:
        population (Particles): A `Particles` instance that represents the algorithm's population.
        pareto_frontier (np.ndarray[np.uint64]): A numpy array of the particle indices for the population matrices.
        memory_size (int): The maximum size of the memory.
    """
    
    def __init__(self, population: Population, pareto_frontier: np.ndarray[np.uint64], memory_size: int) -> None:
        self.position: np.ndarray[np.float64] 
        """ A numpy matrix with the memory position. """
        self.fitness: np.ndarray[np.float64]
        """ A numpy matrix with the memory fitness. """
        self.sigma: Optional[np.ndarray[np.float64]] = None
        """ A numpy matrix with the memory sigma values. This attribute is only used when the sigma method is used. """
        if(len(pareto_frontier) <= memory_size):
            self.position = population.position[pareto_frontier]
            self.fitness = population.fitness[pareto_frontier]
        else:
            # Calculate the crowd distance
            crowd_distances = crowding_distance(population.fitness[pareto_frontier])
            # Sort the Pareto frontier by the crowding distance
            idx = np.argpartition(crowd_distances, -memory_size)[-memory_size:]
            # Initialize the memory with the best solutions
            self.position = population.position[pareto_frontier[idx]]
            self.fitness = population.fitness[pareto_frontier[idx]]