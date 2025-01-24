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
from collections import deque

''' Represents the algorithm's particle personal best '''
class PBest:
    def __init__(self, position, fitness):
        self.position = position.copy()
        self.fitness = fitness.copy()

''' Represents the algorithm's memory '''
class Memory:
    ''' Initialize the instance '''
    def __init__(self, global_best_attribution_type):
        self.position = None # Memory position
        self.fitness = None # Memory fitness
        if global_best_attribution_type < 2:
            self.sigma = None # Memory sigma value
    
    ''' Initialize the memory from the Pareto frontier '''
    def init(self, population, pareto_frontier, memory_size):
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
        return self

''' Represents the algorithm's population (an input for the respective particle) '''
class Particles:
    ''' Initialize the instance'''
    def __init__(self,
                 population_size, # Number of particles
                 max_personal_guides, # Maximum personal guides
                 objective_dim, # Number of objectives
                 position_dim, # Number of variables
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
        # List of personal best
        self.personal_best_list = np.empty(population_size, dtype=object)
        self.personal_best_list[:] = [deque([PBest(self.position[i], self.fitness[i])], maxlen=max_personal_guides) for i in range(population_size)]