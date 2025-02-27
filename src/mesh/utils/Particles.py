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
from typing import Optional, Literal

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

class Population:
    """
    Represents the algorithm's population.

    Args:
        objective_dim (``int``): Number of objectives in the problem.
        position_dim (``int``): Number of variables in the problem.
        position_bounds (``tuple[np.ndarray[np.float64], np.ndarray[np.float64]]``): Tuple with the lower (first numpy array) and upper (second numpy array) bounds of the positions.
        velocity_bounds (``tuple[np.ndarray[np.float64], np.ndarray[np.float64]]``): Tuple with the lower (first numpy array) and upper (second numpy array) bounds of the velocities.
        population_size (``int``): Number of particles.
        global_best_attribution_type (``{0, 1, 2, 3}``): Global best selection method. The options are:

            - ``0``: Applies Sigma method in memory to select the global best.
            - ``1``: Applies Sigma method in fronts to select the global best. Each particle will select its global best from the next front. Particles in Pareto front will select the global best from memory.
            - ``2``: Chooses randomly under uniform distribution a particle from memory.
            - ``3``: Chooses randomly under uniform distribution a particle from fronts. Each particle will select its global best from the next front. Particles in Pareto front will select the global best from memory.
        max_personal_guides (``int``): Number of maximum personal guides.
    """

    def __init__(self,
                 objective_dim: int,
                 position_dim: int,
                 position_bounds: tuple[np.ndarray[np.float64], np.ndarray[np.float64]],
                 velocity_bounds: tuple[np.ndarray[np.float64], np.ndarray[np.float64]],
                 population_size: int,
                 global_best_attribution_type: Literal[0, 1, 2, 3],
                 max_personal_guides: int) -> None:
        self.position: np.ndarray[np.float64, 2]
        ''' Numpy matrix with the particle's positions initialized randomly under uniform distribution. '''
        self.velocity: np.ndarray[np.float64, 2]
        ''' Numpy matrix with the particle's velocities initialized randomly under uniform distribution. '''
        self.fitness: np.ndarray[np.float64, 2]
        ''' Numpy matrix with the particle's fitnesses initialized with ``np.inf`` values. '''
        self.rank: np.ndarray[np.float64]
        ''' Numpy array with the particle's rank. '''
        self.sigma: np.ndarray[np.float64, 2]
        ''' Numpy matrix for the sigma values. Initialized with ``np.inf`` values. Used only if the sigma method is used. '''
        self.global_best: np.ndarray[np.float64, 2]
        ''' Numpy matrix with the best global position for each particle. '''
        self.personal_best_list_pos: np.ndarray[np.float64, 3]
        ''' Numpy tensor with a matrix of personal guide positions for each particle. Each matrix has ``max_personal_guides`` positions.
        Initialized with the respective particle's position repeated for all matrix entries. '''
        self.personal_best_list_fit: np.ndarray[np.float64, 3]
        ''' Numpy tensor with a matrix of personal guide fitnesses for each particle. Each matrix has ``max_personal_guides`` fitnesses. '''

        self.position = np.random.uniform(position_bounds[0], position_bounds[1], (population_size, position_dim))
        self.velocity = np.random.uniform(velocity_bounds[0], velocity_bounds[1], (population_size, position_dim))
        self.fitness = np.full((population_size, objective_dim), np.inf)
        self.rank= np.empty(population_size, dtype=int)
        if global_best_attribution_type < 2:
            self.sigma = np.full((population_size, comb(objective_dim, 2)), np.inf)
        self.global_best = np.empty((population_size, position_dim))
        self.personal_best_list_pos = np.repeat(self.position[:, np.newaxis, :], max_personal_guides, axis=1)
        self.personal_best_list_fit = np.empty((population_size, max_personal_guides, objective_dim))

class Memory:
    """
    Represents the algorithm's memory.

    Args:
        population (``Particles``): A ``Particles`` instance that represents the algorithm's population.
        pareto_frontier (``np.ndarray[np.uint64]``): A numpy array of the particle indices for the population matrices.
        memory_size (``int``): The maximum size of the memory.
    """
    
    def __init__(self, population: Population, pareto_frontier: np.ndarray[np.uint64], memory_size: int) -> None:
        self.position: np.ndarray[np.float64, 2] 
        """ Numpy matrix with the memory position. """
        self.fitness: np.ndarray[np.float64, 2]
        """ Numpy matrix with the memory fitness. """
        self.sigma: Optional[np.ndarray[np.float64]] = None
        """ Numpy matrix with the memory sigma values. This attribute is only used when the sigma method is used. """

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