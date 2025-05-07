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

from mesh.parameters import MeshParameters
from mesh.validations.python_validations import assert_type
from mesh.validations.numpy_validations import assert_np_vector_index

from pygmo import crowding_distance
from math import comb
from typing import Optional

import numpy as np

class Population:
    """ Represents the MESH population.

    Args:
        params (:class:`~mesh.parameters.MeshParameters`): The attributes :attr:`~mesh.parameters.MeshParameters.objective_dim`, :attr:`~mesh.parameters.MeshParameters.position_dim`, :attr:`~mesh.parameters.MeshParameters.position_min_value`, :attr:`~mesh.parameters.MeshParameters.position_max_value`, :attr:`~mesh.parameters.MeshParameters.velocity_min_value`, :attr:`~mesh.parameters.MeshParameters.velocity_max_value`, :attr:`~mesh.parameters.MeshParameters.population_size`, :attr:`~mesh.parameters.MeshParameters.global_best_attribution_type` and :attr:`~mesh.parameters.MeshParameters.max_personal_guides` are used to initialize the population.
    
    Raises:
        TypeError: If the input is not an instance of :class:`~mesh.parameters.MeshParameters`.
    """

    def __init__(self, params: MeshParameters):
        assert_type(params, 'params', MeshParameters)

        self.position: np.ndarray[np.float64, 2]
        ''' Numpy matrix with the particle's positions initialized randomly under uniform distribution. '''
        self.velocity: np.ndarray[np.float64, 2]
        ''' Numpy matrix with the particle's velocities initialized randomly under uniform distribution. '''
        self.fitness: np.ndarray[np.float64, 2]
        ''' Numpy matrix with the particle's fitnesses initialized with ``np.inf`` values. '''
        self.rank: np.ndarray[np.integer]
        ''' Numpy array with the particle's rank. '''
        self.sigma: Optional[np.ndarray[np.float64, 2]] = None
        ''' Numpy matrix for the sigma values. Initialized with ``np.inf`` values. Used only if the Sigma method is used. '''
        self.global_best: np.ndarray[np.float64, 2]
        ''' Numpy matrix with the best global position for each particle. '''
        self.personal_best_pos: np.ndarray[np.float64, 3]
        ''' Numpy tensor with a matrix of personal guide positions for each particle. Each matrix has :attr:`~mesh.parameters.MeshParameters.max_personal_guides` positions.
        Initialized with the respective particle's position repeated for all matrix entries. '''
        self.personal_best_fit: np.ndarray[np.float64, 3]
        ''' Numpy tensor with a matrix of personal guide fitnesses for each particle. Each matrix has :attr:`~mesh.parameters.MeshParameters.max_personal_guides` fitnesses. '''

        self.position = np.random.uniform(params.position_min_value, params.position_max_value, (params.population_size, params.position_dim))
        self.velocity = np.random.uniform(params.velocity_min_value, params.velocity_max_value, (params.population_size, params.position_dim))
        self.fitness = np.full((params.population_size, params.objective_dim), np.inf)
        self.rank= np.empty(params.population_size, dtype=int)
        if params.global_best_attribution_type < 2:
            self.sigma = np.full((params.population_size, comb(params.objective_dim, 2)), np.nan)
        self.global_best = np.full((params.population_size, params.position_dim), np.nan)
        self.personal_best_pos = np.repeat(self.position[:, np.newaxis, :], params.max_personal_guides, axis=1)
        self.personal_best_fit = np.empty((params.population_size, params.max_personal_guides, params.objective_dim))

class Memory:
    """
    Represents the MESH memory.

    Args:
        population (:class:`Population`): The attributes :attr:`~Population.position` and :attr:`~Population.fitness` are used to set the memory position and fitness.
        pareto_front (:type:`np.ndarray[np.integer]`): A numpy array of the particle indices for the population position and fitness matrices.
        params (:class:`~mesh.parameters.MeshParameters`): The attribute :attr:`~mesh.parameters.MeshParameters.memory_size` is used to limit the memory size.

    Raises:
        TypeError: If the input is not of the expected type.
    """
    
    def __init__(self, population: Population, pareto_front: np.ndarray[np.integer], params: MeshParameters) -> None:
        assert_type(population, 'population', Population)
        assert_np_vector_index(pareto_front, 'pareto_front', population.position.shape[0])
        assert_np_vector_index(pareto_front, 'pareto_front', population.fitness.shape[0])
        assert_type(params, 'params', MeshParameters)

        # Set the class attributes
        self.position: np.ndarray[np.float64, 2] 
        """ Numpy matrix with the memory position. """
        self.fitness: np.ndarray[np.float64, 2]
        """ Numpy matrix with the memory fitness. """
        self.sigma: Optional[np.ndarray[np.float64, 2]] = None
        """ Numpy matrix with the memory sigma values. This attribute is only used when the Sigma method is used. """

        if(len(pareto_front) <= params.memory_size):
            self.position = population.position[pareto_front]
            self.fitness = population.fitness[pareto_front]
        else:
            # Calculate the crowd distance
            crowd_distances = crowding_distance(population.fitness[pareto_front])
            # Sort the Pareto front by the crowding distance
            idx = np.argpartition(crowd_distances, -params.memory_size)[-params.memory_size:]
            # Initialize the memory with the best solutions
            self.position = population.position[pareto_front[idx]]
            self.fitness = population.fitness[pareto_front[idx]]
