from libc.stdlib cimport malloc, free
from libc.stdint cimport int64_t
from cython.parallel import prange
import numpy as np
cimport numpy as cnp

ctypedef cnp.int64_t DTYPE_t

# Função principal para avaliação paralela
def parallel_fitness_evaluation(np.ndarray[DTYPE_t, ndim=2] X):
    cdef int n = X.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] results = np.zeros(n, dtype=np.float64)

    # Loop paralelo usando `prange` (roda sem o GIL)
    cdef int i
    for i in prange(n, nogil=True, num_threads=4):
        results[i] = fitness_function(X[i])

    return results