from cython.parallel import prange
import numpy as np
cimport numpy as cnp

ctypedef cnp.int64_t DTYPE_t

# Main fnction for parallel evaluation
def parallel_fitness_evaluation(np.ndarray[DTYPE_t, ndim=2] X, fitness_function):
    cdef int n = X.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] results = np.zeros(n, dtype=np.float64)
    # Parallel loop using `prange`
    cdef int i
    for i in prange(n, nogil=True, num_threads=4):
        results[i] = fitness_function(X[i]) # Python functions need GIL
    return results