import numpy as np
cimport numpy as np

''' Function to optimize sigma evaluation '''
def sigma_evaluation_compiled(np.ndarray[np.float64_t, ndim=2] fitness_matrix,
                               np.ndarray[np.int32_t, ndim=2] pre_alocated_np_tril_indices):
    # Get the squared fitness matrix
    squared_fitnesses = np.square(fitness_matrix)
    # Sum each row of the fitness matrix
    sum_squared_fitnesses = np.sum(squared_fitnesses, axis=1, keepdims=True)
    # Obtain the indexes to perform the combinations (differences)
    row_indexes, col_indexes = pre_alocated_np_tril_indices
    differences = squared_fitnesses[:, row_indexes] - squared_fitnesses[:, col_indexes]
    # Calculate sigma values
    return differences / sum_squared_fitnesses