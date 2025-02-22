import numpy as np
cimport numpy as np

''' Function to optimize sigma evaluation '''
def sigma_evaluation_compiled(np.ndarray[np.float64_t, ndim=2] fitness_matrix, np.ndarray[np.int64_t, ndim=1] row_np_tril_indices, np.ndarray[np.int64_t, ndim=1] col_np_tril_indices):
    # Get the squared fitness matrix
    squared_fitnesses = np.square(fitness_matrix)
    # Sum each row of the fitness matrix
    sum_squared_fitnesses = np.sum(squared_fitnesses, axis=1, keepdims=True)
    # Obtain the np.tril_indices to perform the combinations (differences)
    differences = squared_fitnesses[:, row_np_tril_indices] - squared_fitnesses[:, col_np_tril_indices]
    # Calculate sigma values
    return differences / sum_squared_fitnesses