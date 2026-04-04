from .python_validations import assert_type, is_greater_in_type

from numpy.typing import NDArray

import numpy as np

def assert_np_array_subtype(arr: np.ndarray, arr_name: str, subtype: type) -> None:
  ''' Checks if the ``arr`` is a numpy array with the expected subtype.
  
  Args:
    arr (:type:`np.ndarray`): The input to be checked.
    arr_name (:type:`str`): The name of the input.
    subtype (:type:`type`): The subtype to be checked.

  Raises:
    TypeError: If the input does not have the expected subtype.
  '''

  # Check the input type
  assert_type(arr_name, 'arr_name', str)
  assert_type(arr, arr_name, np.ndarray)
  assert_type(subtype, 'subtype', type)

  # Check the array subtype
  if not np.issubdtype(arr.dtype, subtype):
      raise TypeError(f'The input "{arr_name}" has dtype {arr.dtype}, but expected {subtype}.')

def assert_no_nan_in_np_array(arr: NDArray[np.number], arr_name: str) -> None:
  ''' Checks if the n-dimensional numpy array of numbers ``arr`` does not have NaN values.
  
  Args:
    arr (:type:`NDArray[np.number]`): The input to be checked.
    arr_name (:type:`str`): The name of the input.

  Raises:
    TypeError: If the input does not have the expected type
    ValueError: If the input has NaN values.
  '''

  # Check the input type
  assert_type(arr_name, 'arr_name', str)
  assert_type(arr, arr_name, np.ndarray)
  assert_np_array_subtype(arr, 'arr', np.number)

  # Check the array values
  if np.any(np.isnan(arr)):
    raise ValueError(f'The input "{arr_name}" has NaN values.')

def assert_np_array_for_operations(arr: NDArray[np.number], arr_name: str, shape: tuple) -> None:
  ''' Checks if the ``vec`` is a numpy array with the expected subtype for operations.
  
  Args:
    vec (:type:`NDArray[np.number]`): The input to be checked.
    vec_name (:type:`str`): The name of the input.
    size (:type:`tuple`): The expected shape of the numpy array.

  Raises:
    TypeError: If the input does not have the expected subtype.
    ValueError: If the numpy array has a shape different from the expected or NaN values.
  '''

  # Check the input types
  assert_type(arr_name, 'arr_name', str)
  assert_type(shape, 'shape', tuple)
  assert_no_nan_in_np_array(arr, arr_name)

  # Check the vector shape
  if arr.ndim != len(shape):
    raise ValueError(f'The input "{arr_name}" must have {len(shape)} dimension(s).')
  for i in range(len(shape)):
    if not isinstance(shape[i], (int, np.integer)):
      raise TypeError(f'The parameter "shape" has a non-integer in the position {i}.')
    if arr.shape[i] != shape[i]:
      raise ValueError(f'The input "{arr_name}" has {arr.shape[i]} element(s) in the axis {i}, but expected {shape[i]} element(s).')

def assert_np_vectors_for_boundary(lower: NDArray[np.number], lower_name: str, upper: NDArray[np.number], upper_name: str, size: int) -> None:
  ''' Checks if the ``lower`` and ``upper`` are boundary numpy vectors.
  
  Args:
    lower (:type:`NDArray[np.number]`): The lower boundary numpy vector.
    lower_name (:type:`str`): The name of the lower boundary numpy vector.
    upper (:type:`NDArray[np.number]`): The upper boundary numpyvector.
    upper_name (:type:`str`): The name of the upper boundary numpyvector.
    size (:type:`int | np.integer`): The expected size of the boundary numpy vectors.

  Raises:
    TypeError: If the input does not have the expected subtype.
    ValueError: If the lower boundary vector has an element greater than the respective element in the upper boundary numpy vector.
  '''

  # Check the input types
  assert_type(lower_name, 'lower_name', str)
  assert_type(upper_name, 'upper_name', str)
  is_greater_in_type(size, 'size', int, 0)
  assert_no_nan_in_np_array(lower, lower_name)
  assert_no_nan_in_np_array(upper, upper_name)

  # Check the boundary vectors
  if lower.ndim != 1:
    raise ValueError(f'The input "{lower_name}" must be one-dimensional.')
  if lower.shape[0] != size:
     raise ValueError(f'The input "{lower_name}" has size {lower.shape[0]}, but expected {size}.')
  if upper.ndim != 1:
    raise ValueError(f'The input "{upper_name}" must be one-dimensional.')
  if upper.shape[0] != size:
     raise ValueError(f'The input "{upper_name}" has size {upper.shape[0]}, but expected {size}.')
  if np.any(lower > upper):
	  raise ValueError(f'The input "{lower_name}" must be less than "{upper_name}".')

def assert_np_vector_index(idx_vec: np.ndarray, idx_vec_name: str, max_index: int) -> None:
  ''' Checks if the ``idx_arr`` is an index numpy vector.
  
  Args:
    idx_vec (:type:`np.ndarray`): The numpy vector to be checked.
    idx_vec_name (:type:`str`): The name of the input.
    max_index (:type:`int`): The maximum index value allowed.

  Raises:
    TypeError: If the input numpy vector does not have the expected subtype.
    ValueError: If the input numpy vector has values out of bounds or it is not an one-dimensional array, a size different from the expected or NaN values.
  '''

  # Check the input types
  assert_type(idx_vec_name, 'idx_vec_name', str)
  assert_type(idx_vec, idx_vec_name, np.ndarray)
  assert_type(max_index, "max_index", int)

  # Check the vector subtype
  if not np.issubdtype(idx_vec.dtype, int):
    raise TypeError(f'The input "{idx_vec_name}" has dtype {idx_vec.dtype}, but expected (<class \'int\'>, <class \'numpy.integer\'>).')

  # Check the vector values
  if idx_vec.ndim != 1:
    raise ValueError(f'The input "{idx_vec_name}" must be one-dimensional.')
  if np.any(idx_vec >= max_index) or np.any(idx_vec < -max_index):
    raise ValueError(f'The input "{idx_vec_name}" has indices out of bounds. The maximum index is {max_index}.')