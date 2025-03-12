from typing import Callable

from validations.python import assert_type, is_greater_in_type

import numpy as np
import inspect

def assert_no_nan_in_np_array(arr: np.ndarray, arr_name: str) -> None:
  ''' Checks if the ``arr`` does not have NaN values.
  
  Args:
    arr (:type:`np.ndarray`): The input to be checked.
    arr_name (:type:`str`): The name of the input.

  Raises:
    ValueError: If the input has NaN values.
  '''

  # Check the input type
  assert_type(arr, arr_name, np.ndarray)

  # Check the array values
  if np.any(np.isnan(arr)):
    raise ValueError(f'The input "{arr_name}" has NaN values.')

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
  assert_type(arr, arr_name, np.ndarray)
  assert_type(subtype, 'subtype', type)

  # Check the array subtype
  if not np.issubdtype(arr.dtype, subtype):
      raise TypeError(f'The input "{arr_name}" has dtype "{arr.dtype}", but expected {subtype}.')

def assert_np_vector_for_operations(vec: np.ndarray[np.number], vec_name: str, size: int | np.integer) -> None:
  ''' Checks if the ``vec`` is a numpy vector with the expected subtype for operations.
  
  Args:
    vec (:type:`np.ndarray[np.number]`): The input to be checked.
    vec_name (:type:`str`): The name of the input.
    size (:type:`int | np.integer`): The expected size of the vector.

  Raises:
    TypeError: If the input does not have the expected subtype.
    ValueError: If the numpy vector has a size different from the expected.
  '''

  # Check the input types
  assert_type(size, 'size', (int, np.integer))
  assert_np_array_subtype(vec, vec_name, np.number)
  assert_no_nan_in_np_array(vec, vec_name)

  # Check the vector size
  if vec.ndim != 1:
    raise ValueError(f'The input "{vec_name}" must be one-dimensional.')
  if vec.size != size:
	  raise ValueError(f'The input "{vec_name}" must have size {size}.')

def assert_np_vectors_for_boundary(lower: np.ndarray[np.number], lower_name: str, upper: np.ndarray[np.number], upper_name: str, size: int | np.integer) -> None:
  ''' Checks if the ``lower`` and ``upper`` are boundary numpy vectors.
  
  Args:
    lower (:type:`np.ndarray[np.number]`): The lower boundary numpy vector.
    lower_name (:type:`str`): The name of the lower boundary numpy vector.
    upper (:type:`np.ndarray[np.number]`): The upper boundary numpyvector.
    upper_name (:type:`str`): The name of the upper boundary numpyvector.
    size (:type:`int | np.integer`): The expected size of the boundary numpy vectors.

  Raises:
    TypeError: If the input does not have the expected subtype.
    ValueError: If the lower boundary vector has an element greater than the respective element in the upper boundary numpy vector.
  '''

  # Check the input types
  assert_np_vector_for_operations(lower, lower_name, size)
  assert_np_vector_for_operations(upper, upper_name, size)

  # Check the boundary vectors
  if np.any(lower > upper):
	  raise ValueError(f'The input "{lower_name}" must be less than "{upper_name}".')

def assert_np_vector_index(idx_vec: np.ndarray, idx_vec_name: str, max_index: int) -> None:
  ''' Checks if the ``idx_arr`` is an index numpy vector.
  
  Args:
    idx_vec (:type:`np.ndarray`): The numpy vector to be checked.
    idx_vec_name (:type:`str`): The name of the input.
    max_index (:type:`int`): The maximum value to be compared with.

  Raises:
    TypeError: If the input numpy vector does not have the expected subtype.
    ValueError: If the input numpy vector has values out of bounds or it is not an one-dimensional array.
  '''

  # Check the input types
  assert_type(idx_vec, idx_vec_name, np.ndarray)
  assert_type(max_index, "max_index", (int, np.integer))

  # Check the vector subtype
  if not np.issubdtype(idx_vec.dtype, int) and not np.issubdtype(idx_vec.dtype, np.integer):
    raise TypeError(f'The input "{idx_vec_name}" has dtype "{idx_vec.dtype}", but expected an type of integer.')

  # Check the vector values
  if idx_vec.ndim != 1:
    raise ValueError(f'The input "{idx_vec_name}" must be one-dimensional.')
  if np.any(np.isnan(idx_vec)):
    raise ValueError(f'The input "{idx_vec_name}" has NaN values.')
  if np.any(idx_vec >= max_index) or np.any(idx_vec < -max_index):
    raise ValueError(f'The input "{idx_vec_name}" has indices out of bounds. The maximum index is {max_index}.')


def is_fitness_function(fit_func: Callable[[np.ndarray[np.number]], np.ndarray[np.number]], fit_func_name: str, position_dim: int | np.integer, objective_dim: int | np.integer) -> None:
  ''' Checks if the fitness function is correctly annotated.
  
  Args:
    fit_func (:type:`Callable[[np.ndarray[np.number]], np.ndarray[np.number]]`): The fitness function to be checked.
    fit_func_name (:type:`str`): The name of the fitness function.
    position_dim (:type:`int | np.integer`): The design space dimension.
    objective_dim (:type:`int | np.integer`): The number of objectives.
    
    Raises:
    TypeError: If the input is not of the expected type.
    ValueError: If the input is not a fitness function.
  '''

  # Check the input types
  if not callable(fit_func):
    raise TypeError(f'The input "{fit_func_name}" must be a callable, but it is of type "{type(fit_func)}".')
  assert_type(fit_func_name, 'fit_func_name', str)
  is_greater_in_type(position_dim, 'position_dim', (int, np.integer), 0)
  is_greater_in_type(objective_dim, 'objective_dim', (int, np.integer), 0)

  # Get the fitness function annotation
  annotation = inspect.signature(fit_func)
  fit_func_args = list(annotation.parameters.values())

  # Check the number of arguments without default values
  arg_non_default_list = [arg.default != inspect.Parameter.empty for arg in fit_func_args]
  
  # Check the fitness function arguments
  if not arg_non_default_list:
    raise ValueError('The fitness function must have at least one argument without default values.')
  else:
    mandatory_args = arg_non_default_list.count(True)
    # Check fitness function possibilities
    class_method = fit_func_args[0].name == 'cls' and isinstance(fit_func, classmethod) and mandatory_args == 2
    static_method = isinstance(fit_func, staticmethod) and mandatory_args == 1
    instance_method = fit_func_args[0].name == 'self' and mandatory_args == 2
    pure_function = mandatory_args == 1
    if class_method or static_method or instance_method or pure_function:
      raise ValueError(f'The fitness function must have only one argument without default values, but it has "{arg_non_default_list.count(True)}".')

  # Check the type of the arguments
  try:
    arr_test = np.array([0.0] * position_dim)
    # Check the return type
    result = fit_func(arr_test)
  except Exception:
    raise ValueError(f'The fitness function "{fit_func_name}" must receive a numpy array with the design space dimension ({position_dim}) and return a numpy array with the objective dimension ({objective_dim}).')
  # Check if the return is a numpy array with the correct dimensions and subtype
  assert_np_vector_for_operations(result, 'result', objective_dim)