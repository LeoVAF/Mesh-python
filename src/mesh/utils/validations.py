import numpy as np
from typing import Iterable

def assert_type(var: any, var_name: str, expected_types: type | tuple) -> None:
    ''' Check if the ``var`` is one of the expected types.
    
    Args:
      var (:type:`any`): The input to be checked.
      var_name (:type:`str`): The name of the input.
      expected_types (:type:`type | tuple`): The type or tuple of types to be checked.

    Raises:
      TypeError: If the input is not one of the expected types.
    '''

    # Check the input type
    if not isinstance(var_name, str):
        raise TypeError(f'The parameter "var_name" of this function has type "{type(var_name)}", but expected str.')
    if not isinstance(expected_types, type) and not (isinstance(expected_types, tuple) and all(isinstance(item, type) for item in expected_types)):
        raise TypeError(f'The parameter "expected_types" of this function has type "{type(expected_types)}", but expected type or tuple of types.')
    
    # Check the var type
    if not isinstance(var, expected_types):
        raise TypeError(f'The input "{var_name}" has type "{type(var)}", but expected {expected_types}.')

def assert_type_optional(var: any, var_name: str, expected_types: type | tuple) -> None:
    ''' Check if the ``var`` is one of the expected types, if it is not ``None``.
    
    Args:
      var (:type:`any`): The input to be checked.
      var_name (:type:`str`): The name of the input.
      expected_types (:type:`type | tuple`): The type or tuple of types to be checked.

    Raises:
      TypeError: If the input is not one of the expected types.
    '''

    # Check the input type
    if not isinstance(var_name, str):
        raise TypeError(f'The parameter "var_name" of this function has type "{type(var_name)}", but expected str.')
    if not isinstance(expected_types, type) and not (isinstance(expected_types, tuple) and all(isinstance(item, type) for item in expected_types)):
        raise TypeError(f'The parameter "expected_types" of this function has type "{type(expected_types)}", but expected type or tuple of types.')
    
    # Check the var type
    if var is not None and not isinstance(var, expected_types):
        raise TypeError(f'The input "{var_name}" has type "{type(var)}", but expected {expected_types}.')

def assert_no_nan_in_np_array(arr: np.ndarray, arr_name: str) -> None:
    ''' Check if the ``arr`` does not have NaN values.
    
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
    ''' Check if the ``arr`` is a numpy array with the expected subtype.
    
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

def assert_index_np_array(idx_arr: np.ndarray, idx_arr_name: str, max_index: int) -> None:
    ''' Check if the ``idx_arr`` is an index array.
    
    Args:
      idx_arr (:type:`np.ndarray`): The numpy array to be checked.
      idx_arr_name (:type:`str`): The name of the input.
      max_index (:type:`int`): The maximum value to be compared with.

    Raises:
      TypeError: If the input numpy array does not have the expected subtype.
      ValueError: If the input numpy array has values out of bounds or it is not an one-dimensional array.
    '''

    # Check the input types
    assert_type(idx_arr, idx_arr_name, np.ndarray)
    assert_type(max_index, "max_index", (int, np.integer))

    # Check the array subtype
    if not np.issubdtype(idx_arr.dtype, int) and not np.issubdtype(idx_arr.dtype, np.integer):
        raise TypeError(f'The input "{idx_arr_name}" has dtype "{idx_arr.dtype}", but expected an type of integer.')

    # Check the array values
    if idx_arr.ndim != 1:
        raise ValueError(f'The input "{idx_arr_name}" must be one-dimensional.')
    if np.any(np.isnan(idx_arr)):
        raise ValueError(f'The input "{idx_arr_name}" has NaN values.')
    if np.any(idx_arr >= max_index) or np.any(idx_arr < -max_index):
        raise ValueError(f'The input "{idx_arr_name}" has indices out of bounds. The maximum index is {max_index}.')  

def is_greater_in_type(number: int | float | np.number, number_name: str, number_type: int | float | np.number, value: int | float | np.number) -> None:
    ''' Check if the ``number`` is of a respective type and if it is greater than ``value``.
    
    Args:
      number (:type:`int | float | np.number`): The input to be checked.
      number_name (:type:`str`): The name of the input.
      number_type (:type:`int | float | np.number`): The type to be checked.
      value (:type:`int | float | np.number`): The value to be compared with.

    Raises:
      TypeError: If the input is not of the expected type.
      ValueError: If the input is not greater than the ``value``.
    '''

    # Check the input types
    assert_type(number, number_name, (int, float, np.number))
    assert_type(number_type, 'number_type', type)
    assert_type(value, 'value', (int, float, np.number))

    # Check the number is the correct type
    assert_type(number, number_name, number_type)

    # Check if the input is greater than the value
    if number <= value:
        raise ValueError(f'The input "{number_name}" must be greater than {value}.')
    
def is_greater_in_type_optional(number: int | float | np.number, number_name: str, number_type: int | float | np.number, value: int | float | np.number) -> None:
    ''' Check if the ``number`` is of a respective type and if it is greater than ``value``, if it is not ``None``.
    
    Args:
      number (:type:`int | float | np.number`): The input to be checked.
      number_name (:type:`str`): The name of the input.
      number_type (:type:`int | float | np.number`): The type to be checked.
      value (:type:`int | float | np.number`): The value to be compared with.

    Raises:
      TypeError: If the input is not of the expected type.
      ValueError: If the input is not greater than the ``value``.
    '''

    # Check the input types
    assert_type(number, number_name, (int, float, np.number, type(None)))
    assert_type(number_type, 'number_type', type)
    assert_type(value, 'value', (int, float, np.number))

    # Check the number is the correct type
    assert_type(number, number_name, number_type)

    # Check if the input is greater than the value
    if number is not None and number <= value:
        raise ValueError(f'The input "{number_name}" must be greater than {value}.')

def is_between_inclusive(number: int | float | np.number, number_name: str, lower: int | float | np.number, upper: int | float | np.number) -> None:
    ''' Check if the ``number`` is between ``lower`` and ``upper``, inclusive.
    
    Args:
      number (:type:`int | float | np.number`): The input to be checked.
      number_name (:type:`str`): The name of the input.
      lower (:type:`int | float | np.number`): The lower bound.
      upper (:type:`int | float | np.number`): The upper bound.

    Raises:
      TypeError: If the input is not of the expected type
      ValueError: If the input is not between the bounds.
    '''

    # Check the input types
    assert_type(number, number_name, (int, float, np.number))
    assert_type(lower, 'lower', (int, float, np.number))
    assert_type(upper, 'upper', (int, float, np.number))

    # Check if the input is between the bounds
    if number < lower or number > upper:
        raise ValueError(f'The input "{number_name}" must be between {lower} and {upper}, inclusive.')
    

def is_in_options(option: any, option_name: str, options: Iterable) -> None:
    pass