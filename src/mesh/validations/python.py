from typing import Iterable, Iterator

import numpy as np

def assert_type(var: any, var_name: str, expected_types: type | tuple, is_optional: bool = False) -> None:
  ''' Check if the ``var`` is one of the expected types.
  
  Args:
    var (:type:`any`): The input to be checked.
    var_name (:type:`str`): The name of the input.
    expected_types (:type:`type | tuple`): The type or tuple of types to be checked.
    is_optional (:type:`bool`): If the ``var`` is optional.

  Raises:
    TypeError: If the input is not one of the expected types.
  '''

  # Check the input types
  if not isinstance(var_name, str):
    raise TypeError(f'The parameter "var_name" of this function has type "{type(var_name)}", but expected str.')
  if not isinstance(expected_types, type) and not (isinstance(expected_types, tuple) and all(isinstance(item, type) for item in expected_types)):
    raise TypeError(f'The parameter "expected_types" of this function has type "{type(expected_types)}", but expected type or tuple of types.')
  if not isinstance(is_optional, bool):
    raise TypeError(f'The parameter "is_optional" of this function has type "{type(is_optional)}", but expected bool.')
  
  # Check the optional case
  if is_optional and var is None:
    return
  # Check the var type
  if not isinstance(var, expected_types):
    raise TypeError(f'The input "{var_name}" has type "{type(var)}", but expected {expected_types}.')

def is_greater_in_type(number: int | float | np.number, number_name: str, number_type: int | float | np.number, value: int | float | np.number, is_optional: bool = False) -> None:
  ''' Check if the ``number`` is of a respective type and if it is greater than ``value``.
  
  Args:
    number (:type:`int | float | np.number`): The input to be checked.
    number_name (:type:`str`): The name of the input.
    number_type (:type:`int | float | np.number`): The type to be checked.
    value (:type:`int | float | np.number`): The value to be compared with.
    is_optional (:type:`bool`): If the ``number`` is optional.

  Raises:
    TypeError: If the input is not of the expected type.
    ValueError: If the input is not greater than the ``value``.
  '''

  # Check the input types
  assert_type(number, number_name, (int, float, np.number), is_optional=is_optional)
  assert_type(number_type, 'number_type', type)
  assert_type(value, 'value', (int, float, np.number))

  # Check the optional case
  if is_optional and number is None:
    return
  # Check the number is the correct type
  assert_type(number, number_name, number_type)
  # Check if the input is greater than the value
  if number <= value:
    raise ValueError(f'The input "{number_name}" must be greater than {value}.')

def is_between_inclusive(number: int | float | np.number, number_name: str, lower: int | float | np.number, upper: int | float | np.number, is_optional: bool = False) -> None:
  ''' Check if the ``number`` is between ``lower`` and ``upper``, inclusive.
  
  Args:
    number (:type:`int | float | np.number`): The input to be checked.
    number_name (:type:`str`): The name of the input.
    lower (:type:`int | float | np.number`): The lower bound.
    upper (:type:`int | float | np.number`): The upper bound.
    is_optional (:type:`bool`): If the ``number`` is optional.

  Raises:
    ValueError: If the input is not between the bounds.
  '''

  # Check the input types
  assert_type(number, number_name, (int, float, np.number), is_optional=is_optional)
  assert_type(lower, 'lower', (int, float, np.number))
  assert_type(upper, 'upper', (int, float, np.number))

  # Check the optional case
  if is_optional and number is None:
    return
  # Check if the number is between the bounds
  if number < lower or number > upper:
    raise ValueError(f'The input "{number_name}" must be between {lower} and {upper}, inclusive.')
    

def is_in_options(option: any, option_name: str, options: Iterable | Iterator) -> None:
  ''' Check if the ``option`` is in the ``options``.
  
  Args:
    option (:type:`any`): The input to be checked.
    option_name (:type:`str`): The name of the input.
    options (:type:`Iterable | Iterator`): The options to be checked.

  Raises:
    TypeError: If the input does not have the expected subtype.
    ValueError: If the ``option`` is not in the options.
  '''

  # Check the input types
  assert_type(option_name, 'option_name', str)
  try:
    iter(options)
  except TypeError:
    raise TypeError(f'The parameter "options" of this function must be an iterable or iterator, but it is of type "{type(options)}".')

  # Check if the option is in the options
  if option not in options:
    raise ValueError(f'The input "{option_name}" must be one of the following options: {options}.')