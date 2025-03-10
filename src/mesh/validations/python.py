from typing import Iterable, Iterator, Callable

import numpy as np
import inspect

def assert_type(var: any, var_name: str, expected_types: type | tuple, is_optional: bool = False) -> None:
  ''' Checks if the ``var`` is one of the expected types.
  
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

def assert_type_or_falsy(value: any, value_name: str, expected_types: type | tuple, is_optional: bool = False) -> None:
  ''' Checks if the ``value`` has the correct type or is a falsy value.
  
  Args:
    value (:type:`any`): The input to be checked.
    value_name (:type:`str`): The name of the input.
    expected_types (:type:`type | tuple`): The type or tuple of types to be checked.
    is_optional (:type:`bool`): If the ``value`` is optional.

  Raises:
    TypeError: If the input is not one of the expected types.
  '''

  # Check if the value has the correct type or is a falsy value
  if value:
    assert_type(value, value_name, expected_types, is_optional=is_optional)

def is_greater_in_type(number: int | float | np.number, number_name: str, number_type: int | float | np.number | tuple, value: int | float | np.number, is_optional: bool = False) -> None:
  ''' Checks if the ``number`` is of a respective type and if it is greater than ``value``.
  
  Args:
    number (:type:`int | float | np.number`): The input to be checked.
    number_name (:type:`str`): The name of the input.
    number_type (:type:`int | float | np.number | tuple`): The type to be checked.
    value (:type:`int | float | np.number`): The value to be compared with.
    is_optional (:type:`bool`): If the ``number`` is optional.

  Raises:
    TypeError: If the input is not of the expected type.
    ValueError: If the input is not greater than the ``value``.
  '''

  # Check the input types
  assert_type(number, number_name, (int, float, np.number), is_optional=is_optional)
  assert_type(number_type, 'number_type', (type, tuple))
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
  ''' Checks if the ``number`` is between ``lower`` and ``upper``, inclusive.
  
  Args:
    number (:type:`int | float | np.number`): The input to be checked.
    number_name (:type:`str`): The name of the input.
    lower (:type:`int | float | np.number`): The lower bound.
    upper (:type:`int | float | np.number`): The upper bound.
    is_optional (:type:`bool`): If the ``number`` is optional.

  Raises:
    TypeError: If the input is not of the expected type.
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
  ''' Checks if the ``option`` is in the ``options``.
  
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