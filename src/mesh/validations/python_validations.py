from typing import Any, Iterable, Iterator, Callable

def assert_type(var: Any,
                var_name: str,
                expected_types: type | tuple[type, ...],
                is_optional: bool = False) -> None:
  ''' Checks if the ``var`` is one of the expected types.
  
  Args:
    var (:type:`typing.Any`): The input to be checked.
    var_name (:type:`str`): The name of the input.
    expected_types (:type:`type | tuple[type, ...]`): The type or tuple of types to be checked.
    is_optional (:type:`bool`): If the ``var`` is optional.

  Raises:
    TypeError: If the input is not one of the expected types.
  '''

  # Check the input types
  if not isinstance(var_name, str):
    raise TypeError(f'The parameter "var_name" has type {type(var_name)}, but expected <class \'str\'>.')
  if not isinstance(expected_types, type) and not (isinstance(expected_types, tuple) and all(isinstance(item, type) for item in expected_types)):
    raise TypeError(f'The parameter "expected_types" has type {type(expected_types)}, but expected (<class \'type\'>, <class \'tuple\'>).')
  if not isinstance(is_optional, bool):
    raise TypeError(f'The parameter "is_optional" has type {type(is_optional)}, but expected (<class \'bool\'>, <class \'numpy.bool\'>).')
  
  # Check the optional case
  if var is None:
    if not is_optional:
      raise TypeError(f'The input "{var_name}" is None, but it is not optional.')
    return
  # Check the var type
  if not isinstance(var, expected_types):
    raise TypeError(f'The input "{var_name}" has type {type(var)}, but expected {expected_types}.')

def is_greater_in_type(number: int | float | None,
                       number_name: str,
                       number_type: type[int] | type[float] | tuple[type, ...],
                       value: int | float,
                       is_optional: bool = False) -> None:
  ''' Checks if the ``number`` is of a respective type and if it is greater than ``value``.
  
  Args:
    number (:type:`int | float | None`): The input to be checked.
    number_name (:type:`str`): The name of the input.
    number_type (:type:`type[int] | type[float] | tuple[type, ...]`): The type to be checked.
    value (:type:`int | float`): The value to be compared with.
    is_optional (:type:`bool`): If the ``number`` is optional.

  Raises:
    TypeError: If the input is not of the expected type.
    ValueError: If the input is not greater than the ``value``.
  '''

  # Check the input types
  assert_type(number_name, 'number_name', str)
  assert_type(number, number_name, (int, float), is_optional=is_optional)
  assert_type(number_type, 'number_type', (type, tuple))
  assert_type(value, 'value', (int, float))

  # Check the number is the correct type
  assert_type(number, number_name, number_type, is_optional=is_optional)
  # Check if the input is greater than the value
  if number and number <= value:
    raise ValueError(f'The input "{number_name}" has value {number}, but it must be greater than {value}.')

def is_between_inclusive(number: int | float | None,
                         number_name: str,
                         lower_bound: int | float,
                         upper_bound: int | float,
                         is_optional: bool = False) -> None:
  ''' Checks if the ``number`` is between ``lower_bound`` and ``upper_bound``, inclusive.
  
  Args:
    number (:type:`int | float | None`): The input to be checked.
    number_name (:type:`str`): The name of the input.
    lower_bound (:type:`int | float`): The lower bound.
    upper_bound (:type:`int | float`): The upper_bound bound.
    is_optional (:type:`bool`): If the ``number`` is optional.

  Raises:
    TypeError: If the input is not of the expected type.
    ValueError: If the input is not between the bounds.
  '''

  # Check the input types
  assert_type(number_name, 'number_name', str)
  assert_type(number, number_name, (int, float), is_optional=is_optional)
  assert_type(lower_bound, 'lower_bound', (int, float))
  assert_type(upper_bound, 'upper_bound', (int, float))

  # Check if the boundaries are valids
  if lower_bound > upper_bound:
    raise ValueError('The parameter "lower_bound" must be greater or equal to the parameter "upper_bound".')
  # Check if the number is between the bounds
  if number and (number < lower_bound or number > upper_bound):
    raise ValueError(f'The input "{number_name}" must be between {lower_bound} and {upper_bound}, inclusive.')

def is_in_options(option: Any,
                  option_name: str,
                  options: Iterable | Iterator) -> None:
  ''' Checks if the ``option`` is in the ``options``.
  
  Args:
    option (:type:`typing.Any`): The input to be checked.
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
    raise TypeError(f'The parameter "options" has type {type(options)}, but expected an iterable or iterator.')

  # Check if the option is in the options
  try:
    valid_iter = option not in options
  except Exception:
    raise ValueError('The parameter "options" must be a valid iterable/iterator for "option".')
  if valid_iter:
    raise ValueError(f'The input "{option_name}" must be one of the following options: {options}.')

def is_function(f: Callable, f_name: str) -> None:
  ''' Checks if ``f`` is function.
  
  Args:
    f (:type:`Callable`): The function to be checked.
    f_name (:type:`str`): The name of the function.
    
    Raises:
    TypeError: If the input is not of the expected type.
    ValueError: If the input is not a fitness function.
  '''

  # Check the input types
  assert_type(f_name, 'f_name', str)
  if not callable(f):
    raise TypeError(f'The input "{f_name}" has type {type(f)}, but expected a callable.')