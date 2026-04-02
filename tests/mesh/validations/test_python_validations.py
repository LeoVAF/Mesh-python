from mesh.validations import python_validations as pv

from typing import Callable, cast

import pytest

def test_assert_type_success():
  assert (pv.assert_type(1, 'int_value', int)) is None
  assert (pv.assert_type(1.0, 'float_value', float)) is None
  assert (pv.assert_type(1, 'int_value', (int, float))) is None
  assert (pv.assert_type(None, 'None_value', int, is_optional=True)) is None
    
def test_assert_type_failure():
  with pytest.raises(TypeError, match=r'The parameter "var_name" has type <class \'int\'>, but expected <class \'str\'>.'):
    pv.assert_type(1, cast(str, 2), int)
  with pytest.raises(TypeError, match=r'The parameter "expected_types" has type <class \'NoneType\'>, but expected <class \'type\'> or tuple of <class \'type\'>.'):
    pv.assert_type(1, 'var', cast(type, None))
  with pytest.raises(TypeError, match=r'The parameter "is_optional" has type <class \'int\'>, but expected <class \'bool\'>.'):
    pv.assert_type(1, 'var', int, is_optional=cast(bool, 1))
  with pytest.raises(TypeError, match=r'The parameter "expected_types" has type <class \'tuple\'>, but expected <class \'type\'> or tuple of <class \'type\'>.'):
    pv.assert_type(1, 'var', cast(tuple, (int, 2)))
  with pytest.raises(TypeError, match=r'The input "var" has type <class \'int\'>, but expected \(<class \'float\'>, <class \'list\'>\).'):
    pv.assert_type(1, 'var', (float, list))

def test_is_greater_in_type_success():
  assert (pv.is_greater_in_type(1, 'int_value', int, 0)) is None
  assert (pv.is_greater_in_type(1.0, 'float_value', float, 0)) is None
  assert (pv.is_greater_in_type(1, 'int_value', (int, float), 0)) is None
  assert (pv.is_greater_in_type(1, 'int_value', int, 0.0)) is None
  assert (pv.is_greater_in_type(1.0, 'float_value', float, 0.0)) is None
  assert (pv.is_greater_in_type(1, 'int_value', (int, float), 0.0)) is None
  assert (pv.is_greater_in_type(1, 'int_value', int, 0, is_optional=True)) is None
  assert (pv.is_greater_in_type(None, 'None_value', int, 0, is_optional=True)) is None
  
def test_is_greater_in_type_failure():
  with pytest.raises(TypeError, match=r'The input "number" has type <class \'str\'>, but expected \(<class \'int\'>, <class \'float\'>\).'):
    pv.is_greater_in_type(cast(int, 'number'), 'number', int, 0)
  with pytest.raises(TypeError, match=r'The input "number_name" has type <class \'int\'>, but expected <class \'str\'>.'):
    pv.is_greater_in_type(1, cast(str, 2), int, 0)
  with pytest.raises(TypeError, match=r'The input "number_type" is None, but it must not be None.'):
    pv.is_greater_in_type(1, 'var', cast(type, None), 0, is_optional=True)
  with pytest.raises(TypeError, match=r'The input "value" has type <class \'str\'>, but expected \(<class \'int\'>, <class \'float\'>\).'):
    pv.is_greater_in_type(1, 'var', int, cast(float, '0'))
  with pytest.raises(TypeError, match=r'The parameter "is_optional" has type <class \'int\'>, but expected <class \'bool\'>.'):
    pv.is_greater_in_type(1, 'var', int, 0, is_optional=cast(bool, 1))

  with pytest.raises(ValueError, match=r'The input "int_value" has value 1, but it must be greater than 1.'):
    pv.is_greater_in_type(1, 'int_value', int, 1)
  with pytest.raises(ValueError, match=r'The input "int_value" has value -1, but it must be greater than 1.'):
    pv.is_greater_in_type(-1, 'int_value', int, 1)

def test_is_between_inclusive_success():
  assert (pv.is_between_inclusive(1, 'int_value', 0, 2)) is None
  assert (pv.is_between_inclusive(1.0, 'float_value', 0, 2)) is None
  assert (pv.is_between_inclusive(1, 'int_value', 0, 2)) is None
  assert (pv.is_between_inclusive(1, 'int_value', 0, 1)) is None
  assert (pv.is_between_inclusive(1.0, 'float_value', 0, 1)) is None
  assert (pv.is_between_inclusive(1, 'int_value', 0, 1)) is None
  assert (pv.is_between_inclusive(1, 'int_value', 0, 1)) is None
  assert (pv.is_between_inclusive(1.0, 'float_value', 0, 1)) is None
  assert (pv.is_between_inclusive(1, 'int_value', 0, 1)) is None
  assert (pv.is_between_inclusive(1, 'int_value', 0, 2, is_optional=True)) is None
  assert (pv.is_between_inclusive(None, 'None_value', 0, 2, is_optional=True)) is None

def test_is_between_inclusive_failure():
  with pytest.raises(TypeError, match=r'The input "number" has type <class \'str\'>, but expected \(<class \'int\'>, <class \'float\'>\).'):
    pv.is_between_inclusive(cast(int, 'number'), 'number', 0, 2)
  with pytest.raises(TypeError, match=r'The input "number_name" has type <class \'int\'>, but expected <class \'str\'>.'):
    pv.is_between_inclusive(1, cast(str, 2), 0, 2)
  with pytest.raises(TypeError, match=r'The input "lower_bound" has type <class \'str\'>, but expected \(<class \'int\'>, <class \'float\'>\).'):
    pv.is_between_inclusive(1, 'var', cast(float, '0'), 2)
  with pytest.raises(TypeError, match=r'The input "upper_bound" has type <class \'str\'>, but expected \(<class \'int\'>, <class \'float\'>\).'):
    pv.is_between_inclusive(1, 'var', 0, cast(float, '2'))
  with pytest.raises(TypeError, match=r'The parameter "is_optional" has type <class \'int\'>, but expected <class \'bool\'>.'):
    pv.is_between_inclusive(1, 'var', 0, 2, is_optional=cast(bool, 1))
  
  with pytest.raises(ValueError, match=r'The parameter "lower_bound" must be greater or equal to the parameter "upper_bound".'):
    pv.is_between_inclusive(1, 'int_value', 2, 0)
  with pytest.raises(ValueError, match=r'The input "int_value" must be between 0 and 1, inclusive.'):
    pv.is_between_inclusive(-1, 'int_value', 0, 1)

def test_is_in_options_success():
  assert (pv.is_in_options(1, 'int_value', {0, 1, 2})) is None
  assert (pv.is_in_options(1, 'int_value', [0, 1, 2])) is None
  assert (pv.is_in_options(1, 'int_value', (0, 1, 2))) is None
  assert (pv.is_in_options(1, 'int_value', {0: '0', 1: '1', 2: '2'})) is None
  assert (pv.is_in_options('str', 'str_value', 'a_string')) is None
  
def test_is_in_options_failure():
  with pytest.raises(TypeError, match=r'The input "option_name" has type <class \'int\'>, but expected <class \'str\'>.'):
    pv.is_in_options(1, cast(str, 1), [1])
  with pytest.raises(TypeError, match=r'The parameter "options" has type <class \'int\'>, but expected an iterable or iterator.'):
    pv.is_in_options(1, 'int_value', cast(list, 1))
  
  with pytest.raises(ValueError, match=r'The parameter "options" must be a valid iterable/iterator for "option".'):
    pv.is_in_options(1, 'int_value', 'iterator')
  with pytest.raises(ValueError, match=r'The input "int_value" must be one of the following options: \[0, 2\].'):
    pv.is_in_options(1, 'int_value', [0, 2])

def test_is_fitness_function_success():
  assert (pv.is_function(lambda x: x + 1, 'func')) is None
  assert (pv.is_function(lambda x: x[:5] + 1, 'func')) is None

def test_is_fitness_function_failure():
  with pytest.raises(TypeError, match=r'The input "func" has type <class \'int\'>, but expected a callable.'):
    pv.is_function(cast(Callable, 1), 'func')
  with pytest.raises(TypeError, match=r'The input "f_name" has type <class \'int\'>, but expected <class \'str\'>.'):
    pv.is_function(lambda x: x + 1, cast(str, 1))