from mesh.validations import python_validations as pv

import pytest

def test_assert_type():
  # Successful cases
  assert pv.assert_type(1, 'int_value', int) == None
  assert pv.assert_type(1.0, 'float_value', float) == None
  assert pv.assert_type(1, 'int_value', (int, float)) == None
  assert pv.assert_type(None, 'None_value', int, is_optional=True) == None

  # Failure cases
  with pytest.raises(TypeError, match='The parameter "var_name" has type <class \'int\'>, but expected <class \'str\'>.'):
    pv.assert_type(1, 2, int)
  with pytest.raises(TypeError, match='The parameter "expected_types" has type <class \'NoneType\'>, but expected \(<class \'type\'>, <class \'tuple\'>\).'):
    pv.assert_type(1, 'var', None)
  with pytest.raises(TypeError, match='The parameter "is_optional" has type <class \'int\'>, but expected \(<class \'bool\'>, <class \'numpy.bool\'>\).'):
    pv.assert_type(1, 'var', int, is_optional=1)
  with pytest.raises(TypeError, match='The input "var" has type <class \'int\'>, but expected \(<class \'float\'>, <class \'list\'>\).'):
    pv.assert_type(1, 'var', (float, list))

def test_assert_type_or_falsy():
  # Successful cases
  assert pv.assert_type_or_falsy(1, 'int_value', int) == None
  assert pv.assert_type_or_falsy(1.0, 'float_value', float) == None
  assert pv.assert_type_or_falsy(1, 'int_value', (int, float)) == None
  assert pv.assert_type_or_falsy(None, 'None_value', int, is_optional=True) == None
  assert pv.assert_type_or_falsy(None, 'None_value', int, is_optional=False) == None

  # Failure cases
  with pytest.raises(TypeError, match='The parameter "var_name" has type <class \'int\'>, but expected <class \'str\'>.'):
    pv.assert_type_or_falsy(1, 2, int)
  with pytest.raises(TypeError, match='The parameter "expected_types" has type <class \'NoneType\'>, but expected \(<class \'type\'>, <class \'tuple\'>\).'):
    pv.assert_type_or_falsy(1, 'var', None)
  with pytest.raises(TypeError, match='The parameter "is_optional" has type <class \'int\'>, but expected \(<class \'bool\'>, <class \'numpy.bool\'>\).'):
    pv.assert_type_or_falsy(1, 'var', int, is_optional=1)
  with pytest.raises(TypeError, match='The input "var" has type <class \'int\'>, but expected \(<class \'float\'>, <class \'list\'>\).'):
    pv.assert_type_or_falsy(1, 'var', (float, list))

def test_is_greater_in_type():
  # Successful cases
  assert pv.is_greater_in_type(1, 'int_value', int, 0) == None
  assert pv.is_greater_in_type(1.0, 'float_value', float, 0) == None
  assert pv.is_greater_in_type(1, 'int_value', (int, float), 0) == None
  assert pv.is_greater_in_type(1, 'int_value', int, 0.0) == None
  assert pv.is_greater_in_type(1.0, 'float_value', float, 0.0) == None
  assert pv.is_greater_in_type(1, 'int_value', (int, float), 0.0) == None
  assert pv.is_greater_in_type(1, 'int_value', int, 0.0) == None
  assert pv.is_greater_in_type(1.0, 'float_value', float, 0.0) == None
  assert pv.is_greater_in_type(1, 'int_value', (int, float), 0.0) == None
  assert pv.is_greater_in_type(1, 'int_value', int, 0, is_optional=True) == None
  assert pv.is_greater_in_type(None, 'None_value', int, 0, is_optional=True) == None

  # Failure cases
  with pytest.raises(TypeError, match='The input "number" has type <class \'str\'>, but expected \(<class \'int\'>, <class \'float\'>, <class \'numpy.number\'>\).'):
    pv.is_greater_in_type('number', 'number', int, 0)
  with pytest.raises(TypeError, match='The input "number_name" has type <class \'int\'>, but expected <class \'str\'>.'):
    pv.is_greater_in_type(1, 2, int, 0)
  with pytest.raises(TypeError, match='The input "number_type" has type <class \'NoneType\'>, but expected \(<class \'type\'>, <class \'tuple\'>\).'):
    pv.is_greater_in_type(1, 'var', None, 0)
  with pytest.raises(TypeError, match='The input "value" has type <class \'str\'>, but expected \(<class \'int\'>, <class \'float\'>, <class \'numpy.number\'>\).'):
    pv.is_greater_in_type(1, 'var', int, '0')
  with pytest.raises(TypeError, match='The parameter "is_optional" has type <class \'int\'>, but expected \(<class \'bool\'>, <class \'numpy.bool\'>\).'):
    pv.is_greater_in_type(1, 'var', int, 0, is_optional=1)
  
  with pytest.raises(ValueError, match='The input "int_value" has value 1, but it must be greater than 1.'):
    pv.is_greater_in_type(1, 'int_value', int, 1)
  with pytest.raises(ValueError, match='The input "int_value" has value -1, but it must be greater than 1.'):
    pv.is_greater_in_type(-1, 'int_value', int, 1)

def test_is_between_inclusive():
  # Successful cases
  assert pv.is_between_inclusive(1, 'int_value', 0, 2) == None
  assert pv.is_between_inclusive(1.0, 'float_value', 0, 2) == None
  assert pv.is_between_inclusive(1, 'int_value', 0, 2) == None
  assert pv.is_between_inclusive(1, 'int_value', 0, 1) == None
  assert pv.is_between_inclusive(1.0, 'float_value', 0, 1) == None
  assert pv.is_between_inclusive(1, 'int_value', 0, 1) == None
  assert pv.is_between_inclusive(1, 'int_value', 0, 1) == None
  assert pv.is_between_inclusive(1.0, 'float_value', 0, 1) == None
  assert pv.is_between_inclusive(1, 'int_value', 0, 1) == None
  assert pv.is_between_inclusive(1, 'int_value', 0, 2, is_optional=True) == None
  assert pv.is_between_inclusive(None, 'None_value', 0, 2, is_optional=True) == None

  # Failure cases
  with pytest.raises(TypeError, match='The input "number" has type <class \'str\'>, but expected \(<class \'int\'>, <class \'float\'>, <class \'numpy.number\'>\).'):
    pv.is_between_inclusive('number', 'number', 0, 2)
  with pytest.raises(TypeError, match='The input "number_name" has type <class \'int\'>, but expected <class \'str\'>.'):
    pv.is_between_inclusive(1, 2, 0, 2)
  with pytest.raises(TypeError, match='The input "lower_bound" has type <class \'str\'>, but expected \(<class \'int\'>, <class \'float\'>, <class \'numpy.number\'>\).'):
    pv.is_between_inclusive(1, 'var', '0', 2)
  with pytest.raises(TypeError, match='The input "upper_bound" has type <class \'str\'>, but expected \(<class \'int\'>, <class \'float\'>, <class \'numpy.number\'>\).'):
    pv.is_between_inclusive(1, 'var', 0, '2')
  with pytest.raises(TypeError, match='The parameter "is_optional" has type <class \'int\'>, but expected \(<class \'bool\'>, <class \'numpy.bool\'>\).'):
    pv.is_between_inclusive(1, 'var', 0, 2, is_optional=1)
  
  with pytest.raises(ValueError, match='The parameter "lower_bound" must be greater or equal to the parameter "upper_bound".'):
    pv.is_between_inclusive(1, 'int_value', 2, 0)
  with pytest.raises(ValueError, match='The input "int_value" must be between 0 and 1, inclusive.'):
    pv.is_between_inclusive(-1, 'int_value', 0, 1)

def test_is_in_options():
  # Successful cases
  assert pv.is_in_options(1, 'int_value', {0, 1, 2}) == None
  assert pv.is_in_options(1, 'int_value', [0, 1, 2]) == None
  assert pv.is_in_options(1, 'int_value', (0, 1, 2)) == None
  assert pv.is_in_options(1, 'int_value', {0: '0', 1: '1', 2: '2'}) == None
  assert pv.is_in_options('str', 'str_value', 'a_string') == None

  # Failure cases
  with pytest.raises(TypeError, match='The input "option_name" has type <class \'int\'>, but expected <class \'str\'>.'):
    pv.is_in_options(1, 1, [1])
  
  with pytest.raises(ValueError, match='The parameter "options" must be a valid iterable/iterator for "option".'):
    pv.is_in_options(1, 'int_value', 'iterator')
  with pytest.raises(ValueError, match='The input "int_value" must be one of the following options: \[0, 2\].'):
    pv.is_in_options(1, 'int_value', [0, 2])