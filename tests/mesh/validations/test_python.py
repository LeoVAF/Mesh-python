from validations import python

import pytest


def test_assert_type():
  # Success cases
  assert python.assert_type(1, 'int_value', int) == None
  assert python.assert_type(1.0, 'float_value', float) == None
  assert python.assert_type(1, 'int_value', (int, float)) == None
  assert python.assert_type(None, 'None_value', int, is_optional=True) == None

  # Failure cases
  with pytest.raises(TypeError, match='The parameter "var_name" of this function has type "<class \'int\'>", but expected str.'):
    python.assert_type(1, 2, int)
  with pytest.raises(TypeError, match='The parameter "expected_types" of this function has type "<class \'NoneType\'>", but expected type or tuple of types.'):
    python.assert_type(1, 'var', None)
  with pytest.raises(TypeError, match='The parameter "is_optional" of this function has type "<class \'int\'>", but expected bool.'):
    python.assert_type(1, 'var', int, is_optional=1)
  with pytest.raises(TypeError, match='The input "var" has type "<class \'int\'>", but expected \(<class \'float\'>, <class \'list\'>\).'):
    python.assert_type(1, 'var', (float, list))