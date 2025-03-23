from mesh.validations import python_validations as pv

from unittest import TestCase, main

class TestPythonValidations(TestCase):
  def test_assert_type(self):
    # Successful cases
    self.assertIsNone(pv.assert_type(1, 'int_value', int))
    self.assertIsNone(pv.assert_type(1.0, 'float_value', float))
    self.assertIsNone(pv.assert_type(1, 'int_value', (int, float)))
    self.assertIsNone(pv.assert_type(None, 'None_value', int, is_optional=True))
    
    # Failure cases
    with self.assertRaisesRegex(TypeError, r'The parameter "var_name" has type <class \'int\'>, but expected <class \'str\'>.'):
      pv.assert_type(1, 2, int)
    with self.assertRaisesRegex(TypeError, r'The parameter "expected_types" has type <class \'NoneType\'>, but expected \(<class \'type\'>, <class \'tuple\'>\).'):
      pv.assert_type(1, 'var', None)
    with self.assertRaisesRegex(TypeError, r'The parameter "is_optional" has type <class \'int\'>, but expected \(<class \'bool\'>, <class \'numpy.bool\'>\).'):
      pv.assert_type(1, 'var', int, is_optional=1)
    with self.assertRaisesRegex(TypeError, r'The input "var" has type <class \'int\'>, but expected \(<class \'float\'>, <class \'list\'>\).'):
      pv.assert_type(1, 'var', (float, list))

  def test_is_greater_in_type(self):
    # Successful cases
    self.assertIsNone(pv.is_greater_in_type(1, 'int_value', int, 0))
    self.assertIsNone(pv.is_greater_in_type(1.0, 'float_value', float, 0))
    self.assertIsNone(pv.is_greater_in_type(1, 'int_value', (int, float), 0))
    self.assertIsNone(pv.is_greater_in_type(1, 'int_value', int, 0.0))
    self.assertIsNone(pv.is_greater_in_type(1.0, 'float_value', float, 0.0))
    self.assertIsNone(pv.is_greater_in_type(1, 'int_value', (int, float), 0.0))
    self.assertIsNone(pv.is_greater_in_type(1, 'int_value', int, 0, is_optional=True))
    self.assertIsNone(pv.is_greater_in_type(None, 'None_value', int, 0, is_optional=True))
    
    # Failure cases
    with self.assertRaisesRegex(TypeError, r'The input "number" has type <class \'str\'>, but expected \(<class \'int\'>, <class \'float\'>, <class \'numpy.number\'>\).'):
      pv.is_greater_in_type('number', 'number', int, 0)
    with self.assertRaisesRegex(TypeError, r'The input "number_name" has type <class \'int\'>, but expected <class \'str\'>.'):
      pv.is_greater_in_type(1, 2, int, 0)
    with self.assertRaisesRegex(TypeError, r'The input "number_type" has type <class \'NoneType\'>, but expected \(<class \'type\'>, <class \'tuple\'>\).'):
      pv.is_greater_in_type(1, 'var', None, 0)
    with self.assertRaisesRegex(TypeError, r'The input "value" has type <class \'str\'>, but expected \(<class \'int\'>, <class \'float\'>, <class \'numpy.number\'>\).'):
      pv.is_greater_in_type(1, 'var', int, '0')
    with self.assertRaisesRegex(TypeError, r'The parameter "is_optional" has type <class \'int\'>, but expected \(<class \'bool\'>, <class \'numpy.bool\'>\).'):
      pv.is_greater_in_type(1, 'var', int, 0, is_optional=1)
    
    with self.assertRaisesRegex(ValueError, r'The input "int_value" has value 1, but it must be greater than 1.'):
      pv.is_greater_in_type(1, 'int_value', int, 1)
    with self.assertRaisesRegex(ValueError, r'The input "int_value" has value -1, but it must be greater than 1.'):
      pv.is_greater_in_type(-1, 'int_value', int, 1)
  
  def test_is_between_inclusive(self):
    # Successful cases
    self.assertIsNone(pv.is_between_inclusive(1, 'int_value', 0, 2))
    self.assertIsNone(pv.is_between_inclusive(1.0, 'float_value', 0, 2))
    self.assertIsNone(pv.is_between_inclusive(1, 'int_value', 0, 2))
    self.assertIsNone(pv.is_between_inclusive(1, 'int_value', 0, 1))
    self.assertIsNone(pv.is_between_inclusive(1.0, 'float_value', 0, 1))
    self.assertIsNone(pv.is_between_inclusive(1, 'int_value', 0, 1))
    self.assertIsNone(pv.is_between_inclusive(1, 'int_value', 0, 1))
    self.assertIsNone(pv.is_between_inclusive(1.0, 'float_value', 0, 1))
    self.assertIsNone(pv.is_between_inclusive(1, 'int_value', 0, 1))
    self.assertIsNone(pv.is_between_inclusive(1, 'int_value', 0, 2, is_optional=True))
    self.assertIsNone(pv.is_between_inclusive(None, 'None_value', 0, 2, is_optional=True))

    # Failure cases
    with self.assertRaisesRegex(TypeError, r'The input "number" has type <class \'str\'>, but expected \(<class \'int\'>, <class \'float\'>, <class \'numpy.number\'>\).'):
      pv.is_between_inclusive('number', 'number', 0, 2)
    with self.assertRaisesRegex(TypeError, r'The input "number_name" has type <class \'int\'>, but expected <class \'str\'>.'):
      pv.is_between_inclusive(1, 2, 0, 2)
    with self.assertRaisesRegex(TypeError, r'The input "lower_bound" has type <class \'str\'>, but expected \(<class \'int\'>, <class \'float\'>, <class \'numpy.number\'>\).'):
      pv.is_between_inclusive(1, 'var', '0', 2)
    with self.assertRaisesRegex(TypeError, r'The input "upper_bound" has type <class \'str\'>, but expected \(<class \'int\'>, <class \'float\'>, <class \'numpy.number\'>\).'):
      pv.is_between_inclusive(1, 'var', 0, '2')
    with self.assertRaisesRegex(TypeError, r'The parameter "is_optional" has type <class \'int\'>, but expected \(<class \'bool\'>, <class \'numpy.bool\'>\).'):
      pv.is_between_inclusive(1, 'var', 0, 2, is_optional=1)
    
    with self.assertRaisesRegex(ValueError, r'The parameter "lower_bound" must be greater or equal to the parameter "upper_bound".'):
      pv.is_between_inclusive(1, 'int_value', 2, 0)
    with self.assertRaisesRegex(ValueError, r'The input "int_value" must be between 0 and 1, inclusive.'):
      pv.is_between_inclusive(-1, 'int_value', 0, 1)
  
  def test_is_in_options(self):
    # Successful cases
    self.assertIsNone(pv.is_in_options(1, 'int_value', {0, 1, 2}))
    self.assertIsNone(pv.is_in_options(1, 'int_value', [0, 1, 2]))
    self.assertIsNone(pv.is_in_options(1, 'int_value', (0, 1, 2)))
    self.assertIsNone(pv.is_in_options(1, 'int_value', {0: '0', 1: '1', 2: '2'}))
    self.assertIsNone(pv.is_in_options('str', 'str_value', 'a_string'))
    
    # Failure cases
    with self.assertRaisesRegex(TypeError, r'The input "option_name" has type <class \'int\'>, but expected <class \'str\'>.'):
      pv.is_in_options(1, 1, [1])
    with self.assertRaisesRegex(TypeError, r'The parameter "options" has type <class \'int\'>, but expected an iterable or iterator.'):
      pv.is_in_options(1, 'int_value', 1)
    
    with self.assertRaisesRegex(ValueError, r'The parameter "options" must be a valid iterable/iterator for "option".'):
      pv.is_in_options(1, 'int_value', 'iterator')
    with self.assertRaisesRegex(ValueError, r'The input "int_value" must be one of the following options: \[0, 2\].'):
      pv.is_in_options(1, 'int_value', [0, 2])
  
  def test_is_fitness_function(self):
    # Successful cases
    self.assertIsNone(pv.is_function(lambda x: x + 1, 'func'))
    self.assertIsNone(pv.is_function(lambda x: x[:5] + 1, 'func'))

    # Failure cases
    with self.assertRaisesRegex(TypeError, r'The input "func" has type <class \'int\'>, but expected a callable.'):
      pv.is_function(1, 'func')
    with self.assertRaisesRegex(TypeError, r'The input "f_name" has type <class \'int\'>, but expected <class \'str\'>.'):
      pv.is_function(lambda x: x + 1, 1)

if __name__ == '__main__':
    main()