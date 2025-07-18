from mesh.validations import numpy_validations as npv

import numpy as np
import pytest

def test_assert_np_array_subtype_success():
  assert npv.assert_np_array_subtype(np.array([1, 2, 3]), 'array_value', np.int64) is None
  assert npv.assert_np_array_subtype(np.array([[1.0, 9.0, 5.8]]), 'array_value', np.float64) is None
  assert npv.assert_np_array_subtype(np.array([[1, 9, 5]], dtype=np.uint64), 'array_value', np.uint64) is None
  assert npv.assert_np_array_subtype(np.array([[True, False, False]]), 'array_value', np.bool_) is None
  assert npv.assert_np_array_subtype(np.array(['123', '456', '789']), 'array_value', str) is None
  assert npv.assert_np_array_subtype(np.array([np.array([1, 2, 3]), np.array([-1, -2, -3])]), 'array_value', int) is None
  assert npv.assert_np_array_subtype(np.array([0, 1, np.nan]), 'arr_value', float) is None

def test_assert_np_array_subtype_failure():
  with pytest.raises(TypeError, match=r'The input "array_value" has type <class \'int\'>, but expected <class \'numpy.ndarray\'>.'):
    npv.assert_np_array_subtype(1, 'array_value', int)
  with pytest.raises(TypeError, match=r'The input "arr_name" has type <class \'int\'>, but expected <class \'str\'>.'):
    npv.assert_np_array_subtype(np.array([0, 1]), 1, int)
  with pytest.raises(TypeError, match=r'The input "array_value" has dtype int64, but expected <class \'numpy.float64\'>.'):
    npv.assert_np_array_subtype(np.array([0, 1]), 'array_value', np.float64)
    
def test_assert_no_nan_in_np_array_success():
  assert (npv.assert_no_nan_in_np_array(np.array([1, 2, 3]), 'array_value')) is None
  assert (npv.assert_no_nan_in_np_array(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 'array_value')) is None
  assert (npv.assert_no_nan_in_np_array(np.array([[1.0, 9.0, 5.8]]), 'array_value')) is None

def test_assert_no_nan_in_np_array_failure():
  with pytest.raises(TypeError, match=r'The input "array_value" has type <class \'int\'>, but expected <class \'numpy.ndarray\'>.'):
    npv.assert_no_nan_in_np_array(1, 'array_value')
  with pytest.raises(TypeError, match=r'The input "arr_name" has type <class \'int\'>, but expected <class \'str\'>.'):
    npv.assert_no_nan_in_np_array(np.array([0, 1]), 1)

  with pytest.raises(ValueError, match=r'The input "arr_value" has NaN values.'):
    npv.assert_no_nan_in_np_array(np.array([0, 1, np.nan]), 'arr_value')

def test_assert_np_vector_for_operations_success():
  assert (npv.assert_np_vector_for_operations(np.array([1, 2, 3]), 'array_value', 3)) is None
  assert (npv.assert_np_vector_for_operations(np.array([]), 'array_value', 0)) is None
  assert (npv.assert_np_vector_for_operations(np.array([1, 2, 3, 4, 5]), 'array_value', 5)) is None
  assert (npv.assert_np_vector_for_operations(np.array([1., 2., 3.]), 'array_value', 3)) is None

def test_assert_np_vector_for_operations_failure():
  with pytest.raises(TypeError, match=r'The input "vec_value" has type <class \'int\'>, but expected <class \'numpy.ndarray\'>.'):
    npv.assert_np_vector_for_operations(1, 'vec_value', 1)
  with pytest.raises(TypeError, match=r'The input "vec_name" has type <class \'int\'>, but expected <class \'str\'>.'):
    npv.assert_np_vector_for_operations(np.array([0, 1]), 1, 2)
  with pytest.raises(TypeError, match=r'The input "size" has type <class \'float\'>, but expected \(<class \'int\'>, <class \'numpy.integer\'>\).'):
    npv.assert_np_vector_for_operations(np.array([0, 1]), 'vec_value', 2.)

  with pytest.raises(ValueError, match=r'The input "vec_value" must be one-dimensional.'):
    npv.assert_np_vector_for_operations(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 'vec_value', 9)
  with pytest.raises(ValueError, match=r'The parameter size must be greater than 0.'):
    npv.assert_np_vector_for_operations(np.array([0, 1]), 'vec_value', -2)
  with pytest.raises(ValueError, match=r'The input "vec_value" with size 2 must have size 1.'):
    npv.assert_np_vector_for_operations(np.array([0, 1]), 'vec_value', 1)

def test_assert_np_vectors_for_boundary_success():
  assert (npv.assert_np_vectors_for_boundary(np.array([0, 0, 0]), 'vec1', np.array([1, 1, 1]), 'vec2', 3)) is None
  assert (npv.assert_np_vectors_for_boundary(np.array([0., 0., 0.]), 'vec1', np.array([1., 1., 1.]), 'vec2', 3)) is None
  assert (npv.assert_np_vectors_for_boundary(np.array([0., 0., 0.]), 'vec1', np.array([0, 0, 0]), 'vec2', 3)) is None

def test_assert_np_vectors_for_boundary_failure():
  with pytest.raises(TypeError, match=r'The input "vec1" has type <class \'int\'>, but expected <class \'numpy.ndarray\'>.'):
    npv.assert_np_vectors_for_boundary(1, 'vec1', np.array([1]), 'vec2', 1)
  with pytest.raises(TypeError, match=r'The input "lower_name" has type <class \'int\'>, but expected <class \'str\'>.'):
    npv.assert_np_vectors_for_boundary(np.array([0, 0]), 1, np.array([1, 1]), 'vec2', 2)
  with pytest.raises(TypeError, match=r'The input "vec2" has type <class \'int\'>, but expected <class \'numpy.ndarray\'>.'):
    npv.assert_np_vectors_for_boundary(np.array([1]), 'vec1', 1, 'vec2', 1)
  with pytest.raises(TypeError, match=r'The input "upper_name" has type <class \'int\'>, but expected <class \'str\'>.'):
    npv.assert_np_vectors_for_boundary(np.array([0, 0]), 'vec1', np.array([1, 1]), 1, 2)
  with pytest.raises(TypeError, match=r'The input "size" has type <class \'float\'>, but expected \(<class \'int\'>, <class \'numpy.integer\'>\).'):
    npv.assert_np_vectors_for_boundary(np.array([0, 0]), 'vec1', np.array([1, 1]), 'vec2', 2.)

  with pytest.raises(ValueError, match=r'The input "vec1" must be one-dimensional.'):
    npv.assert_np_vectors_for_boundary(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 'vec1', np.array([1,2,3,4,5,6,7,8,9]), 'vec2', 9)
  with pytest.raises(ValueError, match=r'The input "vec1" with size 2 must have size 1.'):
    npv.assert_np_vectors_for_boundary(np.array([0, 0]), 'vec1', np.array([1]), 'vec2', 1)
  with pytest.raises(ValueError, match=r'The input "vec2" must be one-dimensional.'):
    npv.assert_np_vectors_for_boundary(np.array([1,2,3,4,5,6,7,8,9]), 'vec1', np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 'vec2', 9)
  with pytest.raises(ValueError, match=r'The input "vec2" with size 2 must have size 1.'):
    npv.assert_np_vectors_for_boundary(np.array([0]), 'vec1', np.array([1, 1]), 'vec2', 1)
  with pytest.raises(ValueError, match=r'The parameter size must be greater than 0.'):
    npv.assert_np_vectors_for_boundary(np.array([0, 0]), 'vec1', np.array([1, 1]), 'vec2', -2)
  with pytest.raises(ValueError, match=r'The input "vec1" must be less than "vec2".'):
    npv.assert_np_vectors_for_boundary(np.array([0, 2, 0]), 'vec1', np.array([1, 1, 1]), 'vec2', 3)

def test_assert_np_vector_index_success():
  assert (npv.assert_np_vector_index(np.array([0, 1, 2]), 'idx_vec', 5)) is None
  assert (npv.assert_np_vector_index(np.array([-5, 0, 4]), 'idx_vec', 5)) is None

def test_assert_np_vector_index_failure():
  with pytest.raises(TypeError, match=r'The input "idx_vec" has type <class \'int\'>, but expected <class \'numpy.ndarray\'>.'):
    npv.assert_np_vector_index(1, 'idx_vec', 1)
  with pytest.raises(TypeError, match=r'The input "idx_vec_name" has type <class \'int\'>, but expected <class \'str\'>.'):
    npv.assert_np_vector_index(np.array([0, 0]), 1, 1)
  with pytest.raises(TypeError, match=r'The input "max_index" has type <class \'float\'>, but expected \(<class \'int\'>, <class \'numpy.integer\'>\).'):
    npv.assert_np_vector_index(np.array([0, 0]), 'idx_vec', 1.)
  with pytest.raises(TypeError, match=r'The input "idx_vec" has dtype float64, but expected \(<class \'int\'>, <class \'numpy.integer\'>\).'):
    npv.assert_np_vector_index(np.array([0., 0.]), 'idx_vec', 1)
  with pytest.raises(TypeError, match=r'The input "idx_vec" has dtype float64, but expected \(<class \'int\'>, <class \'numpy.integer\'>\).'):
    npv.assert_np_vector_index(np.array([0, 0, np.nan]), 'idx_vec', 1)
  
  with pytest.raises(ValueError, match=r'The input "idx_vec" must be one-dimensional.'):
    npv.assert_np_vector_index(np.array([[0, 0]]), 'idx_vec', 1)
  with pytest.raises(ValueError, match=r'The input "idx_vec" has indices out of bounds. The maximum index is 1.'):
    npv.assert_np_vector_index(np.array([0, 1]), 'idx_vec', 1)
  with pytest.raises(ValueError, match=r'The input "idx_vec" has indices out of bounds. The maximum index is 2.'):
    npv.assert_np_vector_index(np.array([-3, 0, 1]), 'idx_vec', 2)