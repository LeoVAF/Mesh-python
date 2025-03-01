import numpy as np

''' Type alias for Python types. '''

def assert_type(var: any, var_name: str, expected_type: type) -> None:
    ''' Check if the input is of the expected type.
    
    Args:
      var: The input to be checked.
      var_name: The name of the input.
      type: The type to be checked.

    Raises:
      TypeError: If the input is not of the expected type.
    '''

    # Check the input type
    if not isinstance(var_name, str):
        raise TypeError(f'The parameter "var_name" of this function has type "{type(var_name)}", but expected str.')
    if not isinstance(expected_type, type):
        raise TypeError(f'The parameter "expected_type" of this function has type "{type(expected_type)}", but expected type.')
    
    # Check the var type
    if not isinstance(var, expected_type):
        raise TypeError(f'The input "{var_name}" has type "{type(var)}", but expected {expected_type}.')

def assert_np_array_subtype(arr: np.ndarray, arr_name: str, subtype: type) -> None:
    ''' Check if the input numpy array has the expected subtype.
    
    Args:
      var: The input numpy array to be checked.
      var_name: The name of the input.
      subtype: The subtype to be checked.

    Raises:
      TypeError: If the input numpy array does not have the expected subtype.
      ValueError: If the input numpy array has NaN values.
    '''

    # Check the input type
    assert_type(arr, arr_name, np.ndarray)
    assert_type(subtype, 'subtype', type)

    # Check the array subtype
    if not np.issubdtype(arr.dtype, subtype):
        raise TypeError(f'The input "{arr_name}" has dtype "{arr.dtype}", but expected {subtype}.')
    
    # Check the array values
    if np.any(np.isnan(arr)):
        raise ValueError(f'The input "{arr_name}" has NaN values.')

def assert_index_np_array(idxs: np.ndarray, idxs_name: str, max_index: int) -> None:
    ''' Check if the input numpy array is an index array.
    
    Args:
      var: The input numpy array to be checked.
      var_name: The name of the input.
      max_value: The maximum value to be compared with.

    Raises:
      TypeError: If the input numpy array does not have the expected subtype.
      ValueError: If the input numpy array has values out of bounds.
    '''

    # Check the input types
    assert_type(idxs, idxs_name, np.ndarray)
    assert_type(max_index, "max_index", int)

    # Check the array subtype
    if not np.issubdtype(idxs.dtype, int) and not np.issubdtype(idxs.dtype, np.integer):
        raise TypeError(f'The input "{idxs_name}" has dtype "{idxs.dtype}", but expected an type of integer.')

    # Check the array values
    if np.any(np.isnan(idxs)):
        raise ValueError(f'The input "{idxs_name}" has NaN values.')
    if np.any(idxs >= max_index) or np.any(idxs < -max_index):
        raise ValueError(f'The input "{idxs_name}" has indices out of bounds. The maximum index is {max_index}.')

    

def is_greater(var: any, var_name: str, type: int | float, value: int | float) -> None:
    ''' Check if the input is greater than a value.
    
    Args:
      var: The input to be checked.
      var_name: The name of the input.
      type: The type to be checked.
      value: The value to be compared with.

    Raises:
      TypeError: If the input is not of the expected type.
      ValueError: If the input is not greater than the value.
    '''

    # Check the input types
    assert_type(var, var_name, int | float)
    
    if var <= value:
        raise ValueError(f'The input "{var_name}" must be greater than {value}.')