def is_greater(var: any, var_name: str, type: int | float, value: int | float):
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

    if not isinstance(var, type):
        raise TypeError(f'The input "{var_name}" has type "{type(var)}", but expected {type}.')
    if var <= value:
        raise ValueError(f'The input "{var_name}" must be greater than {value}.')