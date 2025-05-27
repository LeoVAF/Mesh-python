class PublicGrid:
  '''
  Represents a public grid in the microgrid system. This class is used to manage the public grid's properties and behaviors.
  
  Args:
    metering_compensation (:type:`int` | :type:`float`): Compensation percentage when sending energy to the public grid between 0 and 1.

  Raises:
    TypeError: If the input is not the expected type.
    ValueError: If the input is not the allowed value.
  '''

  def __init__(self,
               metering_compensation: int | float = 0):
    
    self.metering_compensation: int | float
    ''' Compensation percentage when sending energy to the public grid between 0 and 1. '''

    self.metering_compensation = metering_compensation
