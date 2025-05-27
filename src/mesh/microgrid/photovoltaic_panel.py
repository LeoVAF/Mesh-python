class PhotovoltaicPanel:
  '''
  Class representing a photovoltaic panel in a microgrid system.
  
  Args:
    rated_power (:type:`int` | :type:`float`): The maximum power output of the photovoltaic panel in kilowatts (kW).
  
  Raises:
    TypeError: If the input is not the expected type.
    ValueError: If the input is not the allowed value.
  '''

  def __init__(self,
               rated_power: int | float):
    
    self.rated_power: int | float
    ''' Maximum power output of the photovoltaic panel in kW. '''

    self.rated_power = rated_power
  