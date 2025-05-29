

class Inverter():
  ''' Class to simulate the microgrid inverter.
  
  Args:
    efficiency (:type:`int | float`): Inverter efficiency between 0 and 1.
    lifetime (:type:`int | float`): Inverter lifetime in years.
  
  Raises:
    TypeError: If the input is not the expected type.
    ValueError: If the input is not the allowed value.
  '''

  def __init__(self, efficiency: int | float = 0.95, lifetime = 24):
    self.efficiency: int | float
    ''' Inverter efficiency between 0 and 1. '''
    self.lifetime: int | float
    ''' Inverter lifetime in years. '''

    self.efficiency = efficiency
    self.lifetime = lifetime