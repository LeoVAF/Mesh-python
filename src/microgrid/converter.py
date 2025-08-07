class Converter():
  ''' Class to simulate the microgrid converter.
  
  Args:
    cost (:type:`int | float`): Cost of the converter in [US$].
    efficiency (:type:`int | float`): Converter efficiency between 0 and 1.
    lifetime (:type:`int | float`): Converter lifetime in [year].
  
  Raises:
    TypeError: If the input is not the expected type.
    ValueError: If the input is not the allowed value.
  '''

  def __init__(self, cost: int | float, efficiency: int | float = 0.95, lifetime = 24):

    self.cost: int | float
    ''' Cost of the converter in [US$]. '''
    self.efficiency: int | float
    ''' Converter efficiency between 0 and 1. '''
    self.lifetime: int | float
    ''' Converter lifetime in [year]. '''
    self.operation_cost: float = 0.0
    ''' Total costs of the converter in the microgrid during the operation simulation in [US$]. '''

    self.cost = cost
    self.efficiency = efficiency
    self.lifetime = lifetime

  def economic_analysis():
    ''' '''

    return 0.0