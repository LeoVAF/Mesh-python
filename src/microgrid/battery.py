import numpy as np

class Battery:
  '''
  Battery object for microgrid simulation.

  Args:
    capacity (:type:`int | float`): Battery capacity in kWh.
    cost_per_kwh (:type:`int | float`): Cost per kWh of the battery.
    efficiency (:type:`int | float`): Battery efficiency between 0 and 1.
    lifetime (:type:`int | float`): Battery lifetime in years.
    number_of_cycles (:type:`int`): Number of cycles the battery can perform. 
    depth_of_discharge (:type:`int | float`): Depth of discharge between 0 and 1.

  Raises:
    TypeError: If the input is not the expected type.
    ValueError: If the input is not the allowed value.
  '''

  def __init__(self,
               capacity: int | float,
               cost_per_kwh: int | float,
               efficiency: int | float,
               lifetime: int | float,
               number_of_cycles: int,
               depth_of_discharge: int | float = 0.8):
    
    self.state_of_charge: np.array[np.float64] | None = None
    ''' Current state of charge in kWh. '''
    self.capacity: int | float
    ''' Battery capacity in kWh. '''
    self.cost_per_kwh: int | float
    ''' Cost per kWh of the battery. '''
    self.efficiency: int | float
    ''' Battery efficiency as a fraction [0, 1]. '''
    self.lifetime: int | float
    ''' Battery lifetime in years. '''
    self.number_of_cycles: int
    ''' Number of cycles the battery can perform. '''
    self.depth_of_discharge: int | float
    ''' Depth of discharge as a fraction [0, 1]. '''

    self.capacity = capacity
    self.cost_per_kwh = cost_per_kwh
    self.efficiency = efficiency
    self.lifetime = lifetime
    self.number_of_cycles = number_of_cycles
    self.depth_of_discharge = depth_of_discharge
