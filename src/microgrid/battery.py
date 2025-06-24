import numpy as np

class Battery:
  ''' Battery object for microgrid simulation.

  Args:
    capacity (:type:`int | float`): Nominal battery capacity in [kWh].
    cost_per_kwh (:type:`int | float`): Cost per kWh of the battery in [US$].
    efficiency (:type:`int | float`): Battery efficiency between 0 and 1.
    lifetime (:type:`int | float`): Battery lifetime in [year].
    number_of_cycles (:type:`int`): Number of charge/discharge cycles the battery can perform. 
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
    ''' Current state of charge in [kWh]. '''
    self.capacity: int | float
    ''' Nominal battery capacity in [kWh]. '''
    self.cost_per_kwh: int | float
    ''' Cost per kWh of the battery. '''
    self.efficiency: int | float
    ''' Battery efficiency as a fraction [0, 1]. '''
    self.lifetime: int | float
    ''' Battery lifetime in [year]. '''
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

  def initialize_state_of_charge(self, length: int) -> None:
    ''' Initializes the state of charge of the battery.

    Args:
      length (:type:`int`): Length of the state of charge array.
    '''

    self.state_of_charge = np.zeros(length)

  def charge(self, surplus_power: np.ndarray[np.float64], indexes: np.ndarray[np.integer]) -> np.ndarray[np.float64]:
    ''' Charges the battery with surplus power.
    
    Args:
      surplus_power (:type:`np.ndarray[np.float64]`): Surplus power array in [kWh].
      indexes (:type:`np.ndarray[np.integer]`): Indexes of the surplus power array to charge.
    '''

    return 0

  def discharge(self, demand: np.ndarray[np.float64], indexes: np.ndarray[np.integer]) -> np.ndarray[np.float64]:
    ''' Discharges the battery to meet demand.
    
    Args:
      demand (:type:`np.ndarray[np.float64]`): Demand array in [kWh].
      indexes (:type:`np.ndarray[np.integer]`): Indexes of the demand array to discharge.
    '''
    
    return 0