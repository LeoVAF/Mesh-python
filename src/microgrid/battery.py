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
    self.state_of_charge: np.array[np.float64] | None = None
    ''' Current state of charge in [kWh]. '''
    self.energy_charged: np.ndarray[np.float64] | None = None
    ''' Numpy array to store the energy charged at each time step in [kWh]. '''
    self.energy_discharged: np.ndarray[np.float64] | None = None
    ''' Numpy array to store the energy discharged at each time step in [kWh]. '''

    self.capacity = capacity
    self.cost_per_kwh = cost_per_kwh
    self.efficiency = efficiency
    self.lifetime = lifetime
    self.number_of_cycles = number_of_cycles
    self.depth_of_discharge = depth_of_discharge

  def initialize(self, hour_steps: int) -> None:
    ''' Initializes the components of the battery.

    Args:
      hour_steps (:type:`int`): The number of hour steps in the simulation.
    '''

    self.state_of_charge = np.zeros(hour_steps)
    self.state_of_charge[0] = self.capacity
    self.energy_charged = np.zeros(hour_steps)
    self.energy_discharged = np.zeros(hour_steps)

  def charge(self, surplus: int | float, t: int) -> int | float:
    ''' Charges the battery with surplus power.
    
    Args:
      surplus (:type:`int | float`):  [kWh].
      t (:type:`int`): Index of the time step.

    Returns:
      :type:`int | float`: Amount of remaining surplus energy in [kWh]
    '''

    return 0

  def discharge(self, demand: int | float, t: int) -> int | float:
    ''' Discharges the battery to meet demand.
    
    Args:
      demand (:type:`np.ndarray[np.float64]`): Demand array in [kWh].
      t (:type:`np.ndarray[np.integer]`): Indexes of the demand array to discharge.
    '''
    
    return 0