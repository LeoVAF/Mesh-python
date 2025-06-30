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
    ''' Battery efficiency as a fraction between 0 and 1. '''
    self.lifetime: int | float
    ''' Battery lifetime in [year]. '''
    self.number_of_cycles: int
    ''' Number of cycles the battery can perform. '''
    self.depth_of_discharge: int | float
    ''' Depth of discharge as a fraction between 0 and 1. '''
    self.state_of_charge: np.array[np.float64] | None = None
    ''' Current state of charge in [kWh]. '''
    self.min_soc: int | float
    ''' Minimum battery state of charge in [kWh]. '''
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
    self.min_soc = capacity * depth_of_discharge

  def initialize(self, hour_steps: int) -> None:
    ''' Initializes the components of the battery.

    Args:
      hour_steps (:type:`int`): The number of hour steps in the simulation.
    '''

    self.state_of_charge = np.zeros(hour_steps + 1)
    self.state_of_charge[0] = self.capacity
    self.energy_charged = np.zeros(hour_steps)
    self.energy_discharged = np.zeros(hour_steps)

  def charge(self, surplus: int | float, t: int) -> int | float:
    ''' Charges the battery with surplus power.
    
    Args:
      surplus (:type:`int | float`): Surplus energy to charge the battery in [kWh].
      t (:type:`int`): Index of the time step.

    Returns:
      :type:`int | float`: Amount of remaining surplus energy after charging the battery in [kWh].
    '''

    # Adjust the state of charge array index to avoid out of bounds error
    t_soc = t + 1
    available_capacity = self.capacity - self.state_of_charge[t]
    if surplus <= available_capacity:
      # Charge normally
      self.state_of_charge[t_soc] = self.state_of_charge[t] + surplus
      self.energy_charged[t] = surplus
      return 0
    else:
      # Only carry what fits
      self.state_of_charge[t_soc] = self.capacity
      self.energy_charged[t] = available_capacity
      # Calculate the remaining surplus after charging
      return surplus - available_capacity

  def discharge(self, demand: int | float, t: int) -> int | float:
    ''' Discharges the battery to meet demand.
    
    Args:
      demand (:type:`int | float`): Energy demand to discharge the battery in [kWh].
      t (:type:`int`): Index of the time step.

    Returns:
      :type:`int | float`: Amount of remaining demand after discharging the battery in [kWh].
    '''
    
    # Adjust the state of charge array index to avoid out of bounds error
    t_soc = t + 1
    adjusted_demand_by_bat_efficiency = demand / self.efficiency
    available_energy = self.state_of_charge[t] - self.min_soc
    if available_energy >= adjusted_demand_by_bat_efficiency:
      # Meets all demand
      self.state_of_charge[t_soc] = self.state_of_charge[t] - adjusted_demand_by_bat_efficiency
      self.energy_discharged[t] = adjusted_demand_by_bat_efficiency
      return 0
    else:
      # Uses everything available up to the minimum SoC
      self.state_of_charge[t_soc] = self.min_soc
      self.energy_discharged[t] = available_energy
      return demand - available_energy * self.efficiency