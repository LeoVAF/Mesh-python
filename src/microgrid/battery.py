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
    self.meet_demand: np.ndarray[np.float64] | None = None
    ''' Energy that will effectively meet demand in [kWh]. '''

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
    self.energy_charged = np.zeros(hour_steps)
    self.energy_discharged = np.zeros(hour_steps)
    self.meet_demand = np.zeros(hour_steps)
    # Start the state of charge with maximum capacity
    self.state_of_charge[0] = self.capacity

  def charge(self, surplus_energy: int | float, t: int) -> int | float:
    ''' Charges the battery with surplus power.
    
    Args:
      surplus_energy (:type:`int | float`): Surplus energy to charge the battery in [kWh].
      t (:type:`int`): Time step.

    Returns:
      :type:`int | float`: Amount of remaining surplus energy after charging the battery in [kWh].
    '''

    # Adjust the state of charge array index to avoid out of bounds error
    t_soc = t + 1
    # Get the state of charge
    state_of_charge = self.state_of_charge[t]
    # Charge the battery
    self.state_of_charge[t_soc] = min(state_of_charge + surplus_energy, self.capacity)
    self.energy_charged[t] = self.state_of_charge[t_soc] - state_of_charge
    # Return the remaining surplus energy after charging
    return surplus_energy - self.energy_charged[t]

  def discharge(self, energy_demanded_adjusted: int | float, inverter_efficiency: int | float, t: int) -> int | float:
    ''' Discharges the battery to meet demand.
    
    Args:
      energy_demanded_adjusted (:type:`int | float`): Energy demanded adjusted by the microgrid inverter to discharge the battery in [kWh].
      inverter_efficiency (:type:`int | float`): The efficiency of the inverter between 0 and 1.
      t (:type:`int`): Time step.

    Returns:
      :type:`int | float`: Amount of remaining demand adjusted after discharging the battery in [kWh].
    '''
    
    # Adjust the state of charge array index to avoid out of bounds error
    t_soc = t + 1
    # Get the state of charge
    state_of_charge = self.state_of_charge[t]
    # Discharge the battery
    self.state_of_charge[t_soc] = max(state_of_charge - energy_demanded_adjusted, self.min_soc)
    energy_to_discharge = state_of_charge - self.state_of_charge[t_soc]
    self.energy_discharged[t] = energy_to_discharge
    self.meet_demand[t] = energy_to_discharge * inverter_efficiency
    # Return the remaining demand adjusted after discharging
    return energy_demanded_adjusted - self.energy_discharged[t]