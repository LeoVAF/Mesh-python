import numpy as np
import numpy.typing as npt

class Battery:
  ''' Battery object for microgrid simulation.

  Args:
    capacity (:type:`int | float`): Nominal battery capacity in [kWh].
    cost_per_kwh (:type:`int | float`): Cost per kWh of the battery in [$].
    efficiency (:type:`int | float`): Battery round-trip efficiency between 0 and 1.
    lifetime (:type:`int | float`): Battery lifetime in time intervals.
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
    self.lifetime: int | float
    ''' Battery lifetime in time intervals. '''
    self.number_of_cycles: int
    ''' Number of cycles the battery can perform. '''
    self.depth_of_discharge: int | float
    ''' Depth of discharge as a fraction between 0 and 1. '''
    self.charge_efficiency: int | float
    '''Battery charging efficiency as a fraction between 0 and 1. '''
    self.discharge_efficiency: int | float
    ''' Battery discharging efficiency as a fraction between 0 and 1. '''
    self.hours_per_interval: int
    ''' Number of hours in each time interval in the simulation. '''
    self.cycles: float
    ''' Number of cycles the battery has performed. '''
    self.energy_level: npt.NDArray[np.floating]
    ''' Current energy level in [kWh]. '''
    self.min_energy_level: int | float
    ''' Minimum battery energy level in [kWh]. '''
    self.energy_per_cycle: float
    ''' Energy required to complete a charge/discharge cycle in [kWh]. '''
    self.energy_charged: npt.NDArray[np.floating]
    ''' Numpy array to store the energy charged at each time step in [kWh]. '''
    self.energy_discharged: npt.NDArray[np.floating]
    ''' Numpy array to store the energy discharged at each time step in [kWh]. '''
    self.meet_demand: npt.NDArray[np.floating]
    ''' Energy that will effectively meet demand in [kWh]. '''
    self.replacements: npt.NDArray[np.floating]
    ''' Numpy array to store the number of replacements at each time interval. '''
    self.last_replacement_hour: int
    ''' Last hour when the battery was replaced. '''

    self.capacity = capacity
    self.cost_per_kwh = cost_per_kwh
    self.lifetime = lifetime
    self.number_of_cycles = number_of_cycles
    self.energy_per_cycle = capacity * depth_of_discharge
    self.depth_of_discharge = depth_of_discharge
    self.charge_efficiency = np.sqrt(efficiency)
    self.discharge_efficiency = np.sqrt(efficiency)
    self.min_energy_level = capacity * (1 - depth_of_discharge)

  def initialize(self, hours: int, hours_per_interval: int) -> None:
    ''' Initializes the components of the battery.

    Args:
      hours (:type:`int`): The number of hours in the simulation.
      hours_per_interval (:type:`int`): The number of hours in each time interval.
    '''

    self.energy_level = np.zeros(hours + 1)
    self.energy_charged = np.zeros(hours)
    self.energy_discharged = np.zeros(hours)
    self.meet_demand = np.zeros(hours)
    self.replacements = np.zeros(hours // hours_per_interval)
    self.hours_per_interval = hours_per_interval

    self.cycles = 0.0
    self.last_replacement_hour = 0
    # Start the energy level with minimum energy level
    self.energy_level[0] = self.min_energy_level

  def charge(self, surplus_energy: int | float, converter_efficiency: int | float, t: int) -> int | float:
    ''' Charges the battery using surplus energy.
    
    Args:
      surplus_energy (:type:`int | float`): Surplus energy to charge the battery in [kWh].
      converter_efficiency (:type:`int | float`): The efficiency of the converter between 0 and 1.
      t (:type:`int`): Time step.

    Returns:
      :type:`int | float`: Amount of remaining surplus energy after charging the battery in [kWh].
    '''

    # Adjust the energy level array index to avoid out of bounds error
    idx = t + 1
    # Get the energy level
    energy_level = self.energy_level[t]
    # Calculate the battery effective efficiency considering the converter efficiency
    effective_efficiency = self.charge_efficiency * converter_efficiency
    # Charge the battery
    energy_to_charge = min(surplus_energy * effective_efficiency, self.capacity - energy_level)
    self.energy_level[idx] = energy_level + energy_to_charge
    self.energy_charged[t] = energy_to_charge
    # Update the battery cycles based on the energy charged
    self.cycles += energy_to_charge / (2 * self.energy_per_cycle)
    # Return the remaining surplus energy after charging
    return surplus_energy - energy_to_charge / effective_efficiency

  def discharge(self,
                deficit_energy: float,
                converter_efficiency: int | float,
                inverter_efficiency: int | float,
                t: int) -> float:
    ''' Discharges the battery to meet demand considering the battery efficiency in this operation.
    
    Args:
      deficit_energy (:type:`float`): Positive energy deficit referred to the DC bus, in [kWh].
      converter_efficiency (:type:`int | float`): The efficiency of the converter between 0 and 1.
      inverter_efficiency (:type:`int | float`): The efficiency of the inverter between 0 and 1.
      t (:type:`int`): Time step.

    Returns:
      :type:`float`: Remaining energy deficit referred to the DC bus, in [kWh].
    '''
    
    # Adjust the energy level array index to avoid out of bounds error
    idx = t + 1
    # Get the energy level
    energy_level = self.energy_level[t]
    # Calculate the battery effective efficiency considering the converter efficiency
    effective_efficiency = self.discharge_efficiency * converter_efficiency
    # Discharge the battery
    energy_to_discharge = min(deficit_energy / effective_efficiency, energy_level - self.min_energy_level)
    self.energy_level[idx] = energy_level - energy_to_discharge
    self.energy_discharged[t] = energy_to_discharge
    # The energy that effectively meets the demand
    self.meet_demand[t] = energy_to_discharge * effective_efficiency * inverter_efficiency
    # Update the battery cycles based on the energy discharged
    self.cycles += energy_to_discharge / (2 * self.energy_per_cycle)
    # Return the remaining demand adjusted after discharging
    return deficit_energy - energy_to_discharge * effective_efficiency

  def check_replacement(self, t: int) -> None:
    ''' Checks if the battery needs to be replaced based on its lifetime and number of cycles.

    Args:
      t (:type:`int`): Time step.
    '''

    # Check if the battery has reached its lifetime or number of cycles
    if self.cycles >= self.number_of_cycles or (t - self.last_replacement_hour) / self.hours_per_interval >= self.lifetime:
        # Replace the battery
        self.energy_level[t] = self.min_energy_level
        self.cycles = 0.0
        self.replacements[max(0, (t-1) // self.hours_per_interval)] += 1
        self.last_replacement_hour = t

  def economic_analysis(self,
                        project_lifetime_intervals: npt.NDArray[np.integer],
                        maintenance_cost_rate: int | float,
                        discount_rate: int | float,
                        resale_rate: int | float,
                        CRF: int | float) -> float:
    r''' Performs the economic analysis of the battery using the Net Present Cost (NPC) approach.

    The total NPC of the battery is given by:

    .. math::
      NPC = \text{IC} + \text{NPV}_{om} + \text{NPV}_{repl} - \text{NPV}_{sv}.

    Where:
    
    - :math:`\text{IC}` is the installation cost;
    - :math:`\text{NPV}_{om}` is the Net Present Value of annual operation and maintenance costs;
    - :math:`\text{NPV}_{repl}` is the Net Present Value of replacement costs during the project lifetime;
    - :math:`\text{NPV}_{sv}` is the Net Present Value of the resale value (salvage value) of the battery at the end of its useful life.

    The installation cost is calculated as:

    .. math::
      \text{IC} = C_{kwh} \cdot B_{cap}.

    :math:`C_{kwh}` is the cost per kWh of nominal capacity for the battery and :math:`B_{cap}` is the nominal capacity of the battery. The operation and maintenance costs are calculated as:

    .. math::
      \text{NPV}_{om} = \sum^{T}_{t=1}\frac{\text{IC} \cdot \tau_{om}}{(1 + d)^t}.

    :math:`T` is the project lifetime in time intervals, :math:`d` is the discount rate per interval (assumed to be constant) in [decimal] and :math:`\tau_{om}` is the operation and maintenance cost rate in [decimal]. The replacement costs occur every :attr:`lifetime` intervals and are equal to the installation cost, discounted to present value according to the following equation:
    
    .. math::
      \text{NPV}_{repl} = \sum^{T}_{t=1}\frac{\left(\left\lfloor \frac{t}{T_{\text{repl}}} \right\rfloor - \left\lfloor \frac{t-1}{T_{\text{repl}}} \right\rfloor\right) \cdot \text{IC}}{(1 + d)^t},

    where :math:`T_{repl}` is the time when the equipment must be replaced. The salvage value is calculated as:

    .. math::
      \text{NPV}_{sv} = \frac{\text{IC} \cdot \tau_{sv} \cdot T_{\text{remaining}}}{(1 + d)^T},

    where :math:`\tau_{sv}` is the resale rate of the battery in [decimal] and :math:`T_{\text{remaining}}` is the remaining lifetime of the battery in [decimal].

    Args:
      project_lifetime_intervals (:type:`npt.NDArray[np.integer]`): Intervals of project lifetime.
      maintenance_cost_rate (:type:`int | float`): Operation and maintenance cost rate based on installation costs in [decimal].
      discount_rate (:type:`int | float`): Discount rate (per interval) during the project lifetime in [decimal].
      resale_rate (:type:`int | float`): Resale rate during the project lifetime in [decimal].
      CRF (:type:`int | float`): Capital Recovery Factor (CRF) during the project lifetime in [decimal].

    Returns:
      :type:`float`: Total Net Present Cost of the battery in present value in [$].
    '''

    # Installation cost (CAPEX)
    installation_cost = self.cost_per_kwh * self.capacity
    NPC = installation_cost
    # O&M costs (discounted)
    NPC += (installation_cost * maintenance_cost_rate) / CRF
    # Replacement costs (discounted)
    NPC += np.sum(installation_cost * (self.replacements) / ((1 + discount_rate) ** project_lifetime_intervals[1:]))
    # Resale | salvage value (discounted)
    project_lifetime = project_lifetime_intervals[-1]
    remaining_cycles = 1 - min(self.cycles, self.number_of_cycles) / self.number_of_cycles
    remaining_time = 1 - (project_lifetime * self.hours_per_interval - self.last_replacement_hour) / (self.hours_per_interval * self.lifetime)
    remaining_lifetime = min(remaining_cycles, remaining_time)
    NPC -= (installation_cost * resale_rate * remaining_lifetime) / ((1 + discount_rate) ** project_lifetime_intervals[-1])
    return float(NPC)