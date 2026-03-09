import numpy as np
import numpy.typing as npt

class Battery:
  ''' Battery object for microgrid simulation.

  Args:
    capacity (:type:`int | float`): Nominal battery capacity in [kWh].
    cost_per_kwh (:type:`int | float`): Cost per kWh of the battery in [$].
    efficiency (:type:`int | float`): Battery efficiency between 0 and 1.
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
    self.efficiency: int | float
    ''' Battery efficiency as a fraction between 0 and 1. '''
    self.lifetime: int | float
    ''' Battery lifetime in time intervals. '''
    self.number_of_cycles: int
    ''' Number of cycles the battery can perform. '''
    self.depth_of_discharge: int | float
    ''' Depth of discharge as a fraction between 0 and 1. '''
    self.state_of_charge: npt.NDArray[np.floating]
    ''' Current state of charge in [kWh]. '''
    self.min_soc: int | float
    ''' Minimum battery state of charge in [kWh]. '''
    self.cycles: float = 0.0
    ''' Number of discharge cycles performed by the battery during the simulation. '''
    self.energy_per_cycle: float
    ''' Energy required to complete a discharge cycle in [kWh]. '''
    self.energy_charged: npt.NDArray[np.floating]
    ''' Numpy array to store the energy charged at each time step in [kWh]. '''
    self.energy_discharged: npt.NDArray[np.floating]
    ''' Numpy array to store the energy discharged at each time step in [kWh]. '''
    self.meet_demand: npt.NDArray[np.floating]
    ''' Energy that will effectively meet demand in [kWh]. '''

    self.capacity = capacity
    self.cost_per_kwh = cost_per_kwh
    self.efficiency = efficiency
    self.lifetime = lifetime
    self.number_of_cycles = number_of_cycles
    self.energy_per_cycle = capacity * depth_of_discharge
    self.depth_of_discharge = depth_of_discharge
    self.min_soc = capacity * (1 - depth_of_discharge)

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

  def charge(self, surplus_energy: int | float, converter_efficiency: int | float, t: int) -> int | float:
    ''' Charges the battery with surplus power.
    
    Args:
      surplus_energy (:type:`int | float`): Surplus energy to charge the battery in [kWh].
      converter_efficiency (:type:`int | float`): The efficiency of the converter between 0 and 1.
      t (:type:`int`): Time step.

    Returns:
      :type:`int | float`: Amount of remaining surplus energy after charging the battery in [kWh].
    '''

    # Adjust the state of charge array index to avoid out of bounds error
    t_soc = t + 1
    # Get the state of charge
    state_of_charge = self.state_of_charge[t]
    # Charge the battery
    surplus_energy_adjusted = surplus_energy * converter_efficiency
    self.state_of_charge[t_soc] = min(state_of_charge + surplus_energy_adjusted, self.capacity)
    energy_to_charge = self.state_of_charge[t_soc] - state_of_charge
    self.energy_charged[t] = energy_to_charge
    # Update the number of cycles
    self.cycles += energy_to_charge / (2 * self.energy_per_cycle)
    # Return the remaining surplus energy after charging
    return surplus_energy_adjusted - energy_to_charge

  def discharge(self, energy_demanded_adjusted: int | float, inverter_efficiency: int | float, t: int) -> int | float:
    ''' Discharges the battery to meet demand considering the battery efficiency in this operation.
    
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
    self.state_of_charge[t_soc] = max(state_of_charge - energy_demanded_adjusted / self.efficiency, self.min_soc)
    energy_to_discharge = state_of_charge - self.state_of_charge[t_soc]
    self.energy_discharged[t] = energy_to_discharge
    # Update the number of cycles
    self.cycles += energy_to_discharge / (2 * self.energy_per_cycle)
    # The energy that effectively meets the demand
    self.meet_demand[t] = energy_to_discharge * self.efficiency * inverter_efficiency
    # Return the remaining demand adjusted after discharging
    return energy_demanded_adjusted - energy_to_discharge * self.efficiency

  def economic_analysis(self,
                        project_lifetime_intervals: npt.NDArray[np.integer],
                        maintenance_cost_rate: int | float,
                        discount_rate: int | float,
                        CRF: int | float) -> float:
    r''' Performs the economic analysis of the battery using the Net Present Cost (NPC) approach.

    The total NPC of the battery is given by:

    .. math::
      NPC_{bat} = \text{IC}_{bat} + \text{NPV}_{om} + \text{NPV}_{repl}.

    Where:
    
    - :math:`\text{IC}_{bat}` is the installation cost;
    - :math:`\text{NPV}_{om}` is the Net Present Value of annual operation and maintenance costs;
    - :math:`\text{NPV}_{repl}` is the Net Present Value of replacement costs during the project lifetime;

    The installation cost is calculated as:

    .. math::
      \text{IC}_{bat} = C_{kwh} \cdot B_{cap}.

    :math:`C_{kwh}` is the cost per kWh of nominal capacity for the battery and :math:`B_{cap}` is the nominal capacity of the battery. The operation and maintenance costs are calculated as:

    .. math::
      \text{NPV}_{om} = \sum^{T}_{t=1}\frac{\text{IC}_{bat} \cdot \tau_{om}}{(1 + d)^t}.

    :math:`T` is the project lifetime in time intervals, :math:`d` is the discount rate per interval (assumed to be constant) in [decimal] and :math:`\tau_{om}` is the operation and maintenance cost rate in [decimal]. The replacement costs occur every :attr:`lifetime` intervals and are equal to the installation cost, discounted to present value according to the following equation:
    
    .. math::
      \text{NPV}_{repl} = \sum^{T}_{t=1}\frac{\left(\left\lfloor \frac{t}{T_{\text{repl}}} \right\rfloor - \left\lfloor \frac{t-1}{T_{\text{repl}}} \right\rfloor\right) \cdot \text{IC}_{bat}}{(1 + d)^t},

    where :math:`T_{repl} = \min\left(I^{\text{lifetime}}_{\text{bat}},\ \dfrac{B^{\text{max}}_{\text{cycles}}}{\sum^{H}_{h=1}B_{\text{cycles}}(h)}\right)` is the time when the equipment must be replaced.

    Args:
      project_lifetime_intervals (:type:`npt.NDArray[np.integer]`): Intervals of project lifetime.
      maintenance_cost_rate (:type:`int | float`): Operation and maintenance cost rate based on installation costs in [decimal].
      discount_rate (:type:`int | float`): Discount rate (per interval) during the project lifetime in [decimal].
      CRF (:type:`int | float`): Capital Recovery Factor (CRF) during the project lifetime in [decimal].

    Returns:
      :type:`float`: Total Net Present Cost of the battery in present value in [$].
    '''

    # Installation cost (CAPEX)
    installation_cost = self.cost_per_kwh * self.capacity
    NPC = installation_cost
    # O&M costs (discounted)
    NPC += (installation_cost * maintenance_cost_rate) / CRF
    # Effective lifetime
    if self.cycles > 0:
      lifetime_cycles = self.number_of_cycles / self.cycles
    else:
      lifetime_cycles = np.inf
    t_eff = min(self.lifetime, lifetime_cycles)
    # Replacement costs (discounted)
    n_repl = np.ceil(project_lifetime_intervals / t_eff)
    NPC += np.sum(installation_cost * (n_repl[1:] - n_repl[:-1]) / ((1 + discount_rate) ** project_lifetime_intervals[1:]))
    return float(NPC)