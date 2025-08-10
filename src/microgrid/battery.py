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
    self.cycles: float = 0.0
    ''' Number of discharge cycles performed by the battery during the simulation. '''
    self.energy_per_cycle: float
    ''' Energy required to complete a discharge cycle in [kWh]. '''
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
    self.cycles += energy_to_discharge / self.energy_per_cycle
    # The energy that effectively meets the demand
    self.meet_demand[t] = energy_to_discharge * self.efficiency * inverter_efficiency
    # Return the remaining demand adjusted after discharging
    return energy_demanded_adjusted - self.energy_discharged[t]

  def economic_analysis(self, project_lifetime: int | float, maintenance_cost_rate: int | float, discount_rate: int | float) -> float:
    r''' Performs the economic analysis of the battery using the Net Present Cost (NPC) approach.

    The total NPC of the battery is given by:

    .. math::
        NPC_{bat} = \text{IC}_{bat} + \text{NPV}_{om} + \text{NPV}_{repl} + C_{deg}.

    Where:
    
    - :math:`\text{IC}_{bat}` is the installation cost;
    - :math:`\text{NPV}_{om}` is the Net Present Value of annual operation and maintenance costs;
    - :math:`\text{NPV}_{repl}` is the Net Present Value of replacement costs during the project lifetime;
    - :math:`C_{deg}` is the degradation cost of the battery.

    The installation cost is calculated as:

    .. math::
      \text{IC}_{bat} = C_{kwh} \cdot E^{nominal}_{bat}.

    :math:`C_{kwh}` is the cost per kWh of nominal capacity for the battery and :math:`E^{nominal}_{bat}` is the nominal capacity of the battery. The operation and maintenance costs are calculated as:

    .. math::
      \text{NPV}_{om} = \sum^{T-1}_{t=0}\frac{\text{IC}_{bat} \cdot \tau_{om}}{(1 + d)^t}.

    :math:`T` is the project lifetime in [years], :math:`d` is the discount rate per year (assumed to be constant) in [decimal] and :math:`\tau_{om}` is the operation and maintenance cost rate in [decimal]. The replacement costs occur every :attr:`lifetime` years and are equal to the installation cost, discounted to present value according to the following equation:
    
    .. math::
      \text{NPV}_{repl} = \sum_{t \in T_{repl}}\frac{\text{IC}_{bat}}{(1 + d)^t},

    where :math:`T_{repl}` is the set of replacement years. The degradation costs of the battery :math:`C_{deg}` are calculated as:

    .. math::
      C_{deg} = \sum^{T-1}_{t=0}\frac{C_{kwh} \cdot E_{dch}}{\text{DoD} \cdot N_{cycles} \cdot (1 + d)^t},

    where :math:`E^{dch}` is the total energy discharged by the battery during the simulation, :math:`\text{DoD}` is the depth of discharge and :math:`N_{cycles}` is the number of cycles performed by the battery.

    Args:
        project_lifetime (:type:`int | float`): Total project lifetime in [years].
        maintenance_cost_rate (:type:`int | float`): Operation and maintenance cost rate based on installation costs in [decimal].
        discount_rate (:type:`int | float`): Discount rate (per year) during the project lifetime in [decimal].

    Returns:
        :type:`float`: Total Net Present Cost of the battery in present value in [US$].
    '''

    years = np.arange(project_lifetime)
    discount = ((1 + discount_rate) ** years)
    # Installation cost (CAPEX)
    installation_cost = self.cost_per_kwh * self.capacity
    NPC = installation_cost
    # O&M costs (discounted)
    OM_cost = (installation_cost * maintenance_cost_rate) / discount
    NPC += np.sum(OM_cost)
    # Replacement costs (discounted)
    t_repl = min(self.lifetime, self.number_of_cycles / self.cycles)
    replacement_years = np.arange(t_repl, project_lifetime, t_repl)
    if len(replacement_years) > 0:
        NPV_repl = installation_cost / ((1 + discount_rate) ** replacement_years)
        NPC += np.sum(NPV_repl)
    # Degradation costs
    degradation_cost = (self.cost_per_kwh * np.sum(self.energy_discharged)) / (self.depth_of_discharge * self.number_of_cycles * discount)
    NPC += np.sum(degradation_cost)
    return NPC