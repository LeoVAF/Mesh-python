import numpy as np


class UtilityGrid:
  ''' Represents an AC utility grid in the microgrid system. This class is used to manage the utility grid's properties and behaviors.
  
  Args:
    cost_per_kwh (:type:`float`): Cost per kWh of utility-grid electricity in [$/kWh]
    tariff_growth (:type:`float`): Tariff growth over the course of the microgrid project between 0 and 1.
    credit_rate (:type:`float`): Credit rate when sending energy to the utility grid between 0 and 1.
  '''

  def __init__(self,
               cost_per_kwh: float = 0.2,
               tariff_growth: float = 0.05,
               credit_rate: float = 0,
               compensation_period_hours: int = 730):
    
    self.cost_per_kwh: float
    ''' Cost per kWh of utility-grid electricity in [$/kWh]. '''
    self.tariff_growth: float
    ''' Tariff growth over the course of the microgrid project between 0 and 1. '''
    self.credit_rate: float
    ''' Compensation percentage when sending energy to the utility grid between 0 and 1. '''
    self.compensation_period_hours: int
    ''' Length of the compensation period in hours. '''
    self.hours_per_interval: int
    ''' Number of hours in each time interval in the simulation. '''
    self.discount_rate: float
    ''' Discount rate for economic analysis. '''
    self.operation_cost: float
    ''' Grid purchasing costs in [$]. '''
    self.energy_credit: float
    ''' Energy credit stored on the utility grid in [kWh]. '''
    self.pending_credit: float
    ''' Energy that will be credited in the next compensation period in [kWh]. '''
    self.current_compensation_period: int
    ''' Variable to mark the compensation period to account for energy credited. '''
    self.purchased_energy: np.typing.NDArray[np.floating]
    ''' Numpy array to store the purchased energy at each time step in [kWh]. '''
    self.exported_energy: np.typing.NDArray[np.floating]
    ''' Numpy array to store the exported energy at each time step in [kWh].'''
    self.released_credit: np.typing.NDArray[np.floating]
    ''' Numpy array to store the credits released into the credit balance at each time step in [kWh]. '''
    self.compensated_energy: np.typing.NDArray[np.floating]
    ''' Numpy array to store the compensated energy at each time step in [kWh]. '''
    self.meet_demand: np.typing.NDArray[np.floating]
    ''' Energy that will effectively meet demand in [kWh]. '''

    self.cost_per_kwh = cost_per_kwh
    self.tariff_growth = tariff_growth
    self.credit_rate = credit_rate
    self.compensation_period_hours = compensation_period_hours

  def initialize(self, hours: int, hours_per_interval: int, discount_rate: float) -> None:
    ''' Initializes the components of the utility grid.
    
    Args:
      hours (:type:`int`): Number of hours in the simulation.
      hours_per_interval (:type:`int`): Number of hours in each time interval in the simulation.
      discount_rate (:type:`float`): The discount rate for economic analysis.
    '''
    
    self.purchased_energy = np.zeros(hours)
    self.exported_energy = np.zeros(hours)
    self.released_credit = np.zeros(hours)
    self.compensated_energy = np.zeros(hours)
    self.meet_demand = np.zeros(hours)
    self.hours_per_interval = hours_per_interval
    self.discount_rate = discount_rate

    self.operation_cost = 0.0
    self.energy_credit = 0.0
    self.pending_credit = 0.0
    self.current_compensation_period = 0

  def update_compensation_period(self, t: int) -> None:
    ''' Updates the compensation period to account for energy compensated.

    Args:
      t (:type:`int`): Time step.
    '''

    # Get month number
    month_number = t // self.compensation_period_hours
    # Update credit if new month started
    if self.current_compensation_period < month_number:
        self.current_compensation_period = month_number
        self.energy_credit += self.pending_credit
        self.released_credit[t] = self.pending_credit
        self.pending_credit = 0.0

  def export_energy(self, surplus_energy: float, inverter_efficiency: float, t: int) -> float:
    ''' Stores the energy credit to compensate.

    Args:
      surplus_energy (:type:`float`): The amount of surplus energy to store in [kWh].
      inverter_efficiency (:type:`float`): The efficiency of the inverter between 0 and 1.
      t (:type:`int`): Time step.
    
    Returns:
      :type:`float`: There is no surplus when utility grid is connected.
    '''

    # Accounts for credited energy
    self.update_compensation_period(t)
    # Export energy
    energy_to_export = surplus_energy * inverter_efficiency
    self.exported_energy[t] = energy_to_export
    # Credit the energy sent to the utility grid
    self.pending_credit += energy_to_export * self.credit_rate
    return 0.0

  def import_energy(self, demanded_energy: float, t: int) -> None:
    ''' Import energy from the utility grid, compensating with available credits.

    Args:
      demanded_energy (:type:`float`): Demanded energy in [kWh].
      t (:type:`int`): Time step.
    '''
    
    # Accounts for compensated energy
    self.update_compensation_period(t)
    # Compensate as much as possible
    compensated = min(demanded_energy, self.energy_credit)
    self.compensated_energy[t] = compensated
    self.energy_credit -= compensated
    # Buy the remaining energy
    energy_to_purchase = demanded_energy - compensated
    if energy_to_purchase > 0:
      self.purchased_energy[t] = energy_to_purchase
      # Calculate the operation cost
      i = t // self.hours_per_interval
      self.operation_cost += energy_to_purchase * self.cost_per_kwh * ((1 + self.tariff_growth) ** (i)) / ((1 + self.discount_rate) ** (i+1))
    # The energy that effectively meets the demand
    self.meet_demand[t] = compensated + energy_to_purchase

  def economic_analysis(self) -> float:
    r''' Performs the economic analysis of the utility grid. It is calculated according to the following equation:

    .. math::
      \mathrm{NPV}_{grid} = C^{kWh}_{grid} \sum_{t=1}^{T}\frac{E^{pur}_{grid}(t)(1 + e)^{i(t)-1}}{(1 + d)^{i(t)}},

    where:
    
    - :math:`T` is the total number of hourly simulation time steps;
    - :math:`C^{kWh}_{grid}` is the initial utility-grid energy tariff in $/kWh;
    - :math:`e` is the tariff growth rate during the project lifetime;
    - :math:`d` is the discount rate during the project lifetime;
    - :math:`i(t) = \lfloor\frac{t-1}{H}\rfloor + 1` is the respective interval at time step :math:`t`.
    
    Returns:
      :type:`float`: Total Net Present Cost of purchasing from the utility grid in present value in [$].
    '''

    # Calculate the Net Present Cost for the purchasing from utility grid
    return self.operation_cost