import numpy as np

class PublicGrid:
  ''' Represents a AC public grid in the microgrid system. This class is used to manage the public grid's properties and behaviors.
  
  Args:
    cost_per_kwh (:type:`int | float`): Cost per kWh of the public grid in [$].
    tariff_growth (:type:`int | float`): Tariff growth over the course of the microgrid project between 0 and 1.
    credit_rate (:type:`int | float`): Credit rate when sending energy to the public grid between 0 and 1.

  Raises:
    TypeError: If the input is not the expected type.
    ValueError: If the input is not the allowed value.
  '''

  def __init__(self,
               cost_per_kwh: int | float = 0.2,
               tariff_growth: int | float = 0.05,
               credit_rate: int | float = 0):
    
    self.cost_per_kwh: int | float
    ''' Cost per kWh of the public grid in [$/kWh]. '''
    self.tariff_growth: int | float
    ''' Tariff growth over the course of the microgrid project between 0 and 1. '''
    self.credit_rate: int | float
    ''' Compensation percentage when sending energy to the public grid between 0 and 1. '''
    self.operation_cost: float = 0.0
    ''' Grid purchasing costs in [$]. '''
    self.energy_purchased: np.ndarray[np.float64] | None = None
    ''' Numpy array to store the energy purchased at each time step in [kWh]. '''
    self.energy_credit: float = 0.0
    ''' Energy credit stored on the public grid in [kWh]. '''
    self.energy_credited: np.ndarray[np.float64] | None = None
    ''' Numpy array to store the energy credited at each time step in [kWh]. '''
    self.energy_to_credit: float = 0.0
    ''' Energy that will be credited next month in [kWh]. '''
    self.next_month: int = 0
    ''' Variable to mark the month to account for energy credited. '''
    self.energy_compensated: np.ndarray[np.float64] | None = None
    ''' Numpy array to store the energy compensated at each time step in [kWh]. '''
    self.meet_demand: np.ndarray[np.float64] | None = None
    ''' Energy that will effectively meet demand in [kWh]. '''

    self.cost_per_kwh = cost_per_kwh
    self.tariff_growth = tariff_growth
    self.credit_rate = credit_rate

  def initialize(self, hour_steps: int) -> None:
    ''' Initializes the components of the public grid.
    
    Args:
      hour_steps (:type:`int`): Number of hour steps in the simulation.
    '''
    
    self.energy_purchased = np.zeros(hour_steps)
    self.energy_credited = np.zeros(hour_steps)
    self.energy_compensated = np.zeros(hour_steps)
    self.meet_demand = np.zeros(hour_steps)

  def update_month(self, t: int) -> None:
    ''' Updates the month to account for energy compensated.

    Args:
      t (:type:`int`): Time step.
    '''

    # Get month number
    month_number = t // 720
    # Update credit if new month started
    if self.next_month < month_number:
        self.next_month = month_number
        self.energy_credit += self.energy_to_credit
        self.energy_credited[t] = self.energy_to_credit
        self.energy_to_credit = 0.0

  def store_energy_credit(self, surplus_energy: int | float, inverter_efficiency: int | float, t: int) -> None:
    ''' Stores the energy credit to compensate.

    Args:
      surplus_energy_adjusted (:type:`int | float`): The amount of surplus energy adjusted by the microgrid inverter to store in [kWh].
      inverter_efficiency (:type:`int | float`): The efficiency of the inverter between 0 and 1.
      t (:type:`int`): Time step.
    
    Returns:
      :type:`float`: Returns 0.0 for compatibility with the Microgrid class.
    '''

    # Credit the energy sent to the public grid
    self.energy_to_credit += surplus_energy * inverter_efficiency * self.credit_rate
    # Accounts for credited energy
    self.update_month(t)
    return 0.0

  def purchase_energy(self, energy_demanded: int | float, t: int) -> int | float:
    ''' Purchases energy from the public grid, compensating with available credits.

    Args:
      energy_demanded (:type:`int | float`): Energy demanded in [kWh].
      t (:type:`int`): Time step.
    '''
    
    # Compensate as much as possible
    compensated = min(energy_demanded, self.energy_credit)
    self.energy_compensated[t] = compensated
    self.energy_credit -= compensated
    # Buy the remaining energy
    energy_to_purchase = energy_demanded - compensated
    self.energy_purchased[t] = energy_to_purchase
    # The energy that effectively meets the demand
    self.meet_demand[t] = compensated + energy_to_purchase
    # Calculate the operation cost
    self.operation_cost += energy_to_purchase * self.cost_per_kwh
    # Accounts for compensated energy
    self.update_month(t)

  def economic_analysis(self, project_lifetime: int | float, discount_rate: int | float) -> float:
    r''' Performs the economic analysis of the public grid. It is calculated according to the following equation:

    .. math::
      \sum^{T}_{t=1}\frac{C_{grid}(1 + e)^t}{(1 + d)^t},

    where:
    
    - :math:`T` is the project lifetime in [years];
    - :math:`C_{grid}` is the simulated purchasing cost during a year in [$];
    - :math:`e` is the tariff growth rate during the project lifetime;
    - :math:`d` is the discount rate during the project lifetime.

    Args:
      project_lifetime (:type:`int | float`): The microgrid project lifetime in [years].
      discout_rate (:type:`int | float`): Discount rate (per year) during the project lifetime.
    
    Returns:
      :type:`float`: Total Net Present Cost of purchasing from the public grid in present value in [$].
    '''

    # Calculate the Net Present Cost for the purchasing from public grid
    if self.tariff_growth == discount_rate:
      # If the tariff growth is equal to the discount rate, the NPV is simply the operation cost times the project lifetime
      NPV = self.operation_cost * project_lifetime
    else:
      NPV = self.operation_cost * (1 + discount_rate) / (discount_rate - self.tariff_growth) * (1 - ((1 + self.tariff_growth) / (1 + discount_rate)) ** project_lifetime)

    return NPV