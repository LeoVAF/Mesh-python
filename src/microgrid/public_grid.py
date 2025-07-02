import numpy as np

class PublicGrid:
  ''' Represents a AC public grid in the microgrid system. This class is used to manage the public grid's properties and behaviors.
  
  Args:
    cost_per_kwh (:type:`int | float`): Cost per kWh of the public grid in [US$].
    credit_rate (:type:`int | float`): Credit rate when sending energy to the public grid between 0 and 1.

  Raises:
    TypeError: If the input is not the expected type.
    ValueError: If the input is not the allowed value.
  '''

  def __init__(self,
               cost_per_kwh: int | float = 0.2,
               credit_rate: int | float = 0):
    
    self.cost_per_kwh: int | float
    ''' Cost per kWh of the public grid in [US$]. '''
    self.credit_rate: int | float
    ''' Compensation percentage when sending energy to the public grid between 0 and 1. '''
    self.grid_cost: float = 0.0
    ''' Total public grid cost in [US$]. '''
    self.energy_purchased: np.ndarray[np.float64] | None = None
    ''' Numpy array to store the energy purchased at each time step in [kWh]. '''
    self.energy_credit: float = 0.0
    ''' Energy credit stored on the public grid in [kWh]. '''
    self.energy_to_compensate: float = 0.0
    ''' Energy that will be credited next month in [kWh]. '''
    self.next_month: int = 0
    ''' Variable to mark the month to account for energy compensated. '''
    self.energy_compensated: np.ndarray[np.float64] | None = None
    ''' Numpy array to store the energy compensated at each time step in [kWh]. '''

    self.cost_per_kwh = cost_per_kwh
    self.credit_rate = credit_rate

  def initialize(self, hour_steps: int) -> None:
    ''' Initializes the components of the public grid.
    
    Args:
      hour_steps (:type:`int`): Number of hour steps in the simulation.
    '''
    
    self.energy_purchased = np.zeros(hour_steps)
    self.energy_compensated = np.zeros(hour_steps)

  def store_energy_credit(self, surplus_energy: int | float) -> None:
    ''' Stores the energy credit to compensate.

    Args:
      surplus_energy (:type:`int | float`): The amount of surplus energy to store in [kWh].
      indexes (:type:`int`): The time step at which the energy is stored.
    '''

    self.energy_to_compensate += surplus_energy * self.credit_rate

  def purchase_energy(self, demanding_energy: int | float, t: int) -> int | float:
    ''' Purchases energy from the public grid, compensating with available credits.

    Args:
      demanding_energy (:type:`int | float`): Energy demand in [kWh].
      t (:type:`int`): Time step.
    '''
    
    # Get month number
    month_number = t // 30
    # Update credit if new month started
    if self.next_month < month_number:
        self.next_month = month_number
        self.energy_credit += self.energy_to_compensate
        self.energy_to_compensate = 0.0
    # Compensate as much as possible
    compensated = min(demanding_energy, self.energy_credit)
    self.energy_compensated[t] = compensated
    self.energy_credit -= compensated
    # Buy the remaining energy
    energy_to_purchase = demanding_energy - compensated
    self.energy_purchased[t] = energy_to_purchase
    self.grid_cost += energy_to_purchase * self.cost_per_kwh