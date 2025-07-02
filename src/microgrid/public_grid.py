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
    self.bought_energy: np.ndarray[np.float64] | None = None
    ''' Numpy array to store the bought energy at each time step in [kWh]. '''
    self.energy_credit: float = 0.0
    ''' Energy credit stored on the public grid in [kWh]. '''
    self.energy_to_compensate: float = 0.0
    ''' Energy that will be credited next month in [kWh]. '''
    self.next_month: int = 0
    ''' Variable to mark the month to account for compensated energy. '''
    self.compensated_energy: np.ndarray[np.float64] | None = None
    ''' Numpy array to store the compensated energy at each time step in [kWh]. '''

    self.cost_per_kwh = cost_per_kwh
    self.credit_rate = credit_rate

  def initialize(self, hour_steps: int) -> None:
    ''' Initializes the components of the public grid.
    
    Args:
      hour_steps (:type:`int`): Number of hour steps in the simulation.
    '''
    
    self.bought_energy = np.zeros(hour_steps)
    self.compensated_energy = np.zeros(hour_steps)

  def store_energy_credit(self, surplus_energy: int | float) -> int | float:
    ''' Stores the energy credit to compensate.

    Args:
      surplus_energy (:type:`int | float`): The amount of surplus energy to store in [kWh].
      indexes (:type:`int`): The time step at which the energy is stored.
    '''

    self.energy_to_compensate += surplus_energy * self.credit_rate
    return 0

  def buy_energy(self, demanding_energy: int | float, t: int) -> int | float:
    ''' Buys energy from the public grid.

    Args:
      demanding_energy (:type:`int | float`): The amount of energy to buy in [kWh].
      indexes (:type:`int`): The time step at which the energy is bought.
    '''
    
    # Get the month number
    month_number = t // 30
    # If you have not yet accounted for the energy sent for compensation, then account for it
    if self.next_month < month_number:
      self.next_month = month_number
      self.energy_credit += self.energy_to_compensate
      self.energy_to_compensate = 0.0
    # Reduce energy purchases with energy credit
    if demanding_energy <= self.energy_credit:
      self.compensated_energy[t] = self.energy_credit - demanding_energy
      self.energy_credit -= demanding_energy
    # Buy energy
    else:
      # Compensates for energy that is in credit
      self.compensated_energy[t] = self.energy_credit
      self.energy_credit = 0
      # Buy the remaining energy
      energy_to_buy = demanding_energy - self.energy_credit
      self.bought_energy[t] = energy_to_buy
      self.cost += energy_to_buy * self.cost_per_kwh