import numpy as np

class PublicGrid:
  ''' Represents a public grid in the microgrid system. This class is used to manage the public grid's properties and behaviors.
  
  Args:
    cost_per_kwh (:type:`int | float`): Cost per kWh of the public grid in [US$].
    metering_compensation (:type:`int | float`): Compensation percentage when sending energy to the public grid between 0 and 1.

  Raises:
    TypeError: If the input is not the expected type.
    ValueError: If the input is not the allowed value.
  '''

  def __init__(self,
               cost_per_kwh: int | float = 0.2,
               metering_compensation: int | float = 0):
    
    self.cost_per_kwh: int | float
    ''' Cost per kWh of the public grid in [US$]. '''
    self.metering_compensation: int | float
    ''' Compensation percentage when sending energy to the public grid between 0 and 1. '''
    self.grid_cost: int | float = 0
    ''' Total public grid cost in [US$]. '''
    self.grid_power: np.ndarray[np.float64] | None = None
    ''' Power array for the public grid in [kWh]. '''

    self.cost_per_kwh = cost_per_kwh
    self.metering_compensation = metering_compensation

  def initialize_power_array(self, hour_steps: int) -> None:
    ''' Initializes the power array for the public grid. 
    
    Args:
      hour_steps (:type:`int`): Number of hour steps in the simulation.
    '''
    
    self.grid_power = np.zeros(hour_steps)

  def store_credit(self, surplus_energy: np.ndarray[np.float64], indexes: np.ndarray[np.integer]) -> None:
    ''' Stores the credit in the public grid.

    Args:
      surplus_energy (:type:`np.ndarray[np.float64]`): The amount of surplus energy to store in [kWh].
      indexes (:type:`np.ndarray[np.integer]`): The time steps at which the energy is stored.
    '''

    pass

  def buy_energy(self, demanding_energy: np.ndarray[np.float64], indexes: np.ndarray[np.integer]) -> None:
    ''' Buys energy from the public grid.

    Args:
      demanding_energy (:type:`np.ndarray[np.float64]`): The amount of energy to buy in [kWh].
      indexes (:type:`np.ndarray[np.integer]`): The time steps at which the energy is bought.
    '''
    
    pass