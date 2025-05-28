import numpy as np

class WindTurbine:
  '''
  Wind turbine simulation.
  
  Args:
    n_turbines (:type:`int`): Number of wind turbines.
    rated_power (:type:`int | float`): Rated power of the wind turbine in kW.

  Raises:
    TypeError: If the input is not the expected type.
    ValueError: If the input is not the allowed value.
  '''

  def __init__(self,
               n_turbines: int,
               rated_power: int | float):
    
    self.rated_power: float
    ''' Rated power of the wind turbine in kW. '''
    self.n_turbines: int
    ''' Number of wind turbines. '''

    self.n_turbines = n_turbines
    self.rated_power = rated_power
  
  def generate_energy(self, wind_velocity: np.ndarray[np.float64]):
    pass