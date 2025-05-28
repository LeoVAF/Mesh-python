import numpy as np

class PhotovoltaicPanel:
  '''
  Class representing a photovoltaic panel in a microgrid system.
  
  Args:
    cost_per_kw (:type:`int | float`): Photovoltaic panel cost per kW maximum power.
    rated_power (:type:`int | float`): The maximum power output of the photovoltaic panel in kilowatts in kW.
  
  Raises:
    TypeError: If the input is not the expected type.
    ValueError: If the input is not the allowed value.
  '''

  def __init__(self,
               cost_per_kw: int | float,
               rated_power: int | float):
    
    self.cost_per_kwh: int | float
    ''' Photovoltaic panel cost per kW maximum power. '''
    self.rated_power: int | float
    ''' Maximum power output of the photovoltaic panel in kW. '''
    self.output_power: np.ndarray[np.float64] | None
    ''' Output power generated. '''

    self.cost_per_kwh = cost_per_kw
    self.rated_power = rated_power
    self.output_power = None

  def generate_energy(self, temperature, solar_radiation):
    ''' Generate energy.
    
      Args:
        temperature (np.ndarray[np.float64]): Numpy array with the temperature in ºC at the time.
        solar_radiation (np.ndarray[np.float64]): Numpy array with the solar radiation in kWh/m^2 at the time.
    '''

    irradiance_ref = 1 # Reference irradiance kWh/m^2
    temperature_ref = 25 # Reference temperature in ºC
    temperature_coefficient = 3.7e-3 # Temperature coefficient of maximum power in 1/°C
    self.output_power = self.rated_power * (solar_radiation/irradiance_ref) * (1 + temperature_coefficient*(temperature + 0.0256 * solar_radiation - temperature_ref))
    return self.output_power