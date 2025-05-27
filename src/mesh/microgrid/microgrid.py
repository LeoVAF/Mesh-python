from mesh.microgrid.photovoltaic_panel import PhotovoltaicPanel
from mesh.microgrid.wind_turbine import WindTurbine
from mesh.microgrid.battery import Battery
from mesh.microgrid.public_grid import PublicGrid

import numpy as np

class Microgrid:
  '''
  Microgrid simulation.

  Args:
    load (:type:`np.ndarray[np.float64]`): A numpy array with the demanding load in kWh.
    temperature (:type:`np.ndarray[np.float64]`): A numpy array with the temperature in ºC.
    solar_radiation (:type:`np.ndarray[np.float64]`): A numpy array with solar radiation in Wh/m^2.
    wind_velocity (:type:`np.ndarray[np.float64]`): A numpy array with the wind velocity in m/s.
    metering_compensation (:type:`float`): Compensation percentage when sending energy to the public grid between 0 and 1.

  Raises:
    TypeError: If the input is not the expected type.
    ValueError: If the input is not the allowed value.
  '''

  def __init__(self,
               load: np.ndarray[np.float64],
               temperature: np.ndarray[np.float64],
               solar_radiation: np.ndarray[np.float64],
               wind_velocity: np.ndarray[np.float64],
               photovoltaic_panel: PhotovoltaicPanel = None,
               wind_turbine: WindTurbine = None,
               battery: Battery = None,
               public_grid: PublicGrid = None):
    
    self.load: np.ndarray[np.float64]
    ''' A numpy array with the demanding load in kWh. '''
    self.temperature: np.ndarray[np.float64]
    ''' A numpy array with the temperature in ºC. '''
    self.solar_radiation: np.ndarray[np.float64]
    ''' A numpy array with solar radiation in Wh/m^2. '''
    self.wind_velocity: np.ndarray[np.float64]
    ''' A numpy array with the wind velocity in m/s. '''
    self.photovoltaic_panel: PhotovoltaicPanel
    ''' Photovoltaic panel object. '''
    self.wind_turbine: WindTurbine
    ''' Wind turbine object. '''
    self.battery: Battery
    ''' Battery object. '''
    self.public_grid: PublicGrid
    ''' Public grid object. '''
    
    self.load = load
    ''' Numpy array with the demanding load in kWh. '''
    self.temperature = temperature
    ''' Numpy array with the temperature in ºC. '''
    self.solar_radiation = solar_radiation
    ''' Numpy array with solar radiation in Wh/m^2. '''
    self.wind_velocity = wind_velocity
    ''' Numpy array with wind velocity in m/s. '''
    self.photovoltaic_panel = photovoltaic_panel
    ''' :class:`~mesh.microgrid.photovoltaic_panel.PhotovoltaicPanel` instance. '''
    self.wind_turbine = wind_turbine
    ''' :class:`~mesh.microgrid.wind_turbine.WindTurbine` instance. '''
    self.battery = battery
    ''' :class:`~mesh.microgrid.battery.Battery` instance. '''
    self.public_grid = public_grid
    ''' :class:`~mesh.microgrid.PublicGrid` instance. '''

  def generate_energy():
    pass