from microgrid.photovoltaic_panel import PhotovoltaicPanel
from microgrid.wind_turbine import WindTurbine
from microgrid.battery import Battery
from microgrid.public_grid import PublicGrid

import numpy as np

class Microgrid:
  '''
  Microgrid simulation.

  Args:
    load (:type:`np.ndarray[np.float64]`): A numpy array with the demanding load in kWh.
    temperature (:type:`np.ndarray[np.float64]`): A numpy array with the temperature in ºC.
    solar_radiation (:type:`np.ndarray[np.float64]`): A numpy array with solar radiation in Wh/m^2.
    wind_velocity (:type:`np.ndarray[np.float64]`): A numpy array with the wind velocity in m/s.
    photovoltaic_panel (:class:`~microgrid.photovoltaic_panel.PhotovoltaicPanel` :type:`| None`): A :class:`~microgrid.photovoltaic_panel.PhotovoltaicPanel` instance.
    wind_turbine (:class:`~microgrid.wind_turbine.WindTurbine` :type:`| None`): A :class:`~microgrid.wind_turbine.WindTurbine` instance.
    battery (:class:`~microgrid.battery.Battery` :type:`| None`): A :class:`~microgrid.battery.Battery` instance.
    public_grid (:class:`~microgrid.public_grid.PublicGrid` :type:`| None`): A :class:`~microgrid.public_grid.PublicGrid` instance.

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
    ''' :class:`~microgrid.photovoltaic_panel.PhotovoltaicPanel` instance. '''
    self.wind_turbine = wind_turbine
    ''' :class:`~microgrid.wind_turbine.WindTurbine` instance. '''
    self.battery = battery
    ''' :class:`~microgrid.battery.Battery` instance. '''
    self.public_grid = public_grid
    ''' :class:`~microgrid.PublicGrid` instance. '''

  def generate_energy():
    pass