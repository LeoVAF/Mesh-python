from microgrid.photovoltaic_panel import PhotovoltaicPanel
from microgrid.wind_turbine import WindTurbine
from microgrid.inverter import Inverter
from microgrid.battery import Battery
from microgrid.public_grid import PublicGrid

import numpy as np

class Microgrid:
  '''
  Microgrid simulation.

  Args:
    load (:type:`np.ndarray[np.float64]`): A numpy array with the demanding load in [kWh].
    temperature (:type:`np.ndarray[np.float64]`): A numpy array with the temperature in [ºC].
    solar_radiation (:type:`np.ndarray[np.float64]`): A numpy array with solar radiation in [kWh/m^2].
    wind_velocity (:type:`np.ndarray[np.float64]`): A numpy array with the wind velocity in [m/s].
    wind_height (:type:`int | float`): The height where the wind speed was measured in [m].
    lifetime (:type:`int | float`): Microgrid lifetime in years.
    photovoltaic_panel (:class:`~microgrid.photovoltaic_panel.PhotovoltaicPanel` :type:`| None`): A :class:`~microgrid.photovoltaic_panel.PhotovoltaicPanel` instance.
    wind_turbine (:class:`~microgrid.wind_turbine.WindTurbine` :type:`| None`): A :class:`~microgrid.wind_turbine.WindTurbine` instance.
    inverter (:class:`~microgrid.inverter.Inverter` :type:`| None`): A :class:`microgrid.inverter.Inverter` instance.
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
               wind_height: int | float,
               lifetime: int | float = 24,
               photovoltaic_panel: PhotovoltaicPanel = None,
               wind_turbine: WindTurbine = None,
               inverter: Inverter = None,
               battery: Battery = None,
               public_grid: PublicGrid = None):
    
    self.load: np.ndarray[np.float64]
    ''' A numpy array with the demanding load in [kWh]. '''
    self.temperature: np.ndarray[np.float64]
    ''' A numpy array with the temperature in [ºC]. '''
    self.solar_radiation: np.ndarray[np.float64]
    ''' A numpy array with solar radiation in [kWh/m^2]. '''
    self.wind_velocity: np.ndarray[np.float64]
    ''' A numpy array with the wind velocity in [m/s]. '''
    self.wind_height: int | float
    ''' The height where the wind speed was measured in [m]. '''
    self.lifetime : int | float
    ''' Microgrid lifetime in years. '''
    self.photovoltaic_panel: PhotovoltaicPanel | None
    ''' A :class:`~microgrid.photovoltaic_panel.PhotovoltaicPanel` instance. '''
    self.wind_turbine: WindTurbine | None
    ''' A :class:`~microgrid.wind_turbine.WindTurbine` instance. '''
    self.inverter: Inverter | None
    ''' A :class:`microgrid.inverter.Inverter` instance. '''
    self.battery: Battery | None
    ''' A :class:`~microgrid.battery.Battery` instance. '''
    self.public_grid: PublicGrid | None
    ''' A :class:`~microgrid.public_grid.PublicGrid` instance. '''
    self.generated_power: np.ndarray[np.float64]
    ''' Numpy array to store the accumulated generated power by generators. '''
    
    self.load = load
    self.temperature = temperature
    self.solar_radiation = solar_radiation
    self.wind_velocity = wind_velocity
    self.wind_height = wind_height
    self.lifetime = lifetime
    self.photovoltaic_panel = photovoltaic_panel
    self.wind_turbine = wind_turbine
    self.inverter = inverter
    self.battery = battery
    self.public_grid = public_grid
    self.generated_power = np.zeros(len(load))

  def generate_power(self):
    ''' Generates power by generators. '''

    # Get the generated power by photovoltaic panels
    if self.photovoltaic_panel is not None:
      self.generated_power += self.photovoltaic_panel.generate_power(self.temperature, self.solar_radiation)
    # Get the generated power by wind_turbines
    if self.wind_turbine is not None:
      self.generated_power += self.wind_turbine.generate_power(self.wind_velocity, self.wind_height)

  def run(self):
    ''' Runs the Microgrid simulation. '''

    # Generate energy by generators
    self.generate_power()
