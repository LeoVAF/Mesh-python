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
    solar_radiation (:type:`np.ndarray[np.float64]`): A numpy array with solar radiation in kWh/m^2.
    wind_velocity (:type:`np.ndarray[np.float64]`): A numpy array with the wind velocity in m/s.
    inverter_eficiency (:type:`int | float`): Efficiency of the inverter between 0 and 1.
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
               inverter_efficiency: int | float = 0.7,
               photovoltaic_panel: PhotovoltaicPanel = None,
               wind_turbine: WindTurbine = None,
               battery: Battery = None,
               public_grid: PublicGrid = None):
    
    self.load: np.ndarray[np.float64]
    ''' A numpy array with the demanding load in kWh. '''
    self.temperature: np.ndarray[np.float64]
    ''' A numpy array with the temperature in ºC. '''
    self.solar_radiation: np.ndarray[np.float64]
    ''' A numpy array with solar radiation in kWh/m^2. '''
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
    self.generated_energy: np.ndarray[np.float64]
    ''' Numpy array to store the accumulated generated energy by generator. '''
    self.inverter_efficiency: int | float
    ''' Efficiency of the inverter between 0 and 1. '''
    
    self.load = load
    self.temperature = temperature
    self.solar_radiation = solar_radiation
    self.wind_velocity = wind_velocity
    self.photovoltaic_panel = photovoltaic_panel
    self.wind_turbine = wind_turbine
    self.battery = battery
    self.public_grid = public_grid
    self.inverter_efficiency = inverter_efficiency
    self.generated_energy = np.zeros(len(load))

  def generate_energy(self):
    ''' Generates Energy by generators. '''

    # Get the generated energy by photovoltaic panels
    self.generated_energy += self.photovoltaic_panel.generate_energy(self.temperature, self.solar_radiation) if self.photovoltaic_panel is not None else 0
    # Get the generated energy by photovoltaic panels
    self.generated_energy += self.wind_turbine.generate_energy(self.wind_velocity) if self.wind_turbine is not None else 0

  def run(self):
    ''' Runs the Microgrid simulation. '''

    # Generate energy by generators
    self.generate_energy()
