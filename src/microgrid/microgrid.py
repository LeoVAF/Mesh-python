from microgrid.photovoltaic_panel import PhotovoltaicPanel
from microgrid.wind_turbine import WindTurbine
from microgrid.battery import Battery
from microgrid.public_grid import PublicGrid
from microgrid.inverter import Inverter
from microgrid.converter import Converter

import numpy as np

class Microgrid:
  ''' Microgrid simulation.

  Args:
    load (:type:`np.ndarray[np.float64]`): A numpy array with the demanding load in [kWh].
    temperature (:type:`np.ndarray[np.float64]`): A numpy array with the temperature in [ºC].
    solar_radiation (:type:`np.ndarray[np.float64]`): A numpy array with solar radiation in [kWh/m^2].
    wind_velocity (:type:`np.ndarray[np.float64]`): A numpy array with the wind velocity in [m/s].
    wind_height (:type:`int | float`): The height where the wind speed was measured in [m].
    lifetime (:type:`int | float`): Microgrid lifetime in [year].
    photovoltaic_panel (:class:`~microgrid.photovoltaic_panel.PhotovoltaicPanel` :type:`| None`): A :class:`~microgrid.photovoltaic_panel.PhotovoltaicPanel` instance.
    wind_turbine (:class:`~microgrid.wind_turbine.WindTurbine` :type:`| None`): A :class:`~microgrid.wind_turbine.WindTurbine` instance.
    battery (:class:`~microgrid.battery.Battery` :type:`| None`): A :class:`~microgrid.battery.Battery` instance.
    public_grid (:class:`~microgrid.public_grid.PublicGrid` :type:`| None`): A :class:`~microgrid.public_grid.PublicGrid` instance.
    inverter (:class:`~microgrid.inverter.Inverter` :type:`| None`): A :class:`microgrid.inverter.Inverter` instance.
    converter (:class:`~microgrid.converter.Converter` :type:`| None`): A :class:`microgrid.converter.Converter` instance.

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
               photovoltaic_panel: PhotovoltaicPanel | None = None,
               wind_turbine: WindTurbine | None = None,
               battery: Battery | None = None,
               public_grid: PublicGrid | None = None,
               inverter: Inverter | None = None,
               converter: Converter | None = None) -> None:
    
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
    ''' Microgrid lifetime in [year]. '''
    self.photovoltaic_panel: PhotovoltaicPanel | None
    ''' A :class:`~microgrid.photovoltaic_panel.PhotovoltaicPanel` instance. '''
    self.wind_turbine: WindTurbine | None
    ''' A :class:`~microgrid.wind_turbine.WindTurbine` instance. '''
    self.battery: Battery | None
    ''' A :class:`~microgrid.battery.Battery` instance. '''
    self.public_grid: PublicGrid | None
    ''' A :class:`~microgrid.public_grid.PublicGrid` instance. '''
    self.inverter: Inverter | None
    ''' A :class:`microgrid.inverter.Inverter` instance. '''
    self.converter: Converter | None
    ''' A :class:`microgrid.converter.Converter` instance. '''
    self.hour_steps: int
    ''' Number of hour steps in the simulation. '''
    self.surplus_energy: np.ndarray[np.float64]
    ''' Numpy array to store the energy surplus that will be throw away at each time step in [kWh]. '''

    self.load = load
    self.temperature = temperature
    self.solar_radiation = solar_radiation
    self.wind_velocity = wind_velocity
    self.wind_height = wind_height
    self.lifetime = lifetime
    self.photovoltaic_panel = photovoltaic_panel
    self.wind_turbine = wind_turbine
    self.battery = battery
    self.public_grid = public_grid
    self.inverter = inverter
    self.converter = converter
    self.hour_steps = len(load)
    self.surplus_energy = np.zeros(self.hour_steps)

  def initialize(self) -> None:
    ''' Initializes the microgrid components. '''
    
    if self.photovoltaic_panel is not None:
      self.photovoltaic_panel.initialize(self.hour_steps)
    if self.wind_turbine is not None:
      self.wind_turbine.initialize(self.hour_steps)
    if self.battery is not None:
      self.battery.initialize(self.hour_steps)
    if self.public_grid is not None:
      self.public_grid.initialize(self.hour_steps)

  def generate_energy(self) -> None:
    ''' Generates energy by generators. '''

    # Get the generated energy by photovoltaic panels
    if self.photovoltaic_panel is not None:
      self.photovoltaic_panel.generate_power(self.temperature, self.solar_radiation)
    # Get the generated energy by wind_turbines
    if self.wind_turbine is not None:
      self.wind_turbine.generate_power(self.wind_velocity, self.wind_height)

  def dispatch_energy(self) -> None:
    ''' Runs the hourly simulation of the microgrid.'''

    # Get the functions to charge and discharge the battery
    if self.battery is not None:
      charge = self.battery.charge
      discharge = self.battery.discharge
      # Check if the battery has a converter to charge energy and get its efficiency
      if self.converter is not None:
        converter_efficiency = self.converter.efficiency
      else:
        converter_efficiency = 1.0
      # Check if the battery has an inverter to discharge energy and get its efficiency
      if self.inverter is not None:
        inverter_efficiency = self.inverter.efficiency
      else:
        inverter_efficiency = 1.0
    else:
      charge = lambda x, y, z: x
      discharge = lambda x, y, z: x
    # # Get the functions to compensate and buy energy from the public grid
    # if self.public_grid is not None:
    #   compensate = self.public_grid.store_credit
    #   buy = self.public_grid.buy_energy
    # else:
    #   compensate = lambda x, t: x

    # Calculate the total generated energy by all generators
    generated_energy = self.photovoltaic_panel.output_power + self.wind_turbine.output_power
    # Adjust demanding load for inverter efficiency
    demanding_load_adjusted = self.load / inverter_efficiency
    # Calculate the time steps in which there is energy surplus
    surplus_mask = np.where(generated_energy > demanding_load_adjusted, True, False)
    # Calculate the difference between generated energy and load
    difference_at_time = np.abs(generated_energy - demanding_load_adjusted)
    for t, there_is_surplus in enumerate(surplus_mask):
      # If there is surplus energy
      if there_is_surplus:
        remaining_surplus_energy = difference_at_time[t]
        # Charge the battery with the surplus energy (if the battery is connected)
        remaining_surplus_energy_after_charging = charge(remaining_surplus_energy, converter_efficiency, t)
        # Send the surplus energy to the public grid (if the public grid is connected)
        ''' Send surplus energy for net metering or throw it away. '''
      # If there is deficit energy
      else:
        remaining_deficit_energy = difference_at_time[t]
        # Discharge the battery to cover the deficit (if the battery is connected)
        remaining_deficit_energy_after_discharging = discharge(remaining_deficit_energy, inverter_efficiency, t)
        # If there is still deficit, buy energy from the public grid (if the public grid is connected)
        ''' Buy energy from the public grid if it exists. '''

  def economic_analysis(self) -> None:
    ''' Performs the economic analysis of the microgrid and its components. '''

    pass
    # # Perform economic analysis for photovoltaic panels
    # if self.photovoltaic_panel is not None:
    #   self.photovoltaic_panel.economic_analysis(self.hour_steps)
    # # Perform economic analysis for wind turbines
    # if self.wind_turbine is not None:
    #   self.wind_turbine.economic_analysis(self.hour_steps)
    # # Perform economic analysis for inverter
    # if self.inverter is not None:
    #   self.inverter.economic_analysis(self.hour_steps)
    # # Perform economic analysis for battery
    # if self.battery is not None:
    #   self.battery.economic_analysis(self.hour_steps)
    # # Perform economic analysis for public grid
    # if self.public_grid is not None:
    #   self.public_grid.economic_analysis(self.hour_steps)

  def run(self) -> None:
    ''' Runs the Microgrid simulation. '''

    # Initialize the microgrid components
    self.initialize()
    # Generates energy by generators
    self.generate_energy()
    # Simulates the energy dispatch
    self.dispatch_energy()
    # Performs economic analysis
    self.economic_analysis()

    # Disconsider the first time step for the battery state of charge
    self.battery.state_of_charge = self.battery.state_of_charge[1:]