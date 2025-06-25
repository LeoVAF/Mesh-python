from microgrid.photovoltaic_panel import PhotovoltaicPanel
from microgrid.wind_turbine import WindTurbine
from microgrid.inverter import Inverter
from microgrid.battery import Battery
from microgrid.public_grid import PublicGrid

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
    ''' Microgrid lifetime in [year]. '''
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
    self.hour_steps: int
    ''' Number of hour steps in the simulation. '''
    self.generated_by_pv: np.ndarray[np.float64]
    ''' Numpy array to store the generated power by photovoltaic panels at each time step in [kWh]. '''
    self.generated_by_wt: np.ndarray[np.float64]
    ''' Numpy array to store the generated power by wind turbines at each time step in [kWh]. '''
    self.battery_charge: np.ndarray[np.float64]
    ''' Numpy array to store the battery charge at each time step in [kWh]. '''
    self.battery_discharge: np.ndarray[np.float64]
    ''' Numpy array to store the battery discharge at each time step in [kWh]. '''
    self.energy_surplus: np.ndarray[np.float64]
    ''' Numpy array to store the energy surplus at each time step in [kWh]. '''
    self.compensated_energy: np.ndarray[np.float64]
    ''' Numpy array to store the compensated energy at each time step in [kWh]. '''

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
    self.hour_steps = len(load)
    self.generated_by_pv = np.zeros(self.hour_steps)
    self.generated_by_wt = np.zeros(self.hour_steps)
    self.battery_charge = np.zeros(self.hour_steps)
    self.battery_discharge = np.zeros(self.hour_steps)
    self.energy_surplus = np.zeros(self.hour_steps)
    self.compensated_energy = np.zeros(self.hour_steps)

  def initialize(self) -> None:
    ''' Initializes the variables. '''
    
    # Initialize the battery state of charge
    if self.battery is not None:
      self.battery.initialize_state_of_charge(self.hour_steps)
    if self.public_grid is not None:
      self.public_grid.initialize_power_array(self.hour_steps)

  def generate_energy(self) -> None:
    ''' Generates energy by generators. '''

    # Get the generated energy by photovoltaic panels
    if self.photovoltaic_panel is not None:
      self.generated_by_pv = self.photovoltaic_panel.generate_power(self.temperature, self.solar_radiation)
    # Get the generated energy by wind_turbines
    if self.wind_turbine is not None:
      self.generated_by_wt = self.wind_turbine.generate_power(self.wind_velocity, self.wind_height)

  def run_hourly_simulation(self) -> None:
    ''' Runs the hourly simulation of the microgrid.'''

    # Calculate the total generated energy by all generators
    generated_energy = self.generated_by_pv + self.generated_by_wt
    # Calculate the time steps in which there is energy surplus
    surplus_mask = np.where(generated_energy > self.load, True, False)
    # Calculate the difference between generated energy and load
    difference_at_time = np.abs(generated_energy - self.load)
    for t, there_is_surplus in enumerate(surplus_mask):
      # If there is surplus energy
      if there_is_surplus:
        remaining_energy_surplus = difference_at_time[t]
        if self.battery is not None:
          # Charge the battery with the surplus energy
          remaining_energy_surplus = self.battery.charge(remaining_energy_surplus, t)
          
        ''' ############ If there is surplus energy, send it to the public grid ############# '''
        if self.public_grid is not None:
          # Store the surplus energy in the public grid
          self.public_grid.store_credit(remaining_energy_surplus, t)
      # If there is deficit
      else:
        pass

  def run(self) -> None:
    ''' Runs the Microgrid simulation. '''
    self.initialize()

    # Generate energy by generators
    self.generate_energy()
    # Run hourly simulation
    self.run_hourly_simulation()