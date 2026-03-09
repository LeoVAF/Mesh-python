from simulation.photovoltaic_panel import PhotovoltaicPanel
from simulation.wind_turbine import WindTurbine
from simulation.battery import Battery
from simulation.public_grid import PublicGrid
from simulation.inverter import Inverter
from simulation.converter import Converter

import numpy as np
import numpy.typing as npt
import pandas as pd

class Microgrid:
  ''' Microgrid simulation.

  Args:
    load (:type:`npt.NDArray[np.floating]`): A numpy array with the demanding load in [kWh].
    temperature (:type:`npt.NDArray[np.floating]`): A numpy array with the temperature in [ºC].
    solar_irradiance (:type:`npt.NDArray[np.floating]`): A numpy array with solar irradiance in [kW/m^2].
    wind_velocity (:type:`npt.NDArray[np.floating]`): A numpy array with the wind velocity in [m/s].
    wind_height (:type:`int | float`): The height where the wind speed was measured in [m].
    lifetime (:type:`int`): Microgrid project lifetime in time intervals.
    maintenance_cost_rate (:type:`int | float`): Operations and maintenance cost rate for installed componentes based on installation costs in [decimal].
    discount_rate (:type:`int | float`): The discount rate during the lifetime project in [decimal].
    photovoltaic_panel (:class:`~simulation.photovoltaic_panel.PhotovoltaicPanel` :type:`| None`): A :class:`~simulation.photovoltaic_panel.PhotovoltaicPanel` instance. Default is ``None``.
    wind_turbine (:class:`~simulation.wind_turbine.WindTurbine` :type:`| None`): A :class:`~simulation.wind_turbine.WindTurbine` instance. Default is ``None``.
    battery (:class:`~simulation.battery.Battery` :type:`| None`): A :class:`~simulation.battery.Battery` instance. Default is ``None``.
    public_grid (:class:`~simulation.public_grid.PublicGrid` :type:`| None`): A :class:`~simulation.public_grid.PublicGrid` instance. Default is ``None``.
    inverter (:class:`~simulation.inverter.Inverter` :type:`| None`): A :class:`simulation.inverter.Inverter` instance. Default is ``None``.
    converter (:class:`~simulation.converter.Converter` :type:`| None`): A :class:`simulation.converter.Converter` instance. Default is ``None``.

  Raises:
    TypeError: If the input is not the expected type.
    ValueError: If the input is not the allowed value.
  '''

  def __init__(self,
               load: npt.NDArray[np.floating],
               temperature: npt.NDArray[np.floating],
               solar_irradiance: npt.NDArray[np.floating],
               wind_velocity: npt.NDArray[np.floating],
               wind_height: int | float,
               lifetime: int = 24,
               maintenance_cost_rate: int | float = 0.02,
               discount_rate: int | float = 0.15,
               photovoltaic_panel: PhotovoltaicPanel | None = None,
               wind_turbine: WindTurbine | None = None,
               battery: Battery | None = None,
               public_grid: PublicGrid | None = None,
               inverter: Inverter | None = None,
               converter: Converter | None = None) -> None:
    
    self.load: npt.NDArray[np.floating]
    ''' A numpy array with the demanding load in [kWh]. '''
    self.temperature: npt.NDArray[np.floating]
    ''' A numpy array with the temperature in [ºC]. '''
    self.solar_irradiance: npt.NDArray[np.floating]
    ''' A numpy array with solar irradiance in [kW/m^2]. '''
    self.wind_velocity: npt.NDArray[np.floating]
    ''' A numpy array with the wind velocity in [m/s]. '''
    self.wind_height: int | float
    ''' The height where the wind speed was measured in [m]. '''
    self.lifetime: int
    ''' Microgrid project lifetime in time intervals. '''
    self.maintenance_cost_rate: int | float
    ''' Operations and maintenance cost rate for installed componentes based on installation costs in [decimal]. '''
    self.discount_rate: int | float
    ''' The discount rate during the lifetime project in [decimal]. '''
    self.photovoltaic_panel: PhotovoltaicPanel | None
    ''' A :class:`~simulation.photovoltaic_panel.PhotovoltaicPanel` instance. Default is ``None``. '''
    self.wind_turbine: WindTurbine | None
    ''' A :class:`~simulation.wind_turbine.WindTurbine` instance. Default is ``None``. '''
    self.battery: Battery | None
    ''' A :class:`~simulation.battery.Battery` instance. Default is ``None``. '''
    self.public_grid: PublicGrid | None
    ''' A :class:`~simulation.public_grid.PublicGrid` instance. Default is ``None``.'''
    self.inverter: Inverter | None
    ''' A :class:`simulation.inverter.Inverter` instance. Default is ``None``.'''
    self.converter: Converter | None
    ''' A :class:`simulation.converter.Converter` instance. Default is ``None``.'''
    self.hour_steps: int
    ''' Number of hour steps in the simulation. '''
    self.surplus_energy: npt.NDArray[np.floating]
    ''' Numpy array to store the energy surplus that will be throw away at each time step in [kWh]. '''
    self.energy_generated: npt.NDArray[np.floating]
    ''' Energy generated at each time step in [kWh]. '''
    # Objectives
    self.lcoe: float = 0.0
    ''' Levelized Cost of Energy in [$/kWh]. '''
    self.renewable_factor: float = 0.0
    ''' Renewable Factor between 0 and 1. '''
    self.meef: float = 0.0
    ''' Microgrid Energy Excess Factor between 0 and 1. '''

    self.load = load
    self.temperature = temperature
    self.solar_irradiance = solar_irradiance
    self.wind_velocity = wind_velocity
    self.wind_height = wind_height
    self.lifetime = lifetime
    self.maintenance_cost_rate = maintenance_cost_rate
    self.discount_rate = discount_rate
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
    
    # Initialization of microgrid attributes
    self.energy_generated = np.zeros(self.hour_steps)
    # Initialization of microgrid components
    if self.photovoltaic_panel:
      self.photovoltaic_panel.initialize(self.hour_steps)
    if self.wind_turbine:
      self.wind_turbine.initialize(self.hour_steps)
    if self.battery:
      self.battery.initialize(self.hour_steps)
    if self.public_grid:
      self.public_grid.initialize(self.hour_steps)

  def generate_energy(self) -> None:
    ''' Generates energy by generators. '''

    # Get the generated energy by photovoltaic panels
    if self.photovoltaic_panel:
      self.photovoltaic_panel.generate_power(self.temperature, self.solar_irradiance)
      self.energy_generated += self.photovoltaic_panel.output_power
    # Get the generated energy by wind_turbines
    if self.wind_turbine:
      self.wind_turbine.generate_power(self.wind_velocity, self.wind_height)
      self.energy_generated += self.wind_turbine.output_power

  def dispatch_energy_by_generators(self, energy_demanded_adjusted: npt.NDArray[np.floating], inverter_efficiency: int | float) -> None:
    ''' Calculates the energy dispatched by generators that effectively met demand.
      
      Args:
        energy_demanded_adjusted (:type:`npt.NDArray[np.floating]`): The energy demanded adjusted by the microgrid inverter in [kWh].
        inverter_efficiency (:type:`int | float`): The efficiency of the inverter between 0 and 1.
    '''

    if self.photovoltaic_panel and self.wind_turbine:
      # Calculate the energy from generators that will supply the demand equaly
      self.photovoltaic_panel.meet_demand[:] = np.minimum(self.photovoltaic_panel.output_power, energy_demanded_adjusted - np.minimum(self.wind_turbine.output_power, energy_demanded_adjusted / 2))
      self.wind_turbine.meet_demand[:] = np.minimum(self.wind_turbine.output_power, energy_demanded_adjusted - self.photovoltaic_panel.meet_demand)
      # Calculate the energy that effectively was sent to demand
      self.photovoltaic_panel.meet_demand *= inverter_efficiency
      self.wind_turbine.meet_demand *= inverter_efficiency
    elif self.photovoltaic_panel:
      self.photovoltaic_panel.meet_demand[:] = np.minimum(self.photovoltaic_panel.output_power, energy_demanded_adjusted)
    elif self.wind_turbine:
      self.wind_turbine.meet_demand[:] = np.minimum(self.wind_turbine.output_power, energy_demanded_adjusted)


  def dispatch_energy(self) -> None:
    ''' Runs the hourly simulation of the microgrid. '''

    # Defining inner functions to handle with None components
    def _no_battery_charge(surplus_energy: int | float, converter_efficiency: int | float, t: int) -> int | float:
      return surplus_energy
    def _no_battery_discharge(energy_demanded_adjusted: int | float, inverter_efficiency: int | float, t: int) -> int | float:
      return energy_demanded_adjusted
    def _no_public_grid_export(surplus_energy: int | float, inverter_efficiency: int | float, t: int) -> float:
      return surplus_energy
    def _no_public_grid_import(energy_demanded: int | float, t: int) -> None:
      return None
    # --------------------------------------------------------

    # Get the functions to charge and discharge the battery
    converter_efficiency = 1.0
    if self.battery:
      charge_battery = self.battery.charge
      discharge_battery = self.battery.discharge
      # Check if the battery has a converter to charge energy and get its efficiency
      if self.converter:
        converter_efficiency = self.converter.efficiency
    else:
      charge_battery = _no_battery_charge
      discharge_battery = _no_battery_discharge
    # Check if the microgrid inverter is connected and get its efficiency
    if self.inverter:
      inverter_efficiency = self.inverter.efficiency
    else:
      inverter_efficiency = 1.0
    # Get the functions to export and import energy from the public grid
    if self.public_grid and self.public_grid.credit_rate > 0:
      export_energy = self.public_grid.export_energy
      import_energy = self.public_grid.import_energy
    elif self.public_grid:
      export_energy = _no_public_grid_export
      import_energy = self.public_grid.import_energy
    else:
      export_energy = _no_public_grid_export
      import_energy = _no_public_grid_import
    # Adjust load demanded by inverter efficiency
    energy_demanded_adjusted = self.load / inverter_efficiency
    # Calculate the energy dispatched by generators that met demand
    self.dispatch_energy_by_generators(energy_demanded_adjusted, inverter_efficiency)
    # Calculate the time steps in which there is energy surplus
    surplus_mask = np.where(self.energy_generated > energy_demanded_adjusted, True, False)
    # Calculate the difference between generated energy and load adjusted
    energy_flow_adjusted = np.abs(self.energy_generated - energy_demanded_adjusted)
    for t, there_is_surplus in enumerate(surplus_mask):
      # If there is surplus energy
      if there_is_surplus:
        remaining_surplus_energy = energy_flow_adjusted[t]
        # Charge the battery with the surplus energy (if the battery is connected)
        remaining_surplus_energy_after_charging = charge_battery(remaining_surplus_energy, converter_efficiency, t) / converter_efficiency
        # Take into account the surplus energy that could not be stored in the battery
        self.meef += remaining_surplus_energy_after_charging
        # Send the surplus energy to the public grid (if the public grid is connected)
        self.surplus_energy[t] = export_energy(remaining_surplus_energy_after_charging, inverter_efficiency, t) / inverter_efficiency
      # If there is deficit energy
      else:
        remaining_deficit_energy_adjusted = energy_flow_adjusted[t]
        # Discharge the battery to cover the deficit adjusted (if the battery is connected)
        remaining_deficit_energy_after_discharging = discharge_battery(remaining_deficit_energy_adjusted, inverter_efficiency, t) * inverter_efficiency
        # If there is still deficit, pruchase energy from the public grid (if the public grid is connected)
        import_energy(remaining_deficit_energy_after_discharging, t)
    # Disconsider the first time step for the battery state of charge
    if self.battery:
      self.battery.state_of_charge = self.battery.state_of_charge[1:]

  def economic_analysis(self, sum_of_loads: int | float) -> None:
    ''' Performs the economic analysis of the microgrid and its components.
    
    Args:
      sum_of_loads (:type:`int | float`): The total load demand over the simulation period in [kWh].
    '''

    # Calculate the Capital Recovery Factor (CRF)
    if self.discount_rate > 0:
      CRF = (self.discount_rate * (1 + self.discount_rate) ** self.lifetime) / ((1 + self.discount_rate) ** self.lifetime - 1)
    else:
      CRF = 1 / self.lifetime
    # Discretization of project lifetime intervals
    project_lifetime_intervals = np.arange(self.lifetime + 1)
    # Get the rated power of the Distributed Energy Resources combined
    der_rated_power = 0.0
    # Perform economic analysis for photovoltaic panels
    if self.photovoltaic_panel:
      self.lcoe += self.photovoltaic_panel.economic_analysis(project_lifetime_intervals, self.maintenance_cost_rate, self.discount_rate, CRF)
      der_rated_power += self.photovoltaic_panel.rated_power
    # Perform economic analysis for wind turbines
    if self.wind_turbine:
      self.lcoe += self.wind_turbine.economic_analysis(project_lifetime_intervals, self.maintenance_cost_rate, self.discount_rate, CRF)
      der_rated_power += self.wind_turbine.rated_power
    # Perform economic analysis for battery
    if self.battery:
      self.lcoe += self.battery.economic_analysis(project_lifetime_intervals, self.maintenance_cost_rate, self.discount_rate, CRF)
    # Perform economic analysis for public grid
    if self.public_grid:
      self.lcoe += self.public_grid.economic_analysis(self.lifetime, self.discount_rate)
    # Perform economic analysis for inverter
    if self.inverter:
      self.lcoe += self.inverter.economic_analysis(der_rated_power * 1.2, project_lifetime_intervals, self.maintenance_cost_rate, self.discount_rate, CRF)
    # Perform economic analysis for converter
    if self.converter:
      self.lcoe += self.converter.economic_analysis(der_rated_power * 1.2, project_lifetime_intervals, self.maintenance_cost_rate, self.discount_rate, CRF)
    # Calculate the Levelized Cost of Energy (lcoe)
    self.lcoe *= CRF / sum_of_loads

  def calculate_renewable_factor(self, sum_of_loads: float) -> None:
    r''' Calculates the Renewable Factor (RF) according to the following equation:

    .. math::
      RF = \frac{\sum^H_{h=1} E^{meet}_{pv}(h) + E^{meet}_{wt}(h) + E^{meet}_{bat}(h)}{\sum^H_{h=1} E_{load}(h)},

    where:

    - :math:`E^{meet}_{pv}(h)` is the energy effectively supplied to the load by the photovoltaic generator at hour :math:`h` [kWh];
    - :math:`E^{meet}_{wt}(h)` is the energy effectively supplied to the load by the wind turbine at hour :math:`h` [kWh];
    - :math:`E^{meet}_{bat}(h)` is the energy discharged from the battery to the load at hour :math:`h` [kWh];
    - :math:`E_{load}(h)` is the energy demanded by the load at hour :math:`h` [kWh].

    The Renewable Factor represents the fraction of the total demand met by renewable sources and battery storage over the simulation period.

    Args:
      sum_of_loads (:type:`float`): The total load demand over the simulation period in [kWh].
    '''

    if self.photovoltaic_panel:
      pv_meet = self.photovoltaic_panel.meet_demand
    else:
      pv_meet = 0
    if self.wind_turbine:
      wt_meet = self.wind_turbine.meet_demand
    else:
      wt_meet = 0
    if self.battery:
      bat_meet = self.battery.meet_demand
    else:
      bat_meet = 0
    self.renewable_factor = np.sum(pv_meet + wt_meet + bat_meet) / sum_of_loads

  def run(self) -> npt.NDArray[np.floating]:
    ''' Runs the Microgrid simulation. '''

    # Initialize the microgrid components
    self.initialize()
    # Generates energy by generators
    self.generate_energy()
    # Simulates the energy dispatch
    self.dispatch_energy()

    # Calculate the objectives
    sum_of_loads = float(np.sum(self.load))
    # Performs economic analysis
    self.economic_analysis(sum_of_loads)
    # Calculate the Renewable Factor
    self.calculate_renewable_factor(sum_of_loads)
    # Calculate the Microgrid Energy Excess Factor (meef)
    self.meef /= float(np.sum(self.energy_generated))
    # Return the Levelized Cost of Energy (lcoe) in $/kWh, Renewable Factor and Microgrid Energy Excess Factor (meef)
    return np.array([self.lcoe, self.renewable_factor, self.meef])

  def logging(self, file_name: str) -> None:
    ''' Logs the microgrid information into a excel file.
    
    Args:
      file_name (:type:`str`): The name of the excel file that the information will be recorded.
    '''

    df = pd.DataFrame({
      'Load [kWh]': self.load,
      'Photovoltaic Panel Generation [kWh]': self.photovoltaic_panel.output_power if self.photovoltaic_panel else np.zeros(self.hour_steps),
      'Wind Turbine Generation [kWh]': self.wind_turbine.output_power if self.wind_turbine else np.zeros(self.hour_steps),
      'Photovoltaic Panel Supply [kWh]': self.photovoltaic_panel.meet_demand if self.photovoltaic_panel else np.zeros(self.hour_steps),
      'Wind Turbine Supply [kWh]': self.wind_turbine.meet_demand if self.wind_turbine else np.zeros(self.hour_steps),
      'Battery State of Charge [kWh]': self.battery.state_of_charge if self.battery else np.zeros(self.hour_steps),
      'Battery Charge [kWh]': self.battery.energy_charged if self.battery else np.zeros(self.hour_steps),
      'Battery Discharge [kWh]': self.battery.energy_discharged if self.battery else np.zeros(self.hour_steps),
      'Battery Supply [kWh]': self.battery.meet_demand if self.battery else np.zeros(self.hour_steps),
      'Energy Purchased [kWh]': self.public_grid.energy_purchased if self.public_grid else np.zeros(self.hour_steps),
      'Energy Credited [kWh]': self.public_grid.energy_credited if self.public_grid else np.zeros(self.hour_steps),
      'Energy Compensated [kWh]': self.public_grid.energy_compensated if self.public_grid else np.zeros(self.hour_steps),
      'Energy Surplus [kWh]': self.surplus_energy
    })

    df.to_excel(file_name + '.xlsx', sheet_name='Microgrid Power Flow', index=False)