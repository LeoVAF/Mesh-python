from microgrid.photovoltaic_panel import PhotovoltaicPanel
from microgrid.wind_turbine import WindTurbine
from microgrid.battery import Battery
from microgrid.public_grid import PublicGrid
from microgrid.inverter import Inverter
from microgrid.converter import Converter

import numpy as np
import pandas as pd

class Microgrid:
  ''' Microgrid simulation.

  Args:
    load (:type:`np.ndarray[np.float64]`): A numpy array with the demanding load in [kWh].
    temperature (:type:`np.ndarray[np.float64]`): A numpy array with the temperature in [ºC].
    solar_radiation (:type:`np.ndarray[np.float64]`): A numpy array with solar radiation in [kWh/m^2].
    wind_velocity (:type:`np.ndarray[np.float64]`): A numpy array with the wind velocity in [m/s].
    wind_height (:type:`int | float`): The height where the wind speed was measured in [m].
    lifetime (:type:`int | float`): Microgrid lifetime project in [year].
    maintenance_cost_rate (:type:`int | float`): Operations and maintenance cost rate for installed componentes based on installation costs in [decimal].
    discount_rate (:type:`int | float`): The discount rate during the lifetime project in [decimal].
    photovoltaic_panel (:class:`~microgrid.photovoltaic_panel.PhotovoltaicPanel` :type:`| None`): A :class:`~microgrid.photovoltaic_panel.PhotovoltaicPanel` instance. Default is ``None``.
    wind_turbine (:class:`~microgrid.wind_turbine.WindTurbine` :type:`| None`): A :class:`~microgrid.wind_turbine.WindTurbine` instance. Default is ``None``.
    battery (:class:`~microgrid.battery.Battery` :type:`| None`): A :class:`~microgrid.battery.Battery` instance. Default is ``None``.
    public_grid (:class:`~microgrid.public_grid.PublicGrid` :type:`| None`): A :class:`~microgrid.public_grid.PublicGrid` instance. Default is ``None``.
    inverter (:class:`~microgrid.inverter.Inverter` :type:`| None`): A :class:`microgrid.inverter.Inverter` instance. Default is ``None``.
    converter (:class:`~microgrid.converter.Converter` :type:`| None`): A :class:`microgrid.converter.Converter` instance. Default is ``None``.

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
               maintenance_cost_rate: int | float = 0.02,
               discount_rate: int | float = 0.15,
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
    self.lifetime: int | float
    ''' Microgrid project lifetime in [year]. '''
    self.maintenance_cost_rate: int | float
    ''' Operations and maintenance cost rate for installed componentes based on installation costs in [decimal]. '''
    self.discount_rate: int | float
    ''' The discount rate during the lifetime project in [decimal]. '''
    self.photovoltaic_panel: PhotovoltaicPanel | None
    ''' A :class:`~microgrid.photovoltaic_panel.PhotovoltaicPanel` instance. Default is ``None``. '''
    self.wind_turbine: WindTurbine | None
    ''' A :class:`~microgrid.wind_turbine.WindTurbine` instance. Default is ``None``. '''
    self.battery: Battery | None
    ''' A :class:`~microgrid.battery.Battery` instance. Default is ``None``. '''
    self.public_grid: PublicGrid | None
    ''' A :class:`~microgrid.public_grid.PublicGrid` instance. Default is ``None``.'''
    self.inverter: Inverter | None
    ''' A :class:`microgrid.inverter.Inverter` instance. Default is ``None``.'''
    self.converter: Converter | None
    ''' A :class:`microgrid.converter.Converter` instance. Default is ``None``.'''
    self.hour_steps: int
    ''' Number of hour steps in the simulation. '''
    self.surplus_energy: np.ndarray[np.float64]
    ''' Numpy array to store the energy surplus that will be throw away at each time step in [kWh]. '''
    self.energy_generated: np.ndarray[np.float64] | None = None
    ''' Energy generated at each time step in [kWh]. Default is ``None``. '''
    # Objectives
    self.planning_cost: float = 0.0
    ''' Cost of electricity in [US$/kWh]. '''
    self.renewable_factor: float = 0.0
    ''' Renewable factor between 0 and 1. '''

    self.load = load
    self.temperature = temperature
    self.solar_radiation = solar_radiation
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
      self.photovoltaic_panel.generate_power(self.temperature, self.solar_radiation)
    # Get the generated energy by wind_turbines
    if self.wind_turbine:
      self.wind_turbine.generate_power(self.wind_velocity, self.wind_height)
    self.energy_generated = self.photovoltaic_panel.output_power + self.wind_turbine.output_power

  def dispatch_energy_by_generators(self, energy_demanded_adjusted: np.ndarray[np.float64], inverter_efficiency: int | float) -> None:
    ''' Calculates the energy dispatched by generators that effectively met demand.
      
      Args:
        energy_demanded_adjusted (:type:`np.ndarray[np.float64]`): The energy demanded adjusted by the microgrid inverter in [kWh].
        inverter_efficiency (:type:`int | float`): The efficiency of the inverter between 0 and 1.
    '''

    # Calculate the energy from generators that will supply the demand equaly
    self.photovoltaic_panel.meet_demand[:] = np.minimum(self.photovoltaic_panel.output_power, energy_demanded_adjusted - np.minimum(self.wind_turbine.output_power, energy_demanded_adjusted / 2))
    self.wind_turbine.meet_demand[:] = np.minimum(self.wind_turbine.output_power, energy_demanded_adjusted - self.photovoltaic_panel.meet_demand)
    # Calculate the energy that effectively was sent to demand
    self.photovoltaic_panel.meet_demand *= inverter_efficiency
    self.wind_turbine.meet_demand *= inverter_efficiency

  def dispatch_energy(self) -> None:
    ''' Runs the hourly simulation of the microgrid. '''

    # Get the functions to charge and discharge the battery
    if self.battery:
      charge_battery = self.battery.charge
      discharge_battery = self.battery.discharge
      # Check if the battery has a converter to charge energy and get its efficiency
      if self.converter:
        converter_efficiency = self.converter.efficiency
      else:
        converter_efficiency = 1.0
    else:
      charge_battery = lambda x, y: x
      discharge_battery = lambda x, y, z: x
    # Check if the microgrid inverter is connected and get its efficiency
    if self.inverter:
      inverter_efficiency = self.inverter.efficiency
    else:
      inverter_efficiency = 1.0
    # Get the functions to compensate and buy energy from the public grid
    if self.public_grid and self.public_grid.credit_rate > 0:
      compensate = self.public_grid.store_energy_credit
      buy = self.public_grid.purchase_energy
    elif self.public_grid:
      compensate = lambda x: x
      buy = self.public_grid.purchase_energy
    else:
      compensate = lambda x: x
      buy = lambda x, t: None
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
        # Send the surplus energy to the public grid (if the public grid is connected)
        self.surplus_energy[t] = compensate(remaining_surplus_energy_after_charging * inverter_efficiency) / inverter_efficiency
      # If there is deficit energy
      else:
        remaining_deficit_energy_adjusted = energy_flow_adjusted[t]
        # Discharge the battery to cover the deficit adjusted (if the battery is connected)
        remaining_deficit_energy_after_discharging = discharge_battery(remaining_deficit_energy_adjusted, inverter_efficiency, t) * inverter_efficiency
        # If there is still deficit, buy energy from the public grid (if the public grid is connected)
        buy(remaining_deficit_energy_after_discharging, t)
    # Disconsider the first time step for the battery state of charge
    self.battery.state_of_charge = self.battery.state_of_charge[1:]

  def economic_analysis(self) -> None:
    ''' Performs the economic analysis of the microgrid and its components. '''

    # Get the rated power of the Distributed Energy Resources combined
    der_rated_power = 0.0
    # Perform economic analysis for photovoltaic panels
    if self.photovoltaic_panel:
      self.planning_cost += self.photovoltaic_panel.economic_analysis(self.lifetime, self.maintenance_cost_rate, self.discount_rate)
      der_rated_power += self.photovoltaic_panel.rated_power
    # Perform economic analysis for wind turbines
    if self.wind_turbine:
      self.planning_cost += self.wind_turbine.economic_analysis(self.lifetime, self.maintenance_cost_rate, self.discount_rate)
      der_rated_power += self.wind_turbine.rated_power
    # Perform economic analysis for battery
    if self.battery:
      self.planning_cost += self.battery.economic_analysis(self.lifetime, self.maintenance_cost_rate, self.discount_rate)
    # Perform economic analysis for public grid
    if self.public_grid:
      self.planning_cost += self.public_grid.economic_analysis(self.lifetime, self.discount_rate)
    # Perform economic analysis for inverter
    if self.inverter:
      self.planning_cost += self.inverter.economic_analysis(der_rated_power, self.lifetime, self.maintenance_cost_rate, self.discount_rate)
    # Perform economic analysis for converter
    if self.converter:
      self.planning_cost += self.converter.economic_analysis(der_rated_power, self.lifetime, self.maintenance_cost_rate, self.discount_rate)

  def calculate_renewable_factor(self, sum_of_loads: int | float) -> None:
    r''' Calculates the Renewable Factor (RF) according to the following equation:

    .. math::
      RF = \frac{\sum^H_{h=1} E_{pv}(h) + E_{wt}(h) + E_{dch}(h)}{\sum^H_{h=1} E_l(h)},

    where:

    - :math:`E_{pv}(h)` is the energy effectively supplied to the load by the photovoltaic generator at hour :math:`h` [kWh];
    - :math:`E_{wt}(h)` is the energy effectively supplied to the load by the wind turbine at hour :math:`h` [kWh];
    - :math:`E_{dch}(h)` is the energy discharged from the battery to the load at hour :math:`h` [kWh];
    - :math:`E_l(h)` is the energy demanded by the load at hour :math:`h` [kWh].

    The Renewable Factor represents the fraction of the total demand met by renewable sources and battery storage over the simulation period.

    Args:
      sum_of_loads (:type:`int | float`): The total load demand over the simulation period in [kWh].
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

  def run(self) -> np.ndarray[np.float64]:
    ''' Runs the Microgrid simulation. '''

    # Initialize the microgrid components
    self.initialize()
    # Generates energy by generators
    self.generate_energy()
    # Simulates the energy dispatch
    self.dispatch_energy()
    # Performs economic analysis
    self.economic_analysis()
    # Calculate the renewable factor
    sum_of_loads = np.sum(self.load)
    self.calculate_renewable_factor(sum_of_loads)
    return np.array([self.planning_cost / sum_of_loads, self.renewable_factor])

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
      'Energy Compensated [kWh]': self.public_grid.energy_compensated if self.public_grid else np.zeros(self.hour_steps),
      'Energy Purchased [kWh]': self.public_grid.energy_purchased if self.public_grid else np.zeros(self.hour_steps),
      'Energy Surplus [kWh]': self.surplus_energy
    })

    df.to_excel(file_name + '.xlsx', sheet_name='Microgrid Power Flow', index=False)