import numpy as np

from simulation.battery import Battery
from simulation.converter import Converter
from simulation.inverter import Inverter
from simulation.microgrid import Microgrid
from simulation.photovoltaic_panel import PhotovoltaicPanel
from simulation.utility_grid import UtilityGrid
from simulation.wind_turbine import WindTurbine


def simulation(pv_rated_power: float,
              wt_rated_power: float,
              bat_capacity: float,
              select_bat: int,
              load: np.typing.NDArray[np.floating],
              temperature: np.typing.NDArray[np.floating],
              solar_data: np.typing.NDArray[np.floating],
              wind_data: np.typing.NDArray[np.floating]) -> Microgrid:
  # Photovoltaic panel input
  pv_cost_per_kwp = 654
  pv_lifetime = 20
  pv_resale_rate = 0.75

  # Wind turbine input
  wt_cost_per_kw = 1079
  cut_in = 2
  wt_rated_wind_speed = 9
  cut_out = 40
  wt_lifetime = 20
  wt_height = 50
  wt_resale_rate = 0.75

  # Battery input: # Lead_Acid(0) Li-ion(1) ZEBRA(2) NaS(3) NiCd(4) NiMH(5) RFV(6) ZnBr(7)
  bat_dod = 0.8
  bat_efficiency_list = [0.8, 0.95, 0.8, 0.85, 0.75, 0.65, 0.75, 0.7]
  # Each battery capacity cost in [US$/kWh]
  bat_cap_cost_list = [130, 1560, 250, 400, 1200, 500, 600, 500]
  # Each battery lifetime in [years]
  bat_lf_list = [10 ,10 ,12, 15, 15, 10, 15, 10]
  # Each battery cycle number
  bat_cycle_list = [1125, 5000, 3000, 3000, 1000, 1050, 12000, 1750]
  bat_resale_rate = 0.75

  # Utility grid input
  grid_cost_per_kwh = 0.12
  grid_tariff_growth = 0.07
  grid_credit_rate = 0.8
  grid_compensation_period_hours = 730

  # Inverter input
  inverter_reference_cost = 180
  inverter_cost_scale = 0.95
  inverter_efficiency = 0.95
  inverter_lifetime = 20
  inverter_resale_rate = 0.75

  # Converter input
  converter_reference_cost = 330
  converter_cost_scale = 0.95
  converter_efficiency = 0.95
  converter_lifetime = 15
  converter_resale_rate = 0.75

  # Microgrid input
  wind_height = 10
  microgrid_lifetime = 24
  microgrid_maintenance_cost_rate = 0.02
  microgrid_discount_rate = 0.1
  microgrid_load_growth_rate = 0.02

  photovoltaic_panel = PhotovoltaicPanel(cost_per_kwp=pv_cost_per_kwp,
                                        rated_power=pv_rated_power,
                                        lifetime=pv_lifetime,
                                        resale_rate=pv_resale_rate)
  
  wind_turbine = WindTurbine(cost_per_kw=wt_cost_per_kw,
                            rated_power=wt_rated_power,
                            rated_wind_speed=wt_rated_wind_speed,
                            cut_in=cut_in,
                            cut_out=cut_out,
                            height=wt_height,
                            lifetime=wt_lifetime,
                            resale_rate=wt_resale_rate)
  
  battery = Battery(capacity=bat_capacity,
                    cost_per_kwh=bat_cap_cost_list[select_bat],
                    efficiency=bat_efficiency_list[select_bat],
                    lifetime=bat_lf_list[select_bat],
                    number_of_cycles=bat_cycle_list[select_bat],
                    depth_of_discharge=bat_dod,
                    resale_rate=bat_resale_rate)
  
  utility_grid = UtilityGrid(cost_per_kwh=grid_cost_per_kwh,
                          tariff_growth=grid_tariff_growth,
                          credit_rate=grid_credit_rate,
                          compensation_period_hours=grid_compensation_period_hours)
  
  inverter = Inverter(reference_cost=inverter_reference_cost,
                      cost_exponent=inverter_cost_scale,
                      efficiency=inverter_efficiency,
                      lifetime=inverter_lifetime,
                      resale_rate=inverter_resale_rate)
  
  converter = Converter(reference_cost=converter_reference_cost,
                        cost_exponent=converter_cost_scale,
                        efficiency=converter_efficiency,
                        lifetime=converter_lifetime,
                        resale_rate=converter_resale_rate)

  microgrid = Microgrid(load=load,
                        temperature=temperature,
                        solar_irradiance=solar_data,
                        wind_velocity=wind_data,
                        wind_height=wind_height,
                        lifetime=microgrid_lifetime,
                        maintenance_cost_rate=microgrid_maintenance_cost_rate,
                        discount_rate=microgrid_discount_rate,
                        load_growth_rate=microgrid_load_growth_rate,
                        photovoltaic_panel=photovoltaic_panel,
                        wind_turbine=wind_turbine,
                        battery=battery,
                        utility_grid=utility_grid,
                        inverter=inverter,
                        converter=converter)
  
  return microgrid

def microgrid_function(pv_rated_power: float,
                       wt_rated_power: float,
                       bat_capacity: float,
                       select_bat: int,
                       load: np.typing.NDArray[np.floating],
                       temperature: np.typing.NDArray[np.floating],
                       solar_data: np.typing.NDArray[np.floating],
                       wind_data: np.typing.NDArray[np.floating]) -> np.typing.NDArray[np.floating]:
  
  # Simulate the microgrid with the given parameters
  microgrid = simulation(pv_rated_power=pv_rated_power,
                          wt_rated_power=wt_rated_power,
                          bat_capacity=bat_capacity,
                          select_bat=select_bat,
                          load=load,
                          temperature=temperature,
                          solar_data=solar_data,
                          wind_data=wind_data)
  
  # Run microgrid
  objectives = microgrid.run()
  # Maximizing Renewable Factor (RF)
  objectives[1] = -objectives[1]
  # Maximizing Renewable Self-Consumption Ratio (RSC)
  objectives[2] = -objectives[2]
  return objectives
