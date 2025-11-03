from microgrid.microgrid import Microgrid
from microgrid.photovoltaic_panel import PhotovoltaicPanel
from microgrid.wind_turbine import WindTurbine
from microgrid.battery import Battery
from microgrid.public_grid import PublicGrid
from microgrid.inverter import Inverter
from microgrid.converter import Converter

import numpy as np

def microgrid_function(pv_rated_power: int | float,
                       wt_rated_power: int | float,
                       bat_capacity: int | float,
                       select_bat: int,
                       load: np.ndarray[np.number],
                       temperature: np.ndarray[np.number],
                       solar_data: np.ndarray[np.number],
                       wind_data: np.ndarray[np.number]) -> np.ndarray[np.float64]:
  # Photovoltaic panel input
  pv_cost_per_kwp = 654
  pv_lifetime = 20

  # Wind turbine input
  wt_cost_per_kw = 1079
  cut_in = 2
  wt_rated_wind_speed = 9
  cut_out = 40
  wt_lifetime = 20
  wt_height = 50

  # Battery input
  bat_dod = 0.8
  bat_efficiency_list = [0.8, 0.95, 0.8, 0.85, 0.75, 0.65, 0.75, 0.7]
  # Each battery capacity cost in [US$/kWh]
  bat_cap_cost_list = [130, 1560, 250, 400, 1200, 500, 600, 500]
  # Each battery lifetime in [years]
  bat_lf_list = [10 ,10 ,12, 15, 15, 10, 15, 10]
  # Each battery cycle number
  bat_cycle_list = [1125, 5000, 3000, 3000, 1000, 1050, 12000, 1750]
  ''' ###### '''
  # bat_efficiency_list = [0.765,0.90,0.92,0.96,0.94,0.938,0.9155,0.95,0.86,0.855,0.70,0.80,0.75,0.70]
  # Each battery capacity cost in [€$]
  # bat_cap_cost_list = [7.31393,28.575,28.575,9.8066225,20.67004375,7.540625,10.1099133,6.746875,38.1,6.99371645,9.55675,31.40075,45.10395475,14.5415,12.4139325]
  # Each battery lifetime in [years]
  # bat_lf_list = [18,17.5,7,15,10,10,10,20,14,13.5,20,3,15,6.5]
  # Each battery cycle number
  # bat_cycle_list = [1400,8000,600,5000,1500,4000,3000,1000,3000,3250,1250,1000,10000,2000]
  ''' ###### '''

  # Public grid input
  grid_cost_per_kwh = 0.12
  grid_tariff_growth = 0.07
  grid_credit_rate = 0.8

  # Inverter input
  inverter_cost_per_kw = 180
  inverter_cost_scale = 0.95
  inverter_efficiency = 0.95
  inverter_lifetime = 20

  # Converter input
  converter_cost_per_kw = 330
  converter_cost_scale = 0.95
  converter_efficiency = 0.95
  converter_lifetime = 15

  # Microgrid input
  wind_height = 10
  microgrid_lifetime = 24
  microgrid_maintenance_cost_rate = 0.02
  microgrid_discount_rate = 0.1
  photovoltaic_panel = PhotovoltaicPanel(cost_per_kwp=pv_cost_per_kwp,
                                        rated_power=pv_rated_power,
                                        lifetime=pv_lifetime)
  wind_turbine = WindTurbine(cost_per_kw=wt_cost_per_kw,
                            rated_power=wt_rated_power,
                            rated_wind_speed=wt_rated_wind_speed,
                            cut_in=cut_in,
                            cut_out=cut_out,
                            height=wt_height,
                            lifetime=wt_lifetime)
  battery = Battery(capacity=bat_capacity,
                    cost_per_kwh=bat_cap_cost_list[select_bat],
                    efficiency=bat_efficiency_list[select_bat],
                    lifetime=bat_lf_list[select_bat],
                    number_of_cycles=bat_cycle_list[select_bat],
                    depth_of_discharge=bat_dod)
  public_grid = PublicGrid(cost_per_kwh=grid_cost_per_kwh,
                          tariff_growth=grid_tariff_growth,
                          credit_rate=grid_credit_rate)
  inverter = Inverter(cost_per_kw=inverter_cost_per_kw,
                      cost_scale=inverter_cost_scale,
                      efficiency=inverter_efficiency,
                      lifetime=inverter_lifetime)
  converter = Converter(cost_per_kw=converter_cost_per_kw,
                        cost_scale=converter_cost_scale,
                        efficiency=converter_efficiency,
                        lifetime=converter_lifetime)

  microgrid = Microgrid(load=load,
                        temperature=temperature,
                        solar_radiation=solar_data,
                        wind_velocity=wind_data,
                        wind_height=wind_height,
                        lifetime=microgrid_lifetime,
                        maintenance_cost_rate=microgrid_maintenance_cost_rate,
                        discount_rate=microgrid_discount_rate,
                        photovoltaic_panel=photovoltaic_panel,
                        wind_turbine=wind_turbine,
                        battery=battery,
                        public_grid=public_grid,
                        inverter=inverter,
                        converter=converter)
  
  # Run microgrid
  objectives = microgrid.run()
  # Maximizing renewable factor
  objectives[1] = -objectives[1]
  return objectives 