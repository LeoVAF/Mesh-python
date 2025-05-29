from microgrid.microgrid import Microgrid
from microgrid.photovoltaic_panel import PhotovoltaicPanel
from microgrid.wind_turbine import WindTurbine
from microgrid.inverter import Inverter
from microgrid.battery import Battery
from microgrid.public_grid import PublicGrid
from microgrid_old.techno_ka import techno_ka

import numpy as np


# Economic input
exchange_rate = 1.14

# Photovoltaic panel input
max_pv = 30
pv_lifetime = 24
pv_efficiency = 0.95
pv_cost = 1500 * exchange_rate

# Wind turbine input
num_wt = 2
wt_rated_power = 5
cut_in = 2.5
cut_out = 40
wt_efficiency = 0.95
wt_lifetime = 24

# Inverter input
inverter_efficiency = 0.95
inverter_lifetime = 24

# Battery input
bat_dod = 0.8
bat_cap = 150
select_bat = 0

# Public grid input
metering_compensation = 0

# Microgrid input
load_ind = np.genfromtxt('scripts/microgrid_old/seasonal_data/loadind.txt')
temperature = np.repeat(np.array([12, 13, 15, 16, 19, 22, 24, 24, 23, 20, 16, 13]), 720)
solar_data = np.genfromtxt('scripts/microgrid_old/seasonal_data/solreal.txt')
wind_data = np.genfromtxt('scripts/microgrid_old/seasonal_data/wind_data.txt')
wind_height = 10
microgrid_lifetime = 24
photovoltaic_panel = PhotovoltaicPanel(cost_per_kw=pv_cost, rated_power=max_pv, lifetime=pv_lifetime)
wind_turbine = WindTurbine(n_turbines=num_wt, rated_power=wt_rated_power, cut_in=cut_in, cut_out=cut_out, efficiency=wt_efficiency, lifetime=wt_lifetime)
inverter = Inverter(efficiency=inverter_efficiency, lifetime=inverter_lifetime)
# battery = Battery()
# public_grid = PublicGrid(metering_compensation=metering_compensation)

microgrid = Microgrid(load=load_ind[:8640],
                      temperature=temperature[:8640],
                      solar_radiation=solar_data[:8640],
                      wind_velocity=wind_data[:8640],
                      wind_height=wind_height,
                      lifetime=microgrid_lifetime,
                      photovoltaic_panel=photovoltaic_panel,
                      wind_turbine=wind_turbine,
                      inverter=inverter,
                      battery=None,
                      public_grid=None)

# Run microgrid
techno_ka(max_pv, num_wt, bat_dod, bat_cap, select_bat, solar_data, wind_data, load_ind)
microgrid.run()