from microgrid.microgrid import Microgrid
from microgrid.photovoltaic_panel import PhotovoltaicPanel
from microgrid_old.techno_ka import techno_ka

import numpy as np

# Input
max_pv = 30
num_wt = 2
bat_dod = 0.8
bat_cap = 150
select_bat = 0
solar_data = np.genfromtxt('scripts/microgrid_old/seasonal_data/solreal.txt')
wind_data = np.genfromtxt('scripts/microgrid_old/seasonal_data/wind_data.txt')
load_ind = np.genfromtxt('scripts/microgrid_old/seasonal_data/loadind.txt')

# Economic input
exchange_rate = 1.14
pv_cost = 1500 * exchange_rate

temperature = np.repeat(np.array([12, 13, 15, 16, 19, 22, 24, 24, 23, 20, 16, 13]), 720)
photovoltaic_panel = PhotovoltaicPanel(cost_per_kw=pv_cost, rated_power=max_pv)


microgrid = Microgrid(load=load_ind[:8640],
                      temperature=temperature[:8640],
                      solar_radiation=solar_data[:8640],
                      wind_velocity=wind_data[:8640],
                      photovoltaic_panel=photovoltaic_panel,
                      wind_turbine=None,
                      battery=None,
                      public_grid=None)

# Run microgrid
techno_ka(max_pv, num_wt, bat_dod, bat_cap, select_bat, solar_data, wind_data, load_ind)
microgrid.run()