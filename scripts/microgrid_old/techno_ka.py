from .CHARGE import charge
from .DISCHARGE import discharge
from .EconomicFast import economic_fast

from statistics import mean
import numpy as np

def techno_ka(max_pv, num_wt, bat_DoD, bat_cap, select_bat, solar_data, wind_data, load):

    max_pv = round(max_pv) # Maximum PV generation [kWh]
    num_wt = round(num_wt) # Number of wind turbines
    bat_DoD = bat_DoD # Depth of discharge [%]
    bat_cap = round(bat_cap) # Battery capacity [kWh]
        
    ## Load inputs
    #http://www.soda-pro.com/web-services/radiation/helioclim-3-archives-for-free

    # Average annual average temperature (12 months)
    #tamb=[0.8 4.2 8.95 13.07 17 17.2 17.05 20.1 17.22 9.4 5 3.9]#ambient temperature-hararate KARLSRUHE (1 ano)
    #tamb=[4.7 5.9 9.1 12 16.4 22.4 25.7 25.3 20.7 14.9 8.6 5.4]#alcala de henares
    tamb = np.array([12, 13, 15, 16, 19, 22, 24, 24, 23, 20, 16, 13]) # Cadiz [Cº]
    tamb = np.repeat(tamb, 720)

    
    #############################################################
    gref = 1 #1000kW/m^2
    tref = tamb.mean() # Temperature at reference condition
    kt = -3.7e-3 # Temperature coefficient of the maximum power(1/c0)
    tc = tamb + 0.0256 # Sum in each element
    tc = solar_data * tc # MATLAB -> tc = g'.*tc';
    upv = 0.986 # PV efficiency with tilted angle>>98.6#
    p_pvout_hourly = upv * (max_pv * (solar_data / gref)) * (1 + kt * (tc - tref)) #output power(kw)_hourly

    del tc
    del solar_data

    factor = 4 # 5.3(loadres)    #3.5(loadin)
  
    load  = np.concatenate((load, [load[-1]])) * factor  

    uinv = 0.7 # System efficiency
    bat_efficiency_list = [0.765,0.90,0.92,0.96,0.94,0.938,0.9155,0.95,0.86,0.855,0.70,0.80,0.75,0.70]
    bat_efficiency = bat_efficiency_list[select_bat]; # Battery efficiency [%]

    cwh = bat_cap / (uinv * bat_efficiency) * (1 + 1 - bat_DoD) # Storage capacity for battery [kWh]

####### Economic analysis #######
    shape_w = 1 ###----------> out of wind xls, from http://www.renewable-energy-concepts.com/fileadmin/user_upload/bilder/windkarte-deutschland-10m.pdf
    wind_data = wind_data * shape_w
    h2 = 18  ###-------------------------------------------------------------->changed
    h0 = 27.3 ###------------------------------------------------------------->changed
    rw = 12 # Blades diameter(m)6.4, 7.4
    pi = np.pi # 3.14159
    aw = pi * rw**2 # Swept Area>>pi x Radius² = Area Swept by the Blades
    uw = 0.95 #
    vco = 20 # Cut out -------------------> changed from 40 to 25 m/s
    vci = 3 # Cut in 2.5
    vr = 9 # Rated speed(m/s)
    pr = 30 # Rated power(kW) 5
    alfa = 0.1 # For heavily forested landscape
    pmax = 30 # Maximum output power [kWh]
    pfurl = 30 # Output power at cut-out speed9kW)
# ############################################################
    v2 = wind_data * ((h2/h0)**alfa)

## grid conected
    Png = 0 # kW output power of diesel generator --------------------------->changed 4 
    Bg = 1 # 1/kW --------------------------->changed 0.246
    Ag = 0 # 1/kW--------------------------->changed 0.08415
    Pg = 1 # Nominal power kW--------------------------->changed 4
# #Price based on in old aproach using RunDieselGeneraor.m
    Fg = Bg * Pg + Ag * Png
## MAIN PROGRAM
    contribution = np.zeros((6, 8640)) #pv,wind, battery, diesel contribution in each hour

    Ebmax = bat_cap * (1 + 1 - bat_DoD) # battery capacity 40 kWh-----------> normally never fully charged
    Ebmin = bat_cap * (1 - bat_DoD) #40kWh
    SOCb = 0.2 #state of charge of the battery>>20#
    Eb = np.zeros(8640)
    time1 = np.zeros(8640)
    gridc = np.zeros(8640)
    Edump = np.zeros(8640)
    Edch = np.zeros(8640)
    Ech = np.zeros(8640)
    Emet = np.zeros(8640)
    DumpCredit = np.zeros(8640)
    Eb[0] = SOCb * Ebmax #state of charge for starting time
#^^^^^^^^^^^^^^START^^^^^^^^^^^^^^^^^^^^^^^^
    #hourly load data for one year
    Pl = load[1:]
#^^^^^^^^^^Out put power calculation^^^^^^^^
#solar power calculation
    Pp = p_pvout_hourly.copy() #output power(kw)_hourly
    Pp[Pp > max_pv] = max_pv

    Pw = np.zeros(8640)
    Pp_mean=mean(Pp)
# wind power calculation
    pwtg = np.zeros(8640)
    pwtg[(vci <= v2) & (v2 <= vr)] = \
        (pr / (vr**3 - vci**3)) * (v2[(vci <= v2) & (v2 <= vr)])**3 - \
            (vci**3 / (vr**3 - vci**3)) * pr
    pwtg[~((vci <= v2) & (v2 <= vr)) & (vr <= v2) & (v2<=vco)] = pr
    Pw[:-1] = pwtg[:-1] * uw * num_wt

    # for t in range(8639):###### COMPROBAR SI ESTÁ BIEN EL ÍNDICE Y ESO
    #     if v2[t]<vci: #v2>>hourly_wind_speed
    #         pwtg.append (0)
    #     elif (vci<=v2[t]) and (v2[t]<=vr):
    #         pwtg.append ((pr/(vr**3-vci**3))*(v2[t])**3-(vci**3/(vr**3-vci**3))*(pr))
    #     elif (vr<=v2[t]) and (v2[t]<=vco):
    #         pwtg.append(pr)
    #     else :
    #         pwtg.append(0)
    #     Pw[t]=pwtg[t]*uw*num_wt

    Pw_mean=mean(Pw)

    for t in range(1,8639):
#^^^^^^^^^^^^^^READ INPUTS^^^^^^^^^^^^^^^^^^

#^^^^^^^^^^^^^^COMPARISON^^^^^^^^^^^^^^^^^^^
        if (Pw[t]+Pp[t]) >= (Pl[t]/uinv):
        #^^^^^^RUN LOAD WITH WIND TURBINE AND PV^^^^^^
         
            if (Pw[t]+Pp[t]) > Pl[t]:
            #^^^^^^^^^^^^^^CHARGE^^^^^^^^^^^^^^^^^^^^^^^^^^
                [Edump,Eb,Ech, DumpCredit] = charge(Pw,Pp,Eb,Ebmax,uinv,bat_efficiency,Pl,t,Edump,Ech,DumpCredit)
                time1[t]=1
                contribution[0, t] = Pp[t]
                contribution[1, t] = Pw[t]
                contribution[2, t] = Edch[t]
                contribution[3, t] = gridc[t]
                contribution[4, t] = Edump[t]
                contribution[5, t] = Emet[t]
            else:
                Eb[t]=Eb(t-1)
#            return #CREO Q HABRÁ Q BORRARLO
        
        else:
       #^^^^^^^^^^^^^^DISCHARGE^^^^^^^^^^^^^^^^^^^
            [Eb,Edump,Edch,gridc,time1,t,Emet,DumpCredit] = discharge(Pw,Pp,Eb,Ebmax,uinv,Pl,t,Ebmin,Edump,Edch,gridc,time1,Emet,DumpCredit)
            contribution[0, t] = Pp[t]
            contribution[1, t] = Pw[t]
            contribution[2, t] = Edch[t]
            contribution[3, t] = gridc[t]
            contribution[4, t] = Edump[t] #contribution(6,t)=Pl(t)
            contribution[5, t] = Emet[t]

            


## plotting
    b = np.sum(contribution, axis=1)
    renewable_factor = ((b[0]+b[1]-(b[2]/(uinv*bat_efficiency)-b[2])-b[4]+b[2])/
                        (b[0]+b[1]-(b[2]/(uinv*bat_efficiency)-b[2])-b[4]+b[2]+b[3]+b[5]))
    
#AÚN NO SÉ HACERLO    #h=pie(b)
    #colormap jet
#AÚN NO SÉ HACERLO    legend('PV','WIND','BATTERY','PUBLIC GRID', 'SURPLUS')

    #reliability
    #lose of load probability=sum(load-pv-wind+battery)/sum(load)
    k=0
    aa=[]
    aa = Pl[:-2] - Pp[:-1] - Pw[:-1] + Eb[:-1]
    k = np.sum(Pl[:-2] > (Pp[:-1] + Pw[:-1] + (Eb[:-1] - Ebmin) + gridc[:-1] + Emet[:-1]))
    # for t in range(8639):
    #     aa.append (Pl[t]-Pp[t]-Pw[t]+Eb[t])
    #     if Pl[t]>(Pp[t]+Pw[t]+(Eb[t]-Ebmin)+gridc[t]):
    #         k=k+1

    LOLP=k/8640
    reliability=sum(aa)/sum(Pl)

    metering_compensation = 0.25

    price_electricity = economic_fast(gridc, Pl, Fg, cwh, max_pv, num_wt, bat_DoD, bat_cap, select_bat, Edch, Emet, metering_compensation)
    ali=[Pp[:168], Pw[:168], Eb[:168], gridc[:168], Pl[:168], Edump[:168]]
    ali2=[Pp[:8640], Pw[:8640], Eb[:8640], Edch[:8640], Ech[:8640], gridc[:8640], Pl[:8640], Edump[:8640], Emet[:8640], DumpCredit[:8640]]
    ali = np.array(ali).T
    ali2 = np.array(ali2).T
    
    return [LOLP, price_electricity, renewable_factor, b, ali, ali2]