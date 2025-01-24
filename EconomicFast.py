from math import ceil

def economic_fast(gridc, Pl, Fg, cwh, max_pv, num_wt, bat_DoD, bat_cap, select_bat, Edch, Emet, metering_compensation):
    exch_rate = 1.14 #-----------> Exchange rate €$ to US$

    WT_COST = 2391*exch_rate # Wind turbine unit cost [€$]
    PV_COST = 1500*exch_rate # PV generation cost [€$]
    
    # Each battery capacity cost [€$]
    bat_cap_cost_list = [7.31393,28.575,28.575,9.8066225,20.67004375,7.540625,10.1099133,6.746875,38.1,6.99371645,9.55675,31.40075,45.10395475,14.5415,12.4139325]
    # Each battery lifetime [Year]
    bat_lf_list = [18,17.5,7,15,10,10,10,20,14,13.5,20,3,15,6.5]
    # Each battery cycle number
    bat_cycle_list = [1400,8000,600,5000,1500,4000,3000,1000,3000,3250,1250,1000,10000,2000]
    
    BAT_CAP_COST = (bat_cap*bat_cap_cost_list[select_bat])*exch_rate
    BAT_BOPC = 440.7 #------------> Balance of Plant Cost (HVAC, Control units etc.) [€$/kWh]
    BAT_BOP = 100 #---------------> Assumed to be 20 kW as trade off as no information is available [€$]
    BAT_OC = 327 #----------------> Installation cost etc. [€$]
    BAT_BOPT = ((BAT_BOPC+BAT_OC)*BAT_BOP)*0 #-->Additional BoP Costs of Battery RFV [€$]
    DSL_COST = 0*exch_rate #-----> maybe set to 0 --> 1000
    DSL_FC = 0.65*exch_rate  #-------------------> Grid connection --> in $$ 0.65   0.41
    INV_CKW = 643*exch_rate #------------------> Dependent on power output [€$]
    INV_MAX = 100 #--------------------> Has to be same as upper boundary for PV in PSO
    INV_COST = INV_CKW*INV_MAX
    PV_reg = 1500
    Wind_reg = 10
    
    INTREST = 6 # Interest/fee [%]
    INFLATION = 1.4 # Inflation [%]
    INFLATION_fuel = 1.4 # Fuel inflation [%]
    #life time
    WT_LF = 20 # Wind turbine lifetime [year]
    PV_LF = 20 # PV lifetime [year]
    BAT_LF = bat_lf_list[select_bat] # Battery lifetime [year]
    DSL_LF = 50 # [year]
    INV_LF = 15 # [year]
    PRJ_LF = 24 # HMGS project lifetime [year]

    BAT_cycle = bat_cycle_list[select_bat] # Life cycles

    # running cost
    OM = 20 # Percentage of initial costs [%]
    # rated power
    DSL_P = 1 #---------------->4 formerly
    
    ####### Economic analysis #######
    k_grid = (gridc[gridc != 0]/1).sum() # Total consumption from grid
    fuel_consumption = Fg * k_grid # Fuel consuption in one year for gridc

    ####### Net metering #######
    k_met = (Emet[Emet != 0]/1).sum() # Total metering [kWh]
    fuel_consumption += (1 - metering_compensation) * Fg * k_met # Fuel consumption in one year for gridc

    k = k_grid + k_met
    k = DSL_LF/k # Year life time
    if k < PRJ_LF:
        n = ceil(PRJ_LF/k) # n is number of repalcement for gridc in project life time
        price_d = DSL_COST*DSL_P*n 
    else:
        k_d = PRJ_LF
        price_d = DSL_COST*DSL_P
        
    ####### Battery cost #######
    k = ceil(PRJ_LF/BAT_LF)
    bat_price = BAT_CAP_COST*cwh*ceil(PRJ_LF/BAT_LF) + BAT_BOPT # Added BoP Cost
    pv_price = PV_COST*max_pv*ceil(PRJ_LF/PV_LF)
    wt_price = WT_COST*num_wt*ceil(PRJ_LF/WT_LF)
    dc = (BAT_CAP_COST*cwh)/(BAT_cycle*cwh*bat_DoD)

    ####### HGMS costs #######
    i = (INTREST-INFLATION)/100 # Real interest rate = monetary interest rate-rate of inflation
    initial_cost = wt_price + pv_price + bat_price + price_d + INV_COST + PV_reg + Wind_reg
    OM = initial_cost*(OM/100)
    initial_cost = initial_cost+OM # Adding operation and maintenance cost

    Anual_cost = initial_cost*((i*(1+i)**PRJ_LF)/(((1+i)**PRJ_LF)-1))

    i = (INTREST-INFLATION_fuel)/100 # Fuel real interest rate = monetary interest rate-rate of inflation
    Anual_cost_fuel = fuel_consumption*PRJ_LF*DSL_FC*((i*(1+i)**PRJ_LF)/(((1+i)**PRJ_LF)-1))
    Anual_cost_batery = dc*sum(Edch)*PRJ_LF*((i*(1+i)**PRJ_LF)/(((1+i)**PRJ_LF)-1)) # Includes cycles + O&M costs of battery + Annuity
    Anual_cost = Anual_cost + Anual_cost_fuel + Anual_cost_batery
    Anual_load = sum(Pl)
    price_electricity = Anual_cost/Anual_load

    return price_electricity

