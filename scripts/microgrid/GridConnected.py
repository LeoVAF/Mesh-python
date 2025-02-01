
def GridConnected(Pw,Pp,Eb,Ebmax,uinv,Pl,t,Edump,gridc,Ebmin, Emet, DumpCredit):
    Eb[t] = Eb[t-1] + (Pw[t] + Pp[t] - ((Pl[t] / uinv) * 1))
    if Eb[t] > Ebmax:
        Edump[t] = Eb[t] - Ebmax
        Eb[t] = Ebmax
        
    if Eb[t]<Ebmin:
        Edump[t] = 0
        Eb[t] = Ebmin
    
    demand_from_grid = (Pl[t]/uinv)-(Eb[t]-Eb[t-1]+Pw[t]+Pp[t])
    total_accumulated_energy = DumpCredit[t-1]

    if demand_from_grid <= total_accumulated_energy:
        Emet[t] = demand_from_grid
        gridc[t] = 0
        DumpCredit[t] = DumpCredit[t-1] - demand_from_grid
        Edump[t] = -demand_from_grid
    else:
        Emet[t] = total_accumulated_energy
        gridc[t] = demand_from_grid - total_accumulated_energy
        DumpCredit[t] = 0
        Edump[t] = -total_accumulated_energy
             
    if gridc[t] > (Pl[t]/uinv):
        gridc[t] = 0 # eh como se nao pudesse ligar a termoeletrica
    
    if Emet[t] > (Pl[t]/uinv):
        Emet[t] = 0
        Edump[t] = 0
        DumpCredit[t] = DumpCredit[t-1]
    if gridc[t] + Emet[t] > (Pl[t]/uinv):
        gridc[t] = 0
        Emet[t] = 0
        Edump[t] = 0
        DumpCredit[t] = DumpCredit[t-1]
        
    if Eb[t]<Ebmin:
        Eb[t]=0
        
    if gridc[t]<0:
        gridc[t]=(Pl[t]/uinv)-(Pw[t]+Pp[t])

    return [Eb,Edump,gridc,t, Emet, DumpCredit]
