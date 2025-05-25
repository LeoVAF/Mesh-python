import numpy as np

from .GridConnected import GridConnected

def discharge(Pw,Pp,Eb,Ebmax,uinv,Pl,t,Ebmin,Edump,Edch,gridc,time1,Emet,DumpCredit):
    # t= t-1 # De momento porque en MATLAB se empieza en 1 y aquí en 0
    Pdch = np.zeros(len(Pw))

    Pdch[t] = (Pl[t]/uinv)-(Pw[t] + Pp[t]) 
    Edch[t] = Pdch[t]*1;    #one hour iteration time


    if (Eb[t-1]-Ebmin) > Edch[t]:
        Eb [t] = Eb[t-1] - Edch[t]
        time1[t] = 2
        DumpCredit[t] = DumpCredit[t-1]

    elif (Eb[t-1]-Ebmin) <= Edch[t]:
        Eb[t] = Ebmin
        Edch[t] = Eb[t-1] - Eb[t]

        #run load with gridc generator and renewable sources#
        [Eb,Edump,gridc,t, Emet, DumpCredit] = GridConnected(Pw,Pp,Eb,Ebmax,uinv,Pl,t,Edump,gridc,Ebmin,Emet,DumpCredit)
        # se actualizan Eb, Edump, gridc, t
        
    return [Eb,Edump,Edch,gridc,time1,t,Emet,DumpCredit]
