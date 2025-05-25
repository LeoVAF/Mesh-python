import numpy as np

def charge(Pw, Pp, Eb, Ebmax, uinv, ub, Pl, t, Edump, Ech, DumpCredit):
    # t = t-1
    Pch = np.zeros(len(Pw))
    
    Pch[t] = (Pw[t] + Pp[t])- (Pl[t]/uinv)
    Ech[t] = Pch[t]*ub;
    if Eb[t] <= Eb[t-1] + Ech[t]:
        Eb [t] = Eb[t-1] + Ech[t]

        if  Eb[t]>= Ebmax:
            Eb[t] = Ebmax
            Ech[t] = Eb[t] - Eb[t-1]
            Edump[t] = Pch[t] - (Ebmax -Eb[t])
        else:
            Edump[t] = 0
    else:
        Eb[t] = Ebmax
        Edump[t] = Pch[t]-(Ebmax - Eb[t])
    
    DumpCredit[t] = DumpCredit[t-1] + Edump[t]
    return [Edump, Eb, Ech, DumpCredit]
