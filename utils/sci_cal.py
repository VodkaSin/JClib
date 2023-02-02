import numpy as np

def unit_time(gk, N_spin, kappa):
    Omega_0 = 2*gk
    lim = Omega_0*np.sqrt(N_spin)
    if kappa < 1.5*lim:
        print("Warning: not in the overdamped regime")
    Gc = Omega_0**2/kappa
    Tr = 1/Gc/N_spin
    return Tr

def delay_time(gk, N_spin, kappa):
    Tr = unit_time(gk, N_spin, kappa)
    if N_spin > 100000:
        Td =  Tr * np.log(N_spin)
    else:
        Td = Tr * np.sum([1/(1+i) for i in range(N_spin)])
    return Td