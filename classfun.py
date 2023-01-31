import helpers
from qutip import qeye, tensor, destroy, basis, sigmap, sigmam, sigmaz, mesolve, basis
from qutip.piqs import *
import numpy as np
import time
import matplotlib.pyplot as plt

delta_a, delta_c = 0,0
gk = 1.6

class exact_sys:
    
    def __init__(self, N_spin, N_cav, E_spin=True):
        # N_spin: number of spins
        # N_cav: the dimension of the Fock space, assume 0 excitation in the cavity
        # E_spin: bool, whether the spins are initially excited
        self.N_spin = N_spin
        self.N_cav = N_cav
        self.E_spin = E_spin
        print("walao")
        self.ID_spin = tensor([qeye(2)]*N_spin)
        # Initial sate
        if E_spin:
            self.psi0 = tensor(basis(N_cav, 0), tensor([basis(2, 0)]*N_spin))
        else:
            self.psi0 =  psi0 = tensor(basis(N_cav, 0), tensor([basis(2, 1)]*N_spin))
        # Operators
        self.a = tensor(destroy(N_cav), self.ID_spin)
        self.ad = tensor(destroy(N_cav).dag(),self.ID_spin)
        self.sp = tensor(qeye(N_cav), helpers.sum_ops(qeye(2), sigmap(), N_spin, 1))
        self.sm = tensor(qeye(N_cav), helpers.sum_ops(qeye(2), sigmam(), N_spin, 1))
        self.sz = tensor(qeye(N_cav), helpers.sum_ops(qeye(2), sigmaz(), N_spin, 1))
        
        # Hamiltonian
        H0 = delta_c*self.ad*self.a + delta_a/2*self.sz + gk*(self.a*self.sp + self.ad*self.sm)
        self.H = H0
    
    def __str__(self):
        return f'Exact solver for {self.N_spin} spins, Fock states = {self.N_cav}'
    
    def run_mesolve(self, tlist, c_ops, e_ops):
        start = time.time()
        out = mesolve(self.H, self.psi0, tlist, c_ops = c_ops, e_ops = e_ops)
        end = time.time()
        print(f'Run time: {end-start}s')
        return out
