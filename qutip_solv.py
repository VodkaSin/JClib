import help_func
from qutip import *
from qutip.piqs import *
import numpy as np
import time
import matplotlib.pyplot as plt

class exact_sys:
    
    def __init__(self, N_spin, N_cav, delta_a, delta_c, gk, E_spin=True):
        # N_spin: number of spins
        # N_cav: the dimension of the Fock space, assume 0 excitation in the cavity
        # delta_a: detuning of spins wrt pump
        # delta_c: detuning of cavity wrt pump
        # E_spin: bool, whether the spins are initially excited
        self.N_spin = N_spin
        self.N_cav = N_cav
        self.delta_a = delta_a
        self.delta_c = delta_c
        self.gk = gk
        self.E_spin = E_spin
        self.ID_spin = tensor([qeye(2)]*N_spin)
        # Initial sate
        if E_spin:
            self.psi0 = tensor(basis(N_cav, 0), tensor([basis(2, 0)]*N_spin))
        else:
            self.psi0 =  psi0 = tensor(basis(N_cav, 0), tensor([basis(2, 1)]*N_spin))
        # Operators
        self.a = tensor(destroy(N_cav), self.ID_spin)
        self.ad = tensor(destroy(N_cav).dag(),self.ID_spin)
        self.sp = tensor(qeye(N_cav), help_func.sum_ops(qeye(2), sigmap(), N_spin, 1))
        self.sm = tensor(qeye(N_cav), help_func.sum_ops(qeye(2), sigmam(), N_spin, 1))
        self.sz = tensor(qeye(N_cav), help_func.sum_ops(qeye(2), sigmaz(), N_spin, 1))
        
        # Hamiltonian
        H0 = self.delta_c*self.ad*self.a + self.delta_a/2*self.sz + gk*(self.a*self.sp + self.ad*self.sm)
        self.H = H0
    
    def __str__(self):
        return f'Exact solver for {self.N_spin} spins, Fock states = {self.N_cav}'
    
    def run_mesolve(self, tlist, c_ops, e_ops):
        start = time.time()
        out = mesolve(self.H, self.psi0, tlist, c_ops = c_ops, e_ops = e_ops)
        end = time.time()
        print(f'Run time: {end-start}s')
        return out


class piqs_sys:
    def __init__(self, N_spin, N_cav, delta_a, delta_c, gk, E_spin=True):
        self.N_spin = N_spin
        self.N_cav = N_cav
        self.delta_a = delta_a
        self.delta_c = delta_c
        self.gk = gk
        self.E_spin = E_spin
        self.ID_spin = to_super(qeye(self.N_spin))
        self.ID_cav = to_super(qeye(self.N_cav))

        # Operators
        self.a = tensor(destroy(N_cav), self.ID_spin)
        self.ad = tensor(destroy(N_cav).dag(),self.ID_spin)
        self.sp = tensor(qeye(N_cav), help_func.sum_ops(qeye(2), sigmap(), N_spin, 1))
        self.sm = tensor(qeye(N_cav), help_func.sum_ops(qeye(2), sigmam(), N_spin, 1))
        self.sz = tensor(qeye(N_cav), help_func.sum_ops(qeye(2), sigmaz(), N_spin, 1))
        
        # Two-level operators
        nds = num_dicke_states(N_spin)
        [jx, jy, jz] = jspin(N_spin)
        jz = 2*jz
        jp, jm = jspin(N_spin, "+"), jspin(N_spin, "-")
        
        # Cavity operators
        a = destroy(N_cav)
        
        # System superoperators
        self.jz_tot = tensor(qeye(N_cav), jz)
        self.jm_tot = tensor(qeye(N_cav), jm)
        self.jp_tot = tensor(qeye(N_cav), jp)
        self.jpjm_tot = tensor(qeye(N_cav), jp*jm)
        self.a_tot = tensor(a, qeye(nds))
        self.ada_tot = tensor(a.dag()*a, qeye(nds))
        if self.E_spin:
            self.psi0 = tensor(fock_dm(N_cav,0), excited(N_spin, basis='dicke'))
        else:
            self.psi0 = tensor(fock_dm(N_cav,0), ground(N_spin, basis='dicke'))
        # Hamiltonian
        H_spin = self.delta_a/2 * self.jz_tot
        H_cav = self.delta_c * self.a_tot.dag() * self.a_tot
        H_int = self.gk * (self.a_tot.dag()*self.jm_tot + self.a_tot*self.jm_tot.dag())
        # if F>0:
        #     self.H = [H_spin+H_cav+H_int, [(self.a_tot.dag()+self.a_tot), Ft]] 
        # else:
        #     self.H = [H_spin+H_cav+H_int]
        
    def __str__(self):
        if self.E_spin:
            return f'PIQS solver for {self.N_spin} excited spins, Fock states = {self.N_cav}'
        else:
            return f'PIQS solver for {self.N_spin} ground state spins, Fock states = {self.N_cav}'
    
    def run_mesolve(self, tlist, c_ops, e_ops):
        start = time.time()
        out = mesolve(self.H, self.psi0, tlist, c_ops = c_ops, e_ops = e_ops)
        end = time.time()
        print(f'Run time: {end-start}s')
        return out