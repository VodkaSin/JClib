import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

import utils.ops as ops
import qutip
import qutip.piqs as piqs
import numpy as np
import time
import matplotlib.pyplot as plt

class sys:
    
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
        self.ID_spin = qutip.tensor([qutip.qeye(2)]*N_spin)
        # Initial sate
        if E_spin:
            self.psi0 = qutip.tensor(qutip.basis(N_cav, 0), qutip.tensor([qutip.basis(2, 0)]*N_spin))
        else:
            self.psi0 = qutip.tensor(qutip.basis(N_cav, 0), qutip.tensor([qutip.basis(2, 1)]*N_spin))
        # Operators
        self.a = qutip.tensor(qutip.destroy(N_cav), self.ID_spin)
        self.ad = qutip.tensor(qutip.destroy(N_cav).dag(),self.ID_spin)
        self.sp = qutip.tensor(qutip.qeye(N_cav), ops.sum_ops(qutip.qeye(2), qutip.sigmap(), N_spin, 1))
        self.sm = qutip.tensor(qutip.qeye(N_cav), ops.sum_ops(qutip.qeye(2), qutip.sigmam(), N_spin, 1))
        self.sz = qutip.tensor(qutip.qeye(N_cav), ops.sum_ops(qutip.qeye(2), qutip.sigmaz(), N_spin, 1))
        
        # Hamiltonian
        H0 = self.delta_c*self.ad*self.a + self.delta_a/2*self.sz + gk*(self.a*self.sp + self.ad*self.sm)
        self.H = H0
    
    def __str__(self):
        return f'Exact solver for {self.N_spin} spins, Fock states = {self.N_cav}'
    
    def run_mesolve(self, tlist, c_ops, e_ops, silent=False):
        start = time.time()
        out = qutip.mesolve(self.H, self.psi0, tlist, c_ops = c_ops, e_ops = e_ops)
        end = time.time()
        if silent == False:
            print(f'Runtime: {end-start}s')
        return out

if __name__ == "__main__":
    t = np.linspace(0,2,1000)
    exact = sys( 1, 20, 0, 0, 1.6)
    exact_out = exact.run_mesolve(t, [np.sqrt(10)*exact.a], [exact.sz, exact.ad*exact.a])
    plt.plot(t, exact_out.expect[0])
    plt.show()