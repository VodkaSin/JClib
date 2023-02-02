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
    def __init__(self, N_spin, N_cav, delta_a, delta_c, gk, theta, phi):
        self.N_spin = N_spin
        self.N_cav = N_cav
        self.A = np.sin(theta/2)
        self.B = np.exp(1j*phi)*np.cos(theta/2)
        self.ID_spin = qutip.qeye(self.N_spin)
        self.ID_cav = qutip.qeye(self.N_cav)

        # Operators
        self.a = qutip.tensor(qutip.destroy(N_cav), self.ID_spin)
        self.ad = qutip.tensor(qutip.destroy(N_cav).dag(),self.ID_spin)
        self.sp = qutip.tensor(qutip.qeye(N_cav), ops.sum_ops(qutip.qeye(2), qutip.sigmap(), N_spin, 1))
        self.sm = qutip.tensor(qutip.qeye(N_cav), ops.sum_ops(qutip.qeye(2), qutip.sigmam(), N_spin, 1))
        self.sz = qutip.tensor(qutip.qeye(N_cav), ops.sum_ops(qutip.qeye(2), qutip.sigmaz(), N_spin, 1))
        
        # Two-level operators
        nds = piqs.num_dicke_states(N_spin)
        [jx, jy, jz] = piqs.jspin(N_spin)
        jz = 2*jz
        jp, jm = piqs.jspin(N_spin, "+"), piqs.jspin(N_spin, "-")
        
        # Cavity operators
        a = qutip.destroy(N_cav)
        
        # System superoperators
        self.jz_tot = qutip.tensor(qutip.qeye(N_cav), jz)
        self.jm_tot = qutip.tensor(qutip.qeye(N_cav), jm)
        self.jp_tot = qutip.tensor(qutip.qeye(N_cav), jp)
        self.jpjm_tot = qutip.tensor(qutip.qeye(N_cav), jp*jm)
        self.a_tot = qutip.tensor(a, qutip.qeye(nds))
        self.ada_tot = qutip.tensor(a.dag()*a, qutip.qeye(nds))
        self.psi0 = qutip.tensor(qutip.fock_dm(N_cav,0), piqs. css(self.N_spin, self.A, self.B))

        # Hamiltonian
        H_spin = delta_a/2 * self.jz_tot
        H_cav = delta_c * self.a_tot.dag() * self.a_tot
        H_int = gk * (self.a_tot.dag()*self.jm_tot + self.a_tot*self.jm_tot.dag())
        self.H = [H_spin+H_cav+H_int] 
        
    def __str__(self):
        return f'PIQS solver for {self.N_spin} spins, {self.A}|e>+{self.B}|g>. Fock states = {self.N_cav}'


    def run_mesolve(self, tlist, c_ops, e_ops):
        start = time.time()
        out = qutip.mesolve(self.H, self.psi0, tlist, c_ops = c_ops, e_ops = e_ops)
        end = time.time()
        print(f'Run time: {end-start}s')
        return out


if __name__ == "__main__":
    t = np.linspace(0,2,1000)
    theta1 = np.pi/2
    theta2 = np.pi/4
    phi1 = 0
    phi2 = np.pi/2
    p1 = sys( 5, 20, 0, 0, 1.6, theta1, phi1)
    p2 = sys( 5, 20, 0, 0, 1.6, theta1, phi2)
    p3 = sys( 5, 20, 0, 0, 1.6, theta2, phi2)
    p_out1 = p1.run_mesolve(t, [np.sqrt(10)*p1.a_tot], [p1.jm_tot, p1.jpjm_tot])
    p_out2 = p2.run_mesolve(t, [np.sqrt(10)*p2.a_tot], [p2.jm_tot, p2.jpjm_tot])
    p_out3 = p3.run_mesolve(t, [np.sqrt(10)*p3.a_tot], [p3.jm_tot, p3.jpjm_tot])

    fig, ax = plt.subplots(2,1)

    ax[0].plot(t, p_out1.expect[0], label='')
    ax[0].plot(t, p_out2.expect[0], label="0.8,0")
    ax[0].plot(t, p_out3.expect[0], label="0.9,0.5")

    ax[1].plot(t, p_out1.expect[1], label="1,0")
    ax[1].plot(t, p_out2.expect[1], label="0.8,0")
    ax[1].plot(t, p_out3.expect[1], label="0.9,0.5")
    
    plt.legend()
    plt.show()
    