import utils.ops as ops
import qutip
import qutip.piqs as piqs
import numpy as np
import time
import matplotlib.pyplot as plt

class sys:
    def __init__(self, N_spin, N_cav, delta_a, delta_c, gk, E_spin=True):
        self.N_spin = N_spin
        self.N_cav = N_cav
        self.E_spin = E_spin
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
        if self.E_spin:
            self.psi0 = qutip.tensor(qutip.fock_dm(N_cav,0), piqs.excited(N_spin, basis='dicke'))
        else:
            self.psi0 = qutip.tensor(qutip.fock_dm(N_cav,0), piqs.ground(N_spin, basis='dicke'))
        # Hamiltonian
        H_spin = delta_a/2 * self.jz_tot
        H_cav = delta_c * self.a_tot.dag() * self.a_tot
        H_int = gk * (self.a_tot.dag()*self.jm_tot + self.a_tot*self.jm_tot.dag())
        self.H = [H_spin+H_cav+H_int] 
        
    def __str__(self):
        if self.E_spin:
            return f'PIQS solver for {self.N_spin} excited spins, Fock states = {self.N_cav}'
        else:
            return f'PIQS solver for {self.N_spin} ground state spins, Fock states = {self.N_cav}'
    
    def run_mesolve(self, tlist, c_ops, e_ops):
        start = time.time()
        out = qutip.mesolve(self.H, self.psi0, tlist, c_ops = c_ops, e_ops = e_ops)
        end = time.time()
        print(f'Run time: {end-start}s')
        return out


if __name__ == "__main__":
    t = np.linspace(0,2,1000)
    p = sys( 3, 20, 0, 0, 1.6)
    p_out = p.run_mesolve(t, [np.sqrt(10)*p.a_tot], [p.jz_tot, p.ada_tot])
    plt.plot(t, p_out.expect[0])
    plt.show()