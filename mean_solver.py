import cupy as cp
import numpy as cp

class sys:
    def __init__(self, pop_inclass, delta_a, delta_c, gk, theta, phi, cav_decay, spin_decay, spin_dephase):
        """
        Initialize basic icputs of the system
        pop_inclass : cp.array
            Contains the number of spins in each class arranged by index
        delta_a : cp.array
            Contains the detuning of the spins in each class arranged by index
        delta_c : float
            Cavity detuning from pump laser
        gk : float
            Universal coupling strength
        theta,phi : float
            Bloch vector: cos(theta/2)|g> + sin(theta/2)exp(i*phi)|e>
            theta = pi: excited
        cav_decay : float
            Cavity decay rate
        spin_decay : float
            Local spin decay rate
        spin_dephase : float
            Local spin dephase rate
        """
        self.pop_inclass = pop_inclass
        self.k = cp.size(pop_inclass) # Total number of classes (ensembles)
        self.delta_a = delta_a
        self.delta_c = delta_c
        self.gk = gk
        self.theta = theta
        self.phi = phi
        self.kappa = cav_decay
        self.gamma = spin_decay
        self.Gamma = spin_dephase
        
        # Convenient parameters to use
        self.w_spin = self.delta_a - 1j*(self.gamma/2 + self.Gamma)
        self.w_cav = self.delta_c - 0.5j*self.kappa

        # Cavity pumping is set to 0 by default
        self.F = 0.

        # Initialize the mean field
        self.initialize()

    def initialize(self):
        """
        Inizializes the mean field matrices
        """
        G = cp.cos(self.theta/2) # Coefficient of ground state
        E = cp.sin(self.theta/2)*cp.exp(1j*self.phi) # Coefficient of excited state

        self.a = 0.+ 0j
        self.ada = 0.
        self.a2 = 0.

        self.sz = -cp.cos(self.theta)*cp.ones(self.k)
        self.sm = G*E*cp.ones(self.k)
        self.sp = cp.conjugate(E)*G*cp.ones(self.k)

        self.a_sz = cp.zeros(self.k, dtype=cp.cfloat)
        self.a_sm = cp.zeros(self.k, dtype=cp.cfloat)
        self.a_sp = cp.zeros(self.k, dtype=cp.cfloat)

        self.sz_sm = 0.25*cp.sin(2*self.theta)*cp.exp(-1j*self.phi) * cp.ones((self.k, self.k), dtype=cp.cfloat)
        self.sz_sz = cp.cos(self.theta)**2 *cp.ones((self.k, self.k), dtype=cp.cfloat)
        self.sm_sm = 0.25*cp.sin(self.theta)**2 * cp.ones((self.k, self.k), dtype=cp.cfloat)
        self.sp_sm = G**2* cp.abs(E)**2 * cp.ones((self.k, self.k), dtype=cp.cfloat)
    
    """
    Functions to calculate mean-field increments at each time step
    """

    def cal_da(self):
        return (-1j * self.w_cav * self.a - 1j * self.gk * sum(self.pop_inclass * self.sm) - 1j * self.F) 
    
    def cal_da2(self):
        return (-2j * self.w_cav * self.a2 - 2j * self.gk * cp.sum(self.pop_inclass * self.a_sm) - 2j * self.F * self.a)
    
    def cal_dada(self):
        """
        Returns a real number: Intrafield cavity photons
        """
        return cp.real(-2 * self.gk * cp.sum(self.pop_inclass * cp.imag(self.a_sp)) - 2 * self.F * cp.imag(self.a) - self.kappa * self.ada)

    def cal_dsz(self):
        """
        Returns a real array (self.k): Average spin inversion in each class
        """
        return cp.real(4 * self.gk * cp.imag(self.a_sp) - self.gamma * (1 + self.sz))
    
    def cal_dsm(self):
        return (-1j * self.w_spin * self.sm + 1j * self.a_sz * self.gk)
    
    def cal_da_sz(self):
        return (-1j * self.w_cav * self.a_sz - 1j * self.gk * (-self.sm + (self.pop_inclass-1) * self.sm_sz_s) - \
                2j * self.gk * ((self.a2 * self.sp + 2 * self.a * self.a_sp - 2 * self.a**2 * self.sp) - (cp.conjugate(self.a) * self.a_sm + self.a * cp.conjugate(self.a_sp) +\
                 self.ada * self.sm - 2*cp.abs(self.a)**2 * self.sm)) - self.gamma * (self.a + self.a_sz) - 1j * self.F_t * self.sz - \
                1j * self.gk * cp.sum(self.sz_sm_d * self.pop_inclass[..., None], axis=0))