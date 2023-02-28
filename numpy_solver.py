import numpy as np
from numpy.random import seed, normal
import matplotlib.pyplot as plt
import time

A = [0, 1/4, 3/8, 12/13, 1, 1/2]

B = np.asarray([[],
     [1/4],
     [3/32, 9/32],
     [1932/2197, -7200/2197, 7296/2197],
     [439/216, -8, 3680/513, -845/4104],
     [-8/27, 2, -3544/2565, 1859/4104, -11/40]], dtype=object)
C = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]
CH = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]
CT = [1/360, 0, -128/4275, -2197/75240, 1/50, 2/55]

def RK45(fun, t0, y0, h, iter):
    """
    Returns the new stepsize and value
    Input:
    fun: function
    y0: current value
    """
    k1 = h * fun(t0 + A[0]*h, y0)
    k2 = h * fun(t0 + A[1]*h, y0 + B[1]*k1)
    k3 = h * fun(t0 + A[2]*h, y0 + B[2][0]*k1 + B[2][1]*k2)
    k4 = h * fun(t0 + A[3]*h, y0 + B[3][0]*k1 + B[3][1]*k2 + B[3][2]*k3)
    k5 = h * fun(t0 + A[4]*h, y0 + B[4][0]*k1 + B[4][1]*k2 + B[4][2]*k3 +B[4][3]*k4)
    k6 = h * fun(t0 + A[5]*h, y0 + B[5][0]*k1 + B[5][1]*k2 + B[5][2]*k3 +B[5][3]*k4
                 + B[5][4]*k5)
    k = np.asarray([k1,k2,k3,k4,k5,k6])
    y1 = y0 + sum([CH[i]*k[i] for i in range(6)])
    TE = np.abs(sum([CT[i]*k[i] for i in range(6)])).max()
    print(TE)
    
    fac = 0.9
    min_step = 1e-6
    max_step = 1e-3
    tol = 1e-5

    h_new = min(max_step, max(min_step, h * fac * (tol/(TE+1e-6))**0.2))
    if TE > tol:
        print("Not precise enough")
        iter += 1
        if iter>5:
            return y0, min_step
        else:
            return RK45(fun, t0, y0, h_new, iter)
    else:
        return y1, h_new

def gauss(x, x_0, sigma):
    """
    PDF function of normal distribution
    x_0, sigma: mean and std of the normal distribution
    """
    return 1*np.exp(-((x-x_0) / sigma) ** 2)

def gen_rand_pop(N_spin, k, sigma, min_pop):
    """
    Returns randomly generated population distribution following normal given standard deviation
    Here the maximum detuning is by default 3*sigma
    By default round up, if after rounding up, the class contains fewer than min_pop spins, discard class
    """
    det = np.linspace(0, 3*sigma, k)
    interval = 3*sigma/k
    prob = np.asarray([gauss((i+.5)*interval, 0, sigma) for i in range(k)])
    prob_sum = np.sum(prob)
    pop = np.ceil(N_spin/prob_sum*prob)
    keep = np.where(pop>=min_pop)
    pop = pop[keep]
    new_k = pop.size
    return pop, det, new_k

def ind_where(arr, target, tol):
    """
    Returns the center index of when the array reaches the target value within a tolerance
    """
    pts = np.where(abs(arr) - target <tol)
    center = np.take(pts, len(pts)//2)
    return center


def unit_time(gk, N_spin, kappa):
    """
    Returns the unit time as defined for Dicke superradiance
    """
    Omega_0 = 2*gk
    lim = Omega_0*np.sqrt(N_spin)
    if kappa < 1.5*lim:
        print("Warning: not in the overdamped regime")
    Gc = Omega_0**2/kappa
    Tr = 1/Gc/N_spin
    return Tr

def delay_time(gk, N_spin, kappa, theta):
    """
    Returns the theoretical delay time for Dicke superradiance
    qinit: float
        Initial excitation J-M, in mean field context, N_spin/2 * (1+cos(theta))
    """
    Tr = unit_time(gk, N_spin, kappa)
    qinit = N_spin/2 * (1+np.cos(theta))
    if N_spin > 1000:
        Td =  Tr * np.log(N_spin/(qinit+1))
    else:
        Td = Tr * (np.sum([1/(1+i) for i in range(N_spin)])-np.log(qinit+1))
    return Td


class sys:
    def __init__(self, pop_inclass, delta_a, delta_c, gk, theta, phi, cav_decay, spin_decay, spin_dephase):
        """
        Initialize basic inputs of the system
        pop_inclass : np.array
            Contains the number of spins in each class arranged by index
        delta_a : np.array
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
        self.k = np.size(pop_inclass) # Total number of classes (ensembles)
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
        self.w_spin_r = np.tile(self.w_spin, self.k).reshape(self.k, self.k) # For convenience
        self.w_cav = self.delta_c - 0.5j*self.kappa

        # Cavity pumping is set to 0 by default
        self.F = 0.

        # Initialize the mean field
        self.initialize()

    def initialize(self):
        """
        Inizializes the mean field matrices
        """
        G = np.cos(self.theta/2) # Coefficient of ground state
        E = np.sin(self.theta/2)*np.exp(1j*self.phi) # Coefficient of excited state

        self.a = 0.+ 0j
        self.ad = np.conjugate(self.a)
        self.ada = 0.
        self.a2 = 0.

        self.sz = -np.cos(self.theta)*np.ones(self.k)
        self.sm = G*E*np.ones(self.k)
        self.sp = np.conjugate(E)*G*np.ones(self.k)

        self.a_sz = np.zeros(self.k, dtype=np.cfloat)
        self.a_sm = np.zeros(self.k, dtype=np.cfloat)
        self.a_sp = np.zeros(self.k, dtype=np.cfloat)

        self.sz_sm = 0.25*np.sin(2*self.theta)*np.exp(-1j*self.phi)\
                     * np.ones((self.k, self.k), dtype=np.cfloat)
        self.sz_sz = np.cos(self.theta)**2\
                     * np.ones((self.k, self.k), dtype=np.cfloat)
        self.sm_sm = 0.25*np.sin(self.theta)**2\
                     * np.ones((self.k, self.k), dtype=np.cfloat)
        self.sp_sm = G**2* np.abs(E)**2\
                     * np.ones((self.k, self.k), dtype=np.cfloat)
    
    """
    Functions to calculate mean-field increments at each time step
    """

    def cal_da(self):
        return (-1j * self.w_cav * self.a - 1j * self.gk * sum(self.pop_inclass
                 * self.sm) - 1j * self.F) 
    
    def cal_da2(self):
        return (-2j * self.w_cav * self.a2 - 2j * self.gk * np.sum(self.pop_inclass
                 * self.a_sm) - 2j * self.F * self.a)
    
    def cal_dada(self):
        """
        Returns a real number: Intrafield cavity photons
        """
        return np.real(-2 * self.gk * np.sum(self.pop_inclass * np.imag(self.a_sp))
                 - 2 * self.F * np.imag(self.a) - self.kappa * self.ada)

    def cal_dsz(self):
        """
        Returns a real array (self.k): Average spin inversion in each class
        """
        return np.real(4 * self.gk * np.imag(self.a_sp) - self.gamma * (1 + self.sz))
    
    def cal_dsm(self):
        return (-1j * self.w_spin * self.sm + 1j * self.a_sz * self.gk)
    
    def cal_da_sz(self):
        return (-1j * self.w_cav * self.a_sz - 1j * self.gk * (-self.sm 
                 - np.diagonal(self.sz_sm)) - 2j * self.gk * ((self.a2 * self.sp
                 + 2 * self.a * self.a_sp - 2 * self.a**2 * self.sp)
                 - (self.ad * self.a_sm + self.a * np.conjugate(self.a_sp)
                 + self.ada * self.sm - 2 * np.abs(self.a)**2 * self.sm))
                 - self.gamma * (self.a + self.a_sz) - 1j * self.F * self.sz
                 - 1j * self.gk * np.sum(self.sz_sm * self.pop_inclass[..., None], axis=0))
    
    def cal_da_sm(self):
        return (-1j * (self.w_spin + self.w_cav) * self.a_sm 
                 + 1j * self.gk * (np.diagonal(self.sm_sm) + 2 * self.a_sz * self.a 
                 + self.a2 * self.sz - 2 * self.a**2 * self.sz) 
                 - 1j * self.F * self.sm - 1j * self.gk * np.sum(self.sm_sm 
                 * self.pop_inclass[..., None], axis=0))

    def cal_da_sp(self):
        return (1j * (np.conj(self.w_spin) - self.w_cav) * self.a_sp 
                 - .5j * self.gk * (1 - self.sz) + 1j * self.gk * np.diagonal(self.sp_sm)
                 -1j * self.gk * np.sum(self.sp_sm * self.pop_inclass[..., None], axis=0) 
                 - 1j * self.gk * ((1 + self.ada) * self.sz 
                 + 2 * np.real(self.a * np.conjugate(self.a_sz))
                 - 2 * np.abs(self.a)**2 * self.sz) -1j * self.F * self.sp)

    def cal_dsm_sm(self):
        return (-1j * (self.w_spin_r+self.w_spin_r.T) * self.sm_sm 
                 + 1j * self.gk * (self.a * self.sz_sm
                 + self.a_sz * np.vstack(self.sm) + self.sz * np.vstack(self.a_sm)
                 - self.a * self.sz * np.vstack(self.sm)) 
                 + 1j * self.gk * (self.a * np.conjugate(self.sz_sm).T 
                 + self.a_sm * np.vstack(self.sz) + self.sm * np.vstack(self.a_sz)
                 - 2 * self.a * self.sm * np.vstack(self.sz)))
    
    def cal_dsz_sz(self):
        return (4 * self.gk * ((self.a_sp * np.vstack(self.sz) 
                 + self.sp * np.vstack(self.a_sz) + self.a * np.conjugate(self.sz_sm).T 
                 - 2 * self.a * self.sp * np.vstack(self.sz))
                 + (self.a_sp.T * self.sz + self.a_sz * np.vstack(self.sp)
                 + self.a * np.conjugate(self.sz_sm) 
                 - 2 * self.a * np.vstack(self.sp) * self.sz))
                 + self.gamma * (np.tile(self.sz, self.k).reshape(self.k, self.k).T
                 + self.sz_sz))

    def cal_dsz_sm(self):
        return ((-1j * self.w_spin_r + self.gamma) * self.sz_sm 
                 + 1j * self.gk * (self.a_sz * np.vstack(self.sz)
                 + np.vstack(self.a_sz)*self.sz + self.a * self.sz_sz
                 - 2 * self.a * self.sz * np.vstack(self.sz)) 
                 - 2j * self.gk * (self.a_sp * np.vstack(self.sm) 
                 + np.vstack(self.a_sm) * self.sp + self.a * self.sp_sm
                 - 2 * self.a * self.sp * np.vstack(self.sm)
                 - (np.conjugate(self.a_sp) * np.vstack(self.sm) 
                 + np.conjugate(self.a_sp).T * self.sm + self.ad * self.sm_sm
                 - 2 * self.ad * self.sm * np.vstack(self.sm)))
                 - self.gamma * np.tile(self.sm, self.k).reshape(self.k, self.k).T)

    def cal_dsp_sm(self):
        return (-1j * (self.w_spin_r - self.w_spin_r.T) * self.sp_sm
                 + 1j * (self.a_sp * np.vstack(self.sz) 
                 + np.vstack(self.a_sz) * self.sp + self.a * np.conjugate(self.sz_sm).T
                 - 2 * self.a * self.sp * np.vstack(self.sz))
                 - 1j * (np.conjugate(self.a_sz) * np.vstack(self.sm)
                 + np.conjugate(self.a_sp).T * self.sz + self.ad * self.sz_sm
                 - 2 * self.ad * self.sz * np.vstack(self.sm)))
    
    def update(self, F, dt):
        """
        Update the mean-field values for one time step of size dt
        """
        self.F = F

        self.a += self.cal_da() * dt
        self.ad = np.conjugate(self.a)
        self.ada += self.cal_dada() * dt
        self.a2 += self.cal_da2() * dt

        self.sz += self.cal_dsz() * dt
        self.sm += self.cal_dsm() * dt
        self.sp = np.conjugate(self.sm)

        self.a_sz += self.cal_da_sz() * dt
        self.a_sm += self.cal_da_sm() * dt
        self.a_sp += self.cal_da_sp() * dt

        self.sz_sm += self.cal_dsz_sm() * dt
        self.sz_sz += self.cal_dsz_sz() * dt
        self.sm_sm += self.cal_dsm_sm() * dt
        self.sp_sm += self.cal_dsp_sm() * dt

    def solve_constant(self, tlist):
        """
        Returns 
        tlist: np.array
            All the time steps with constant dt
        F_t: np.array
            Cavity pump amplitude with respect to time
        """
        intervals = np.size(tlist)
        F_t = np.zeros(intervals)
        dt = tlist[-1]/intervals
        e_ada = np.zeros(intervals)
        e_sz = np.zeros((intervals, self.k))
        e_sp_sm = np.zeros((intervals, self.k))
        print("Start solving, dt = ", dt)
        start = time.time()
        for t in range(intervals):
            self.update(F_t[t], dt)
            e_ada[t] = self.ada
            e_sz[t] = self.sz
            e_sp_sm[t] = np.diagonal(self.sp_sm)
        end = time.time()
        print(end-start)

        return [e_ada, e_sz, e_sp_sm]


    def solve_adapt(self, endtime, h0):
        """
        
        """
        self.F = 0
        t = 0
        h_store = []
        e_ada = []
        e_sz = []
        e_sp_sm = []
        while t + h0 < endtime:
            h_store.append(h0)
            [sz, h0] = RK45(self.cal_dsz, t, self.sz, h0, 1)
            print("sz", sz)
            print("stepsize", h0)
            self.sz = sz
            self.update(self.F, h0)
            e_ada.append(self.ada)
            e_sz.append(self.sz)
            e_sp_sm.append(np.diagonal(self.sp_sm))
        return [h_store, e_ada, e_sz, e_sp_sm]

if __name__ == "__main__":
    pop_inclass = np.asarray([1000 for i in range(200)])
    delta_a = np.asarray([10 for i in range(200)])
    delta_c = 0
    gk = 1.6
    theta = np.pi
    phi = 0
    cav_decay = 160
    spin_decay = 0
    spin_dephase = 0

    test_sys = sys(pop_inclass, delta_a, delta_c, gk, theta, phi, 
                 cav_decay, spin_decay, spin_dephase)
    
    num_iter = 6e4
    tlist = np.linspace(0,0.2,int(num_iter))
    F_t = np.zeros(int(num_iter))
    results = test_sys.solve_constant(tlist)
    Td_theory = delay_time(gk, 100, cav_decay, theta)
    Td_simulate = ind_where(results[1], 0.0, 0.5)
    print(Td_theory, Td_simulate)

    fig, ax = plt.subplots(3,1, sharex='col')
    ax[0].plot()
    ax[0].set_title(f"{np.size(pop_inclass)} classes, {num_iter} iterations")
    ax[2].set_xlabel(r"t($\mu s$)")
    ax[0].set_ylabel(r"$\langle a^\dagger a\rangle$")
    ax[1].set_ylabel(r"$\langle\sigma_z\rangle$")
    ax[2].set_ylabel(r"$\langle\sigma_+\sigma_-\rangle$")

    ax[0].plot(tlist, results[0])
    ax[1].plot(tlist, results[1])
    ax[2].plot(tlist, results[2])

    plt.savefig("np.png")