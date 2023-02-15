import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

import utils.ops as ops
import numpy as np
import time
import matplotlib.pyplot as plt

class sys:
    
    def __init__(self, N_class, delta_a, delta_c, gk, kappa, gamma, E_spin=True, F=0):
        
      ###########################################################################################################################      
      # Class for systems to be solved with mean-field equations
      # N_class: array storing the number of spins in different classes
      # delta_a: array of spin detuning of each class
      # delta_c: detuning between field and pump
      # kappa: cavity linewidth
      # gamma: spin dephasing
      # E_spin: Boolean, true if all spins started excited
      ###########################################################################################################################
        
        self.N_class = N_class
        self.k = len(N_class)
        self.delta_c = delta_c
        self.delta_a = delta_a
        self.gk = gk
        self.kappa = kappa
        self.gamma = gamma
        self.w_spin = self.delta_a - 1j*self.gamma/2.
        self.w_cav = self.delta_c - 1j*self.kappa/2.
        self.E_spin = E_spin
       
        self.F = F
        self.F_t = 0 #FOR NOW

    ###########################################################################################################################
        # Initialization of operators
    ###########################################################################################################################       
        
        self.a = 0.+0j
        self.da = 0.
        self.ada = 0.
        self.dada = 0.
        self.a2 = 0.
        self.da2 = 0.
        self.sm = np.zeros(self.k, dtype=np.cfloat) 
        self.dsm = np.zeros(self.k, dtype=np.cfloat) 
        self.sp = np.zeros(self.k, dtype=np.cfloat)
        self.dsp = np.zeros(self.k, dtype=np.cfloat) 
        if E_spin:
            self.sz = np.ones(self.k, dtype=np.cfloat) # Added self.N_class*
        else:
            self.sz = -np.ones(self.k, dtype=np.cfloat)
        self.dsz = np.zeros(self.k, dtype=np.cfloat)

        self.a_sz = np.zeros(self.k, dtype=np.cfloat) # Product of expectation value of a and sz
        self.da_sz = np.zeros(self.k, dtype=np.cfloat) 
        self.a_sm = np.zeros(self.k, dtype=np.cfloat)
        self.da_sm = np.zeros(self.k, dtype=np.cfloat) 
        self.a_sp = np.zeros(self.k, dtype=np.cfloat)
        self.da_sp = np.zeros(self.k, dtype=np.cfloat) 
        
        # For different spins in then same class, each pair is equivalent
        self.sm_sz_s = np.zeros(self.k, dtype=np.cfloat)
        self.dsm_sz_s = np.zeros(self.k, dtype=np.cfloat) 
        self.sm_sp_s = np.zeros(self.k, dtype=np.cfloat)
        self.dsm_sp_s = np.zeros(self.k, dtype=np.cfloat) 
        self.sm_sm_s = np.zeros(self.k, dtype=np.cfloat)
        self.dsm_sm_s = np.zeros(self.k, dtype=np.cfloat) 
        self.sz_sz_s = np.ones(self.k, dtype=np.cfloat)
        self.dsz_sz_s = np.zeros(self.k, dtype=np.cfloat) 

        # Different class all set to 0, ignore if there is only one class
        self.sm_sm_d = np.zeros((self.k, self.k), dtype=np.cfloat)
        self.dsm_sm_d = np.zeros((self.k, self.k), dtype=np.cfloat)
        if self.k>1:
            self.sz_sz_d = np.ones((self.k, self.k), dtype=np.cfloat)
        else:
            self.sz_sz_d = np.zeros((self.k, self.k), dtype=np.cfloat)
        self. dsz_sz_d = np.zeros((self.k, self.k), dtype=np.cfloat)
        np.fill_diagonal(self.sz_sz_d, 0)
        self.sz_sm_d = np.zeros((self.k, self.k), dtype=np.cfloat)
        self.dsz_sm_d = np.zeros((self.k, self.k), dtype=np.cfloat)
        self.sp_sm_d = np.zeros((self.k, self.k), dtype=np.cfloat)
        self.dsp_sm_d = np.zeros((self.k, self.k), dtype=np.cfloat)
        

    def __str__(self):
        if self.E_spin:
            return f'Mean-field solver for {self.N_spin} excited spins'
        else:
            return f'Mean-field solver for {self.N_spin} ground state spins'
    
    ###########################################################################################################################
        # Computing 1st and 2nd order differentials
    ###########################################################################################################################
    
    
    def cal_da(self):
        return (-1j * self.w_cav * self.a - 1j * self.gk * sum(self.N_class * self.sm) - 1j * self.F_t) 
    
    def cal_dda(self):
        return (-1j * self.w_cav * self.da - 1j * self.gk * sum(self.N_class * self.dsm))

    def cal_da2(self):
        return (-2j * self.w_cav * self.a2 - 2j * self.gk * np.sum(self.N_class * self.a_sm) - 2j * self.F_t * self.a) 
    
    def cal_dda2(self):
        return (-2j * self.w_cav * self.da2 - 2j * self.gk * np.sum(self.N_class * self.da_sm) - 2j * self.F_t * self.da)

    def cal_dada(self):
        return (-2 * self.gk * np.sum(self.N_class * np.imag(self.a_sp)) - 2 * self.F_t * np.imag(self.a) - self.kappa * self.ada)
    
    def cal_ddada(self):
        return (-2 * self.gk * np.sum(self.N_class * np.imag(self.da_sp)) - 2 * self.F_t * np.imag(self.da) - self.kappa * self.ada)
    
    ###########################################################################################################################
    ## Same class s expectation values
        
    
    def cal_dsm(self):
        return (-1j * self.w_spin * self.sm + 1j * self.a_sz * self.gk)

    def cal_dsz(self):
        return (4 * self.gk * np.imag(self.a_sp) - self.gamma * (1 + self.sz))
    

    def cal_da_sz(self):
        return (-1j * self.w_cav * self.a_sz - 1j * self.gk * (-self.sm + (self.N_class-1) * self.sm_sz_s) - \
                2j * self.gk * ((self.a2 * self.sp + 2 * self.a * self.a_sp - 2 * self.a**2 * self.sp) - (np.conjugate(self.a) * self.a_sm + self.a * np.conjugate(self.a_sp) +\
                 self.ada * self.sm - 2*np.abs(self.a)**2 * self.sm)) - self.gamma * (self.a + self.a_sz) - 1j * self.F_t * self.sz - \
                1j * self.gk * np.sum(self.sz_sm_d * self.N_class[..., None], axis=0)) # Assuming that N is a row vector of size k


    def cal_da_sm(self):
        return (-1j * (self.w_spin + self.w_cav) * self.a_sm - 1j * self.gk * ((self.N_class-1) * self.sm_sm_s - 2 * self.a_sz * self.a - self.a2 * self.sz\
                + 2 * self.a**2 * self.sz) - 1j * self.F_t * self.sm - 1j * self.gk * np.sum(self.sm_sm_d * self.N_class[..., None], axis=0))
    
    
    def cal_da_sp(self):
        return (1j * (np.conj(self.w_spin) - self.w_cav) * self.a_sp - .5j * self.gk * (1-self.sz) - 1j * self.gk * (self.N_class-1) * self.sm_sp_s\
                -1j * self.gk * np.sum(self.sp_sm_d * self.N_class[..., None], axis=0) - 1j * self.gk * ((1+self.ada) * self.sz + 2 * np.real(self.a * np.conjugate(self.a_sz)) - 2 * np.abs(self.a)**2 * self.sz)\
                -1j * self.F_t * self.sp)


    ###########################################################################################################################
    # Spin-spin interactions from the same class
    def cal_dsm_sz_s(self):
        ad = np.conjugate(self.a)
        return (-1j * self.w_spin * self.sm_sz_s + 1j * self.gk * (self.a * self.sz_sz_s + 2 * self.sz * self.a_sz - 2 * self.a * self.sz**2)- 2j * self.gk\
            *(self.a_sm * self.sp + self.a_sp * self.sm + self.a * self.sm_sp_s - 2 * self.a * self.sm * self.sp - (ad * self.sm_sm_s + 2*np.conjugate(self.a_sp) * self.sm\
            - 2 * ad * self.sm**2)) - self.gamma*(self.sm + self.sm_sz_s))

    def cal_dsm_sp_s(self):
        return (2 * np.imag(self.w_spin) * self.sm_sp_s + \
                2* self.gk * np.imag(self.sz * np.conjugate(self.a_sp) + \
                np.conjugate(self.a) * self.sm_sz_s + self.sm * np.conjugate(self.a_sz)-\
                2 * np.conjugate(self.a)* self.sm * self.sz))
    
    def cal_dsm_sm_s(self):
        return (-2j * self.w_spin * self.sm_sm_s + 2j * self.gk * (self.a * self.sm_sz_s + self.a_sm * self.sz + self.a_sz * self.sm-\
                    2 * self.a * self.sm * self.sz))

    def cal_dsz_sz_s(self):
        return (8 * self.gk * np.imag(self.a * np.conjugate(self.sm_sz_s) + self.a_sp * self.sz + self.a_sz * self.sp - 2 * self.a * self.sp * self.sz) - \
                   2 *self.gamma * (self.sz + self.sz_sz_s))

    ###########################################################################################################################
    # Spin-spin interactions from a different class

    
    def cal_dsm_sm_d(self):
        N = np.size(self.w_spin)
        w_spin_r = np.tile(self.w_spin, N).reshape(N, N)
        w_spin_c = w_spin_r.T
        a_sz_c = np.vstack(self.a_sz)
        sm_c = np.vstack(self.sm)
        a_sm_c = np.vstack(self.a_sm)
        sz_c = np.vstack(self.sz)

        d = -1j * (w_spin_r + w_spin_c) * self.sm_sm_d + \
                   1j * self.gk * (self.a * self.sz_sm_d + sm_c * self.a_sz + self.sz * a_sm_c - 2 * self.a * self.sz * sm_c)\
                   + 1j * self.gk * (self.a * self.sz_sm_d.T + self.sm * a_sz_c + sz_c * self.a_sm - 2 * self.sm * sz_c)
        np.fill_diagonal(d, 0)
        return d


    def cal_dsz_sz_d(self):
        N = np.size(self.sz)
        a_sz_c = np.vstack(self.a_sz)
        a_sp_c = np.vstack(self.a_sp)
        sp_c = np.vstack(self.sp)
        sz_c = np.vstack(self.sz)
        sz_mat_c = np.tile(sz_c, N)
        sz_mat_r = np.tile(self.sz, N).reshape(N, N)

        d = 4 * self.gk * np.imag(self.a * np.conjugate(self.sz_sm_d).T + \
                    self.sp * a_sz_c + self.a_sp * sz_c - 2 * self.a * self.sp * sz_c) + \
                   4 * self.gk * np.imag(self.a * np.conjugate(self.sz_sm_d) + \
                    self.a_sz * sp_c + self.sz * a_sp_c - 2 * self.a * self.sz * sp_c) -\
                   self.gamma*(sz_mat_c + sz_mat_r + 2 * self.sz_sz_d)
        np.fill_diagonal(d, 0)
        return d

    def cal_dsz_sm_d(self):
        ad = np.conjugate(self.a)
        N = np.size(self.sm)
        w_spin_c = np.vstack(self.w_spin)
        sz_c = np.vstack(self.sz)
        sm_c = np.vstack(self.sm)
        a_sm_c = np.vstack(self.a_sm)
        a_sz_c = np.vstack(self.a_sz)
        ad_sm = np.conjugate(self.a_sp)
        ad_sm_c = np.vstack(ad_sm)
        sm_mat_c = np.tile(sm_c, N)

        d = (-1j * w_spin_c * self.sz_sm_d\
                    + 1j * self.gk * (self.a * self.sz_sz_d + self.a_sz * sz_c + a_sz_c * self.sz\
                   - 2 * self.a * self.sz * sz_c) - 2j * self.gk * ((self.a * self.sp_sm_d + self.a_sp * sm_c + a_sm_c * self.sp - 2 * self.a * self.sp * sm_c)\
                   -(ad * self.sm_sm_d + ad_sm * sm_c + ad_sm_c * self.sm - 2 * ad * self.sm *sm_c))\
                   -self.gamma * (sm_mat_c + self.sz_sm_d))
        np.fill_diagonal(d, 0)
        return d


    def cal_dsp_sm_d(self):
        ad = np.conjugate(self.a)
        w_spin_c = np.vstack(self.w_spin)
        sp_sz_d = np.conjugate(self.sz_sm_d).T
        ad_sz = np.conjugate(self.a_sz)
        ad_sm_c = np.conjugate(self.a_sp).T
        sz_c = np.vstack(self.sz)
        sm_c = np.vstack(self.sm)
        a_sz_c = np.vstack(self.a_sz)
        d = -1j * (w_spin_c - np.conjugate(self.w_spin)) * self.sp_sm_d + 1j * self.gk*\
                    (self.a * sp_sz_d) + self.a_sp * sz_c + a_sz_c * self.sz - 2 * self.a * self.sp * sz_c -1j * self.gk *\
                    (ad * self.sz_sm_d + ad_sz * sm_c + ad_sm_c * self.sz - 2 * ad * self.sz * sm_c)
        np.fill_diagonal(d,0)
        return d                                                                   

    
    def bound_array(self, ub, val, val_pre):
        
        # val: the newly evaluated expectation of increments
        # val_pre: the expectation of increment from last iteration
    
        val_abs = np.abs(val)
        index = np.where(val_abs>ub)
        val[index] = val_pre[index]
        return val
    
    def bound_scalar(self, ub, val, val_pre):
        val_abs = np.abs(val)
        if val_abs>ub:
            return val_pre
        return(val)

    ###########################################################################################################################
        # Time evolution 
    ###########################################################################################################################
        
    def update(self):
        self.intervals = 1000
        self.dt = 0.001
        n_sz = np.zeros((self.intervals,self.k), dtype=np.cfloat)
        n_ada = np.zeros((self.intervals), dtype=np.cfloat)
        start = time.time()
        t = 0 # not used, positional variable
        for i in range(self.intervals):
            #self.F_t = self.pulse[i]
            self.F_t = self.F
            n_sz[i] =self.sz
            n_ada[i] = self.ada
            self.da = self.cal_da() * self.dt
            self.dada = self.cal_dada() * self.dt
            self.da2 = self.cal_da2() * self.dt
            self.dsm = self.cal_dsm() * self.dt
            self.dsz = self.cal_dsz() * self.dt
            self.da_sz = self.cal_da_sz() * self.dt
            self.da_sm = self.cal_da_sm() * self.dt
            self.da_sp = self.cal_da_sp() * self.dt
            self.dsm_sz_s = self.cal_dsm_sz_s() * self.dt
            self.dsm_sp_s = self.cal_dsm_sp_s() * self.dt
            self.dsm_sm_s = self.cal_dsm_sm_s() * self.dt
            self.dsz_sz_s = self.cal_dsz_sz_s() * self.dt
            if self.k > 1:
                self.dsm_sm_d = self.cal_dsm_sm_d() * self.dt
                self.dsz_sz_d = self.cal_dsz_sz_d() * self.dt
                self.dsz_sm_d = self.cal_dsz_sm_d() * self.dt
                self.dsp_sm_d = self.cal_dsp_sm_d() * self.dt
            
    
            self.a = self.bound_scalar(sum(self.N_class), self.a + self.da, self.a)
            self.ada = self.bound_scalar(sum(self.N_class), self.ada + self.dada, self.ada)
            if self.ada < 0:
                self.ada = 0
            self.a2 = self.bound_scalar(self.a**2, self.a2 + self.da2 , self.a2)
            self.sm = self.bound_array(1, self.sm + self.dsm , self.sm)
            self.sz = self.bound_array(1, self.sz + self.dsz, self.sz)
            self.a_sz += self.da_sz 
            self.a_sm += self.da_sm 
            self.a_sp += self.da_sp 
            self.sm_sz_s += self.dsm_sz_s 
            self.sm_sp_s += self.dsm_sp_s 
            self.sm_sm_s += self.dsm_sm_s 
            self.sz_sz_s = self.bound_array(1, self.dsz_sz_s, self.sz_sz_s)
            if self.k > 1:
                self.sm_sm_d += self.dsm_sm_d
                self.sz_sm_d += self.dsz_sm_d
                self.sz_sz_d += self.dsz_sz_d
                self.sp_sm_d += self.dsp_sm_d
            # clear_output(wait=True)
            # display('Iteration ' +str(i))
        end = time.time()
        print(f'Run time: {end-start}s')
        return [n_sz, n_ada]