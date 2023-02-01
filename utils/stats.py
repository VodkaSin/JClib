import numpy as np
from numpy.random import seed, normal
from scipy.stats import norm

def gauss(x, x_0, sigma):
    # PDF function of normal distribution
    # x_0, sigma: mean and std of the normal distribution
    return 1*np.exp(-((x-x_0) / sigma) ** 2)

def gen_same_pop(N_spin, k, max_det):
    # Return array of N_spin in k classes (uniformly distributed) and detuning distribution (k array)
    # The detuning distribution follows a standard normal distribution with classes spread across 99% (3*sigma) of [0, max_det]
    pop = np.asarray([int(N_spin/k) for i in range(k)])
    det = max_det * np.asarray([2*(1-norm.cdf(i)) for i in np.linspace(0,3,k)])
    return pop, det

def gen_rand_pop(N_spin, k, sigma, min_pop):
    # Returns randomly generated population distribution following normal given standard deviation
    # Here the maximum detuning is by default 3*sigma
    # By default round up, if after rounding up, the class contains fewer than min_pop spins, discard class
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
    # Returns the center index of when the array reaches the target value within a tolerance
    pts = np.where(abs(arr) - target <tol)
    center = np.take(pts, len(pts)//2)
    return center