from qutip import *
from qutip.piqs import *
import numpy as np

def gauss(x, x_0, sigma):
    return 1*np.exp(-((x-x_0) / sigma) ** 2)

def ind_op(op1, op2, N_spin, index):
    # create a tensor product of op1 everywhere but at index, where it's op2
    # returns tensor_i!=index op1 tensor op2
    a = [op1] * N_spin
    a[index] = op2
    return tensor(a)

def sum_ops(op1, op2, N_spin, c):
    # sum of all permutations where all operators are op1 but one is op2
    return sum([c*ind_op(op1, op2, N_spin, i) for i in range(N_spin)])
