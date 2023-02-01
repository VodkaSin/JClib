import qutip

def ind_op(op1, op2, N_spin, index):
    # Returns a operator (tensor product of N*op1) with op2 at given index
    a = [op1] * N_spin
    a[index] = op2
    return qutip.tensor(a)

def sum_ops(op1, op2, N_spin, c):
    # sum of all permutations where all operators are op1 but one is op2
    return sum([c*ind_op(op1, op2, N_spin, i) for i in range(N_spin)])