import utils
import solver.exact_solve as exact
import solver.piqs_solve as piqs
import numpy as np
import matplotlib.pyplot as plt


Tr = utils.unit_time(1.6, 1000, 160)
Td = utils.delay_time(1.6, 12, 40)
print(Tr, Td)

t = np.linspace(0,2,1000)
# se = exact.sys( 1, 20, 0, 0, 1.6)
kappa = 40
for N_spin in [6,8,10,12]:
    sp = piqs.sys( N_spin, 30, 0, 0, 1.6, np.pi, 0)
# exact_out = se.run_mesolve(t, [np.sqrt(10)*se.a], [se.sz, se.ad * se.a])
    piqs_out = sp.run_mesolve(t,[np.sqrt(kappa)*sp.a_tot], [sp.jz_tot, sp.ada_tot])
    print(f"{N_spin} spins, Td = {utils.delay_time(1.6, N_spin, kappa)}, Tmes = {t[utils.ind_where(piqs_out.expect[0], 0, 0.05)]}")
# plt.plot(t, exact_out.expect[0],label='Exact')
    plt.plot(t, piqs_out.expect[0],label = N_spin)
plt.legend()
plt.show()