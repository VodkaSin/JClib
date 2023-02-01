import utils
import solver.exact_solve as exact
# import piqs_solve as piqs
import numpy as np
import matplotlib.pyplot as plt


Tr = utils.unit_time(1.6, 1000, 160)
Td = utils.delay_time(1.6, 1000, 160)
print(Tr, Td)

t = np.linspace(0,2,1000)
se = exact.sys( 1, 20, 0, 0, 1.6)
# sp = piqs.sys( 1, 20, 0, 0, 1.6)
exact_out = se.run_mesolve(t, [np.sqrt(10)*se.a], [se.sz, se.ad * se.a])
# piqs_out = sp.run_mesolve(t,[np.sqrt(10)*sp.a_tot], [sp.jz_tot, sp.ada_tot])
plt.plot(t, exact_out.expect[0],label='Exact')
# plt.plot(t, piqs_out.expect[0],label='PIQS')
plt.legend()
plt.show()