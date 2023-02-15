import utils
import solver.exact_solve as exact
import solver.piqs_solve as piqs
import solver.mean_solve as mf
import numpy as np
import matplotlib.pyplot as plt
import qutip


t = np.linspace(0,2,1000)
# se = exact.sys( 1, 20, 0, 0, 1.6)
kappa = 40
N_spin = 3
# fig, ax = plt.subplots(3, 1, sharex=True)
# ax[0].set(ylabel = "jz")
# ax[1].set(ylabel = 'jpjm')
# ax[2].set(xlabel = "time (us)",ylabel = "ada")

# sp1 = piqs.sys(N_spin, 30, 0, 0, 1.6, np.pi, 0)
# sp2 = piqs.sys(N_spin, 30, 0, 0, 1.6, np.pi/2, 0)
# sp3 = piqs.sys(N_spin, 30, 0, 0, 1.6, 0, 0)
# piqs_out1 = sp1.run_mesolve(t,[np.sqrt(kappa)*sp1.a_tot], [sp1.jz_tot, sp1.ada_tot, sp1.jpjm_tot, sp1.jm_tot])
# piqs_out2 = sp2.run_mesolve(t,[np.sqrt(kappa)*sp2.a_tot], [sp2.jz_tot, sp2.ada_tot, sp2.jpjm_tot, sp2.jm_tot])
# piqs_out3 = sp3.run_mesolve(t,[np.sqrt(kappa)*sp3.a_tot], [sp3.jz_tot, sp3.ada_tot, sp3.jpjm_tot, sp3.jm_tot])
sysmf = mf.sys(np.asarray([10]), np.asarray([0]), 0, 1.6, 16, 0)
out = sysmf.update()
plt.plot(np.real(out[0]))
plt.show()

# for N_spin in [8]:
#     theta1 = np.pi
#     phi1 = 0
#     theta2 = np.pi/2
#     phi2 = 0
#     theta3 = 0
#     phi3 = 0
#     sp1 = piqs.sys( N_spin, 30, 0, 0, 1.6, theta1, phi1)
#     sp2 = piqs.sys( N_spin, 30, 0, 0, 1.6, theta2, phi2)
#     sp3 = piqs.sys( N_spin, 30, 0, 0, 1.6, theta3, phi3)
#     # exact_out = se.run_mesolve(t, [np.sqrt(10)*se.a], [se.sz, se.ad * se.a])
#     piqs_out1 = sp1.run_mesolve(t,[np.sqrt(kappa)*sp1.a_tot], [sp1.jz_tot, sp1.ada_tot, sp1.jpjm_tot, sp1.jm_tot])
#     piqs_out2 = sp2.run_mesolve(t,[np.sqrt(kappa)*sp2.a_tot], [sp2.jz_tot, sp2.ada_tot, sp2.jpjm_tot, sp2.jm_tot])
#     piqs_out3 = sp3.run_mesolve(t,[np.sqrt(kappa)*sp3.a_tot], [sp3.jz_tot, sp3.ada_tot, sp3.jpjm_tot, sp3.jm_tot])
#     ax[0].plot(t, piqs_out1.expect[0],label = f"{theta1}, {phi1}")
#     ax[0].plot(t, piqs_out2.expect[0],label = f"{theta2}, {phi2}")
#     ax[0].plot(t, piqs_out3.expect[0],label = f"{theta3}, {phi3}")
#     ax[0].legend()
#     ax[1].plot(t, piqs_out1.expect[2],label = f"{theta1}, {phi1}")
#     ax[1].plot(t, piqs_out2.expect[2],label = f"{theta2}, {phi2}")
#     ax[1].plot(t, piqs_out3.expect[2],label = f"{theta3}, {phi3}")
#     ax[1].legend()
#     ax[2].plot(t, piqs_out1.expect[1],label = f"{theta1}, {phi1}")
#     ax[2].plot(t, piqs_out2.expect[1],label = f"{theta2}, {phi2}")
#     ax[2].plot(t, piqs_out3.expect[1],label = f"{theta3}, {phi3}")
#     ax[2].legend()
#     plt.show()
#     print(N_spin)
#     print(piqs_out1.expect[2][0])
#     print(piqs_out2.expect[2][0])
#     print(piqs_out3.expect[2][0])
#     print("jpjm for spins", N_spin ,qutip.expect(sp1.jpjm_tot, sp1.psi0))
# fig.show()