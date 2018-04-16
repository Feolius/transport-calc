import numpy as np
import unum.units as unt
import unum.core
import scipy.constants as constants
import matplotlib.pyplot as plt

meV = unum.core.new_unit('meV', 10 ** -3 * unt.eV, 'meV')
T = 4.2 * unt.K
B = 0.4 * unt.T
n = 7.7 * 10 ** 15 * unt.m ** -2
m = 0.068 * constants.m_e * unt.kg
e = constants.e * unt.C
k = constants.k * unt.J / unt.K
tau_q = 4.7 * unt.ps
mu = 200 * unt.m ** 2 / (unt.V * unt.s)
tau_tr = mu * m / e
tau_tr = tau_tr.cast_unit(unt.ps)
I_dc = 25 * unt.uA
W = 50 * unt.um
E_dc = (I_dc / W) * B / (n * e)
E_dc = E_dc.cast_unit(unt.V / unt.m)
hbar = constants.hbar * unt.J * unt.s
w_c = e * B / m
dos = m / (constants.pi * hbar ** 2)
Ef = n / dos
Ef = Ef.cast_unit(unt.J)

dingle = - constants.pi / (w_c * tau_q)
dingle = dingle.cast_unit(unt.unitless)
dingle = np.exp(dingle.number())

def fermi_dirac(E, Ef, T):
    exp_arg = (E - Ef)/(k*T)
    exp_arg = exp_arg.cast_unit(unt.unitless)
    exp_arg = exp_arg.number()
    return 1.0 / (np.exp(exp_arg) + 1)





def fermi_dirac_vector(E):
    return fermi_dirac(E, Ef, T)

Ef = Ef.cast_unit(unt.J)
x = np.linspace(0, 2 * Ef.number(), 1000) * unt.J
y = map(fermi_dirac_vector, x)
# y_der = np.diff(y)
# y_der = np.append(y_der, [y_der[-1]])
x = map(lambda x: x.number(), x)
plt.plot(x, y)
plt.show()


