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
v_f = (2 * Ef / m) ** 0.5

#@TODO calculate later
tau_in = 0.1 * tau_tr.number() * tau_tr.unit()
Q_dc = 2 * (tau_tr / tau_in) * (e * E_dc * v_f / w_c) ** 2 * (constants.pi / (hbar * w_c)) ** 2
Q_dc = Q_dc.cast_unit(unt.unitless)

dingle = - constants.pi / (w_c * tau_q)
dingle = dingle.cast_unit(unt.unitless)
dingle = np.exp(dingle.number())


def f_osc_arg_get_E(arg):
    return arg * hbar * w_c + Ef


f_osc_arg = np.linspace(-5, 5, 1000)
dE = f_osc_arg_get_E(f_osc_arg[1]) - f_osc_arg_get_E(f_osc_arg[0])
dE = dE.cast_unit(meV)


def fermi_dirac(E, Ef, T):
    exp_arg = (E - Ef)/(k*T)
    exp_arg = exp_arg.cast_unit(unt.unitless)
    exp_arg = exp_arg.number()
    return 1.0 / (np.exp(exp_arg) + 1)


def fermi_dirac_wrapper(E):
    return fermi_dirac(E, Ef, T)


def fermi_dirac_der_wrapper(E):
    fermi_dirac_der = (fermi_dirac(E + dE, Ef, T) - fermi_dirac(E, Ef, T)) / dE
    return fermi_dirac_der.number()


def f_osc_sin_arg(E):
    arg = 2 * constants.pi * E / (hbar * w_c) * (4 * Q_dc / (1 + Q_dc))
    return arg.cast_unit(unt.unitless).number()


f_osc_arg = np.linspace(-5, 5, 1000)

# x = np.linspace(0, 2 * Ef.number(), 1000) * Ef.unit()
f_0 = map(lambda arg: fermi_dirac_wrapper(f_osc_arg_get_E(arg)), f_osc_arg)
f_0_der = map(lambda arg: fermi_dirac_der_wrapper(f_osc_arg_get_E(arg)), f_osc_arg)
f_osc_sin = map(lambda arg: np.sin(f_osc_sin_arg(f_osc_arg_get_E(arg))), f_osc_arg)
f_osc_prefix = dingle * (hbar * w_c / (2 * constants.pi * dE.unit()))
f_osc_prefix = f_osc_prefix.cast_unit(unt.unitless)
f_osc = f_osc_prefix * np.multiply(f_0_der, f_osc_sin)
f = f_osc + f_0


# x = map(lambda x: x.number(), x)
plt.plot(f_osc_arg, f)
plt.plot(f_osc_arg, f_0)
plt.show()



