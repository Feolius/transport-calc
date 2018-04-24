import numpy as np
import unum.units as unt
import unum.core
import scipy.constants as constants
import matplotlib.pyplot as plt

meV = unum.core.new_unit('meV', 10 ** -3 * unt.eV, 'meV')
T = 4.2 * unt.K
B = 0.3 * unt.T
n = 7.7 * 10 ** 15 * unt.m ** -2
m = 0.068 * constants.m_e * unt.kg
epsilon = 12
epsilon_0 = constants.epsilon_0 * unt.F / unt.m
mu_0 = constants.mu_0 * unt.N / unt.A ** 2
k = 1 / (4 * constants.pi * epsilon_0 * epsilon)
e = constants.e * unt.C
kb = constants.k * unt.J / unt.K
tau_q = 4.7 * unt.ps
mu = 200 * unt.m ** 2 / (unt.V * unt.s)
tau_tr = mu * m / e
tau_tr = tau_tr.cast_unit(unt.ps)
I_dc = 25 * unt.uA
W = 50 * unt.um
L = 100 * unt.um
w = 2 * constants.pi * 130 * 10 ** 9 * unt.Hz
P = 4 * 10 ** -3 * unt.W
hbar = constants.hbar * unt.J * unt.s
w_c = e * B / m
dos = m / (constants.pi * hbar ** 2)
Ef = n / dos
v_f = (2 * Ef / m) ** 0.5

# Calculation of tau inelastic here
kappa = 2 * constants.pi * e ** 2 * dos * k
tau_in_log_arg = kappa * v_f / (w_c * (w_c * tau_tr) ** 0.5)
tau_in_log_arg = tau_in_log_arg.cast_unit(unt.unitless)
tau_in_log_arg = tau_in_log_arg.number()
tau_in_log = np.log(tau_in_log_arg)
tau_in_inv = (((constants.pi * kb * T) ** 2) / (4 * constants.pi * Ef * hbar)) * tau_in_log
tau_in = tau_in_inv ** -1
tau_in = tau_in.cast_unit(unt.ps)
# tau_in = 200 * unt.ps
# Calculation of Q_dc here
Q_dc = 0
if I_dc.number() != 0:
    E_dc = (I_dc / W) * B / (n * e)
    E_dc = E_dc.cast_unit(unt.V / unt.m)
    Q_dc = 2 * (tau_in / tau_tr) * (e * E_dc * v_f / w_c) ** 2 * (constants.pi / (hbar * w_c)) ** 2
    Q_dc = Q_dc.cast_unit(unt.unitless)
Q_dc = 0
# Calculate P_w
P_w = 0
if P.number() != 0 and w.number() != 0:
    E_w = ((mu_0 / epsilon_0) ** 0.5 * P / (W * L)) ** 0.5
    E_w = E_w.cast_unit(unt.V / unt.m)
    #@TODO something wrong with E_w calue here. That's why taking E_dc instead of E_w for now
    P_w = (tau_in / tau_tr) * (e * E_dc * v_f / w) ** 2 * (w_c ** 2 + w ** 2) / ((w ** 2 - w_c ** 2) ** 2 * hbar ** 2)
    P_w = P_w.cast_unit(unt.unitless)


# Dingle factor
dingle = - constants.pi / (w_c * tau_q)
dingle = dingle.cast_unit(unt.unitless)
dingle = np.exp(dingle.number())


def f_osc_arg_get_E(arg):
    return arg * hbar * w_c + Ef


f_osc_arg = np.linspace(-5, 5, 1000)
dE = f_osc_arg_get_E(f_osc_arg[1]) - f_osc_arg_get_E(f_osc_arg[0])
dE = dE.cast_unit(meV)


def fermi_dirac(E, Ef, T):
    exp_arg = (E - Ef)/(kb*T)
    exp_arg = exp_arg.cast_unit(unt.unitless)
    exp_arg = exp_arg.number()
    return 1.0 / (np.exp(exp_arg) + 1)


def fermi_dirac_wrapper(E):
    return fermi_dirac(E, Ef, T)


def fermi_dirac_der_wrapper(E):
    fermi_dirac_der = (fermi_dirac(E + dE, Ef, T) - fermi_dirac(E, Ef, T)) / dE
    return fermi_dirac_der.number()


p_w_sin_arg = constants.pi * w / w_c
p_w_sin_arg = p_w_sin_arg.cast_unit(unt.unitless)
p_w_sin_arg = p_w_sin_arg.number()
f_osc_sin_prefix = (2 * constants.pi / (hbar * w_c)) * (P_w * 2 * constants.pi * w * np.sin(2 * p_w_sin_arg) / w_c + 4 * Q_dc) / \
          (1 + P_w * np.sin(p_w_sin_arg) ** 2 + Q_dc)
def f_osc_sin_arg(E):
    arg = f_osc_sin_prefix * E
    return arg.cast_unit(unt.unitless).number()


f_osc_arg = np.linspace(-5, 5, 1000)

f_0 = map(lambda arg: fermi_dirac_wrapper(f_osc_arg_get_E(arg)), f_osc_arg)
f_0_der = map(lambda arg: fermi_dirac_der_wrapper(f_osc_arg_get_E(arg)), f_osc_arg)
f_osc_sin = map(lambda arg: np.sin(f_osc_sin_arg(f_osc_arg_get_E(arg))), f_osc_arg)
f_osc_prefix = dingle * (hbar * w_c / (2 * constants.pi * dE.unit()))
f_osc_prefix = f_osc_prefix.cast_unit(unt.unitless).number()
f_osc = f_osc_prefix * np.multiply(f_0_der, f_osc_sin)
f = f_osc + f_0


plt.plot(f_osc_arg, f)
plt.plot(f_osc_arg, f_0)
plt.show()



