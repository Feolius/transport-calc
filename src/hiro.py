import numpy as np
import unum.units as unt
import unum.core
import scipy.constants as constants
import matplotlib.pyplot as plt
from scipy.fftpack import rfft, irfft, fftfreq

meV = unum.core.new_unit('meV', 10 ** -3 * unt.eV, 'meV')
hbar = constants.hbar * unt.J * unt.s
T = 1.6 * unt.K
n = 8.3 * 10 ** 15 * unt.m ** -2
a = 200 * unt.nm
m = 0.068 * constants.m_e * unt.kg
e = constants.e * unt.C
kb = constants.k * unt.J / unt.K
V_0 = 0.35 * meV

dos = m / (constants.pi * hbar ** 2)
Ef = n / dos
v_f = (2 * Ef / m) ** 0.5
x = np.arange(16.0).reshape(8, 2)
data = np.genfromtxt('data/hiro/original_data.csv', delimiter=',')
data = np.hsplit(data, 2)
data[0] = np.reshape(data[0], len(data[0]))
data[1] = np.reshape(data[1], len(data[1]))
data_cut = (data[0][574:3149], data[1][574:3149])

def R_c(B):
    return (2 * constants.pi * n) ** 0.5 * (hbar / (e * abs(B)))

cos_arg_prefix = 2 * constants.pi / a
V_B_prefix = (a/(constants.pi ** 2)) ** 0.5
def V_B_per_V_0_wrapper(B):
    cos_arg = cos_arg_prefix * R_c(B)- constants.pi / 4
    cos_arg = cos_arg.cast_unit(unt.unitless).number()
    V_B = V_B_prefix * R_c(B) ** -0.5 * np.cos(cos_arg)
    return abs(V_B.cast_unit(unt.unitless).number())
#
# V_B = map(V_B_per_V_0_wrapper, data_cut[0]* unt.T)
# plt.plot(data_cut[0], V_B)
inv_data = data_cut[0] ** -1
arg = np.linspace(inv_data[0], inv_data[-1], 3000)
trf = np.interp(arg, inv_data, data_cut[1])
plt.plot(inv_data, data_cut[1], 'o')
plt.plot(arg, trf)
# freq = rfft(data_cut[1])
# plt.plot(freq)
plt.show()