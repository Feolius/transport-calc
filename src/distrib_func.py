import numpy as np
from unum.units import *
from unum.core import new_unit
import scipy.constants as constants
import matplotlib.pyplot as plt

meV = new_unit('meV', 10 ** -3 * eV, 'meV')


def fermi_dirac(E, Ef, T):
    E = E.cast_unit(J)
    Ef = Ef.cast_unit(J)
    T = T.cast_unit(K)
    k = constants.k * J / K
    exp_arg = (E - Ef)/(k*T)
    exp_arg = exp_arg.number()
    return 1.0 / (np.exp(exp_arg) + 1)



Ef = 10.0 * meV
T = 4.2 * K
x = np.linspace(0, 2 * Ef.number(), 1000) * meV

def fermi_dirac_vector(E):
    return fermi_dirac(E, Ef, T)

y = map(fermi_dirac_vector, x)

x = map(lambda x: x.number(), x)
plt.plot(x, y)
plt.show()


