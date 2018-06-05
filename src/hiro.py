import numpy as np
import unum.units as unt
import unum.core
import scipy.constants as constants
import matplotlib.pyplot as plt
import os.path
import scipy.special as spc

meV = unum.core.new_unit('meV', 10 ** -3 * unt.eV, 'meV')
hbar = constants.hbar * unt.J * unt.s
T = 1.6 * unt.K
n = 8.3 * 10 ** 15 * unt.m ** -2
a = 200 * unt.nm
m = 0.068 * constants.m_e * unt.kg
e = constants.e * unt.C
kb = constants.k * unt.J / unt.K
V_0 = 0.35 * meV
I_dc = 80 * unt.uA
W = 50 * unt.um
mu = 215 * unt.m ** 2 / (unt.V * unt.s)
R_0 = 1 / (e * n * mu)
R_0 = R_0.cast_unit(unt.OHM)

dos = m / (constants.pi * hbar ** 2)
Ef = n / dos
v_f = (2 * Ef / m) ** 0.5


def read_from_csv(filename, delimeter=','):
    data = np.genfromtxt(filename, delimiter=',')
    data = np.hsplit(data, 2)
    data[0] = np.reshape(data[0], len(data[0]))
    data[1] = np.reshape(data[1], len(data[1]))
    return (data[0], data[1])


data = read_from_csv('data/hiro/original_data.csv')
data = (data[0][574:3149], data[1][574:3149])

def R_c(B):
    return (2 * constants.pi * n) ** 0.5 * (hbar / (e * abs(B)))

cos_arg_prefix = 2 * constants.pi / a
V_B_prefix = (a/(constants.pi ** 2)) ** 0.5
def V_B_per_V_0_wrapper(B):
    cos_arg = cos_arg_prefix * R_c(B) - constants.pi / 4
    cos_arg = cos_arg.cast_unit(unt.unitless).number()
    V_B = V_B_prefix * R_c(B) ** -0.5 * np.cos(cos_arg)
    return abs(V_B.cast_unit(unt.unitless).number())


temp_refactor_prefix = (2 * constants.pi ** 2 * kb * T / hbar)
w_c_prefix = e / m
def R_sdh_per_R_0_envelop(B, tau_q, V_0, A_SdH=0.85):
    w_c = w_c_prefix * B
    temp_prefactor_arg = temp_refactor_prefix / w_c
    temp_prefactor_arg = temp_prefactor_arg.cast_unit(unt.unitless).number()
    temp_prefactor = temp_prefactor_arg / np.sinh(temp_prefactor_arg)
    V_B = V_0 * V_B_per_V_0_wrapper(B)
    bessel_arg = 2 * constants.pi * V_B / (hbar * w_c)
    bessel_arg = bessel_arg.cast_unit(unt.unitless).number()
    dingle_arg = -constants.pi / (w_c * tau_q)
    dingle_arg = dingle_arg.cast_unit(unt.unitless).number()
    return 4 * A_SdH * temp_prefactor * spc.j0(bessel_arg) * np.exp(dingle_arg)

def R_sdh_per_R_0(B, tau_q, V_0, A_SdH=0.85):
    envelop = R_sdh_per_R_0_envelop(B, tau_q, V_0, A_SdH)
    cos_arg = 2 * constants.pi * Ef / (hbar * w_c_prefix * B)
    cos_arg = cos_arg.cast_unit(unt.unitless).number()
    return envelop * np.cos(cos_arg)

E_dc_prefix = (I_dc / W) / (n * e)
def r_hiro_per_R_0(B, tau_q, A_hiro=0.85):
    w_c = w_c_prefix * B
    dingle_arg = -2 * constants.pi / (w_c * tau_q)
    dingle_arg = dingle_arg.cast_unit(unt.unitless).number()
    cos_arg = 4 * constants.pi * R_c(B) * e * E_dc_prefix * B / (hbar * w_c)
    cos_arg = cos_arg.cast_unit(unt.unitless).number()
    return A_hiro * np.exp(dingle_arg) * np.cos(cos_arg)

#
# V_B = map(V_B_per_V_0_wrapper, data_cut[0]* unt.T)
# plt.plot(data_cut[0], V_B)
inv_data = (np.flip(data[0] ** -1, 0), np.flip(data[1], 0))
interp_arg = np.linspace(inv_data[0][0], inv_data[0][-1], 4000)
interp_val = np.interp(interp_arg, inv_data[0], inv_data[1])
interp_data = (interp_arg, interp_val)

def collect_dots(subplot, filename):
    fig = subplot.figure
    if os.path.isfile(filename):
        data = np.genfromtxt(filename, delimiter=',')
        data = np.hsplit(data, 2)
    else:
        data = (np.array([]), np.array([]))
    line, = subplot.plot(data[0], data[1], marker='o', linestyle='')


    def onclick(event):
        xdata = line.get_xdata(True)
        ydata = line.get_ydata(True)
        if event.button == 2:
            xdata = np.append(xdata, event.xdata)
            ydata = np.append(ydata, event.ydata)
        elif event.button == 3:
            xdata = xdata[:-1]
            ydata = ydata[:-1]
        line.set_xdata(xdata)
        line.set_ydata(ydata)
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)


    def handle_close(event):
        xdata = line.get_xdata(True)
        ydata = line.get_ydata(True)
        xdata = np.reshape(xdata, (len(xdata),1))
        ydata = np.reshape(ydata, (len(ydata),1))
        data = np.hstack((xdata, ydata))
        np.savetxt(filename, data, delimiter=',')
        line.remove()

    fig.canvas.mpl_connect('close_event', handle_close)
    plt.show()

# Subtract monotonic part.
maximums_data_file = 'data/hiro/maximums.csv'
minimums_data_file = 'data/hiro/minimums.csv'
# fig = plt.figure()
# subplot = fig.add_subplot(111)
# subplot.plot(interp_data[0], interp_data[1])
# collect_dots(subplot, maximums_data_file)

# fig = plt.figure()
# subplot = fig.add_subplot(111)
# subplot.plot(interp_data[0], interp_data[1])
# collect_dots(subplot, minimums_data_file)

data = read_from_csv(maximums_data_file)
p = np.polyfit(data[0], data[1], 12)
max_data_interp = np.polyval(p, interp_arg)
data = read_from_csv(minimums_data_file)
p = np.polyfit(data[0], data[1], 12)
min_data_interp = np.polyval(p, interp_arg)
average_data_interp = (max_data_interp + min_data_interp) / 2
interp_val_average = interp_val - average_data_interp
# fig = plt.figure()
# subplot = fig.add_subplot(111)
# subplot.plot(interp_arg, interp_val)
# subplot.plot(interp_arg, max_data_interp)
# subplot.plot(interp_arg, min_data_interp)
# plt.show()
# fig = plt.figure()
# subplot = fig.add_subplot(111)
# subplot.plot(interp_arg, interp_val_average)
# plt.show()

# Extract CO
freq = np.fft.rfft(interp_val_average)
freq[:3] = 0
freq[50:] = 0
# fig = plt.figure()
# subplot = fig.add_subplot(111)
# subplot.plot(freq)
# plt.show()
interp_val_co = np.fft.irfft(freq, len(interp_arg))
# fig = plt.figure()
# subplot = fig.add_subplot(111)
# subplot.plot(interp_arg, interp_val_co)
# plt.show()

# Extract SdH
# fig = plt.figure()
# subplot = fig.add_subplot(111)
# subplot.plot(interp_arg[:1150], interp_val_average[:1150])
# plt.show()
freq = np.fft.rfft(interp_val_average[:1150])
freq[:42] = 0
freq[95:] = 0
# fig = plt.figure()
# subplot = fig.add_subplot(111)
# subplot.plot(freq)
# plt.show()
interp_val_sdh = np.fft.irfft(freq, len(interp_val_average[:1150]))
# fig = plt.figure()
# subplot = fig.add_subplot(111)
# subplot.plot(interp_arg[:1150], interp_val_sdh)
# plt.show()

# Calc SdH envelop
# interp_val_sdg_calc_envelop = map(lambda arg: R_sdh_per_R_0_envelop(arg ** -1 * unt.T, 2.3 * unt.ps, 0 * unt.eV), interp_arg[:1150])
# fig = plt.figure()
# subplot = fig.add_subplot(111)
# subplot.plot(interp_arg[:1150], interp_val_sdg_calc_envelop)
# subplot.plot(interp_arg[:1150], interp_val_sdh)
# plt.show()

# Calc SdH
# interp_val_sdg_calc = map(lambda arg: R_sdh_per_R_0(arg ** -1 * unt.T, 2.3 * unt.ps, 0.7 * meV), interp_arg[:1150])
# fig = plt.figure()
# subplot = fig.add_subplot(111)
# subplot.plot(interp_arg[:1150], interp_val_sdg_calc)
# subplot.plot(interp_arg[:1150], interp_val_sdg_calc_envelop)
# plt.show()

# DC PART
data = read_from_csv('data/hiro/dc_data.csv')
interp_val_dc = np.interp(interp_arg, data[0], data[1])
# fig = plt.figure()
# subplot = fig.add_subplot(111)
# subplot.plot(interp_arg, interp_val_dc)
# plt.show()

# Subtract monotonic part.
maximums_data_file = 'data/hiro/maximums_dc.csv'
minimums_data_file = 'data/hiro/minimums_dc.csv'
# fig = plt.figure()
# subplot = fig.add_subplot(111)
# subplot.plot(interp_arg, interp_val_dc)
# collect_dots(subplot, maximums_data_file)
#
# fig = plt.figure()
# subplot = fig.add_subplot(111)
# subplot.plot(interp_arg, interp_val_dc)
# collect_dots(subplot, minimums_data_file)
data = read_from_csv(maximums_data_file)
p = np.polyfit(data[0], data[1], 15)
max_data_interp = np.polyval(p, interp_arg)
data = read_from_csv(minimums_data_file)
p = np.polyfit(data[0], data[1], 15)
min_data_interp = np.polyval(p, interp_arg)
average_data_interp = (max_data_interp + min_data_interp) / 2
interp_val_average_dc = interp_val_dc - average_data_interp
# fig = plt.figure()
# subplot = fig.add_subplot(111)
# subplot.plot(interp_arg, interp_val_dc)
# subplot.plot(interp_arg, max_data_interp)
# subplot.plot(interp_arg, min_data_interp)
# plt.show()
# fig = plt.figure()
# subplot = fig.add_subplot(111)
# subplot.plot(interp_arg, interp_val_average_dc)
# plt.show()

# Extract CO
# freq = np.fft.rfft(interp_val_average_dc)
# freq[:10] = 0
# freq[30:] = 0
# fig = plt.figure()
# subplot = fig.add_subplot(111)
# subplot.plot(freq)
# plt.show()
# interp_val_co_dc = np.fft.irfft(freq, len(interp_arg))
# fig = plt.figure()
# subplot = fig.add_subplot(111)
# subplot.plot(interp_arg, interp_val_co)
# subplot.plot(interp_arg, interp_val_co_dc)
# plt.show()

# Calc HIRO
interp_val_dc_calc = map(lambda arg: r_hiro_per_R_0(arg ** -1 * unt.T, 2.3 * unt.ps), interp_arg)
fig = plt.figure()
subplot = fig.add_subplot(111)
subplot.plot(interp_arg, interp_val_dc_calc)
plt.show()