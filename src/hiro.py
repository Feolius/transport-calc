import numpy as np
import unum.units as unt
import unum.core
import scipy.constants as constants
import matplotlib.pyplot as plt
import os.path
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


def read_from_csv(filename, delimeter=','):
    data = np.genfromtxt(filename, delimiter=',')
    data = np.hsplit(data, 2)
    data[0] = np.reshape(data[0], len(data[0]))
    data[1] = np.reshape(data[1], len(data[1]))
    return (data[0], data[1])


data = read_from_csv('data/hiro/original_data.csv')
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
inv_data = (np.flip(data_cut[0] ** -1, 0), np.flip(data_cut[1], 0))
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
p = np.polyfit(data[0], data[1], 8)
max_data_interp = np.polyval(p, interp_arg)
data = read_from_csv(minimums_data_file)
p = np.polyfit(data[0], data[1], 8)
min_data_interp = np.polyval(p, interp_arg)
average_data_interp = (max_data_interp + min_data_interp) / 2
interp_val_average = interp_val - average_data_interp
# fig = plt.figure()
# subplot = fig.add_subplot(111)
# subplot.plot(interp_arg, average_data_interp)
# subplot.plot(interp_arg, max_data_interp)
# subplot.plot(interp_arg, min_data_interp)
# plt.show()
# fig = plt.figure()
# subplot = fig.add_subplot(111)
# subplot.plot(interp_arg, interp_val_average)

freq = np.fft.rfft(interp_val_average)
freq[:7] = 0
freq[23:] = 0
# fig = plt.figure()
# subplot = fig.add_subplot(111)
# subplot.plot(freq)
# plt.show()
interp_val_average = np.fft.irfft(freq, len(interp_arg))

data = read_from_csv('data/hiro/dc_data.csv')
interp_val_dc = np.interp(interp_arg, data[0], data[1])
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
freq = np.fft.rfft(interp_val_average_dc)
freq[:9] = 0
freq[20:] = 0
# fig = plt.figure()
# subplot = fig.add_subplot(111)
# subplot.plot(freq)
# plt.show()
interp_val_average_dc = np.fft.irfft(freq, len(interp_arg))
fig = plt.figure()
subplot = fig.add_subplot(111)
subplot.plot(interp_arg, interp_val_average)
subplot.plot(interp_arg, interp_val_average_dc)
plt.show()
