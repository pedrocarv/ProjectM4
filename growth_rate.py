import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

def find_growth_rate_1(time, energy, wpe):
    '''
    This is the first method for estimating growth rate
    '''
    # Fit a univariate spline to the array
    spls = UnivariateSpline(time, energy, k = 4, s = 0)
    # Get the points in time where the derivatives are zero
    roots = spls.derivative().roots()
    # Find the energy difference between a peak and through and divide by the time
    growth_rates = []
    for i in range(len(roots)-1):
        growth_rates.append(abs(spls(roots[i]) - spls(roots[i+1]))/(roots[i+1] - roots[i]))
    # Calculate the average growth rate
    avg_growth_rate = sum(growth_rates)/len(growth_rates)

    return avg_growth_rate/wpe

def find_growth_rate_2(time, energy):
    '''
    This is the second method for estimating growth rate
    '''
    # Fit a univariate spline to the array
    spls = UnivariateSpline(time, energy, k = 4, s = 0)
    # Get the points in time where the derivatives are zero
    roots = spls.derivative().roots()
    # Find the energy difference between a peak and through and divide by the time
    maxima = []
    maxima_loc = []
    for i in range(len(roots)-1):
        if spls(roots[i]) < spls(roots[i+1]):
            maxima.append(spls(roots[i+1]))
            maxima_loc.append(roots[i+1])

    z = np.poly1d(np.polyfit(maxima_loc, maxima, 3))
    data_fit = []
    for t in time:
        data_fit.append(z(t))

    def model_func(x, a, k, b):
        return a*np.exp(k*x) + b 

    opt, _ = curve_fit(model_func, time, data_fit)
    a, k, b = opt

    plt.plot(time, model_func(time, a, k, b))
    plt.plot(time, energy)
    plt.scatter(maxima_loc, maxima)
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    # Show the minor grid as well. Style it in very light gray as a thin,
    # dotted line.
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    # Make the minor ticks and gridlines show.
    plt.minorticks_on()
    plt.show()

    return k

def find_growth_rate_3(time, energy, wpe):
    '''
    This is the third method for estimating growth rate
    '''
    # Fit a univariate spline to the array
    spls = UnivariateSpline(time, energy, k = 4, s = 0)
    # Get the points in time where the derivatives are zero
    roots = spls.derivative().roots()
    # Find the energy difference between a peak and through and divide by the time
    growth_rates = []
    for i in range(len(roots)-1):
        growth_rates.append(np.sqrt(spls(roots[i])))
    # Calculate the average growth rate
    avg_growth_rate = sum(growth_rates)/len(growth_rates)

    return avg_growth_rate/wpe


data01 = np.loadtxt('Electrostatic energy - k = 0.1.txt')
data02 = np.loadtxt('Electrostatic energy - k = 0.2.txt')
data03 = np.loadtxt('Electrostatic energy - k = 0.3.txt')
data04 = np.loadtxt('Electrostatic energy - k = 0.4.txt')
data05 = np.loadtxt('Electrostatic energy - k = 0.5.txt')
data06 = np.loadtxt('Electrostatic energy - k = 0.6.txt')

time_euler = np.arange(0, 2000*2e-2, 2e-2)
time_euler_K06 = np.arange(0, 10000*3e-3, 3e-3)

wpe = 1.0

grk01 = find_growth_rate_3(time_euler, data01, wpe)
grk02 = find_growth_rate_3(time_euler, data02, wpe)
grk03 = find_growth_rate_3(time_euler, data03, wpe)
grk04 = find_growth_rate_3(time_euler, data04, wpe)
grk05 = find_growth_rate_3(time_euler, data05, wpe)
grk06 = find_growth_rate_3(time_euler_K06, data06, wpe)


growth_rates_numerical = [grk01, grk02, grk03, grk04, grk05, grk06]
kgrowth = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

omega_p = 1.0
Vo = 1.0

growth_rates_theoretical_imag = []
growth_rates_theoretical_real = []

for k in kgrowth:
    growth_rates_theoretical_imag.append(np.sqrt(k**2 * Vo**2 + omega_p**2 - omega_p * np.sqrt(omega_p**2 + 4 * k**2 * Vo**2, dtype=np.cdouble), dtype=np.cdouble))
    growth_rates_theoretical_real.append(np.sqrt(k**2 * Vo**2 + omega_p**2 + omega_p * np.sqrt(omega_p**2 + 4 * k**2 * Vo**2)))

#plt.plot(k, growth_rates_numerical, label = 'Numerical')
plt.plot(kgrowth, np.imag(growth_rates_theoretical_imag), color = 'black', label = 'Theoretical')
plt.plot(kgrowth, growth_rates_theoretical_real, color = 'black')
plt.scatter(kgrowth, growth_rates_numerical, color = 'blue', label = 'Numerical')
plt.legend()
plt.xlabel(r'$k \lambda_D$')
plt.ylabel(r'$\omega / \omega_p$')
# Show the major grid and style it sightly.
plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
# Show the minor grid as well. Style it in very light gray as a thin,
# dotted line.
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
# Make the minor ticks and gridlines show.
plt.minorticks_on()
plt.savefig('Growth rate.png', dpi = 600)
plt.show()



