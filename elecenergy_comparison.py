import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

data1 = np.loadtxt('Electrostatic energy - 10000.txt')
data2 = np.loadtxt('Electrostatic energy - 50000.txt')
data3 = np.loadtxt('Electrostatic energy - 100000.txt')
data4 = np.loadtxt('Electrostatic energy - 1000000.txt')
data5 = np.loadtxt('Electrostatic energy - euler.txt')

time_PIC = np.arange(0, 200*5e-11, 5e-11)
time_euler = np.arange(0, 2000*2e-2, 2e-2)

wpe = 5641460274.996249

plt.semilogy(time_PIC*wpe, data1, label=r'PIC - $10^4$ particles')
plt.semilogy(time_PIC*wpe, data2, label=r'PIC - $5 \times 10^4$ particles')
plt.semilogy(time_PIC*wpe, data3, label=r'PIC - $10^5$ particles')
plt.semilogy(time_PIC*wpe, data4, label=r'PIC - $10^6$ particles')
plt.semilogy(time_euler, data5, label=r'Eulerian')
plt.legend()
plt.xlim([0,20])
plt.xlabel(r'$t \omega_p$')
plt.ylabel('Electrostatic energy')
# Show the major grid and style it sightly.
plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
# Show the minor grid as well. Style it in very light gray as a thin,
# dotted line.
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
# Make the minor ticks and gridlines show.
plt.minorticks_on()
plt.savefig('Electrostatic energy comparison.png', dpi = 600)
plt.show()


