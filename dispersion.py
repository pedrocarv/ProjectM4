import numpy as np
import matplotlib.pyplot as plt

qe   = 1.60217646e-19
me   = 9.10938188e-31
mp   = 1.67262158e-27
KB   = 1.38000000e-23
mu0  = 4*np.pi*1e-7
lux  = 299792458
eps0 = 1.0/(mu0*lux*lux)
ev2k = qe/KB

# Define the values of Vo and omega_p
Te = 1.0*ev2k
Vo = np.sqrt(KB*Te/me)
ne = 1e16
omega_p = np.sqrt(ne*qe**2/(eps0*me))
debye = np.sqrt(eps0*KB*Te/(ne*qe**2))

# Define a range of values for k
k_range = np.linspace(0, 1.6/debye, 100)

# Initialize empty arrays to hold the real and imaginary parts of the equation
omega_real = []
omega_imag = []

kdebye = [k/debye for k in k_range]

# Loop over the values of k
for k in k_range:
    # Calculate the real and imaginary parts of the equation
    omega_real.append(np.sqrt(k**2 * Vo**2 + omega_p**2 + omega_p * np.sqrt(omega_p**2 + 4 * k**2 * Vo**2))/omega_p)
    omega_imag.append(np.sqrt(k**2 * Vo**2 + omega_p**2 - omega_p * np.sqrt(omega_p**2 + 4 * k**2 * Vo**2, dtype=np.cdouble), dtype=np.cdouble)/omega_p)

# Plot the real and imaginary parts of the equation
plt.plot(k_range*debye, omega_real, color = 'black', label = r'Re($\omega$)')
# plt.plot(k_range*debye, np.sqrt(np.real(omega_imag)**2 + np.imag(omega_imag)**2), color = 'blue', label = r'Im($\omega$)')
plt.plot(k_range*debye, np.imag(omega_imag), color = 'blue', label = r'Im($\omega$)')
plt.plot(k_range*debye, np.real(omega_imag), color = 'black')
plt.xlabel(r'$k \lambda_D$')
plt.ylabel(r'$\omega/\omega_p$')
plt.ylim(bottom=0)
# Show the major grid and style it slightly.
plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
# Show the minor grid as well. Style it in very light gray as a thin,
# dotted line.
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
# Make the minor ticks and gridlines show.
plt.minorticks_on()
plt.legend()
plt.title('Real and imaginary parts of the dispersion relation')
plt.savefig('Dispersion relation.png', dpi=900)

plt.show()