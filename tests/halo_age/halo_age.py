import numpy as np
from pylab import *
import commah
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from astropy.cosmology import Planck13 as cosmo

def func(x,a,b,c):
    f = a+b*x+c*x**2
    return f

Mi_output = np.arange(8,14.2,0.2)
Mi_output = 10**Mi_output
z_output = np.arange(0,5,0.1)
tage = np.zeros(len(Mi_output))
i = 0
for Mi in Mi_output:
    output = commah.run('Planck13',zi=0,Mi=Mi,z=z_output)
    M_output = output['Mz'][0]
    f = interp1d(M_output, z_output)
    z_half = f(Mi/2)
    tage[i] = cosmo.age(0).value-cosmo.age(z_half).value
    i += 1

popt, pcov = curve_fit(func, np.log10(Mi_output), tage)
print(popt)

params = {
    "font.size": 12,
    "font.family": "Times",
    "text.usetex": True,
    "figure.figsize": (4, 3),
    "figure.subplot.left": 0.18,
    "figure.subplot.right": 0.95,
    "figure.subplot.bottom": 0.18,
    "figure.subplot.top": 0.95,
    "figure.subplot.wspace": 0.25,
    "figure.subplot.hspace": 0.25,
    "lines.markersize": 6,
    "lines.linewidth": 1.5,
    "figure.max_open_warning": 0,
}
rcParams.update(params)
#######################
# Plot the density profile
figure()
ax = plt.subplot(1, 1, 1)
grid(True)

plot(Mi_output, tage, '-', color='black', label='Data')
plot(Mi_output, func(np.log10(Mi_output),*popt), '--', label='Fit')

text(2e8,11.5,'$f=a+bx+cx^{2}$, $a=7.657,~b=0.841,~c=-0.063$',fontsize=9)
ylabel('Halo age [Gyr]')
xlabel('$M_{200}$ [M$_{\odot}$]')
xscale('log')
axis([1e8,1e14,6,12])
legend(loc="lower left", labelspacing=0.2, handlelength=1.5, handletextpad=0.4, frameon=False)
ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
plt.savefig('halo_age.png',dpi=200)