import numpy as np
from pylab import *
#from scipy.optimize import fsolve
#from scipy.interpolate import interp1d

def read_data(file):
    data = np.loadtxt(file)
    # Remove potential zeros before we begin
    nozero = data[:, 1] > 0
    data = data[nozero, :]
    min_rho = np.log10(data[:, 1]) >= 3.
    data = data[min_rho, :]

    # Next:
    x = data[:, 0]
    y = data[:, 1]
    yerr = np.ones(len(x)) * 0.5
    errorbar = np.ones(len(x)) * 5
    return x, y, yerr

for i in range(0,10):
    output_name = 'Profile_halos_M11.0_L006N188_%i_hydro_comparison.png'%i

    file = '../Individual_sample/Profile_halos_M11.0_DML006N188_SigmaConstant01_%i.txt'%i
    x, y, yerr = read_data(file)

    file = '../Individual_sample/Profile_halos_M11.0_RefL006N188_SigmaConstant01_%i.txt'%i
    #file = '../../L006N188_SigmaConstant10/Individual_sample/Profile_halos_M11.0_DML006N188_SigmaConstant10_%i.txt'%i
    x2, y2, yerr2 = read_data(file)


    # Plot parameters
    params = {
        "font.size": 12,
        "font.family":"Times",
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

    plot(x, y, '-', lw=2, color='black', label='Sigma 1/DMONLY')
    plt.fill_between(x, y - yerr / 2, y + yerr / 2, alpha=0.4, color='grey')

    plot(x2, y2, '-', lw=2, color='tab:blue', label='Sigma 1/Ref')
    plt.fill_between(x2, y2 - yerr2 / 2, y2 + yerr2 / 2, alpha=0.4, color='tab:blue')

    xscale('log')
    yscale('log')
    ylabel(r'Density profile [M$_{\odot}$/kpc$^{3}$]')
    xlabel(r'Radius [kpc]')
    plt.legend(loc="lower left", labelspacing=0.2, handlelength=1.5, handletextpad=0.4, frameon=False)
    axis([5e-1, 1e2, 1e4, 1e8])
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig(output_name, dpi=200)

