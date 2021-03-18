import numpy as np
from pylab import *
from scipy.optimize import fsolve
from scipy.optimize import root
from scipy.interpolate import interp1d
from functions import find_r1, M_isothermal, rho_isothermal, find_nfw, rho_nfw
from functions import rho_joint_profiles

def calcM200(R200, z):
    Om = 0.307
    Ol = 1.0 - Om
    # Input R200 [kpc]
    rhocrit = 2.775e11 * 0.6777 ** 2 / (1e3) ** 3  # Msun/kpc^3
    rhocrit *= (Om * (1. + z) ** 3 + Ol)
    M200 = (4. * np.pi / 3.) * 200. * rhocrit * R200 ** 3
    return np.log10(M200)


def calcR200(x, rs, Ms, z):
    Om = 0.307
    Ol = 1.0 - Om
    rhocrit = 2.775e11 * 0.6777 ** 2 / (1e3) ** 3  # Msun/kpc^3
    rhocrit *= (Om * (1. + z) ** 3 + Ol)

    # Input rhos [Msun/kpc^3], rs [kpc]
    c = x / rs
    Yc = np.log(1. + c) - c / (1. + c)
    Y1 = np.log(2.) - 0.5
    M200_1 = (4. * np.pi / 3.) * 200. * rhocrit * x ** 3  # Msun
    M200_1 = np.log10(M200_1)
    M200_2 = Ms + np.log10( Yc / Y1 )
    f = M200_1 - M200_2
    return f


def calc_Ms(rs, rhos):
    r = np.arange(-3, 3, 0.001)
    r = 10 ** r
    deltar = r[1:] - r[:-1]
    deltar = np.append(deltar, deltar[-1])
    rho = rhos * rho_nfw(r / rs)
    mass = 4. * np.pi * np.cumsum(rho * r ** 2 * deltar)
    mass = np.log10(mass)
    interpolate = interp1d(r, mass)
    Ms = interpolate(rs)
    return Ms

def calculate_additional_params(rs, rhos):
    Ms = calc_Ms(rs, rhos)  # Msun
    R200 = fsolve(calcR200, 100., args=(rs, Ms, 0.))
    M200 = calcM200(R200, 0.)
    c200 = R200 / rs
    return c200, M200

def model_params(N0, v0, sigma0):
    Msun_in_cgs = 1.98848e33
    kpc_in_cgs = 3.08567758e21

    t_age = 7.5  # Gyr - assuming constant halo age
    t_age_cgs = t_age * 1e9 * 365.24 * 24 * 3600  # sec
    v0_cgs = v0 * 1e5  # cm/s
    rho0_cgs = N0 / (t_age_cgs * (4. / np.sqrt(np.pi)) * v0_cgs * sigma0)  # g / cm^3
    rho0 = rho0_cgs * kpc_in_cgs ** 3 / Msun_in_cgs  # Msun / kpc^3

    G = 4.3e-6  # kpc km^2 Msun^-1 s^-2
    r0 = v0 ** 2 / (4. * np.pi * G * rho0)
    r0 = np.sqrt(r0)  # kpc

    sol = fsolve(find_r1, 20, args=(rho0_cgs, v0_cgs, sigma0, t_age_cgs))
    r1 = sol[0] * r0  # kpc

    M1 = M_isothermal(r1, r0, rho0)  # Msun
    rho1 = rho0 * rho_isothermal(r1 / r0)  # Msun /kpc^3

    sol = root(find_nfw, [30, np.log10(rho1)], args=(r1, np.log10(rho1), np.log10(M1)),
               method='hybr', tol=1e-4)

    rs = sol.x[0]
    rhos = 10 ** sol.x[1]
    return r1, rho1, r0, rho0, rs, rhos

def plot_solution(xdata, ydata, yerrdata, soln, output_name, output_file):

    # Extract solution:
    N0, v0, sigma0 = soln.x
    N0 = 10 ** N0
    v0 = 10 ** v0
    sigma0 = 10 ** sigma0

    # Calculate additional params:
    r1, rho1, r0, rho0, rs, rhos = model_params(N0, v0, sigma0)
    c200, M200 = calculate_additional_params(rs, rhos)

    # Output best-fit model params to file:
    header = "Maximum likelihood estimates \n"
    header += "Units: radius [kpc], density [log10Msun/kpc3], mass [log10Msun]."
    names = np.array(['r1', 'rho1', 'r0','rho0', 'rs', 'rhos',
                      'c200', 'M200', 'N0', 'v0', 'sigma0'])

    floats = np.array([r1, np.log10(rho1), r0, np.log10(rho0),
                       rs, np.log10(rhos), c200, M200,
                       N0, v0, sigma0], dtype="object")
    ab = np.zeros(names.size, dtype=[('var1', 'U6'), ('var2', float)])
    ab['var1'] = names
    ab['var2'] = floats
    np.savetxt(output_file, ab, fmt="%6s %10.3f", header=header)

    # Plotting best-fit model:
    xrange = np.arange(-1, 3, 0.1)
    xrange = 10 ** xrange
    model = rho_joint_profiles(xrange, r1, r0, rho0, rs, rhos)
    model_nfw = rhos * rho_nfw(xrange / rs)
    model_iso = rho0 * rho_isothermal(xrange / r0)

    print("SIDM model:")
    print("r1 = {0:.2f}".format(r1))
    print("r0 = {0:.2f}".format(r0))
    print("log10rho0 = {0:.2f}".format(np.log10(rho0)))
    print("rs = {0:.2f}".format(rs))
    print("log10rhos = {0:.2f}".format(np.log10(rhos)))

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

    plot(xdata, ydata, '-', lw=2, color='black', label='Data')
    plt.fill_between(xdata, ydata - yerrdata / 2, ydata + yerrdata / 2, alpha=0.4, color='grey')

    plot(xrange, model_iso, '--',color='tab:red', label='Isothermal-profile fit')
    plot(xrange, model_nfw, '--',color='tab:orange', label='NFW-profile fit')
    plot(xrange, model, '-',color='tab:blue', label='SIDM-profile fit')
    plot(r1, [rho1], 'o')

    xscale('log')
    yscale('log')
    ylabel(r'Density profile [M$_{\odot}$/kpc$^{3}$]')
    xlabel(r'Radius [kpc]')
    plt.legend(loc="lower left", labelspacing=0.2, handlelength=1.5, handletextpad=0.4, frameon=False)
    axis([1e-1, 5e2, 1e2, 5e8])
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig(output_name, dpi=200)
    return


def plot_isothermal_fit(xdata, ydata, yerrdata, N0, v0, sigma0, output_name):
    Msun_in_cgs = 1.98848e33
    kpc_in_cgs = 3.08567758e21

    t_age = 7.5  # Gyr - assuming constant halo age
    t_age_cgs = t_age * 1e9 * 365.24 * 24 * 3600  # sec
    v0_cgs = v0 * 1e5  # cm/s
    rho0_cgs = N0 / (t_age_cgs * (4. / np.sqrt(np.pi)) * v0_cgs * sigma0)  # g / cm^3
    rho0 = rho0_cgs * kpc_in_cgs ** 3 / Msun_in_cgs  # Msun / kpc^3

    G = 4.3e-6  # kpc km^2 Msun^-1 s^-2
    r0 = v0 ** 2 / (4. * np.pi * G * rho0)
    r0 = np.sqrt(r0)  # kpc

    xrange = np.arange(-1,3,0.1)
    xrange = 10**xrange
    model = rho_isothermal(xrange / r0)
    model = rho0 * model

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

    plot(xdata, ydata, '-', label='Data')
    plt.fill_between(xdata, ydata - yerrdata / 2, ydata + yerrdata / 2, alpha=0.4)

    plot(xrange, model, '--', label='Isothermal-profile fit')

    xscale('log')
    yscale('log')
    ylabel(r'Density profile [M$_{\odot}$/kpc$^{3}$]')
    xlabel(r'Radius [kpc]')
    plt.legend(loc=[0.4, 0.8], labelspacing=0.2, handlelength=1.5, handletextpad=0.4, frameon=False)
    axis([1e-1, 5e2, 1e2, 5e8])
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig(output_name, dpi=200)
    return

