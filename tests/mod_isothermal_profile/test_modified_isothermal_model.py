import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import curve_fit
from pylab import *
from scipy.optimize import fsolve
from scipy.optimize import minimize

def read_data_single_halo(file):

    data = np.loadtxt(file)

    # Remove potential zeros before we begin
    nozero = data[:,1] > 0
    data = data[nozero,:]
    min_rho = np.log10(data[:,1]) >= 3.
    data = data[min_rho, :]

    # Next:
    x = data[:,0]
    y = np.log10(data[:,1])
    yerr = np.ones(len(x)) * 0.2
    errorbar = np.ones(len(x)) * 0.2

    z = data[:,2]
    zerrorbar = np.ones(len(x)) * 0.2


    return x, y, yerr, errorbar, z, zerrorbar

def read_data(file):

    data = np.loadtxt(file)

    # Remove potential zeros before we begin
    nozero = data[:,2] > 0
    data = data[nozero,:]
    min_rho = np.log10(data[:,1]) >= 3.
    data = data[min_rho, :]

    # Next:
    x = data[:,0]
    y = np.log10(data[:,1])
    yerr = np.sqrt( (y-np.log10(data[:,2]))**2 + (np.log10(data[:,3])-y)**2)
    errorbar = np.log10( np.sqrt( (data[:,1]-data[:,2])**2 + (data[:,3]-data[:,1])**2) )

    z = np.log10(data[:,4])
    zerrorbar = np.sqrt( (data[:,4]-data[:,6])**2 + (data[:,4]-data[:,5])**2)

    return x, y, yerr, errorbar, z, zerrorbar

def velocity_dispersion(x, n):
    """
        Args: x = r/rs
        Returns: y = sigma_v(r)/sigma_v0
    """
    f = x**(n/2)
    return f

def diff_isothermal_equation(f,x,n):
    """
    Differential equation that describes the isothermal profile
    """
    y, z = f
    dfdx = [z,-(n+2)*(1./x)*z-n*(n+1)*(1./x**2)-(1./x**n)*np.exp(y)]
    return dfdx

def cross_section(x, sigma0, w0):
    f = 4 * sigma0 * w0 ** 4 / x ** 4
    f *= (2 * np.log(1. + 0.5 * x ** 2 / w0 ** 2) - np.log(1. + x ** 2 / w0 ** 2))
    # f = sigma0 / (1 + x**2 / w0**2)**2
    return f

def integrand(x, sigma0, w0, v):
    """
    Momentum transfer cross section
    """
    sigma = cross_section(x, sigma0, w0)
    y = x / v
    f = sigma * y**3 * np.exp(-y**2 / 4.)
    return f

def sigma_vel_weight(x, ns0, v0, sigma0, w0):
    """
    Function : <sigma/m v>(r)
    It determines the scattering rate computed by integrating
    the cross section, sigma/m, weithed by the relative velocity v
    over a Maxwell-Boltzmann distribution:
    <sigma/m v> = 1/[2 sqrt(pi) vrms^3] int_0^infty sigma/m(v) v^3 exp(-v^2/4 vrms^2) dv
                = 4/qrt(pi) vrms sigma/m (if sigma/m is constant!)
    """
    v = velocity_dispersion(x, ns0)
    v *= v0

    integral = quad(integrand, 0, np.inf, args=(sigma0, w0, v))[0]
    weight_function = v / (2 * np.sqrt(np.pi))
    weight_function *= integral
    return weight_function

def calculate_log_v0(r0, rho0):
    G = 4.3e-6  # kpc km^2 Msun^-1 s^-2
    v0 = r0**2 * 4. * np.pi * G * rho0
    v0 = np.sqrt(v0)
    return np.log10(v0) #km/s

def calculate_log_N0(rho0, v0, ns0, sigma0, w0):
    """
    Calculates the number of scattering events within r0.
    """
    Msun_in_cgs = 1.98848e33
    kpc_in_cgs = 3.08567758e21

    t_age = 7.5 # Gyr - assuming constant halo age
    t_age_cgs = t_age * 1e9 * 365.24 * 24 * 3600 # sec
    rho0_cgs = rho0 * Msun_in_cgs / kpc_in_cgs**3

    sigma_v_weight = sigma_vel_weight(1, ns0, v0, sigma0, w0) # km/s cm^2/g
    sigma_v_weight *= 1e5 # cm^3/s/g

    N0 = t_age_cgs * sigma_v_weight * rho0_cgs
    return np.log10(N0)

def fit_isothermal_model(xdata, a, b, n):

    xrange = np.arange(-5, 5, 0.01)
    xrange = 10**xrange
    xrange = xrange / a
    y0 = [0, 0]
    sol = odeint(diff_isothermal_equation, y0, xrange, args=(n,))
    yrange = np.exp(sol[:, 0])
    yrange = np.log10(yrange)
    finterpolate = interp1d(xrange, yrange)
    x = xdata / a
    ydata = finterpolate(x)
    f = b + ydata

    return f

def rho_isothermal(x,n):
    """
        Args: x = r/r0
        Returns: y = rho(r)/rho0
    """
    xrange = np.arange(-4, 5, 0.01)
    xrange = 10 ** xrange
    y0 = [0, 0]
    sol = odeint(diff_isothermal_equation, y0, xrange, args=(n,))
    yrange = np.exp(sol[:, 0])
    finterpolate = interp1d(xrange, yrange)

    if np.isscalar(x):
        if x <= 1e-4:
            y = 0
        elif x >= 1e5:
            y = finterpolate(1e5)
        else:
            y = finterpolate(x)
    else:
        y = np.zeros(len(x))
        in_range = np.logical_and(x > 1e-4, x <= 1e4)
        y[in_range] = finterpolate(x[in_range])

    return y


def find_r1(x, rho0, v0, ns0, t_age, sigma0, w0):
    """
    Important function. Solves the following equation (Kaplinghat +16)
    1 = Gamma(r1) x tAGE
    1 = <sigma/m v>(r1) x rho(r1) x tAGE
    1 = (4/sqrt(pi) ) x (sigma/m) x sigma_vel(r1) x rho(r1) x tAGE

    Args: rho0 : normalization density profile.
          v0 : normalization of velocity dispersion.
          sigma0 : cross section per unit mass.
          ns0 : slope of velocity dispersion model, could be 0 for pure isothermal.
          t_age : Assumes constant halo age.
    """
    sigma_v_weight = sigma_vel_weight(x, ns0, v0, sigma0, w0) # km/s cm^2/g
    sigma_v_weight *= 1e5 # cm^3 / s / g
    rho = rho_isothermal(x, ns0) * rho0
    f = t_age * sigma_v_weight * rho
    f -= 1.0
    return f


def get_r1(rho0, v0, ns0, sigma0, w0):
    Msun_in_cgs = 1.98848e33
    kpc_in_cgs = 3.08567758e21

    t_age = 7.5  # Gyr - assuming constant halo age
    t_age_cgs = t_age * 1e9 * 365.24 * 24 * 3600  # sec
    rho0_cgs = rho0 * Msun_in_cgs / kpc_in_cgs ** 3  # g/cm^3

    G = 4.3e-6  # kpc km^2 Msun^-1 s^-2
    r0 = v0 ** 2 / (4. * np.pi * G * rho0)
    r0 = np.sqrt(r0)  # kpc

    sol = fsolve(find_r1, 20, args=(rho0_cgs, v0, ns0, t_age_cgs, sigma0, w0))
    r1 = sol[0] * r0  # kpc

    return r1

def log_likelihood(theta, x, y, yerr):
    """
    The natural logarithm of the joint likelihood.

    Args:
        theta (tuple): a sample containing individual parameter values
        x (array): values over which the data/model is defined
        y (array): the set of data
        yerr (array): the standard deviation of the data points
    """
    r0, rho0, ns0 = theta

    model = rho_isothermal(x / r0, ns0)
    model = 10**rho0 * model

    sigma2 = yerr**2
    log_l = -0.5 * np.sum((y - model) ** 2 / sigma2)
    return log_l


def log_likelihood_iso(theta, x, y, yerr):
    """
    The natural logarithm of the joint likelihood.

    Args:
        theta (tuple): a sample containing individual parameter values
        x (array): values over which the data/model is defined
        y (array): the set of data
        yerr (array): the standard deviation of the data points
    """
    r0, rho0 = theta

    model = rho_isothermal(x / r0, 0.)
    model = 10**rho0 * model

    sigma2 = yerr**2
    log_l = -0.5 * np.sum((y - model) ** 2 / sigma2)
    return log_l


def fit_models(input_file,i):
    if i == 100:
        x, y, yerr, yerrorbar, z, zerrorbar = read_data(input_file)
    else:
        x, y, yerr, yerrorbar, z, zerrorbar = read_data_single_halo(input_file)

    # Initial cross section input
    sigma0 = 108
    sigma0 = np.log10(sigma0)
    ns0 = 0
    w0 = 30
    w0 = np.log10(w0)

    # First fit profile based on Isothermal model
    popt, pcov = curve_fit(fit_isothermal_model, x, y, p0=[1, 10, ns0])
    r0 = popt[0]
    rho0 = popt[1]
    ns0 = popt[2]

    # Log10 values of free params
    v0 = calculate_log_v0(r0, 10**rho0)
    r1 = get_r1(10**rho0, 10**v0, ns0, 10**sigma0, 10**w0)

    print("======")
    print("Initial estimates:")
    print("rho0 = {0:.3f}".format(rho0))
    print("r0 = {0:.3f}".format(r0))
    print("ns0 = {0:.3f}".format(ns0))
    print("v0 = {0:.3f}".format(v0))
    print("r1 = {0:.3f}".format(r1))
    print("======")


    # New fit
    select = x <= r1

    np.random.seed(42)
    nll = lambda *args: -log_likelihood(*args)
    initial = np.array([r0, rho0, ns0])
    soln = minimize(nll, initial, args=(x[select], 10**y[select], 10**yerr[select]))
    r0_ml, rho0_ml, ns0_ml = soln.x

    # Log10 values of free params
    v0_ml = calculate_log_v0(r0_ml, 10**rho0_ml)
    r1_ml = get_r1(10**rho0_ml, 10**v0_ml, ns0_ml, 10**sigma0, 10**w0)

    print("======")
    print("Max-likelihood estimates:")
    print("rho0 = {0:.3f}".format(rho0_ml))
    print("r0 = {0:.3f}".format(r0_ml))
    print("ns0 = {0:.3f}".format(ns0_ml))
    print("v0 = {0:.3f}".format(v0_ml))
    print("r1 = {0:.3f}".format(r1))
    print("======")


    ##########
    np.random.seed(42)
    nll = lambda *args: -log_likelihood_iso(*args)
    initial = np.array([r0, rho0])
    soln = minimize(nll, initial, args=(x[select], 10**y[select], 10**yerr[select]))
    r0_ml_iso, rho0_ml_iso = soln.x

    # Log10 values of free params
    v0_ml_iso = calculate_log_v0(r0_ml_iso, 10**rho0_ml_iso)

    r1_ml_iso = get_r1(10**rho0_ml_iso, 10**v0_ml_iso, 0., 10**sigma0, 10**w0)

    print("======")
    print("Max-likelihood estimates (Isothermal profile)")
    print("rho0 = {0:.3f}".format(rho0_ml_iso))
    print("r0 = {0:.3f}".format(r0_ml_iso))
    print("v0 = {0:.3f}".format(v0_ml_iso))
    print("r1 = {0:.3f}".format(r1_ml_iso))
    print("======")
    ##########


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

    plot(x, 10**y, '-', label='Data')
    plt.fill_between(x, 10**y - 10**yerr / 2, 10**y + 10**yerrorbar / 2, alpha=0.4)

    xrange = np.arange(-1,3,0.1)
    xrange = 10**xrange

    model = rho_isothermal(xrange / r0_ml, ns0_ml)
    model = 10**rho0_ml * model
    plot(xrange, model, '--', label='Modified-isothermal profile',color='tab:orange')
    plot([r1_ml],[10**rho0_ml * rho_isothermal(r1_ml / r0_ml, ns0_ml)],'o',color='tab:orange')

    model = rho_isothermal(xrange / r0_ml_iso, 0)
    model = 10**rho0_ml_iso * model
    plot(xrange, model, '--', label='Isothermal profile',color='tab:green')
    plot([r1_ml_iso],[10**rho0_ml_iso * rho_isothermal(r1_ml_iso / r0_ml_iso, 0)],'o',color='tab:green')

    xscale('log')
    yscale('log')
    ylabel(r'Density profile [M$_{\odot}$/kpc$^{3}$]')
    xlabel(r'Radius [kpc]')
    plt.legend(loc=[0.22, 0.7], labelspacing=0.2, handlelength=1.5, handletextpad=0.4, frameon=False)
    axis([0.5, 1e2, 1e3, 1e8])
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)

    if i == 100:
        plt.savefig("density_profile_test.png", dpi=200)
    else :
        plt.savefig("density_profile_test_%i.png"%i, dpi=200)

    #######################
    # Plot the velocity dispersion
    figure()
    ax = plt.subplot(1, 1, 1)
    grid(True)

    plot(x, z, '-', label='Data')
    plt.fill_between(x, z - zerrorbar / 2, z + zerrorbar / 2, alpha=0.4)

    veldisp = v0_ml * velocity_dispersion(xrange/ r0_ml, ns0_ml)
    plot(xrange, veldisp, '--', label='Modified-isothermal profile')

    veldisp = v0_ml_iso * velocity_dispersion(xrange/ r0_ml_iso, 0)
    plot(xrange, veldisp, '--', label='Isothermal profile')

    xscale('log')
    ylabel(r'Velocity dispersion [km/s]')
    xlabel(r'Radius [kpc]')
    plt.legend(loc=[0.22, 0.7], labelspacing=0.2, handlelength=1.5, handletextpad=0.4, frameon=False)
    axis([0.5, 1e2, -4, 8])
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)

    if i == 100:
        plt.savefig("velocity_dispersion_test.png", dpi=200)
    else :
        plt.savefig("velocity_dispersion_test_%i.png"%i, dpi=200)



# for i in range(0,20):
#     input_file = "./data/Profile_halos_M9.0_DML006N188_SigmaVel100_%i.txt"%i
#     fit_models(input_file, i)


input_file = "./data/Profile_halos_M9.0_DML006N188_SigmaVel100.txt"
fit_models(input_file, 100)