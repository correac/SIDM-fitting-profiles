import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d

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

    return x, y, yerr, errorbar


def diff_isothermal_equation(f,x):
    """
    Differential equation that describes the isothermal profile
    """
    y, z = f
    n = 0.0
    dfdx = [z,-(n+2)*(1./x)*z-n*(n+1)*(1./x**2)-(1./x**n)*np.exp(y)]
    return dfdx

def rho_nfw(x):
    """
        Args: x = r/rs
        Returns: y = rho(r)/rho_s
    """
    y = 1. / (x * (1.+x)**2)
    return y

def rho_isothermal(x):
    """
        Args: x = r/r0
        Returns: y = rho(r)/rho0
    """
    xrange = np.arange(-4, 5, 0.1)
    xrange = 10 ** xrange
    y0 = [0, 0]
    sol = odeint(diff_isothermal_equation, y0, xrange)
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

def M_isothermal(r, r0, rho0):
    """
    Mass enclosed within r1. Calculated using Isothermal profile.
    """
    ri = np.arange(-3, np.log10(r), 0.1)
    ri = 10**ri
    delta_ri = ri[1:]-ri[:-1]
    delta_ri = np.append(delta_ri,delta_ri[-1])

    rhoi = rho0 * rho_isothermal( ri/r0 )
    Mint = 4. * np.pi * ri**2 * rhoi * delta_ri
    Mint = np.sum(Mint)
    return Mint

def M_nfw(r1, rs, rhos):
    """
    Mass enclosed within r1. Calculated using NFW profile.
    """
    ri = np.arange(-3, np.log10(r1), 0.1)
    ri = 10**ri
    delta_ri = ri[1:]-ri[:-1]
    delta_ri = np.append(delta_ri,delta_ri[-1])

    rhoi = rhos * rho_nfw( ri/rs )
    Mint = 4. * np.pi * ri**2 * rhoi * delta_ri
    Mint = np.sum(Mint)
    return Mint

def find_r1(x, rho0, v0, sigma0, t_age):
    f = t_age * (4. / np.sqrt(np.pi)) * v0 * sigma0
    f *= rho_isothermal(x) * rho0
    f -= 1.0
    return f

def find_nfw(x, r1, rho1, M1):

    rhoNFW = rho_nfw(r1 / x[0])
    if rhoNFW <= 0: rhoNFW = 1.

    MNFW = M_nfw(r1, x[0], 10**x[1])
    if MNFW <= 0: MNFW = 1.

    f = [np.log10(rhoNFW) + x[1] - rho1,
         np.log10(MNFW) - M1]
    return f


def rho_joint_profiles(r, r1, r0, rho0, rs, rhos):

    rho = np.zeros(len(r))

    for i in range(0,len(r)):
        if r[i] <= r1:
            rho[i] = rho0 * rho_isothermal(r[i] / r0)
        else :
            rho[i] = rhos * rho_nfw(r[i] / rs)
    return rho

def calculate_log_v0(r0, rho0):
    G = 4.3e-6  # kpc km^2 Msun^-1 s^-2
    v0 = r0**2 * 4. * np.pi * G * rho0
    v0 = np.sqrt(v0)
    return np.log10(v0)

def calculate_log_N0(v0, rho0, sigma0):
    Msun_in_cgs = 1.98848e33
    kpc_in_cgs = 3.08567758e21

    t_age = 7.5 # Gyr - assuming constant halo age
    t_age_cgs = t_age * 1e9 * 365.24 * 24 * 3600 # sec
    v0_cgs = v0 * 1e5 # cm/s
    rho0_cgs = rho0 * Msun_in_cgs / kpc_in_cgs**3

    N0 = t_age_cgs * (4. / np.sqrt(np.pi)) * v0_cgs * sigma0 * rho0_cgs
    return np.log10(N0)

def fit_isothermal_model(xdata, a, b):
    xrange = np.arange(-5,5,0.1)
    xrange = 10**xrange
    xrange = xrange / a
    y0 = [0, 0]
    sol = odeint(diff_isothermal_equation, y0, xrange)
    yrange = np.exp(sol[:, 0])
    yrange = np.log10(yrange)
    finterpolate = interp1d(xrange, yrange)
    x = xdata / a
    ydata = finterpolate(x)
    f = b + ydata
    return f