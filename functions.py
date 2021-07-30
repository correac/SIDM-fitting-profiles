import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.optimize import root


def read_single_halo(file):
    data = np.loadtxt(file)
    # Remove potential zeros before we begin
    nozero = data[:,1] > 0
    data = data[nozero,:]
    min_rho = np.log10(data[:,1]) >= 3.
    data = data[min_rho, :]

    # Next:
    x = data[:,0]
    y = np.log10(data[:,1])
    yerr = np.zeros(len(x))
    errorbar = np.zeros(len(x))

    return x, y, yerr, errorbar

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

def rho_nfw(x):
    """
        Args: x = r/rs
        Returns: y = rho(r)/rho_s
    """
    y = 1. / (x * (1.+x)**2)
    return y

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

def M_isothermal(r, r0, rho0, ns0):
    """
    Mass enclosed within r1. Calculated using Isothermal profile.
    """
    rmin = -3
    step = 0.05

    if np.log10(r) > rmin+step:
        ri = np.arange(rmin, np.log10(r), step)
        ri = 10**ri
        delta_ri = ri[1:]-ri[:-1]
        delta_ri = np.append(delta_ri,delta_ri[-1])

        rhoi = rho0 * rho_isothermal( ri/r0, ns0)
        Mint = 4. * np.pi * ri**2 * rhoi * delta_ri
        Mint = np.sum(Mint)
    else : Mint = 0
    return Mint

def M_nfw(r1, rs, rhos):
    """
    Mass enclosed within r1. Calculated using NFW profile.
    """
    rmin = -3
    step = 0.05

    if np.log10(r1) > rmin+step:
        ri = np.arange(rmin, np.log10(r1), step)
        ri = 10**ri
        delta_ri = ri[1:]-ri[:-1]
        delta_ri = np.append(delta_ri,delta_ri[-1])

        rhoi = rhos * rho_nfw( ri/rs )
        Mint = 4. * np.pi * ri**2 * rhoi * delta_ri
        Mint = np.sum(Mint)
    else: Mint = 0
    return Mint

def cross_section_weighted(v0, sigma0, w0):
    """
    Function : <sigma/m v>(r) / <v(r)>
    Returns effective velocity-independent momentum-transfer
    cross-section in the isothermal region.
    """
    integral = quad(integrand, 0, np.inf, args=(sigma0, w0, v0))[0]

    sigma_weight = v0 / (2 * np.sqrt(np.pi))
    sigma_weight *= integral

    v_weight = v0 * 4 / np.sqrt(np.pi)
    sigma_weight /= v_weight
    return sigma_weight

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
    #weight_function = (4. / np.sqrt(np.pi)) * v * sigma0

    integral = quad(integrand, 0, np.inf, args=(sigma0, w0, v))[0]

    weight_function = v / (2 * np.sqrt(np.pi))
    weight_function *= integral

    return weight_function

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

def find_rho0(N0, t_age, v0, ns0, sigma0, w0):
    Msun_in_cgs = 1.98848e33
    kpc_in_cgs = 3.08567758e21

    t_age_cgs = t_age * 1e9 * 365.24 * 24 * 3600  # sec
    sigma_v_weight = sigma_vel_weight(1, ns0, v0, sigma0, w0) * 1e5  # 1/s cm^3/g
    rho0_cgs = N0 / (t_age_cgs * sigma_v_weight)  # g / cm^3
    rho0 = rho0_cgs * kpc_in_cgs ** 3 / Msun_in_cgs  # Msun / kpc^3
    return rho0

def find_nfw(x, r1, rho1, M1):

    rhoNFW = rho_nfw(r1 / x[0])
    if rhoNFW <= 0: rhoNFW = 1.

    MNFW = M_nfw(r1, x[0], 10**x[1])
    if MNFW <= 0: MNFW = 1.

    f = [np.log10(rhoNFW) + x[1] - rho1,
         np.log10(MNFW) - M1]
    return f


def rho_joint_profiles(r, r1, r0, rho0, rs, rhos, ns0):

    rho = np.zeros(len(r))

    for i in range(0,len(r)):
        if r[i] <= r1:
            rho[i] = rho0 * rho_isothermal(r[i] / r0, ns0)
        else :
            rho[i] = rhos * rho_nfw(r[i] / rs)
    return rho

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


def calc_R200(M200):
    z = 0
    Om = 0.307
    Ol = 1.0 - Om
    rhocrit = 2.775e11 * 0.6777 ** 2 / (1e3) ** 3  # Msun/kpc^3
    rhocrit *= (Om * (1. + z) ** 3 + Ol)

    R200 = M200 / ((4. * np.pi / 3.) * 200. * rhocrit )
    R200 = R200**(1. / 3.) #kpc

    return R200

def calc_rhos(c200):
    z = 0
    Om = 0.307
    Ol = 1.0 - Om
    rhocrit = 2.775e11 * 0.6777 ** 2 / (1e3) ** 3  # Msun/kpc^3
    rhocrit *= (Om * (1. + z) ** 3 + Ol)

    Yc200 = np.log(1.+c200) - c200/(1.+c200)
    rhos = (200./3.) * rhocrit * c200**3 / Yc200 # Msun/kpc^3
    return rhos

def calc_rs(M200, c200):
    R200 = calc_R200(M200) #kpc
    rs = R200 / c200 #kpc
    return rs

def find_c200_M200(x, r1, rho1, M1):

    rs = calc_rs(10**x[0], 10**x[1]) #kpc
    rhoNFW = rho_nfw(r1 / rs)
    if rhoNFW <= 0: rhoNFW = 1.

    rhos = calc_rhos(10**x[1])
    MNFW = M_nfw(r1, rs, rhos)
    if MNFW <= 0: MNFW = 1.

    f = [np.log10(rhoNFW) + np.log10(rhos) - rho1,
         np.log10(MNFW) - M1]
    return f

def find_nfw_params(N0, v0, ns0, sigma0, w0, log10M200, log10c200):

    Msun_in_cgs = 1.98848e33
    kpc_in_cgs = 3.08567758e21

    t_age = 7.5 # Gyr - assuming constant halo age
    rho0 = find_rho0(N0, t_age, v0, ns0, sigma0, w0)
    t_age_cgs = t_age * 1e9 * 365.24 * 24 * 3600  # sec
    rho0_cgs = rho0 * Msun_in_cgs / kpc_in_cgs ** 3 # g/cm^3

    G = 4.3e-6  # kpc km^2 Msun^-1 s^-2
    r0 = v0**2 / (4. * np.pi * G * rho0)
    r0 = np.sqrt(r0) # kpc

    sol = fsolve(find_r1, 20, args=(rho0_cgs, v0, ns0, t_age_cgs, sigma0, w0))
    r1 = sol[0] * r0 # kpc

    M1 = M_isothermal(r1, r0, rho0, ns0) # Msun
    rho1 = rho0 * rho_isothermal( r1/r0, ns0) # Msun /kpc^3

    sol = root(find_c200_M200, [log10M200, log10c200], args=(r1, np.log10(rho1), np.log10(M1)), method='hybr', tol=1e-4)

    log10M200 = sol.x[0]
    log10c200 = sol.x[1]

    return log10M200, log10c200


def c_M_relation(log_M0):
    """
    Concentration-mass relation from Correa et al. (2015).
    This relation is most suitable for Planck cosmology.
    """
    z = 0
    # Best-fit params:
    alpha = 1.7543 - 0.2766 * (1. + z) + 0.02039 * (1. + z) ** 2
    beta = 0.2753 + 0.0035 * (1. + z) - 0.3038 * (1. + z) ** 0.0269
    gamma = -0.01537 + 0.02102 * (1. + z) ** (-0.1475)

    log_10_c200 = alpha + beta * log_M0 * (1. + gamma * log_M0 ** 2)
    c200 = 10 ** log_10_c200
    return c200
