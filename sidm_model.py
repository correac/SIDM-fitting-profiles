import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import root
from functions import rho_isothermal, M_isothermal, find_r1, \
    find_nfw, rho_joint_profiles, find_rho0

def sidm_halo_model(r, N0, v0, ns0, sigma0, w0):
    """
        Calculates density profile for an SIDM halo
        assuming a truncation between an isothermal profile
        and an NFW profile at radius r1.

        Args:
            x  : values over which the data/model is defined (radius, ri)
            N0 : Number of scatterings per particle [no units]
            v0 : 1D velocity dispersion [units km/s]
            sigma0 : cross section per unit mass [units cm^2/g]
            ns0 : Inner slope of velocity dispersion
        """
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

    sol = root(find_nfw, [1, np.log10(rho0)], args=(r1, np.log10(rho1), np.log10(M1)),
               method='hybr', tol=1e-4)

    rs = sol.x[0]
    rhos = 10**sol.x[1]
    if rs < 0: rs = 1

    rho = rho_joint_profiles(r, r1, r0, rho0, rs, rhos, ns0)
    return np.log10(rho)



