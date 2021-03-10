import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import root
from functions import rho_isothermal, M_isothermal, find_r1, find_nfw, rho_joint_profiles

def sidm_halo_model(r, N0, v0, sigma0):
    """
        Calculates density profile for an SIDM halo
        assuming a trunctaction between an isothermal profile
        and an NFW profile at radius r1.

        Args:
            x  : values over which the data/model is defined (radius, ri)
            N0 : Number of scatterings per particle [no units]
            v0 : 1D velocity dispersion [units km/s]
            sigma0 : cross section per unit mass [units cm^2/g]
        """
    Msun_in_cgs = 1.98848e33
    kpc_in_cgs = 3.08567758e21

    t_age = 7.5 # Gyr - assuming constant halo age
    t_age_cgs = t_age * 1e9 * 365.24 * 24 * 3600 # sec
    v0_cgs = v0 * 1e5 # cm/s
    rho0_cgs = N0 / ( t_age_cgs * (4. / np.sqrt(np.pi)) * v0_cgs * sigma0 ) #g / cm^3
    rho0 = rho0_cgs * kpc_in_cgs**3 / Msun_in_cgs #Msun / kpc^3

    G = 4.3e-6  # kpc km^2 Msun^-1 s^-2
    r0 = v0**2 / (4. * np.pi * G * rho0)
    r0 = np.sqrt(r0) # kpc

    sol = fsolve(find_r1, 20, args=(rho0_cgs, v0_cgs, sigma0, t_age_cgs))
    r1 = sol[0] * r0 # kpc

    M1 = M_isothermal(r1, r0, rho0) # Msun
    rho1 = rho0 * rho_isothermal( r1/r0 ) # Msun /kpc^3

    sol = root(find_nfw, [20, np.log10(rho0)], args=(r1, np.log10(rho1), np.log10(M1)),
               method='hybr', tol=1e-4)

    rs = sol.x[0]
    rhos = 10**sol.x[1]
    if rs < 0: rs = 1

    rho = rho_joint_profiles(r, r1, r0, rho0, rs, rhos)
    return np.log10(rho)



