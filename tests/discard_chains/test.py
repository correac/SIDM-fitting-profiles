import numpy as np
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from functions import fit_isothermal_model, calculate_log_v0, calculate_log_N0, calculate_log_sigma0
from functions import read_data, read_single_halo
from functions import find_nfw_params, c_M_relation

import emcee
import corner
from pylab import *
from sidm_model import sidm_halo_model
from functions import find_nfw_params, c_M_relation, calculate_log_N0
from multiprocessing import Pool
import time

import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

def quality(N0, v0, sigma0, x, y):

    #logM200, logc200 = find_nfw_params(10 ** N0, 10 ** v0, 10 ** sigma0, 10., np.log10(c_M_relation(10.)))

    Nbins = len(x)
    rho_model = sidm_halo_model(x, 10**N0, 10**v0, 10**sigma0)

    quality = np.sum( (y-rho_model)**2 ) / Nbins
    return quality


def find_initial_sigma0(rho0, r0, v0, x, y, yerr):

    # Uniform sigma0 priors
    sigma0_list = np.arange(-1,2,0.25)

    # Initial values
    quality_of_fit_prev = 100
    best_sigma0 = 0

    for sigma0 in sigma0_list:

        N0 = calculate_log_N0(10 ** rho0, 10 ** v0, 10 ** sigma0)

        np.random.seed(37)
        nll = lambda *args: -log_ml(*args)
        initial = np.array([N0, v0, sigma0])
        soln = minimize(nll, initial, args=(x, y, yerr))
        N0_ml, v0_ml, sigma0_ml = soln.x

        quality_of_fit = quality(N0_ml, v0_ml, sigma0_ml, x, y)

        if quality_of_fit < quality_of_fit_prev:
            best_sigma0 = sigma0_ml
            quality_of_fit_prev = quality_of_fit


    return best_sigma0

def log_prior_c200(c0, log10M0):

    delta_sig_c200 = 0.13
    w = np.log10(c0 / c_M_relation(log10M0))
    w /= delta_sig_c200
    w = -1 * w**2

    return w

def log_prior(theta):
    """
    The natural logarithm of the prior probability.

    It sets prior to 1 (log prior to 0) if params are in range, and zero (-inf) otherwise.

    Args:
        theta (tuple): a sample containing individual parameter values
    """
    N0, v0, sigma0 = theta

    log_prior = -np.inf

    # Here we convert from N0 to M200, c200
    if 0 < N0 < 5 and -1. < v0 < 3.5 and -2. < sigma0 < 4.:

        #logM200, logc200 = find_nfw_params(10**N0, 10**v0, 10**sigma0, 10., np.log10(c_M_relation(10.)))

        # Flat priors on all except c200:
        if -2. < sigma0 < 4. and 6 < logM200 < 14 and 0 < logc200 < 2:

            # Prior distribution on c200:
            #c200 = 10 ** logc200
            #log_prior = log_prior_c200(c200, logM200)

            log_prior = 0


    return log_prior

def log_posterior(theta):
    """
    The natural logarithm of the joint posterior.

    Args:
        theta (tuple): a sample containing individual parameter values
        x (array): values over which the data/model is defined
        y (array): the set of data
        yerr (array): the standard deviation of the data points
    """
    N0, v0, sigma0 = theta
    #logM200, logc200 = find_nfw_params(10 ** N0, 10 ** v0, 10 ** sigma0, 10., np.log10(c_M_relation(10.)))

    lp = log_prior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta)

def log_ml(theta, x, y, yerr):
    """
    The natural logarithm of the joint likelihood.

    Args:
        theta (tuple): a sample containing individual parameter values
        x (array): values over which the data/model is defined
        y (array): the set of data
        yerr (array): the standard deviation of the data points
    """
    N0, v0, sigma0 = theta
    #logM200, _ = find_nfw_params(10 ** N0, 10 ** v0, 10 ** sigma0, 10., np.log10(c_M_relation(10.)))

    model = sidm_halo_model(x, 10**N0, 10**v0, 10**sigma0)
    sigma2 = yerr**2
    log_l = -0.5 * np.sum((y - model) ** 2 / sigma2)
    return log_l

def log_likelihood(theta):
    """
    The natural logarithm of the joint likelihood.

    Args:
        theta (tuple): a sample containing individual parameter values
        x (array): values over which the data/model is defined
        y (array): the set of data
        yerr (array): the standard deviation of the data points
    """
    N0, v0, sigma0 = theta
    model = sidm_halo_model(x_global, 10**N0, 10**v0, 10**sigma0)
    sigma2 = yerr_global ** 2
    log_l = -0.5 * np.sum((y_global - model) ** 2 / sigma2)
    return log_l


def likelihood_of_chain(chain):

    steps = len(chain[:,0])
    lk = 0

    for k in range(0, steps):

        N0, v0, sigma0 = chain[k,:]
        #logM200, _ = find_nfw_params(10 ** N0, 10 ** v0, 10 ** sigma0, 10., np.log10(c_M_relation(10.)))

        lk += log_likelihood(chain[k,:])

    lk /= steps
    return lk


if __name__ == '__main__':

    input_file = "../../data/L006N188_SigmaConstant10/Individual_sample/Profile_halos_M10.0_DML006N188_SigmaConstant10_5.txt"
    output_folder = "./"
    name = "DML006N188_SigmaConstant10_M10.0_5"

    # Output data
    output_file = output_folder+"Output_"+name+".txt"
    output_plot_final = output_folder+"Initial_fit"+name+".png"
    output_corner_plot = output_folder+"corner_"+name+".png"

    # Read initial data
    x, y, yerr, errorbar = read_single_halo(input_file)

    # First fit profile based on Isothermal model
    popt, pcov = curve_fit(fit_isothermal_model, x, y, p0=[1, 10])
    r0 = popt[0]
    rho0 = popt[1]

    # Log10 values of free params
    v0 = calculate_log_v0(r0, 10**rho0)

    # Make a guess for initial sigma0
    print("Finding best-initial sigma0..")
    #sigma0 = find_initial_sigma0(rho0, r0, v0, x, y, yerr)
    #sigma0 = calculate_log_sigma0(10**rho0, 10**v0, 10)
    sigma0 = 1
    N0 = calculate_log_N0(10**rho0, 10**v0, 10**sigma0)

    print("======")
    print("Initial estimates:")
    print("rho0 = {0:.3f}".format(rho0))
    print("r0 = {0:.3f}".format(r0))
    print("N0 = {0:.3f}".format(N0))
    print("sigma0 = {0:.3f}".format(10**sigma0))
    print("v0 = {0:.3f}".format(v0))
    print("======")

    np.random.seed(42)
    nll = lambda *args: -log_ml(*args)
    initial = np.array([N0, v0, sigma0])
    soln = minimize(nll, initial, args=(x, y, yerr))
    N0_ml, v0_ml, sigma0_ml = soln.x

    logM200, logc200 = find_nfw_params(10 ** N0_ml, 10 ** v0_ml, 10 ** sigma0_ml,
                                       10., np.log10(c_M_relation(10.)))
    sol = np.array([N0_ml, v0_ml, sigma0_ml, logM200, logc200])


    global x_global, y_global, yerr_global
    x_global, y_global, yerr_global = x, y, yerr

    #pos = soln.x + 1e-4 * np.random.randn(64, 3)
    pos = soln.x + 1e-2 *np.random.randn(64, 3)
    nwalkers, ndim = pos.shape

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool)
        start = time.time()
        sampler.run_mcmc(pos, 500, progress=True)
        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} minutes".format(multi_time / 60))

    samples = sampler.get_chain(discard=100, thin=1, flat=False)

    samples_log_prob = sampler.get_log_prob()

    print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(sampler.get_autocorr_time(quiet=True))))

    # Removing stuck iterations
    ll_walkers = np.sum(samples_log_prob[:,:],axis=0) / len(samples_log_prob[:,0])
    index = np.argsort(ll_walkers)
    ll_walkers = ll_walkers[index]

    C = 100.
    ll_k_diff = ll_walkers[1:] - ll_walkers[:-1]
    ll_k_diff -= C * (ll_walkers[1:]-ll_walkers[0]) / np.arange(1,nwalkers)
    select_chain = np.where(ll_k_diff < 0)[0] # select where difference is smaller than average difference

    new_chain = np.arange(0, int(nwalkers / 4)) #let's keep first 64/4 walkers
    new_chain = np.append(new_chain, select_chain[select_chain > nwalkers / 4])

    new_sample = samples[:, index[new_chain]]
    s = list(new_sample.shape[1:])
    s[0] = np.prod(new_sample.shape[:2])
    new_sample = new_sample.reshape(s) # flatting..

    labels = ["log$_{10}$N$_{0}$",
              "log$_{10}$v$_{0}$",
              "log$_{10}$($\sigma_{0}$/m/cm$^{2}$g$^{-1}$)"]

    N0 = np.median(new_sample[:, 0])
    v0 = np.median(new_sample[:, 1])
    sigma0 = np.median(new_sample[:, 2])

    # Plot parameters
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

    # Make the base corner plot
    figure = corner.corner(
        new_sample, labels=labels, quantiles=[0.16, 0.84],
        truths=[N0, v0, sigma0], show_titles=True,
        title_kwargs={"fontsize": 16}
    )

    plt.savefig(output_corner_plot, dpi=200)
