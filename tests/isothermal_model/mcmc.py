import numpy as np
import emcee
import corner
from pylab import *
from scipy.optimize import minimize
from sidm_model import sidm_halo_model
from plotter import output_best_fit_params
from functions import find_nfw_params, c_M_relation, calculate_log_N0
from plotter import plot_solution
from multiprocessing import Pool
import time


def convert_params(theta):
    """
        Returns c200, Log10M200 and Log10sigma0
        based on sampling from Log10N0, Log10v0 and Log10sigma0.
    Args:
        theta (tuple): Log10N0, Log10v0 and Log10sigma0
    """
    N0 = theta[:,0]
    v0 = theta[:,1]
    sigma0 = theta[:, 2]

    M200 = np.zeros(len(N0))
    c200 = np.zeros(len(N0))

    # Calculate NFW profile params:
    for i in range(len(N0)):
        M200[i], c200[i] = find_nfw_params(10**N0[i], 10**v0[i], 10**sigma0[i], 10., np.log10(c_M_relation(10.)))

    params = np.array([10**c200, M200, sigma0])
    return params.T

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
        #if -2. < sigma0 < 4. and 6 < logM200 < 14 and 0 < logc200 < 2:

            # Prior distribution on c200:
        #    c200 = 10 ** logc200
        #    log_prior = log_prior_c200(c200, logM200)
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
    lk = np.zeros(steps)

    for k in range(0, steps):

        N0, v0, sigma0 = chain[k,:]
        lk[k] = log_likelihood(theta[k,:])

    lk = np.cumsum(lk) / steps
    return lk


def check_chain(samples, samples_log_prob, nwalkers):

    # Removing stuck iterations
    ll_walkers = np.sum(samples_log_prob[:, :], axis=0) / len(samples_log_prob[:, 0])

    index = np.argsort(ll_walkers)
    ll_walkers = ll_walkers[index]

    C = 100.
    ll_k_diff = ll_walkers[1:] - ll_walkers[:-1]
    ll_k_diff -= C * (ll_walkers[1:] - ll_walkers[0]) / np.arange(1, nwalkers)
    select_chain = np.where(ll_k_diff < 0)[0]  # select where difference is smaller than average difference

    new_chain = np.arange(0, int(nwalkers / 4))  # let's keep first 64/4 walkers
    new_chain = np.append(new_chain, select_chain[select_chain > nwalkers / 4])

    new_sample = samples[:, index[new_chain]]

    s = list(new_sample.shape[1:])
    s[0] = np.prod(new_sample.shape[:2])
    new_sample = new_sample.reshape(s)  # flatting..
    return new_sample

def run_mcmc(soln, x, y, yerr, errorbar, name, output_folder):

    global x_global, y_global, yerr_global
    x_global, y_global, yerr_global = x, y, yerr

    output_corner_plot = output_folder + "corner_" + name + ".png"

    #N0, v0, sigma0 = soln.x

    #pos = soln.x + 1e-4 * np.random.randn(64, 3)
    pos = soln + np.random.randn(64, 3)
    nwalkers, ndim = pos.shape

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool)
        start = time.time()
        sampler.run_mcmc(pos, 500, progress=True)
        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} minutes".format(multi_time / 60))

    samples = sampler.get_chain(discard=100, thin=1, flat=True)
    soln = np.array([np.median(samples[:, 0]), np.median(samples[:, 1]), np.median(samples[:, 2])])

    pos = soln + 1e-4 * np.random.randn(64, 3)

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool)
        start = time.time()
        sampler.run_mcmc(pos, 1000, progress=True)
        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} minutes".format(multi_time / 60))

    samples = sampler.get_chain(discard=200, thin=1, flat=False)
    samples_log_prob = sampler.get_log_prob()

    print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(sampler.get_autocorr_time(quiet=True))))

    # Removing stuck iterations
    flat_samples = check_chain(samples, samples_log_prob, nwalkers)

    labels = ["log$_{10}$N$_{0}$",
              "log$_{10}$v$_{0}$",
              "log$_{10}$($\sigma_{0}$/m/cm$^{2}$g$^{-1}$)"]

    N0 = np.median(flat_samples[:, 0])
    v0 = np.median(flat_samples[:, 1])
    sigma0 = np.median(flat_samples[:, 2])

    # Make the base corner plot
    figure = corner.corner(
        flat_samples, labels=labels, quantiles=[0.16, 0.84],
        truths=[N0, v0, sigma0], show_titles=True,
        title_kwargs={"fontsize": 16}
    )

    plt.savefig(output_corner_plot, dpi=200)

    output_data = output_folder + "samples_" + name + ".txt"
    np.savetxt(output_data, flat_samples, fmt="%s")



def quality(N0, v0, sigma0, x, y):

    logM200, logc200 = find_nfw_params(10 ** N0, 10 ** v0, 10 ** sigma0, 10., np.log10(c_M_relation(10.)))

    Nbins = len(x)
    rho_model = sidm_halo_model(x, 10**N0, 10**v0, 10**sigma0)

    quality = np.sum( (y-rho_model)**2 ) / Nbins
    return quality

def find_initial_sigma0(rho0, r0, v0, x, y, yerr):

    # Uniform sigma0 priors
    sigma0_list = np.arange(-1,2,0.5)

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

