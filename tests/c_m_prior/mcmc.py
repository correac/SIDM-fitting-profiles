import numpy as np
import emcee
import corner
from pylab import *
from sidm_model import sidm_halo_model_default
from plotter import model_params, plot_solution_default, output_best_fit_params
from functions import c_M_relation

from multiprocessing import Pool
import time

def log_prior_c200(c0, log10M0):
    M0 = 10**log10M0
    delta_sig_c200 = 0.13
    w = np.log10(c0 / c_M_relation(M0))
    w /= delta_sig_c200
    w = np.exp(-1 * w**2)
    return np.log10(w)

def log_prior_default(theta):
    """
    The natural logarithm of the prior probability.

    It sets prior to 1 (log prior to 0) if params are in range, and zero (-inf) otherwise.

    Args:
        theta (tuple): a sample containing individual parameter values
    """
    log10N0, log10v0, ns0, log10sigma0, log10w0, log10M200, log10c200 = theta

    # Flat priors on all except c200:
    if 0 < log10N0 < 5 and -1. < log10v0 < 3.5 and -0.5 < ns0 < 0.5 \
            and -2. < log10sigma0 < 4. and -2. < log10w0 < 3. and \
            4 < log10M200 < 16 and 0 < log10c200 < 2:
        # Prior distribution on c200
        c200 = 10**log10c200
        log_prior = log_prior_c200(c200, log10M200)
        return log_prior
    else:
        return -np.inf

def log_posterior_default(theta, x, y, yerr):
    """
    The natural logarithm of the joint posterior.

    Args:
        theta (tuple): a sample containing individual parameter values
        x (array): values over which the data/model is defined
        y (array): the set of data
        yerr (array): the standard deviation of the data points
    """
    lp = log_prior_default(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_default(theta, x, y, yerr)

def log_likelihood_default(theta, x, y, yerr):
    """
    The natural logarithm of the joint likelihood.

    Args:
        theta (tuple): a sample containing individual parameter values
        x (array): values over which the data/model is defined
        y (array): the set of data
        yerr (array): the standard deviation of the data points
    """
    log10N0, log10v0, ns0, log10sigma0, log10w0, log10M200, log10c200 = theta

    N0 = 10**log10N0
    v0 = 10**log10v0
    sigma0 = 10**log10sigma0
    w0 = 10**log10w0
    c200 = 10**log10c200

    model = sidm_halo_model_default(x, N0, v0, ns0, sigma0, w0, log10M200, c200)
    sigma2 = yerr**2
    log_l = -0.5 * np.sum((y - model) ** 2 / sigma2)
    return log_l


def run_mcmc_default(soln, x, y, yerr, errorbar, name, output_folder):

    output_corner_plot = output_folder+"corner_"+name+".png"

    pos = soln.x + 1e-4 * np.random.randn(64, 7)
    nwalkers, ndim = pos.shape

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_default, args=(x, y, yerr), pool=pool)
        start = time.time()
        sampler.run_mcmc(pos, 500, progress=True)
        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} minutes".format(multi_time / 60))

    flat_samples = sampler.get_chain(discard=10, thin=15, flat=True)

    labels = ["log$_{10}$N$_{0}$",
              "log$_{10}$v$_{0}$",
              "n$_{s0}$",
              "log$_{10}$($\sigma_{0}$/m/cm$^{2}$g$^{-1}$)",
              "log$_{10}$($w_{0}$/km/s)",
              "log$_{10}$(M$_{200}$/M$_{\odot}$)",
              "log$_{10}$c$_{200}$"]

    log10N0 = np.median(flat_samples[:,0])
    log10v0 = np.median(flat_samples[:,1])
    ns0 = np.median(flat_samples[:,2])
    log10sigma0 = np.median(flat_samples[:,3])
    log10w0 = np.median(flat_samples[:,4])
    log10M200 = np.median(flat_samples[:,5])
    log10c200 = np.median(flat_samples[:,6])

    # Make the base corner plot
    figure = corner.corner(
        flat_samples, labels=labels, quantiles=[0.16, 0.84],
        truths=[log10N0, log10v0, ns0, log10sigma0, log10w0, log10M200, log10c200], show_titles=True,
        title_kwargs={"fontsize": 16}
    )

    plt.savefig(output_corner_plot, dpi=200)

    mcmc_sol = np.array([log10N0, log10v0, ns0, log10sigma0, log10w0, log10M200, log10c200])
    output_file = output_folder+"MCMC_Output_"+name+".txt"
    output_fig = output_folder+"MCMC_fit_"+name+".png"
    plot_solution_default(x, 10**y, 10**errorbar, mcmc_sol, output_fig, output_file)

    # Output bestfit paramter range
    sol_median = np.array([10**log10N0, 10**log10v0, ns0, 10**log10sigma0, 10**log10w0, log10M200, 10**log10c200])

    log10N0 = np.percentile(flat_samples[:,0],84)
    log10v0 = np.percentile(flat_samples[:,1],84)
    ns0 = np.percentile(flat_samples[:,2],84)
    log10sigma0 = np.percentile(flat_samples[:,3],84)
    log10w0 = np.percentile(flat_samples[:,4],84)
    log10M200 = np.percentile(flat_samples[:,5],84)
    log10c200 = np.percentile(flat_samples[:,6],84)
    sol_upper = np.array([10**log10N0, 10**log10v0, ns0, 10**log10sigma0, 10**log10w0, log10M200, 10**log10c200])

    log10N0 = np.percentile(flat_samples[:,0],16)
    log10v0 = np.percentile(flat_samples[:,1],16)
    ns0 = np.percentile(flat_samples[:,2],16)
    log10sigma0 = np.percentile(flat_samples[:,3],16)
    log10w0 = np.percentile(flat_samples[:,4],16)
    log10M200 = np.percentile(flat_samples[:,5],16)
    log10c200 = np.percentile(flat_samples[:,6],16)
    sol_lower = np.array([10**log10N0, 10**log10v0, ns0, 10**log10sigma0, 10**log10w0, log10M200, 10**log10c200])

    output_file = output_folder+"MCMC_parameter_range_"+name+".txt"

    output_best_fit_params(sol_median, sol_upper, sol_lower, output_file)
