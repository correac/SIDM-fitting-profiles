import numpy as np
import emcee
import corner
from pylab import *
from sidm_model import sidm_halo_model
from plotter import output_best_fit_params
from functions import find_nfw_params, c_M_relation
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
    ns0 = theta[:,2]
    sigma0 = theta[:, 3]

    M200 = np.zeros(len(N0))
    c200 = np.zeros(len(N0))

    # Calculate NFW profile params:
    for i in range(len(N0)):
        M200[i], c200[i] = find_nfw_params(10**N0[i], 10**v0[i], ns0[i], 10**sigma0[i], 10., np.log10(c_M_relation(10.)))

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
    N0, v0, ns0, sigma0 = theta

    log_prior = -np.inf

    # Here we convert from N0, v0, ns0 to M200, c200

    if 0 < N0 < 5 and -1. < v0 < 3.5 and -2. < sigma0 < 4. and -0.5 < ns0 < 0.5:

        logM200, logc200 = find_nfw_params(10**N0, 10**v0, ns0, 10**sigma0, 10., np.log10(c_M_relation(10.)))

        # Flat priors on all except c200:
        if -2. < sigma0 < 4. and 6 < logM200 < 14 and 0 < logc200 < 2:
            # Prior distribution on c200
            c200 = 10 ** logc200
            log_prior = log_prior_c200(c200, logM200)


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
    N0, v0, ns0, sigma0 = theta
    model = sidm_halo_model(x, 10**N0, 10**v0, ns0, 10**sigma0)
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
    N0, v0, ns0, sigma0 = theta
    model = sidm_halo_model(x_global, 10**N0, 10**v0, ns0, 10**sigma0)
    sigma2 = yerr_global ** 2
    log_l = -0.5 * np.sum((y_global - model) ** 2 / sigma2)
    return log_l

def run_mcmc(soln, x, y, yerr, errorbar, name, output_folder):

    global x_global, y_global, yerr_global
    x_global, y_global, yerr_global = x, y, yerr

    output_corner_plot = output_folder + "corner_" + name + ".png"

    N0, v0, ns0, sigma0 = soln.x

    pos = soln.x + 1e-4 * np.random.randn(64, 4)
    nwalkers, ndim = pos.shape

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool)
        start = time.time()
        sampler.run_mcmc(pos, 500, progress=True)
        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} minutes".format(multi_time / 60))


    flat_samples = sampler.get_chain(discard=10, thin=15, flat=True)

    labels = ["log$_{10}$N$_{0}$",
              "log$_{10}$v$_{0}$",
              "n$_{s0}$",
              "log$_{10}$($\sigma_{0}$/m/cm$^{2}$g$^{-1}$)"]

    N0 = np.median(flat_samples[:, 0])
    v0 = np.median(flat_samples[:, 1])
    ns0 = np.median(flat_samples[:, 2])
    sigma0 = np.median(flat_samples[:, 3])

    # Make the base corner plot
    figure = corner.corner(
        flat_samples, labels=labels, quantiles=[0.16, 0.84],
        truths=[N0, v0, ns0, sigma0], show_titles=True,
        title_kwargs={"fontsize": 16}
    )

    plt.savefig(output_corner_plot, dpi=200)

    labels = ["c$_{200}$",
              "log$_{10}$(M$_{200}$/M$_{\odot}$)",
              "log$_{10}$($\sigma_{0}$/m/cm$^{2}$g$^{-1}$)"]

    nfw_params = convert_params(flat_samples)

    c200 = np.median(nfw_params[:, 0])
    logM200 = np.median(nfw_params[:, 1])
    log10sigma0 = np.median(nfw_params[:, 2])
    log10N0 = np.median(flat_samples[:, 0])
    log10v0 = np.median(flat_samples[:, 1])
    ns0 = np.median(flat_samples[:, 2])

    mcmc_sol = np.array([log10N0, log10v0, ns0, log10sigma0, logM200, np.log10(c200)])
    output_file = output_folder + "MCMC_Output_" + name + ".txt"
    output_fig = output_folder + "MCMC_fit_" + name + ".png"
    plot_solution(x, 10**y, 10**errorbar, mcmc_sol, output_fig, output_file)

    # Make the base corner plot
    figure = corner.corner(
        nfw_params, labels=labels, quantiles=[0.16, 0.84],
        truths=[c200, logM200, log10sigma0], show_titles=True,
        title_kwargs={"fontsize": 16}
    )

    output_corner_plot = output_folder + "corner_nfw_" + name + ".png"
    plt.savefig(output_corner_plot, dpi=200)

    # Output bestfit paramter range
    sol_median = np.array(
        [10 ** log10N0, 10 ** log10v0, ns0, 10 ** log10sigma0, logM200, c200])

    log10N0 = np.percentile(flat_samples[:, 0], 84)
    log10v0 = np.percentile(flat_samples[:, 1], 84)
    ns0 = np.percentile(flat_samples[:, 2], 84)
    log10sigma0 = np.percentile(flat_samples[:, 3], 84)
    log10M200 = np.percentile(nfw_params[:, 1], 84)
    c200 = np.percentile(nfw_params[:, 0], 84)
    sol_upper = np.array(
        [10 ** log10N0, 10 ** log10v0, ns0, 10 ** log10sigma0, log10M200, c200])

    log10N0 = np.percentile(flat_samples[:, 0], 16)
    log10v0 = np.percentile(flat_samples[:, 1], 16)
    ns0 = np.percentile(flat_samples[:, 2], 16)
    log10sigma0 = np.percentile(flat_samples[:, 3], 16)
    log10M200 = np.percentile(nfw_params[:, 1], 16)
    c200 = np.percentile(nfw_params[:, 0], 16)
    sol_lower = np.array(
        [10 ** log10N0, 10 ** log10v0, ns0, 10 ** log10sigma0, log10M200, c200])

    output_file = output_folder + "MCMC_parameter_range_" + name + ".txt"

    output_best_fit_params(sol_median, sol_upper, sol_lower, output_file)
