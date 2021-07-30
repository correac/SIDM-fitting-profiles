import numpy as np
import emcee
import corner
from pylab import *
from sidm_model import sidm_halo_model
from plotter import model_params,calculate_additional_params

from multiprocessing import Pool
import time

# If runing with mpi uncomment ...
#from schwimmbad import MPIPool

def convert_params(theta):
    """
        Returns c200, Log10M200 and Log10sigma0
        based on sampling from Log10N0, Log10v0 and Log10sigma0.
    Args:
        theta (tuple): Log10N0, Log10v0 and Log10sigma0
    """
    N0 = theta[:,0]
    v0 = theta[:,1]
    sigma0 = theta[:,2]
    ns0 = theta[:,3]
    w0 = theta[:,4]

    M200 = np.zeros(len(N0))
    c200 = np.zeros(len(N0))

    for i in range(len(N0)):

        r1, rho1, r0, rho0, rs, rhos = model_params(10**N0[i], 10**v0[i], ns0[i], 10**sigma0[i], 10**w0[i])
        c200[i], M200[i] = calculate_additional_params(rs, rhos)

    params = np.array([c200, M200, ns0, sigma0, w0])

    return params.T

def log_prior(theta):
    """
    The natural logarithm of the prior probability.

    It sets prior to 1 (log prior to 0) if params are in range, and zero (-inf) otherwise.

    Args:
        theta (tuple): a sample containing individual parameter values
    """
    N0, v0, ns0, sigma0, w0 = theta

    if 0 < N0 < 5 and -1. < v0 < 3.5 and -2. < sigma0 < 4. and -0.5 < ns0 < 0.5 and -2. < w0 < 3.:
        return 0.0
    return -np.inf

def log_posterior(theta, x, y, yerr):
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
    return lp + log_likelihood(theta, x, y, yerr)

def log_likelihood(theta, x, y, yerr):
    """
    The natural logarithm of the joint likelihood.

    Args:
        theta (tuple): a sample containing individual parameter values
        x (array): values over which the data/model is defined
        y (array): the set of data
        yerr (array): the standard deviation of the data points
    """
    N0, v0, ns0, sigma0, w0 = theta
    model = sidm_halo_model(x, 10**N0, 10**v0, ns0, 10**sigma0, 10**w0)
    sigma2 = yerr**2
    log_l = -0.5 * np.sum((y - model) ** 2 / sigma2)
    return log_l



def run_mcmc(soln, x, y, yerr, output_file):

    N0, v0, ns0, sigma0, w0 = soln.x

    r1, rho1, r0, rho0, rs, rhos = model_params(10**N0, 10**v0, ns0, 10**sigma0, 10**w0)
    c200_ml, M200_ml = calculate_additional_params(rs, rhos)

    pos = soln.x + 1e-4 * np.random.randn(64, 5)
    nwalkers, ndim = pos.shape

    # If runing with mpi ...
    #with MPIPool() as pool:
    #    if not pool.is_master():
    #        pool.wait()
    #        sys.exit(0)

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(x, y, yerr), pool=pool)
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
              "log$_{10}$($w_{0}$/km/s)"]

    # Make the base corner plot
    figure = corner.corner(
        flat_samples, labels=labels, quantiles=[0.16, 0.5, 0.84],
        truths=[N0, v0, ns0, sigma0, w0], show_titles=True,
        title_kwargs={"fontsize": 16}
        #range=[(0, 50), (9, 12), (-1, 1), (-2, 2), (-2, 3)]
    )

    plt.savefig("flat_samples.png", dpi=200)

    # labels = ["c$_{200}$",
    #           "log$_{10}$(M$_{200}$/M$_{\odot}$)",
    #           "n$_{s}$",
    #           "log$_{10}$($\sigma_{0}$/m/cm$^{2}$g$^{-1}$)",
    #           "log$_{10}$($w_{0}$/km/s)"]
    #
    # nfw_params = convert_params(flat_samples)
    #
    # # Make the base corner plot
    # figure = corner.corner(
    #     nfw_params, labels=labels, quantiles=[0.16, 0.5, 0.84],
    #     truths=[c200_ml, M200_ml, ns0, sigma0, w0], show_titles=True,
    #     title_kwargs={"fontsize": 16},
    #     range=[(0,50),(9,12),(-1,1),(-2,2),(-2,3)]
    # )

    # Extract the axes
    #axes = np.array(figure.axes).reshape((ndim, ndim))
    #median_value = np.mean(nfw_params, axis=0)

    # Loop over the diagonal
    #for i in range(ndim):
    #    ax = axes[i, i]
    #    ax.axvline(median_value[i], color="tab:red")

    # Loop over the histograms
    #for yi in range(ndim):
    #    for xi in range(yi):
    #        ax = axes[yi, xi]
    #        ax.axvline(median_value[xi], color="tab:red")
    #        ax.axhline(median_value[yi], color="tab:red")
    #        ax.plot(median_value[xi], median_value[yi], "sr")

    # plt.savefig(output_file, dpi=200)