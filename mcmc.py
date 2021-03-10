import numpy as np
import emcee
import corner
from pylab import *
from sidm_model import sidm_halo_model
from plotter import model_params,calculate_additional_params

def log_prior(theta):
    """
    The natural logarithm of the prior probability.

    It sets prior to 1 (log prior to 0) if params are in range, and zero (-inf) otherwise.

    Args:
        theta (tuple): a sample containing individual parameter values
    """
    N0, v0, sigma0 = theta

    if 0 < N0 < 5 and -1. < v0 < 3.5 and -2. < sigma0 < 2.0:
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
    N0, v0, sigma0 = theta
    model = sidm_halo_model(x, 10**N0, 10**v0, 10**sigma0)
    sigma2 = yerr**2
    log_l = -0.5 * np.sum((y - model) ** 2 / sigma2)
    return log_l

def run_mcmc(soln, x, y, yerr):

    N0, v0, sigma0 = soln.x

    r1, rho1, r0, rho0, rs, rhos = model_params(N0, v0, sigma0)
    c200, M200 = calculate_additional_params(rs, rhos)

    param_space = np.array([c200, M200, sigma0])

    #pos = soln.x + 1e-4 * np.random.randn(32, 3)
    pos = param_space + 1e-4 * np.random.randn(32, 3)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(x, y, yerr))
    sampler.run_mcmc(pos, 500, progress=True)

    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    labels = ["N0", "v0", "sigma0"]

    fig = corner.corner(
        flat_samples, labels=labels, truths=[c200, M200, sigma0]
    )

    plt.savefig('corner.png')