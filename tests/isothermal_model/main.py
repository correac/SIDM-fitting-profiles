import numpy as np
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from plotter import plot_solution
from mcmc import log_ml, run_mcmc, find_initial_sigma0
from functions import fit_isothermal_model, calculate_log_v0, calculate_log_N0, calculate_log_sigma0
from functions import read_data, read_single_halo
from functions import find_nfw_params, c_M_relation

import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

if __name__ == '__main__':

    from utils import *

    input_file = args.input
    output_folder = args.output
    name = args.name
    op_sample = args.halosample


    # Output data
    output_file = output_folder+"Output_"+name+".txt"
    output_plot_final = output_folder+"Initial_fit"+name+".png"
    output_corner_plot = output_folder+"corner_"+name+".png"

    # Read initial data
    if op_sample == 'joint':
        x, y, yerr, errorbar = read_data(input_file)
    else :
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
    sigma0 = calculate_log_sigma0(10**rho0, 10**v0, 10)
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

    print("======")
    print("Maximum likelihood estimates:")
    print("N0 = {0:.3f}".format(N0_ml))
    print("v0 = {0:.3f}".format(v0_ml))
    print("sigma0 = {0:.3f}".format(10**sigma0_ml))
    print("M200 = {0:.3f}".format(logM200))
    print("c200 = {0:.3f}".format(10**logc200))
    print("======")

    plot_solution(x, 10**y, 10**errorbar, sol, output_plot_final, output_file)

    print("Running mcmc sampler..")
    run_mcmc(soln, x, y, yerr, errorbar, name, output_folder)
