import numpy as np
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from plotter import plot_solution_default
from mcmc import log_likelihood_default, run_mcmc_default
from functions import fit_isothermal_model, fit_modified_isothermal_model, calculate_log_v0, calculate_log_N0
from functions import read_data, read_single_halo, c_M_relation, find_nfw_params

import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

if __name__ == '__main__':

    from utils import *

    input_file = args.input
    output_folder = args.output
    name = args.name
    op_density = args.density_model
    op_sigma = args.cross_section_model
    op_sample = args.halo_sample

    # Output data
    output_file = output_folder+"Output_"+name+".txt"
    output_plot_final = output_folder+name+"_fit.png"

    # Read initial data
    if op_sample == 'joint':
        x, y, yerr, errorbar = read_data(input_file)
    else :
        x, y, yerr, errorbar = read_single_halo(input_file)

    if op_density == "modified-isothermal":

        ns0 = args.variable

        # First fit profile based on Isothermal model
        popt, pcov = curve_fit(fit_modified_isothermal_model, x, y, p0=[1, 10, ns0])
        r0 = popt[0]
        rho0 = 10 ** popt[1]
        ns0 = popt[2]

        # Initial cross section input
        sigma0 = args.sigma
        log10sigma0 = np.log10(sigma0)
        w0 = args.wvel
        log10w0 = np.log10(w0)

        # Log10 values of free params
        log10v0 = calculate_log_v0(r0, rho0)
        v0 = 10 ** log10v0
        log10N0 = calculate_log_N0(rho0, v0, ns0, sigma0, w0, op_sigma)
        N0 = 10 ** log10N0

        # Initial guess
        log10M200 = 10.
        log10c200 = np.log10(c_M_relation(10 ** log10M200))

        # Find M200, c200 params
        log10M200, log10c200 = find_nfw_params(N0, v0, ns0, sigma0, w0, log10M200, log10c200, op_sigma)

        print("======")
        print("Initial estimates:")
        print("rho0 = {0:.3f}".format(np.log10(rho0)))
        print("r0 = {0:.3f}".format(r0))
        print("ns0 = {0:.3f}".format(ns0))
        print("N0 = {0:.3f}".format(log10N0))
        print("v0 = {0:.3f}".format(log10v0))
        print("M200 = {0:.3f}".format(log10M200))
        print("c200 = {0:.3f}".format(10 ** log10c200))
        print("======")

        if op_sigma == "velocity-dependent":

            np.random.seed(42)
            nll = lambda *args: -log_likelihood_default(*args)
            initial = np.array([log10N0, log10v0, ns0, log10sigma0, log10w0, log10M200, log10c200])
            soln = minimize(nll, initial, args=(x, y, yerr))
            log10N0_ml, log10v0_ml, ns0_ml, log10sigma0_ml, log10w0_ml, log10M200_ml, log10c200_ml = soln.x

            print("======")
            print("Maximum likelihood estimates:")
            print("N0 = {0:.3f}".format(log10N0_ml))
            print("v0 = {0:.3f}".format(log10v0_ml))
            print("ns0 = {0:.3f}".format(ns0_ml))
            print("sigma0 = {0:.3f}".format(log10sigma0_ml))
            print("w0 = {0:.3f}".format(log10w0_ml))
            print("M200 = {0:.3f}".format(log10M200_ml))
            print("c200 = {0:.3f}".format(10**log10c200_ml))
            print("======")

            plot_solution_default(x, 10**y, 10**errorbar, soln.x, output_plot_final, output_file)

            run_mcmc_default(soln, x, y, yerr, errorbar, name, output_folder)

