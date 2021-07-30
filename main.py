import numpy as np
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from plotter import plot_solution, plot_isothermal_fit
from mcmc import log_likelihood, run_mcmc
from functions import fit_isothermal_model, calculate_log_v0, calculate_log_N0
from functions import read_data, read_single_halo

import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

if __name__ == '__main__':

    from utils import *

    input_file = args.input
    output_folder = args.output
    name = args.name

    # Output data
    output_file = output_folder+"Output_"+name+".txt"
    #output_plot_initial = output_folder+name+"_initial_fit.png"
    #output_plot_veldisp = output_folder+name+"_velocity_dispersion.png"
    output_plot_final = output_folder+name+"_fit.png"
    output_corner_plot = output_folder+"corner_"+name+".png"

    # Read initial data
    #x, y, yerr, errorbar = read_data(input_file)
    x, y, yerr, errorbar = read_single_halo(input_file)

    # Initial cross section input
    sigma0 = args.sigma
    sigma0 = np.log10(sigma0)
    ns0 = args.variable
    w0 = args.wvel
    w0 = np.log10(w0)

    # First fit profile based on Isothermal model
    popt, pcov = curve_fit(fit_isothermal_model, x, y, p0=[1, 10, ns0])
    r0 = popt[0]
    rho0 = popt[1]
    ns0 = popt[2]
    print("Initial estimates:")
    print("rho0 = {0:.3f}".format(rho0))
    print("r0 = {0:.3f}".format(r0))
    print("ns0 = {0:.3f}".format(ns0))

    # Log10 values of free params
    v0 = calculate_log_v0(r0, 10**rho0)
    N0 = calculate_log_N0(10**rho0, 10**v0, ns0, 10**sigma0, 10**w0)
    print("N0 = {0:.3f}".format(N0))
    print("v0 = {0:.3f}".format(v0))
    print("======")

    # First fit profile based on Isothermal model
    # plot_isothermal_fit(x, 10**y, 10**errorbar, 10**N0, 10**v0, ns0, 10**sigma0, 10**w0,
    #                    output_plot_initial, output_plot_veldisp)


    np.random.seed(42)
    nll = lambda *args: -log_likelihood(*args)
    initial = np.array([N0, v0, ns0, sigma0, w0])
    soln = minimize(nll, initial, args=(x, y, yerr))
    N0_ml, v0_ml, ns0_ml, sigma0_ml, w0_ml = soln.x

    print("Maximum likelihood estimates:")
    print("N0 = {0:.3f}".format(N0_ml))
    print("v0 = {0:.3f}".format(v0_ml))
    print("ns0 = {0:.3f}".format(ns0_ml))
    print("sigma0 = {0:.3f}".format(sigma0_ml))
    print("w0 = {0:.3f}".format(w0_ml))

    print("======")
    plot_solution(x, 10**y, 10**errorbar, soln, output_plot_final, output_file)

    run_mcmc(soln, x, y, yerr, output_corner_plot)
