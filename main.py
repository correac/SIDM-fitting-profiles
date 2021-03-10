import numpy as np
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from plotter import plot_solution, plot_isothermal_fit
from mcmc import log_likelihood, run_mcmc
from functions import fit_isothermal_model, calculate_log_v0, calculate_log_N0
from functions import read_data

import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')


if __name__ == '__main__':


    # Read initial data
    #file_name = "../../../FermionDSph/Code/Final_Data/Carina/output_rho.txt"
    file_name = "./data/Profile_halo_M12_L025N376_sigma_10.txt"

    # Output data
    output_folder = './output/'
    output_file = output_folder+"Output_M12_L025N376_sigma_10.txt"
    output_plot_initial = output_folder+"M12_L025N376_sigma_10_initial_fit.png"
    output_plot_final = output_folder+"M12_L025N376_sigma_10_final_fit.png"
    output_corner_plot = output_folder+"corner_M12_L025N376_sigma_10.png"

    x, y, yerr, errorbar = read_data(file_name)

    # First fit profile based on Isothermal model
    popt, pcov = curve_fit(fit_isothermal_model, x, y)
    r0 = popt[0]
    rho0 = popt[1]
    print("Initial estimates:")
    print("rho0 = {0:.3f}".format(rho0))
    print("r0 = {0:.3f}".format(r0))

    # Log10 values of free params
    sigma0 = 0.0
    v0 = calculate_log_v0(r0, 10**rho0)
    N0 = calculate_log_N0(10**v0, 10**rho0, 10**sigma0)
    print("N0 = {0:.3f}".format(N0))
    print("v0 = {0:.3f}".format(v0))
    print("======")

    plot_isothermal_fit(x, 10**y, 10**errorbar, 10**N0, 10**v0, 10**sigma0, output_plot_initial)

    # First fit profile based on Isothermal model
    np.random.seed(42)
    nll = lambda *args: -log_likelihood(*args)
    initial = np.array([N0, v0, sigma0])
    soln = minimize(nll, initial, args=(x, y, yerr))
    N0_ml, v0_ml, sigma0_ml = soln.x

    print("Maximum likelihood estimates:")
    print("N0 = {0:.3f}".format(N0_ml))
    print("v0 = {0:.3f}".format(v0_ml))
    print("sigma0 = {0:.3f}".format(sigma0_ml))

    print("======")
    plot_solution(x, 10**y, 10**errorbar, soln, output_plot_final, output_file)

    run_mcmc(soln, x, y, yerr,output_corner_plot)