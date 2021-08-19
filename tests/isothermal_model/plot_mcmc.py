import numpy as np
from plotter import plot_solution
from functions import read_single_halo
from functions import find_nfw_params, c_M_relation
from pylab import *
from scipy.stats import mode

def convert_params(theta):
    """
        Returns c200, Log10M200 and Log10sigma0
        based on sampling from Log10N0, Log10v0 and Log10sigma0.
    Args:
        theta (tuple): Log10N0, Log10v0 and Log10sigma0
    """
    N0 = theta[0]
    v0 = theta[1]
    sigma0 = theta[2]

    M200, c200 = find_nfw_params(10**N0, 10**v0, 10**sigma0, 10., np.log10(c_M_relation(10.)))

    params = np.array([10**c200, M200, sigma0])
    return params.T

def make_mcmc_plots(x, y, errorbar, output_folder, name):

    output_data = output_folder + "samples_" + name + ".txt"

    flat_samples = np.loadtxt(output_data)

    labels = ["c$_{200}$",
              "log$_{10}$(M$_{200}$/M$_{\odot}$)",
              "log$_{10}$($\sigma_{0}$/m/cm$^{2}$g$^{-1}$)"]

    median_samples = np.array([np.median(flat_samples[:,0]), np.median(flat_samples[:,1]), np.median(flat_samples[:,2])])
    nfw_params = convert_params(median_samples)

    c200, logM200, log10sigma0 = nfw_params
    log10N0, log10v0 = median_samples[0:2]

    # c200 = np.median(nfw_params[:, 0])
    # logM200 = np.median(nfw_params[:, 1])
    # log10sigma0 = np.median(nfw_params[:, 2])
    # log10N0 = np.median(median_samples[:, 0])
    # log10v0 = np.median(median_samples[:, 1])

    mcmc_sol = np.array([log10N0, log10v0, log10sigma0, logM200, np.log10(c200)])
    output_file = output_folder + "MCMC_Output_" + name + ".txt"
    output_fig = output_folder + "MCMC_fit_" + name + ".png"
    plot_solution(x, 10**y, 10**errorbar, mcmc_sol, output_fig, output_file)

    # # Make the base corner plot
    # figure = corner.corner(
    #     nfw_params, labels=labels, quantiles=[0.16, 0.84],
    #     truths=[c200, logM200, log10sigma0], show_titles=True,
    #     title_kwargs={"fontsize": 16}
    # )
    #
    # output_corner_plot = output_folder + "corner_nfw_" + name + ".png"
    # plt.savefig(output_corner_plot, dpi=200)

    # Output bestfit paramter range
    # sol_median = np.array(
    #     [10 ** log10N0, 10 ** log10v0, 10 ** log10sigma0, logM200, c200])
    #
    # log10N0 = np.percentile(flat_samples[:, 0], 84)
    # log10v0 = np.percentile(flat_samples[:, 1], 84)
    # log10M200 = np.percentile(nfw_params[:, 1], 84)
    # c200 = np.percentile(nfw_params[:, 0], 84)
    # log10sigma0 = np.percentile(flat_samples[:, 2], 84)
    # sol_upper = np.array(
    #     [10 ** log10N0, 10 ** log10v0, 10 ** log10sigma0, log10M200, c200])
    #
    # log10N0 = np.percentile(flat_samples[:, 0], 16)
    # log10v0 = np.percentile(flat_samples[:, 1], 16)
    # log10sigma0 = np.percentile(flat_samples[:, 2], 16)
    # log10M200 = np.percentile(nfw_params[:, 1], 16)
    # c200 = np.percentile(nfw_params[:, 0], 16)
    # sol_lower = np.array(
    #     [10 ** log10N0, 10 ** log10v0, 10 ** log10sigma0, log10M200, c200])
    #
    # output_file = output_folder + "MCMC_parameter_range_" + name + ".txt"
    #
    # output_best_fit_params(sol_median, sol_upper, sol_lower, output_file)


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
    else:
        x, y, yerr, errorbar = read_single_halo(input_file)

    make_mcmc_plots(x, y, errorbar, output_folder, name)