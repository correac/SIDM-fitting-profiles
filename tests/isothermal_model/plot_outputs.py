import numpy as np
from pylab import *

def plot_joint_outputs():

    # Plot parameters
    params = {
        "font.size": 12,
        "font.family":"Times",
        "text.usetex": True,
        "figure.figsize": (4, 3),
        "figure.subplot.left": 0.18,
        "figure.subplot.right": 0.95,
        "figure.subplot.bottom": 0.18,
        "figure.subplot.top": 0.95,
        "figure.subplot.wspace": 0.25,
        "figure.subplot.hspace": 0.25,
        "lines.markersize": 3,
        "lines.linewidth": 1.5,
        "figure.max_open_warning": 0,
    }
    rcParams.update(params)

    #######################
    # Do a nice plot
    figure()
    ax = plt.subplot(1, 1, 1)
    grid(True)

    name = 'DML006N188_SigmaConstant00_M'
    vpair, sigma_vpair = read_data_joint(name)

    yerrl = sigma_vpair[0,:]-sigma_vpair[2,:]
    yerrp = sigma_vpair[1,:]-sigma_vpair[0,:]
    xerrl = vpair[0,:]-vpair[2,:]
    xerrp = vpair[1,:]-vpair[0,:]
    errorbar(vpair[0,:], sigma_vpair[0,:], yerr=[yerrl,yerrp] ,xerr=[xerrl,xerrp] ,fmt='o',
             color='lightblue', ecolor='lightblue', alpha=0.5,label='DML006N188/Sigma 0 cm$^{2}$/g')

    name = 'DML006N188_SigmaConstant01_M'
    vpair, sigma_vpair = read_data_joint(name)

    yerrl = sigma_vpair[0,:]-sigma_vpair[2,:]
    yerrp = sigma_vpair[1,:]-sigma_vpair[0,:]
    xerrl = vpair[0,:]-vpair[2,:]
    xerrp = vpair[1,:]-vpair[0,:]
    errorbar(vpair[0,:], sigma_vpair[0,:] ,yerr=[yerrl,yerrp], xerr=[xerrl,xerrp], fmt='o',
             color='orange', ecolor='orange', alpha=0.5,label='DML006N188/Sigma 1 cm$^{2}$/g')

    # x = np.arange(10,101,0.5)
    # y = cross_section(x, 108, 33.35)
    # plot(x,y, '-', lw=1, color='black')

    axis([0,60,1e-1,1e2])
    yscale('log')
    ylabel(r'$\sigma_{T}/m_{\chi}$ [cm$^{2}$/g]')
    xlabel(r'$\langle v_{\mathrm{pair}}\rangle$ [km/s]')
    plt.legend(loc="upper right", labelspacing=0.2, handlelength=1.5, handletextpad=0.4, frameon=False)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig('test_joint.png', dpi=200)

def c_M_relation(log_M0):
    """
    Concentration-mass relation from Correa et al. (2015).
    This relation is most suitable for Planck cosmology.
    """
    z = 0
    # Best-fit params:
    alpha = 1.7543 - 0.2766 * (1. + z) + 0.02039 * (1. + z) ** 2
    beta = 0.2753 + 0.0035 * (1. + z) - 0.3038 * (1. + z) ** 0.0269
    gamma = -0.01537 + 0.02102 * (1. + z) ** (-0.1475)

    log_10_c200 = alpha + beta * log_M0 * (1. + gamma * log_M0 ** 2)
    c200 = 10 ** log_10_c200
    return c200


def read_data_joint(name):

    num_halos = 3
    sigma_vpair = np.zeros((3,num_halos))
    vpair = np.zeros((3,num_halos))
    c200 = np.zeros((3,num_halos))
    M200 = np.zeros((3,num_halos))

    for i in range(0, num_halos):
        if i == 0: file = './output/MCMC_parameter_range_'+name+'9.0.txt'
        if i == 1: file = './output/MCMC_parameter_range_'+name+'9.5.txt'
        if i == 2: file = './output/MCMC_parameter_range_'+name+'10.0.txt'
        data = np.loadtxt(file, usecols=(1,2,3))
        sigma_vpair[:,i] = data[4,:]
        vpair[:,i] = data[3,:]

    return vpair, sigma_vpair

def read_data(name, num_halos):

    sigma_vpair = np.zeros((3,num_halos))
    vpair = np.zeros((3,num_halos))
    #c200 = np.zeros((3,num_halos))
    #M200 = np.zeros((3,num_halos))

    output_folder = './output/cartesius/'

    # for i in range(0, num_halos):
    #     file = './output/Individual_sample/MCMC_parameter_range_'+name+'_%i.txt'%i
    #     data = np.loadtxt(file, usecols=(1,2,3))
    #     sigma_vpair[:,i] = data[4,:]
    #     vpair[:,i] = data[3,:]
    #     c200[:,i] = data[0,:]
    #     M200[:,i] = data[1,:]

    for i in range(0, num_halos):
        file = output_folder + "samples_" + name + "_%i.txt"%i
        data = np.loadtxt(file)
        v0 = np.median(10**data[:, 1]) * 4 / np.sqrt(np.pi)
        low_v0 = np.percentile(10**data[:, 1], 16) * 4 / np.sqrt(np.pi)
        up_v0 = np.percentile(10**data[:, 1], 84) * 4 / np.sqrt(np.pi)
        sigma0 = np.median(10**data[:, 2])
        low_sigma0 = np.percentile(10**data[:, 2], 16)
        up_sigma0 = np.percentile(10**data[:, 2], 84)

        vpair[:, i] = np.array([v0, up_v0, low_v0])
        sigma_vpair[:, i] = np.array([sigma0, up_sigma0, low_sigma0])

    yerrl = sigma_vpair[0,:]-sigma_vpair[2,:]
    yerrp = sigma_vpair[1,:]-sigma_vpair[0,:]
    xerrl = vpair[0,:]-vpair[2,:]
    xerrp = vpair[1,:]-vpair[0,:]

    xerr = np.array([xerrl, xerrp])
    yerr = np.array([yerrl, yerrp])
    x = vpair[0,:]
    y = sigma_vpair[0,:]

    error_range = np.sqrt(yerrl**2+yerrp**2)
    #select = np.where((error_range<100)&(error_range>1e-3))[0]
    select = np.where(error_range>1e-3)[0]
    return x[select], y[select], xerr[:,select], yerr[:,select]


def plot_outputs():

    # Plot parameters
    params = {
        "font.size": 12,
        "font.family":"Times",
        "text.usetex": True,
        "figure.figsize": (4, 3),
        "figure.subplot.left": 0.18,
        "figure.subplot.right": 0.95,
        "figure.subplot.bottom": 0.18,
        "figure.subplot.top": 0.95,
        "figure.subplot.wspace": 0.25,
        "figure.subplot.hspace": 0.25,
        "lines.markersize": 3,
        "lines.linewidth": 1,
        "figure.max_open_warning": 0,
    }
    rcParams.update(params)

    #######################
    # Do a nice plot
    figure()
    ax = plt.subplot(1, 1, 1)
    grid(True)

    name = 'DML006N188_SigmaConstant01_M10.0'
    x, y, xerr, yerr = read_data(name, 19)
    errorbar(x, y ,yerr=yerr, xerr=xerr, fmt='o', ecolor='tab:blue',color='tab:blue', alpha=0.5)

    name = 'DML006N188_SigmaConstant01_M10.5'
    x, y, xerr, yerr = read_data(name, 9)
    errorbar(x, y ,yerr=yerr, xerr=xerr, fmt='o', ecolor='tab:blue',color='tab:blue', alpha=0.5, label='DML006N188')

    name = 'RefL006N188_SigmaConstant01_M10.0'
    x, y, xerr, yerr = read_data(name, 16)
    errorbar(x, y ,yerr=yerr, xerr=xerr, fmt='v', ecolor='tab:orange',color='tab:orange', alpha=0.5, label='RefL006N188')

    name = 'RefL006N188_SigmaConstant01_M10.5'
    x, y, xerr, yerr = read_data(name, 6)
    errorbar(x, y ,yerr=yerr, xerr=xerr, fmt='v', ecolor='tab:orange',color='tab:orange', alpha=0.5)

    plot(np.array([0,150]), np.array([1,1]), '--', lw=1, color='black')

    axis([0,150,1e-1,1e3])
    yscale('log')
    ylabel(r'$\sigma/m_{\chi}$ [cm$^{2}$/g]')
    xlabel(r'$\langle v_{\mathrm{pair}}\rangle$ [km/s]')
    plt.legend(loc="upper left", labelspacing=0.2, handlelength=1.5, handletextpad=0.4, frameon=False)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig('cross_section.png', dpi=200)


    #######################
    # Do a nice plot
    figure()
    ax = plt.subplot(1, 1, 1)
    grid(True)

    name = 'DML006N188_SigmaConstant10_M10.5'
    x, y, xerr, yerr = read_data(name, 7)
    errorbar(x, y ,yerr=yerr, xerr=xerr, fmt='v', ecolor='tab:red',color='tab:red', alpha=0.5, label='DML006N188/Sigma10')

    name = 'DML006N188_SigmaConstant10_M10.0'
    x, y, xerr, yerr = read_data(name, 8)
    errorbar(x, y ,yerr=yerr, xerr=xerr, fmt='o', ecolor='tab:red',color='tab:red', alpha=0.5)

    plot(np.array([0,150]), np.array([10,10]), '--', lw=1, color='black')

    axis([0,150,1e-1,1e3])
    yscale('log')
    ylabel(r'$\sigma/m_{\chi}$ [cm$^{2}$/g]')
    xlabel(r'$\langle v_{\mathrm{pair}}\rangle$ [km/s]')
    plt.legend(loc="upper left", labelspacing=0.2, handlelength=1.5, handletextpad=0.4, frameon=False)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig('cross_section_sigma10.png', dpi=200)

    # ##########################
    # # Plot some plot
    # figure()
    # ax = plt.subplot(1, 1, 1)
    # grid(True)
    #
    # name = 'DML006N188_SigmaConstant01_M9'
    # #num_halos = 5
    # _, _, M200, c200 = read_data(name, num_halos)
    #
    # yerrl = c200[0,:]-c200[2,:]
    # yerrp = c200[1,:]-c200[0,:]
    # errorbar(M200[0,:], c200[0,:] ,yerr=[yerrl,yerrp], fmt='o', ecolor='lightblue', color='lightblue', alpha=0.5)
    #
    # name = 'DML006N188_SigmaConstant01_M9.5'
    # num_halos = 11
    # _, _, M200, c200 = read_data(name, num_halos)
    #
    # yerrl = c200[0,:]-c200[2,:]
    # yerrp = c200[1,:]-c200[0,:]
    # errorbar(M200[0,:], c200[0,:] ,yerr=[yerrl,yerrp], fmt='o', ecolor='green', color='green', alpha=0.5)
    #
    # name = 'DML006N188_SigmaConstant01_M10.0'
    # num_halos = 11
    # _, _, M200, c200 = read_data(name, num_halos)
    #
    # yerrl = c200[0,:]-c200[2,:]
    # yerrp = c200[1,:]-c200[0,:]
    # errorbar(M200[0,:], c200[0,:] ,yerr=[yerrl,yerrp], fmt='o', ecolor='orange',color='orange', alpha=0.5)
    #
    #
    # M0 = np.arange(8,13,0.2)
    # c0 = c_M_relation(M0)
    # plot(M0, c0, '--', lw=1, color='black')
    #
    # axis([8,11,0,30])
    # ylabel(r'$c_{200}$')
    # xlabel(r'$\log_{10}M_{200}~[M_{\odot}]$')
    # #plt.legend(loc="lower left", labelspacing=0.2, handlelength=1.5, handletextpad=0.4, frameon=False)
    # ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    # plt.savefig('concentration_mass_relation.png', dpi=200)





if __name__ == '__main__':
    plot_outputs()
    #plot_joint_outputs()