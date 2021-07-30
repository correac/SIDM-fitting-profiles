import numpy as np
from pylab import *

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
        sigma_vpair[:,i] = data[8,:]
        vpair[:,i] = data[7,:]

    return vpair, sigma_vpair

def read_data(name, num_halos):

    sigma_vpair = np.zeros((3,num_halos))
    vpair = np.zeros((3,num_halos))
    c200 = np.zeros((3,num_halos))
    M200 = np.zeros((3,num_halos))

    for i in range(0, num_halos):
        file = './output/MCMC_parameter_range_'+name+'_%i.txt'%i
        data = np.loadtxt(file, usecols=(1,2,3))
        sigma_vpair[:,i] = data[8,:]
        vpair[:,i] = data[7,:]
        c200[:,i] = data[0,:]
        M200[:,i] = data[1,:]

    return vpair, sigma_vpair, M200, c200

def cross_section(x, sigma0, w0):
    f = 4 * sigma0 * w0 ** 4 / x ** 4
    f *= (2 * np.log(1. + 0.5 * x ** 2 / w0 ** 2) - np.log(1. + x ** 2 / w0 ** 2))
    return f

def error_cross_section(x, sigma0, w0, err_sigma0, err_w0):
    f = cross_section(x, sigma0, w0)
    dfdsigma0 = f / sigma0
    term = 1./ (1 + 0.5 * x**2/w0**2) - 1/(1. + x**2/w0**2)
    dfdw0 = 4 * f / w0 + 8 * sigma0 * (w0**2/ x**2) * term / w0

    error = np.sqrt( dfdsigma0**2 * err_sigma0**2 + dfdw0**2 * err_w0**2)
    return error

def read_other_data(name, num_halos):

    sigma_vpair = np.zeros((3,num_halos))
    vpair = np.zeros(num_halos)

    for i in range(0, num_halos):
        file = './output/MCMC_parameter_range_'+name+'_%i.txt'%i
        data = np.loadtxt(file, usecols=(1,2,3))
        vpair[i] = data[7,0]

        sigmaT0 = data[5,:]
        w0 = data[6,:]

        sigma = cross_section(vpair[i],sigmaT0[0],w0[0])
        sigma_vpair[0,i] = sigmaT0[0]

        err_sigma0 = sigmaT0[1]-sigmaT0[0]
        err_w0 = w0[1]-w0[0]
        error_high = error_cross_section(vpair[i], sigmaT0[0], w0[0], err_sigma0, err_w0)
        sigma_vpair[1,i] = error_high

        err_sigma0 = sigmaT0[0]-sigmaT0[2]
        err_w0 = w0[0]-w0[2]
        error_low = error_cross_section(vpair[i], sigmaT0[0], w0[0], err_sigma0, err_w0)
        sigma_vpair[2,i] = error_low

    return vpair, sigma_vpair

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
        "lines.linewidth": 1.5,
        "figure.max_open_warning": 0,
    }
    rcParams.update(params)

    #######################
    # Do a nice plot
    figure()
    ax = plt.subplot(1, 1, 1)
    grid(True)

    name = 'DML006N188_SigmaVel100_M9.5'
    num_halos = 40
    vpair, sigma_vpair, _, _ = read_data(name, num_halos)

    yerrl = sigma_vpair[0,:]-sigma_vpair[2,:]
    yerrp = sigma_vpair[1,:]-sigma_vpair[0,:]
    xerrl = vpair[0,:]-vpair[2,:]
    xerrp = vpair[1,:]-vpair[0,:]
    errorbar(vpair[0,:], sigma_vpair[0,:], yerr=[yerrl,yerrp] ,xerr=[xerrl,xerrp] ,fmt='o', ecolor='lightblue', alpha=0.5)

    name = 'DML006N188_SigmaVel100_M9.0'
    num_halos = 7
    vpair, sigma_vpair, M200, c200 = read_data(name, num_halos)

    yerrl = sigma_vpair[0,:]-sigma_vpair[2,:]
    yerrp = sigma_vpair[1,:]-sigma_vpair[0,:]
    xerrl = vpair[0,:]-vpair[2,:]
    xerrp = vpair[1,:]-vpair[0,:]
    errorbar(vpair[0,:], sigma_vpair[0,:] ,yerr=[yerrl,yerrp], xerr=[xerrl,xerrp], fmt='o', ecolor='orange', alpha=0.5)

    x = np.arange(10,55,0.5)
    y = cross_section(x, 108, 33.35)
    plot(x,y, '-', lw=1, color='black')

    axis([10,50,1e-1,1e3])
    yscale('log')
    ylabel(r'$\sigma_{T}/m_{\chi}$ [cm$^{2}$/g]')
    xlabel(r'$\langle v_{\mathrm{pair}}\rangle$ [km/s]')
    #plt.legend(loc="lower left", labelspacing=0.2, handlelength=1.5, handletextpad=0.4, frameon=False)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig('test.png', dpi=200)

    #######################
    # Do a nice plot
    figure()
    ax = plt.subplot(1, 1, 1)
    grid(True)

    name = 'DML006N188_SigmaVel100_M9.5'
    num_halos = 40
    vpair, sigma_vpair = read_other_data(name, num_halos)
    errorbar(vpair, sigma_vpair[0,:] ,yerr=sigma_vpair[1:3,:], fmt='o', ecolor='lightblue', alpha=0.5)

    name = 'DML006N188_SigmaVel100_M9.0'
    num_halos = 7
    vpair, sigma_vpair = read_other_data(name, num_halos)
    errorbar(vpair, sigma_vpair[0,:] ,yerr=sigma_vpair[1:3,:], fmt='o', ecolor='orange', alpha=0.5)

    x = np.arange(10,55,0.5)
    y = cross_section(x, 108, 33.35)
    plot(x,y, '-', lw=1, color='black')

    axis([10,50,1e-1,1e3])
    yscale('log')
    ylabel(r'$\sigma_{T}/m_{\chi}$ [cm$^{2}$/g]')
    xlabel(r'$\langle v_{\mathrm{pair}}\rangle$ [km/s]')
    #plt.legend(loc="lower left", labelspacing=0.2, handlelength=1.5, handletextpad=0.4, frameon=False)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig('test_2.png', dpi=200)

    ##########################
    # Plot some plot
    figure()
    ax = plt.subplot(1, 1, 1)
    grid(True)

    name = 'DML006N188_SigmaVel100_M9.5'
    num_halos = 40
    _, _, M200, c200 = read_data(name, num_halos)

    yerrl = c200[0,:]-c200[2,:]
    yerrp = c200[1,:]-c200[0,:]
    errorbar(M200[0,:], c200[0,:] ,yerr=[yerrl,yerrp], fmt='o', ecolor='lightblue', alpha=0.5)

    name = 'DML006N188_SigmaVel100_M9.0'
    num_halos = 9
    _, _, M200, c200 = read_data(name, num_halos)

    yerrl = c200[0,:]-c200[2,:]
    yerrp = c200[1,:]-c200[0,:]
    errorbar(M200[0,:], c200[0,:] ,yerr=[yerrl,yerrp], fmt='o', ecolor='orange', alpha=0.5)

    M0 = np.arange(8,13,0.2)
    c0 = c_M_relation(M0)
    plot(M0, c0, '-', lw=1, color='black')

    axis([8,11,0,30])
    ylabel(r'$c_{200}$')
    xlabel(r'$\log_{10}M_{200}~[M_{\odot}]$')
    #plt.legend(loc="lower left", labelspacing=0.2, handlelength=1.5, handletextpad=0.4, frameon=False)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig('test_3.png', dpi=200)


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

    name = 'DML006N188_SigmaVel100_M'
    vpair, sigma_vpair = read_data_joint(name)

    yerrl = sigma_vpair[0,:]-sigma_vpair[2,:]
    yerrp = sigma_vpair[1,:]-sigma_vpair[0,:]
    xerrl = vpair[0,:]-vpair[2,:]
    xerrp = vpair[1,:]-vpair[0,:]
    errorbar(vpair[0,:], sigma_vpair[0,:], yerr=[yerrl,yerrp] ,xerr=[xerrl,xerrp] ,fmt='o', ecolor='lightblue', alpha=0.5)

    name = 'DML006N188_SigmaConstant01_M'
    vpair, sigma_vpair = read_data_joint(name)

    yerrl = sigma_vpair[0,:]-sigma_vpair[2,:]
    yerrp = sigma_vpair[1,:]-sigma_vpair[0,:]
    xerrl = vpair[0,:]-vpair[2,:]
    xerrp = vpair[1,:]-vpair[0,:]
    errorbar(vpair[0,:], sigma_vpair[0,:] ,yerr=[yerrl,yerrp], xerr=[xerrl,xerrp], fmt='o', ecolor='orange', alpha=0.5)

    x = np.arange(10,101,0.5)
    y = cross_section(x, 108, 33.35)
    plot(x,y, '-', lw=1, color='black')

    axis([10,100,1e-1,1e3])
    yscale('log')
    ylabel(r'$\sigma_{T}/m_{\chi}$ [cm$^{2}$/g]')
    xlabel(r'$\langle v_{\mathrm{pair}}\rangle$ [km/s]')
    #plt.legend(loc="lower left", labelspacing=0.2, handlelength=1.5, handletextpad=0.4, frameon=False)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig('test_joint.png', dpi=200)


if __name__ == '__main__':
    #plot_outputs()
    plot_joint_outputs()