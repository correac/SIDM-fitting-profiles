
import numpy as np
from scipy.integrate import odeint
from pylab import *
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

def velocity_dispersion(x, n):
    """
        Args: x = r/rs
        Returns: y = sigma_v(r)/sigma_v0
    """
    f = x**(n/2)
    return f

def diff_isothermal_equation(f,x,n):
    """
    Differential equation that describes the isothermal profile
    """
    y, z = f
    dfdx = [z,-(n+2)*(1./x)*z-n*(n+1)*(1./x**2)-(1./x**n)*np.exp(y)]
    return dfdx

def rho_nfw(x):
    """
        Args: x = r/rs
        Returns: y = rho(r)/rho_s
    """
    y = 1. / (x * (1.+x)**2)
    return y

def rho_isothermal(x,n):
    """
        Args: x = r/r0
        Returns: y = rho(r)/rho0
    """
    xrange = np.arange(-4, 5, 0.01)
    xrange = 10 ** xrange
    y0 = [0, 0]
    sol = odeint(diff_isothermal_equation, y0, xrange, args=(n,))
    yrange = np.exp(sol[:, 0])
    finterpolate = interp1d(xrange, yrange)

    if np.isscalar(x):
        if x <= 1e-4:
            y = 0
        elif x >= 1e5:
            y = finterpolate(1e5)
        else:
            y = finterpolate(x)
    else:
        y = np.zeros(len(x))
        in_range = np.logical_and(x > 1e-4, x <= 1e4)
        y[in_range] = finterpolate(x[in_range])

    return y

# Plot parameters
params = {
    "font.size": 12,
    "text.usetex": True,
    "figure.figsize": (6, 3),
    "figure.subplot.left": 0.12,
    "figure.subplot.right": 0.95,
    "figure.subplot.bottom": 0.18,
    "figure.subplot.top": 0.95,
    "figure.subplot.wspace": 0.35,
    "figure.subplot.hspace": 0.35,
    "lines.markersize": 6,
    "lines.linewidth": 1.5,
    "figure.max_open_warning": 0,
}
rcParams.update(params)
rc("font", **{"family": "sans-serif", "sans-serif": ["Times"]})

#######################
# Plot the density profile
figure()
ax = plt.subplot(1,2,1)
grid(True)

x =  10 ** (np.arange(-1,2,0.1))
y0 = rho_isothermal(x,0.0)
y1 = rho_isothermal(x,-0.1)
y2 = rho_isothermal(x,0.1)

plot(x,y0, '-',label='$n=0$')
plot(x,y1, '-',label='$n=-0.1$')
plot(x,y2, '-',label='$n=0.1$')

xscale('log')
yscale('log')
ylabel(r'Modified-isothermal profile ($\rho/\rho_{0}$)')
xlabel('r/r$_{0}$')
plt.legend(loc=[0.0,0.1],labelspacing=0.2,handlelength=1.5,handletextpad=0.4,frameon=False,
           columnspacing=0.1,ncol=1)
ax.tick_params(direction='in', axis='both', which='both', pad=4.5)

axis([1e-1, 1e2, 1e-4, 3])

#####################
ax = plt.subplot(1,2,2)
grid(True)

x =  10 ** (np.arange(-1,2,0.1))
v0 = velocity_dispersion(x, 0.0)
v1 = velocity_dispersion(x, -0.1)
v2 = velocity_dispersion(x, 0.1)

plot(x,v0, '-')
plot(x,v1, '-')
plot(x,v2, '-')

xscale('log')
ylabel(r'Velocity dispersion ($\sigma_v$(r)/$\sigma_{v0}$)')
xlabel('r/r$_{0}$')
axis([1e-1, 1e2, 0.7, 1.3])
ax.tick_params(direction='in', axis='both', which='both', pad=4.5)

plt.savefig('Isothermal_model.png',dpi=200)

