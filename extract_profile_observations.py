import numpy as np
from scipy.interpolate import interp1d

def read_data(input_file, output_file):

    data = np.loadtxt(input_file)
    x = data[:,0]
    y = np.log10(data[:,1])
    yup = np.log10(data[:,3])
    ylow = np.log10(data[:,2])


    finterpolate = interp1d(x, y)
    xdata = np.arange(-1,np.log10(np.max(x)),0.1)
    xdata = 10**xdata

    y = finterpolate(xdata)
    finterpolate = interp1d(x, yup)
    yup = finterpolate(xdata)
    finterpolate = interp1d(x, ylow)
    ylow = finterpolate(xdata)
    x = xdata

    # Finallly output
    output = np.zeros((len(x), 4))
    output[:, 0] = x
    output[:, 1] = 10**y
    output[:, 2] = 10**ylow
    output[:, 3] = 10**yup
    np.savetxt(output_file, output, fmt="%s")

galaxy="Carina"
file_name = "../FermionDSph/Code/Final_Data/"+galaxy+"/output_rho.txt"
output_file = "./data/FermionDSph_"+galaxy+".txt"
read_data(file_name, output_file)
