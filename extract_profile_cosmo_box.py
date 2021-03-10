import numpy as np
import h5py
import scipy.stats as stat
from scipy.special import spence

def bin_volumes(radial_bins):
    """Returns the volumes of the bins. """

    single_vol = lambda x: (4.0 / 3.0) * np.pi * x ** 3
    outer = single_vol(radial_bins[1:])
    inner = single_vol(radial_bins[:-1])
    return outer - inner


def bin_centers(radial_bins):
    """Returns the centers of the bins. """

    outer = radial_bins[1:]
    inner = radial_bins[:-1]
    return 0.5 * (outer + inner)


def analyse_halo(mass, pos):
    # Define radial bins [log scale, kpc units]
    radial_bins = np.arange(0, 5, 0.1)
    radial_bins = 10 ** radial_bins

    # Radial coordinates [kpc units]
    r = np.sqrt(np.sum(pos ** 2, axis=1))

    SumMasses, _, _ = stat.binned_statistic(x=r, values=np.ones(len(r)) * mass[0], statistic="sum", bins=radial_bins, )
    density = (SumMasses / bin_volumes(radial_bins))  # Msun/kpc^3
    return density

radial_bins = np.arange(0, 5, 0.1)
radial_bins = 10**radial_bins
centers = bin_centers(radial_bins) #kpc

snap = 60
folder = '/Users/camila/TangoSIDM/TangoSIDM_swift/tests/Cosmo_boxes/L025N376/L025N376_sigma_10/'

with h5py.File(folder+"snapshot_00%02i.hdf5"%snap) as hf:
    mass = hf['PartType1/Masses'][:] * 1e10 #Msun
    pos = hf['PartType1/Coordinates'][:][:]
    vel = hf['PartType1/Velocities'][:][:]
    unit_length_in_cgs = hf["/Units"].attrs["Unit length in cgs (U_L)"]
    a = hf["/Header"].attrs["Scale-factor"]

snapshot_file = h5py.File(folder+"snapshot_00%02i.hdf5"%snap)
group_file = h5py.File(folder+"subhalo_00%02i.catalog_groups"%snap)
particles_file = h5py.File(folder+"subhalo_00%02i.catalog_particles"%snap)
properties_file = h5py.File(folder+"subhalo_00%02i.properties"%snap)

m200c = properties_file["Mass_200crit"][:] * 1e10
m200c = np.log10(m200c)
CoP = np.zeros((len(m200c), 3))
CoP[:, 0] = properties_file["Xcminpot"][:] / a
CoP[:, 1] = properties_file["Ycminpot"][:] / a
CoP[:, 2] = properties_file["Zcminpot"][:] / a

select_halos = np.where((m200c >= 11.9) & (m200c <= 12.1))[0]  # >10 star parts

M200 = np.median(10 ** m200c[select_halos])
num_halos = len(select_halos)
print('Num halos', num_halos, np.log10(M200))

density_all = np.zeros((len(centers), num_halos))

for halo in range(0, num_halos):
    halo_j = select_halos[halo]
    print('reading:',halo)

    # Grab the start position in the particles file to read from
    halo_start_position = group_file["Offset"][halo_j]
    halo_end_position = group_file["Offset"][halo_j + 1]
    particle_ids_in_halo = particles_file["Particle_IDs"][halo_start_position:halo_end_position]
    particle_ids_from_snapshot = snapshot_file["PartType1/ParticleIDs"][...]

    _, indices_v, indices_p = np.intersect1d(particle_ids_in_halo,
                                             particle_ids_from_snapshot,
                                             assume_unique=True,
                                             return_indices=True, )

    particles_mass = mass[indices_p].copy()
    particles_pos = pos[indices_p, :].copy()
    particles_pos -= CoP[halo_j, :]  # centering
    particles_pos *= 1e3  # kpc
    density_all[:, halo] = analyse_halo(particles_mass, particles_pos)

densityM = np.median(density_all[:, :], axis=1)
densityUp = np.percentile(density_all[:, :], 84, axis=1)
densityLow = np.percentile(density_all[:, :], 16, axis=1)

# Output final median profile:
output = np.zeros((len(centers),4))
output[:,0] = centers
output[:,1] = densityM
output[:,2] = densityLow
output[:,3] = densityUp
np.savetxt("./data/Profile_halo_M12_L025N376_sigma_10.txt", output, fmt="%s")
