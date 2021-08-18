import numpy as np
import h5py
import scipy.stats as stat
import os

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


def analyse_halo(mass, pos, radial_bins):

    # Radial coordinates [kpc units]
    r = np.sqrt(np.sum(pos ** 2, axis=1))

    SumMasses, _, _ = stat.binned_statistic(x=r, values=np.ones(len(r)) * mass[0], statistic="sum", bins=radial_bins, )
    density = (SumMasses / bin_volumes(radial_bins))  # Msun/kpc^3
    return density

def read_data(which_halos,siminfo,mass_select):

    radial_bins = np.arange(-0.3, 3, 0.1)
    radial_bins = 10**radial_bins
    centers = bin_centers(radial_bins) #kpc

    with h5py.File(siminfo.snapshot,"r") as hf:
        a = hf["/Header"].attrs["Scale-factor"]
        mass = hf['PartType1/Masses'][:] * 1e10 #Msun
        pos = hf['PartType1/Coordinates'][:][:] * a

    snapshot_file = h5py.File(siminfo.snapshot,"r")
    group_file = h5py.File(siminfo.catalog_groups,"r")
    particles_file = h5py.File(siminfo.catalog_particles,"r")
    properties_file = h5py.File(siminfo.subhalo_properties,"r")

    m200c = properties_file["Mass_200crit"][:] * 1e10
    m200c[m200c == 0] = 1
    m200c = np.log10(m200c)
    CoP = np.zeros((len(m200c), 3))
    CoP[:, 0] = properties_file["Xcminpot"][:] * a
    CoP[:, 1] = properties_file["Ycminpot"][:] * a
    CoP[:, 2] = properties_file["Zcminpot"][:] * a
    subtype = properties_file["Structuretype"][:]

    select_halos = np.where((m200c >= mass_select-0.2) & (m200c <= mass_select+0.2))[0]  # >10 star parts

    # Checking sample
    if which_halos == 'subhalos':
        select = np.where(subtype[select_halos] > 10)[0]
        select_halos = select_halos[select]
    else:
        select = np.where(subtype[select_halos] == 10)[0]
        select_halos = select_halos[select]

    if len(select_halos) >= 20:
        #select_random = np.random.random_integers(len(select_halos) - 1, size=(20))
        #select_halos = select_halos[select_random]
        select_halos = select_halos[0:20]


    M200 = np.median(10 ** m200c[select_halos])
    M200 = np.log10(M200)
    num_halos = len(select_halos)

    density_all = np.zeros((len(centers), num_halos))

    for halo in range(0, num_halos):
        halo_j = select_halos[halo]

        # # Grab the start position in the particles file to read from
        # halo_start_position = group_file["Offset"][halo_j]
        # halo_end_position = group_file["Offset"][halo_j + 1]
        # particle_ids_in_halo = particles_file["Particle_IDs"][halo_start_position:halo_end_position]
        # particle_ids_from_snapshot = snapshot_file["PartType1/ParticleIDs"][...]
        #
        # _, indices_v, indices_p = np.intersect1d(particle_ids_in_halo,
        #                                          particle_ids_from_snapshot,
        #                                          assume_unique=True,
        #                                          return_indices=True, )
        #
        # particles_mass = mass[indices_p].copy()
        # particles_pos = pos[indices_p, :].copy()
        particles_mass = mass.copy()
        particles_pos = pos.copy()
        particles_pos -= CoP[halo_j, :]  # centering
        particles_pos *= 1e3  # kpc
        if len(particles_mass) == 0 :continue
        density_all[:, halo] = analyse_halo(particles_mass, particles_pos, radial_bins)

        density = analyse_halo(particles_mass, particles_pos, radial_bins)
        output = np.zeros((len(centers),2))
        output[:, 0] = centers
        output[:, 1] = density

        if which_halos == 'subhalos':
            np.savetxt(siminfo.output_path+"Profile_subhalos_M%0.1f"%mass_select+"_"+siminfo.name+"_%i.txt"%halo, output, fmt="%s")
        else:
            np.savetxt(siminfo.output_path+"Profile_halos_M%0.1f"%mass_select+"_"+siminfo.name+"_%i.txt"%halo, output, fmt="%s")


    densityM = np.median(density_all[:, :], axis=1)
    densityUp = np.percentile(density_all[:, :], 84, axis=1)
    densityLow = np.percentile(density_all[:, :], 16, axis=1)

    # Output final median profile:
    output = np.zeros((len(centers),4))
    output[:,0] = centers
    output[:,1] = densityM
    output[:,2] = densityLow
    output[:,3] = densityUp


    if which_halos == 'subhalos':
        np.savetxt(siminfo.output_path+"Profile_subhalos_M%0.1f"%mass_select+"_"+siminfo.name+".txt", output, fmt="%s")
    else:
        np.savetxt(siminfo.output_path+"Profile_halos_M%0.1f"%mass_select+"_"+siminfo.name+".txt", output, fmt="%s")



class SimInfo:
    def __init__(self, folder, snap, output_path, name):
        self.name = name
        self.output_path = output_path

        snapshot = os.path.join(folder,"snapshot_%04i.hdf5"%snap)
        if os.path.exists(snapshot):
            self.snapshot = os.path.join(folder,"snapshot_%04i.hdf5"%snap)

        properties = os.path.join(folder, "halo_%04i.properties" % snap)
        if os.path.exists(properties):
            self.subhalo_properties = os.path.join(folder, "halo_%04i.properties" % snap)
        else:
            self.subhalo_properties = os.path.join(folder, "subhalo_%04i.properties" % snap)

        catalog = os.path.join(folder,"halo_%04i.catalog_groups"%snap)
        if os.path.exists(catalog):
            self.catalog_groups = os.path.join(folder,"halo_%04i.catalog_groups"%snap)
        else:
            self.catalog_groups = os.path.join(folder,"subhalo_%04i.catalog_groups"%snap)

        catalog_particles = os.path.join(folder, "halo_%04i.catalog_particles" % snap)
        if os.path.exists(catalog_particles):
            self.catalog_particles = os.path.join(folder, "halo_%04i.catalog_particles" % snap)
        else:
            self.catalog_particles = os.path.join(folder, "subhalo_%04i.catalog_particles" % snap)

        snapshot_file = h5py.File(self.snapshot, "r")
        self.softening = float(snapshot_file["/Parameters"].attrs["Gravity:comoving_DM_softening"][:])
        self.softening *= 1e3 #kpc units


if __name__ == '__main__':
    
    from utils import *

    output_path = args.output
    folder = args.input
    snapshot = int(args.snapshot)
    name = args.name

    siminfo = SimInfo(folder, snapshot, output_path, name)

    # mass = 9.0
    # read_data("halos",siminfo,mass)
    # read_data("subhalos",siminfo,mass)

    # mass = 9.5
    # read_data("halos",siminfo,mass)
    # read_data("subhalos",siminfo,mass)

    mass = 10.5
    read_data("halos",siminfo,mass)
    # read_data("subhalos",siminfo,mass)
