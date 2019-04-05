import numpy as np
from amuse.community.fractalcluster.interface import new_fractal_cluster_model
from amuse.ic.plummer import new_plummer_model 
from amuse.ic.kroupa import new_kroupa_mass_distribution
from amuse.ic.salpeter import new_salpeter_mass_distribution   
from amuse.community.fractalcluster.interface import new_fractal_cluster_model
from amuse.ic.kingmodel import new_king_model
from amuse.datamodel import units
from amuse.units import nbody_system, constants
import random
import os
from matplotlib import pyplot as plt
from amuse.datamodel.particles import Particles
from matplotlib import cm
from amuse.units.quantities import zero
import numpy
from amuse.ext.orbital_elements import *
from MASC.binaries import orbital_period_to_semi_major_axis
from MASC.binaries import *
import random
def MakeMeANewCluster(N=100, HalfMassRadius=1.|units.parsec, mdist='K',
                    kind="P", frac_dim=None, 
                    W=None, mmin=0.1, mmax=100., 
                    SEED=None):
    # np.random.seed(SEED)
    # random.seed(SEED)
    if mdist.lower()=="k":
        mZAMS = new_kroupa_mass_distribution(N,
                mass_max=mmax|units.MSun)
    if mdist.lower()=="s":
        mZAMS = new_salpeter_mass_distribution(N,
                mass_min=mmin|units.MSun,
                mass_max=mmax|units.MSun)
    mtot = mZAMS.sum()
    rvir = (16./(3.*np.pi))*(HalfMassRadius/1.3) 
    converter = nbody_system.nbody_to_si(
            mtot,
            rvir
            )

    if kind.lower()=="p":
        bodies = new_plummer_model(N, converter,
                                   random=np.random.seed(SEED))
        print "--------------"
        print "Made Plummer Model"
        print "--------------"
    if kind.lower()=="f":
        if frac_dim==None:
            frac_dim=1.6
        bodies = new_fractal_cluster_model(N, converter,
                fractal_dimension=frac_dim, random_seed=SEED)
        print "--------------"
        print "Made Fractal Cluster with d=",frac_dim
        print "--------------"
    if kind.lower()=="k":
        if W==None: W=6
        bodies = new_king_model(N,W,converter)
        print "--------------"
        print "Made King Model with W0=",W
        print "--------------"

    bodies.mass = mZAMS
    return bodies, converter

def binary_fraction_calc(total_num, bin_frac):
    
    number_of_binary_pairs = (total_num * bin_frac) / np.float(2)
    number_of_single_stars = total_num - total_num*bin_frac

    sim_initial_stars = int(number_of_binary_pairs)+int(number_of_single_stars)

    print bin_frac, 2.*number_of_binary_pairs / \
           (number_of_single_stars+2.*number_of_binary_pairs)

    return sim_initial_stars, int(number_of_binary_pairs)


def new_binary_orbit(mass1, mass2, semi_major_axis,
                     eccentricity = 0, keyoffset = 1):
    total_mass = mass1 + mass2
    mass_fraction_particle_1 = mass1 / (total_mass)
    
    binary = Particles(2)
    binary[0].mass = mass1
    binary[1].mass = mass2
    
    mu = constants.G * total_mass
    
    velocity_perihelion \
        = np.sqrt( mu / semi_major_axis * ((1.0 + eccentricity)
                                               /(1.0 - eccentricity)))
    radius_perihelion = semi_major_axis * (1.0 - eccentricity)
    print velocity_perihelion
    
    binary[0].position = ((1.0 - mass_fraction_particle_1) \
                           * radius_perihelion * [1.0,0.0,0.0])
    binary[1].position = -(mass_fraction_particle_1 \
                            * radius_perihelion * [1.0,0.0,0.0])
    
    binary[0].velocity = ((1.0 - mass_fraction_particle_1) \
                           * velocity_perihelion * [0.0,1.0,0.0])
    binary[1].velocity = -(mass_fraction_particle_1 \
                            * velocity_perihelion * [0.0,1.0,0.0])

    return binary

# see Eggleton 2006 Equation 1.6.3 (2006epbm.book.....E)
def random_semimajor_axis_PPE(Mprim, Msec, P_min=10|units.day,
                              P_max=100.|units.yr):

    Pmax = P_max.value_in(units.day)
    Pmin = P_min.value_in(units.day)
    mpf = (Mprim.value_in(units.MSun)**2.5)/5.e+4
    rnd_max = (Pmax * mpf)**(1./3.3) / (1 + (Pmin * mpf)**(1./3.3))
    rnd_min = (Pmin * mpf)**(1./3.3) / (1 + (Pmax * mpf)**(1./3.3))
    rnd_max = min(rnd_max, 1)
    rnd =np.random.uniform(rnd_min, rnd_max, 1)
    Porb = ((rnd/(1.-rnd))**3.3)/mpf | units.day
    Mtot = Mprim + Msec
    a = ((constants.G*Mtot) * (Porb/(2*np.pi))**2)**(1./3.)
    return a


def make_secondaries_with_massives(center_of_masses, Nbin):
    """
    UPDATE 11-01-2018
    This module now subdivides the random sample of stars
    into massive and non massive stars. The massive stars 
    paired up with their next largest partner while the 
    non-massive stars are paired up from a random uniform
    distribution. 

    ORIGINAL CODE FROM kira.py in AMUSE/EXAMPLES/TEXTBOOK
    CAN BE FOUND IN THE AMUSECODE GITHUB REPOSITORY AT
    https://github.com/amusecode/amuse
    """
    resulting_binaries = Particles()
    singles_in_binaries = Particles()
    binaries = center_of_masses.random_sample(Nbin)

    massive_stars = binaries[binaries.mass >= 5|units.MSun]
    massive_star_key = massive_stars.key[np.argsort(massive_stars.mass)][::-1]
    n_massive_stars = len(massive_star_key)
    massive_star_indices = np.arange(n_massive_stars)
    mmin = center_of_masses.mass.min()
    print mmin
    for bi in binaries:
        mp = bi.mass
        if mp >= 5|units.MSun:
            location = massive_star_indices[np.in1d(massive_star_key,bi.key)]
            print location
            if location!=(n_massive_stars-1):
                ms_key = massive_star_key[location+1]
                ms = binaries.mass[binaries.key==ms_key]
                print "Found a partner"
            if location==(n_massive_stars-1):
                ms_key = location
                ms = binaries[binaries.key==massive_star_key[location]]
                ms = ms.mass * 0.9
                print "Made a partner"            
            print "Making Massive Binary..."
            print ms
        if mp < 5|units.MSun:
            ms = numpy.random.uniform(mmin.value_in(units.MSun),
                                      mp.value_in(units.MSun)) | units.MSun
            print "Making non-Massive Binary..."

            
        a = random_semimajor_axis_PPE(mp, ms)
        e = numpy.sqrt(numpy.random.random())

        nb = new_binary_orbit(mp, ms, a, e) 
        nb.position += bi.position
        nb.velocity += bi.velocity
        nb = singles_in_binaries.add_particles(nb)
        nb.radius = 0.01 * a 

        bi.radius = a 
        binary_particle = bi.copy()
        binary_particle.child1 = nb[0]
        binary_particle.child2 = nb[1]
        binary_particle.semi_major_axis = a
        binary_particle.eccentricity = e
        resulting_binaries.add_particle(binary_particle)

    # resulting_binaries.add_particles(massive_binaries)
    # singles_in_binaries.add_particles(singles_in_massive_binaries)

    single_stars = center_of_masses-binaries
    return single_stars, resulting_binaries, singles_in_binaries


def make_secondaries_with_massives2(center_of_masses, Nbin):
    """
    UPDATE 11-01-2018
    This module now subdivides the random sample of stars
    into massive and non massive stars. The massive stars 
    paired up with their next largest partner while the 
    non-massive stars are paired up from a random uniform
    distribution. 

    ORIGINAL CODE FROM kira.py in AMUSE/EXAMPLES/TEXTBOOK
    CAN BE FOUND IN THE AMUSECODE GITHUB REPOSITORY AT
    https://github.com/amusecode/amuse
    """
    resulting_binaries = Particles()
    singles_in_binaries = Particles()
    binaries = center_of_masses.random_sample(Nbin)

    massive_stars = binaries[binaries.mass >= 5|units.MSun]
    massive_star_key = massive_stars.key[np.argsort(massive_stars.mass)][::-1]
    n_massive_stars = len(massive_star_key)
    massive_star_indices = np.arange(n_massive_stars)
    mmin = center_of_masses.mass.min()
    print mmin,center_of_masses.mass.max()
    for bi in binaries:
        mp = bi.mass
        if mp >= 5|units.MSun:
            location = massive_star_indices[np.in1d(massive_star_key,bi.key)]
            print location
            if location!=(n_massive_stars-1):
                ms_key = massive_star_key[location+1]
                ms = binaries.mass[binaries.key==ms_key]
                print "Found a partner"
            if location==(n_massive_stars-1):
                ms_key = location
                ms = binaries[binaries.key==massive_star_key[location]]
                ms = ms.mass * 0.9
                print "Made a partner"            
            print "Making Massive Binary..."
            print ms
        if mp < 5|units.MSun:
            ms = numpy.random.uniform(mmin.value_in(units.MSun),
                                      mp.value_in(units.MSun)) | units.MSun
            print "Making non-Massive Binary..."

        mean_p = 5.
        sigma_p = 2.3
        porb = np.random.lognormal(size=1,
                                    mean=mean_p,
                                    sigma=sigma_p) | units.day
        a = orbital_period_to_semi_major_axis(
                            porb,
                            mp,
                            ms
            )
        #a = random_semimajor_axis_PPE(mp, ms)
        e = numpy.sqrt(numpy.random.random())
        inclination = np.pi*random.random()|units.rad
        true_anomaly = 2.*np.pi*random.random()|units.rad
        longitude_of_ascending_node = 2.*np.pi*random.random()|units.rad
        argument_of_periapsis = 2.*np.pi*random.random()|units.rad

        primary,secondary = generate_binaries(
            mp,
            ms,
            a,
            eccentricity=e,
            inclination=inclination,
            true_anomaly=true_anomaly,
            longitude_of_the_ascending_node=longitude_of_ascending_node,
            argument_of_periapsis=argument_of_periapsis,
            G=constants.G
            )
        # print primary.position[0]
        nb = Particles(2)
        nb[0].mass = mp
        nb[0].position = primary.position[0]
        nb[0].velocity = primary.velocity[0]

        nb[1].mass = ms
        nb[1].position = secondary.position[0]
        nb[1].velocity = secondary.velocity[0]
        # nb = new_binary_orbit(mp, ms, a, e) 
        nb.position += bi.position
        nb.velocity += bi.velocity
        nb[0].radius = nb[0].mass.value_in(units.MSun)**(3./7) | units.RSun
        nb[1].radius = nb[1].mass.value_in(units.MSun)**(3./7) | units.RSun
        nb = singles_in_binaries.add_particles(nb)

        # bi.radius = (nb[0].position - nb[1].position).length()
        binary_particle = bi.copy()
        binary_particle.child1 = nb[0]
        binary_particle.child2 = nb[1]
        binary_particle.semi_major_axis = a
        binary_particle.eccentricity = e
        resulting_binaries.add_particle(binary_particle)

    # resulting_binaries.add_particles(massive_binaries)
    # singles_in_binaries.add_particles(singles_in_massive_binaries)

    single_stars = center_of_masses-binaries
    return single_stars, resulting_binaries, singles_in_binaries

def MakeMeANewClusterWithBinaries(N=100, HalfMassRadius=1.|units.parsec, mdist='K',
                    kind="P", frac_dim=None, 
                    W=None, mmin=0.1, mmax=100., 
                    SEED=None, bin_frac=0.5):

    N, number_of_binaries = binary_fraction_calc(N,bin_frac)

    # np.random.seed(SEED)
    # random.seed(SEED)
    if mdist.lower()=="k":
        mZAMS = new_kroupa_mass_distribution(N,
                mass_max=mmax|units.MSun)
    if mdist.lower()=="s":
        mZAMS = new_salpeter_mass_distribution(N,
                mass_min=mmin|units.MSun,
                mass_max=mmax|units.MSun)
    mtot = mZAMS.sum()
    rvir = (16./(3.*np.pi))*(HalfMassRadius/1.3) 
    converter = nbody_system.nbody_to_si(
            mtot,
            rvir
            )

    if kind.lower()=="p":
        bodies = new_plummer_model(N, converter,
                                   random=np.random.seed(SEED))
        print "--------------"
        print "Made Plummer Model"
        print "--------------"
    if kind.lower()=="f":
        if frac_dim==None:
            frac_dim=1.6
        bodies = new_fractal_cluster_model(N, converter,
                fractal_dimension=frac_dim, random_seed=SEED)
        print "--------------"
        print "Made Fractal Cluster with d=",frac_dim
        print "--------------"
    if kind.lower()=="k":
        if W==None: W=6
        bodies = new_king_model(N,W,converter)
        print "--------------"
        print "Made King Model with W0=",W
        print "--------------"

    bodies.mass = mZAMS

    single_stars, binary_stars, singles_in_binaries \
        = make_secondaries(bodies, number_of_binaries)
    converter = nbody_system.nbody_to_si(single_stars.mass.sum()+singles_in_binaries.mass.sum(),
                                        rvir)
    return single_stars, binary_stars, singles_in_binaries, converter


def binary_reference_frame(binary_systems, bsingles, return_status=True):
    contact_status = []
    for bi in binary_systems:
        child_1 = bi.child1
        child_2 = bi.child2

        child1 = child_1.as_particle_in_set(bsingles)
        child2 = child_2.as_particle_in_set(bsingles)
        sep = (child1.position - child2.position).length()
        rtot = (child1.radius+child2.radius)

        if rtot > sep and sep != 0.0|units.m:
            contact_status.append(["RLOF","RLOF"])
        if sep==0.0|units.m:
            contact_status.append(["MERGED","MERGED"])
        if rtot < sep:
            contact_status.append(["DETACHED","DETACHED"])

        child_1.as_particle_in_set(bsingles).position += bi.position
        child_2.as_particle_in_set(bsingles).position += bi.position

        child_1.as_particle_in_set(bsingles).velocity += bi.velocity
        child_2.as_particle_in_set(bsingles).velocity += bi.velocity

    if return_status:    return bsingles, np.ravel(contact_status)
    if not return_status: return bsingles


def spatial_plot_module(single_stars, 
                        bsingle_stars=None, 
                        binary_stars=None,
                        cluster_length=None, 
                        time=None, x=None, 
                        direc=None):
    """
    FUNCTION THAT PLOTS THE 2-D X-Y DISTRIBUTION OF STARS WITHIN A 
    STAR CLUSTER, DIFFERENTIATING THE BINARIES FROM THE SINGLE STARS
    AS WELL AS SHOWING CHANGES OCCURING WITHIN BINARIES SUCH AS 
    RLOF AND BINARY MERGERS.

    STARS ARE SEPARATED INTO THREE DIFFERENT SUBSETS
        -- STARS IN BINARIES
        -- SINGLE STARS
        -- MERGED STARS
        -- BINARY STARS IN RLOF
    THEY ARE COLORED BY HOW MUCH HIGHER THEIR OVERALL VELOCITY IS
    COMPARED TO THE VELOCITY DISPERSION OF THE STAR CLUSTER

    PARAMETERS:
    -- SINGLE_STARS; PARTICLE SET OF SINGLE STARS NOT IN BINARIES
    -- BSINGLE_STARS; PARTICLE SET OF INDIVIDUAL STARS THAT ARE IN
        BINARIES
    -- BINARY_STARS; PARTICLE SET OF BINARY STAR ORBITAL PARAMETERS
    -- CLUSTER LENGTH; SIZE OF X-Y AXIS TO PLOT STAR CLUSTER
    -- TIME; TIMESTAMP TO ADD TO EACH IMAGE
    -- X; QUEUE STAMP TO ADD TO IMAGE FILENAME
    -- DIREC; DIRECTORY TO OUTPUT INDIVIDUAL IMAGES TO
    """
    image_dir = direc
    index = str(x)
    index = index.zfill(6)
    ax = plt.figure(figsize=(8,8))
    ax1 = ax.add_subplot(111) 

    tmin,tmax = 0,3

    # single_stars.move_to_center()

    if not binary_stars==None:
        translated_bstars, contact_status = binary_reference_frame(binary_stars, bsingle_stars)
        all_singles = Particles(particles=[single_stars, translated_bstars])

        all_singles.move_to_center()

        detached_mask = np.in1d(contact_status,"DETACHED")
        rlof_mask = np.in1d(contact_status,"RLOF")
        merge_mask = np.in1d(contact_status,"MERGED")    

        # translated_bstars.move_to_center()
    # total_COM = all_singles.center_of_mass()
    # hmr = all_singles.LagrangianRadii(cm=all_singles.center_of_mass(),mf=[0.75])[0]
    # centering_subset = all_singles[(all_singles.position - total_COM).lengths() <= hmr]
    # centering_com = centering_subset.center_of_mass() 
        # all_vel = all_singles.velocity
        # vel_mean = all_vel.mean()
        # vel_disp = all_vel.std()

        bstar_pos = translated_bstars.position-all_singles.center_of_mass()
        bstar_vel = translated_bstars.velocity.lengths().value_in(units.kms)# - translated_bstars.center_of_mass_velocity()
        # bstar_vel_sigma = abs(bstar_vel - vel_mean)#/vel_disp

        detached_stars = bstar_pos[detached_mask]
        # v = ((bstar_vel.x[detached_mask]**2. + \
            # bstar_vel.y[detached_mask]**2)**.5).value_in(units.kms)
        v = bstar_vel[detached_mask]
        bin_x = detached_stars.x.value_in(units.parsec)
        bin_y = detached_stars.y.value_in(units.parsec)

        scatter1 = ax1.scatter(bin_x, bin_y, marker='o', s=75, c=np.log10(v),
                                label="N_det: "+str(detached_mask.sum()),
                                alpha=0.50, zorder=1, cmap=cm.gnuplot,
                                vmin=tmin, vmax=tmax) 

        rlof_stars = bstar_pos[rlof_mask]
        # v = ((bstar_vel.x[rlof_mask]**2. + \
            # bstar_vel.y[rlof_mask]**2)**.5).value_in(units.kms)
        v = bstar_vel[rlof_mask]
        bin_x = rlof_stars.x.value_in(units.parsec)
        bin_y = rlof_stars.y.value_in(units.parsec)
        scatter2 = ax1.scatter(bin_x, bin_y, marker='s', s=75, c=np.log10(v),
                                label="N_rlof: "+str(rlof_mask.sum()),
                                alpha=0.50, zorder=1, cmap=cm.gnuplot,
                                vmin=tmin, vmax=tmax) 

        merged_stars = bstar_pos[merge_mask]
        # v = ((bstar_vel.x[merge_mask]**2. + \
            # bstar_vel.y[merge_mask]**2)**.5).value_in(units.kms)
        v = bstar_vel[merge_mask]
        bin_x = merged_stars.x.value_in(units.parsec)
        bin_y = merged_stars.y.value_in(units.parsec)

        scatter3 = ax1.scatter(bin_x, bin_y, marker='x', s=75, c=np.log10(v),
                                label="N_merg: "+str(merge_mask.sum()),
                                alpha=0.50, zorder=1, cmap=cm.gnuplot,
                                vmin=tmin, vmax=tmax) 

    if binary_stars==None:
        all_vel = single_stars.velocity
        vel_mean = all_vel.mean()
        vel_disp = all_vel.std()


    star_pos = single_stars.position-all_singles.center_of_mass()
    star_vel = single_stars.velocity.lengths().value_in(units.kms)# - single_stars.center_of_mass_velocity()
    # star_vel_sigma = abs(star_vel - vel_mean)/vel_disp



    # v = ((star_vel.x**2. + star_vel.y**2.)**.5).value_in(units.kms)
    v = star_vel
    star_x = star_pos.x.value_in(units.parsec)
    star_y = star_pos.y.value_in(units.parsec)
    scatter4 = ax1.scatter(star_y, star_x, marker='^', s=50, c=np.log10(v),
                            label="N_ms: "+str(len(single_stars)),
                            cmap=cm.gnuplot, alpha=0.50, zorder=0, 
                            vmin=tmin, vmax=tmax)

    ax1.set_xlabel('X [Parsecs]')
    ax1.set_ylabel('Y [Parsecs]')

    ax1.set_title(str(np.round(time.value_in(units.Myr),3))+' Myr')

    # cluster_length = (5|units.parsec).value_in(units.parsec)
    plt.legend(loc='upper right', frameon=False, scatterpoints=1, markerscale=0.5)
    ax1.set_xlim(-cluster_length, cluster_length)
    ax1.set_ylim(-cluster_length, cluster_length)
    cb = plt.colorbar(scatter4, pad=0.005)
    cb.set_label(r"$\sigma_{V,SC}$")
    plt.draw()

    plt.savefig(image_dir+'bin_evo_pos_plot'+index+'.png',bbox_inches='tight', dpi=300)
    plt.cla()
    plt.close()


def energy_conv_plot(FILENAME=None):
    import pandas as pd
    data = pd.read_csv(FILENAME, float_precision="high")
    
    absolute_EdE = abs(data.energy_error)
    ax = plt.figure(figsize=(8,8))
    ax1 = ax.add_subplot(111) 

    ax1.plot(data.step_time,absolute_EdE)
    ax1.set_yscale('log')
    ax1.set_ylabel(r"$(E_{0}-E)/E_{0}$",fontsize=12)
    ax1.set_xlabel(r"$N_{step}$",fontsize=12)

    plt.draw()
    plt.savefig("energy_conservation_plot.png",bbox_inches='tight', dpi=300)
    plt.cla()
    plt.close()

def simple_spatial_plot_module(single_stars, 
                        bsingle_stars=None, 
                        binary_stars=None,
                        cluster_length=None, 
                        time=None, x=None, 
                        direc=None,
                        com=zero,
                        core=zero):
    """
    FUNCTION THAT PLOTS THE 2-D X-Y DISTRIBUTION OF STARS WITHIN A 
    STAR CLUSTER, DIFFERENTIATING THE BINARIES FROM THE SINGLE STARS
    AS WELL AS SHOWING CHANGES OCCURING WITHIN BINARIES SUCH AS 
    RLOF AND BINARY MERGERS.

    STARS ARE SEPARATED INTO THREE DIFFERENT SUBSETS
        -- STARS IN BINARIES
        -- SINGLE STARS
        -- MERGED STARS
        -- BINARY STARS IN RLOF
    THEY ARE COLORED BY HOW MUCH HIGHER THEIR OVERALL VELOCITY IS
    COMPARED TO THE VELOCITY DISPERSION OF THE STAR CLUSTER

    PARAMETERS:
    -- SINGLE_STARS; PARTICLE SET OF SINGLE STARS NOT IN BINARIES
    -- BSINGLE_STARS; PARTICLE SET OF INDIVIDUAL STARS THAT ARE IN
        BINARIES
    -- BINARY_STARS; PARTICLE SET OF BINARY STAR ORBITAL PARAMETERS
    -- CLUSTER LENGTH; SIZE OF X-Y AXIS TO PLOT STAR CLUSTER
    -- TIME; TIMESTAMP TO ADD TO EACH IMAGE
    -- X; QUEUE STAMP TO ADD TO IMAGE FILENAME
    -- DIREC; DIRECTORY TO OUTPUT INDIVIDUAL IMAGES TO
    """
    image_dir = direc
    index = str(x)
    index = index.zfill(6)
    ax = plt.figure(figsize=(8,8))
    ax1 = ax.add_subplot(111) 

    if np.all(com==zero): 
        single_stars.move_to_center()
        star_pos = single_stars.position
        cx = single_stars.center_of_mass()[0].value_in(units.parsec)
        cy = single_stars.center_of_mass()[1].value_in(units.parsec)
    if not np.all(com==zero):
        center_of_mass = com
        star_pos = single_stars.position - center_of_mass
        cx = center_of_mass[0].value_in(units.parsec)
        cy = center_of_mass[1].value_in(units.parsec)

    contact_status = binary_reference_frame(binary_stars, bsingle_stars)[1]
    detached_size = np.in1d(contact_status,"DETACHED").sum()
    rlof_size = np.in1d(contact_status,"RLOF").sum()
    merge_size = np.in1d(contact_status,"MERGED").sum()
    num_single_stars = len(single_stars)-len(bsingle_stars)

    label_str=r"$N_{ms}:\ $"+str(num_single_stars)+"\n"+\
                r"$Bin_{det}:\ $"+str(detached_size/2)+"\n"+\
                r"$Bin_{RLOF}:\ $"+str(rlof_size/2)+"\n"+\
                r"$Bin_{merged}:\ $"+str(merge_size/2)
    

    star_x = star_pos.x.value_in(units.parsec)
    star_y = star_pos.y.value_in(units.parsec)
    scatter4 = ax1.scatter(star_y, star_x, marker='o', s=5.0, c='gray',alpha=0.5,
                            label=label_str)

    ax1.set_xlabel('X [Parsecs]')
    ax1.set_ylabel('Y [Parsecs]')

    ax1.set_title(str(np.round(time.value_in(units.Myr),3))+' Myr')

    # cluster_length = (5|units.parsec).value_in(units.parsec)
    plt.legend(loc='upper right', frameon=False, scatterpoints=1, markerscale=0.5)
    ax1.set_xlim(-cluster_length, cluster_length)
    ax1.set_ylim(-cluster_length, cluster_length)
    # cb = plt.colorbar(scatter4, pad=0.005)
    # cb.set_label(r"$\sigma_{V,SC}$")
    plt.draw()

    plt.savefig(image_dir+'bin_evo_pos_plot'+index+'.png',bbox_inches='tight', dpi=300)
    plt.cla()
    plt.close()


def spatial_plot_module2(single_stars, 
                        bsingle_stars=None, 
                        binary_stars=None,
                        cluster_length=None, 
                        time=None, x=None, 
                        direc=None,
                        com=zero,
                        core=zero):
    """
    FUNCTION THAT PLOTS THE 2-D X-Y DISTRIBUTION OF STARS WITHIN A 
    STAR CLUSTER, COLORED BY THEIR MASS, AND LISTING THE NUMBER OF
    DETACHED, RLOF, AND MERGED BINARIES

    STARS ARE SEPARATED INTO THREE DIFFERENT SUBSETS
        -- STARS IN BINARIES
        -- SINGLE STARS
        -- MERGED STARS
        -- BINARY STARS IN RLOF
    THEY ARE COLORED BY HOW MUCH HIGHER THEIR OVERALL VELOCITY IS
    COMPARED TO THE VELOCITY DISPERSION OF THE STAR CLUSTER

    PARAMETERS:
    -- SINGLE_STARS; PARTICLE SET OF SINGLE STARS NOT IN BINARIES
    -- BSINGLE_STARS; PARTICLE SET OF INDIVIDUAL STARS THAT ARE IN
        BINARIES
    -- BINARY_STARS; PARTICLE SET OF BINARY STAR ORBITAL PARAMETERS
    -- CLUSTER LENGTH; SIZE OF X-Y AXIS TO PLOT STAR CLUSTER
    -- TIME; TIMESTAMP TO ADD TO EACH IMAGE
    -- X; QUEUE STAMP TO ADD TO IMAGE FILENAME
    -- DIREC; DIRECTORY TO OUTPUT INDIVIDUAL IMAGES TO
    """
    image_dir = direc
    index = str(x)
    index = index.zfill(6)
    ax = plt.figure(figsize=(8,8))
    ax1 = ax.add_subplot(111) 

    tmin,tmax = -1,2

    # single_stars.move_to_center()
    contact_status = binary_reference_frame(binary_stars, bsingle_stars)[1]
    detached_size = np.in1d(contact_status,"DETACHED").sum()
    rlof_size = np.in1d(contact_status,"RLOF").sum()
    merge_size = np.in1d(contact_status,"MERGED").sum()

    num_single_stars = len(single_stars)# - len(binary_stars)#(detached_size+rlof_size+merge_size)

    if np.all(com==zero): 
        single_stars.move_to_center()
        star_pos = single_stars.position
        cx = single_stars.center_of_mass()[0].value_in(units.parsec)
        cy = single_stars.center_of_mass()[1].value_in(units.parsec)
    if not np.all(com==zero):
        center_of_mass = com
        star_pos = single_stars.position - center_of_mass
        cx = center_of_mass[0].value_in(units.parsec)
        cy = center_of_mass[1].value_in(units.parsec)
    # star_vel = single_stars.velocity.lengths().value_in(units.kms)# - single_stars.center_of_mass_velocity()
    # star_vel_sigma = abs(star_vel - vel_mean)/vel_disp


    # v = ((star_vel.x**2. + star_vel.y**2.)**.5).value_in(units.kms)
    m = np.log10(single_stars.mass.value_in(units.MSun))
    star_x = star_pos.x.value_in(units.parsec)
    star_y = star_pos.y.value_in(units.parsec)
    scatter4 = ax1.scatter(star_y, star_x, marker='o', s=50, c=m,
                            label=r"$N_{ms}:\ $"+str(num_single_stars)+"\n"+\
                            r"$Bin_{det}:\ $"+str(detached_size/2)+"\n"+\
                            r"$Bin_{RLOF}:\ $"+str(rlof_size/2)+"\n"+\
                            r"$Bin_{merged}:\ $"+str(merge_size/2),
                            cmap=cm.viridis, alpha=0.50, zorder=0, 
                            vmin=tmin, vmax=tmax)
    if not core==zero:
        c1 = plt.Circle((cx,cy),core.value_in(units.parsec),fill=False,color='black',zorder=1)
        ax1.add_artist(c1)
    ax1.set_xlabel('X [Parsecs]')
    ax1.set_ylabel('Y [Parsecs]')

    ax1.set_title(str(np.round(time.value_in(time.unit),3))+str(time.unit))

    # cluster_length = (5|units.parsec).value_in(units.parsec)
    plt.legend(loc='upper right', frameon=False, scatterpoints=1, markerscale=0.5)
    ax1.set_xlim(-cluster_length, cluster_length)
    ax1.set_ylim(-cluster_length, cluster_length)
    cb = plt.colorbar(scatter4, pad=0.005)
    cb.set_label(r"$Mass\ [M_{\odot}]$")
    plt.draw()

    plt.savefig(image_dir+'bin_evo_pos_plot'+index+'.png',bbox_inches='tight', dpi=300)
    plt.cla()
    plt.close()


def simple_2d_movie_maker(filename,fps=None,img_dir=None,output_dir=None):
    import os
    """
    MAKES A MOVIE OUT OF 2-D PLOTS OF THE STAR CLUSTER EVOLUTION
    USING FFMPEG

    PARAMETERS
    -- FILENAME; NAME OF OUTPUT MOVIE
    -- IMG_DIR; DIRECTORY IN WHICH TO FIND INDIVIDUAL IMAGES
    -- OUTPUT_DIR; DIRECTORY IN WHICH TO OUTPUT THE COMPILED
        MOVIE. SET TO NONE, IN WHICH CASE THE MOVIE IS 
        OUTPUTTED IN THE SAME DIRECTORY AS THE STORED IMAGES 
    """
    CWD = os.getcwd()
    if filename==None: filename = "spatial_evolution_movie.mp4"
    if fps==None: fps=15
    if img_dir==None: img_dir = CWD
    if not filename[-4:]=='.mp4': filename = filename+".mp4"
    if not output_dir==None: movie_dir = output_dir
    if output_dir==None: movie_dir = img_dir
    
    print "making evolution movie"
    os.system("ffmpeg -framerate "+str(fps)+" -pix_fmt yuv420p -pattern_type glob -i '"+img_dir+"*.png' "+movie_dir+filename)
    print "finished evolution movie"
