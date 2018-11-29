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




def MakeMeANewCluster(N=100, HalfMassRadius=1.|units.parsec, mdist='K',
                    kind="P", frac_dim=None, 
                    W=None, mmin=0.1, mmax=100., 
                    SEED=None):
    np.random.seed(SEED)
    random.seed(SEED)
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

def make_secondaries(center_of_masses, Nbin):
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
            ms = np.random.uniform(mmin.value_in(units.MSun),
                                      mp.value_in(units.MSun)) | units.MSun
            print "Making non-Massive Binary..."
        a = random_semimajor_axis_PPE(mp, ms)
        e = np.sqrt(np.random.random())

        nb = new_binary_orbit(mp, ms, a, e) 
        nb.position += bi.position
        nb.velocity += bi.velocity
        nb = singles_in_binaries.add_particles(nb)
        nb.radius = 0.01 * a 

        bi.radius = 3*a 
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

    np.random.seed(SEED)
    random.seed(SEED)
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
    # converter = nbody_system.nbody_to_si(single_stars.mass.sum()+singles_in_binaries.mass.sum(),
                                        # rvir)
    return single_stars, binary_stars, singles_in_binaries, converter


def binary_reference_frame(binary_systems, bsingles):
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

    return bsingles, np.ravel(contact_status)



def spatial_plot_module(single_stars, 
                        bsingle_stars, 
                        binary_stars,
                        cluster_length, 
                        time, x, 
                        direc):
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
    index = index.zfill(4)
    ax = plt.figure(figsize=(8,8))
    ax1 = ax.add_subplot(111) 

    tmin,tmax = -3, 3

    translated_bstars, contact_status = binary_reference_frame(binary_stars, bsingle_stars)

    all_singles = Particles(particles=[single_stars, translated_bstars])
    
    # total_COM = all_singles.center_of_mass()
    # hmr = all_singles.LagrangianRadii(cm=all_singles.center_of_mass(),mf=[0.75])[0]
    # centering_subset = all_singles[(all_singles.position - total_COM).lengths() <= hmr]
    # centering_com = centering_subset.center_of_mass() 


    all_vel = all_singles.velocity
    vel_mean = all_vel.mean()
    vel_disp = all_vel.std()

    star_pos = single_stars.position# - centering_com
    star_vel = single_stars.velocity.lengths()# - single_stars.center_of_mass_velocity()
    star_vel_sigma = (star_vel - vel_mean)/vel_disp

    bstar_pos = translated_bstars.position# - centering_com
    bstar_vel = translated_bstars.velocity.lengths()# - translated_bstars.center_of_mass_velocity()
    bstar_vel_sigma = (bstar_vel - vel_mean)/vel_disp

    detached_mask = np.in1d(contact_status,"DETACHED")
    rlof_mask = np.in1d(contact_status,"RLOF")
    merge_mask = np.in1d(contact_status,"MERGED")

    detached_stars = bstar_pos[detached_mask]
    # v = ((bstar_vel.x[detached_mask]**2. + \
        # bstar_vel.y[detached_mask]**2)**.5).value_in(units.kms)
    v = bstar_vel_sigma[detached_mask]
    bin_x = detached_stars.x.value_in(units.parsec)
    bin_y = detached_stars.y.value_in(units.parsec)

    scatter1 = ax1.scatter(bin_x, bin_y, marker='o', s=75, c=np.log10(v),
                            label="N_det: "+str(detached_mask.sum()),
                            alpha=0.50, zorder=1, cmap=cm.gnuplot,
                            vmin=tmin, vmax=tmax) 

    rlof_stars = bstar_pos[rlof_mask]
    # v = ((bstar_vel.x[rlof_mask]**2. + \
        # bstar_vel.y[rlof_mask]**2)**.5).value_in(units.kms)
    v = bstar_vel_sigma[rlof_mask]
    bin_x = rlof_stars.x.value_in(units.parsec)
    bin_y = rlof_stars.y.value_in(units.parsec)
    scatter2 = ax1.scatter(bin_x, bin_y, marker='s', s=75, c=np.log10(v),
                            label="N_rlof: "+str(rlof_mask.sum()),
                            alpha=0.50, zorder=1, cmap=cm.gnuplot,
                            vmin=tmin, vmax=tmax) 

    merged_stars = bstar_pos[merge_mask]
    # v = ((bstar_vel.x[merge_mask]**2. + \
        # bstar_vel.y[merge_mask]**2)**.5).value_in(units.kms)
    v = bstar_vel_sigma[merge_mask]
    bin_x = merged_stars.x.value_in(units.parsec)
    bin_y = merged_stars.y.value_in(units.parsec)

    scatter3 = ax1.scatter(bin_x, bin_y, marker='x', s=75, c=np.log10(v),
                            label="N_merg: "+str(merge_mask.sum()),
                            alpha=0.50, zorder=1, cmap=cm.gnuplot,
                            vmin=tmin, vmax=tmax) 

    # v = ((star_vel.x**2. + star_vel.y**2.)**.5).value_in(units.kms)
    v = star_vel_sigma
    star_x = star_pos.x.value_in(units.parsec)
    star_y = star_pos.y.value_in(units.parsec)
    scatter4 = ax1.scatter(star_y, -1*star_x, marker='^', s=50, c=np.log10(v),
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
    cb = plt.colorbar(scatter1, pad=0.005)
    cb.set_label(r"$\sigma_{V,SC}$")
    plt.draw()

    plt.savefig(image_dir+'bin_evo_pos_plot'+index+'.png',bbox_inches='tight', dpi=300)
    plt.cla()
    plt.close()


def simple_2d_movie_maker(filename, img_dir, output_dir=None):
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
    if not filename[-4:]=='.mp4': filename = filename+".mp4"
    if not output_dir==None: movie_dir = output_dir
    if output_dir==None: movie_dir = img_dir
    
    print "making evolution movie"
    os.system("ffmpeg -framerate 30 -pix_fmt yuv420p -pattern_type glob -i '"+img_dir+"*.png' "+movie_dir+filename)
    print "finished evolution movie"
