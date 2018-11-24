import numpy as np
from amuse.community.fractalcluster.interface import new_fractal_cluster_model
from amuse.ic.plummer import new_plummer_model 
from amuse.ic.kroupa import new_kroupa_mass_distribution
from amuse.ic.salpeter import new_salpeter_mass_distribution   
from amuse.community.fractalcluster.interface import new_fractal_cluster_model
from amuse.ic.kingmodel import new_king_model
from amuse.datamodel import units
from amuse.units import nbody_system
import random
import os






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
    
    total_COM = all_singles.center_of_mass()
    hmr = all_singles.LagrangianRadii(cm=all_singles.center_of_mass(),mf=[0.75])[0]
    centering_subset = all_singles[(all_singles.position - total_COM).lengths() <= hmr]
    centering_com = centering_subset.center_of_mass() 


    all_vel = all_singles.velocity
    vel_mean = all_vel.mean()
    vel_disp = all_vel.std()

    star_pos = single_stars.position - centering_com
    star_vel = single_stars.velocity.lengths()# - single_stars.center_of_mass_velocity()
    star_vel_sigma = (star_vel - vel_mean)/vel_disp

    bstar_pos = translated_bstars.position - centering_com
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
    scatter4 = ax1.scatter(star_y, -1*star_x, marker='.', s=75, c=np.log10(v),
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
    print "making evolution movie"
    previous_movie_check = os.path.isfile(sim_dir+movie_filename)
    # if previous_movie_check==True:
        # os.remove(sim_dir+movie_filename)
        # print "Removed Previous Simulation Movie" 
    os.system("ffmpeg -framerate 30 -pix_fmt yuv420p -pattern_type glob -i '"+image_dir+"*.png' "+sim_dir+movie_filename)
    print "finished evolution movie"
