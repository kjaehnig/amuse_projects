"""
THIS VERSION OF THE STAR CLUSTER AMUSE SIMULATION SCRIPT USES THE MINIMUM 
TIMESTEP FROM EITHER DYNAMICS, STELLAR EVOLUTION, OR USER-SUPPLIED.
STELLAR EVOLUTION IS HANDLED WITH SEBA
DYNAMICS WITH THE MULTIPLE MODULE AND PH4
SUPERNOVA ARE HANDLED AND RESOLVED WITHIN THE CODE
PRIMORDIAL BINARIES ARE INCLUDED IN THE CODE
    -RANDOM ECCENTRICITY VALUES
    -UNIFORM SAMPLING OF PERIODS AND SEMI-MAJOR AXES(NOT DUQUENNEY DIST)
  
"""
from amuse.lab import *
from amuse.units.quantities import as_vector_quantity
from amuse.couple import encounters
from amuse.couple import collision_handler
from amuse.units import quantities
import numpy
import logging
import csv
import datetime
import os
from matplotlib import pyplot as plt
import numpy as np
from amuse.datamodel.particles import ParticlesSuperset
#import pandas as pd
from amuse.datamodel.rotation import add_spin
from amuse.ext.orbital_elements import new_binary_from_orbital_elements
from amuse.ext.orbital_elements import orbital_elements_from_binary
from collections import Counter
from matplotlib import cm
# plt.rcParams['axes.facecolor'] = 'grey'
import random
from support_functions import MakeMeANewCluster
from support_functions import spatial_plot_module
from support_functions import simple_2d_movie_maker

data_dir = "/home/l_reclusa/Desktop/GRAD/SC/simulation_data/"
# data_dir = "/Users/karljaehnig/Desktop/GRAD/sc_gal_sim/data_files/"
#data_dir = "/home/jaehniko/data_files"

def write_csv(data, filename):
    with open(filename+'.csv', 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)

def binary_fraction_calc(total_num, bin_frac):
    
    number_of_binary_pairs = (total_num * bin_frac) / np.float(2)
    number_of_single_stars = total_num - total_num*bin_frac

    sim_initial_stars = int(number_of_binary_pairs)+int(number_of_single_stars)

    print bin_frac, 2.*number_of_binary_pairs / \
           (number_of_single_stars+2.*number_of_binary_pairs)

    return sim_initial_stars, int(number_of_binary_pairs)

def print_diagnostics(grav, E0=None):

    KE = grav.gravity_code.kinetic_energy
    PE = grav.gravity_code.potential_energy
    Emul = grav.get_total_energy_of_all_multiples()
    Nmul = len(grav.binaries)
    print ""
    # print "time= ", grav.gravity_code.get_time().in_(units.Myr)
    print "    top-level kinetic energy = ", KE
    print "    top-level potential energy = ", PE
    print "    total top-level energy = ", KE+PE
    print "    ",Nmul," multiples, ","total energy = ", Emul
    E= KE + PE + Emul
    print "     UNCORRECTED TOTAL ENERGY =",E

    Etid = grav.multiples_external_tidal_correction \
            + grav.multiples_internal_tidal_correction
    Eerr = grav.multiples_integration_energy_error

    E-= Etid+Eerr
    print "     CALCULATED CORRECTED TOTAL ENERGY =",E
#    print "     INTERNAL CORRECTED TOTAL ENERGY =", grav.corrected_total_energy
    if E0 is not None: print "     relative energy error =", (E-E0)/E0

    return E

def new_smalln(converter, smalln_pro):
    result = SmallN(converter, number_of_workers=smalln_pro)
    result.parameters.timestep_parameter = 0.001
    return result

def new_kepler(converter):
    kepler = Kepler(converter)
    kepler.initialize_code()
    kepler.set_longitudinal_unit_vector(1.0,0.0, 0.0)
    kepler.set_transverse_unit_vector(0.0, 1.0, 0)
    return kepler

def new_binary_orbit(mass1, mass2, semi_major_axis,
                     eccentricity = 0, keyoffset = 1):
    total_mass = mass1 + mass2
    mass_fraction_particle_1 = mass1 / (total_mass)
    
    binary = Particles(2)
    binary[0].mass = mass1
    binary[1].mass = mass2
    
    mu = constants.G * total_mass
    
    velocity_perihelion \
        = numpy.sqrt( mu / semi_major_axis * ((1.0 + eccentricity)
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
    rnd =numpy.random.uniform(rnd_min, rnd_max, 1)
    Porb = ((rnd/(1.-rnd))**3.3)/mpf | units.day
    Mtot = Mprim + Msec
    a = ((constants.G*Mtot) * (Porb/(2*numpy.pi))**2)**(1./3.)
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


# def make_massive_binaries(massive_stars):
#     massive_binaries = Particles()
#     singles_in_massive_binaries = Particles()
#     massive_stars = massive_stars[np.argsort(massive_stars.mass)][::-1]
#     num_massive_stars = len(massive_stars.mass)
#     for bi in np.arange(num_massive_stars):
#         if bi < num_massive_stars - 1:
#             mp = massive_stars.mass[bi]
#             ms = massive_stars.mass[bi+1]
#         if bi == num_massive_stars -1:
#             mp = massive_stars.mass[bi]
#             ms = massive_stars.mass[bi]*0.9
#         a = random_semimajor_axis_PPE(mp, ms)
#         e = numpy.sqrt(numpy.random.random())

#         nb = new_binary_orbit(mp, ms, a, e) 
#         nb.position += massive_stars.position[bi]
#         nb.velocity += massive_stars.velocity[bi]
#         nb = singles_in_massive_binaries.add_particles(nb)
#         nb.radius = 0.01 * a 

#         massive_stars.radius[bi] = 3*a 
#         binary_particle = massive_stars[bi].copy()
#         binary_particle.child1 = nb[0]
#         binary_particle.child2 = nb[1]
#         binary_particle.semi_major_axis = a
#         binary_particle.eccentricity = e
#         massive_binaries.add_particle(binary_particle)

#     return massive_binaries, singles_in_massive_binaries

def calculate_orbital_elementss(bi, converter):
    kep = new_kepler(converter)
    comp1 = bi.child1
    comp2 = bi.child2
    mass = (comp1.mass + comp2.mass)
    pos = (comp2.position - comp1.position)
    vel = (comp2.velocity - comp1.velocity)
    kep.initialize_from_dyn(mass, pos[0], pos[1], pos[2],
                            vel[0], vel[1], vel[2])
    a,e = kep.get_elements()
    kep.stop()
    return a, e


###BOOKLISTSTART###
def resolve_changed_binaries(stopping_condition, stellar, multiples_code, converter):
    new_binaries = stopping_condition.particles(0)
    for bi in new_binaries:
        print "add new binary:", bi
        a, e = calculate_orbital_elementss(bi, converter)
        bi.semi_major_axis = a
        bi.eccentricity = e
        stellar.binaries.add_particle(bi)
        # print "new binary parameters", a, e
        print bi

    lost_binaries = stopping_condition.particles(1)
    for bi in lost_binaries:
        print "remove old binary:", bi.key
        stellar.binaries.remove_particle(bi)

    changed_binaries = stopping_condition.particles(2)
    for bi in changed_binaries:
        bs = bi.as_particle_in_set(stellar.binaries)
        a, e = calculate_orbital_elementss(bi, converter)
        bs.semi_major_axis = a
        bs.eccentricity = e
        print "Modified binary parameters", a, e
        # print bs
###BOOKLISTSTOP##



def update_dynamical_binaries_from_stellar(stellar, multiples_code, converter):
    kep = new_kepler(converter)

    # THIS NEEDS TO BE CHECKED!
    print "++++++++++++ THIS NEEDS TO BE CHECKED ++++++++++++++++++++"

    print "Number of binaries=", len(stellar.binaries)
    for bi in stellar.binaries:
        bs = bi.as_particle_in_set(multiples_code.binaries)
        total_mass = bi.child1.mass+bi.child2.mass 
        kep.initialize_from_elements(total_mass, bi.semi_major_axis,
                                     bi.eccentricity)
        rel_position = as_vector_quantity(kep.get_separation_vector())
        rel_velocity = as_vector_quantity(kep.get_velocity_vector())
        mu = bi.child1.mass / total_mass 
        bs.child1.position = mu * rel_position 
        bs.child2.position = -(1-mu) * rel_position 
        bs.child1.velocity = mu * rel_velocity
        bs.child2.velocity = -(1-mu) * rel_velocity
        # print "semi_major_axis=", bi.semi_major_axis, total_mass, \
        #       bi.child1.mass, bi.child2.mass, bi.eccentricity
    kep.stop()
        



def resolve_supernova(supernova_detection, bodies, time):
    if supernova_detection.is_set():
        print "At time=", time.in_(units.Myr), "supernova detected=", len(supernova_detection.particles(0))
        Nsn = 0
        group = bodies.particles
        if supernova_detection.particles(0)[0] in bodies.singles_in_binaries:
            group = bodies.singles_in_binaries
            print "Supernova in binary"
        if supernova_detection.particles(0)[0] in bodies.particles:
            group = bodies.particles
            print "Single stellar supernova"
        for ci in range(len(supernova_detection.particles(0))):
            small_timestep = True
            print supernova_detection.particles(0)[0]
            particles_in_supernova = Particles(particles=supernova_detection.particles(0))
            natal_kick_x = particles_in_supernova.natal_kick_x
            natal_kick_y = particles_in_supernova.natal_kick_y
            natal_kick_z = particles_in_supernova.natal_kick_z

            print "natal kick vector: ",natal_kick_x, natal_kick_y, natal_kick_z
            particles_in_supernova = particles_in_supernova.get_intersecting_subset_in(group)
            particles_in_supernova.vx += natal_kick_x
            particles_in_supernova.vy += natal_kick_y
            particles_in_supernova.vz += natal_kick_z

            Nsn+=1
        print "Resolve supernova Number:", Nsn
        
        return small_timestep

def runaway_detector(multiples_code):
    stars = multiples_code.all_singles
    runaway_condition = multiples_code.all_singles.velocity.lengths_squared().sqrt().value_in(units.kms) >= 600 
    number_runaways = len(multiples_code.all_singles[runaway_condition])
    return number_runaways

    
def resolve_binary_merger(stellar, multiples_code):
    condition = stellar.binaries.semi_major_axis.value_in(units.parsec) == 0.0 
    merger_binaries = stellar.binaries[condition]

    for bi in merger_binaries:
        bm = bi.as_particle_in_set(multiples_code.binaries)
        comp1 = bm.child1.as_particle_in_set(multiples_code.all_singles)
        comp2 = bm.child2.as_particle_in_set(multiples_code.all_singles)

        print comp1

        dyn_comp1 = multiples_code.all_singles[multiples_code.all_singles.key==comp1.key]
        dyn_comp2 = multiples_code.all_singles[multiples_code.all_singles.key==comp2.key]
        dyn_bin = multiples_code.binaries[multiples_code.binaries.key==bi.key]

        bstar1 = stellar.particles[stellar.particles.key==comp1.key]
        bstar2 = stellar.particles[stellar.particles.key==comp2.key]

        dynamical_set = Particles(particles=[comp1, comp2])
        new_key = comp1.key+comp2.key
        new_particle = Particle(new_key)
        new_particle.mass = bm.mass
        new_particle.age = 0.0 | units.Myr
        new_particle.velocity = dynamical_set.center_of_mass_velocity()
        new_particle.position = dynamical_set.center_of_mass()

        print "Binary Merger Occurance: "
        print "Components:    ",comp1.key,"       ", comp2.key
        print "Stellar Type: ",bi.child1.stellar_type, bi.child1.mass.in_(units.MSun)
        print "              ",bi.child2.stellar_type, bi.child2.mass.in_(units.MSun)

        stellar.particles.remove_particle(comp1)
        stellar.particles.remove_particle(comp2)
        stellar.binaries.remove_particle(bi)
        print "Removed Binary ", bi.key

        multiples_code.singles_in_binaries.remove_particle(dyn_comp1)
        multiples_code.singles_in_binaries.remove_particle(dyn_comp2)

        multiples_code.binaries.remove_particle(dyn_bin)

        stellar.particles.add_particle(new_particle)
        multiples_code.particles.add_particle(new_particle)

        print "New single star added ", new_particle.key,"\n"
        print "Sanity Check: ",len(multiples_code.singles_in_binaries), len(multiples_code.singles), len(stellar.particles)



def check_for_merger(stellar, multiples_code):
    condition_value = False

    for bi in stellar.binaries:
        r_sum = (bi.child1.radius + bi.child2.radius)
        bs1 = bi.child1.as_particle_in_set(multiples_code.singles_in_binaries)
        bs2 = bi.child2.as_particle_in_set(multiples_code.singles_in_binaries)
        sep = (bs1.position - bs2.position).length()
        if r_sum < sep:
            condition_value = False 
        else: 
            print "Binary Merger Detected"
            condition_value = True
            break

    return condition_value

def merge_two_stars(bodies, particles_in_encounter):
    com_pos = particles_in_encounter.center_of_mass()
    com_vel = particles_in_encounter.center_of_mass_velocity()
    new_particle=Particles(1)
    new_particle.mass = particles_in_encounter.total_mass()
    new_particle.age = min(particles_in_encounter.age) * max(particles_in_encounter.mass)/new_particle.mass
    new_particle.position = com_pos
    new_particle.velocity = com_vel
    new_particle.radius = 0 | units.RSun
    bodies.add_particles(new_particle)
    print "Two stars (M=",particles_in_encounter.mass,") collided at d=", com_pos.length()
    bodies.remove_particles(particles_in_encounter)



def resolve_collision(collision_detection, gravity, stellar, bodies):
    if collision_detection.is_set():
        E_coll = gravity.gravity_code.kinetic_energy + gravity.gravity_code.potential_energy
        print "Collision at time=", gravity.model_time.in_(units.Myr)
        for ci in range(len(collision_detection.particles(0))): 
            particles_in_encounter = Particles(particles=[collision_detection.particles(0)[ci], collision_detection.particles(1)[ci]])
            particles_in_encounter = particles_in_encounter.get_intersecting_subset_in(bodies)
            d = (particles_in_encounter[0].position-particles_in_encounter[1].position).length()
            if particles_in_encounter.collision_radius.sum()>d:
                merge_two_stars(bodies, particles_in_encounter)
                bodies.synchronize_to(gravity.particles)
                bodies.synchronize_to(stellar.particles)
            else:
                print "Encounter failed to resolve, because the stars were too small."
            dE_coll = E_coll - (gravity.gravity_code.kinetic_energy + gravity.gravity_code.potential_energy)
        print "Energy error in the collision: dE =", dE_coll 







def main(
        N=100,
        Binfrac=0.5,
        HMR=1.,
        VR=0.5,
        Tend=10,
        Numsteps=10,
        Mdist='K',
        Sdist='P',
        Mmin=1.0,
        Mmax=100.,
        filename='cluster1',
        seed=2501,
        nproc=1,
        smalln_pro=1,
        NF=1.0,
        HBF=3.0,
        SF=10.0
        ):

# N=100
# Binfrac=0.50
# HMR=1.  | units.parsec
# VR=0.25
# Tend=10  | units.Myr
# Numsteps=100
# Mdist='K'
# Sdist='F'
# Mmin=10.
# Mmax=100.
# filename='cluster1'
# seed=2503
# nproc=1
# smalln_pro=1
# NF=1.0
# HBF=3.0
# SF=10.0



    start_time = datetime.datetime.now()
    timestamp = "-"+str(start_time.year)+'-'+str(start_time.month)+'-'+str(start_time.day)
    filename = filename+timestamp+"-"+str(start_time.microsecond)
    write_csv(['step', 'time', 'energy_error', 'step_time'], data_dir+filename)

    sim_num,num_bin = binary_fraction_calc(N, Binfrac)
    tend, nsteps, N, Nbin = Tend, Numsteps, sim_num, num_bin
    # print seed
    random.seed(seed)
    np.random.seed(seed)

    filename_number = str(
                        np.random.choice(
                            np.arange(0,10000),size=1
                            )
                        )
    #    filename = "cluster"+filename_number
    print filename
    file_dir_check = os.path.isdir(data_dir+str(filename))


    if file_dir_check == False:
        os.makedirs(data_dir+str(filename))

    sim_dir = data_dir+str(filename)+"/"
    
    print "Made image directory"
    image_dir = sim_dir+"images/"
    os.makedirs(image_dir)
    logging.basicConfig(level=logging.ERROR)

    print N, Mmin, Mmax,HMR
    stars, converter = MakeMeANewCluster(N=sim_num, HalfMassRadius=HMR, mdist=Mdist,
                        kind=Sdist, frac_dim=1.6, 
                        W=None, mmin=Mmin, mmax=Mmax, SEED=seed)

    code = ph4(converter, mode = 'cpu', number_of_workers=nproc-1)
    code.initialize_code()
    code.parameters.set_defaults()
    code.parameters.epsilon_squared = 0.01|units.RSun**2.
    # code.parameters.dt_param = 0.0001
    # code.parameters.use_gpu = 1
    code.parameters.timestep_parameter = 0.001
    stop_cond = code.stopping_conditions.collision_detection
    stop_cond.enable()

    code1 = datetime.datetime.now()
    code1_delta = (code1-start_time).total_seconds()
    write_csv([-2,code1_delta, 999, 999],data_dir+filename)

    single_stars, binary_stars, singles_in_binaries \
        = make_secondaries(stars, Nbin)

    single_stars.move_to_center()
    binary_stars.move_to_center()
    single_stars.scale_to_standard(convert_nbody=converter,virial_ratio=VR/2.)
    binary_stars.scale_to_standard(convert_nbody=converter,virial_ratio=VR/2.)

    stellar = SeBa()
    stellar.parameters.metallicity = 0.02
    stellar.particles.add_particles(single_stars)
    stellar.particles.add_particles(singles_in_binaries)
    stellar.binaries.add_particles(binary_stars)

    code2 = datetime.datetime.now()
    code2_delta = (code2-code1).total_seconds()
    write_csv([-1,code2_delta, 999, 999],data_dir+filename)

    supernova_detection = stellar.stopping_conditions.supernova_detection
    supernova_detection.enable()

    encounter_code = encounters.HandleEncounter(
        kepler_code =  new_kepler(converter),
        resolve_collision_code = new_smalln(converter, smalln_pro=smalln_pro),
        interaction_over_code = None,
        G = constants.G
    )
    multiples_code = encounters.Multiples(
        gravity_code = code,
        handle_encounter_code = encounter_code,
        G = constants.G
    )


    multiples_code.particles.add_particles((stars-binary_stars).copy())
    multiples_code.singles_in_binaries.add_particles(singles_in_binaries)
    multiples_code.binaries.add_particles(binary_stars)
    multiples_code.commit_particles()

    #    multiples_code.global_debug = 3
    multiples_code.handle_encounter_code.parameters.neighbours_factor = NF
    multiples_code.handle_encounter_code.parameters.hard_binary_factor = HBF
    multiples_code.handle_encounter_code.parameters.scatter_factor = SF    

    #OmegaVector = [0., 0., 1e-15]|units.s**-1
    #add_spin(multiples_code.particles, OmegaVector)


    multiples_code.evolve_model(0.0|units.Myr)

    individual_stars = multiples_code.singles.copy()
    binary_stars = multiples_code.binaries.copy()
    singles_in_binaries = multiples_code.singles_in_binaries.copy()

    channel_SE_to_DYN = stellar.particles.new_channel_to(multiples_code.particles,
                        attributes=["mass","radius"],
                        target_names=["mass","radius"])
    channel_SE_to_DYN.copy()

    channel_SE_to_DYN_bins = stellar.binaries.new_channel_to(multiples_code.binaries,
                       attributes=["eccentricity","semi_major_axis"],
                       target_names=["eccentricity","semi_major_axis"])


    se_attributes = stellar.particles.get_attribute_names_defined_in_store()
    channel_SE_to_stars = stellar.particles.new_channel_to(individual_stars,
                       attributes=se_attributes,
                       target_names=se_attributes)
    channel_SE_to_stars.copy()

    dyn_attributes = multiples_code.particles.get_attribute_names_defined_in_store()
    channel_DYN_to_stars = multiples_code.singles.new_channel_to(individual_stars,
                       attributes=["x","y","z","vx","vy","vz"],
                       target_names=["x","y","z","vx","vy","vz"])
    channel_DYN_to_stars.copy()

    channel_DYN_to_bins = multiples_code.particles.new_channel_to(binary_stars,
                       attributes=["x","y","z","vx","vy","vz"],
                       target_names=["x","y","z","vx","vy","vz"])
    channel_DYN_to_bins.copy()

    channel_DYN_to_bstars = multiples_code.singles_in_binaries.new_channel_to(singles_in_binaries,
                       attributes=["x","y","z","vx","vy","vz"],
                       target_names=["x","y","z","vx","vy","vz"])
    channel_DYN_to_bstars.copy()


    SEB_attributes = stellar.binaries.get_attribute_names_defined_in_store()
    channel_SE_to_bins = stellar.binaries.new_channel_to(binary_stars)
    channel_SE_to_bins.copy()

    stopping_condition \
        = multiples_code.stopping_conditions.binaries_change_detection
    stopping_condition.enable()

    i = 0

    time = 0.0|tend.unit
    x=0

    # spatial_plot_module(individual_stars, singles_in_binaries, binary_stars, time, x, image_dir)
    # write_set_to_file(individual_stars.savepoint(time),ssf_dir+"single_stars.hdf5","hdf5")
    # write_set_to_file(singles_in_binaries.savepoint(time), bsf_dir+"binary_singles.hdf5","hdf5")
    # write_set_to_file(binary_stars.savepoint(time),bsf_dir+"binary_stars.hdf5","hdf5")

    print tend, Numsteps
    sim_dt = tend.value_in(units.Myr) / np.float(nsteps)
    print sim_dt

    snapshot_timesteps = np.arange(sim_dt,tend.value_in(units.Myr)+sim_dt, sim_dt).astype("float")
    print snapshot_timesteps

    sim_dt = tend.value_in(units.Myr)/ np.float(nsteps*10.)

    print "--------------------------------------------"
    print "Starting the simulation now..."
    print "--------------------------------------------"

    sim_start = datetime.datetime.now()
    sim_start_delta = (code2 - sim_start).total_seconds()
    write_csv([0, sim_start_delta, 0.0, 0.0], data_dir+filename)

    ii=1
    while time < tend:
        loop_time1 = datetime.datetime.now()
        index = str(x)
        index = index.zfill(4)

        se_timestep = stellar.particles.time_step.min()
        dt_min = min(sim_dt|units.Myr, se_timestep)
        print "Time steps:",sim_dt, se_timestep.in_(units.Myr) 

        time += dt_min

        E0 = print_diagnostics(multiples_code)
        print "Evolving Model..."
        multiples_code.evolve_model(time)
        print "Done Evolving Model."
        ER = print_diagnostics(multiples_code, E0)

        RelErr = (ER - E0) / E0
        channel_DYN_to_stars.copy()
        channel_DYN_to_bstars.copy()
        channel_DYN_to_bins.copy()

        print "at t=", multiples_code.model_time, \
              "Nmultiples:", len(stellar.binaries[stellar.binaries.semi_major_axis.in_(units.m)!=0.0|units.m])

        print "Checking for any binary changes..."
        if stopping_condition.is_set():
            resolve_changed_binaries(stopping_condition, stellar, multiples_code, converter)
        print "Done checking for changed binaries."
        
        print "Evolving the stellar equations..."
        stellar.evolve_model(time)
        print "Done evolving stellar equations."
        small_timestep=resolve_supernova(supernova_detection, multiples_code, time)

        stellar_separations = stellar.binaries.semi_major_axis.value_in(units.RSun)

        update_dynamical_binaries_from_stellar(stellar, multiples_code,
                                               converter)

        channel_SE_to_DYN.copy()
        
        loop_time2 = datetime.datetime.now()
        loop_delta = (loop_time2 - loop_time1).total_seconds()
        write_csv([ii, loop_delta, RelErr, time.value_in(units.Myr)], data_dir+filename)
        ii+=1
        # if x==5: break

        channel_SE_to_stars.copy()
        channel_SE_to_bins.copy()

        if len(snapshot_timesteps)==0: break
        epsilon = abs(time.value_in(units.Myr)-snapshot_timesteps[0]) 

        x += 1
         
         
        epsilon_check = epsilon < 0.25

        if epsilon_check:
           print "wrote out files at time: ", np.round(time.value_in(units.Myr),2), time.value_in(units.Myr)
           spatial_plot_module(individual_stars, singles_in_binaries, binary_stars, 1,time, x, image_dir)
           write_set_to_file(individual_stars.savepoint(time),sim_dir+"single_stars.hdf5","hdf5")
           write_set_to_file(singles_in_binaries.savepoint(time), sim_dir+"binary_singles.hdf5","hdf5")
           write_set_to_file(binary_stars.savepoint(time),sim_dir+"binary_stars.hdf5","hdf5")
           snapshot_timesteps = np.delete(snapshot_timesteps, 0)
           print snapshot_timesteps



        print "t, Energy=", time, multiples_code.get_total_energy()

    simple_2d_movie_maker("output_movie_evolution", img_dir=image_dir)


def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()
    result.add_option("-N", dest="N",type="int", default=100)
    result.add_option("--Binfrac", dest="Binfrac",type="float", default=0.5)
    result.add_option("--HMR", unit=units.parsec,dest="HMR",type="float", default=1|units.parsec)
    result.add_option("--Virialratio", dest="VR", type="float", default=0.5)
    result.add_option("--Tend", unit=units.Myr,dest="Tend",type="float", default=10.|units.Myr)
    result.add_option("--Numsteps", dest="Numsteps",type="int", default=10)
    result.add_option("--Mdist", dest="Mdist", default="K")
    result.add_option("--Sdist", dest="Sdist", default="P")
    result.add_option("--Mmin", dest="Mmin", type="float", default=0.5)
    result.add_option("--Mmax", dest="Mmax", type="float", default=100.)
    result.add_option("--Filename", dest="filename", default="cluster1")
    result.add_option("--Seed", dest="seed",type="int", default=2501)
    result.add_option("--Npro", dest="nproc",type="int", default=1)
    result.add_option("--Smalln_pro",dest="smalln_pro",type="int",default=1)
    result.add_option("--NF", dest="NF", type="float", default=1.0)
    result.add_option("--HBF", dest="HBF", type="float", default=3.0)
    result.add_option("--SF", dest="SF",type="float", default=10.0)
    return result



if __name__ == "__main__":
    opt, arguments  = new_option_parser().parse_args()
    main(**opt.__dict__)
    set_printing_strategy("custom", 
                      preferred_units = [units.MSun, units.parsec, units.Myr], 
                      precision = 4, prefix = "", 
                      separator = " [", suffix = "]")

    # options, arguments  = new_option_parser().parse_args()
    # if options.seed >= 0:
    #     numpy.random.seed(options.seed)
    #     # This is only for random.sample, which apparently does not use numpy
    #     import random
    #     random.seed(options.seed)
    # main(opt.N, 
    #     opt.Binfrac, 
    #     opt.HMR, 
    #     opt.Virialratio,
    #     opt.Tend,
    #     opt.Numsteps,
    #     opt.Mdist,
    #     opt.Sdist,
    #     opt.Mmin,
    #     opt.Mmax,
    #     opt.Filename,
    #     opt.Seed,
    #     opt.Npro,
    #     opt.Smalln_pro,
    #     opt.NF,
    #     opt.HBF,
    #     opt.SF)






