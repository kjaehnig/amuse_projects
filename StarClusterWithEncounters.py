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
from support_functions import make_secondaries_with_massives2
from support_functions import MakeMeANewCluster
from support_functions import simple_spatial_plot_module
from support_functions import simple_2d_movie_maker
from support_functions import energy_conv_plot
from amuse.support.console import set_printing_strategy
from MASC.binaries import orbital_period_to_semi_major_axis
from amuse.ext.orbital_elements import generate_binaries
# from MASC.binaries import *




set_printing_strategy("custom", 
                  preferred_units = [units.MSun, units.parsec, units.Myr], 
                  precision = 4, prefix = "", 
                  separator = " [", suffix = "]")

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

    # KE = grav.gravity_code.kinetic_energy
    # PE = grav.gravity_code.potential_energy
    Emul = grav.get_total_energy()
    Nmul = len(grav.binaries)
    print ""
    print "time=    ", grav.gravity_code.get_time().in_(units.Myr)
    # print "    virial-ratio of cluster=   ", 2*KE/(abs(PE))
    # print "    top-level kinetic energy = ", KE
    # print "    top-level potential energy = ", PE
    # print "    total top-level energy = ", KE+PE
    print "    ",Nmul," multiples, ","total energy = ", Emul
    E=Emul
    print "     UNCORRECTED TOTAL ENERGY =",E

    Etid = grav.multiples_external_tidal_correction \
            + grav.multiples_internal_tidal_correction
    Eerr = grav.multiples_integration_energy_error

    E-= Etid+Eerr
    print "     CALCULATED CORRECTED TOTAL ENERGY =",E
#    print "     INTERNAL CORRECTED TOTAL ENERGY =", grav.corrected_total_energy
    if E0 is not None: print "     relative energy error =", (E-E0)/E0

    return E


# def new_smalln(converter, smalln_pro):
#     result = SmallN(converter, number_of_workers=smalln_pro)
#     result.parameters.timestep_parameter = .0001
#     result.parameters.cm_index = 10001
#     # result.parameters.allow_full_unperturbed = 0
#     return result

SMALLN = None
def init_smalln(converter,smalln_pro):
    global SMALLN
    SMALLN = SmallN(converter, number_of_workers=smalln_pro)
    SMALLN.parameters.timestep_parameter = .05
    SMALLN.parameters.cm_index = 10001


def new_smalln():
    SMALLN.reset()
    return SMALLN

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
def random_semimajor_axis_PPE(Mprim, Msec, P_min=1.0|units.day,
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
    resulting_binaries = Particles()
    singles_in_binaries = Particles()
    binaries = center_of_masses.random_sample(Nbin)
    mmin = center_of_masses.mass.min()
    print mmin
    for bi in binaries:
        mp = bi.mass
        ms = numpy.random.uniform(mmin.value_in(units.MSun),
                                  mp.value_in(units.MSun)) | units.MSun
        a = random_semimajor_axis_PPE(mp, ms)
        e = numpy.sqrt(numpy.random.random())

        nb = new_binary_orbit(mp, ms, a, e) 
        nb.position += bi.position
        nb.velocity += bi.velocity
        nb[0].radius = nb[0].mass.value_in(units.MSun)**(3./7) | units.RSun
        nb[1].radius = nb[1].mass.value_in(units.MSun)**(3./7) | units.RSun
        nb = singles_in_binaries.add_particles(nb)
        # nb.radius = a 

        bi.radius = (nb[0].position - nb[1].position).length()
        binary_particle = bi.copy()
        binary_particle.child1 = nb[0]
        binary_particle.child2 = nb[1]
        binary_particle.semi_major_axis = a
        binary_particle.eccentricity = e
        resulting_binaries.add_particle(binary_particle)

    single_stars = center_of_masses-binaries
    return single_stars, resulting_binaries, singles_in_binaries

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
        nb[0].radius = nb[0].mass.value_in(units.MSun)**(3./7) | units.RSun
        nb[1].radius = nb[1].mass.value_in(units.MSun)**(3./7) | units.RSun
        nb = singles_in_binaries.add_particles(nb)
        # nb.radius = a 

        # binary_particle.radius = (nb[0].position - nb[1].position).length()
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
    stellar.commit_particles()
    multiples_code.commit_particles()
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

    
def resolve_binary_dynamics(converter, stellar, multiples_code, 
                        individual_stars, singles_in_binaries, binary_stars, 
                        timestep, stopping_condition):

    resolve_changed_binaries(stopping_condition, stellar, multiples_code, converter)

    bsingles_to_remove = Particles()

    condition = stellar.binaries.semi_major_axis.value_in(units.parsec) == 0.0 
    merged_binaries = stellar.binaries.key[condition]
    number_mergers = 0
    for ii in range(len(multiples_code.binaries)):

        bi = multiples_code.binaries[ii]
        bis = stellar.binaries[ii]

        loc_in_multiples_parts = np.in1d(multiples_code.particles.key,bi.key)
        loc_in_stellar_parts = np.in1d(stellar.particles.key,bis.key)

        mA,mB = bi.child1.mass, bi.child2.mass
        total_mass = mA + mB
        com = bi.position
        comv = bi.velocity
        if (bi.semi_major_axis == 0 |units.parsec):
            number_mergers += 1
            print "Binary Merger Occurance: "
            print "Components:    ",bi.child1.key,"       ", bi.child2.key
            print "Stellar Type: ",bis.child1.stellar_type, bis.child2.mass.in_(units.MSun)
            print "              ",bis.child2.stellar_type, bis.child2.mass.in_(units.MSun)

            bc = binary_stars[ii].as_particle_in_set(multiples_code.particles)
            bc_in_stellar = binary_stars[ii].as_particle_in_set(stellar.particles)
            if bc == None: bc = binary_stars[ii]

            merged = Particle()
            merged.mass = 0.95 * total_mass
            merged.position = com + (comv*timestep)
            merged.velocity = comv
            merged.stellar_type = 1 | units.stellar_type

            ### REMOVE FROM THE PARTICLE SETS #####
            # singles_in_binaries.remove_particle(binary_stars[ii].child1)
            # singles_in_binaries.remove_particle(binary_stars[ii].child1)
            bsingles_to_remove.add_particle(binary_stars[ii].child1)
            bsingles_to_remove.add_particle(binary_stars[ii].child2)
            stellar.particles.remove_particle(stellar.particles[loc_in_stellar_parts])
            multiples_code.particles.remove_particle(multiples_code.particles[loc_in_multiples_parts])

            #### REMOVE FROM THE BINARY SETS #####
            stellar.binaries.remove_particle(bis)
            multiples_code.binaries.remove_particle(bi)
            binary_stars.remove_particle(binary_stars[ii])

            ### ADD NEW PARTICLE
            stellar.particles.add_particle(merged)
            multiples_code.particles.add_particle(merged)
            individual_stars.add_particle(merged)

    singles_in_binaries.remove_particles(bsingles_to_remove)
    return number_mergers

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


def particle_dataset_bookkeeping(stellar, 
                                multiples_code,
                                single_stars,
                                singles_in_binaries,
                                binary_stars):

    zero_mass_stars = stellar.particles.mass == 0.0|units.MSun
    if np.sum(zero_mass_stars) > 0:
        print("There are non-existent stars that need to be removed...")

        binary_removals = []
        single_star_removals = []
        single_star_additions = Particles()

        for bi in stellar.binaries:
            bs = bi.as_particle_in_set(multiples_code.binaries)
            
            children = Particles()
            children.add_particle(bi.child1.as_particle_in_set(multiples_code.singles_in_binaries))
            children.add_particle(bi.child2.as_particle_in_set(multiples_code.singles_in_binaries))

            children.position += bs.position
            children.velocity += bs.velocity
            children.mass[0] = bi.child1.mass
            children.mass[1] = bi.child2.mass

            if np.sum(children.mass == 0|units.MSun) > 0:
                print("----------------------------------------------")
                print("Found the problematic binary stars...")
                print("Making replacement particle...")
                id_creation = datetime.datetime.now()
                random_id = int(str(id_creation.year) +\
                                str(id_creation.month) +\
                                    str(id_creation.day) +\
                                        str(id_creation.hour) +\
                                            str(id_creation.minute) +\
                                                str(id_creation.microsecond)) 
                print(random_id)
                replacement_single = Particle(keys=random_id)

                problematic_binary_key = bi.key
                binary_removals.append(problematic_binary_key)
                single_star_removals.append(children.key)

                zero_mass_star = children[children.mass == 0.0|units.MSun]
                non_zero_mass_star = children[children.mass != 0.0|units.MSun]

                replacement_single.mass = non_zero_mass_star.mass
                replacement_single.position = [non_zero_mass_star.x,
                                                non_zero_mass_star.y,
                                                non_zero_mass_star.z]#bs.position
                replacement_single.velocity = bs.velocity
                replacement_single.age = 0|units.Myr
                single_star_additions.add_particle(replacement_single)

        print("Removing binaries from binary subsets...")
        binary_stars.remove_particle(
                binary_stars[np.in1d(multiples_code.binaries.key,binary_removals)]
                                    )
        multiples_code.binaries.remove_particle(
                multiples_code.binaries[np.in1d(multiples_code.binaries.key,binary_removals)]
                                                )
        stellar.binaries.remove_particle(
            stellar.binaries[np.in1d(stellar.binaries.key,binary_removals)]
                                        )

        print("Removing individual stars from singles subsets...")
        singles_in_binaries.remove_particles(singles_in_binaries[np.in1d(singles_in_binaries.key,single_star_removals)])
        multiples_code.singles_in_binaries.remove_particles(multiples_code.singles_in_binaries[np.in1d(multiples_code.singles_in_binaries.key, single_star_removals)])
        multiples_code.all_singles.remove_particles(multiples_code.all_singles[np.in1d(multiples_code.all_singles.key, single_star_removals)])

        stellar.particles.remove_particles(stellar.particles[np.in1d(stellar.particles.key,single_star_removals)])


        print("Adding in replacement particles...")
        single_stars.add_particle(single_star_additions)
        multiples_code.particles.add_particle(single_star_additions)
        stellar.particles.add_particle(single_star_additions)

        # multiples_code.commit_particles()
        # stellar.commit_particles()

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
        TimeUnit='myr',
        Numsteps=10,
        Mdist='K',
        Sdist='P',
        Mmin=1.0,
        Mmax=100.,
        filename='SC',
        rot=False,
        seed=2501,
        nproc=1,
        smalln_pro=1,
        DTPARAM=0.05,
        NF=1.0,
        HBF=3.0,
        SF=10.0,
        RUN_GPU=1,
        IMSIZE=5, 
        DIR=None,
        MACHINE='linux'
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

# data_dir = "/home/l_reclusa/Desktop/GRAD/SC/simulation_data/"
    if MACHINE.lower() == 'mac':
        data_dir = "/Users/karljaehnig/Desktop/GRAD/sc_gal_sim/data_files/"
    if MACHINE.lower() == 'accre': 
        data_dir = "/home/jaehniko/data_files"
    if MACHINE.lower() == 'linux':
        data_dir = "/home/l_reclusa/Desktop/GRAD/SC/simulation_data/"

    # if DIR==None:
        # data_dir = data_dir
    if DIR!=None:
        data_dir = data_dir+DIR+"/"

    if TimeUnit.lower()=='myr': tend = Tend|units.Myr
    if TimeUnit.lower()=='yr': tend = Tend|units.yr

    start_time = datetime.datetime.now()
    timestamp = "-"+str(start_time.year)+'-'+str(start_time.month)+'-'+str(start_time.day)
    if VR == 0.5: vflag = '-V'
    if VR > 0.5: vflag = '-SV'
    if VR < 0.5: vflag = '-sV'
    seed_tag = str(seed)
    filename = filename+vflag+timestamp+"-"+str(start_time.microsecond)+"-"+seed_tag
    print filename
    file_dir_check = os.path.isdir(data_dir+str(filename))

    if file_dir_check == False:
        os.makedirs(data_dir+str(filename))

    sim_dir = data_dir+str(filename)+"/"

    write_csv(['step', 'time', 'energy_error', 'step_time'], sim_dir+filename)

    sim_num,num_bin = binary_fraction_calc(N, Binfrac)
    nsteps, N, Nbin = Numsteps, sim_num, num_bin
    # print seed
    random.seed(seed)
    np.random.seed(seed)

    filename_number = str(
                        np.random.choice(
                            np.arange(0,10000),size=1
                            )
                        )
    #    filename = "cluster"+filename_number

    
    print "Made image directory"
    image_dir = sim_dir+"images/"
    os.makedirs(image_dir)
    logging.basicConfig(level=logging.ERROR)

    print N, Mmin, Mmax,HMR
    stars, converter = MakeMeANewCluster(N=sim_num, HalfMassRadius=HMR, mdist=Mdist,
                        kind=Sdist, frac_dim=1.6, 
                        W=None, mmin=Mmin, mmax=Mmax, SEED=seed)
    # converter = generic_unit_converter.ConvertBetweenGenericAndSiUnits(1|units.MSun,
    #                                                                     1|units.parsec,
    #                                                                     1|units.kms)
    if RUN_GPU:
        code = ph4(converter, 
                    mode = 'gpu',
                    redirection='none')
        code.initialize_code()
        code.parameters.set_defaults()
        code.parameters.use_gpu = 1
        code.parameters.timestep_parameter = DTPARAM
    if not RUN_GPU:
        code = ph4(converter, 
                    mode = 'cpu',
                    redirection='none',
                    number_of_workers=nproc)
        code.initialize_code()
        code.parameters.set_defaults()
        code.parameters.timestep_parameter = DTPARAM
    if RUN_GPU==2:
        code = Hermite(converter,
                        number_of_workers=nproc)
        code.initialize_code()
        code.parameters.set_defaults()
        code.parameters.dt_param = DTPARAM
    # code.parameters.dt_param = 0.001
    # code.parameters.force_sync = 1
    code.parameters.epsilon_squared = 1|units.RSun**2.
    # code.parameters.block_steps = 1 
    # code.parameters.dt_param = 0.0001
    # code.parameters.use_gpu = 1
    # stop_cond = code.stopping_conditions.collision_detection
    # stop_cond.enable()

    code1 = datetime.datetime.now()
    code1_delta = (code1-start_time).total_seconds()
    # write_csv([-2,code1_delta, 999, 999],sim_dir+filename)

    single_stars, binary_stars, singles_in_binaries \
        = make_secondaries_with_massives2(stars, Nbin)

    # single_stars.move_to_center()
    # binary_stars.move_to_center()

    # single_stars.scale_to_standard(convert_nbody=converter,
    #                                 virial_ratio=VR/2.,
    #                                 smoothing_length_squared=code.parameters.epsilon_squared)
    # binary_stars.scale_to_standard(convert_nbody=converter,
    #                                 virial_ratio=VR/2.,
    #                             smoothing_length_squared=code.parameters.epsilon_squared)

    stellar = SeBa()
    stellar.initialize_code()
    stellar.parameters.set_defaults()
    stellar.parameters.metallicity = 0.02
    stellar.particles.add_particles(single_stars)
    stellar.particles.add_particles(singles_in_binaries)
    stellar.binaries.add_particles(binary_stars)
    stellar.commit_particles()

    code2 = datetime.datetime.now()
    code2_delta = (code2-code1).total_seconds()
    # write_csv([-1,code2_delta, 999, 999],sim_dir)

    supernova_detection = stellar.stopping_conditions.supernova_detection
    supernova_detection.enable()

    init_smalln(converter,smalln_pro)
    encounter_code = encounters.HandleEncounter(
        kepler_code =  new_kepler(converter),
        resolve_collision_code = new_smalln(),
        interaction_over_code = None,
        G = constants.G
    )
    multiples_code = encounters.Multiples(
        gravity_code = code,
        handle_encounter_code = encounter_code,
        G = constants.G
    )


    multiples_code.particles.add_particles(single_stars)
    multiples_code.singles_in_binaries.add_particles(singles_in_binaries)
    multiples_code.binaries.add_particles(binary_stars)
    multiples_code.commit_particles()

    # if rot:
    # OmegaVector = [0., 0., 1e-14]|units.s**-1
    # add_spin(multiples_code.particles, OmegaVector)

    multiples_code.particles.scale_to_standard(convert_nbody=converter,
                                    virial_ratio=VR,
                                smoothing_length_squared=code.parameters.epsilon_squared)

    # multiples_code.neighbor_perturbation_limit = 0.05
    # multiples_code.global_debug = 3
    # multiples_code.neighbor_veto = True
    # multiples_code.final_scale_factor = 1.01
    #    multiples_code.global_debug = 3
    multiples_code.handle_encounter_code.parameters.neighbours_factor = NF
    multiples_code.handle_encounter_code.parameters.hard_binary_factor = HBF
    multiples_code.handle_encounter_code.parameters.scatter_factor = SF    



    multiples_code.evolve_model(1.0|units.yr)

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
                       attributes=["mass","x","y","z","vx","vy","vz"],
                       target_names=["mass","x","y","z","vx","vy","vz"])
    channel_DYN_to_stars.copy()

    channel_DYN_to_bins = multiples_code.particles.new_channel_to(binary_stars,
                       attributes=["x","y","z","vx","vy","vz"],
                       target_names=["x","y","z","vx","vy","vz"])
    channel_DYN_to_bins.copy()

    channel_DYN_to_bstars = multiples_code.singles_in_binaries.new_channel_to(singles_in_binaries,
                       attributes=["mass","x","y","z","vx","vy","vz"],
                       target_names=["mass","x","y","z","vx","vy","vz"])
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
    write_set_to_file(individual_stars.savepoint(time),sim_dir+"single_stars.hdf5","hdf5")
    write_set_to_file(singles_in_binaries.savepoint(time), sim_dir+"binary_singles.hdf5","hdf5")
    write_set_to_file(binary_stars.savepoint(time),sim_dir+"binary_stars.hdf5","hdf5")
    print tend, Numsteps
    sim_dt = tend.value_in(tend.unit) / np.float(nsteps)
    print sim_dt

    snapshot_timesteps = np.arange(sim_dt,tend.value_in(tend.unit)+sim_dt, sim_dt).astype("float")
    print snapshot_timesteps

    sim_dt = tend.value_in(tend.unit)/ np.float(nsteps*2.)

    print "--------------------------------------------"
    print "Starting the simulation now..."
    print "--------------------------------------------"

    sim_start = datetime.datetime.now()
    sim_start_delta = (code2 - sim_start).total_seconds()
    write_csv([0, sim_start_delta, 0.0, 0.0], sim_dir+filename)
    ii=1
    while time < tend:
        loop_time1 = datetime.datetime.now()
        index = str(x)
        index = index.zfill(6)

        se_timestep = stellar.particles.time_step.min()
        dt_min = min(sim_dt|tend.unit, se_timestep)
        print "Time steps:",sim_dt, se_timestep.in_(units.Myr) 

        time += dt_min

        E0 = multiples_code.get_total_energy()
        print "Total Energy of System:      ",E0 #print_diagnostics(multiples_code)
        multiples_code.evolve_model(time)
        ER1 = multiples_code.get_total_energy()
        relerr1 = (ER1 - E0)/E0
        print "Relative Dynamics Error:     ",relerr1
        # time = multiples_code.model_time

        if stopping_condition.is_set():
            resolve_changed_binaries(stopping_condition, stellar, multiples_code, converter)

        stellar.evolve_model(time)
        ER2 = multiples_code.get_total_energy()
        relerr2 = (ER2 - ER1)/ER1
        print "Relative Stellar Evol Error: ",relerr2

        resolve_supernova(supernova_detection, multiples_code, time)
        
        update_dynamical_binaries_from_stellar(stellar, multiples_code,
                                               converter)
        ER3 = multiples_code.get_total_energy()
        relerr3 = (ER3 - ER2)/ER2
        print "Book-keeping Error:          ",relerr3

        stellar_mergers = len(stellar.binaries[stellar.binaries.semi_major_axis.in_(units.m)==0.0|units.m])

        # number_mergers = resolve_binary_dynamics(converter, stellar, multiples_code, 
        #                     individual_stars, singles_in_binaries, binary_stars, 
        #                     dt_min, stopping_condition)
        print "at t=", multiples_code.model_time, \
              "Nmultiples:", len(stellar.binaries) - stellar_mergers

        # collision_detection = len(stellar.binaries[stellar.binaries.semi_major_axis.in_(units.m)==0.0|units.m])
        # if collision_detection > 0:

        # ER = print_diagnostics(multiples_code, E0)
        particle_dataset_bookkeeping(stellar, multiples_code,
                                    individual_stars,
                                    singles_in_binaries,
                                    binary_stars)

        if len(stellar.particles) > 50:
            DensCtr,CoreR,CoreDens = multiples_code.all_singles.densitycentre_coreradius_coredens(converter)
            # DensCtr = multiples_code.particles.center_of_mass()
            print("Core Radius [pc]:          ", CoreR.value_in(units.parsec))
            print("Core Density [Msun/pc**3]: ", CoreDens.value_in(units.MSun/units.parsec**3))
        else:
            DensCtr = multiples_code.particles.center_of_mass()
            CoreR   = 0|units.parsec

        channel_SE_to_DYN.copy()
        channel_DYN_to_stars.copy()
        channel_DYN_to_bstars.copy()
        channel_DYN_to_bins.copy()

        RelErr = relerr1+relerr2+relerr3 #(ER - E0) / E0
        print "Total Simulation Error:     ",RelErr
        print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        ii+=1
        # if x==5: break

        channel_SE_to_stars.copy()
        channel_SE_to_bins.copy()

        if len(snapshot_timesteps)==0: break
        epsilon = abs(time.value_in(tend.unit)-snapshot_timesteps[0]) 

        x += 1
         
         
        epsilon_check = epsilon < 0.10

        if epsilon_check:
           print "wrote out files at time: ", np.round(time.value_in(tend.unit),2), time.value_in(tend.unit)
           write_set_to_file(individual_stars.savepoint(time),sim_dir+"single_stars.hdf5","hdf5")
           write_set_to_file(singles_in_binaries.savepoint(time), sim_dir+"binary_singles.hdf5","hdf5")
           write_set_to_file(binary_stars.savepoint(time),sim_dir+"binary_stars.hdf5","hdf5")
           print(len(individual_stars),len(singles_in_binaries),len(binary_stars))
           simple_spatial_plot_module(multiples_code.all_singles, singles_in_binaries, binary_stars, IMSIZE,time, x, image_dir,com=DensCtr,core=CoreR)
           snapshot_timesteps = np.delete(snapshot_timesteps, 0)
           print snapshot_timesteps

        loop_time2 = datetime.datetime.now()
        loop_delta = (loop_time2 - loop_time1).total_seconds()
        write_csv([ii, loop_delta, RelErr, time.value_in(units.Myr)], sim_dir+filename)
        energy_conv_plot(FILENAME=sim_dir+filename+'.csv',DIR=sim_dir)
        print "Timestep Computation Time: ", loop_delta," seconds"
        print "t, Energy=", time, multiples_code.get_total_energy()

    sim_end = datetime.datetime.now()
    comp_time = sim_end - sim_start
    print "Total computation time", comp_time.total_seconds()
    simple_2d_movie_maker("output_movie_evolution",fps=45, img_dir=image_dir, output_dir=sim_dir)


def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()
    result.add_option("-N", dest="N",type="int", default=100)
    result.add_option("--Binfrac", dest="Binfrac",type="float", default=0.5)
    result.add_option("--HMR", unit=units.parsec,dest="HMR",type="float", default=1|units.parsec)
    result.add_option("--Virialratio", dest="VR", type="float", default=0.5)
    result.add_option("--Tend",dest="Tend",type="float", default=10.)
    result.add_option("--TimeUnit", dest="TimeUnit",default='myr')
    result.add_option("--Numsteps", dest="Numsteps",type="int", default=10)
    result.add_option("--Mdist", dest="Mdist", default="K")
    result.add_option("--Sdist", dest="Sdist", default="P")
    result.add_option("--Mmin", dest="Mmin", type="float", default=0.5)
    result.add_option("--Mmax", dest="Mmax", type="float", default=100.)
    result.add_option("--Filename", dest="filename", default="SC")
    result.add_option("--Rotation", dest='rot', default=False)
    result.add_option("--Seed", dest="seed",type="int", default=2501)
    result.add_option("--Npro", dest="nproc",type="int", default=1)
    result.add_option("--Smalln_pro",dest="smalln_pro",type="int",default=1)
    result.add_option("--Dtparam",dest='DTPARAM',type='float',default=0.05)
    result.add_option("--NF", dest="NF", type="float", default=1.0)
    result.add_option("--HBF", dest="HBF", type="float", default=3.0)
    result.add_option("--SF", dest="SF",type="float", default=10.0)
    result.add_option("--Run_gpu",dest="RUN_GPU",type='int',default=0)
    result.add_option("--Imsize",dest='IMSIZE',type='float',default=5)
    result.add_option("--Dir",dest="DIR",default=None)
    result.add_option("--Machine",dest='MACHINE', default='linux')
    return result



if __name__ == "__main__":
    opt, arguments  = new_option_parser().parse_args()
    main(**opt.__dict__)


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






