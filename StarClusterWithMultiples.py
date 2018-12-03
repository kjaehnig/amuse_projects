import numpy
import time
import os
from amuse.units import nbody_system,units
from amuse.ic.plummer import new_plummer_model
from amuse.community.ph4.interface import ph4
from amuse.community.seba.interface import SeBa
from amuse.community.smalln.interface import SmallN
from amuse.community.kepler.interface import Kepler
from amuse.couple import multiples
from amuse.ic.salpeter import new_salpeter_mass_distribution   
from amuse.units import constants
from support_functions import MakeMeANewCluster,MakeMeANewClusterWithBinaries,binary_reference_frame
from amuse.io import write_set_to_file, read_set_from_file
from amuse.support.console import set_printing_strategy
from amuse.datamodel.particles import Particles
from support_functions import spatial_plot_module
from support_functions import simple_2d_movie_maker
# Awkward syntax here because multiples needs a function that resets
# and returns a small-N integrator.
set_printing_strategy("custom", preferred_units=[units.MSun, units.parsec, units.Myr],
                      precision=9, prefix="", separator="[", suffix="]")


DIR = "/home/l_reclusa/Desktop/GRAD/SC/simulation_data/"
SMALLN = None
def init_smalln(converter):
    global SMALLN
    SMALLN = SmallN(convert_nbody=converter)

def new_smalln():
    SMALLN.reset()
    return SMALLN

def stop_smalln():
    global SMALLN
    SMALLN.stop()

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

def resolve_supernova(supernova_detection, bodies, time):
    if supernova_detection.is_set():
        print "At time=", time.in_(units.Myr), "supernova detected=", len(supernova_detection.particles(0))
        Nsn = 0
        for ci in range(len(supernova_detection.particles(0))):
            print supernova_detection.particles(0)
            particles_in_supernova = Particles(particles=supernova_detection.particles(0))
            natal_kick_x = particles_in_supernova.natal_kick_x
            natal_kick_y = particles_in_supernova.natal_kick_y
            natal_kick_z = particles_in_supernova.natal_kick_z

            particles_in_supernova = particles_in_supernova.get_intersecting_subset_in(bodies)
            particles_in_supernova.vx += natal_kick_x
            particles_in_supernova.vy += natal_kick_y
            particles_in_supernova.vz += natal_kick_z

            Nsn+=1
        print "Resolve supernova Number:", Nsn
        

def resolve_collision(collision_detection, gravity, stellar, bodies):
    if collision_detection.is_set():
        E_coll = gravity.kinetic_energy + gravity.potential_energy
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
            dE_coll = E_coll - (gravity.kinetic_energy + gravity.potential_energy)
        print "Energy error in the collision: dE =", dE_coll 


def print_diagnostics(grav, E0=None):

    # Simple diagnostics.

    ke = grav.kinetic_energy
    pe = grav.potential_energy
    Nmul, Nbin, Emul = grav.get_total_multiple_energy()
    print ''
    print 'Time =', grav.get_time()
    print '    top-level kinetic energy =', ke
    print '    top-level potential energy =', pe
    print '    total top-level energy =', ke + pe
    print '   ', Nmul, 'multiples,', 'total energy =', Emul
    E = ke + pe + Emul
    print '    uncorrected total energy =', E
    
    # Apply known corrections.
    
    Etid = grav.multiples_external_tidal_correction \
            + grav.multiples_internal_tidal_correction  # tidal error
    Eerr = grav.multiples_integration_energy_error  # integration error

    E -= Etid + Eerr
    print '    corrected total energy =', E

    if E0 is not None: print '    relative energy error=', (E-E0)/E0
    
    return E

def integrate_system(
                    N=100,
                    Binfrac=0.5,
                    HMR=1.,
                    VR=0.5,
                    Tend=10,
                    Numsteps=10,
                    Mdist='K',
                    Sdist='P',
                    Mmin=.5,
                    Mmax=100.,
                    filename='cluster1',
                    seed=2501
                    ):
    # ZAMS = new_salpeter_mass_distribution(N, 0.1|units.MSun, 100|units.MSun)
    # converter = nbody_system.nbody_to_si(ZAMS.sum(), .1|units.parsec)
    file_dir_check = os.path.isdir(DIR+str(filename))
    if file_dir_check == False:
        os.makedirs(DIR+str(filename))
    sim_dir = DIR+str(filename)+"/"

    print "Made image directory"
    image_dir = sim_dir+"images/"
    os.makedirs(image_dir)

    stars,bstars,sbstars,converter =\
                 MakeMeANewClusterWithBinaries(N=N, 
                    HalfMassRadius=0.001*HMR,
                    kind=Sdist, frac_dim=1.6, mmin=Mmin,
                     mmax=Mmax, SEED=seed, bin_frac=Binfrac)

    gravity = ph4(converter, mode='gpu', redirection='none')
    gravity.initialize_code()
    gravity.parameters.set_defaults()
    gravity.parameters.epsilon_squared = 0.1 | units.RSun**2
    if seed is not None: numpy.random.seed(seed)
    # stars = new_plummer_model(N, converter)
    # stars.mass = ZAMS
    
    translated_bstars, contact_status = binary_reference_frame(bstars, sbstars)

    all_singles = Particles(particles=[stars, translated_bstars])

    all_singles.move_to_center()
    all_singles.scale_to_standard(convert_nbody=converter,
                             virial_ratio = 0.40)

    gravity.particles.add_particles(all_singles)

    stopping_condition = gravity.stopping_conditions.collision_detection
    stopping_condition.enable()

    # stellar = SeBa()
    # stellar.parameters.metallicity = 0.02
    # stellar.parameters.supernova_kick_velocity = 300 | units.km / units.s
    # stellar.particles.add_particles(all_singles)

    # sn_detection = stellar.stopping_conditions.supernova_detection
    # sn_detection.enable()

    id = numpy.arange(N)
    all_singles.id = id+1

    # Set dynamical radii for encounters.

    all_singles.radius = all_singles.mass.number ** 0.8 | units.RSun

    init_smalln(converter)
    kep = Kepler(unit_converter=converter)
    kep.initialize_code()
    multiples_code = multiples.Multiples(gravity, 
                                        new_smalln, 
                                        kep, 
                                        gravity_constant=constants.G)
    multiples_code.neighbor_perturbation_limit = 0.001
    multiples_code.neighbor_veto = False
    multiples_code.global_debug = 3

    to_stars = multiples_code.particles.new_channel_to(all_singles,
                attributes=["mass","radius","x","y","z","vx","vy","vz","potential_in_code"],
                target_names=["mass","radius","x","y","z","vx","vy","vz","potential_in_code"])

    #   global_debug = 0: no output from multiples
    #                  1: minimal output
    #                  2: debugging output
    #                  3: even more output

    print ''
    print 'multiples_code.neighbor_veto =', \
        multiples_code.neighbor_veto
    print 'multiples_code.neighbor_perturbation_limit =', \
        multiples_code.neighbor_perturbation_limit
    print 'multiples_code.retain_binary_apocenter =', \
        multiples_code.retain_binary_apocenter
    print 'multiples_code.wide_perturbation_limit =', \
        multiples_code.wide_perturbation_limit

    # Advance the system.
    dt = Tend/numpy.float(Numsteps)
    T = 0.0|units.Myr
    start_time = time.time()
    i=0
    print all_singles.LagrangianRadii(cm=all_singles.center_of_mass(), mf=[0.5])[0].value_in(units.parsec)
    while T < Tend:
        T += dt
        E0 = print_diagnostics(multiples_code)
        multiples_code.evolve_model(T)

        # multiples_code.particles.synchronize_to(stars)
        to_stars.copy()
        write_set_to_file(all_singles.savepoint(T), sim_dir+filename+".hdf5", 'hdf5')
        spatial_plot_module(multiples_code.particles,cluster_length=3, time=T, x=i, direc=image_dir)
        print_diagnostics(multiples_code, E0)
        i+=1


    gravity.stop()
    kep.stop()
    stop_smalln()

    comptime = time.time() - end_time
    print 'SIMULATION FINISHED, computation time: ', comptime
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

    return result
if __name__ in ('__main__'):
    opt, arguments  = new_option_parser().parse_args()
    integrate_system(**opt.__dict__)
