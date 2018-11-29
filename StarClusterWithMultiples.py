import numpy
import time
from amuse.units import nbody_system,units
from amuse.ic.plummer import new_plummer_model
from amuse.community.ph4.interface import ph4
from amuse.community.smalln.interface import SmallN
from amuse.community.kepler.interface import Kepler
from amuse.couple import multiples
from amuse.ic.salpeter import new_salpeter_mass_distribution   
from amuse.units import constants
from support_functions import MakeMeANewCluster,MakeMeANewClusterWithBinaries,binary_reference_frame
from amuse.io import write_set_to_file, read_set_from_file
from amuse.support.console import set_printing_strategy
from amuse.datamodel.particles import Particles

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

def integrate_system(N, t_end, seed=None):
    # ZAMS = new_salpeter_mass_distribution(N, 0.1|units.MSun, 100|units.MSun)
    # converter = nbody_system.nbody_to_si(ZAMS.sum(), .1|units.parsec)

    stars,bstars,sbstars,converter =\
                 MakeMeANewClusterWithBinaries(N=N, 
                    HalfMassRadius=0.1|units.parsec,
                    kind='f', frac_dim=2.0, mmin=0.5,
                     mmax=120, SEED=2501, bin_frac=0.5)

    gravity = ph4(converter, mode='gpu', redirection='none')
    gravity.initialize_code()
    gravity.parameters.set_defaults()
    gravity.parameters.epsilon_squared = 0 | units.AU**2
    if seed is not None: numpy.random.seed(seed)
    # stars = new_plummer_model(N, converter)
    # stars.mass = ZAMS
    
    translated_bstars, contact_status = binary_reference_frame(bstars, sbstars)

    all_singles = Particles(particles=[stars, translated_bstars])

    all_singles.move_to_center()
    all_singles.scale_to_standard(convert_nbody=converter,smoothing_length_squared
                             = gravity.parameters.epsilon_squared,
                             virial_ratio = 0.25)

    # id = numpy.arange(N)
    all_singles.id = all_singles.key

    # Set dynamical radii for encounters.

    all_singles.radius = all_singles.mass.number ** 0.8 | units.RSun
    gravity.particles.add_particles(all_singles)

    stopping_condition = gravity.stopping_conditions.collision_detection
    stopping_condition.enable()

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
                attributes=["mass","radius","x","y","z","vx","vy","vz"],
                target_names=["mass","radius","x","y","z","vx","vy","vz"])

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
    dt = 0.1|units.Myr
    T = 0.0|units.Myr
    t0 = time.time()

    while T < t_end:
        T+=dt
        print T
        E0 = print_diagnostics(multiples_code)
        multiples_code.evolve_model(T)
        # multiples_code.particles.synchronize_to(stars)
        to_stars.copy()
        write_set_to_file(all_singles.savepoint(T), DIR+'multiples_star_cluster_BIG.hdf5', 'hdf5')

        print_diagnostics(multiples_code, E0)

    gravity.stop()
    kep.stop()
    stop_smalln()

    comptime = time.time() - t0
    print 'SIMULATION FINISHED, computation time: ', comptime
if __name__ in ('__main__'):
    N = 5000
    t_end = 100.0 | units.Myr
    integrate_system(N, t_end) #, 42)
