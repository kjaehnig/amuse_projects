from amuse.ext.plummer import new_plummer_model
from amuse.ext.kingmodel import new_king_model
from amuse.community.fractalcluster.interface import new_fractal_cluster_model
from amuse.units import units
from amuse.ic.salpeter import new_salpeter_mass_distribution
from amuse.ic.brokenimf import new_kroupa_mass_distribution
from amuse.datamodel.rotation import add_spin
import csv
import numpy as np
from amuse.support.console import set_printing_strategy
import pandas as pd
from amuse.units.nbody_system import nbody_to_si



set_printing_strategy("custom", 
                  preferred_units = [units.MSun, units.parsec, units.Myr], 
                  precision = 4, prefix = "", 
                  separator = " [", suffix = "]")




def virial_from_hmr(HMR):
   RVIR = (16./(3.*np.pi))*(HMR/1.3)
   return RVIR

def random_ic_parameters(model='plummer'):
    N_range = [10, 1e5]
    HMR_range = [0.5, 100]  #### PARSECS
    Virial_state = [0.25, 1.0] 
    Frac_dim = [1.6, 3.0]
    W0_parameter = [0, 16]
    Spin_kick = [1e-15,1e-5 ] ##### S ** -1 
    Add_spin_kick = np.random.choice([True,False])

    Spin_flag = None

    output_dict = dict([])

    if model.lower()=='plummer':
        Num = np.random.randint(N_range[0],N_range[1],size=1)[0]
        output_dict['Model'] = "P"
        output_dict["W0_FD"] = []
        output_dict['N'] = Num
        Mzams = new_salpeter_mass_distribution(Num, 0.08|units.Msun, 100|units.MSun)
        Mass = Mzams.sum()
        output_dict["M"] = Mass.value_in(units.MSun)
        HMR = np.random.randint(HMR_range[0],HMR_range[1],size=1)[0]
        output_dict['HMR'] = HMR
        RVIR = virial_from_hmr(HMR)
        Virial = np.random.uniform(Virial_state[0],Virial_state[1],size=1)[0]
        output_dict['Q'] = Virial
        converter = nbody_to_si(Mass, RVIR|units.parsec, 1.0|units.kms)

        stars = new_plummer_model(Num, convert_nbody = converter)
        stars.mass = Mzams

        if Add_spin_kick:
            Spin = np.random.uniform(Spin_kick[0], Spin_kick[1],size=3)
            Omega_vector = [Spin[0], Spin[1], Spin[2]] | units.s ** -1
            add_spin(stars, Omega_vector)
            Spin_flag = 1
            omegax, omegay, omegaz = Spin[0], Spin[1], Spin[2]

        if not Add_spin_kick: 
            Spin_flag = 0
            omegax, omegay, omegaz = 0,0,0
        output_dict['Sflag'] = Spin_flag
        output_dict['omegax'] = omegax
        output_dict['omegay'] = omegay
        output_dict['omegaz'] = omegaz

        stars.scale_to_standard(convert_nbody=converter,
                                virial_ratio = Virial)

        output_df = pd.DataFrame(output_dict, columns=output_dict.keys())
    
    return output_df 






