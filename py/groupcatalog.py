import numpy as np
import pandas as pd
import astropy.coordinates as coord
import astropy.units as u
import os
import sys
import emcee
import pickle
from astropy.io import ascii
import subprocess as sp
from astropy.table import Table
import astropy.io.fits as fits
import astropy.units as u
import copy
import sys
import math
import struct
from io import BufferedReader
from multiprocessing import Pool
from joblib import Parallel, delayed
import signal 

if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from pyutils import *
from redshift_guesser import SimpleRedshiftGuesser, PhotometricRedshiftGuesser
from hdf5_to_dat import pre_process_mxxl
from uchuu_to_dat import pre_process_uchuu
from bgs_helpers import *
import wp
from calibrationdata import *

# Must keep this protocol syncronized with the C++ code in groups.hpp
MSG_REQUEST = 0
MSG_FSAT = 1
MSG_LHMR = 2
MSG_LSAT = 3
MSG_HOD = 4
MSG_HODFIT = 5
MSG_COMPLETED = 6
MSG_ABORTED = 7
TYPE_FLOAT = 0
TYPE_DOUBLE = 1

# Sentinal value for no truth redshift
NO_TRUTH_Z = -99.99

# Shared bins for various purposes
Mhalo_bins = np.logspace(9, 15.4, 155-90)
Mhalo_labels = Mhalo_bins[0:len(Mhalo_bins)-1] 

Lgal_bins_for_lsat = np.linspace(8.8, 10.7, 20)

mstar_bins = np.logspace(6, 13, 30)
mstar_labels = mstar_bins[0:len(mstar_bins)-1]

Mr_gal_bins = log_solar_L_to_abs_mag_r(np.log10(L_gal_bins))
Mr_gal_labels = log_solar_L_to_abs_mag_r(np.log10(L_gal_labels))

# This is for the broken direction clustering measurements on the group catalog, not the mock ones.
CLUSTERING_MAG_BINS = [-16, -17, -18, -19, -20, -21, -22, -23]

GF_PROPS_BGS_VANILLA = {
    'zmin':0, 
    'zmax':0,
    'frac_area':0, # should be filled in
    'fluxlim':2,
    'color':1,
}

# ω_L_sf, σ_sf, ω_L_q, σ_q, ω_0_sf, ω_0_q, β_0q, β_Lq, β_0sf, β_Lsf
GOOD_TEN_PARAMETERS = np.array([
    [19.737870,6.394322,24.657218,11.571343,33.116540,9.403598,-3.755194,16.879988,9.941906,0.958446],
    [13.1,2.42,12.9,4.84,17.4,2.67,-0.92,10.25,12.993,-8.04],
    [20.266485,6.981479,20.417991,9.467296,30.750261,6.597792,-1.896981,16.674611,10.527099,1.341537],
    [16.678, 4.460, 19.821, 8.312, 26.087, 6.361, -2.149, 15.190, 12.316, -2.440,],
    [16.703, 4.449, 20.839, 9.096, 28.473, 7.129, -2.792, 16.698, 12.878, -1.552,],
    [16.085, 4.231, 17.800, 7.393, 24.378, 5.448, -2.335, 13.921, 12.962, -2.611,],
    [17.244, 4.737, 20.855, 9.040, 29.335, 6.746, -2.556, 17.334, 12.444, -2.083,],
    [16.996, 4.719, 20.392, 9.025, 28.510, 7.018, -3.166, 16.200, 13.577, -0.381,],
    [16.546, 4.449, 18.628, 7.764, 25.308, 5.657, -2.405, 13.964, 12.704, -2.656,],
    [16.282, 4.222, 19.610, 8.317, 25.916, 6.212, -2.352, 15.420, 12.629, -2.366,],
    [18.164, 5.378, 22.283, 9.641, 29.212, 7.854, -2.662, 14.937, 11.956, -0.798,],
    [16.479, 4.100, 17.130, 6.873, 27.722, 2.841, -4.811, 20.308, 13.020, -2.867,],
    [16.434, 4.338, 17.842, 7.859, 26.438, 3.922, -3.719, 17.966, 13.318, -2.801,],
    [19.603, 6.398, 22.167, 10.035, 29.776, 7.459, -2.735, 15.314, 11.259, -0.649,],
    [16.851, 4.626, 20.415, 8.839, 27.402, 6.785, -2.461, 15.311, 11.867, -1.280,],
    [17.482, 5.155, 21.416, 10.300, 27.547, 7.046, -2.988, 15.631, 12.334, -2.553,],
    [16.851, 4.647, 19.172, 8.232, 26.273, 5.859, -2.518, 15.238, 12.262, -2.230,],
    [16.812, 4.511, 20.105, 8.392, 26.276, 6.416, -2.102, 15.491, 12.314, -2.457,],
    [16.682, 4.516, 19.678, 8.421, 28.002, 6.950, -2.308, 15.839, 13.020, -0.616,],
    ])

# A 10 Parameter set found from MCMC SV3 with SDSS data.
GF_PROPS_BGS_COLORS_C1 = {
    'zmin':0, 
    'zmax':0,
    'frac_area':0, # should be filled in
    'fluxlim':2,
    'color':1,
    'omegaL_sf':GOOD_TEN_PARAMETERS[0][0],
    'sigma_sf':GOOD_TEN_PARAMETERS[0][1],
    'omegaL_q':GOOD_TEN_PARAMETERS[0][2],
    'sigma_q':GOOD_TEN_PARAMETERS[0][3],
    'omega0_sf':GOOD_TEN_PARAMETERS[0][4],
    'omega0_q':GOOD_TEN_PARAMETERS[0][5],
    'beta0q':GOOD_TEN_PARAMETERS[0][6],
    'betaLq':GOOD_TEN_PARAMETERS[0][7],
    'beta0sf':GOOD_TEN_PARAMETERS[0][8],
    'betaLsf':GOOD_TEN_PARAMETERS[0][9]
}

# Weird other good one
#({'omegaL_sf': 13.03086801, 'sigma_sf': 1.85056851, 'omegaL_q': 8.92398122, 'sigma_q': -0.27906515, 'omega0_sf': 15.34144908, 'omega0_q': -1.27133105, 'beta0q': -0.59290738, 'betaLq': 14.81630656, 'beta0sf': 12.17260624, 'betaLsf': -6.73894304})

MASTER_PROCESS_LIST = [] # List of all processes that should be killed if we die unexpectedly
def cleanup():
    for p in MASTER_PROCESS_LIST:
        if p is sp.Popen:
            if p.poll() is None:
                p.terminate()
                p.wait()   
def signal_handler(signum, frame):
    cleanup()             
    sys.exit(1)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
def exception_handler(exc_type, exc_value, exc_traceback):
    cleanup()
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
    sys.exit(1)
sys.excepthook = exception_handler


class GroupCatalog:

    def __init__(self, name):
        self.name = name
        folder_name = name.upper().replace(" ", "_").replace("<", "").replace(">", "") + "/"
        self.set_output_folder(OUTPUT_FOLDER + folder_name)  
        self.color = 'k' # plotting color; nothing to do with galaxies
        self.marker = '-'
        self.preprocess_file: str = None
        self.mag_cut: float = None
        self.GF_props = {} # Properties that are sent as command-line arguments to the group finder executable
        self.extra_params = None # Tuple of parameters values
        self.sampler: emcee.EnsembleSampler = None
        self.proc : sp.Popen = None
        self.pipereader : BufferedReader = None
        self.outstream = None

        self.has_truth = False
        self.Mhalo_bins = Mhalo_bins
        self.labels = Mhalo_labels
        self.Mhalo_labels = Mhalo_labels
        self.all_data: pd.DataFrame = None
        self.centrals: pd.DataFrame = None
        self.sats: pd.DataFrame = None
        self.L_gal_bins = L_gal_bins
        self.L_gal_labels = L_gal_labels
        self.Mr_gal_bins = Mr_gal_bins
        self.Mr_gal_labels = Mr_gal_labels

        # Properties pertaining the popmock option
        self.caldata: CalibrationData = None
        self.mockfile = MOCK_FILE_FOR_POPMOCK
        self.mocksize = 250.0 # Mpc/h

        # Geneated from popmock option in group finder
        self.lsat_r = None 
        self.lsat_b = None 

        # Generated from run_corrfunc; holds the wp measurements on the mock populated with this group catalog's HOD.
        self.wp_mock = {}

        # These are direct wp measurements, not on the mock. This code doesn't work BUG
        self.wp_all = None # (rbins, wp_all, wp_r, wp_b)
        self.wp_all_extra = None # (rbins, wp_all, wp_r, wp_b)
        self.wp_slices = np.array(len(CLUSTERING_MAG_BINS) * [None]) # Tuple of (rbins, wp) at each index
        self.wp_slices_extra = np.array(len(CLUSTERING_MAG_BINS) * [None]) # Tuple of (rbins, wp) at each index

        self.f_sat = None # per Lgal bin 
        self.Lgal_counts = None # size of Lgal bins 

        # Given from GF process via monitor_pipe
        self.fsat : np.ndarray = None
        self.fsatr : np.ndarray = None
        self.fsatb : np.ndarray = None
        self.lhmr_m : np.ndarray = None # lhmr model
        self.lhmr_std : np.ndarray = None # lhmr model scatter
        self.lhmr_r_m : np.ndarray = None # lhmr model red
        self.lhmr_r_std : np.ndarray = None # lhmr model red scatter
        self.lhmr_b_m : np.ndarray = None # lhmr model blue
        self.lhmr_b_std : np.ndarray = None # lhmr model blue scatter
        self.lsat_ratios : np.ndarray = None # lsat ratios, 107-88+1 values
        self.hod : np.ndarray = None # Raw HOD
        self.hodfit : np.ndarray = None # HOD with modifications; this is what was used to populate mocks

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ['proc', 'pipereader', 'outstream']:
            if key in state:
                del state[key]
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.proc = None
        self.pipereader = None
        self.outstream = None

    def set_output_folder(self, folder):
        self.output_folder = folder
        self.file_pattern = self.output_folder + self.name
        self.GF_outfile = self.file_pattern + ".out"
        self.results_file = self.file_pattern + ".pickle"

    def get_mock_wp(self, mag: int, color: str, wp_err):
        """
        Get the wp and estimate of wp error from the mock populated with this group catalog's HOD. 
        Returns the wp and the error, which is an ad-hoc construction.

        :param mag: mass bin without the - sign. e.g. 17
        :param color: 'red' or 'blue' or 'all'
        :param wp_err: the wp error from the matching data, which is used in creating the error for the mock
        :return: wp_mock and wp_mock_error
        """
        mag = abs(mag)
        idx = self.caldata.mag_to_idx(mag)

        wp_mock = self.wp_mock[(color, mag)][:,4] # the wp values
        
        # Idea is to take the error bars from the data and use them on the mock
        # But since the volumes are different, we need to scale them.
        vfac = (self.caldata.volumes[idx]/250.0**3)**.5 # factor by which to multiply errors
        
        # Add in an additional error term that is a fraction of the wp value itself as well. 
        # This is to account for the fact that the mock is not a perfect representation of the data.
        # and there are likely systematic errors in the mock itself.
        efac = 0.1 
        wp_mock_error = vfac*wp_err + efac*wp_mock
        
        return wp_mock, wp_mock_error


    def setup_GF_mcmc(self, mcmc_num: int = None):
        print("Setting up GF for MCMC...")

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # If output folder is already an mcmc one, strip it off
        last_folder = self.output_folder.split('/')[-2]
        if last_folder.startswith('mcmc_'):
            self.set_output_folder(self.output_folder.replace(last_folder + '/', ''))

        # Change self.output_folder to be a new subfolder called mcmc_{mcmc_num}. If None, set to the next integer.
        if mcmc_num is None:
            mcmc_num = len([name for name in os.listdir(self.output_folder) if os.path.isdir(os.path.join(self.output_folder, name))])
        self.set_output_folder(self.output_folder + f"mcmc_{mcmc_num}/")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        if 'earlyexit' not in self.GF_props:
            self.GF_props['earlyexit'] = 1

        # Setup emcee backend
        backfile = self.output_folder + "gf_mcmc.h5"
        if os.path.isfile(backfile):
            print(f'BACKEND ALREADY EXISTS: {backfile}')
            backend = emcee.backends.HDFBackend(backfile)
            nwalkers = backend.get_chain().shape[1]
            ndim = backend.get_chain().shape[2]
        else:
            print(f'CREATING BACKEND: {backfile}')
            backend = emcee.backends.HDFBackend(backfile)
            nwalkers = 25
            ndim = 10

        self.sampler = emcee.EnsembleSampler(
            nwalkers, 
            ndim, 
            self.lnprob,
            backend=backend
        )

    def run_GF_mcmc(self, niter: int):

        if self.sampler is None:
            print("Warning: run_GF_mcmc() called without sampler set. Call setup_GF_mcmc() first.")
            return

        # Run the group finder in interactive mode
        print("Calling group finder with interactive on for inital run.")
        self.run_group_finder(popmock=True, profile=False, interactive=True, silent=True)
        
        # If there is a state, continue from there
        if self.sampler.backend.iteration > 0:
            print(f"Resuming from iteration {self.sampler.backend.iteration}")
            pos, prob, state = self.sampler.run_mcmc(None, niter, progress=True)
        else:
            print("Starting fresh")
            # Start fresh using IC's centered around known good values
            if self.sampler.ndim == 10:
                # TODO improve the starting random walker locations
                initial = GOOD_TEN_PARAMETERS[3]
            elif self.sampler.ndim == 14:
                initial = np.array([1.312225e+01, 2.425592e+00, 1.291072e+01, 4.857720e+00, 1.745350e+01, 2.670356e+00, -9.231342e-01, 1.028550e+01, 1.301696e+01, -8.029334e+00, 2.689616e+00, 1.102281e+00, 2.231206e+00, 4.823592e-01])
            
            spread_factor = 0.5
            p0 = initial + spread_factor * (np.random.randn(self.sampler.nwalkers, self.sampler.ndim) - 0.5)
            
            # Set a walker to each set of saved off parameters
            for i in range(len(GOOD_TEN_PARAMETERS)):
                if i < self.sampler.nwalkers:
                    p0[i] = GOOD_TEN_PARAMETERS[i]

            print(np.shape(p0))

            pos, prob, state = self.sampler.run_mcmc(p0, niter, progress=True)

        # Anything else to state before saving off?
        self.dump()      

    # --- log-likelihood
    def lnlike(self, theta):
        chi = self.run_and_calc_chisqr(theta)
        # Side effect of running is that some properties were set via pipe messages
        # Stack all the blob data as one long array
        blobs = np.concatenate((self.fsat, self.fsatr, self.fsatb, self.lhmr_m, self.lhmr_std, self.lhmr_r_m, self.lhmr_r_std, self.lhmr_b_m, self.lhmr_b_std, self.lsat_r, self.lsat_b))
        
        # TODO maybe store HOD too
        #blobs = np.concatenate((self.fsat, self.fsatr, self.fsatb, self.lhmr_m, self.lhmr_std, self.lhmr_r_m, self.lhmr_r_std, self.lhmr_b_m, self.lhmr_b_std, self.lsat_r, self.lsat_b, np.flatten(self.hodfit)))
        
        assert len(blobs) == 3*40 + 65*6 + 40, f"Expected {3*40 + 65*6 + 40} metadata entries, but got {len(blobs)}"
        return -0.5 * chi, blobs

    # --- set the priors (no priors right now)
    def lnprior(self, theta):
        return 0.0

    # -- combine the two above
    def lnprob(self, theta):
        lp = self.lnprior(theta)

        # TODO also return lhmr, lsat as blob metadata
        # Want to avoid a postprocess() call in MCMC loop, so things are calculated in C GF and sent via a pipe to us.
        like = self.lnlike(theta)
        return lp + like[0], like[1]

    def get_backends(self):
        # Read in all subdirs of output_folder that start with mcmc_
        main_output = self.output_folder
        last_folder = self.output_folder.split('/')[-2]
        if last_folder.startswith('mcmc_'):
            main_output = self.output_folder.replace(last_folder + '/', '')
            self.set_output_folder(main_output)
        
        mcmc_folders = [name for name in os.listdir(main_output) if os.path.isdir(os.path.join(main_output, name)) and name.startswith('mcmc_')]
        print(f"Found {len(mcmc_folders)} mcmc/optuna folders")

        backends = []
        # Get the best parameters from each mcmc run and choose the best ones
        for mcmc_folder in mcmc_folders:
            backend = self.get_backend_for_run(int(mcmc_folder.split('_')[1]), read_only=True)
            if backend is None:
                continue
            backends.append(backend)
        return backends, mcmc_folders

    def get_backend_for_run(self, mcmc_num: int, read_only=True) -> emcee.backends.Backend:
        
        main_output = self.output_folder
        last_folder = self.output_folder.split('/')[-2]
        if last_folder.startswith('mcmc_'):
            main_output = self.output_folder.replace(last_folder + '/', '')

        backfile = main_output + f"mcmc_{mcmc_num}/gf_mcmc.h5"
        backend = emcee.backends.HDFBackend(backfile, read_only=read_only)
        return backend
    
    def load_best_params_from_run(self, run):
        backend = self.get_backend_for_run(run, read_only=True)
        if backend is None:
            return None
        if isinstance(backend, emcee.backends.Backend):
            idx = np.argmax(backend.get_log_prob(flat=True))
            return backend.get_chain(flat=True)[idx]
        else: 
            raise Exception("Unknown backend type")


    def load_best_params_across_runs(self, save=False):
        backends, mcmc_folders = self.get_backends()

        # Get the best parameters from each mcmc run and choose the best ones
        best_params_list = []
        for backend, folder in zip(backends, mcmc_folders):
            if isinstance(backend, emcee.backends.Backend):
                idx = np.argmax(backend.get_log_prob(flat=True))
                values = backend.get_chain(flat=True)[idx]
                chisqr = (-2) * backend.get_log_prob(flat=True)[idx]
                print(f"Best chi^2 for {folder} (N={len(backend.get_log_prob(flat=True))}) (emcee): {chisqr}")
                best_params_list.append((chisqr, values))

        best_params_list.sort(key=lambda x: x[0])

        if save:
            with open(os.path.join(self.output_folder, 'best_params_list.pkl'), 'wb') as f:
                pickle.dump(best_params_list, f)

        # Copy the best parameters into the GF_props
        if len(best_params_list) > 0:
            best_params = best_params_list[0][1]
            if len(best_params) != 10 and len(best_params) != 14:
                print(f"Warning: reader has wrong number of parameters. Expected 10 or 14 but got {len(best_params)}")
            self.GF_props = {
                'zmin': self.GF_props['zmin'],
                'zmax': self.GF_props['zmax'],
                'frac_area': self.GF_props['frac_area'],
                'fluxlim': self.GF_props['fluxlim'],
                'color': self.GF_props['color'],
                'omegaL_sf': best_params[0],
                'sigma_sf': best_params[1],
                'omegaL_q': best_params[2],
                'sigma_q': best_params[3],
                'omega0_sf': best_params[4],
                'omega0_q': best_params[5],
                'beta0q': best_params[6],
                'betaLq': best_params[7],
                'beta0sf': best_params[8],
                'betaLsf': best_params[9],
            }
            if len(best_params) == 14:
                self.GF_props['omega_chi_0_sf'] = best_params[10]
                self.GF_props['omega_chi_0_q'] = best_params[11]
                self.GF_props['omega_chi_L_sf'] = best_params[12]
                self.GF_props['omega_chi_L_q'] = best_params[13]

        print(f"Best chi squared: {best_params_list[0][0] if len(best_params_list) > 0 else 'N/A'}")
        if len(best_params_list) > 0:
            return best_params_list[0][1]
        return None

    def sanity_tests(self):
        print(f"Running sanity tests on {self.name}")
        df = self.all_data

        good_ltot = df.loc[:, 'L_TOT'] >= 0.9999*df.loc[:, 'L_GAL']
        if 'TARGETID' in df.columns:
            assert np.all(good_ltot), f"Total luminosity should be greater than galaxy luminosity, but {df.loc[~good_ltot, 'TARGETID'].to_numpy()} are not."
            assert np.all(df['N_SAT'] >= 0), f"Number of satellites should be >= 0, but {df.loc[df['N_SAT'] < 0, 'TARGETID'].to_numpy()} have negative NSAT."
        else:
            assert np.all(good_ltot), f"Total luminosity should be greater than galaxy luminosity, but {np.sum(~good_ltot)} are not."
            assert np.all(df['N_SAT'] >= 0), f"Number of satellites should be >= 0, but {np.sum(df['N_SAT'] < 0)} have negative NSAT."

        sats = df.loc[df['IS_SAT']]
        assert np.all(sats['P_SAT'] > 0.499999), f"Everything marked as a sat should have P_sat > 0.5, but {np.sum(sats['P_SAT'] < 0.5)} do not."
        assert np.all(sats.index != sats['IGRP']), "Satellites should have igrp != index"

        cens = df.loc[~df['IS_SAT']]
        assert np.all(cens['P_SAT'] < 0.500001), f"Everything marked as a central should have P_sat < 0.5, but {np.sum(cens['P_SAT'] > 0.5)} do not."
        assert np.all(cens.index == cens['IGRP']), "Centrals should have igrp == index"

        assert len(cens) == len(df['IGRP'].unique()), f"Counts of centrals should be count of unique groups, but {len(df.loc[~df['IS_SAT']])} != {len(df['IGRP'].unique())}"

        if len(df) > 1000:
            bighalos = cens.loc[cens['Z'] < 0.1].sort_values('M_HALO', ascending=False).head(20)
            assert np.all(bighalos['N_SAT'] > 0), f"Big halos at low z should have satellites, but {np.sum(bighalos['N_SAT'] == 0)} do not."

    def write_sharable_output_file(self, name=None):
        if name is None:
            name = str.replace(self.GF_outfile, ".out", "_Catalog.csv").replace(" ", "_").replace("<", "").replace(">", "")
        elif not name.endswith('.csv'):
            name = name + '.csv'

        fitsname = name.replace(".csv", ".fits")

        columns_to_write = [
            'TARGETID', 
            'RA',
            'DEC',
            'Z',
            'L_GAL', 
            'VMAX',
            'P_SAT', 
            'M_HALO',
            'N_SAT', 
            'L_TOT', 
            'IGRP', 
            'WEIGHT', 
            'APP_MAG_R', 
            'Z_ASSIGNED_FLAG',
            'G_R',
            'IS_SAT', 
            'QUIESCENT', 
            'MSTAR' 
        ]
        for c in columns_to_write.copy():
            if c not in self.all_data.columns:
                print("WARNING - column not found: " + c)
                columns_to_write.remove(c)

        df_to_write = self.all_data.loc[:, columns_to_write]

        #print(f"Writing a sharable output file: {name}")
        #df_to_write.to_csv(name, index=False, header=True)

        print(f"Writing a sharable output file: {fitsname}")
        table = Table.from_pandas(
            df_to_write,
            units={ 
                'RA': u.degree,
                'DEC': u.degree,
                'L_GAL': u.solLum,
                'VMAX': u.Mpc**3,
                'M_HALO': u.solMass,
                'L_TOT': u.solLum,
                'MSTAR': u.solMass
            } # Others are dimensionless
            )
        
        table.write(fitsname, overwrite=True)

        # Add a name to the table, to confrom to DESI VAC standards
        hdul = fits.open(fitsname, memmap=True)
        hdul[1].name = "GALAXIES"
        hdul.writeto(fitsname, overwrite=True)

        return fitsname

    def get_best_wp_all(self):
        if self.wp_all_extra is not None:
            return self.wp_all_extra
        return self.wp_all
    
    def get_best_wp_slices(self):
        if self.wp_slices_extra is not None and np.all(self.wp_slices_extra != None):
            return self.wp_slices_extra
        return self.wp_slices

    def get_completeness(self):
        return spectroscopic_complete_percent(self.all_data['Z_ASSIGNED_FLAG'].to_numpy())

    def get_lostgal_neighbor_used(self):
        arr = self.all_data['Z_ASSIGNED_FLAG'].to_numpy()
        return np.sum(z_flag_is_neighbor(arr)) / np.sum(z_flag_is_not_spectro_z(arr))

    def basic_stats(self):
        groups = self.centrals[self.centrals['N_SAT'] >= 1]
        clusters = self.centrals[self.centrals['N_SAT'] >= 10]

        print(f"Basic stats for {self.name}")
        print(f"  Total galaxies: {len(self.all_data)}")
        print(f"  Total groups (sats >= 1): {len(groups)}")
        print(f"  Total clusters (sats >= 10): {len(clusters)}")
        print(f"  Total satellites: {len(self.sats)}")
        print(f"  Spectroscopic completeness: {self.get_completeness():.2%}")
        print(f"  Footprint: {self.GF_props['frac_area'] * DEGREES_ON_SPHERE:.1f} deg^2")     

    def dump(self):
        self.__class__ = eval(self.__class__.__name__) #reset __class__ attribute
        with open(self.results_file, 'wb') as f:
            pickle.dump(self, f)

    def cleanup_gfproc(self):
        if self.proc is not None:
            try:
                if self.proc.poll() is None:  # If the process is still running
                    self.proc.send_signal(signal.SIGINT)  # Send interrupt signal
                    self.proc.wait(timeout=5)  # Wait for it to finish
            except sp.TimeoutExpired:
                self.proc.kill()
        

    def monitor_pipe(self):
        while self.proc.poll() is None: # while the group finder process is running
            
            header = self.pipereader.read(6)
            if len(header) == 0:
                continue
            if len(header) < 6:
                raise Exception("Incomplete header, len=" + str(len(header)))

            msg_type, data_type, count = struct.unpack("<BBI", header)
            if msg_type not in (MSG_FSAT, MSG_LHMR, MSG_LSAT, MSG_HOD, MSG_HODFIT, MSG_COMPLETED, MSG_ABORTED):
                raise Exception("Unexpected response")
            
            if data_type not in (TYPE_FLOAT, TYPE_DOUBLE):
                raise Exception("Unexpected data type")
            
            dtype_marker = 'd' if data_type == TYPE_DOUBLE else 'f'
            dtype = np.dtype(dtype_marker)

            bytes_needed = count * (8 if data_type == TYPE_DOUBLE else 4)

            payload = self.pipereader.read(bytes_needed)
            if len(payload) < bytes_needed:
                raise Exception(f"Incomplete payload, expected {bytes_needed} bytes but got {len(payload)} bytes") 
            data = struct.unpack(f"<{count}{dtype_marker}", payload)
            assert count == len(data)

            if msg_type == MSG_FSAT:
                if count != 120:
                    raise Exception(f"Unexpected fsat data count: {count}, expected 120")
                else:
                    self.fsat = np.array(data[0:40], dtype=dtype)
                    self.fsatr = np.array(data[40:80], dtype=dtype)
                    self.fsatb = np.array(data[80:120], dtype=dtype)
            elif msg_type == MSG_LHMR:
                if count != 65*3*2:
                    raise Exception(f"Unexpected lhmr data count: {count}, expected 65*3*2")
                else:
                    data = np.array(data, dtype=dtype).reshape((6, 65))
                    self.lhmr_m = np.array(data[0], dtype=dtype)
                    self.lhmr_std = np.array(data[1], dtype=dtype)
                    self.lhmr_r_m = np.array(data[2], dtype=dtype)
                    self.lhmr_r_std = np.array(data[3], dtype=dtype)
                    self.lhmr_b_m = np.array(data[4], dtype=dtype)
                    self.lhmr_b_std = np.array(data[5], dtype=dtype)

            elif msg_type == MSG_LSAT:
                if count != 40:
                    raise Exception(f"Unexpected lsat data count: {count}")
                else:
                    self.lsat_r = np.array(data[0:20], dtype=dtype)
                    self.lsat_b = np.array(data[20:40], dtype=dtype)

            elif msg_type == MSG_HOD:
                cols = self.caldata.bincount*7 + 1
                rows = len(data) // cols
                self.hod = np.array(data, dtype=dtype).reshape((rows, cols))

            elif msg_type == MSG_HODFIT:
                cols = self.caldata.bincount*7 + 1
                rows = len(data) // cols
                self.hodfit = np.array(data, dtype=dtype).reshape((rows, cols))
            
            elif msg_type == MSG_COMPLETED:
                print("Group Finder completed successfully.")
                return True
            
            elif msg_type == MSG_ABORTED:
                print("Group Finder was aborted.")
                return False
            
            else:
                raise Exception(f"Unexpected message type: {msg_type}")

        return True


    def run_group_finder(self, popmock=False, silent=False, verbose=False, profile=False, interactive=False):
        t1 = time.time()
        print("Running Group Finder for " + self.name)

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        if self.preprocess_file is None:
            print("Warning: no input file set. Cannot run group finder.")
            return
        
        if not os.path.exists(self.preprocess_file):
            print(f"Warning: preprocess_file {self.preprocess_file} does not exist. Cannot run group finder.")
            return
        
        # Group Finder expects these files in working directory
        sp.run(["cp", HALO_MASS_FUNC_FILE , self.output_folder])
        sp.run(["cp", LSAT_LOOKUP_FILE , self.output_folder])

        sys.stdout.flush()

        if profile:
            args = ["perf", "record",  "-g", BIN_FOLDER + "PerfGroupFinder", self.preprocess_file]
        else:
            args = [BIN_FOLDER + "kdGroupFinder_omp", self.preprocess_file]

        args.append(str(self.GF_props['zmin']))
        args.append(str(self.GF_props['zmax']))
        args.append(str(self.GF_props['frac_area']))
        if 'fluxlim' in self.GF_props:
            args.append(f"--fluxlim={self.mag_cut},{self.GF_props['fluxlim']}")
        if self.GF_props.get('color') == 1:
            args.append("-c")
        if 'iterations' in self.GF_props:
            args.append("--iterations=" + str(self.GF_props['iterations']))
        if silent:
            args.append("-s")
        if 'earlyexit' in self.GF_props and self.GF_props['earlyexit'] == 1:
            args.append("-e")
        if popmock:
            # Save a file with the volume bin info, which the C code will read
            self.caldata.write_volume_bins_file(self.output_folder + "volume_bins.dat")
            args.append(f"--popmock={MOCK_FILE_FOR_POPMOCK},volume_bins.dat")
        if verbose:
            args.append("-v")
        if interactive:
            args.append("-k") # keep alive
        if 'omegaL_sf' in self.GF_props:
            args.append(f"--wcen={self.GF_props['omegaL_sf']},{self.GF_props['sigma_sf']},{self.GF_props['omegaL_q']},{self.GF_props['sigma_q']},{self.GF_props['omega0_sf']},{self.GF_props['omega0_q']}")
        if 'beta0q' in self.GF_props:
            args.append(f"--bsat={self.GF_props['beta0q']},{self.GF_props['betaLq']},{self.GF_props['beta0sf']},{self.GF_props['betaLsf']}")
        if 'omega_chi_0_sf' in self.GF_props:
            args.append(f"--chi1={self.GF_props['omega_chi_0_sf']},{self.GF_props['omega_chi_0_q']},{self.GF_props['omega_chi_L_sf']},{self.GF_props['omega_chi_L_q']}")            

        read_fd, write_fd = os.pipe()
        fds = (write_fd,)
        args.append(f"--pipe={write_fd}")

        print(args)

        # The galaxies are written to stdout, so send ot the GF_outfile file stream
        self.outstream = open(self.GF_outfile, "w")

        # TODO stderr to a seperate log?
        self.proc = sp.Popen(args, cwd=self.output_folder, stdout=self.outstream, stdin=sp.PIPE, pass_fds=fds)
        MASTER_PROCESS_LIST.append(self.proc)
        self.pipereader = os.fdopen(read_fd, "rb")
        success = self.monitor_pipe()

        if not interactive:
            # Cleanup
            self.proc.stdin.close()
            self.pipereader.close()
            self.proc.wait()
            self.pipereader = None
            self.outstream.close()
            self.outstream = None


            # TODO Group Finder does not consistently return >0 for errors.
            if self.proc.returncode != 0:
                print(f"ERROR: Group Finder failed with return code {self.proc.returncode}.")
                self.proc = None
                return False
            
            self.proc = None
        #if popmock:
        #    # TODO switch these to use pipe instead of file
        #    hodout = f'{self.output_folder}hod.out'
        #    hodfitout = f'{self.output_folder}hod_fit.out'
        #    if os.path.exists(hodout):
        #        self.hod = np.loadtxt(hodout, skiprows=4, dtype='float', delimiter=' ')
        #    if os.path.exists(hodfitout):
        #        self.hodfit = np.loadtxt(hodfitout, skiprows=4, dtype='float', delimiter=' ')

        t2 = time.time()
        print(f"run_group_finder() took {t2-t1:.1f} seconds.")
        return success


    def calc_wp_for_mock(self):
        """
        Run corrfunc on the mock populated with an HOD built from this sample. 
        """
        print("Running Corrfunc on mock populated with HOD from this sample.")
        t1 = time.time()
        if self.GF_outfile is None:
            print("Warning: calc_wp_for_mock() called without GF_outfile set.")
            return
        if not os.path.exists(self.GF_outfile):
            print(f"Warning: calc_wp_for_mock() should be called after run_group_finder().")
            return
        
        # Define the path to the Corrfunc binary
        corrfunc_path = "/export/sirocco1/tinker/src/Corrfunc/bin/"
        if not os.path.exists(os.path.join(corrfunc_path, "wp")):
            corrfunc_path = "/mount/sirocco1/tinker/src/Corrfunc/bin/"
        
        mag_starts = self.caldata.magbins[:-1]

        nthreads = os.cpu_count()
        pimax = 40
        boxsize = 250

        # Call corrfunc compiled executable to compute wp on the mock that was populated with the HOD extracted from this catalog
        for i in range(len(mag_starts)):
            m = abs(mag_starts[i])
            for color in ["red", "blue", "all"]:
                if color == "red" and not self.caldata.color_separation[i]:
                    continue
                if color == "blue" and not self.caldata.color_separation[i]:
                    continue
                if color == "all" and self.caldata.color_separation[i]:
                    continue

                # The mock file is written by the group finder as mock_{col}_M{m}.dat
                outf = f"wp_mock_{color}_M{m}.dat"
                cmd = f"{corrfunc_path}/wp {boxsize} mock_{color}_M{m}.dat a {self.caldata.rpbinsfile} {pimax} {nthreads} > wp_mock_{color}_M{m}.dat 2> wp_stderr.txt"
                result = sp.run(cmd, cwd=self.output_folder, shell=True, check=True)
                if result.returncode != 0:
                    print(f"Error running command: {cmd}")
                self.wp_mock[(color, m)] = np.loadtxt(f'{self.output_folder}{outf}', skiprows=0, dtype='float')

        t2 = time.time()
        print(f"Done with wp on mock populated with HOD from this sample (time = {t2-t1:.1f}s).")

    def refresh_df_views(self):
        if self.all_data is not None:
            # Compute some common aggregations upfront here
            # TODO make these lazilly evaluated properties on the GroupCatalog object
            # Can put more of them into this pattern from elsewhere in plotting code then
            self.f_sat = self.all_data.groupby('LGAL_BIN', observed=False).apply(fsat_vmax_weighted)
            self.f_sat_sf = self.all_data.loc[~self.all_data['QUIESCENT']].groupby('LGAL_BIN', observed=False).apply(fsat_vmax_weighted)
            self.f_sat_q = self.all_data.loc[self.all_data['QUIESCENT']].groupby('LGAL_BIN', observed=False).apply(fsat_vmax_weighted)
            self.Lgal_counts = self.all_data.groupby('LGAL_BIN', observed=False).RA.count()

            # TODO this is incomplete; just counting galaxies right now
            #self.f_sat_cic = cic_binning(self.all_data['L_GAL'].to_numpy(), [self.L_gal_bins])
            #self.f_sat_sf_cic = cic_binning(self.all_data.loc[~self.all_data['QUIESCENT'], 'L_GAL'].to_numpy(), [self.L_gal_bins])
            #self.f_sat_q_cic = cic_binning(self.all_data.loc[self.all_data['QUIESCENT'], 'L_GAL'].to_numpy(), [self.L_gal_bins])

            # Setup some convenience subsets of the DataFrame
            self.centrals = self.all_data.loc[self.all_data.index == self.all_data['IGRP']]
            self.sats = self.all_data.loc[self.all_data.index != self.all_data['IGRP']]

    def postprocess(self):
        if self.all_data is not None:
            self.refresh_df_views()
        else:
            print("Warning: postprocess called with all_data DataFrame is not set yet. Override postprocess() or after calling run_group_finder() set it.")

    def calculate_projected_clustering(self, with_extra_randoms=False):
        pass

    def calculate_projected_clustering_in_magbins(self, with_extra_randoms=False):
        pass

    def chisqr(self):
        """ 
        Evaluate the quality of the HOD implied by the group finder results
        by comparing a mock populated with the HOD to the external datasets.
        """
        if len(self.wp_mock) == 0:
            print("WARNING: chisqr() called without having populated the mock.")
            return

        dof = 0

        with np.printoptions(precision=0, suppress=True, linewidth=300):
            chi = 0
            clustering_chisqr_r = []
            clustering_chisqr_b = []
            clustering_chisqr_all = []
            lsat_chisqr = []

            # PROJECTED CLUSTERING COMPARISON
            mag_limits = self.caldata.magbins[:-1]

            for i in range(len(mag_limits)):
                mag = abs(mag_limits[i])

                if self.caldata.color_separation[i]:
                    wp, wp_err, radius = self.caldata.get_wp_red(mag)
                    wp_model, wp_err_model = self.get_mock_wp(mag, 'red', wp_err)

                    chivec = (wp_model-wp)**2/(wp_err**2 + wp_err_model**2) 
                    dof += len(chivec)
                    clustering_chisqr_r.append(np.sum(chivec))

                    wp, wp_err, radius = self.caldata.get_wp_blue(mag)
                    wp_model, wp_err_model = self.get_mock_wp(mag, 'blue', wp_err)

                    chivec = (wp_model-wp)**2/(wp_err**2 + wp_err_model**2) 
                    dof += len(chivec)
                    clustering_chisqr_b.append(np.sum(chivec))

                    clustering_chisqr_all.append(0)

                else:
                    wp, wp_err, radius = self.caldata.get_wp_all(mag)
                    wp_model, wp_err_model = self.get_mock_wp(mag, 'all', wp_err)

                    chivec = (wp_model-wp)**2/(wp_err**2 + wp_err_model**2) 
                    dof += len(chivec)
                    clustering_chisqr_all.append(np.sum(chivec))

                    clustering_chisqr_r.append(0)
                    clustering_chisqr_b.append(0)


            print("Red Clustering χ^2: ", np.array(clustering_chisqr_r))
            print("Blue Clustering χ^2: ", np.array(clustering_chisqr_b))
            print("No sep Clustering χ^2: ", np.array(clustering_chisqr_all))

            # LSAT COMPARISON
            lsat_chisqr = compute_lsat_chisqr(self.caldata.lsat_observations, self.lsat_r, self.lsat_b)
            dof += len(lsat_chisqr)
            
            # TODO automate whether this is on or off depending on GF parameters?
            # This is for the second parameter (galaxy concentration)    
            """
            # now do lsat vs second parameter BLUE
            fname = datafolder + "lsat_sdss_con.dat"
            data = ascii.read(fname, delimiter='\s', format='no_header')
            y = np.array(data['col2'][...], dtype='float')
            e = np.array(data['col3'][...], dtype='float')

            fname = self.output_folder + "lsat_groups_propx_blue.out"
            data = ascii.read(fname, delimiter='\s', format='no_header')
            m = np.array(data['col2'][...], dtype='float')
            
            em = m*(e/y)
            chivec = (y-m)**2/(e**2+em**2)
            chi = chi + np.sum(chivec)
                
            # now do lsat vs second parameter RED
            fname = datafolder + "lsat_sdss_con.dat"
            data = ascii.read(fname, delimiter='\s', format='no_header')
            y = np.array(data['col4'][...], dtype='float')
            e = np.array(data['col5'][...], dtype='float')

            fname = self.output_folder + "lsat_groups_propx_red.out"
            data = ascii.read(fname, delimiter='\s', format='no_header')
            m = np.array(data['col2'][...], dtype='float')

            em = m*(e/y)
            chivec = (y-m)**2/(e**2+em**2)
            chi = chi + np.sum(chivec)
            """

            chi = np.sum(lsat_chisqr) + np.sum(clustering_chisqr_r) + np.sum(clustering_chisqr_b) + np.sum(clustering_chisqr_all)

            # Print off the chi squared value and model info and return it 
            #print(f'MODEL {ncount}')
            print(f'χ^2: {chi:.1f}. χ^2/DOF: {chi/dof:.3f} (dof={dof})')
            #os.system('date')
            sys.stdout.flush()

        return chi, clustering_chisqr_r, clustering_chisqr_b, clustering_chisqr_all, lsat_chisqr

    def run_and_calc_chisqr(self, params):

        if len(params) != 10 and len(params) != 14:
            print("Warning: chisqr called with wrong number of parameters. Expected 10 or 14.")
        if self.proc is None:
            raise Exception("Group finder process is not running, but expected to be.")
        if self.pipereader is None:
            raise Exception("Group finder pipe reader is no longer open, but expected to be.")

        self.GF_props = {
            'zmin':self.GF_props['zmin'],
            'zmax':self.GF_props['zmax'],
            'frac_area':self.GF_props['frac_area'],
            'fluxlim':self.GF_props['fluxlim'],
            'color':self.GF_props['color'],
            'omegaL_sf':params[0],
            'sigma_sf':params[1],
            'omegaL_q':params[2],
            'sigma_q':params[3],
            'omega0_sf':params[4],  
            'omega0_q':params[5],    
            'beta0q':params[6],    
            'betaLq':params[7],
            'beta0sf':params[8],
            'betaLsf':params[9],
        }
        if len(params) == 14:
            self.GF_props['omega_chi_0_sf'] = params[10]
            self.GF_props['omega_chi_0_q'] = params[11]
            self.GF_props['omega_chi_L_sf'] = params[12]
            self.GF_props['omega_chi_L_q'] = params[13]

        # Send message to GF TODO
        # &(WCEN_MASS), &(WCEN_SIG), &(WCEN_MASSR), &(WCEN_SIGR), &(WCEN_NORM), &(WCEN_NORMR), &(BPROB_RED), &(BPROB_XRED), &(BPROB_BLUE), &(BPROB_XBLUE)
        print("Sending message for next GF iteration")
        msg = struct.pack("<BBI", MSG_REQUEST, TYPE_DOUBLE, len(params)) + struct.pack(f"<{len(params)}d", *params)
        self.proc.stdin.write(msg)
        self.proc.stdin.flush()

        # Wait for the group finder to re-run with these parameters
        success = self.monitor_pipe()  
        if not success:
            # GF aborted early for very bad parameters or something similar
            return np.inf

        # No issues, calculate goodness of fit and give it back to emcee
        self.calc_wp_for_mock()
        overall, clust_r, clust_b, clust_nosep, lsat = self.chisqr()
        return overall


    def get_true_z_from(self, truth_df: pd.DataFrame):
        """
        Adds a column to the catalog's all_data DataFrame with the true redshifts from the truth_df DataFrame 
        for rows with z_assigned_flag != 0.
        """
        #self.all_data = self.all_data.convert_dtypes()
        
        if self.has_truth:
            self.all_data.drop(columns=['TARGETID_T', 'Z_T', 'Z_ASSIGNED_FLAG_T', 'L_GAL_T', 'G_R_T', 'IS_SAT_T', 'LGAL_BIN_T'], inplace=True, errors='ignore')
        
        truth_df = truth_df[['TARGETID', 'Z', 'Z_ASSIGNED_FLAG', 'L_GAL', 'G_R', 'IS_SAT', 'LGAL_BIN']].copy()
        truth_df['TARGETID'] = truth_df['TARGETID'].astype('Int64')
        truth_df.index = truth_df.TARGETID
        self.all_data = self.all_data.join(truth_df, on='TARGETID', how='left', rsuffix='_T', validate="1:1")
        # I want target_id_T to be int, but there are NaNs in the join, so turn NaN into -1
        rows_to_nan = z_flag_is_not_spectro_z(self.all_data['Z_ASSIGNED_FLAG_T'])
        self.all_data.loc[rows_to_nan, 'Z_T'] = NO_TRUTH_Z
        print(f"{np.sum(rows_to_nan)} galaxies to have no truth redshift.")
        self.has_truth = True

        # If we failed to match on target_id for most galaxies, let's instead use match_coordinate_sky
        #if np.sum(rows_to_nan) > 0.5 * len(self.all_data):
        #    print("Warning: get_true_z_from() failed to match target_id for many galaxies. Falling back to match_coordinate_sky.")
            # drop the columns we added
        #    self.all_data.drop(columns=['Z_T', 'IS_SAT_T'], inplace=True)

            # match on sky coordinates
        #    coords_catalog = coord.SkyCoord(ra=self.all_data.RA.to_numpy() * u.degree, dec=self.all_data['DEC'].to_numpy() * u.degree)
        #    coords_truth = coord.SkyCoord(ra=truth_df.RA.to_numpy() * u.degree, dec=truth_df['DEC'].to_numpy() * u.degree)
        #    idx, d2d, _ = coord.match_coordinates_sky(coords_catalog, coords_truth, nthneighbor=1)
        #    matched = d2d < 1 * u.arcsec  # 1 arcsecond tolerance

        #    self.all_data['Z_T'] = NO_TRUTH_Z
        #    self.all_data.loc[matched, 'Z_T'] = truth_df.iloc[idx[matched]]['Z']
        #    self.all_data['IS_SAT_T'] = 0
        #    self.all_data.loc[matched, 'IS_SAT_T'] = truth_df.iloc[idx[matched]]['IS_SAT']
            


class SDSSGroupCatalog(GroupCatalog):
    
    @staticmethod
    def from_MCMC(reader: emcee.backends.HDFBackend, name: str, preprocessed_file: str, galprops_file: str):
        gc = SDSSGroupCatalog(name, preprocessed_file, galprops_file)

        # Use lowest chi squared (highest log prob) parameter set
        idx = np.argmax(reader.get_log_prob(flat=True))
        values = reader.get_chain(flat=True)[idx]
        if len(values) != 10 and len(values) != 14:
            print("Warning: reader has wrong number of parameters. Expected 10 or 14.")
        gc.GF_props = {
            'zmin':0,
            'zmax':1.0,
            'frac_area':0.179,
            'fluxlim':1,
            'color':1,
            'omegaL_sf':values[0],
            'sigma_sf':values[1],
            'omegaL_q':values[2],
            'sigma_q':values[3],
            'omega0_sf':values[4],
            'omega0_q':values[5],
            'beta0q':values[6],
            'betaLq':values[7],
            'beta0sf':values[8],
            'betaLsf':values[9],
        }
        if len(values) == 14:
            gc.GF_props['omega_chi_0_sf'] = values[10]
            gc.GF_props['omega_chi_0_q'] = values[11]
            gc.GF_props['omega_chi_L_sf'] = values[12]
            gc.GF_props['omega_chi_L_q'] = values[13]
        
        return gc

    def __init__(self, name: str, preprocessed_file: str, galprops_file: str, gfprops: dict):
        super().__init__(name)
        self.preprocess_file = preprocessed_file
        self.galprops_file = galprops_file
        self.L_gal_bins = self.L_gal_bins[15:] # Cutoff Sloan plots where the data falls off
        self.L_gal_labels = self.L_gal_labels[15:]
        self.Mr_gal_bins = self.Mr_gal_bins[15:]
        self.Mr_gal_labels = self.Mr_gal_labels[15:]
        self.mag_cut = 17.7
        self.GF_props = gfprops
        self.caldata = CalibrationData.SDSS_5bin(self.mag_cut, self.GF_props['frac_area'])

        # TODO BUG right volumes?
        #Volume of bin 0 is 344276.781250
        #Volume of bin 1 is 1321019.625000
        #Volume of bin 2 is 4954508.500000
        #Volume of bin 3 is 18012528.000000
        #Volume of bin 4 is 62865016.000000

        # Copied from Jeremy's groupfind_mcmc.py
        #volume = [ 3.181e+05, 1.209e+06, 4.486e+06, 1.609e+07, 5.517e+07 ] # if including -17
        #volume = [ 1.209e+06, 4.486e+06, 1.609e+07, 5.517e+07 ] #if starting at -18
        #volume = [  1.721e+06, 6.385e+06, 2.291e+07, 7.852e+07 ] # actual SDSS

        #self.x_volume = np.array([1321019, 4954508, 18012528, 62865016]) 
        #vfac = (self.x_volume/250.0**3)**.5 # factor by which to multiply errors

    def postprocess(self):
        origprops = pd.read_csv(self.preprocess_file, delimiter=' ', names=('RA', 'DEC', 'Z', 'LOGLGAL', 'VMAX', 'QUIESCENT', 'CHI'))
        galprops = pd.read_csv(self.galprops_file, delimiter=' ', names=('MAG_G', 'MAG_R', 'SIGMA_V', 'DN4000', 'CONCENTRATION', 'LOG_M_STAR', 'Z_ASSIGNED_FLAG'))
        galprops['G_R'] = galprops['MAG_G'] - galprops['MAG_R'] 
        galprops.rename(columns={'MAG_R': 'APP_MAG_R'}, inplace=True)
        galprops['QUIESCENT'] = origprops['QUIESCENT'].astype(bool)
        self.all_data = read_and_combine_gf_output(self, galprops)
        self.all_data['MSTAR'] = np.power(10, self.all_data.LOG_M_STAR)
        self.all_data['Mstar_bin'] = pd.cut(x = self.all_data['MSTAR'], bins = mstar_bins, labels = mstar_labels, include_lowest = True)
        super().postprocess()

class SDSSPublishedGroupCatalog(GroupCatalog):

    def __init__(self, name):
        super().__init__(name)
        self.preprocess_file = None
        self.GF_outfile = SDSS_FOLDER + 'sdss_kdgroups_v1.0.dat'
        self.color = get_color(4)
        self.marker = '-'
        self.GF_props = {
            'zmin':0,
            'zmax':1.0,
            'frac_area':0.179,
            'fluxlim':1,
            'color':1,
            'omegaL_sf':13.1,
            'sigma_sf':2.42,
            'omegaL_q':12.9,
            'sigma_q':4.84,
            'omega0_sf':17.4,  
            'omega0_q':2.67,    
            'beta0q':-0.92,    
            'betaLq':10.25,
            'beta0sf':12.993,
            'betaLsf':-8.04,
            'omega_chi_0_sf':2.68,  
            'omega_chi_0_q':1.10,
            'omega_chi_L_sf':2.23,
            'omega_chi_L_q':0.48,
        }

    def postprocess(self):
        origprops = pd.read_csv(SDSS_v1_DAT_FILE, delimiter=' ', names=
                                ('RA', 'DEC', 'Z', 'LOGLGAL', 'VMAX', 'QUIESCENT', 'CHI'))
        galprops = pd.read_csv(SDSS_v1_1_GALPROPS_FILE, delimiter=' ', names=
                               ('MAG_G', 'MAG_R', 'SIGMA_V', 'DN4000', 'CONCENTRATION', 'LOG_M_STAR', 'Z_ASSIGNED_FLAG'))
        galprops['G_R'] = galprops['MAG_G'] - galprops['MAG_R'] 
        galprops['QUIESCENT'] = origprops['QUIESCENT'].astype(bool)
        galprops.rename(columns={'MAG_R': "app_mag"}, inplace=True)
        
        main_df = pd.read_csv(self.GF_outfile, delimiter=' ', names=
                              ('RA', 'DEC', 'Z', 'L_GAL', 'VMAX', 'P_SAT', 'M_HALO', 'N_SAT', 'L_TOT', 'IGRP', 'WEIGHT'))
        
        df = pd.merge(main_df, galprops, left_index=True, right_index=True, validate="1:1")

        # add columns indicating if galaxy is a satellite
        df['IS_SAT'] = (df.index != df['IGRP']).astype(bool)
        df['LOGLGAL'] = np.log10(df['L_GAL'])

        # add column for halo mass bins and Lgal bins
        df['Mh_bin'] = pd.cut(x = df['M_HALO'], bins = self.Mhalo_bins, labels = self.Mhalo_labels, include_lowest = True)
        df['LGAL_BIN'] = pd.cut(x = df['L_GAL'], bins = self.L_gal_bins, labels = self.L_gal_labels, include_lowest = True)

        self.all_data = df
        self.all_data['MSTAR'] = np.power(10, self.all_data.LOG_M_STAR)
        self.all_data['Mstar_bin'] = pd.cut(x = self.all_data['MSTAR'], bins = mstar_bins, labels = mstar_labels, include_lowest = True)
        super().postprocess()

class TestGroupCatalog(GroupCatalog):
    """
    A miniature flux-limited sample cut from SDSS data for quick testing purposes.

    The COSMOS survey is centered at (J2000):
    RA +150.11916667 (10:00:28.600)
    DEC +2.20583333 (+02:12:21.00)

    This sample is cut to around this, +/- 1 degree in RA and DEC.

    """
    def __init__(self, name):
        super().__init__(name)
        self.preprocess_file = TEST_DAT_FILE
        self.GF_props = {
            'zmin':0,
            'zmax':1.0,
            'frac_area':4.0/DEGREES_ON_SPHERE,
            'fluxlim':1,
            'color':0,
        }
        self.mag_cut = 17.7
        self.caldata = CalibrationData.SDSS_5bin(self.mag_cut, self.GF_props['frac_area'])

    def create_test_dat_files(self):
        gals = pd.read_csv(SDSS_v1_DAT_FILE, delimiter=' ', names=('RA', 'DEC', 'Z', 'LOGLGAL', 'VMAX', 'QUIESCENT', 'CHI'))
        galprops = pd.read_csv(SDSS_v1_1_GALPROPS_FILE, delimiter=' ', names=('MAG_G', 'MAG_R', 'SIGMA_V', 'DN4000', 'CONCENTRATION', 'LOG_M_STAR', 'Z_ASSIGNED_FLAG'))

        cut_gals = gals[np.logical_and(gals.RA > 149.119, gals.RA < 151.119)]
        cut_gals = cut_gals[np.logical_and(cut_gals['DEC'] > 1.205, cut_gals['DEC'] < 3.205)]
        indexes = cut_gals.index
        print(f"Cut to {len(indexes)} galaxies.")
        cut_galprops = galprops.iloc[indexes]

        # write to TEST_DAT_FILE and TEST_GALPROPS_FILE
        cut_gals.to_csv(TEST_DAT_FILE, sep=' ', header=False, index=False)
        cut_galprops.to_csv(TEST_GALPROPS_FILE, sep=' ', header=False, index=False)

    def postprocess(self):
        origprops = pd.read_csv(TEST_DAT_FILE, delimiter=' ', names=('RA', 'DEC', 'Z', 'LOGLGAL', 'VMAX', 'QUIESCENT', 'CHI'))
        galprops = pd.read_csv(TEST_GALPROPS_FILE, delimiter=' ', names=('MAG_G', 'MAG_R', 'SIGMA_V', 'DN4000', 'CONCENTRATION', 'LOG_M_STAR', 'Z_ASSIGNED_FLAG'))
        galprops['G_R'] = galprops['MAG_G'] - galprops['MAG_R'] 
        galprops['QUIESCENT'] = origprops['QUIESCENT'].astype(bool)
        galprops.rename(columns={'MAG_R': 'APP_MAG_R'}, inplace=True)
        self.all_data = read_and_combine_gf_output(self, galprops)
        add_halo_columns(self)
        return super().postprocess()

class MXXLGroupCatalog(GroupCatalog):

    def __init__(self, name, mode: Mode, mag_cut: float, catalog_mag_cut: float, use_colors: bool, gfprops: dict):
        super().__init__(name)
        self.mode = mode
        self.mag_cut = mag_cut
        self.catalog_mag_cut = catalog_mag_cut
        self.use_colors = use_colors
        self.color = mode_to_color(mode)
        self.GF_props = gfprops
        

    def preprocess(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        fname, props = pre_process_mxxl(MXXL_FILE, self.mode.value, self.file_pattern, self.mag_cut, self.catalog_mag_cut, self.use_colors)
        self.preprocess_file = fname
        for p in props:
            self.GF_props[p] = props[p]

    def run_group_finder(self, popmock=False, silent=False, verbose=False, profile=False, interactive=False):
        if self.preprocess_file is None:
            self.preprocess()
        return super().run_group_finder(popmock=popmock, silent=silent, profile=profile, interactive=interactive)


    def postprocess(self):

        filename_props_fast = str.replace(self.GF_outfile, ".out", "_galprops.pkl")
        filename_props_slow = str.replace(self.GF_outfile, ".out", "_galprops.dat")
        if os.path.exists(filename_props_fast):
            galprops = pd.read_pickle(filename_props_fast)
        else:
            galprops = pd.read_csv(filename_props_slow, delimiter=' ', names=('APP_MAG_R', 'G_R', 'galaxy_type', 'mxxl_halo_mass', 'Z_ASSIGNED_FLAG', 'assigned_halo_mass', 'Z_OBS', 'mxxl_halo_id', 'assigned_halo_id'), dtype={'mxxl_halo_id': np.int32, 'assigned_halo_id': np.int32, 'Z_ASSIGNED_FLAG': np.int32})
        
        self.all_data = read_and_combine_gf_output(self, galprops)
        df = self.all_data
        self.has_truth = True#self.mode.value == Mode.ALL.value
        df['IS_SAT_T'] = np.logical_or(df.galaxy_type == 1, df.galaxy_type == 3)
        df['Z_T'] = df['Z_OBS'] # MXXL truth values are always there
        if self.has_truth:
            df['Mh_bin_T'] = pd.cut(x = df['mxxl_halo_mass']*10**10, bins = Mhalo_bins, labels = Mhalo_labels, include_lowest = True)
            df['L_GAL_T'] = np.power(10, abs_mag_r_to_log_solar_L(app_mag_to_abs_mag_k(df.app_mag.to_numpy(), df.z_obs.to_numpy(), df.G_R.to_numpy())))
            df['LGAL_BIN_T'] = pd.cut(x = df['L_GAL_T'], bins = self.L_gal_bins, labels = self.L_gal_labels, include_lowest = True)
            self.truth_f_sat = df.groupby('LGAL_BIN_T').apply(fsat_truth_vmax_weighted)
            self.centrals_T = df[np.invert(df.IS_SAT_T)]
            self.sats_T = df[df.IS_SAT_T]

        # TODO if we switch to using bins we need a Truth version of this
        df['QUIESCENT'] = is_quiescent_BGS_gmr(df['LOGLGAL'], df.G_R)

        super().postprocess()

class UchuuGroupCatalog(GroupCatalog):
   
    def __init__(self, name, mode: Mode, mag_cut: float, catalog_mag_cut: float, use_colors: bool, gfprops: dict):
        super().__init__(name)
        self.mode = mode
        self.mag_cut = mag_cut
        self.catalog_mag_cut = catalog_mag_cut
        self.use_colors = use_colors
        self.color = get_color(9)
        self.GF_props = gfprops

    def preprocess(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        fname, props = pre_process_uchuu(UCHUU_FILE, self.mode.value, self.file_pattern, self.mag_cut, self.catalog_mag_cut, self.use_colors)
        self.preprocess_file = fname
        for p in props:
            self.GF_props[p] = props[p]

    def run_group_finder(self, popmock=False, silent=False, verbose=False, profile=False, interactive=False):
        if self.preprocess_file is None:
            self.preprocess()
        return super().run_group_finder(popmock=popmock, silent=silent, profile=profile, interactive=interactive)


    def postprocess(self):
        galprops = pd.read_pickle(str.replace(self.GF_outfile, ".out", "_galprops.pkl"))
        
        df = read_and_combine_gf_output(self, galprops)
        self.all_data = df

        self.has_truth = True
        self['IS_SAT_T'] = np.invert(df.central)
        self['Mh_bin_T'] = pd.cut(x = self['uchuu_halo_mass']*10**10, bins = Mhalo_bins, labels = Mhalo_labels, include_lowest = True)
        # TODO BUG Need L_GAL_T, the below is wrong!
        truth_f_sat = df.groupby('LGAL_BIN').apply(fsat_truth_vmax_weighted)
        self.truth_f_sat = truth_f_sat
        self.centrals_T = df[np.invert(df.IS_SAT_T)]
        self.sats_T = df[df.IS_SAT_T]

        # TODO add quiescent column

        super().postprocess()

class BGSGroupCatalog(GroupCatalog):
    
    extra_prop_df: pd.DataFrame = None

    def __init__(self, name, mode: Mode, mag_cut: float, catalog_mag_cut: float, sdss_fill: bool = True, num_passes: int = 3, drop_passes: int = 0, data_cut: str = "Y1-Iron", gfprops=None, extra_params = None, caldata_ctor=None):
        super().__init__(name)
        self.mode = mode
        self.mag_cut = mag_cut
        self.catalog_mag_cut = catalog_mag_cut
        self.color = mode_to_color(mode)
        self.sdss_fill = sdss_fill
        self.num_passes = num_passes
        self.drop_passes = drop_passes
        self.data_cut = data_cut
        self.is_centered_version = False
        self.centered = None # SV3 Centered version shortcut.
        self.extra_params = extra_params
        self.GF_props = gfprops
        frac_area = get_footprint_fraction(data_cut, mode, num_passes)
        self.GF_props['frac_area'] = frac_area
        if caldata_ctor is None:
            self.caldata = CalibrationData.BGS_Y1_6bin(self.mag_cut, self.GF_props['frac_area'])
        else:
            self.caldata = caldata_ctor(self.mag_cut, self.GF_props['frac_area'])

    @staticmethod
    def from_MCMC(reader: emcee.backends.HDFBackend, mode: Mode):

        idx = np.argmax(reader.get_log_prob(flat=True))
        p = reader.get_chain(flat=True)[idx]
        if len(p) != 13:
            raise ValueError("reader has wrong number of parameters. Expected 13.")
        
        print(f"Using MCMC parameters: {p}")

        gc = BGSGroupCatalog(f"BGS SV3 MCMC {mode_to_str(mode)}", mode, 19.5, 23.0, sdss_fill=False, num_passes=10, drop_passes=3, data_cut="sv3", gfprops=GF_PROPS_BGS_VANILLA.copy(), extra_params=p)
        if mode.value == Mode.PHOTOZ_PLUS_v1.value:
            gc.color = 'g'
        elif mode.value == Mode.PHOTOZ_PLUS_v2.value:
            gc.color = 'darkorange'
        elif mode.value == Mode.PHOTOZ_PLUS_v3.value:
            gc.color = 'purple'
        
        gc.marker = '--'

        return gc

    def preprocess(self, silent=False):
        t1 = time.time()
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        print("Pre-processing...")
        if self.data_cut == "Y1-Iron":
            infile = IAN_BGS_Y1_MERGED_FILE
        elif self.data_cut == "Y1-Iron-Mini":
            infile = IAN_BGS_Y1_MERGED_FILE
        elif self.data_cut == "Y3-Kibo":
            raise ValueError("Y3 Kibo no longer supported")
        elif self.data_cut == "Y3-Kibo-SV3Cut":
            raise ValueError("Y3 Kibo no longer supported")
        elif self.data_cut == "Y3-Loa":
            infile = IAN_BGS_Y3_MERGED_FILE_LOA
        elif self.data_cut == "Y3-Loa-SV3Cut":
            infile = IAN_BGS_Y3_MERGED_FILE_LOA_SV3CUT
        elif self.data_cut == "sv3":
            infile = IAN_BGS_SV3_MERGED_FILE
        else:
            raise ValueError("Unknown data_cut value")
        
        if silent:
            with SuppressPrint():
                fname, props = pre_process_BGS(infile, self.mode.value, self.file_pattern, self.mag_cut, self.catalog_mag_cut, self.sdss_fill, self.num_passes, self.drop_passes, self.data_cut, self.extra_params)
        else:
            fname, props = pre_process_BGS(infile, self.mode.value, self.file_pattern, self.mag_cut, self.catalog_mag_cut, self.sdss_fill, self.num_passes, self.drop_passes, self.data_cut, self.extra_params)
        
        self.preprocess_file = fname
        for p in props:
            self.GF_props[p] = props[p]
        
        t2 = time.time()
        print(f"Pre-processing complete in {t2-t1:.2} seconds.")

    def setup_GF_mcmc(self, mcmc_num: int = None):
        #self.run_group_finder(popmock=True)
        super().setup_GF_mcmc(mcmc_num)

    def run_group_finder(self, popmock=False, silent=False, verbose=False, profile=False, interactive=False):
        if self.preprocess_file is None:
            self.preprocess(silent=silent)
        else:
            print("Skipping pre-processing")
        return super().run_group_finder(popmock=popmock, silent=silent, verbose=verbose, profile=profile, interactive=interactive)

    def add_bootstrapped_f_sat(self, N_ITERATIONS = 100):

        df = self.all_data

        if self.data_cut == 'sv3':
            print("Bootstrapping for fsat error estimate...")
            t1 = time.time()
            # label the SV3 region each galaxy is in
            df['region'] = tile_to_region(df['NTID'])

            # Add bootstrapped error bars for fsat
            f_sat_realizations = []
            f_sat_sf_realizations = []
            f_sat_q_realizations = []

            def bootstrap_iteration(region_indices):
                relevent_columns = ['LGAL_BIN', 'IS_SAT', 'VMAX', 'QUIESCENT']
                alt_df = pd.DataFrame(columns=relevent_columns)
                for idx in region_indices:
                    rows_to_add = df.loc[df.region == idx, relevent_columns]
                    if len(alt_df) > 0:
                        alt_df = pd.concat([alt_df, rows_to_add])
                    else:
                        alt_df = rows_to_add

                f_sat = alt_df.groupby('LGAL_BIN', observed=False).apply(fsat_vmax_weighted)
                f_sat_sf = alt_df[alt_df['QUIESCENT'] == False].groupby('LGAL_BIN', observed=False).apply(fsat_vmax_weighted)
                f_sat_q = alt_df[alt_df['QUIESCENT'] == True].groupby('LGAL_BIN', observed=False).apply(fsat_vmax_weighted)
                return f_sat, f_sat_sf, f_sat_q

            #results = Parallel(n_jobs=-1)(delayed(bootstrap_iteration)(np.random.choice(range(len(sv3_regions_sorted)), len(sv3_regions_sorted), replace=True)) for _ in range(N_ITERATIONS))
            results = [bootstrap_iteration(np.random.choice(range(len(sv3_regions_sorted)), len(sv3_regions_sorted), replace=True)) for _ in range(N_ITERATIONS)]

            f_sat_realizations, f_sat_sf_realizations, f_sat_q_realizations = zip(*results)

            self.f_sat_err = np.std(f_sat_realizations, axis=0)
            self.f_sat_sf_err = np.std(f_sat_sf_realizations, axis=0)
            self.f_sat_q_err = np.std(f_sat_q_realizations, axis=0)

            t2 = time.time()
            print(f"Bootstrapping complete in {t2-t1:.2} seconds.")
        else:
            pass

    def postprocess(self):
        print("Post-processing...")
        galprops = pd.read_pickle(str.replace(self.GF_outfile, ".out", "_galprops.pkl"))
        df = read_and_combine_gf_output(self, galprops)
        
        bad = df.loc[np.isnan(df.L_GAL)]
        if len(bad) > 0:
            print(f"Warning: {len(bad)} galaxies have nan L_GAL.")

        # Get extra fastspecfit columns. Could have threaded these through with galprops
        # But if they aren't used in group finding or preprocessing this is easier to update
        if BGSGroupCatalog.extra_prop_df is None:
            print("Getting fastspecfit data...", end='\r')
            BGSGroupCatalog.extra_prop_df = get_extra_bgs_fastspectfit_data()
            print("Getting fastspecfit data... done")
        
        prior_len = len(df)
        df = pd.merge(df, BGSGroupCatalog.extra_prop_df, on='TARGETID', how='left')
        assert prior_len == len(df)
        df['Mstar_bin'] = pd.cut(x = df['MSTAR'], bins = mstar_bins, labels = mstar_labels, include_lowest = True)

        self.all_data = df

        super().postprocess()

        if self.data_cut == 'sv3':
            self.centered = filter_SV3_to_avoid_edges(self)

        #if self.data_cut == 'sv3' or self.data_cut == 'Y3-Kibo-SV3Cut' or self.data_cut == 'Y3-Loa-SV3Cut':
        #   self.calculate_projected_clustering()
        #   self.calculate_projected_clustering_in_magbins()

        print("Post-processing done.")

    def basic_stats(self):
        super().basic_stats()
        print(f"  Lost Galaxy Handling: {self.mode}")     
        if self.extra_params is not None:
            print(f"    Parameters: {self.extra_params}")
            zflag = self.all_data['Z_ASSIGNED_FLAG']
            print(f"    Neighbor usage %: {z_flag_is_neighbor(zflag).sum() / z_flag_is_not_spectro_z(zflag).sum() * 100:.2f}")
        print(f"  Magnitude Limit: {self.mag_cut}")     
        print(f"  Neighbor Magnitude Limit: {self.catalog_mag_cut}")     
        print(f"  Use SDSS Redshifts: {self.sdss_fill}")
        print(f"  Min. # of Passes: {self.num_passes}")
        print(f"  Data Version: {self.data_cut}")

    def get_volume_limited_sample(self, dim_mag: float, bright_mag=None, zmin=0):
        """
        Construct a volume-limited sample of galaxies with galaxies brighter than dim_mag.
        Optionally include a bright mag limit as well.
        The maximum z will be calculated automatiaclly for the given flux limit of this catalog.
        """
        zmax = get_max_observable_z(dim_mag, self.mag_cut)
        zmax = zmax.value

        if zmin > zmax:
            raise ValueError(f"zmin {zmin:.5f} is greater than zmax {zmax:.5f}. Cannot create volume-limited sample.")
        
        if bright_mag is not None and bright_mag > dim_mag:
            raise ValueError(f"bright_mag {bright_mag} is greater than dim_mag {dim_mag}. Cannot create volume-limited sample.")

        print(f"Creating volume-limited sample with zmin={zmin:.5f}, zmax={zmax:.5f}, dim_mag={dim_mag:.2f}.")
        if bright_mag is not None:
            print(f"Bright magnitude limit: {bright_mag:.2f}")

        df = self.all_data
        df = df.loc[np.logical_and(df['Z'] > zmin, df['Z'] < zmax)]
        if bright_mag is not None:
            bright_lim = abs_mag_r_to_log_solar_L(bright_mag)
            df = df.loc[df['LOGLGAL'] < bright_lim]
        dim_lim = abs_mag_r_to_log_solar_L(dim_mag)
        df = df.loc[df['LOGLGAL'] > dim_lim]

        df.reset_index(drop=True, inplace=True)

        print(f'Volume-limited sample has {len(df):,} galaxies of the original {len(self.all_data):,} galaxies.')
        return df, zmax        


    def get_randoms(self):
        if self.data_cut == 'sv3' or self.data_cut == 'Y3-Kibo-SV3Cut' or self.data_cut == 'Y3-Loa-SV3Cut':
            return get_sv3_randoms_inner()
        else:
            print("Randoms not available for this data cut.")
        
    def get_randoms_mini(self):
        if self.data_cut == 'sv3' or self.data_cut == 'Y3-Kibo-SV3Cut' or self.data_cut == 'Y3-Loa-SV3Cut':
            return get_sv3_randoms_inner_mini()
        else:
            print("Randoms not available for this data cut.")

    def calculate_projected_clustering(self, with_extra_randoms=False):
        raise NotImplemented("This code is old and likely wrong!")

        # TODO weights for fiber collisions corrections

        df = self.all_data
        if df.get('mag_R') is None:
            df['mag_R'] = log_solar_L_to_abs_mag_r(np.log10(df['L_GAL']))
            df['mag_bin'] = np.digitize(df.mag_R, CLUSTERING_MAG_BINS)

        if with_extra_randoms:
            randoms = self.get_randoms() # TODO clustering or full randoms???
        else:
            randoms = self.get_randoms_mini() # TODO clustering or full randoms???

        if randoms is not None:
            print(f"Calculating mini projected clustering: Random count / data count = {len(randoms['RA'])} / {len(df)}")
            if with_extra_randoms:
                self.wp_all_extra = wp.calculate_wp_from_df(df, randoms) 
            else:
                self.wp_all = wp.calculate_wp_from_df(df, randoms)
        else:
            print("Randoms not available for this data cut.")

        serialize(self)

    def calculate_projected_clustering_in_magbins(self, with_extra_randoms=False):
        print("Calculating luminosity dependent clustering...")
        raise NotImplemented("This code is old and likely wrong!")
    
        # BUG Wrong zmax per bin?
        #Mag-5log(h) < -14:  zmax theory=0.01650  zmax obs=0.015029
        #Mag-5log(h) < -15:  zmax theory=0.02595  zmax obs=0.027139
        #Mag-5log(h) < -16:  zmax theory=0.04067  zmax obs=0.043211
        #Mag-5log(h) < -17:  zmax theory=0.06336  zmax obs=0.06597
        #Mag-5log(h) < -18:  zmax theory=0.09792  zmax obs=0.101448
        #Mag-5log(h) < -19:  zmax theory=0.14977  zmax obs=0.154458
        #Mag-5log(h) < -20:  zmax theory=0.22620  zmax obs=0.231856
        #Mag-5log(h) < -21:  zmax theory=0.33694  zmax obs=0.331995
        #Mag-5log(h) < -22:  zmax theory=0.49523  zmax obs=0.459593
        #Mag-5log(h) < -23:  zmax theory=0.72003  zmax obs=0.49971
        zmax = [0.04067, 0.06336, 0.09792, 0.14977, 0.22620, 0.331995, 0.459593, 0.49971]
        #Not doing zmin's for now; no need for indepdenent samples for this

        df = self.all_data
        if df.get('mag_R') is None:
            df['mag_R'] = log_solar_L_to_abs_mag_r(np.log10(df['L_GAL']))
            df['mag_bin'] = np.digitize(df.mag_R, CLUSTERING_MAG_BINS)

        if with_extra_randoms:
            randoms = self.get_randoms()# TODO clustering or full randoms???
        else:
            randoms = self.get_randoms_mini()# TODO clustering or full randoms???

        for i in range(len(CLUSTERING_MAG_BINS)):
            txt = "extra" if with_extra_randoms else "mini"
            if with_extra_randoms and self.wp_slices_extra is not None and self.wp_slices_extra[i] is not None:
                print(f"Skipping already calculated wp mag bin {i} with {txt} randoms...")
                continue
            print(f"Calculating wp for mag bin {i} with {txt} randoms...")
            in_z_range = df.z < zmax[i]
            in_mag_bin = df.mag_bin == i
            mag_min = CLUSTERING_MAG_BINS[i-1] if i > 0 else CLUSTERING_MAG_BINS[i]
            mag_max = CLUSTERING_MAG_BINS[i]
            rows = in_z_range & in_mag_bin

            rbins, wp_a, wp_r, wp_b = wp.calculate_wp_from_df(df.loc[rows], randoms)

            if with_extra_randoms:
                self.wp_slices_extra[i] = (rbins, wp_a, wp_r, wp_b, mag_min, mag_max)
            else:
                self.wp_slices[i] = (rbins, wp_a, wp_r, wp_b, mag_min, mag_max)

            serialize(self)
            
    def add_jackknife_err_to_proj_clustering(self, with_extra_randoms=False, for_mag_bins=False):
        raise NotImplemented("This code is old and likely wrong!")

        if self.data_cut != 'sv3' and self.data_cut != 'Y3-Kibo-SV3Cut' and self.data_cut != 'Y3-Loa-SV3Cut':
            print("Warning: add_jackknife_err_to_proj_clustering called for non-SV3 data cut. Skipping.")
            return
        
        t1 = time.time()
        print("Adding jackknife error to projected clustering...")
        
        df = self.all_data
        if with_extra_randoms:
            randoms = self.get_randoms()
        else:
            randoms = self.get_randoms_mini()

        # label the SV3 region each galaxy is in
        df['region'] = tile_to_region(df['NTID']).astype(int)
        region_ids = df['region'].unique()
        n = len(region_ids)

        def jacknife_iteration(region_idx_to_drop, mag_bin=None):
            if mag_bin is not None:
                in_mag_bin = df.mag_bin == mag_bin
                in_z_range = df.z < zmax[mag_bin]
                alt_df = df.loc[np.all([(df['region'] != region_idx_to_drop), in_mag_bin, in_z_range], axis=0)]
            else:
                alt_df = df.loc[df['region'] != region_idx_to_drop]
            print(f"Jackknife cut out region {region_idx_to_drop} with {len(df)-len(alt_df)} galaxies.")
            return wp.calculate_wp_from_df(alt_df, randoms)

        def cov(realizations):
            mean = np.mean(realizations, axis=0)
            cov = np.zeros((mean.size, mean.size))
            for i in range(n):
                diff = realizations[i] - mean
                cov += np.outer(diff, diff)
            cov *= (n - 1) / n
            return cov

        if for_mag_bins:
            zmax = [0.04067, 0.06336, 0.09792, 0.14977, 0.22620, 0.331995, 0.459593, 0.49971]
            self.wp_slices_cov = np.array(len(CLUSTERING_MAG_BINS) * [None])
            self.wp_slices_r_cov = np.array(len(CLUSTERING_MAG_BINS) * [None])
            self.wp_slices_b_cov = np.array(len(CLUSTERING_MAG_BINS) * [None])
            self.wp_slices_err = np.array(len(CLUSTERING_MAG_BINS) * [None])
            self.wp_slices_r_err = np.array(len(CLUSTERING_MAG_BINS) * [None])
            self.wp_slices_b_err = np.array(len(CLUSTERING_MAG_BINS) * [None])

            for i in range(len(CLUSTERING_MAG_BINS)):
                wp_realizations = np.zeros((len(region_ids), wp.NBINS), dtype=float)
                wp_r_realizations = np.zeros((len(region_ids), wp.NBINS), dtype=float)
                wp_b_realizations = np.zeros((len(region_ids), wp.NBINS), dtype=float)

                j = 0
                for rid in region_ids:
                    rbins, wp_a, wp_r, wp_b = jacknife_iteration(rid, mag_bin=i)
                    wp_realizations[j, :] = wp_a
                    wp_r_realizations[j, :] = wp_r
                    wp_b_realizations[j, :] = wp_b
                    j += 1

                self.wp_slices_cov[i] = cov(wp_realizations)
                self.wp_slices_r_cov[i] = cov(wp_r_realizations)
                self.wp_slices_b_cov[i] = cov(wp_b_realizations)
                self.wp_slices_err[i] = np.sqrt(np.diag(self.wp_slices_cov[i]))
                self.wp_slices_r_err[i] = np.sqrt(np.diag(self.wp_slices_r_cov[i]))
                self.wp_slices_b_err[i] = np.sqrt(np.diag(self.wp_slices_b_cov[i]))

        else:
            wp_realizations = np.zeros((len(region_ids), wp.NBINS), dtype=float)
            wp_r_realizations = np.zeros((len(region_ids), wp.NBINS), dtype=float)
            wp_b_realizations = np.zeros((len(region_ids), wp.NBINS), dtype=float)

            i = 0
            for rid in region_ids:
                rbins, wp_a, wp_r, wp_b = jacknife_iteration(rid)
                wp_realizations[i, :] = wp_a
                wp_r_realizations[i, :] = wp_r
                wp_b_realizations[i, :] = wp_b
                i += 1

            self.wp_cov = cov(wp_realizations)
            self.wp_r_cov = cov(wp_r_realizations)
            self.wp_b_cov = cov(wp_b_realizations)
            self.wp_err = np.sqrt(np.diag(self.wp_cov))
            self.wp_r_err = np.sqrt(np.diag(self.wp_r_cov))
            self.wp_b_err = np.sqrt(np.diag(self.wp_b_cov))

        t2 = time.time()
        print(f"Jackknife error of wp complete in {t2-t1:.2} seconds.")


    def refresh_df_views(self):
        super().refresh_df_views()
        self.add_bootstrapped_f_sat()


def get_extra_bgs_fastspectfit_data():
    fname = OUTPUT_FOLDER + 'bgs_mstar.pkl'
    if os.path.isfile(fname):
        return pickle.load(open(fname, 'rb'))
    else:
        hdul = fits.open(BGS_Y1_FASTSPEC_FILE, memmap=True)
        data = hdul[1].data
        fastspecfit_id = data['TARGETID']
        log_mstar = data['LOGMSTAR'].astype("<f8")
        mstar = np.power(10, log_mstar)
        #Dn4000 = data['DN4000'].astype("<f8")
        hdul.close()

        df = pd.DataFrame({'TARGETID': fastspecfit_id, 'MSTAR': mstar})
        pickle.dump(df, open(fname, 'wb'))
        return df


def filter_SV3_to_avoid_edges(gc: GroupCatalog, INNER_RADIUS = 1.3):
    """
    Take the built group catalog for SV3 and remove galaxies near the edges of the footprint.
    """
    df = gc.all_data
    if gc.data_cut != "sv3":
        print("Warning: filter_SV3_to_avoid_edges called for non-SV3 data cut.")
        return

    print("Creating centered variant of the SV3 catalog...")
    
    # Get distance to nearest SV3 region center point for each galaxy
    gals_coord = coord.SkyCoord(ra=df.RA.to_numpy()*u.degree, dec=df['DEC'].to_numpy()*u.degree, frame='icrs')
    close_array = get_objects_near_sv3_regions(gals_coord, INNER_RADIUS)

    # Filter out galaxies within INNER_RADIUS of any SV3 region center
    inner_df = df.loc[close_array].copy()

    new = copy.deepcopy(gc)
    new.is_centered_version = True
    new.results_file = str.replace(new.results_file, '.pickle', '_cen.pkl')
    new.all_data = inner_df
    new.refresh_df_views()

    print(f"Filtered out {len(df) - len(inner_df)} ({(len(df)-len(inner_df)) / len(df)}) galaxies near the edges of the SV3 footprint.")

    return new

def serialize(gc: GroupCatalog):
    # TODO this is a hack to get around class redefinitions invalidating serialized objects
    # Mess up subclasses?!
    gc.__class__ = eval(gc.__class__.__name__) #reset __class__ attribute
    with open(gc.results_file, 'wb') as f:
        pickle.dump(gc, f)

def deserialize(gc: GroupCatalog):
    gc.__class__ = eval(gc.__class__.__name__) #reset __class__ attribute
    with open(gc.results_file, 'rb') as f:    
        try:
            o: GroupCatalog = pickle.load(f)
            if o.all_data is None:
                print(f"Warning: deserialized object {o.name} has no all_data DataFrame.")
        except:
            print(f"Error deserializing {gc.results_file}")
            o = None
        return o


def drop_SV3_passes(drop_passes: int, tileid: np.ndarray, unobserved: np.ndarray):
    # For SV3 Analysis: remove higher numbered (later observed) tiles from the list in each patch
    # Note this increases the size of the catalog slightly as drop_passes goes up. 
    # It is because some observations are stars or galaxies outside reshift range, which would have been removed.
    # Now they will be in the catalog as unobserved galaxies.
    if drop_passes > 0:
        for patch_number in range(len(sv3_regions_sorted)):
            tilelist = sv3_regions_sorted[patch_number]
            
            # Remove tiles in reverse TILEID order
            for i in np.flip(np.arange(len(tilelist) - drop_passes, len(tilelist))):
                if drop_passes > 0:
                    active_tile = tilelist[i]
                    # TODO We are unsure if this is actually right
                    observed_by_this_tile = tileid == active_tile

                    # Count this tile's observations as unobserved
                    unobserved = np.logical_or(unobserved, observed_by_this_tile)
    
    return unobserved

def add_bsat_column(catalog: GroupCatalog):
    # This replicates what the C Group Finder does.
    bprob = 10
    if 'beta0q' in catalog.GF_props:
        beta0q = catalog.GF_props['beta0q']
        beta0sf = catalog.GF_props['beta0sf']
        betaLq = catalog.GF_props['betaLq']
        betaLsf = catalog.GF_props['betaLsf']
        bprob = np.zeros(len(catalog.all_data))
        bprob = np.where(catalog.all_data['QUIESCENT'], beta0q + betaLq*(catalog.all_data['LOGLGAL']-9.5), beta0sf + betaLsf*(catalog.all_data['LOGLGAL']-9.5))
        bprob = np.where(bprob < 0.001, 0.001, bprob)

    catalog.all_data['BSAT'] = bprob

def add_halo_columns(catalog: GroupCatalog):
    """
    # TODO make work for UCHUU too; need refactoring regarding halo mass property names, etc
    """
    df: pd.DataFrame = catalog.all_data
    
    # Calculate additional halo properties
    if 'mxxl_halo_mass' in df.columns:
        mxxl_masses = df.loc[:, 'mxxl_halo_mass'].to_numpy() * 1E10 * u.solMass
        df.loc[:, 'mxxl_halo_vir_radius_guess'] = get_vir_radius_mine(mxxl_masses)
        # TODO comoving or proper?
        as_per_kpc = get_cosmology().arcsec_per_kpc_proper(df['Z'].to_numpy())
        df.loc[:, 'mxxl_halo_vir_radius_guess_arcsec'] =  df.loc[:, 'mxxl_halo_vir_radius_guess'].to_numpy() * as_per_kpc.to(u.arcsec / u.kpc).value

    masses = df['M_HALO'].to_numpy() * u.solMass # / h        
    df['halo_radius_kpc'] = get_vir_radius_mine(masses) # kpc / h
    # TODO comoving or proper?
    as_per_kpc = get_cosmology().arcsec_per_kpc_proper(df['Z'].to_numpy())
    df['halo_radius_arcsec'] = df['halo_radius_kpc'].to_numpy() * as_per_kpc.to(u.arcsec / u.kpc).value

    catalog.refresh_df_views()
    # Luminosity distance to z_obs
    #df.loc[:, 'ldist_true'] = z_to_ldist(df.z_obs.to_numpy())

def update_properties_for_indices(idx, app_mag_r, app_mag_g, g_r_apparent, z_eff, abs_mag_R, abs_mag_R_k, abs_mag_G, abs_mag_G_k, gmr_best, dn4000_model, log_L_gal, quiescent):
    """
    Updates the properties for the given indices in the arrays. This is used for lost galaxies when we assigned
    redshifts to them generally.
    """
    # Ensure app_mag_r and z_eff are not NaN or Inf at idx
    assert np.isnan(app_mag_r[idx]).any() == False, f"app_mag_r has NaN values at {np.where(np.isnan(app_mag_r))}"
    assert np.isnan(z_eff[idx]).any() == False, f"z_eff has NaN values at {np.where(np.isnan(z_eff))}"
    assert z_eff[idx].all() > 0.001, f"z_eff has too low values at {np.where(z_eff <= 0.001)}"
    np.put(abs_mag_R, idx, app_mag_to_abs_mag(app_mag_r[idx], z_eff[idx]))
    nanindex = np.isnan(abs_mag_R[idx])
    if nanindex.any():
        print(f"Warning: abs_mag_R has NaN values at {np.where(nanindex)}")
        print(f"app_mag_r[idx]: {app_mag_r[idx]}")
        print(f"z_eff[idx]: {z_eff[idx]}")
        print(f"abs_mag_R[idx]: {abs_mag_R[idx]}")
    assert np.isnan(abs_mag_R[idx]).any() == False, f"abs_mag_R[idx] has NaN values at {np.where(np.isnan(abs_mag_R[idx]))}"
    np.put(abs_mag_R_k, idx, k_correct(abs_mag_R[idx], z_eff[idx], g_r_apparent[idx], band='r'))
    np.put(abs_mag_G, idx, app_mag_to_abs_mag(app_mag_g[idx], z_eff[idx]))
    np.put(abs_mag_G_k, idx, k_correct(abs_mag_G[idx], z_eff[idx], g_r_apparent[idx], band='g'))
    np.put(log_L_gal, idx, abs_mag_r_to_log_solar_L(abs_mag_R_k[idx]))
    G_R_k = abs_mag_G_k - abs_mag_R_k
    np.put(gmr_best, idx, G_R_k[idx])
    lookup = dn4000lookup()
    assert np.isnan(abs_mag_R_k[idx]).any() == False, f"abs_mag_R_k[idx] has NaN values at {np.where(np.isnan(abs_mag_R_k[idx]))}"
    assert np.isnan(G_R_k[idx]).any() == False, f"G_R_k[idx] has NaN values at {np.where(np.isnan(G_R_k[idx]))}"
    np.put(dn4000_model, idx, lookup.query(abs_mag_R_k[idx], G_R_k[idx]))
    np.put(quiescent, idx, is_quiescent_BGS_dn4000(log_L_gal[idx], dn4000_model[idx], G_R_k[idx]))

def get_footprint_fraction(data_cut, mode, num_passes_required):
    # These are calculated from randoms in BGS_study.ipynb
    if data_cut == "Y1-Iron":
        # For Y1-Iron  
        FOOTPRINT_FRAC_1pass = 0.1876002 # 7739 deg^2
        FOOTPRINT_FRAC_2pass = 0.1153344 # 4758 deg^2
        FOOTPRINT_FRAC_3pass = 0.0649677 # 2680 deg^2
        FOOTPRINT_FRAC_4pass = 0.0228093 # 940 deg^2
        # 0% 5pass coverage
    elif data_cut == "Y1-Iron-Mini":
        FOOTPRINT_FRAC_1pass = 141.6 / DEGREES_ON_SPHERE 
        FOOTPRINT_FRAC_2pass = 141.6 / DEGREES_ON_SPHERE 
        FOOTPRINT_FRAC_3pass = 141.6 / DEGREES_ON_SPHERE 
    elif data_cut == "Y3-Kibo" or data_cut == "Y3-Loa":
        FOOTPRINT_FRAC_1pass = 0.30968189465008605 # 12775 deg^2
        FOOTPRINT_FRAC_2pass = 0.2859776210215015 # 11797 deg^2
        FOOTPRINT_FRAC_3pass = 0.23324031706784962 # 9621 deg^2
        FOOTPRINT_FRAC_4pass = 0.1148695997866822 # 4738 deg^2
    elif data_cut == "sv3":
        # These are for the 18/20 patches being used. We dropped two due to poor Y3 overlap.
        FOOTPRINT_FRAC_1pass =  156.2628 / DEGREES_ON_SPHERE 
        FOOTPRINT_FRAC_10pass = 124.2812 / DEGREES_ON_SPHERE 
    elif data_cut == "Y3-Kibo-SV3Cut" or data_cut == "Y3-Loa-SV3Cut":
        # Here the data was cut to the SV3 10p footprint. 
        # But the num_passes is now referring to actualy Y3 main survey passes.
        FOOTPRINT_FRAC_1pass = 124.2812 / DEGREES_ON_SPHERE 
        # Not computed for other situations.
    else:
        print("Invalid data cut. Exiting.")
        exit(2)

    # TODO update footprint with new calculation from ANY. It shouldn't change.
    if mode == Mode.ALL.value or num_passes_required == 1:
        return FOOTPRINT_FRAC_1pass
    elif num_passes_required == 2:
        return FOOTPRINT_FRAC_2pass
    elif num_passes_required == 3:
        return FOOTPRINT_FRAC_3pass
    elif num_passes_required == 4:
        return FOOTPRINT_FRAC_4pass
    elif num_passes_required == 10:
        return FOOTPRINT_FRAC_10pass
    else:
        print(f"Need footprint calculation for num_passes_required = {num_passes_required}. Exiting")
        exit(2)        
        

def pre_process_BGS(fname, mode, outname_base, APP_MAG_CUT, CATALOG_APP_MAG_CUT, sdss_fill, num_passes_required, drop_passes, data_cut, extra_params):
    """
    Pre-processes the BGS data for use with the group finder.
    """
    Z_MIN = 0.001 # BUG The Group Finder blows up if you lower this
    Z_MAX = 0.5

    # TODO BUG One galaxy is lost from this to group finder...
    
    print("Reading data from ", fname)
    # Unobserved galaxies have masked rows in appropriate columns of the table
    table = Table.read(fname, format='fits')

    if drop_passes > 0 and data_cut != "sv3":
        raise ValueError("Dropping passes is only for the sv3 study")
    if extra_params is not None and Mode.is_photoz_plus(mode) == False:
        raise ValueError("Extra parameters are only for the PHOTOZ_PLUS_v1 mode.")

    frac_area = get_footprint_fraction(data_cut, mode, num_passes_required)
    wants_MCMC = False

    if mode == Mode.ALL.value:
        print("\nMode ALL NOT SUPPORTED DUE TO FIBER INCOMPLETENESS")
        exit(2)
    elif mode == Mode.FIBER_ASSIGNED_ONLY.value:
        print(f"\nMode FIBER ASSIGNED ONLY {num_passes_required}+ PASSES")
    elif mode == Mode.NEAREST_NEIGHBOR.value:
        print("\nMode NEAREST_NEIGHBOR")
    elif mode == Mode.FANCY.value:
        print("\nMode FANCY NOT SUPPORTED")
        exit(2)
    elif mode == Mode.SIMPLE.value:
        print("\nMode SIMPLE v2")
    elif mode == Mode.SIMPLE_v4.value:
        print("\nMode SIMPLE v4")
    elif mode == Mode.SIMPLE_v5.value:
        print("\nMode SIMPLE v5")
    elif Mode.is_photoz_plus(mode):
        print("\nMode PHOTOZ PLUS")
        if extra_params is not None:
            if len(extra_params) == 2:
                NEIGHBORS, BB_PARAMS = extra_params
                print("Using one set of extra parameter values for all color combinations.")
                params = (BB_PARAMS, BB_PARAMS, BB_PARAMS, BB_PARAMS)
            elif len(extra_params) == 5:
                NEIGHBORS, BB_PARAMS, RB_PARAMS, BR_PARAMS, RR_PARAMS = extra_params
                params = (BB_PARAMS, RB_PARAMS, BR_PARAMS, RR_PARAMS)
            elif len(extra_params) == 6:
                print("Using new Lorentzian parameter set")
                NEIGHBORS = extra_params[0]
                params = extra_params[1:]
            elif len(extra_params) == 13:
                NEIGHBORS = extra_params[0]
                BB_PARAMS, RB_PARAMS, BR_PARAMS, RR_PARAMS = extra_params[1:].reshape(4, 3)
                params = (BB_PARAMS, RB_PARAMS, BR_PARAMS, RR_PARAMS)
            else:
                raise ValueError("Extra parameters must be a tuple of length 2 or 5")
            NEIGHBORS = int(NEIGHBORS)
        else:
            print("Extra parameters not provided for PHOTOZ_PLUS mode; will MCMC them.")
            NEIGHBORS = 10
            wants_MCMC = True
    else:
        print("Invalid mode. Exiting.")
        exit(2)


    # Some versions of the LSS Catalogs use astropy's Table used masked arrays for unobserved spectral targets    
    if np.ma.is_masked(table['Z']):
        z_obs = table['Z'].data.data
        unobserved = table['Z'].mask # the masked values are what is unobserved
        no_truth_z = unobserved.copy()
    else:
        # SV3 version didn't do this
        z_obs = table['Z']
        unobserved = table['Z'].astype("<i8") >  100
        no_truth_z = unobserved.copy()

    obj_type = get_tbl_column(table, 'SPECTYPE')
    deltachi2 = get_tbl_column(table, 'DELTACHI2')
    ff_g = get_tbl_column(table, 'FRACFLUX_G')
    ff_r = get_tbl_column(table, 'FRACFLUX_R')
    ff_z = get_tbl_column(table, 'FRACFLUX_Z')
    dec = get_tbl_column(table, 'DEC', required=True)
    ra = get_tbl_column(table, 'RA', required=True)
    maskbits = get_tbl_column(table, 'MASKBITS')
    ref_cat = get_tbl_column(table, 'REF_CAT')
    tileid = get_tbl_column(table, 'TILEID')
    target_id = get_tbl_column(table, 'TARGETID')
    ntid = get_tbl_column(table, 'NEAREST_TILEIDS')
    if ntid is not None:
        ntid = ntid[:,0] # just need to nearest tile for our purposes
    app_mag_r = get_tbl_column(table, 'APP_MAG_R', required=True)
    app_mag_g = get_tbl_column(table, 'APP_MAG_G', required=True)
    g_r_apparent = app_mag_g - app_mag_r
    abs_mag_R = get_tbl_column(table, 'ABS_MAG_R', required=True)
    abs_mag_R_k = get_tbl_column(table, 'ABS_MAG_R_K', required=True)
    abs_mag_G = get_tbl_column(table, 'ABS_MAG_G', required=True)
    abs_mag_G_k = get_tbl_column(table, 'ABS_MAG_G_K', required=True)
    gmr_best = get_tbl_column(table, 'G_R_BEST', required=True) # absolute G-R using fastspecfit k-corr when possible or polynomial for unobserved
    log_L_gal = get_tbl_column(table, 'LOG_L_GAL', required=True)
    quiescent = get_tbl_column(table, 'QUIESCENT', required=True)
    p_obs = get_tbl_column(table, 'PROB_OBS')
    z_sv3 = get_tbl_column(table, 'Z_SV3')
    if p_obs is None:
        print("WARNING: PROB_OBS column not found in FITS file. Using 0.689 for all unobserved galaxies.")
        p_obs = np.ones(len(z_obs)) * 0.689
    nan_pobs = np.isnan(p_obs)
    if np.any(nan_pobs):
        print(f"WARNING: {np.sum(nan_pobs)} galaxies have nan p_obs. Setting those to 0.689, the mean of Y3.")
        p_obs[nan_pobs] = 0.689
    z_phot = get_tbl_column(table, 'Z_PHOT')
    if z_phot is None:
        print("WARNING: Z_PHOT column not found in FITS file. Will be set to nan for all.")
        z_phot = np.ones(len(z_obs)) * np.nan
    dn4000_model = get_tbl_column(table, 'DN4000_MODEL')
    if dn4000_model is None:
        print("WARNING: DN4000_MODEL column not found in FITS file. Will be set to 0 for all.")
        dn4000_model = np.zeros(len(z_obs))

    # A manually curated list of bad targets, usually from visual inspection of images
    bad_targets = [39627705590745283, 39628011489723373]

    # For SV3 Analysis we can pretend to not have observed some galaxies
    # This procedure is really accurate and doesn't produce a main-like situation. 
    # Instead for our fiber incompleteness analysis, we take Y3 and cut to SV3 isntead.
    # So this code path is not critical anymore.
    if data_cut == "sv3":
        unobserved = drop_SV3_passes(drop_passes, tileid, unobserved)
    if data_cut == "Y3-Loa-SV3Cut" and wants_MCMC:
        if z_sv3 is None:
            print("ERROR: to run PZP MCMC on Y3-Loa-SV3Cut, you need to provide the Z_SV3 column.")
            exit(2)
        # Copy over SV3 redshifts to the unobserved galaxies for use in the MCMC
        sv3_z_missing = np.isnan(z_sv3) | (z_sv3 > 100)
        no_truth_z = sv3_z_missing & unobserved 
        z_obs[unobserved] = z_sv3[unobserved]

    orig_count = len(dec)
    print(f"{orig_count:,} objects in file")

    # If an observation was made, some automated system will evaluate the spectra and auto classify the SPECTYPE
    # as GALAXY, QSO, STAR. It is null (and masked) for non-observed targets.
    # NTILE tracks how many DESI pointings could have observed the target (at fiber level)
    # NTILE_MINE gives how many tiles include just from inclusion in circles drawn around tile centers
    # null values (masked rows) are unobserved targets; not all columns are masked though

    ##############################################################################
    # Make filter arrays (True/False values)
    ##############################################################################
    multi_pass_filter = table['NTILE_MINE'] >= num_passes_required
    galaxy_observed_filter = obj_type == b'GALAXY'
    app_mag_filter = app_mag_r < APP_MAG_CUT
    redshift_filter = z_obs > Z_MIN
    redshift_hi_filter = z_obs < Z_MAX
    deltachi2_filter = deltachi2 > 40 # Ensures that there wasn't another z with similar likelihood from the z fitting code
    bad_targets_filter = ~np.isin(target_id, bad_targets)
    
    # Special version cut to look like SV3 - choose only the ones inside the SV3 footprint
    if data_cut == "Y3-Kibo-SV3Cut" or data_cut == "Y3-Loa-SV3Cut":
        ntid_sv3 = get_tbl_column(table, 'NEAREST_TILEIDS_SV3', required=True)[:,0] # just need to nearest tile for our purposes
        region = tile_to_region(ntid_sv3)
        to_remove = np.isin(region, sv3_poor_y3overlap)
        in_good_sv3regions = ~to_remove
        multi_pass_filter = np.all([multi_pass_filter, table['NTILE_MINE_SV3'] >= 10, in_good_sv3regions], axis=0)

    # For Y1-Iron-Mini, we want to cut to a smaller region
    if data_cut == "Y1-Iron-Mini":
        multi_pass_filter &= ra > 160
        multi_pass_filter &= ra < 175
        multi_pass_filter &= dec > -7
        multi_pass_filter &= dec < 3

    # Roughly remove HII regions of low z, high angular size galaxies (SGA catalog)
    if maskbits is not None and ref_cat is not None:
        sga_collision = (maskbits & MASKBITS['GALAXY']) != 0
        sga_central = ref_cat == b'L3'
        to_remove_blue = sga_collision & ~sga_central & (g_r_apparent < 0.8)
        print(f"{np.sum(to_remove_blue):,} galaxies ({np.sum(to_remove_blue) / len(dec) * 100:.2f}%) have a SGA collision, are not SGA centrals, and are blue enough to remove.")
        no_SGA_Issues = np.invert(to_remove_blue)
    else:
        no_SGA_Issues = np.ones(len(dec), dtype=bool)

    # Fiberflux cuts, too remove confusing overlapping objects which likely have bad spectra.
    ff_req = np.ones(len(dec), dtype=bool)
    if ff_g is not None and ff_r is not None and ff_z is not None:
       FF_CUT = 0.5 # LOW Z folks used 0.35 for this in target selection. LSSCats don't cut on it at all. 
       ff_g_req = np.logical_or(ff_g < FF_CUT, np.isnan(ff_g))
       ff_r_req = np.logical_or(ff_r < FF_CUT, np.isnan(ff_r))
       ff_z_req = np.logical_or(ff_z < FF_CUT, np.isnan(ff_z))
       ff_req = np.sum([ff_g_req, ff_r_req, ff_z_req], axis=0) >= 2 # Two+ bands with low enough fracflux required
       print(f"{np.sum(~ff_req):,} galaxies ({np.sum(~ff_req) / len(dec) * 100:.2f}%) have fracflux in two bands too high to keep.")

    observed_requirements = np.all([galaxy_observed_filter, app_mag_filter, redshift_filter, redshift_hi_filter, deltachi2_filter, no_SGA_Issues, ff_req], axis=0)

    # treat low deltachi2 as unobserved. Must pass the photometric quality control still.
    treat_as_unobserved = np.all([galaxy_observed_filter, app_mag_filter, no_SGA_Issues, ff_req, np.invert(deltachi2_filter)], axis=0)
    #print(f"We have {np.count_nonzero(treat_as_unobserved)} observed galaxies with deltachi2 < 40 to add to the unobserved pool")
    unobserved = np.all([app_mag_filter, np.logical_or(unobserved, treat_as_unobserved)], axis=0)

    if mode == Mode.FIBER_ASSIGNED_ONLY.value:
        keep = np.all([bad_targets_filter, multi_pass_filter, observed_requirements], axis=0)

    if mode == Mode.NEAREST_NEIGHBOR.value or Mode.is_simple(mode) or Mode.is_photoz_plus(mode):
        keep = np.all([bad_targets_filter, multi_pass_filter, np.logical_or(observed_requirements, unobserved)], axis=0)

        # Filter down inputs to the ones we want in the catalog for NN and similar calculations
        # TODO why bother with this for the real data? Use all we got, right? 
        # I upped the cut to 21 so it doesn't do anything
        catalog_bright_filter = app_mag_r < CATALOG_APP_MAG_CUT 
        catalog_keep = np.all([galaxy_observed_filter, catalog_bright_filter, redshift_filter, redshift_hi_filter, deltachi2_filter, no_SGA_Issues, ~unobserved], axis=0)
        catalog_ra = ra[catalog_keep]
        catalog_dec = dec[catalog_keep]
        z_obs_catalog = z_obs[catalog_keep]
        catalog_quiescent = quiescent[catalog_keep]
        print(f"{len(z_obs_catalog):,} galaxies in the neighbor catalog.")

    # Apply filters
    obj_type = obj_type[keep]
    dec = dec[keep]
    ra = ra[keep]
    z_obs = z_obs[keep]
    target_id = target_id[keep]
    app_mag_r = app_mag_r[keep]
    app_mag_g = app_mag_g[keep]
    p_obs = p_obs[keep]
    deltachi2 = deltachi2[keep]
    g_r_apparent = g_r_apparent[keep]
    abs_mag_R = abs_mag_R[keep]
    abs_mag_R_k = abs_mag_R_k[keep]
    abs_mag_G = abs_mag_G[keep]
    abs_mag_G_k = abs_mag_G_k[keep]
    gmr_best = gmr_best[keep]
    log_L_gal = log_L_gal[keep]
    quiescent = quiescent[keep]
    dn4000_model = dn4000_model[keep]
    ntid = ntid[keep]
    z_phot = z_phot[keep]
    unobserved = unobserved[keep]
    no_truth_z = no_truth_z[keep]

    observed = np.invert(unobserved)
    idx_unobserved = np.flatnonzero(unobserved)
    z_assigned_flag = np.zeros(len(z_obs), dtype=np.int32)

    count = len(dec)
    print(f"{count:,} galaxies left for main catalog after filters.")
    first_need_redshift_count = unobserved.sum()
    print(f'{first_need_redshift_count} ({100*first_need_redshift_count / len(unobserved) :.1f})% need redshifts')

    z_eff = np.copy(z_obs)

    ############################################################################
    # If a lost galaxy matches the SDSS catalog, grab it's redshift and use that
    ############################################################################
    if unobserved.sum() > 0 and sdss_fill:
        sdss_vanilla = deserialize(SDSSGroupCatalog("SDSS Vanilla v2", SDSS_v2_DAT_FILE, SDSS_v2_GALPROPS_FILE, {'frac_area': 0.179}))
        if sdss_vanilla.all_data is not None:
            sdss_has_specz = z_flag_is_spectro_z(sdss_vanilla.all_data.Z_ASSIGNED_FLAG)
            observed_sdss = sdss_vanilla.all_data.loc[sdss_has_specz]

            sdss_catalog = coord.SkyCoord(ra=observed_sdss.RA.to_numpy()*u.degree, dec=observed_sdss['DEC'].to_numpy()*u.degree, frame='icrs')
            to_match = coord.SkyCoord(ra=ra[idx_unobserved]*u.degree, dec=dec[idx_unobserved]*u.degree, frame='icrs')
            print(f"Matching {len(to_match):,} lost galaxies to {len(sdss_catalog):,} SDSS galaxies")
            idx, d2d, d3d = coord.match_coordinates_sky(to_match, sdss_catalog, nthneighbor=1, storekdtree=False)
            ang_dist = d2d.to(u.arcsec).value
            sdss_z = sdss_vanilla.all_data.iloc[idx]['Z'].to_numpy()

            # if angular distance is < 1", then we consider it a match to SDSS catalog and copy over it's z
            ANGULAR_DISTANCE_MATCH_OLD = 3.0
            ANGULAR_DISTANCE_MATCH = 1.0
            matched_old = ang_dist < ANGULAR_DISTANCE_MATCH_OLD
            matched = ang_dist < ANGULAR_DISTANCE_MATCH

            print(f"{matched.sum():,} of {first_need_redshift_count:,} lost galaxies matched to SDSS catalog (would have matched {matched_old.sum():,} with 3\")")

            # If sloan z is very different from z_phot, we should probably not use it
            # This is a bit of a hack to avoid using the SDSS z for galaxies that are likely to be wrong
            # TODO but what if it's just a really bad photo-z? Spot check some of these...
            matched = np.logical_and(matched, np.abs(z_phot[idx_unobserved] - sdss_z) < 0.1)
            print(f"{matched.sum():,} are reasonable matches given the photo-z.")
            
            z_eff[idx_unobserved] = np.where(matched, sdss_z, np.nan) # Set to SDSS redshift of nan if not matched
            z_assigned_flag[idx_unobserved] = np.where(matched, AssignedRedshiftFlag.SDSS_SPEC.value, -4) 
            
            # Take Dn4000 and QUIESCENT from SDSS, keep the magnitude related things as DESI calculations
            idx_from_sloan = idx_unobserved[matched]
            update_properties_for_indices(idx_from_sloan, app_mag_r, app_mag_g, g_r_apparent, z_eff, abs_mag_R, abs_mag_R_k, abs_mag_G, abs_mag_G_k, gmr_best, dn4000_model, log_L_gal, quiescent)
            np.put(dn4000_model, idx_from_sloan, observed_sdss.iloc[idx]['DN4000'].to_numpy())
            np.put(quiescent, idx_from_sloan, observed_sdss.iloc[idx]['QUIESCENT'].to_numpy())
            unobserved[idx_unobserved] = np.where(matched, False, unobserved[idx_unobserved])
            observed = np.invert(unobserved)
            idx_unobserved = np.flatnonzero(unobserved)
     
            print(f"{matched.sum():,} of {first_need_redshift_count:,} redshifts taken from SDSS.")
            print(f"{unobserved.sum():,} remaining galaxies need redshifts.")
        else:
            print("No SDSS catalog to match to. Skipping.")

    ##################################################################################################
    # Now, depending on the mode chosen, we will assign redshifts to the remaining unobserved galaxies
    ##################################################################################################

    if mode == Mode.NEAREST_NEIGHBOR.value:
        print("Getting nearest neighbor redshifts...")
        catalog = coord.SkyCoord(ra=catalog_ra*u.degree, dec=catalog_dec*u.degree, frame='icrs')
        to_match = coord.SkyCoord(ra=ra[idx_unobserved]*u.degree, dec=dec[idx_unobserved]*u.degree, frame='icrs')
        idx, d2d, d3d = coord.match_coordinates_sky(to_match, catalog, storekdtree=False)
        z_eff[idx_unobserved] = z_obs_catalog[idx]
        z_assigned_flag[idx_unobserved] = 1
        print("Copying over nearest neighbor properties complete.")

    target_quiescent = quiescent[idx_unobserved]
    q_missing = np.isnan(target_quiescent)
    if np.any(q_missing):
        print(f"WARNING: Quiescent missing for {np.sum(q_missing):,} galaxies. Using g-r color without k-corrections to guess.")
        target_quiescent = np.where(np.isnan(target_quiescent), is_quiescent_lost_gal_guess(app_mag_g[idx_unobserved] - app_mag_r[idx_unobserved]).astype(int), target_quiescent)

    if Mode.is_simple(mode):
        if mode == Mode.SIMPLE.value:
            ver = '2.0'
        elif mode == Mode.SIMPLE_v4.value:
            ver = '4.0'
        elif mode == Mode.SIMPLE_v5.value:
            ver = '5.0'
        
        with SimpleRedshiftGuesser(app_mag_r[observed], z_eff[observed], ver) as scorer: 
            print(f"Assigning missing redshifts... ")   
            catalog = coord.SkyCoord(ra=catalog_ra*u.degree, dec=catalog_dec*u.degree, frame='icrs')
            to_match = coord.SkyCoord(ra=ra[idx_unobserved]*u.degree, dec=dec[idx_unobserved]*u.degree, frame='icrs')

            # neighbor_indexes is the index of the nearest galaxy in the catalog arrays
            neighbor_indexes, d2d, d3d = coord.match_coordinates_sky(to_match, catalog, storekdtree=False)
            ang_dist = d2d.to(u.arcsec).value

            assert len(target_quiescent) == len(ang_dist)

            chosen_z, isNN = scorer.choose_redshift(z_obs_catalog[neighbor_indexes], ang_dist, p_obs[idx_unobserved], app_mag_r[idx_unobserved], target_quiescent, catalog_quiescent[neighbor_indexes])
            z_eff[idx_unobserved] = chosen_z
            z_assigned_flag[idx_unobserved] = np.where(isNN, AssignedRedshiftFlag.NEIGHBOR_ONE.value, AssignedRedshiftFlag.PSEUDO_RANDOM.value)
            print(f"Assigning missing redshifts complete.")   

    if Mode.is_photoz_plus(mode):
        with PhotometricRedshiftGuesser.from_files(BGS_Y3_LOST_APP_TO_Z_FILE, BGS_Y3_LOST_APP_AND_ZPHOT_TO_Z_FILE, NEIGHBOR_ANALYSIS_SV3_BINS_SMOOTHED_FILE_V2, Mode(mode)) as scorer:
            print(f"Assigning missing redshifts... ")   

            if wants_MCMC:
               MAX_NEIGHBORS = 20
            else:
                MAX_NEIGHBORS = NEIGHBORS

            catalog = coord.SkyCoord(ra=catalog_ra*u.degree, dec=catalog_dec*u.degree, frame='icrs')
            to_match = coord.SkyCoord(ra=ra[idx_unobserved]*u.degree, dec=dec[idx_unobserved]*u.degree, frame='icrs')
           
            shape = (MAX_NEIGHBORS, len(to_match))
            neighbor_indexes = np.zeros(shape, dtype=np.int64)
            n_z = np.zeros(shape, dtype=np.float64)
            ang_dist = np.zeros(shape, dtype=np.float64)
            n_q = np.zeros(shape, dtype=np.float64)

            for n in range(MAX_NEIGHBORS):
                neighbor_indexes[n, :], d2d, d3d = coord.match_coordinates_sky(to_match, catalog, nthneighbor=n+1, storekdtree='nn_kdtree')
                ang_dist[n, :] = d2d.to(u.arcsec).value
                n_z[n, :] = z_obs_catalog[neighbor_indexes[n, :]]
                n_q[n, :] = catalog_quiescent[neighbor_indexes[n, :]]

            if wants_MCMC:
                if data_cut != "sv3" or drop_passes == 0:
                    if data_cut != 'Y3-Loa-SV3Cut' or z_sv3 is None:
                        raise ValueError("MCMC optimization of parameters is only possible with SV3 and dropped passes or Y3-Loa-SV3Cut data supplemented with SV3 redshifts.")

                print("Performing MCMC optimization of PhotometricRedshiftGuesser parameters")
                # Can only use the galaxies that were observed but we're pretending are unobserved 
                idx =  np.flatnonzero(np.logical_and(~no_truth_z, unobserved))
                # from the neighbor arrays, need to discard the ones that are not in the idx
                n_selector = (~no_truth_z)[unobserved] # True/False array of the ones that were observed but we're pretending are unobserved of length idx_unobserved
                
                NEIGHBORS, params = find_optimal_parameters_mcmc(scorer, mode, app_mag_r[idx], p_obs[idx], z_phot[idx], target_quiescent[n_selector], ang_dist[:, n_selector], n_z[:, n_selector], n_q[:, n_selector], z_obs[idx])
                NEIGHBORS = int(NEIGHBORS)
                print(f"Best params found: N={NEIGHBORS}, p={params}")
                #a, b = find_optimal_parameters(scorer, app_mag_r[idx], p_obs[idx], z_phot[idx], target_quiescent[n_selector], ang_dist[:, n_selector], n_z[:, n_selector], n_q[:, n_selector], z_obs[idx])
                #pickle.dump(params, open(OUTPUT_FOLDER + "photoz_plus_params_v1_4.pkl", "wb"))
            else :
                #A_WEIGHT, B_WEIGHT = pickle.load(open(OUTPUT_FOLDER + "photoz_plus_params.pkl", "rb"))
                print(f"Using given parameters: N={NEIGHBORS}, p={params}")

            chosen_z, flag_value = scorer.choose_redshift(n_z[:NEIGHBORS, :], ang_dist[:NEIGHBORS, :], z_phot[idx_unobserved], p_obs[idx_unobserved], app_mag_r[idx_unobserved], target_quiescent, n_q[:NEIGHBORS, :], params)
            z_eff[idx_unobserved] = chosen_z
            z_assigned_flag[idx_unobserved] = flag_value

            print(f"Assigning missing redshifts complete.")   
    
    assert np.isnan(z_eff).sum() == 0


    print(f"Neighbor usage %: {z_flag_is_neighbor(z_assigned_flag).sum() / z_flag_is_not_spectro_z(z_assigned_flag).sum() * 100:.2f}")

    # Now that we have redshifts for lost galaxies, we can calculate the rest of the properties
    if len(idx_unobserved) > 0:
        update_properties_for_indices(idx_unobserved, app_mag_r, app_mag_g, g_r_apparent, z_eff, abs_mag_R, abs_mag_R_k, abs_mag_G, abs_mag_G_k, gmr_best, dn4000_model, log_L_gal, quiescent)

    print(f"Catalog contains {quiescent.sum():,} quiescent and {len(quiescent) - quiescent.sum():,} star-forming galaxies")

    # the vmax should be calculated from un-k-corrected magnitudes
    V_max = get_max_observable_volume(abs_mag_R, z_eff, APP_MAG_CUT, frac_area)

    # TODO get galaxy concentration from somewhere
    chi = np.zeros(count, dtype=np.int32) 

    ####################################################################################
    # FINAL QUALITY CONTROL
    ####################################################################################
    
    # Redshift assignments could have placed lost galaxies outside the range of the catalog. Remove them.
    qa1 = np.logical_and(z_eff > Z_MIN, z_eff < Z_MAX)
    print(f"{np.sum(~qa1):,} galaxies have redshifts outside the range of the catalog and will be removed.")

    # Ensure implied luminosity (for assigned z only) isn't totally crazy. If so, remove them.
    L_GAL_MAX = np.log10(4e11)
    qa2 = ~np.logical_and(log_L_gal > L_GAL_MAX, unobserved)
    print(f"{np.sum(~qa2):,} unobserved galaxies have implied log(L_gal) > {L_GAL_MAX:.2f} and will be removed.")

    qa3 = ~np.isnan(log_L_gal)
    print(f"{np.sum(~qa3):,} galaxies have nan log_L_gal and will be removed.")

    assert np.all(z_assigned_flag >= -3), "z_assigned_flag is unset for some targets."

    final_selection = np.all([qa1, qa2, qa3], axis=0)

    print(f"Final Catalog Size: {np.sum(final_selection):,}.")

    ####################################################################################
    # Write the completed preprocess files for the group finder / post-processing to use
    ####################################################################################
    t1 = time.time()
    galprops= pd.DataFrame({
        'APP_MAG_R': app_mag_r[final_selection].astype("<f8"),
        'TARGETID': target_id[final_selection].astype("<i8"),
        'Z_ASSIGNED_FLAG': z_assigned_flag[final_selection].astype("<i4"),
        'G_R': gmr_best[final_selection].astype("<f8"), # this is k corrected to z=0.1
        'DN4000_MODEL': dn4000_model[final_selection].astype("<f8"),
        'NTID': ntid[final_selection].astype("<i8"),
        'Z_PHOT': z_phot[final_selection].astype("<f8"),
        'Z_OBS': z_obs[final_selection].astype("<f8"),
        'QUIESCENT': quiescent[final_selection].astype("bool"), 
    })
    galprops.to_pickle(outname_base + "_galprops.pkl")  
    t2 = time.time()
    print(f"Galprops pickling took {t2-t1:.4f} seconds")

    write_dat_files(ra[final_selection], dec[final_selection], z_eff[final_selection], log_L_gal[final_selection], V_max[final_selection], quiescent[final_selection], chi[final_selection], outname_base)

    return outname_base + ".dat", {'zmin': np.min(z_eff[final_selection]), 'zmax': np.max(z_eff[final_selection]), 'frac_area': frac_area }


n_range = [1, 20.99999] # we floor this value for the num neighbors to use
a_range = [0.0, 2.5]
b_range = [0.0, 4.0]
s_range = [1.5, 7.0]
#p_range = [1.0, 8.0]

def log_prior(params):
    n = params[0]
    if not n_range[0] <= n <= n_range[1]:
        return -np.inf
    bb, rb, br, rr = np.reshape(params[1:], (4,3))
    ranges = [a_range, b_range, s_range]#, p_range]
    for param_set in [bb, rb, br, rr]:
        if not all(low <= param <= high for param, (low, high) in zip(param_set, ranges)):
            return -np.inf  # log(0)
    return 0.0  # log(1), all good

def log_likelihood(params, scorer: PhotometricRedshiftGuesser, app_mag_r, p_obs, z_phot, t_q, ang_dist, n_z, n_q, z_truth):
    n = math.floor(params[0])
    bb, rb, br, rr = np.reshape(params[1:], (4,3))    
    chosen_z, assignment_type = scorer.choose_redshift(n_z[:n, :], ang_dist[:n, :], z_phot, p_obs, app_mag_r, t_q, n_q[:n, :], (bb, rb, br, rr))
    score = photoz_plus_metric_4(chosen_z, z_truth, assignment_type)
    return -score  # Negative because we want to maximize the likelihood

def log_probability(params, scorer, app_mag_r, p_obs, z_phot, t_q, ang_dist, n_z, n_q, z_truth):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf # throw away this possibility
    return lp + log_likelihood(params, scorer, app_mag_r, p_obs, z_phot, t_q, ang_dist, n_z, n_q, z_truth)

def find_optimal_parameters_mcmc(scorer: PhotometricRedshiftGuesser, mode, app_mag_r, p_obs, z_phot, t_q, ang_dist, n_z, n_q, z_truth):
    assert len(app_mag_r) == len(p_obs) == len(z_phot) == len(t_q) == len(ang_dist[0]) == len(n_z[0]) == len(n_q[0]) == len(z_truth)
    
    if mode == Mode.PHOTOZ_PLUS_v4.value:
        ndim = 5
        n_walkers = 26
        n_steps = 20000
        new_a_range = [0.5, 2.0]
        new_b_range = [0.5, 2.0]
        # Create random arrays of length 5 of values for each walker
        # Range of index 0 new_a_range, rest is new_b_range
        pos = np.array([np.random.uniform(low=[new_a_range[0], new_b_range[0], s_range[0], 0.0, 0.0],
                                           high=[new_a_range[1], new_b_range[1], s_range[1], 2.5, 4.0]) for i in range(n_walkers*4)])
        pos = pos.reshape(n_walkers, 5)
        pos = np.insert(pos, 0, np.arange(n_walkers)%20 +1, axis=1)

        print(f"Initial positions: {pos}")

    else:
        ndim = 13
        n_walkers=ndim*2
        n_steps=20000
        pos = np.array([np.random.uniform(low=[a_range[0], b_range[0], s_range[0]], 
                                            high=[a_range[1], b_range[1], s_range[1]]) for i in range(n_walkers*4)])
        pos = pos.reshape(n_walkers, 12)
        pos = np.insert(pos, 0, np.arange(n_walkers)%20 +1, axis=1)

        # Get score_b values cached and ready for fast access
        scorer.use_score_cache_for_mcmc(len(app_mag_r))
        _ = scorer.choose_redshift(n_z, ang_dist, z_phot, p_obs, app_mag_r, t_q, n_q, np.reshape(pos[0, 1:], (4,3)))

        # Insert some manual favorites as ICs; these came from past MCMC runs
        for i in range(20):
            pos[i] = [i + 1, 1.2938, 1.5467, 3.0134, 1.2229, 0.8628, 2.5882, 0.8706, 0.6126, 2.4447, 1.1163, 1.2938, 3.1650]
            if i > 0:
                pos[i] *= np.random.uniform(0.90, 0.999, size=13)
        

    if mode == Mode.PHOTOZ_PLUS_v1.value:
        backfile = BASE_FOLDER + "mcmc13_m4_1_7.h5"
    elif mode == Mode.PHOTOZ_PLUS_v2.value:
        backfile = BASE_FOLDER + "mcmc13_m4_2_6.h5"
    elif mode == Mode.PHOTOZ_PLUS_v3.value:
        backfile = BASE_FOLDER + "mcmc13_m4_3_1.h5"
    elif mode == Mode.PHOTOZ_PLUS_v4.value:
        backfile = BASE_FOLDER + "mcmc6_m4_4_1.h5"
    if os.path.exists(backfile):
        print("Loaded existing MCMC sampler")
        backend = emcee.backends.HDFBackend(backfile)
        n_walkers = backend.shape[0]
    else:
        backend = emcee.backends.HDFBackend(backfile)
        backend.reset(n_walkers, ndim)

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability, args=(scorer, app_mag_r, p_obs, z_phot, t_q, ang_dist, n_z, n_q, z_truth), backend=backend, pool=Pool())

    print("Running MCMC...")
    sampler.run_mcmc(pos, n_steps, progress=True)

    samples = sampler.get_chain(flat=True)
    params = samples[np.argmax(sampler.get_log_prob())]
    return int(params[0]), params[1:] # TODO wrong but we usually never get here


##########################
# Processing Group Finder Output File
##########################

def read_and_combine_gf_output(gc: GroupCatalog, galprops_df):
    # TODO instead of reading GF output from disk, have option to just keep in memory
    main_df = pd.read_csv(gc.GF_outfile, delimiter=' ', names=
                          ('RA', 'DEC', 'Z', 'L_GAL', 'VMAX', 'P_SAT', 'M_HALO', 'N_SAT', 'L_TOT', 'IGRP', 'WEIGHT', 'CHI1_WEIGHT'),
                          dtype={'RA': np.float64, 'DEC': np.float64, 'Z': np.float64, 'L_GAL': np.float64, 'VMAX': np.float64,
                                 'P_SAT': np.float64, 'M_HALO': np.float64, 'N_SAT': np.int32, 'L_TOT': np.float64, 'IGRP': np.int64, 'WEIGHT': np.float64, 'CHI1_WEIGHT': np.float64})
    df = pd.merge(main_df, galprops_df, left_index=True, right_index=True, validate='1:1')

    # Drop bad data, should have been cleaned up earlier though!
    orig_count = len(df)
    df = df[df['M_HALO'] != 0]
    new_count = len(df)
    if (orig_count != new_count):
        print("WARNING: Dropped {0} bad galaxies".format(orig_count - new_count))

    # add columns indicating if galaxy is a satellite
    df['IS_SAT'] = (df.index != df['IGRP']).astype(bool)
    df['LOGLGAL'] = np.log10(df['L_GAL'])

    # add column for halo mass bins and Lgal bins
    df['Mh_bin'] = pd.cut(x = df['M_HALO'], bins = gc.Mhalo_bins, labels = gc.Mhalo_labels, include_lowest = True)
    df['LGAL_BIN'] = pd.cut(x = df['L_GAL'], bins = gc.L_gal_bins, labels = gc.L_gal_labels, include_lowest = True)

    return df # TODO update callers



##########################
# Aggregation Helpers
##########################

def count_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        return len(series) / np.average(series['VMAX'])

def fsat_truth_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        return np.average(series['IS_SAT_T'], weights=1/series['VMAX'])
    
def fsat_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        return np.average(series['IS_SAT'], weights=1/series['VMAX'])

def Mhalo_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        return np.average(series['M_HALO'], weights=1/series['VMAX'])
    
def Mhalo_std_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        totweight = np.sum(1/series['VMAX'])
        mu = np.log10(Mhalo_vmax_weighted(series))
        values = np.log10(series['M_HALO'])
        return np.sqrt(np.sum((values - mu)**2 * 1/series['VMAX']) / totweight)

def Lgal_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        return np.average(series['L_GAL'], weights=1/series['VMAX'])

def LogLgal_std_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        totweight = np.sum(1/series['VMAX'])
        mu = np.log10(Lgal_vmax_weighted(series))
        values = np.log10(series['L_GAL'])
        return np.sqrt(np.sum((values - mu)**2 * 1/series['VMAX']) / totweight)

def z_flag_is_spectro_z(arr):
    return np.logical_or(arr == AssignedRedshiftFlag.SDSS_SPEC.value, arr == AssignedRedshiftFlag.DESI_SPEC.value)

def z_flag_is_neighbor(arr):
    return arr >= AssignedRedshiftFlag.NEIGHBOR_ONE.value

def z_flag_is_random(arr):
    return arr == AssignedRedshiftFlag.PSEUDO_RANDOM.value

def z_flag_is_photo_z(arr):
    return arr == AssignedRedshiftFlag.PHOTO_Z.value

def z_flag_is_not_spectro_z(arr):
    return ~z_flag_is_spectro_z(arr)

def mstar_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        if 'Z_ASSIGNED_FLAG' in series.columns:
            should_mask = np.logical_or(series.Z_ASSIGNED_FLAG != 0, np.isnan(series['MSTAR']))
        else:
            should_mask = np.isnan(series['MSTAR'])
        masked_mstar = np.ma.masked_array(series['MSTAR'], should_mask)
        masked_vmax = np.ma.masked_array(series['VMAX'], should_mask)
        return np.average(masked_mstar, weights=1/masked_vmax)

# TODO not sure right way to do std error for this sort of data
def mstar_std_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        should_mask = np.logical_or(series.Z_ASSIGNED_FLAG != 0, np.isnan(series['MSTAR']))
        masked_mstar = np.ma.masked_array(series['MSTAR'], should_mask)
        masked_vmax = np.ma.masked_array(series['VMAX'], should_mask)
        return np.sqrt(np.average((masked_mstar - mstar_vmax_weighted(series))**2, weights=1/masked_vmax))

def qf_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        return np.average(series['QUIESCENT'], weights=1/series['VMAX'])

def qf_Dn4000MODEL_smart_eq_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        return np.average(is_quiescent_BGS_dn4000(series['LOGLGAL'], series['DN4000_MODEL'], series['G_R']), weights=1/series['VMAX'])

def qf_Dn4000_smart_eq_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        return np.average(is_quiescent_BGS_dn4000(series['LOGLGAL'], series['DN4000'], series.G_R), weights=1/series['VMAX'])

def qf_BGS_gmr_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        return np.average(is_quiescent_BGS_gmr(series['LOGLGAL'], series.G_R), weights=1/series['VMAX'])
    
def nsat_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        print(series['N_SAT'])
        return np.average(series['N_SAT'], weights=1/series['VMAX'])
    


##############################
def compute_lsat_chisqr(observed, model_lsat_r, model_lsat_b):

    #obs_lcen = data[:,0] # log10 already
    obs_lsat_r = observed[:,1] # fr
    obs_err_r = observed[:,2] # er
    obs_lsat_b = observed[:,3] # fb
    obs_err_b = observed[:,4] # eb

    obs_ratio = obs_lsat_r/obs_lsat_b
    # Dividing two quantities with errors, so we need to propagate the errors
    obs_ratio_err = obs_ratio * ((obs_err_r/obs_lsat_r)**2 + (obs_err_b/obs_lsat_b)**2)**.5

    # Get Lsat for r/b centrals from the group finder's output
    model_ratio = model_lsat_r/model_lsat_b

    # Chi squared
    lsat_chisqr = (obs_ratio - model_ratio)**2 / obs_ratio_err**2 
    print("LSat χ^2: ", lsat_chisqr)
    return lsat_chisqr