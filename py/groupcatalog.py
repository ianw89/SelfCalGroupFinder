import numpy as np
import pandas as pd
import astropy.coordinates as coord
import astropy.units as u
import os
import emcee
import pickle
import subprocess as sp
from astropy.table import Table
import astropy.io.fits as fits
import copy
import sys
import wp
import math
from multiprocessing import Pool
from joblib import Parallel, delayed

if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from pyutils import *
from redshift_guesser import SimpleRedshiftGuesser, PhotometricRedshiftGuesser
from hdf5_to_dat import pre_process_mxxl
from uchuu_to_dat import pre_process_uchuu
from bgs_helpers import *

# Sentinal value for no truth redshift
NO_TRUTH_Z = -99.99

# Shared bins for various purposes
Mhalo_bins = np.logspace(10, 15.5, 40)
Mhalo_labels = Mhalo_bins[0:len(Mhalo_bins)-1] 

L_gal_bins = np.logspace(6, 12.5, 40)
L_gal_labels = L_gal_bins[0:len(L_gal_bins)-1]

mstar_bins = np.logspace(6, 13, 30)
mstar_labels = mstar_bins[0:len(mstar_bins)-1]

Mr_gal_bins = log_solar_L_to_abs_mag_r(np.log10(L_gal_bins))
Mr_gal_labels = log_solar_L_to_abs_mag_r(np.log10(L_gal_labels))

CLUSTERING_MAG_BINS = [-16, -17, -18, -19, -20, -21, -22, -23]

GF_PROPS_VANILLA = {
    'zmin':0, 
    'zmax':0,
    'frac_area':0, # should be filled in
    'fluxlim':1,
    'color':1,
}
GF_PROPS_COLORS = {
    'zmin':0, 
    'zmax':0,
    'frac_area':0, # should be filled in
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
}

class GroupCatalog:

    def __init__(self, name):
        self.name = name
        self.output_folder = OUTPUT_FOLDER
        self.file_pattern = self.output_folder + self.name
        self.GF_outfile = self.file_pattern + ".out"
        self.results_file = self.file_pattern + ".pickle"
        self.color = 'k' # plotting color; nothing to do with galaxies
        self.marker = '-'
        self.preprocess_file = None
        self.GF_props = {} # Properties that are sent as command-line arguments to the group finder executable
        self.extra_params = None # Tuple of parameters values

        self.has_truth = False
        self.Mhalo_bins = Mhalo_bins
        self.labels = Mhalo_labels
        self.Mhalo_labels = Mhalo_labels
        self.all_data: pd.DataFrame = None
        self.centrals: pd.DataFrame = None
        self.sats: pd.DataFrame = None
        self.L_gal_bins = L_gal_bins
        self.L_gal_labels = L_gal_labels

        self.wp_all = None # (rbins, wp_all, wp_r, wp_b)
        self.wp_all_extra = None # (rbins, wp_all, wp_r, wp_b)
        self.wp_slices = np.array(len(CLUSTERING_MAG_BINS) * [None]) # Tuple of (rbins, wp) at each index
        self.wp_slices_extra = np.array(len(CLUSTERING_MAG_BINS) * [None]) # Tuple of (rbins, wp) at each index

        # Geneated from popmock option in group finder
        self.mock_b_M17 = None
        self.mock_r_M17 = None
        self.mock_b_M18 = None
        self.mock_r_M18 = None
        self.mock_b_M19 = None
        self.mock_r_M19 = None
        self.mock_b_M20 = None
        self.mock_r_M20 = None
        self.mock_b_M21 = None
        self.mock_r_M21 = None
        self.lsat_groups = None # lsat_model
        self.lsat_groups2 = None # lsat_model_scatter

        # Generated from run_corrfunc
        self.wp_mock_b_M17 = None
        self.wp_mock_r_M17 = None
        self.wp_mock_b_M18 = None
        self.wp_mock_r_M18 = None
        self.wp_mock_b_M19 = None
        self.wp_mock_r_M19 = None
        self.wp_mock_b_M20 = None
        self.wp_mock_r_M20 = None
        self.wp_mock_b_M21 = None
        self.wp_mock_r_M21 = None

        self.f_sat = None # per Lgal bin 
        self.Lgal_counts = None # size of Lgal bins 

    def get_best_wp_all(self):
        if self.wp_all_extra is not None:
            return self.wp_all_extra
        return self.wp_all
    
    def get_best_wp_slices(self):
        if self.wp_slices_extra is not None and np.all(self.wp_slices_extra != None):
            return self.wp_slices_extra
        return self.wp_slices

    def get_completeness(self):
        return spectroscopic_complete_percent(self.all_data.z_assigned_flag.to_numpy())

    def get_lostgal_neighbor_used(self):
        arr = self.all_data.z_assigned_flag.to_numpy()
        return np.sum(z_flag_is_neighbor(arr)) / np.sum(z_flag_is_not_spectro_z(arr))

    def dump(self):
        self.__class__ = eval(self.__class__.__name__) #reset __class__ attribute
        with open(self.results_file, 'wb') as f:
            pickle.dump(self, f)

    def run_group_finder(self, popmock=False, silent=False):
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
        #sp.run(["cp", HALO_MASS_FUNC_FILE , self.output_folder])
        #sp.run(["cp", LSAT_LOOKUP_FILE , self.output_folder])

        with open(self.GF_outfile, "w") as f:
            args = [BIN_FOLDER + "kdGroupFinder_omp", self.preprocess_file]
            args.append(str(self.GF_props['zmin']))
            args.append(str(self.GF_props['zmax']))
            args.append(str(self.GF_props['frac_area']))
            if self.GF_props.get('fluxlim') == 1:
                args.append("-f")
            if self.GF_props.get('color') == 1:
                args.append("-c")
            if silent:
                args.append("-s")
            if popmock:
                args.append("--popmock")
            if 'omegaL_sf' in self.GF_props:
                args.append(f"--wcen={self.GF_props['omegaL_sf']},{self.GF_props['sigma_sf']},{self.GF_props['omegaL_q']},{self.GF_props['sigma_q']},{self.GF_props['omega0_sf']},{self.GF_props['omega0_q']}")
            if 'beta0q' in self.GF_props:
                args.append(f"--bsat={self.GF_props['beta0q']},{self.GF_props['betaLq']},{self.GF_props['beta0sf']},{self.GF_props['betaLsf']}")
            if 'omega_chi_0_sf' in self.GF_props:
                args.append(f"--chi1={self.GF_props['omega_chi_0_sf']},{self.GF_props['omega_chi_0_q']},{self.GF_props['omega_chi_L_sf']},{self.GF_props['omega_chi_L_q']}")            

            print(args)
            # The galaxies are written to stdout, so send ot the GF_outfile file stream
            self.results = sp.run(args, cwd=self.output_folder, stdout=f)
            
            #print(self.results.returncode)
        
        # TODO what about if there was an error? returncode for the GF doesn't seem useful right now
        if popmock:
            self.mock_b_M17 = np.loadtxt(f'{self.output_folder}mock_blue_M17.dat', skiprows=0, dtype='float')
            self.mock_r_M17 = np.loadtxt(f'{self.output_folder}mock_red_M17.dat', skiprows=0, dtype='float')
            self.mock_b_M18 = np.loadtxt(f'{self.output_folder}mock_blue_M18.dat', skiprows=0, dtype='float')
            self.mock_r_M18 = np.loadtxt(f'{self.output_folder}mock_red_M18.dat', skiprows=0, dtype='float')
            self.mock_b_M19 = np.loadtxt(f'{self.output_folder}mock_blue_M19.dat', skiprows=0, dtype='float')
            self.mock_r_M19 = np.loadtxt(f'{self.output_folder}mock_red_M19.dat', skiprows=0, dtype='float')
            self.mock_b_M20 = np.loadtxt(f'{self.output_folder}mock_blue_M20.dat', skiprows=0, dtype='float')
            self.mock_r_M20 = np.loadtxt(f'{self.output_folder}mock_red_M20.dat', skiprows=0, dtype='float')
            self.mock_b_M21 = np.loadtxt(f'{self.output_folder}mock_blue_M21.dat', skiprows=0, dtype='float')
            self.mock_r_M21 = np.loadtxt(f'{self.output_folder}mock_red_M21.dat', skiprows=0, dtype='float')
            
            self.lsat_groups = np.loadtxt(f'{self.output_folder}lsat_groups.out', skiprows=0, dtype='float')
            if os.path.exists(f'{self.output_folder}lsat_groups2.out'):
                self.lsat_groups2 = np.loadtxt(f'{self.output_folder}lsat_groups2.out', skiprows=0, dtype='float')

        t2 = time.time()
        print(f"run_group_finder() took {t2-t1:.2} seconds.")


    def run_corrfunc(self):
        """
        Run corrfunc on the mock populated with an HOD built from this sample. 
        This does NOT calculate the projected correlation functino of this sample directly!
        TODO rename and refactor all this.
        """

        if self.GF_outfile is None:
            print("Warning: run_corrfunc() called without GF_outfile set.")
            return
        if not os.path.exists(self.GF_outfile):
            print(f"Warning: run_corrfunc() should be called after run_group_finder(). File {self.GF_outfile} does not exist.")
            return
        
        wp.run_corrfunc(self.output_folder)


        self.wp_mock_b_M17 = np.loadtxt(f'{self.output_folder}wp_mock_blue_M17.dat', skiprows=0, dtype='float')
        self.wp_mock_r_M17 = np.loadtxt(f'{self.output_folder}wp_mock_red_M17.dat', skiprows=0, dtype='float')
        self.wp_mock_b_M18 = np.loadtxt(f'{self.output_folder}wp_mock_blue_M18.dat', skiprows=0, dtype='float')
        self.wp_mock_r_M18 = np.loadtxt(f'{self.output_folder}wp_mock_red_M18.dat', skiprows=0, dtype='float')
        self.wp_mock_b_M19 = np.loadtxt(f'{self.output_folder}wp_mock_blue_M19.dat', skiprows=0, dtype='float')
        self.wp_mock_r_M19 = np.loadtxt(f'{self.output_folder}wp_mock_red_M19.dat', skiprows=0, dtype='float')
        self.wp_mock_b_M20 = np.loadtxt(f'{self.output_folder}wp_mock_blue_M20.dat', skiprows=0, dtype='float')
        self.wp_mock_r_M20 = np.loadtxt(f'{self.output_folder}wp_mock_red_M20.dat', skiprows=0, dtype='float')
        self.wp_mock_b_M21 = np.loadtxt(f'{self.output_folder}wp_mock_blue_M21.dat', skiprows=0, dtype='float')
        self.wp_mock_r_M21 = np.loadtxt(f'{self.output_folder}wp_mock_red_M21.dat', skiprows=0, dtype='float')


    def refresh_df_views(self):
        if self.all_data is not None:
            # Compute some common aggregations upfront here
            # TODO make these lazilly evaluated properties on the GroupCatalog object
            # Can put more of them into this pattern from elsewhere in plotting code then
            self.f_sat = self.all_data.groupby('Lgal_bin', observed=False).apply(fsat_vmax_weighted)
            self.f_sat_sf = self.all_data.loc[~self.all_data.quiescent].groupby('Lgal_bin', observed=False).apply(fsat_vmax_weighted)
            self.f_sat_q = self.all_data.loc[self.all_data.quiescent].groupby('Lgal_bin', observed=False).apply(fsat_vmax_weighted)
            self.Lgal_counts = self.all_data.groupby('Lgal_bin', observed=False).RA.count()

            # TODO this is incomplete; just counting galaxies right now
            #self.f_sat_cic = cic_binning(self.all_data['L_gal'].to_numpy(), [self.L_gal_bins])
            #self.f_sat_sf_cic = cic_binning(self.all_data.loc[~self.all_data.quiescent, 'L_gal'].to_numpy(), [self.L_gal_bins])
            #self.f_sat_q_cic = cic_binning(self.all_data.loc[self.all_data.quiescent, 'L_gal'].to_numpy(), [self.L_gal_bins])

            # Setup some convenience subsets of the DataFrame
            self.centrals = self.all_data.loc[self.all_data.index == self.all_data.igrp]
            self.sats = self.all_data.loc[self.all_data.index != self.all_data.igrp]

    def postprocess(self):
        if self.all_data is not None:
            self.refresh_df_views()
        else:
            print("Warning: postprocess called with all_data DataFrame is not set yet. Override postprocess() or after calling run_group_finder() set it.")

    def calculate_projected_clustering(self, with_extra_randoms=False):
        pass

    def calculate_projected_clustering_in_magbins(self, with_extra_randoms=False):
        pass

    def get_true_z_from(self, truth_df: pd.DataFrame):
        """
        Adds a column to the catalog's all_data DataFrame with the true redshifts from the truth_df DataFrame 
        for rows with z_assigned_flag != 0.
        """
        #self.all_data = self.all_data.convert_dtypes()
        
        if self.has_truth:
            self.all_data.drop(columns=['z_T', 'L_gal_T', 'logLgal_T', 'g_r_T', 'is_sat_truth', 'Lgal_bin_T', 'target_id_T', 'z_assigned_flag_T'], inplace=True, errors='ignore')
        
        truth_df = truth_df[['target_id', 'z', 'z_assigned_flag', 'L_gal', 'logLgal', 'g_r', 'Lgal_bin', 'is_sat']].copy()
        truth_df['target_id'] = truth_df['target_id'].astype('Int64')
        truth_df.index = truth_df.target_id
        self.all_data = self.all_data.join(truth_df, on='target_id', how='left', rsuffix='_T', validate="1:1")
        # I want target_id_T to be int, but there are NaNs in the join, so turn NaN into -1
        rows_to_nan = z_flag_is_not_spectro_z(self.all_data['z_assigned_flag_T'])
        self.all_data.loc[rows_to_nan, 'z_T'] = NO_TRUTH_Z
        print(f"{np.sum(rows_to_nan)} galaxies to have no truth redshift.")
        #self.all_data.drop(columns=['target_id_T', 'z_assigned_flag_T'], inplace=True)
        self.all_data.rename(columns={'is_sat_T': 'is_sat_truth'}, inplace=True)
        self.has_truth = True

        # If we failed to match on target_id for most galaxies, let's instead use match_coordinate_sky
        #if np.sum(rows_to_nan) > 0.5 * len(self.all_data):
        #    print("Warning: get_true_z_from() failed to match target_id for many galaxies. Falling back to match_coordinate_sky.")
            # drop the columns we added
        #    self.all_data.drop(columns=['z_T', 'is_sat_truth'], inplace=True)

            # match on sky coordinates
        #    coords_catalog = coord.SkyCoord(ra=self.all_data.RA.to_numpy() * u.degree, dec=self.all_data.Dec.to_numpy() * u.degree)
        #    coords_truth = coord.SkyCoord(ra=truth_df.RA.to_numpy() * u.degree, dec=truth_df.Dec.to_numpy() * u.degree)
        #    idx, d2d, _ = coord.match_coordinates_sky(coords_catalog, coords_truth, nthneighbor=1)
        #    matched = d2d < 1 * u.arcsec  # 1 arcsecond tolerance

        #    self.all_data['z_T'] = NO_TRUTH_Z
        #    self.all_data.loc[matched, 'z_T'] = truth_df.iloc[idx[matched]]['z']
        #    self.all_data['is_sat_truth'] = 0
        #    self.all_data.loc[matched, 'is_sat_truth'] = truth_df.iloc[idx[matched]]['is_sat']
            


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

    def __init__(self, name: str, preprocessed_file: str, galprops_file: str):
        super().__init__(name)
        self.preprocess_file = preprocessed_file
        self.galprops_file = galprops_file
        self.L_gal_bins = self.L_gal_bins[15:]
        self.L_gal_labels = self.L_gal_labels[15:]

        self.volume = np.array([1.721e+06, 6.385e+06, 2.291e+07, 7.852e+07]) # Copied from Jeremy's groupfind_mcmc.py
        self.vfac = (self.volume/250.0**3)**.5 # factor by which to multiply errors
        self.efac = 0.1 # let's just add a constant fractional error bar

    def postprocess(self):
        origprops = pd.read_csv(self.preprocess_file, delimiter=' ', names=('ra', 'dec', 'z', 'logLgal', 'Vmax', 'quiescent', 'chi'))
        galprops = pd.read_csv(self.galprops_file, delimiter=' ', names=('Mag_g', 'Mag_r', 'sigma_v', 'Dn4000', 'concentration', 'log_M_star', 'z_assigned_flag'))
        galprops['g_r'] = galprops.Mag_g - galprops.Mag_r 
        galprops['quiescent'] = origprops.quiescent.astype(bool)
        galprops.rename(columns={'Mag_r': "app_mag"}, inplace=True)
        self.all_data = read_and_combine_gf_output(self, galprops)
        #self.all_data['quiescent'] = is_quiescent_SDSS_Dn4000(self.all_data.logLgal, self.all_data.Dn4000)
        self.all_data['mstar'] = np.power(10, self.all_data.log_M_star)
        self.all_data['Mstar_bin'] = pd.cut(x = self.all_data['mstar'], bins = mstar_bins, labels = mstar_labels, include_lowest = True)
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
        origprops = pd.read_csv(SDSS_v1_DAT_FILE, delimiter=' ', names=('ra', 'dec', 'z', 'logLgal', 'Vmax', 'quiescent', 'chi'))
        galprops = pd.read_csv(SDSS_v1_1_GALPROPS_FILE, delimiter=' ', names=('Mag_g', 'Mag_r', 'sigma_v', 'Dn4000', 'concentration', 'log_M_star', 'z_assigned_flag'))
        galprops['g_r'] = galprops.Mag_g - galprops.Mag_r 
        galprops['quiescent'] = origprops.quiescent.astype(bool)
        galprops.rename(columns={'Mag_r': "app_mag"}, inplace=True)
        
        main_df = pd.read_csv(self.GF_outfile, delimiter=' ', names=('RA', 'Dec', 'z', 'L_gal', 'V_max', 'P_sat', 'M_halo', 'N_sat', 'L_tot', 'igrp', 'weight'))
        
        df = pd.merge(main_df, galprops, left_index=True, right_index=True)

        # add columns indicating if galaxy is a satellite
        df['is_sat'] = (df.index != df.igrp).astype(int)
        df['logLgal'] = np.log10(df.L_gal)

        # add column for halo mass bins and Lgal bins
        df['Mh_bin'] = pd.cut(x = df['M_halo'], bins = self.Mhalo_bins, labels = self.Mhalo_labels, include_lowest = True)
        df['Lgal_bin'] = pd.cut(x = df['L_gal'], bins = self.L_gal_bins, labels = self.L_gal_labels, include_lowest = True)

        self.all_data = df
        #self.all_data['quiescent'] = is_quiescent_SDSS_Dn4000(self.all_data.logLgal, self.all_data.Dn4000)
        self.all_data['mstar'] = np.power(10, self.all_data.log_M_star)
        self.all_data['Mstar_bin'] = pd.cut(x = self.all_data['mstar'], bins = mstar_bins, labels = mstar_labels, include_lowest = True)
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

    def create_test_dat_files(self):
        gals = pd.read_csv(SDSS_v1_DAT_FILE, delimiter=' ', names=('ra', 'dec', 'z', 'logLgal', 'Vmax', 'quiescent', 'chi'))
        galprops = pd.read_csv(SDSS_v1_1_GALPROPS_FILE, delimiter=' ', names=('Mag_g', 'Mag_r', 'sigma_v', 'Dn4000', 'concentration', 'log_M_star', 'z_assigned_flag'))

        cut_gals = gals[np.logical_and(gals.ra > 149.119, gals.ra < 151.119)]
        cut_gals = cut_gals[np.logical_and(cut_gals.dec > 1.205, cut_gals.dec < 3.205)]
        indexes = cut_gals.index
        print(f"Cut to {len(indexes)} galaxies.")
        cut_galprops = galprops.iloc[indexes]

        # write to TEST_DAT_FILE and TEST_GALPROPS_FILE
        cut_gals.to_csv(TEST_DAT_FILE, sep=' ', header=False, index=False)
        cut_galprops.to_csv(TEST_GALPROPS_FILE, sep=' ', header=False, index=False)

    def postprocess(self):
        origprops = pd.read_csv(TEST_DAT_FILE, delimiter=' ', names=('ra', 'dec', 'z', 'logLgal', 'Vmax', 'quiescent', 'chi'))
        galprops = pd.read_csv(TEST_GALPROPS_FILE, delimiter=' ', names=('Mag_g', 'Mag_r', 'sigma_v', 'Dn4000', 'concentration', 'log_M_star', 'z_assigned_flag'))
        galprops['g_r'] = galprops.Mag_g - galprops.Mag_r 
        galprops['quiescent'] = origprops.quiescent.astype(bool)
        galprops.rename(columns={'Mag_r': "app_mag"}, inplace=True)
        self.all_data = read_and_combine_gf_output(self, galprops)
        #self.all_data['quiescent'] = is_quiescent_SDSS_Dn4000(self.all_data.logLgal, self.all_data.Dn4000)
        add_halo_columns(self)
        return super().postprocess()

class MXXLGroupCatalog(GroupCatalog):

    def __init__(self, name, mode: Mode, mag_cut: float, catalog_mag_cut: float, use_colors: bool):
        super().__init__(name)
        self.mode = mode
        self.mag_cut = mag_cut
        self.catalog_mag_cut = catalog_mag_cut
        self.use_colors = use_colors
        self.color = mode_to_color(mode)
        

    def preprocess(self):
        fname, props = pre_process_mxxl(MXXL_FILE, self.mode.value, self.file_pattern, self.mag_cut, self.catalog_mag_cut, self.use_colors)
        self.preprocess_file = fname
        for p in props:
            self.GF_props[p] = props[p]

    def run_group_finder(self, popmock=False):
        if self.preprocess_file is None:
            self.preprocess()
        super().run_group_finder(popmock=popmock)


    def postprocess(self):

        filename_props_fast = str.replace(self.GF_outfile, ".out", "_galprops.pkl")
        filename_props_slow = str.replace(self.GF_outfile, ".out", "_galprops.dat")
        if os.path.exists(filename_props_fast):
            galprops = pd.read_pickle(filename_props_fast)
        else:
            galprops = pd.read_csv(filename_props_slow, delimiter=' ', names=('app_mag', 'g_r', 'galaxy_type', 'mxxl_halo_mass', 'z_assigned_flag', 'assigned_halo_mass', 'z_obs', 'mxxl_halo_id', 'assigned_halo_id'), dtype={'mxxl_halo_id': np.int32, 'assigned_halo_id': np.int32, 'z_assigned_flag': np.int8})
        
        self.all_data = read_and_combine_gf_output(self, galprops)
        df = self.all_data
        self.has_truth = True#self.mode.value == Mode.ALL.value
        df['is_sat_truth'] = np.logical_or(df.galaxy_type == 1, df.galaxy_type == 3)
        df['z_T'] = df['z_obs'] # MXXL truth values are always there
        if self.has_truth:
            df['Mh_bin_T'] = pd.cut(x = df['mxxl_halo_mass']*10**10, bins = Mhalo_bins, labels = Mhalo_labels, include_lowest = True)
            df['L_gal_T'] = np.power(10, abs_mag_r_to_log_solar_L(app_mag_to_abs_mag_k(df.app_mag.to_numpy(), df.z_obs.to_numpy(), df.g_r.to_numpy())))
            df['Lgal_bin_T'] = pd.cut(x = df['L_gal_T'], bins = self.L_gal_bins, labels = self.L_gal_labels, include_lowest = True)
            self.truth_f_sat = df.groupby('Lgal_bin_T').apply(fsat_truth_vmax_weighted)
            self.centrals_T = df[np.invert(df.is_sat_truth)]
            self.sats_T = df[df.is_sat_truth]

        # TODO if we switch to using bins we need a Truth version of this
        df['quiescent'] = is_quiescent_BGS_gmr(df.logLgal, df.g_r)

        super().postprocess()

class UchuuGroupCatalog(GroupCatalog):
   
    def __init__(self, name, mode: Mode, mag_cut: float, catalog_mag_cut: float, use_colors: bool):
        super().__init__(name)
        self.mode = mode
        self.mag_cut = mag_cut
        self.catalog_mag_cut = catalog_mag_cut
        self.use_colors = use_colors
        self.color = get_color(9)

    def preprocess(self):
        fname, props = pre_process_uchuu(UCHUU_FILE, self.mode.value, self.file_pattern, self.mag_cut, self.catalog_mag_cut, self.use_colors)
        self.preprocess_file = fname
        for p in props:
            self.GF_props[p] = props[p]

    def run_group_finder(self, popmock=False):
        if self.preprocess_file is None:
            self.preprocess()
        super().run_group_finder(popmock=popmock)


    def postprocess(self):
        filename_props_fast = str.replace(self.GF_outfile, ".out", "_galprops.pkl")
        filename_props_slow = str.replace(self.GF_outfile, ".out", "_galprops.dat")
        if os.path.exists(filename_props_fast):
            galprops = pd.read_pickle(filename_props_fast)
        else:
            galprops = pd.read_csv(filename_props_slow, delimiter=' ', names=('app_mag', 'g_r', 'central', 'uchuu_halo_mass', 'uchuu_halo_id'), dtype={'uchuu_halo_id': np.int64, 'central': np.bool_})
        
        df = read_and_combine_gf_output(self, galprops)
        self.all_data = df

        self.has_truth = True
        self['is_sat_truth'] = np.invert(df.central)
        self['Mh_bin_T'] = pd.cut(x = self['uchuu_halo_mass']*10**10, bins = Mhalo_bins, labels = Mhalo_labels, include_lowest = True)
        # TODO BUG Need L_gal_T, the below is wrong!
        truth_f_sat = df.groupby('Lgal_bin').apply(fsat_truth_vmax_weighted)
        self.truth_f_sat = truth_f_sat
        self.centrals_T = df[np.invert(df.is_sat_truth)]
        self.sats_T = df[df.is_sat_truth]

        # TODO add quiescent column

        super().postprocess()

class BGSGroupCatalog(GroupCatalog):
    
    extra_prop_df: pd.DataFrame = None

    def __init__(self, name, mode: Mode, mag_cut: float, catalog_mag_cut: float, sdss_fill: bool = True, num_passes: int = 3, drop_passes: int = 0, data_cut: str = "Y1-Iron", extra_params = None):
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

    @staticmethod
    def from_MCMC(reader: emcee.backends.HDFBackend, mode: Mode):

        idx = np.argmax(reader.get_log_prob(flat=True))
        p = reader.get_chain(flat=True)[idx]
        if len(p) != 13:
            raise ValueError("reader has wrong number of parameters. Expected 13.")
        
        print(f"Using MCMC parameters: {p}")

        gc = BGSGroupCatalog(f"BGS SV3 MCMC {mode_to_str(mode)}", mode, 19.5, 23.0, sdss_fill=False, num_passes=10, drop_passes=3, data_cut="sv3", extra_params=p)
        gc.GF_props = GF_PROPS_VANILLA.copy()
        if mode.value == Mode.PHOTOZ_PLUS_v1.value:
            gc.color = 'g'
        elif mode.value == Mode.PHOTOZ_PLUS_v2.value:
            gc.color = 'darkorange'
        elif mode.value == Mode.PHOTOZ_PLUS_v3.value:
            gc.color = 'purple'
        
        gc.marker = '--'

        return gc

    def preprocess(self):
        print("Pre-processing...")
        if self.data_cut == "Y1-Iron":
            infile = IAN_BGS_MERGED_FILE
        elif self.data_cut == "Y1-Iron-v1.2":
            infile = IAN_BGS_MERGED_FILE_OLD
        elif self.data_cut == "Y3-Kibo":
            infile = IAN_BGS_Y3_MERGED_FILE_KIBO
        elif self.data_cut == "Y3-Kibo-SV3Cut":
            infile = IAN_BGS_Y3_MERGED_FILE_KIBO
        elif self.data_cut == "Y3-Loa":
            infile = IAN_BGS_Y3_MERGED_FILE
        elif self.data_cut == "Y3-Loa-SV3Cut":
            infile = IAN_BGS_Y3_MERGED_FILE
        elif self.data_cut == "Y3-Jura":
            infile = IAN_BGS_Y3_MERGED_FILE_JURA
        elif self.data_cut == "sv3":
            infile = IAN_BGS_SV3_MERGED_FILE
        else:
            raise ValueError("Unknown data_cut value")
        
        fname, props = pre_process_BGS(infile, self.mode.value, self.file_pattern, self.mag_cut, self.catalog_mag_cut, self.sdss_fill, self.num_passes, self.drop_passes, self.data_cut, self.extra_params)
        self.preprocess_file = fname
        for p in props:
            self.GF_props[p] = props[p]

    def run_group_finder(self, popmock=False):
        if self.preprocess_file is None:
            self.preprocess()
        else:
            print("Skipping pre-processing")
        super().run_group_finder(popmock=popmock)

    def add_bootstrapped_f_sat(self, N_ITERATIONS = 100):

        df = self.all_data

        if self.data_cut == 'sv3':
            print("Bootstrapping for fsat error estimate...")
            t1 = time.time()
            # label the SV3 region each galaxy is in
            df['region'] = tile_to_region(df['nearest_tile_id'])

            # Add bootstrapped error bars for fsat
            f_sat_realizations = []
            f_sat_sf_realizations = []
            f_sat_q_realizations = []

            def bootstrap_iteration(region_indices):
                relevent_columns = ['Lgal_bin', 'is_sat', 'V_max', 'quiescent']
                alt_df = pd.DataFrame(columns=relevent_columns)
                for idx in region_indices:
                    rows_to_add = df.loc[df.region == idx, relevent_columns]
                    if len(alt_df) > 0:
                        alt_df = pd.concat([alt_df, rows_to_add])
                    else:
                        alt_df = rows_to_add

                f_sat = alt_df.groupby('Lgal_bin', observed=False).apply(fsat_vmax_weighted)
                f_sat_sf = alt_df[alt_df.quiescent == False].groupby('Lgal_bin', observed=False).apply(fsat_vmax_weighted)
                f_sat_q = alt_df[alt_df.quiescent == True].groupby('Lgal_bin', observed=False).apply(fsat_vmax_weighted)
                return f_sat, f_sat_sf, f_sat_q

            results = Parallel(n_jobs=-1)(delayed(bootstrap_iteration)(np.random.choice(range(len(sv3_regions_sorted)), len(sv3_regions_sorted), replace=True)) for _ in range(N_ITERATIONS))
            #results = [bootstrap_iteration(np.random.choice(range(len(sv3_regions_sorted)), len(sv3_regions_sorted), replace=True)) for _ in range(N_ITERATIONS)]

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
        filename_props_fast = str.replace(self.GF_outfile, ".out", "_galprops.pkl")
        filename_props_slow = str.replace(self.GF_outfile, ".out", "_galprops.dat")
        if os.path.exists(filename_props_fast):
            galprops = pd.read_pickle(filename_props_fast)
        else:
            # g_r is k-corrected
            galprops = pd.read_csv(filename_props_slow, delimiter=' ', names=('app_mag', 'target_id', 'z_assigned_flag', 'g_r', 'Dn4000'), dtype={'target_id': np.int64, 'z_assigned_flag': np.int8})
        
        df = read_and_combine_gf_output(self, galprops)

        # TODO we write this to the .dat file, use that instead of re-evaluating and double check all is the same
        df['quiescent'] = is_quiescent_BGS_gmr(df.logLgal, df.g_r)

        # Get extra fastspecfit columns. Could have threaded these through with galprops
        # But if they aren't used in group finding or preprocessing this is easier to update
        if BGSGroupCatalog.extra_prop_df is None:
            print("Getting fastspecfit data...", end='\r')
            BGSGroupCatalog.extra_prop_df = get_extra_bgs_fastspectfit_data()
            print("Getting fastspecfit data... done")
        
        prior_len = len(df)
        df = pd.merge(df, BGSGroupCatalog.extra_prop_df, on='target_id', how='left')
        assert prior_len == len(df)
        df['Mstar_bin'] = pd.cut(x = df['mstar'], bins = mstar_bins, labels = mstar_labels, include_lowest = True)

        self.all_data = df

        super().postprocess()

        if self.data_cut == 'sv3':
            self.centered = filter_SV3_to_avoid_edges(self)

        if self.data_cut == 'sv3' or self.data_cut == 'Y3-Kibo-SV3Cut' or self.data_cut == 'Y3-Loa-SV3Cut':
            self.calculate_projected_clustering()
            self.calculate_projected_clustering_in_magbins()

        print("Post-processing done.")

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

        # TODO weights for fiber collisions corrections

        df = self.all_data
        if df.get('mag_R') is None:
            df['mag_R'] = log_solar_L_to_abs_mag_r(np.log10(df['L_gal']))
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

        #Mag-5log(h) > -14:  zmax theory=0.01650  zmax obs=0.015029
        #Mag-5log(h) > -15:  zmax theory=0.02595  zmax obs=0.027139
        #Mag-5log(h) > -16:  zmax theory=0.04067  zmax obs=0.043211
        #Mag-5log(h) > -17:  zmax theory=0.06336  zmax obs=0.06597
        #Mag-5log(h) > -18:  zmax theory=0.09792  zmax obs=0.101448
        #Mag-5log(h) > -19:  zmax theory=0.14977  zmax obs=0.154458
        #Mag-5log(h) > -20:  zmax theory=0.22620  zmax obs=0.231856
        #Mag-5log(h) > -21:  zmax theory=0.33694  zmax obs=0.331995
        #Mag-5log(h) > -22:  zmax theory=0.49523  zmax obs=0.459593
        #Mag-5log(h) > -23:  zmax theory=0.72003  zmax obs=0.49971
        zmax = [0.04067, 0.06336, 0.09792, 0.14977, 0.22620, 0.331995, 0.459593, 0.49971]
        #Not doing zmin's for now; no need for indepdenent samples for this

        df = self.all_data
        if df.get('mag_R') is None:
            df['mag_R'] = log_solar_L_to_abs_mag_r(np.log10(df['L_gal']))
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
        df['region'] = tile_to_region(df['nearest_tile_id']).astype(int)
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

    def write_sharable_output_file(self, name=None):
        print("Writing a sharable output file")
        if name is None:
            name = str.replace(self.GF_outfile, ".out", " Catalog.csv")
        elif not name.endswith('.csv'):
            name = name + '.csv'
        df = self.all_data.drop(columns=['Mstar_bin', 'Mh_bin', 'Lgal_bin', 'logLgal', 'Dn4000'])
        print(df.columns)
        df.to_csv(name, index=False, header=True)


def get_extra_bgs_fastspectfit_data():
    fname = OUTPUT_FOLDER + 'bgs_mstar.pkl'
    if os.path.isfile(fname):
        return pickle.load(open(fname, 'rb'))
    else:
        hdul = fits.open(BGS_FASTSPEC_FILE, memmap=True)
        data = hdul[1].data
        fastspecfit_id = data['TARGETID']
        log_mstar = data['LOGMSTAR'].astype("<f8")
        mstar = np.power(10, log_mstar)
        #Dn4000 = data['DN4000'].astype("<f8")
        hdul.close()

        df = pd.DataFrame({'target_id': fastspecfit_id, 'mstar': mstar})
        pickle.dump(df, open(fname, 'wb'))


def get_objects_near_sv3_regions(gals_coord, radius_deg):
    """
    Returns a true/false array of len(gals_coord) that is True for objects within radius_deg 
    of an SV3 region.
    """

    SV3_tiles = pd.read_csv(BGS_Y3_TILES_FILE, delimiter=',', usecols=['TILEID', 'FAFLAVOR', 'TILERA', 'TILEDEC', 'TILERA', 'TILEDEC'])
    SV3_tiles = SV3_tiles.loc[SV3_tiles.FAFLAVOR == 'sv3bright']
    SV3_tiles.reset_index(inplace=True)

    # Cut to the regions of interest
    center_ra = []
    center_dec = []
    for region in sv3_regions_sorted:
        tiles = SV3_tiles.loc[SV3_tiles.TILEID.isin(region)]
        center_ra.append(np.mean(tiles.TILERA))
        center_dec.append(np.mean(tiles.TILEDEC))
    
    tiles_coord = coord.SkyCoord(ra=center_ra*u.degree, dec=center_dec*u.degree, frame='icrs')
    idx, d2d, d3d = coord.match_coordinates_sky(gals_coord, tiles_coord, nthneighbor=1, storekdtree='kdtree_sv3_tiles')
    ang_distances = d2d.to(u.degree).value

    return ang_distances < radius_deg

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
    gals_coord = coord.SkyCoord(ra=df.RA.to_numpy()*u.degree, dec=df.Dec.to_numpy()*u.degree, frame='icrs')
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
        as_per_kpc = get_cosmology().arcsec_per_kpc_proper(df['z'].to_numpy())
        df.loc[:, 'mxxl_halo_vir_radius_guess_arcsec'] =  df.loc[:, 'mxxl_halo_vir_radius_guess'].to_numpy() * as_per_kpc.to(u.arcsec / u.kpc).value

    masses = df['M_halo'].to_numpy() * u.solMass # / h
    df['halo_radius_kpc'] = get_vir_radius_mine(masses) # kpc / h
    # TODO comoving or proper?
    as_per_kpc = get_cosmology().arcsec_per_kpc_proper(df['z'].to_numpy())
    df['halo_radius_arcsec'] = df['halo_radius_kpc'].to_numpy() * as_per_kpc.to(u.arcsec / u.kpc).value

    # Luminosity distance to z_obs
    #df.loc[:, 'ldist_true'] = z_to_ldist(df.z_obs.to_numpy())

def update_properties_for_indices(idx, app_mag_r, app_mag_g, g_r_apparent, z_eff, abs_mag_R, abs_mag_R_k, abs_mag_G, abs_mag_G_k, log_L_gal, quiescent):
    np.put(abs_mag_R, idx, app_mag_to_abs_mag(app_mag_r[idx], z_eff[idx]))
    np.put(abs_mag_R_k, idx, k_correct(abs_mag_R[idx], z_eff[idx], g_r_apparent[idx], band='r'))
    np.put(abs_mag_G, idx, app_mag_to_abs_mag(app_mag_g[idx], z_eff[idx]))
    np.put(abs_mag_G_k, idx, k_correct(abs_mag_G[idx], z_eff[idx], g_r_apparent[idx], band='g'))
    np.put(log_L_gal, idx, abs_mag_r_to_log_solar_L(abs_mag_R_k[idx]))
    G_R_k = abs_mag_G_k - abs_mag_R_k
    np.put(quiescent, idx, is_quiescent_BGS_gmr(None, G_R_k[idx]))
    return G_R_k

def get_tbl_column(tbl, colname, required=False):
    if colname in tbl.columns:
        if np.ma.is_masked(tbl[colname]):
            return tbl[colname].data.data
        return tbl[colname]
    else:
        if required:
            raise ValueError(f"Required column {colname} not found in table.")
        return None

def get_footprint_fraction(data_cut, mode, num_passes_required):
    # These are calculated from randoms in BGS_study.ipynb
    if data_cut == "Y1-Iron" or data_cut == "Y1-Iron-v1.2":
        # For Y1-Iron  
        FOOTPRINT_FRAC_1pass = 0.1876002 # 7739 degrees
        FOOTPRINT_FRAC_2pass = 0.1153344 # 4758 degrees
        FOOTPRINT_FRAC_3pass = 0.0649677 # 2680 degrees
        FOOTPRINT_FRAC_4pass = 0.0228093 # 940 degrees
        # 0% 5pass coverage
    elif data_cut == "Y3-Jura":
        FOOTPRINT_FRAC_1pass = 0.310691 # 12816 degrees
        FOOTPRINT_FRAC_2pass = 0.286837 # 11832 degrees
        FOOTPRINT_FRAC_3pass = 0.233920 # 9649 degrees
        FOOTPRINT_FRAC_4pass = 0.115183 # 4751 degrees
    elif data_cut == "Y3-Kibo" or data_cut == "Y3-Loa":
        FOOTPRINT_FRAC_1pass = 0.30968189465008605 # 12775 degrees
        FOOTPRINT_FRAC_2pass = 0.2859776210215015 # 11797 degrees
        FOOTPRINT_FRAC_3pass = 0.23324031706784962 # 9621 degrees
        FOOTPRINT_FRAC_4pass = 0.1148695997866822 # 4738 degrees
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
            wants_MCMC = False
            if len(extra_params) == 2:
                NEIGHBORS, BB_PARAMS = extra_params
                print("Using one set of extra parameter values for all color combinations.")
                RR_PARAMS = BR_PARAMS = RB_PARAMS = BB_PARAMS
            elif len(extra_params) == 5:
                NEIGHBORS, BB_PARAMS, RB_PARAMS, BR_PARAMS, RR_PARAMS = extra_params
            elif len(extra_params) == 13:
                NEIGHBORS = extra_params[0]
                BB_PARAMS, RB_PARAMS, BR_PARAMS, RR_PARAMS = extra_params[1:].reshape(4, 3)
            else:
                raise ValueError("Extra parameters must be a tuple of length 2 or 5")
            NEIGHBORS = int(NEIGHBORS)
        else:
            print("Extra parameters not provided for PHOTOZ_PLUS mode; will MCMC them.")
            NEIGHBORS = 10
            wants_MCMC = True


    # Some versions of the LSS Catalogs use astropy's Table used masked arrays for unobserved spectral targets    
    if np.ma.is_masked(table['Z']):
        z_obs = table['Z'].data.data
        unobserved = table['Z'].mask # the masked values are what is unobserved
        unobserved_orginal = unobserved.copy()
    else:
        # SV3 version didn't do this
        z_obs = table['Z']
        unobserved = table['Z'].astype("<i8") == 999999
        unobserved_orginal = unobserved.copy()

    obj_type = get_tbl_column(table, 'SPECTYPE')
    deltachi2 = get_tbl_column(table, 'DELTACHI2')
    dec = get_tbl_column(table, 'DEC', required=True)
    ra = get_tbl_column(table, 'RA', required=True)
    maskbits = get_tbl_column(table, 'MASKBITS')
    ref_cat = get_tbl_column(table, 'REF_CAT')
    tileid = get_tbl_column(table, 'TILEID')
    target_id = get_tbl_column(table, 'TARGETID')
    ntid = get_tbl_column(table, 'NEAREST_TILEIDS')[:,0] # just need to nearest tile for our purposes
    app_mag_r = get_tbl_column(table, 'APP_MAG_R', required=True)
    app_mag_g = get_tbl_column(table, 'APP_MAG_G', required=True)
    g_r_apparent = app_mag_g - app_mag_r
    abs_mag_R = get_tbl_column(table, 'ABS_MAG_R', required=True)
    abs_mag_R_k = get_tbl_column(table, 'ABS_MAG_R_K', required=True)
    abs_mag_G = get_tbl_column(table, 'ABS_MAG_G', required=True)
    abs_mag_G_k = get_tbl_column(table, 'ABS_MAG_G_K', required=True)
    log_L_gal = get_tbl_column(table, 'LOG_L_GAL', required=True)
    quiescent = get_tbl_column(table, 'QUIESCENT', required=True)
    p_obs = get_tbl_column(table, 'PROB_OBS')
    if p_obs is None:
        print("WARNING: PROB_OBS column not found in FITS file. Using 0.689 for all unobserved galaxies.")
        p_obs = np.ones(len(z_obs)) * 0.689
    nan_pobs = np.isnan(p_obs)
    if np.any(nan_pobs):
        print(f"WARNING: {np.sum(nan_pobs)} galaxies have nan p_obs. Setting those to 0.689, the mean of Y3.")
        p_obs[nan_pobs] = 0.689
    z_phot = get_tbl_column(table, 'Z_PHOT')
    have_z_phot = True
    if z_phot is None:
        print("WARNING: Z_PHOT column not found in FITS file. Will be set to nan for all.")
        z_phot = np.ones(len(z_obs)) * np.nan
        have_z_phot = False
    dn4000 = get_tbl_column(table, 'DN4000')
    if dn4000 is None:
        dn4000 = np.zeros(len(z_obs))

    # For SV3 Analysis we can pretend to not have observed some galaxies
    # TODO BUG this procedure is not right accordin to Ashley Ross
    if data_cut == "sv3":
        unobserved = drop_SV3_passes(drop_passes, tileid, unobserved)

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
    
    # Special version cut to look like SV3 - choose only the ones inside the SV3 footprint
    if data_cut == "Y3-Kibo-SV3Cut" or data_cut == "Y3-Loa-SV3Cut":
        ntid_sv3 = get_tbl_column(table, 'NEAREST_TILEIDS_SV3', required=True)[:,0] # just need to nearest tile for our purposes
        region = tile_to_region(ntid_sv3)
        to_remove = np.isin(region, sv3_poor_y3overlap)
        in_good_sv3regions = ~to_remove
        multi_pass_filter = np.all([multi_pass_filter, table['NTILE_MINE_SV3'] >= 10, in_good_sv3regions], axis=0)

    # Roughly remove HII regions of low z, high angular size galaxies (SGA catalog)
    if maskbits is not None and ref_cat is not None:
        BITMASK_SGA = 0x1000 
        sga_collision = (maskbits & BITMASK_SGA) != 0
        sga_central = ref_cat == b'L3'
        to_remove_blue = sga_collision & ~sga_central & (g_r_apparent < 0.5)
        print(f"{np.sum(to_remove_blue):,} galaxies ({np.sum(to_remove_blue) / len(dec) * 100:.2f}%) have a SGA collision, are not SGA centrals, and are blue enough to remove.")
        no_SGA_Issues = np.invert(to_remove_blue)
    else:
        no_SGA_Issues = np.ones(len(dec), dtype=bool)

    observed_requirements = np.all([galaxy_observed_filter, app_mag_filter, redshift_filter, redshift_hi_filter, deltachi2_filter, no_SGA_Issues], axis=0)

    # treat low deltachi2 as unobserved
    treat_as_unobserved = np.all([galaxy_observed_filter, app_mag_filter, no_SGA_Issues, np.invert(deltachi2_filter)], axis=0)
    #print(f"We have {np.count_nonzero(treat_as_unobserved)} observed galaxies with deltachi2 < 40 to add to the unobserved pool")
    unobserved = np.all([app_mag_filter, np.logical_or(unobserved, treat_as_unobserved)], axis=0)

    if mode == Mode.FIBER_ASSIGNED_ONLY.value: # means 3pass 
        keep = np.all([multi_pass_filter, observed_requirements], axis=0)

    if mode == Mode.NEAREST_NEIGHBOR.value or Mode.is_simple(mode) or Mode.is_photoz_plus(mode):
        keep = np.all([multi_pass_filter, np.logical_or(observed_requirements, unobserved)], axis=0)

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
    log_L_gal = log_L_gal[keep]
    quiescent = quiescent[keep]
    dn4000 = dn4000[keep]
    ntid = ntid[keep]
    z_phot = z_phot[keep]
    unobserved = unobserved[keep]
    unobserved_orginal = unobserved_orginal[keep]

    observed = np.invert(unobserved)
    idx_unobserved = np.flatnonzero(unobserved)
    z_assigned_flag = np.zeros(len(z_obs), dtype=np.int8)

    count = len(dec)
    print(f"{count:,} galaxies left for main catalog after filters.")
    first_need_redshift_count = unobserved.sum()
    print(f'{first_need_redshift_count} ({100*first_need_redshift_count / len(unobserved) :.1f})% need redshifts')

    z_eff = np.copy(z_obs)

    ############################################################################
    # If a lost galaxy matches the SDSS catalog, grab it's redshift and use that
    ############################################################################
    if unobserved.sum() > 0 and sdss_fill:
        sdss_vanilla = deserialize(SDSSGroupCatalog("SDSS Vanilla v2", SDSS_v2_DAT_FILE, SDSS_v2_GALPROPS_FILE))
        if sdss_vanilla.all_data is not None:
            sdss_has_specz = z_flag_is_spectro_z(sdss_vanilla.all_data.z_assigned_flag)
            observed_sdss = sdss_vanilla.all_data.loc[sdss_has_specz]

            sdss_catalog = coord.SkyCoord(ra=observed_sdss.RA.to_numpy()*u.degree, dec=observed_sdss.Dec.to_numpy()*u.degree, frame='icrs')
            to_match = coord.SkyCoord(ra=ra[idx_unobserved]*u.degree, dec=dec[idx_unobserved]*u.degree, frame='icrs')
            print(f"Matching {len(to_match):,} lost galaxies to {len(sdss_catalog):,} SDSS galaxies")
            idx, d2d, d3d = coord.match_coordinates_sky(to_match, sdss_catalog, nthneighbor=1, storekdtree=False)
            ang_dist = d2d.to(u.arcsec).value
            sdss_z = sdss_vanilla.all_data.iloc[idx]['z'].to_numpy()

            # if angular distance is < 3", then we consider it a match to SDSS catalog and copy over it's z
            ANGULAR_DISTANCE_MATCH = 3
            matched = ang_dist < ANGULAR_DISTANCE_MATCH
            idx_from_sloan = idx_unobserved[matched]
            
            z_eff[idx_unobserved] = np.where(matched, sdss_z, np.nan)    
            z_assigned_flag[idx_unobserved] = np.where(matched, AssignedRedshiftFlag.SDSS_SPEC.value, z_assigned_flag[idx_unobserved])
            
            update_properties_for_indices(idx_from_sloan, app_mag_r, app_mag_g, g_r_apparent, z_eff, abs_mag_R, abs_mag_R_k, abs_mag_G, abs_mag_G_k, log_L_gal, quiescent)
            unobserved[idx_unobserved] = np.where(matched, False, unobserved[idx_unobserved])
            observed = np.invert(unobserved)
            idx_unobserved = np.flatnonzero(unobserved)
     
            print(f"{matched.sum():,} of {first_need_redshift_count:,} redshifts taken from SDSS.")
            print(f"{unobserved.sum():,} remaining galaxies need redshifts.")
            #print(f"z_eff, after SDSS match: {z_eff[0:20]}")   
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

    # We need to guess a color (target_quiescent) for the unobserved galaxies to help the redshift guesser
    if Mode.is_simple(mode) or Mode.is_photoz_plus(mode):
        # Multiple possible ideas
        # 1) Could use the NN's redshift to k-correct but I don't like it
        # 2) Use the lost galaxies' photometric redshift to k-correct
        if have_z_phot:
            lost_abs_mag_R = app_mag_to_abs_mag(app_mag_r[idx_unobserved], z_phot[idx_unobserved])
            lost_abs_mag_R_k = k_correct(lost_abs_mag_R, z_phot[idx_unobserved], g_r_apparent[idx_unobserved])
            lost_abs_mag_G = app_mag_to_abs_mag(app_mag_g[idx_unobserved], z_phot[idx_unobserved])
            lost_abs_mag_G_k = k_correct(lost_abs_mag_G, z_phot[idx_unobserved], g_r_apparent[idx_unobserved], band='g')
            #lost_log_L_gal = abs_mag_r_to_log_solar_L(lost_abs_mag_R_k)
            lost_G_R_k = lost_abs_mag_G_k - lost_abs_mag_R_k
            target_quiescent = is_quiescent_BGS_gmr(None, lost_G_R_k)
        else:
        # 3) Use an uncorrected apparent g-r color cut to guess if the galaxy is quiescent or not
            target_quiescent = is_quiescent_lost_gal_guess(app_mag_g[idx_unobserved] - app_mag_r[idx_unobserved]).astype(int)
        
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
        with PhotometricRedshiftGuesser.from_files(BGS_Y3_LOST_APP_TO_Z_FILE, BGS_Y3_LOST_APP_AND_ZPHOT_TO_Z_FILE, NEIGHBOR_ANALYSIS_SV3_BINS_SMOOTHED_FILE, Mode(mode)) as scorer:
            print(f"Assigning missing redshifts... ")   

            if wants_MCMC:
               MAX_NEIGHBORS = 20
            else:
                MAX_NEIGHBORS = NEIGHBORS
                params = (BB_PARAMS, RB_PARAMS, BR_PARAMS, RR_PARAMS)

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
                    raise ValueError("MCMC optimization of parameters is only possible with SV3 and dropped passes.")
                
                print("Performing MCMC optimization of PhotometricRedshiftGuesser parameters")
                # Can only use the galaxies that were observed but we're pretending are unobserved 
                idx =  np.flatnonzero(np.logical_and(~unobserved_orginal, unobserved))
                # from the neighbor arrays, need to discard the ones that are not in the idx
                n_selector = (~unobserved_orginal)[unobserved] # True/False array of the ones that were observed but we're pretending are unobserved of length idx_unobserved
                
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

    # Redshift assignments could have placed lost galaxies outside the range of the catalog. Remove them.
    final_selection = np.logical_and(z_eff > Z_MIN, z_eff < Z_MAX)
    print(f"{np.sum(~final_selection):,} galaxies have redshifts outside the range of the catalog and will be removed.")
    
    print(f"Neighbor usage %: {z_flag_is_neighbor(z_assigned_flag).sum() / z_flag_is_not_spectro_z(z_assigned_flag).sum() * 100:.2f}")

    # Now that we have redshifts for lost galaxies, we can calculate the rest of the properties
    if len(idx_unobserved) > 0:
        G_R_k = update_properties_for_indices(idx_unobserved, app_mag_r, app_mag_g, g_r_apparent, z_eff, abs_mag_R, abs_mag_R_k, abs_mag_G, abs_mag_G_k, log_L_gal, quiescent)
    else:
        G_R_k = abs_mag_G_k - abs_mag_R_k

    print(f"Catalog contains {quiescent.sum():,} quiescent and {len(quiescent) - quiescent.sum():,} star-forming galaxies")
     #print(f"Quiescent agreement between g-r and Dn4000 for observed galaxies: {np.sum(quiescent_gmr[observed] == quiescent[observed]) / np.sum(observed)}")

    # the vmax should be calculated from un-k-corrected magnitudes
    V_max = get_max_observable_volume(abs_mag_R, z_eff, APP_MAG_CUT, frac_area)

    # TODO get galaxy concentration from somewhere
    chi = np.zeros(count, dtype=np.int8) 

    ####################################################################################
    # Write the completed preprocess files for the group finder / post-processing to use
    ####################################################################################
    t1 = time.time()
    galprops= pd.DataFrame({
        'app_mag': app_mag_r[final_selection].astype("<f8"),
        'target_id': target_id[final_selection].astype("<i8"),
        'z_assigned_flag': z_assigned_flag[final_selection].astype("<i1"),
        'g_r': G_R_k[final_selection].astype("<f8"), # TODO name this G_R_k ?
        'Dn4000': dn4000[final_selection].astype("<f8"),
        'nearest_tile_id': ntid[final_selection].astype("<i8"),
        'z_phot': z_phot[final_selection].astype("<f8"),
        'z_obs': z_obs[final_selection].astype("<f8"),
    })
    galprops.to_pickle(outname_base + "_galprops.pkl")  
    t2 = time.time()
    print(f"Galprops pickling took {t2-t1:.4f} seconds")

    write_dat_files_v2(ra[final_selection], dec[final_selection], z_eff[final_selection], log_L_gal[final_selection], V_max[final_selection], quiescent[final_selection], chi[final_selection], outname_base)

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
        pos[i] *= np.random.uniform(0.90, 0.999, size=13)


    if mode == Mode.PHOTOZ_PLUS_v1.value:
        backfile = BASE_FOLDER + "mcmc13_m4_1_7.h5"
    elif mode == Mode.PHOTOZ_PLUS_v2.value:
        backfile = BASE_FOLDER + "mcmc13_m4_2_4.h5"
    elif mode == Mode.PHOTOZ_PLUS_v3.value:
        backfile = BASE_FOLDER + "mcmc13_m4_3_1.h5"
    if os.path.exists(backfile):
        new = False
        print("Loaded existing MCMC sampler")
        backend = emcee.backends.HDFBackend(backfile)
        n_walkers = backend.shape[0]
    else:
        new = True
        backend = emcee.backends.HDFBackend(backfile)
        backend.reset(n_walkers, ndim)

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability, args=(scorer, app_mag_r, p_obs, z_phot, t_q, ang_dist, n_z, n_q, z_truth), backend=backend, pool=Pool())

    print("Running MCMC...")
    #if not new:
    #    pos = None
    sampler.run_mcmc(pos, n_steps, progress=True)

    samples = sampler.get_chain(flat=True)
    params = samples[np.argmax(sampler.get_log_prob())]
    bb, rb, br, rr = np.reshape(params[1:], (4,4))
    return params[0], (bb, rb, br, rr)


##########################
# Processing Group Finder Output File
##########################

def read_and_combine_gf_output(gc: GroupCatalog, galprops_df):
    # TODO instead of reading GF output from disk, have option to just keep in memory
    main_df = pd.read_csv(gc.GF_outfile, delimiter=' ', names=('RA', 'Dec', 'z', 'L_gal', 'V_max', 'P_sat', 'M_halo', 'N_sat', 'L_tot', 'igrp', 'weight'))
    df = pd.merge(main_df, galprops_df, left_index=True, right_index=True)

    # Drop bad data, should have been cleaned up earlier though!
    orig_count = len(df)
    df = df[df.M_halo != 0]
    new_count = len(df)
    if (orig_count != new_count):
        print("Dropped {0} bad galaxies".format(orig_count - new_count))

    # add columns indicating if galaxy is a satellite
    df['is_sat'] = (df.index != df.igrp).astype(int)
    df['logLgal'] = np.log10(df.L_gal)

    # add column for halo mass bins and Lgal bins
    df['Mh_bin'] = pd.cut(x = df['M_halo'], bins = gc.Mhalo_bins, labels = gc.Mhalo_labels, include_lowest = True)
    df['Lgal_bin'] = pd.cut(x = df['L_gal'], bins = gc.L_gal_bins, labels = gc.L_gal_labels, include_lowest = True)

    return df # TODO update callers



# TODO might be wise to double check that my manual calculation of q vs sf matches what the group finder was fed by
# checking the input data. I think for now it should be exactly the same, but maybe we want it to be different for
# apples to apples comparison between BGS and SDSS



##########################
# Aggregation Helpers
##########################

def count_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        return len(series) / np.average(series.V_max)

def fsat_truth_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        return np.average(series.is_sat_truth, weights=1/series.V_max)
    
def fsat_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        return np.average(series.is_sat, weights=1/series.V_max)

def Lgal_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        return np.average(series.L_gal, weights=1/series.V_max)

# TODO not sure right way to do std error for this sort of data
def Lgal_std_vmax_weighted(series):
    """Lognormal error on Lgal"""
    if len(series) == 0:
        return 0
    else:
        return np.power(10, np.sqrt(np.average((np.log(series.L_gal) - np.log(Lgal_vmax_weighted(series)))**2, weights=1/series.V_max)))

def z_flag_is_spectro_z(arr):
    return np.logical_or(arr == AssignedRedshiftFlag.SDSS_SPEC.value, arr == AssignedRedshiftFlag.DESI_SPEC.value)

def z_flag_is_neighbor(arr):
    return arr >= AssignedRedshiftFlag.NEIGHBOR_ONE.value

def z_flag_is_random(arr):
    return arr == AssignedRedshiftFlag.PSEUDO_RANDOM.value

def z_flag_is_not_spectro_z(arr):
    return ~z_flag_is_spectro_z(arr)

def mstar_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        if 'z_assigned_flag' in series.columns:
            should_mask = np.logical_or(series.z_assigned_flag != 0, np.isnan(series.mstar))
        else:
            should_mask = np.isnan(series.mstar)
        masked_mstar = np.ma.masked_array(series.mstar, should_mask)
        masked_vmax = np.ma.masked_array(series.V_max, should_mask)
        return np.average(masked_mstar, weights=1/masked_vmax)

# TODO not sure right way to do std error for this sort of data
def mstar_std_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        should_mask = np.logical_or(series.z_assigned_flag != 0, np.isnan(series.mstar))
        masked_mstar = np.ma.masked_array(series.mstar, should_mask)
        masked_vmax = np.ma.masked_array(series.V_max, should_mask)
        return np.sqrt(np.average((masked_mstar - mstar_vmax_weighted(series))**2, weights=1/masked_vmax))

def qf_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        return np.average(series.quiescent, weights=1/series.V_max)

def qf_Dn4000_1_6_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        return np.average((series.Dn4000 >  1.6), weights=1/series.V_max)

def qf_Dn4000_smart_eq_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        return np.average(is_quiescent_BGS_smart(series.logLgal, series.Dn4000, series.g_r), weights=1/series.V_max)

def qf_BGS_gmr_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        return np.average(is_quiescent_BGS_gmr(series.logLgal, series.g_r), weights=1/series.V_max)
    
def nsat_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        print(series.N_sat)
        return np.average(series.N_sat, weights=1/series.V_max)
    

