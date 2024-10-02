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
import wp as wp
import asyncio

if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from pyutils import *
from redshift_guesser import SimpleRedshiftGuesser, PhotometricRedshiftGuesser
from hdf5_to_dat import pre_process_mxxl
from uchuu_to_dat import pre_process_uchuu


# Shared bins for various purposes
Mhalo_bins = np.logspace(10, 15.5, 40)
Mhalo_labels = Mhalo_bins[0:len(Mhalo_bins)-1] 

L_gal_bins = np.logspace(6, 12.5, 40)
L_gal_labels = L_gal_bins[0:len(L_gal_bins)-1]

mstar_bins = np.logspace(6, 13, 40)
mstar_labels = mstar_bins[0:len(mstar_bins)-1]

Mr_gal_labels = log_solar_L_to_abs_mag_r(np.log10(L_gal_labels))

# I built this list of tiles by looking at https://www.legacysurvey.org/viewer-desi and viewing DESI EDR tiles (look for SV3)
sv3_regions = [
    [122, 128, 125, 124, 120, 127, 126, 121, 123, 129],
    [499, 497, 503, 500, 495, 502, 501, 496, 498, 504],
    [14,  16,  20,  19,  13,  12,  21,  18,  15,  17 ],
    [41,  47,  49,  44,  43,  39,  46,  45,  40,  42,  48],
    [68,  74,  76,  71,  70,  66,  73,  72,  67,  69,  75],
    [149, 155, 152, 147, 151, 154, 148, 156, 150, 153], 
    [527, 533, 530, 529, 525, 532, 531, 526, 528, 534], 
    [236, 233, 230, 228, 238, 234, 232, 231, 235, 237, 229],
    [265, 259, 257, 262, 263, 256, 260, 264, 255, 258, 261],
    [286, 284, 289, 290, 283, 287, 291, 282, 285, 288],
    [211, 205, 203, 208, 209, 202, 206, 210, 201, 204, 207],
    [397, 394, 391, 400, 399, 392, 393, 398, 396, 395, 390],
    [373, 365, 371, 367, 368, 363, 369, 370, 366, 364, 372],
    [346, 338, 340, 344, 343, 341, 336, 342, 339, 337, 345],
    [592, 589, 586, 595, 587, 593, 590, 594, 585, 588, 591],
    [313, 316, 319, 311, 317, 314, 309, 310, 318, 312, 315],
    [176, 182, 184, 179, 178, 174, 181, 180, 175, 177, 183],
    [564, 558, 556, 561, 562, 555, 559, 560, 565, 563, 557],
    [421, 424, 427, 419, 425, 422, 417, 423, 418, 420, 426],
    [95,  101, 103, 98,  97,  93,  100, 99,  94,  96,  102],
]
sv3_regions_sorted = []
for region in sv3_regions:
    a = region.copy()
    a.sort()
    sv3_regions_sorted.append(a)

# Build a dictionary of tile_id to region index
sv3_tile_to_region = {}
for i, region in enumerate(sv3_regions):
    for tile in region:
        sv3_tile_to_region[tile] = i
def tile_to_region_raw(key):
    return sv3_tile_to_region.get(key, None)  # Return None if key is not found
tile_to_region = np.vectorize(tile_to_region_raw)

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

        self.has_truth = False
        self.Mhalo_bins = Mhalo_bins
        self.labels = Mhalo_labels
        self.all_data: pd.DataFrame = None
        self.centrals: pd.DataFrame = None
        self.sats: pd.DataFrame = None
        self.L_gal_bins = L_gal_bins
        self.L_gal_labels = L_gal_labels

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

    def get_completeness(self):
        return np.sum(self.all_data.z_assigned_flag == 0) / len(self.all_data.z_assigned_flag)

    def get_lostgal_neighbor_used(self):
        arr = self.all_data.z_assigned_flag.to_numpy()
        return np.sum(z_flag_is_neighbor(arr)) / (np.sum(z_flag_is_random(arr)) + np.sum(z_flag_is_neighbor(arr)))

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
            self.f_sat_sf = self.all_data[self.all_data.quiescent == False].groupby('Lgal_bin', observed=False).apply(fsat_vmax_weighted)
            self.f_sat_q = self.all_data[self.all_data.quiescent == True].groupby('Lgal_bin', observed=False).apply(fsat_vmax_weighted)
            self.Lgal_counts = self.all_data.groupby('Lgal_bin', observed=False).RA.count()

            # Setup some convenience subsets of the DataFrame
            # TODO check memory implications of this
            self.centrals = self.all_data.loc[self.all_data.index == self.all_data.igrp]
            self.sats = self.all_data.loc[self.all_data.index != self.all_data.igrp]

    def postprocess(self):
        if self.all_data is not None:
            self.refresh_df_views()
        else:
            print("Warning: postprocess called with all_data DataFrame is not set yet. Override postprocess() or after calling run_group_finder() set it.")

    def get_true_z_from(self, truth_df: pd.DataFrame):
        """
        Adds a column to the catalog's all_data DataFrame with the true redshifts from the truth_df DataFrame 
        for rows with z_assigned_flag != 0.
        """
        truth_df = truth_df[['target_id', 'z', 'z_assigned_flag', 'L_gal', 'logLgal', 'g_r', 'Lgal_bin']].copy()
        truth_df.index = truth_df.target_id
        self.all_data = self.all_data.join(truth_df, on='target_id', how='left', rsuffix='_T')
        rows_to_nan = self.all_data['z_assigned_flag_T'] != AssignedRedshiftFlag.DESI_SPEC.value
        rows_to_nan = rows_to_nan & self.all_data['z_assigned_flag_T'] != AssignedRedshiftFlag.SDSS_SPEC.value
        self.all_data.loc[rows_to_nan, 'z_T'] = -99.99
        self.all_data.drop(columns=['target_id_T', 'z_assigned_flag_T'], inplace=True)
        self.has_truth = True

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

        self.volume = np.array([1.721e+06, 6.385e+06, 2.291e+07, 7.852e+07]) # Copied from Jeremy's groupfind_mcmc.py
        self.vfac = (self.volume/250.0**3)**.5 # factor by which to multiply errors
        self.efac = 0.1 # let's just add a constant fractional error bar

    def postprocess(self):
        galprops = pd.read_csv(self.galprops_file, delimiter=' ', names=('Mag_g', 'Mag_r', 'sigma_v', 'Dn4000', 'concentration', 'log_M_star', 'z_assigned_flag'))
        galprops['g_r'] = galprops.Mag_g - galprops.Mag_r 
        galprops.rename(columns={'Mag_r': "app_mag"}, inplace=True)
        self.all_data = read_and_combine_gf_output(self, galprops)
        self.all_data['quiescent'] = is_quiescent_SDSS_Dn4000(self.all_data.logLgal, self.all_data.Dn4000)
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
        galprops = pd.read_csv(SDSS_v1_GALPROPS_FILE, delimiter=' ', names=('Mag_g', 'Mag_r', 'sigma_v', 'Dn4000', 'concentration', 'log_M_star'))
        galprops['g_r'] = galprops.Mag_g - galprops.Mag_r 
        galprops.rename(columns={'Mag_r': "app_mag"}, inplace=True)
        print(len(galprops))
        
        main_df = pd.read_csv(self.GF_outfile, delimiter=' ', names=('RA', 'Dec', 'z', 'L_gal', 'V_max', 'P_sat', 'M_halo', 'N_sat', 'L_tot', 'igrp', 'weight'))
        print(len(main_df))
        
        df = pd.merge(main_df, galprops, left_index=True, right_index=True)

        # add columns indicating if galaxy is a satellite
        df['is_sat'] = (df.index != df.igrp).astype(int)
        df['logLgal'] = np.log10(df.L_gal)

        # add column for halo mass bins and Lgal bins
        df['Mh_bin'] = pd.cut(x = df['M_halo'], bins = Mhalo_bins, labels = Mhalo_labels, include_lowest = True)
        df['Lgal_bin'] = pd.cut(x = df['L_gal'], bins = L_gal_bins, labels = L_gal_labels, include_lowest = True)

        self.all_data = df
        self.all_data['quiescent'] = is_quiescent_SDSS_Dn4000(self.all_data.logLgal, self.all_data.Dn4000)
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
        galprops = pd.read_csv(SDSS_v1_GALPROPS_FILE, delimiter=' ', names=('Mag_g', 'Mag_r', 'sigma_v', 'Dn4000', 'concentration', 'log_M_star'))

        cut_gals = gals[np.logical_and(gals.ra > 149.119, gals.ra < 151.119)]
        cut_gals = cut_gals[np.logical_and(cut_gals.dec > 1.205, cut_gals.dec < 3.205)]
        indexes = cut_gals.index
        print(f"Cut to {len(indexes)} galaxies.")
        cut_galprops = galprops.iloc[indexes]

        # write to TEST_DAT_FILE and TEST_GALPROPS_FILE
        cut_gals.to_csv(TEST_DAT_FILE, sep=' ', header=False, index=False)
        cut_galprops.to_csv(TEST_GALPROPS_FILE, sep=' ', header=False, index=False)

    def postprocess(self):
        galprops = pd.read_csv(TEST_GALPROPS_FILE, delimiter=' ', names=('Mag_g', 'Mag_r', 'sigma_v', 'Dn4000', 'concentration', 'log_M_star'))
        galprops['g_r'] = galprops.Mag_g - galprops.Mag_r 
        galprops.rename(columns={'Mag_r': "app_mag"}, inplace=True)
        self.all_data = read_and_combine_gf_output(self, galprops)
        self.all_data['quiescent'] = is_quiescent_SDSS_Dn4000(self.all_data.logLgal, self.all_data.Dn4000)
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
        self.has_truth = self.mode.value == Mode.ALL.value
        df['is_sat_truth'] = np.logical_or(df.galaxy_type == 1, df.galaxy_type == 3)
        if self.has_truth:
            df['Mh_bin_T'] = pd.cut(x = df['mxxl_halo_mass']*10**10, bins = Mhalo_bins, labels = Mhalo_labels, include_lowest = True)
            df['L_gal_T'] = np.power(10, abs_mag_r_to_log_solar_L(app_mag_to_abs_mag_k(df.app_mag.to_numpy(), df.z_obs.to_numpy(), df.g_r.to_numpy())))
            df['Lgal_bin_T'] = pd.cut(x = df['L_gal_T'], bins = L_gal_bins, labels = L_gal_labels, include_lowest = True)
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

    def __init__(self, name, mode: Mode, mag_cut: float, catalog_mag_cut: float, sdss_fill: bool = True, num_passes: int = 3, drop_passes: int = 0, data_cut: str = "Y1-Iron"):
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

    def preprocess(self):
        print("Pre-processing...")
        if self.data_cut == "Y1-Iron":
            infile = IAN_BGS_MERGED_FILE
        elif self.data_cut == "Y1-Iron-v1.2":
            infile = IAN_BGS_MERGED_FILE_OLD
        elif self.data_cut == "Y3-Kibo":
            infile = IAN_BGS_Y3_MERGED_FILE
        elif self.data_cut == "Y3-Jura":
            infile = IAN_BGS_Y3_MERGED_FILE_JURA
        elif self.data_cut == "sv3":
            infile = IAN_BGS_SV3_MERGED_FILE
        else:
            raise ValueError("Unknown data_cut value")
        
        fname, props = pre_process_BGS(infile, self.mode.value, self.file_pattern, self.mag_cut, self.catalog_mag_cut, self.sdss_fill, self.num_passes, self.drop_passes, self.data_cut)
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
            # label the SV3 region each galaxy is in
            df['region'] = tile_to_region(df['nearest_tile_id'])

            # Add bootstrapped error bars for fsat
            f_sat_realizations = []
            f_sat_sf_realizations = []
            f_sat_q_realizations = []

            for i in range(N_ITERATIONS):
                region_indices = np.random.choice(range(len(sv3_regions_sorted)), len(sv3_regions_sorted), replace=True)
                alt_df = pd.DataFrame(columns=df.columns)
                for idx in region_indices:
                    rows_to_add = df.loc[df.region == idx].copy()
                    if len(rows_to_add) > 0:
                        alt_df = alt_df._append(rows_to_add, ignore_index=True)
                f_sat_realizations.append(alt_df.groupby('Lgal_bin', observed=False).apply(fsat_vmax_weighted))
                f_sat_sf_realizations.append(alt_df[alt_df.quiescent == False].groupby('Lgal_bin', observed=False).apply(fsat_vmax_weighted))
                f_sat_q_realizations.append(alt_df[alt_df.quiescent == True].groupby('Lgal_bin', observed=False).apply(fsat_vmax_weighted))

            self.f_sat_err = np.std(f_sat_realizations, axis=0)
            self.f_sat_sf_err = np.std(f_sat_sf_realizations, axis=0)
            self.f_sat_q_err = np.std(f_sat_q_realizations, axis=0)
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

        print("Post-processing done.")


    def refresh_df_views(self):
        super().refresh_df_views()
        #self.add_bootstrapped_f_sat()

    def write_sharable_output_file(self):
        print("Writing a sharable output file")
        filename_out = str.replace(self.GF_outfile, ".out", " Catalog.csv")
        df = self.all_data.drop(columns=['Mstar_bin', 'Mh_bin', 'Lgal_bin', 'logLgal', 'Dn4000'])
        print(df.columns)
        df.to_csv(filename_out, index=False, header=True)


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

def pre_process_BGS(fname, mode, outname_base, APP_MAG_CUT, CATALOG_APP_MAG_CUT, sdss_fill, num_passes_required, drop_passes, data_cut):
    """
    Pre-processes the BGS data for use with the group finder.
    """
    Z_MIN = 0.001 # BUG The Group Finder blows up if you lower this
    Z_MAX = 0.8

    # TODO BUG One galaxy is lost from this to group finder...
    
    # TODO conversion of fluxes to apparent and absolute mags can be done ahead of time
    # so use that directly from the merged file.


    print("Reading FITS data from ", fname)
    # Unobserved galaxies have masked rows in appropriate columns of the table
    table = Table.read(fname, format='fits')

    if drop_passes > 0 and data_cut != "sv3":
        raise ValueError("Dropping passes is only for the sv3 study")
    
    # These are calculated from randoms in BGS_study.ipynb
    if data_cut == "Y1-Iron" or data_cut == "Y1-Iron-v1.2":
        # For Y1-Iron  
        FOOTPRINT_FRAC_1pass = 0.1876002 # 7739 degrees
        FOORPRINT_FRAC_2pass = 0.1153344 # 4758 degrees
        FOOTPRINT_FRAC_3pass = 0.0649677 # 2680 degrees
        FOOTPRINT_FRAC_4pass = 0.0228093 # 940 degrees
        # 0% 5pass coverage
    elif data_cut == "Y3-Jura":
        # For Y3-Jura
        FOOTPRINT_FRAC_1pass = 0.310691 # 12816 degrees
        FOORPRINT_FRAC_2pass = 0.286837 # 11832 degrees
        FOOTPRINT_FRAC_3pass = 0.233920 # 9649 degrees
        FOOTPRINT_FRAC_4pass = 0.115183 # 4751 degrees
    elif data_cut == "Y3-Kibo":
        # For Y3-Kibo
        FOOTPRINT_FRAC_1pass = 0.3112278 # 12839 degrees
        FOORPRINT_FRAC_2pass = 0.2870291 # 11840 degrees
        FOOTPRINT_FRAC_3pass = 0.2338243 # 9645 degrees
        FOOTPRINT_FRAC_4pass = 0.1150345 # 4745 degrees
    elif data_cut == "sv3":
        FOOTPRINT_FRAC_1pass = 173.87 / DEGREES_ON_SPHERE 
        FOOTPRINT_FRAC_10pass = 138.192 / DEGREES_ON_SPHERE 
    else:
        print("Invalid data cut. Exiting.")
        exit(2)

    # TODO update footprint with new calculation from ANY. It shouldn't change.
    if mode == Mode.ALL.value or num_passes_required == 1:
        frac_area = FOOTPRINT_FRAC_1pass
    elif num_passes_required == 2:
        frac_area = FOORPRINT_FRAC_2pass
    elif num_passes_required == 3:
        frac_area = FOOTPRINT_FRAC_3pass
    elif num_passes_required == 4:
        frac_area = FOOTPRINT_FRAC_4pass
    elif num_passes_required == 10:
        frac_area = FOOTPRINT_FRAC_10pass
    else:
        print(f"Need footprint calculation for num_passes_required = {num_passes_required}. Exiting")
        exit(2)

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

    # Some versions of the LSS Catalogs use astropy's Table used masked arrays for unobserved spectral targets    
    if np.ma.is_masked(table['Z']):
        z_obs = table['Z'].data.data
        unobserved = table['Z'].mask # the masked values are what is unobserved
    else:
        # SV3 version didn't do this
        z_obs = table['Z']
        unobserved = table['Z'].astype("<i8") == 999999

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
        print("WARNING: PROB_OBS column not found in FITS file. Using 0.5 for all unobserved galaxies.")
        p_obs = np.ones(len(z_obs)) * 0.5
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
    print(orig_count, "objects in FITS file")

    # If an observation was made, some automated system will evaluate the spectra and auto classify the SPECTYPE
    # as GALAXY, QSO, STAR. It is null (and masked) for non-observed targets.
    # NTILE tracks how many DESI pointings could have observed the target (at fiber level)
    # NTILE_MINE gives how many tiles include just from inclusion in circles drawn around tile centers
    # null values (masked rows) are unobserved targets; not all columns are masked though

    # Make filter arrays (True/False values)
    multi_pass_filter = table['NTILE_MINE'] >= num_passes_required
    galaxy_observed_filter = obj_type == b'GALAXY'
    app_mag_filter = app_mag_r < APP_MAG_CUT
    redshift_filter = z_obs > Z_MIN
    redshift_hi_filter = z_obs < Z_MAX
    deltachi2_filter = deltachi2 > 40 # Ensures that there wasn't another z with similar likelihood from the z fitting code
    
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

    if mode == Mode.NEAREST_NEIGHBOR.value or mode == Mode.SIMPLE.value or mode == Mode.SIMPLE_v4.value or mode == Mode.SIMPLE_v5.value or mode == Mode.PHOTOZ_PLUS_v1.value:
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

    observed = np.invert(unobserved)
    idx_unobserved = np.flatnonzero(unobserved)
    z_assigned_flag = np.zeros(len(z_obs), dtype=np.int8)

    count = len(dec)
    print(count, "galaxies left for main catalog after filters.")
    first_need_redshift_count = unobserved.sum()
    print(f'{first_need_redshift_count} ({100*first_need_redshift_count / len(unobserved) :.1f})% need redshifts')

    z_eff = np.copy(z_obs)

    # If a lost galaxy matches the SDSS catalog, grab it's redshift and use that
    if unobserved.sum() > 0 and sdss_fill:
        sdss_vanilla = deserialize(SDSSGroupCatalog("SDSS Vanilla v2", SDSS_v2_DAT_FILE, SDSS_v2_GALPROPS_FILE))
        #sdss_vanilla = deserialize(SDSSGroupCatalog("SDSS Vanilla", SDSS_v1_DAT_FILE, SDSS_v1_GALPROPS_FILE))
        if sdss_vanilla.all_data is not None:
            observed_sdss = sdss_vanilla.all_data.loc[sdss_vanilla.all_data.z_assigned_flag == 0]

            sdss_catalog = coord.SkyCoord(ra=observed_sdss.RA.to_numpy()*u.degree, dec=observed_sdss.Dec.to_numpy()*u.degree, frame='icrs')
            to_match = coord.SkyCoord(ra=ra[idx_unobserved]*u.degree, dec=dec[idx_unobserved]*u.degree, frame='icrs')
            print(f"Matching {len(to_match):,} lost galaxies to {len(sdss_catalog):,} SDSS galaxies")
            idx, d2d, d3d = coord.match_coordinates_sky(to_match, sdss_catalog, nthneighbor=1, storekdtree=False)
            ang_distances = d2d.to(u.arcsec).value
            sdss_z = sdss_vanilla.all_data.iloc[idx]['z'].to_numpy()

            # if angular distance is < 3", then we consider it a match to SDSS catalog and copy over it's z
            ANGULAR_DISTANCE_MATCH = 3
            matched = ang_distances < ANGULAR_DISTANCE_MATCH
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

    if mode == Mode.SIMPLE.value or mode == Mode.SIMPLE_v4.value or mode == Mode.SIMPLE_v5.value:
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
            ang_distances = d2d.to(u.arcsec).value

            # We need to guess a color for the unobserved galaxies to help the redshift guesser
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
            
            assert len(target_quiescent) == len(ang_distances)

            chosen_z, isNN = scorer.choose_redshift(z_obs_catalog[neighbor_indexes], ang_distances, p_obs[idx_unobserved], app_mag_r[idx_unobserved], target_quiescent, catalog_quiescent[neighbor_indexes])
            z_eff[idx_unobserved] = chosen_z
            z_assigned_flag[idx_unobserved] = np.where(isNN, AssignedRedshiftFlag.NEIGHBOR_ONE.value, AssignedRedshiftFlag.PSEUDO_RANDOM.value)
            print(f"Assigning missing redshifts complete.")   

    if mode == Mode.PHOTOZ_PLUS_v1.value:
        with PhotometricRedshiftGuesser.from_files(IAN_MXXL_LOST_APP_TO_Z_FILE, NEIGHBOR_ANALYSIS_SV3_BINS_FILE) as scorer:
            NEIGHBORS = 5
            print(f"Considering {NEIGHBORS} neighbors for redshift assignment")
            print(f"Assigning missing redshifts... ")   

            catalog = coord.SkyCoord(ra=catalog_ra*u.degree, dec=catalog_dec*u.degree, frame='icrs')
            to_match = coord.SkyCoord(ra=ra[idx_unobserved]*u.degree, dec=dec[idx_unobserved]*u.degree, frame='icrs')
           
            target_quiescent = is_quiescent_lost_gal_guess(app_mag_g[idx_unobserved] - app_mag_r[idx_unobserved]).astype(int)

            shape = (NEIGHBORS, len(to_match))
            neighbor_indexes = np.zeros(shape, dtype=np.int64)
            n_z = np.zeros(shape, dtype=np.float64)
            ang_distances = np.zeros(shape, dtype=np.float64)
            n_q = np.zeros(shape, dtype=np.float64)

            for n in range(NEIGHBORS):
                neighbor_indexes[n, :], d2d, d3d = coord.match_coordinates_sky(to_match, catalog, nthneighbor=n+1, storekdtree='nn_kdtree')
                ang_distances[n, :] = d2d.to(u.arcsec).value
                n_z[n, :] = z_obs_catalog[neighbor_indexes[n, :]]
                n_q[n, :] = catalog_quiescent[neighbor_indexes[n, :]]

            chosen_z, neighbor_used = scorer.choose_redshift(n_z, ang_distances, z_phot[idx_unobserved], p_obs[idx_unobserved], app_mag_r[idx_unobserved], target_quiescent, n_q)
            z_eff[idx_unobserved] = chosen_z
            z_assigned_flag[idx_unobserved] = np.where(np.isnan(neighbor_used), AssignedRedshiftFlag.PSEUDO_RANDOM.value, neighbor_used)

            print(f"Assigning missing redshifts complete.")   

    #print(f"z_eff, after assignment: {z_eff[0:20]}")   
    assert np.all(z_eff > 0.0)

    # Now that we have redshifts for lost galaxies, we can calculate the rest of the properties
    idx_unobserved = np.flatnonzero(unobserved)
    if len(idx_unobserved) > 0:
        G_R_k = update_properties_for_indices(idx_unobserved, app_mag_r, app_mag_g, g_r_apparent, z_eff, abs_mag_R, abs_mag_R_k, abs_mag_G, abs_mag_G_k, log_L_gal, quiescent)
    else:
        G_R_k = abs_mag_G_k - abs_mag_R_k

    print(f"Catalogs contains {quiescent.sum():,} quiescent and {len(quiescent) - quiescent.sum():,} star-forming galaxies")
     #print(f"Quiescent agreement between g-r and Dn4000 for observed galaxies: {np.sum(quiescent_gmr[observed] == quiescent[observed]) / np.sum(observed)}")

    # the vmax should be calculated from un-k-corrected magnitudes
    V_max = get_max_observable_volume(abs_mag_R, z_eff, APP_MAG_CUT, frac_area)

    # TODO get galaxy concentration from somewhere
    chi = np.zeros(count, dtype=np.int8) 

    # Output files
    t1 = time.time()
    galprops= pd.DataFrame({
        'app_mag': app_mag_r.astype("<f8"),
        'target_id': target_id.astype("<i8"),
        'z_assigned_flag': z_assigned_flag.astype("<i1"),
        'g_r': G_R_k.astype("<f8"), # TODO name this G_R_k ?
        'Dn4000': dn4000.astype("<f8"),
        'nearest_tile_id': ntid.astype("<i8"),
        'z_phot': z_phot.astype("<f8"),
        'z_obs': z_obs.astype("<f8"),
    })
    galprops.to_pickle(outname_base + "_galprops.pkl")  
    t2 = time.time()
    print(f"Galprops pickling took {t2-t1:.4f} seconds")

    write_dat_files_v2(ra, dec, z_eff, log_L_gal, V_max, quiescent, chi, outname_base)

    return outname_base + ".dat", {'zmin': np.min(z_eff), 'zmax': np.max(z_eff), 'frac_area': frac_area }







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
    df['Mh_bin'] = pd.cut(x = df['M_halo'], bins = Mhalo_bins, labels = Mhalo_labels, include_lowest = True)
    df['Lgal_bin'] = pd.cut(x = df['L_gal'], bins = L_gal_bins, labels = L_gal_labels, include_lowest = True)

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
    

