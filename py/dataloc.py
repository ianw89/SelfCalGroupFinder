import os

###################################################
# Information on where files are located
###################################################

#  /global/cfs/cdirs/desi/ == https://data.desi.lbl.gov/desi/

# This is the root folder of this repository; update it to your own path
#BASE_FOLDER = '/Users/ianw89/Repos/SelfCalGroupFinder/'
#BASE_FOLDER = os.environ["HOME"] + '/'
BASE_FOLDER = '/mount/sirocco1/imw2293/GROUP_CAT/'
REPO_FOLDER = BASE_FOLDER + 'SelfCalGroupFinder/'

# Subfolders
BIN_FOLDER = REPO_FOLDER + 'bin/'
PY_SRC_FOLDER = REPO_FOLDER + 'py/'
PARAMS_FOLDER = PY_SRC_FOLDER + 'parameters/'
PARAMS_SDSS_FOLDER = PARAMS_FOLDER + 'sdss/'
PARAMS_BGSY1_FOLDER = PARAMS_FOLDER + 'bgs_y1/'
OUTPUT_FOLDER = BASE_FOLDER + 'OUTPUT/'
#OUTPUT_FOLDER = "/Volumes/Seagate Backup Plus Drive/galaxy-groups-data/OUTPUT/"
DATA_FOLDER = BASE_FOLDER + 'DATA/'
#DATA_FOLDER = "/Volumes/Seagate Backup Plus Drive/galaxy-groups-data/DATA/"
SDSS_FOLDER = DATA_FOLDER + 'SDSS/'
BGS_IMAGES_FOLDER = DATA_FOLDER + 'BGS_IMAGES/'
BGS_Y1_FOLDER = DATA_FOLDER + 'BGS_IRON/'
BGS_FUJI_FOLDER = DATA_FOLDER + 'BGS_FUJI/'
BGS_Y3_FOLDER_KIBO = DATA_FOLDER + 'BGS_KIBO/'
BGS_Y3_FOLDER_LOA = DATA_FOLDER + 'BGS_LOA/'
#MXXL_DATA_DIR="/export/sirocco2/tinker/DESI/MXXL_MOCKS/"
MXXL_DATA_DIR=DATA_FOLDER + "MXXL/"
UCHUU_FILES_FOLDER="/export/sirocco2/tinker/DESI/UCHUU_MOCKS/"
K_CORR_PARAMETERS = PY_SRC_FOLDER + 'kcorr/parameters'


# Parameter Files
#################
WP_RADIAL_BINS_SDSS_FILE = PARAMS_SDSS_FOLDER + 'wp_rbins.dat'
WP_RADIAL_BINS_DESI_FILE = PARAMS_BGSY1_FOLDER + 'wp_rbins_desi.dat'
WP_RADIAL_EDGE_DESI_FILE = PARAMS_BGSY1_FOLDER + 'wp_redges_desi.dat'
HALO_MASS_FUNC_FILE = REPO_FOLDER + 'halo_mass_function.dat'
LSAT_LOOKUP_FILE = REPO_FOLDER + 'lsat_lookup.dat'
LSAT_OBSERVATIONS_FILE = PARAMS_SDSS_FOLDER + 'Lsat_SDSS_DnGMM.dat'
MOCK_FILE_FOR_POPMOCK = DATA_FOLDER + 'POPMOCK/' + 'hosthalo_z0.0_M1e10_Lsat.dat'

# SDSS Data Files
#################
# These files were acquried from https://cosmo.nyu.edu/~tinker/GROUP_FINDER/SELFCAL_GROUPS/sdss_fluxlim_v1.0.dat
# They should have been built by Jeremy from the below DR7 NYU VAGC files
SDSS_v1_DAT_FILE = SDSS_FOLDER + 'sdss_fluxlim_v1.0.dat'
SDSS_v2_DAT_FILE = SDSS_FOLDER + 'sdss_fluxlim_v2.0.dat' # Built by me where DESI Y1 redshifts are added to SDSS missing ones
SDSS_BGSCUT_DAT_FILE = SDSS_FOLDER + 'sdss_fluxlim_bgscut.dat' # Built by me where DESI Y1 redshifts are added to SDSS missing ones
SDSS_v1_GALPROPS_FILE = SDSS_FOLDER + "sdss_galprops_v1.0.dat"
SDSS_v1_1_GALPROPS_FILE = SDSS_FOLDER + "sdss_galprops_v1.1.dat"
SDSS_v2_GALPROPS_FILE = SDSS_FOLDER + "sdss_galprops_v2.0.dat"
SDSS_BGSCUT_GALPROPS_FILE = SDSS_FOLDER + "sdss_galprops_bgscut.dat"
# Acquired from http://sdss.physics.nyu.edu/vagc-dr7/vagc2/object_catalog.fits
SDSS_DR7_OBJECT_FILE = SDSS_FOLDER + "object_catalog.fits"
# Acquired from http://sdss.physics.nyu.edu/vagc-dr7/vagc2/collisions/collisions.nearest.fits
SDSS_DR7_COLLISIONS_FILE = SDSS_FOLDER + "collisions.nearest.fits"
# Acquired from http://sdss.physics.nyu.edu/vagc-dr7/vagc2/object_sdss_spectro.fits    
SDSS_DR7_SPECTRO_FILE = SDSS_FOLDER + "object_sdss_spectro.fits"
# Acquired from http://sdss.physics.nyu.edu/vagc/flatfiles/object_sdss_imaging.fits.html
SDSS_DR7_IMAGING_FILE = SDSS_FOLDER + "object_sdss_imaging.fits"

SDSS_DR7B_ID_FILE = SDSS_FOLDER + "id.dr72bright34.dat"
SDSS_DR7B_LSS_FILE = SDSS_FOLDER + "lss.dr72bright34.dat"
SDSS_DR7B_PHOTO_FILE = SDSS_FOLDER + "photoinfo.dr72bright34.dat"

# TEST Data Files
#################
TEST_DAT_FILE = DATA_FOLDER + 'test_mini_fluxlim.dat'
TEST_GALPROPS_FILE = DATA_FOLDER + 'test_mini_galprops.dat'

# Simulation Data Files
#######################
MXXL_FILE = MXXL_DATA_DIR + "weights_3pass.hdf5"
UCHUU_FILE = UCHUU_FILES_FOLDER + "BGS_LC_Uchuu.fits"

# BGS SV3 DATA FILES
###################
BGS_SV3_COMBINED_PHOTOMETRIC_CATALOG = BGS_FUJI_FOLDER + "targetphot-fuji-combined.fits"
# File was acquired from https://data.desi.lbl.gov/desi/survey/catalogs/SV3/LSS/fuji/LSScats/3.1/BGS_ANY_full.dat.fits
BGS_SV3_ANY_FULL_FILE = BGS_FUJI_FOLDER + "BGS_ANY_full.dat.fits"
# File was acquired from https://data.desi.lbl.gov/desi/survey/catalogs/SV3/LSS/fuji/LSScats/3.1/BGS_ANY_X_full.ran.fits where X is 0 to 17
BGS_SV3_RAND_FILE = BGS_FUJI_FOLDER + "BGS_ANY_X_full.ran.fits"
# File was acquired from https://data.desi.lbl.gov/desi/survey/catalogs/SV3/LSS/fuji/Alltiles_bright_tilelocs.dat.fits   
BGS_SV3_PROB_OBS_FILE = BGS_FUJI_FOLDER + "Alltiles_bright_tilelocs.dat.fits"
BGS_SV3_FASTSPEC_FILE = BGS_FUJI_FOLDER + "fastspec-fuji-sv3-bright.fits"

# This file is BUILT by running code in the BGS_study.ipynb notebook
# It is a joined BGS file, with a filtered down set of rows and columns
IAN_BGS_SV3_MERGED_FILE = BGS_FUJI_FOLDER + "ian_BGS_SV3_merged.fits"
IAN_BGS_SV3_MERGED_NOY3_FILE = BGS_FUJI_FOLDER + "ian_BGS_SV3_merged_noY3.fits"

BGS_SV3_CLUSTERING_N_BRIGHT_FILE = BGS_FUJI_FOLDER + "BGS_BRIGHT_N_clustering.dat.fits"
BGS_SV3_CLUSTERING_S_BRIGHT_FILE = BGS_FUJI_FOLDER + "BGS_BRIGHT_S_clustering.dat.fits"
BGS_SV3_CLUSTERING_RAND_FILE = BGS_FUJI_FOLDER + "BGS_BRIGHT_N_X_clustering.ran.fits"


# BGS Y1 DATA FILES
###################
# File was acquired from /global/cfs/cdirs/desi/public/dr1/vac/dr1/lsdr9-photometry/iron/v1.1/observed-targets/targetphot-iron.fits
# Files were acquired from /global/cfs/cdirs/desi/public/dr1/vac/dr1/lsdr9-photometry/iron/v1.1/potential-targets/...
# Combination and reduce version of the previous files. See fetch_build_photometric_catalog.py
BGS_Y1_COMBINED_PHOTOMETRIC_CATALOG = BGS_Y1_FOLDER + "targetphot-iron-combined.fits"
# File was acquired from https://data.desi.lbl.gov/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/BGS_ANY_full.dat.fits
BGS_ANY_FULL_FILE = BGS_Y1_FOLDER + "BGS_ANY_full.dat.fits"
# File was acquired from https://data.desi.lbl.gov/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.2/BGS_ANY_full.dat.fits
BGS_ANY_FULL_FILE_OLD = BGS_Y1_FOLDER + "BGS_ANY_full.dat.fits~"
# File was acquired from https://data.desi.lbl.gov/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/BGS_ANY_0_full.ran.fits
BGS_Y1_RAND_FILE = BGS_Y1_FOLDER + "BGS_ANY_X_full.ran.fits"
# File was acquired from https://data.desi.lbl.gov/desi/survey/catalogs/Y1/LSS/iron/LSScats/mainbw-bright-allTiles_v1.fits
BGS_PROB_OBS_FILE = BGS_Y1_FOLDER + "mainbw-bright-allTiles_v1.fits"
# File was acquired from Index of /global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/v2.1/catalogs/fastspec-iron-main-bright.fits
BGS_FASTSPEC_FILE = BGS_Y1_FOLDER + "fastspec-iron-main-bright.fits"
# File was acquired from https://data.desi.lbl.gov/desi/spectro/redux/iron/tiles-iron.csv
BGS_TILES_FILE = BGS_Y1_FOLDER + "tiles-iron.csv"
# This file is BUILT by running code in the BGS_study.ipynb notebook
# It is a joined BGS file, with a filtered down set of rows and columns
IAN_BGS_Y1_MERGED_FILE = BGS_Y1_FOLDER + "ian_BGS_merged.fits"
IAN_BGS_Y1_MERGED_FILE_OLD = BGS_Y1_FOLDER + "ian_BGS_merged.fits~"


# BGS Y3 DATA FILES
###################
# TODO get fastspecfit when available
BGS_Y3_COMBINED_PHOTOMETRIC_CATALOG = BGS_Y3_FOLDER_LOA + "targetphot-loa-combined.fits"
# File was acquired from https://data.desi.lbl.gov/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/BGS_ANY_full.dat.fits
BGS_Y3_ANY_FULL_FILE = BGS_Y3_FOLDER_LOA + "BGS_ANY_full.dat.fits"
# File was acquried from https://data.desi.lbl.gov/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/nonkp/BGS_BRIGHT_0_full.ran.fits
BGS_Y3_RAND_FILE = BGS_Y3_FOLDER_LOA + "BGS_BRIGHT_X_full.ran.fits"
BGS_Y3_CLUSTERING_RAND_FILE = BGS_Y3_FOLDER_LOA + "BGS_BRIGHT_X_clustering.ran.fits"
# File was built in NERSC from /pscratch/sd/i/ioannis/fastspecfit/data/loa/catalogs/fastspec-loa-main-bright-nside1-hpXX.fits
BGS_Y3_FASTSPEC_FILE = BGS_Y3_FOLDER_LOA + "fastspec-loa-main-bright-ian.fits"

# File was acquired from https://data.desi.lbl.gov/desi/spectro/redux/loa/tiles-kibo.csv
BGS_Y3_TILES_FILE = BGS_Y3_FOLDER_LOA + "tiles-kibo.csv"
#
BGS_Y3_CLUSTERING_FILE = BGS_Y3_FOLDER_LOA + "BGS_BRIGHT_clustering.dat.fits"
# This file is BUILT by running code in the BGS_study.ipynb notebook
# It is a joined BGS file, with a filtered down set of rows and columns
IAN_BGS_Y3_MERGED_FILE_KIBO = BGS_Y3_FOLDER_KIBO + "ian_BGS_Y3_merged.fits"
IAN_BGS_Y3_MERGED_FILE_LOA = BGS_Y3_FOLDER_LOA + "ian_BGS_Y3_merged.fits"


# DESI LEGACY IMAGING SURVEY FILES
##################################
# These are constructed from the sweeps because the sweeps are big. Use photoz.py to build them.
IAN_PHOT_Z_FILE = BGS_IMAGES_FOLDER + "IAN_PHOTZ_MATCHES.pkl"
IAN_PHOT_Z_FILE_NOSPEC = BGS_IMAGES_FOLDER + "IAN_PHOTZ_MATCHES_NS_NOSPEC.pkl"
IAN_PHOT_Z_FILE_WSPEC = BGS_IMAGES_FOLDER + "IAN_PHOTZ_MATCHES_NS_WSPEC.pkl"
BRICKS_TO_SKIP_S_FILE = BGS_IMAGES_FOLDER + "BRICKS_TO_SKIP.pkl"
BRICKS_TO_SKIP_N_FILE = BGS_IMAGES_FOLDER + "BRICKS_TO_SKIP_N.pkl"


# SV3 DERIVED AUXILERY FILES
############################
NEIGHBOR_ANALYSIS_SV3_BINS_FILE = OUTPUT_FOLDER + 'BGS_cic_binned_data.pkl'
NEIGHBOR_ANALYSIS_SV3_BINS_SMOOTHED_FILE = OUTPUT_FOLDER + 'BGS_cic_smoothed_binned_data.pkl'
NEIGHBOR_ANALYSIS_MXXL_BINS_FILE = OUTPUT_FOLDER + 'MXXL_cic_binned_data.pkl'

# MXXL DERIVED AUXILERY FILES
#############################

# This file is BUILT by running code in the MXXL_study.ipynb notebook
IAN_MXXL_LOST_APP_TO_Z_FILE = BGS_Y1_FOLDER + "mxxl_lost_appmag_to_z_map.dat"
# This file is BUILT by running code in the MXXL_study.ipynb notebook
MXXL_PROB_OBS_FILE = OUTPUT_FOLDER + "prob_obs.npy"
MXXL_ABS_MAG_R_FILE = OUTPUT_FOLDER + "mxxl_abs_mag_r_mine.npy"

# BGS DERIVED AUXILERY FILES
###############################
BGS_Y3_LOST_APP_TO_Z_FILE = BGS_Y1_FOLDER + "bgsy3_lost_appmag_to_z_map.dat"
BGS_Y3_LOST_APP_AND_ZPHOT_TO_Z_FILE = BGS_Y1_FOLDER + "bgsy3_lost_appmag_zphot_to_z_map.dat"


# RANDOMS FILES MATCHING OUR FOOTPRINTS
#######################################

# ##### SV3 (FUJI) Randoms #####
# These versions of randoms are for use with unweighted clustering where the untargeted galaxies have been assigned redshifts somehow
MY_RANDOMS_SV3 = OUTPUT_FOLDER + "randoms_df_sv3.pkl"
MY_RANDOMS_SV3_20 = OUTPUT_FOLDER + "randoms_df_sv3_20.pkl"
MY_RANDOMS_SV3_MINI = OUTPUT_FOLDER + "randoms_df_sv3_mini.pkl"
MY_RANDOMS_SV3_MINI_20 = OUTPUT_FOLDER + "randoms_df_sv3_mini_20.pkl"

# Clustering versions of randoms have weights from the LSS team. Meant for use with PIP weighted clustering
MY_RANDOMS_SV3_CLUSTERING = OUTPUT_FOLDER + "randoms_df_sv3_clustering.pkl"
MY_RANDOMS_SV3_CLUSTERING_20 = OUTPUT_FOLDER + "randoms_df_sv3_clustering_20.pkl"
MY_RANDOMS_SV3_CLUSTERING_MINI = OUTPUT_FOLDER + "randoms_df_sv3_clustering_mini.pkl"
MY_RANDOMS_SV3_CLUSTERING_MINI_20 = OUTPUT_FOLDER + "randoms_df_sv3_clustering_mini_20.pkl"

# ##### Y3 cut to SV3 10p footprint ######
# Clustering versions of randoms have weights from the LSS team. Meant for use with PIP weighted clustering
MY_RANDOMS_Y3_LIKESV3_CLUSTERING = OUTPUT_FOLDER + "randoms_df_y3likesv3_clustering.pkl"
MY_RANDOMS_Y3_LIKESV3_CLUSTERING_20 = OUTPUT_FOLDER + "randoms_df_y3likesv3_clustering_20.pkl"
MY_RANDOMS_Y3_LIKESV3_CLUSTERING_MINI = OUTPUT_FOLDER + "randoms_df_y3likesv3_clustering_mini.pkl"
MY_RANDOMS_Y3_LIKESV3_CLUSTERING_MINI_20 = OUTPUT_FOLDER + "randoms_df_y3likesv3_clustering_mini_20.pkl"



MY_RANDOMS_Y1_MINI = OUTPUT_FOLDER + "randoms_df_y1iron_mini.pkl"

MY_RANDOMS_Y3_MINI = OUTPUT_FOLDER + "randoms_df_y3loa_mini.pkl"



# CUSTOM CLUSTERING RESULTS DIRECTORY
#######################################################

CUSTOM_CLUSTERING_RESULTS_FOLDER = "/global/cfs/cdirs/desi/users/ianw89/clustering/"
