###################################################
# Information on where files are located
###################################################

# This is the root folder of this repository; update it to your own path
#BASE_FOLDER = '/home/users/imw2293/'
BASE_FOLDER = '/mount/sirocco1/imw2293/GROUP_CAT/'
REPO_FOLDER = BASE_FOLDER + 'SelfCalGroupFinder/'

# Subfolders
BIN_FOLDER = REPO_FOLDER + 'bin/'
PY_SRC_FOLDER = REPO_FOLDER + 'py/'
PARAMS_FOLDER = PY_SRC_FOLDER + 'parameters/'
OUTPUT_FOLDER = BASE_FOLDER + 'OUTPUT/'
DATA_FOLDER = BASE_FOLDER + 'DATA/'
SDSS_FOLDER = DATA_FOLDER + 'SDSS/'
BGS_IMAGES_FOLDER = DATA_FOLDER + 'BGS_IMAGES/'
BGS_FOLDER = DATA_FOLDER + 'BGS_IRON/'
BGS_FUJI_FOLDER = DATA_FOLDER + 'BGS_FUJI/'
BGS_Y3_FOLDER = DATA_FOLDER + 'BGS_JURA/'
MXXL_DATA_DIR="/export/sirocco2/tinker/DESI/MXXL_MOCKS/"
UCHUU_FILES_FOLDER="/export/sirocco2/tinker/DESI/UCHUU_MOCKS/"


# Parameter Files
#################
WP_RADIAL_BINS_FILE = PARAMS_FOLDER + 'wp_rbins.dat'
HALO_MASS_FUNC_FILE = REPO_FOLDER + 'halo_mass_function.dat'
LSAT_LOOKUP_FILE = REPO_FOLDER + 'lsat_lookup.dat'
LSAT_OBSERVATIONS_FILE = PARAMS_FOLDER + 'Lsat_SDSS_DnGMM.dat'

# SDSS Data Files
#################
# These files were acquried from https://cosmo.nyu.edu/~tinker/GROUP_FINDER/SELFCAL_GROUPS/sdss_fluxlim_v1.0.dat
# They should have been built by Jeremy from the below DR7 NYU VAGC files
SDSS_v1_DAT_FILE = SDSS_FOLDER + 'sdss_fluxlim_v1.0.dat'
SDSS_v2_DAT_FILE = SDSS_FOLDER + 'sdss_fluxlim_v2.0.dat' # Built by me where DESI Y1 redshifts are added to SDSS missing ones
SDSS_v1_GALPROPS_FILE = SDSS_FOLDER + "sdss_galprops_v1.0.dat"
SDSS_v2_GALPROPS_FILE = SDSS_FOLDER + "sdss_galprops_v2.0.dat"
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
# File was acquired from https://data.desi.lbl.gov/desi/survey/catalogs/SV3/LSS/fuji/LSScats/3.1/BGS_ANY_full.dat.fits
BGS_SV3_ANY_FULL_FILE = BGS_FUJI_FOLDER + "BGS_ANY_full.dat.fits"
# File was acquired from https://data.desi.lbl.gov/desi/survey/catalogs/SV3/LSS/fuji/LSScats/3.1/BGS_ANY_0_full.ran.fits
BGS_SV3_RAND_FILE = BGS_FUJI_FOLDER + "BGS_ANY_0_full.ran.fits"
# File was acquired from https://data.desi.lbl.gov/desi/survey/catalogs/SV3/LSS/fuji/Alltiles_bright_tilelocs.dat.fits   
BGS_SV3_PROB_OBS_FILE = BGS_FUJI_FOLDER + "Alltiles_bright_tilelocs.dat.fits"
# This file is BUILT by running code in the BGS_study.ipynb notebook
# It is a joined BGS file, with a filtered down set of rows and columns
IAN_BGS_SV3_MERGED_FILE = BGS_FUJI_FOLDER + "ian_BGS_SV3_merged.fits"


# BGS Y1 DATA FILES
###################
# File was acquired from https://data.desi.lbl.gov/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/BGS_ANY_full.dat.fits
BGS_ANY_FULL_FILE = BGS_FOLDER + "BGS_ANY_full.dat.fits"
# File was acquired from https://data.desi.lbl.gov/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.2/BGS_ANY_full.dat.fits
BGS_ANY_FULL_FILE_OLD = BGS_FOLDER + "BGS_ANY_full.dat.fits~"
# File was acquired from https://data.desi.lbl.gov/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/BGS_ANY_0_full.ran.fits
BGS_RAND_FILE = BGS_FOLDER + "BGS_ANY_0_full.ran.fits"
# File was acquired from https://data.desi.lbl.gov/desi/survey/catalogs/Y1/LSS/iron/LSScats/mainbw-bright-allTiles_v1.fits
BGS_PROB_OBS_FILE = BGS_FOLDER + "mainbw-bright-allTiles_v1.fits"
# File was acquired from Index of /public/dr1/vac/dr1/fastspecfit/iron/v2.1/catalogs/fastspec-iron-main-bright.fits
BGS_FASTSPEC_FILE = BGS_FOLDER + "fastspec-iron-main-bright.fits"
# File was acquired from https://data.desi.lbl.gov/desi/spectro/redux/iron/tiles-iron.csv
BGS_TILES_FILE = BGS_FOLDER + "tiles-iron.csv"
# This file is BUILT by running code in the BGS_study.ipynb notebook
# It is a joined BGS file, with a filtered down set of rows and columns
IAN_BGS_MERGED_FILE = BGS_FOLDER + "ian_BGS_merged.fits"
IAN_BGS_MERGED_FILE_OLD = BGS_FOLDER + "ian_BGS_merged.fits~"


# BGS Y3 DATA FILES
###################
# TODO Jura is unofficial Y3, replace later
# TODO this is noveto version, want BGS_ANY_full.dat.fits instead
# TODO need PROB_OBS for Y3...
# File was acquired from https://data.desi.lbl.gov/desi/survey/catalogs/DA2/LSS/jura-v1/LSScats/test/BGS_ANY_full_noveto.dat.fits    
BGS_Y3_ANY_FULL_FILE = BGS_Y3_FOLDER + "BGS_ANY_full_noveto.dat.fits"
# File was acquired from https://data.desi.lbl.gov/desi/survey/catalogs/DA2/LSS/jura-v1/LSScats/test/BGS_BRIGHT_0_full.ran.fits
BGS_Y3_RAND_FILE = BGS_Y3_FOLDER + "BGS_BRIGHT_0_full.ran.fits"
# File was acquired from https://data.desi.lbl.gov/desi/spectro/redux/jura/tiles-jura.csv
BGS_Y3_TILES_FILE = BGS_Y3_FOLDER + "tiles-jura.csv"
# This file is BUILT by running code in the BGS_study.ipynb notebook
# It is a joined BGS file, with a filtered down set of rows and columns
IAN_BGS_Y3_MERGED_FILE = BGS_Y3_FOLDER + "ian_BGS_Y3_merged.fits"

IAN_PHOT_Z_FILE = BGS_IMAGES_FOLDER + "IAN_PHOTZ_MATCHES.fits"


# MXXL DERIVED AUXILERY FILES
#############################

# This file is BUILT by running code in the MXXL_study.ipynb notebook
IAN_MXXL_LOST_APP_TO_Z_FILE = BGS_FOLDER + "mxxl_lost_appmag_to_z_map.dat"
# This file is BUILT by running code in the MXXL_study.ipynb notebook
MXXL_PROB_OBS_FILE = OUTPUT_FOLDER + "prob_obs.npy"