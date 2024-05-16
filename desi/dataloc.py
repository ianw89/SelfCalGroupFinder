###################################################
# Information on where files are located
###################################################

# This is the root folder of this repository; TODO make this automatic
#BASE_FOLDER = "/Volumes/Seagate Backup Plus Drive/galaxy-groups-data/"
#BASE_FOLDER = "/mnt/f/galaxy-groups-data/"
BASE_FOLDER = '/home/users/imw2293/SelfCalGroupFinder/'

# Subfolders
BIN_FOLDER = BASE_FOLDER + 'bin/'
ROOT_FOLDER = BIN_FOLDER #TODO cleanup: replace usages with BIN_FOLDER instead
DATA_FOLDER = BASE_FOLDER + 'data/'
MXXL_DATA_DIR="/export/sirocco2/tinker/DESI/MXXL_MOCKS/"
UCHUU_FILES_FOLDER="/export/sirocco2/tinker/DESI/UCHUU_MOCKS/"

# This file was acquried from https://cosmo.nyu.edu/~tinker/GROUP_FINDER/SELFCAL_GROUPS/sdss_fluxlim_v1.0.dat
SDSS_v1_DAT_FILE = DATA_FOLDER + 'sdss_fluxlim_v1.0.dat'
SDSS_v1_GALPROPS_FILE = DATA_FOLDER + "sdss_galprops_v1.0.dat"

BGS_PREVIOUS_FOLDER = DATA_FOLDER + "BGS v1.0/"

# Simulation Data Files
MXXL_FILE = MXXL_DATA_DIR + "weights_3pass.hdf5"
UCHUU_FILE = UCHUU_FILES_FOLDER + "BGS_LC_Uchuu.fits"

# File was acquired from https://data.desi.lbl.gov/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.2/BGS_ANY_full.dat.fits
BGS_ANY_FULL_FILE = DATA_FOLDER + "BGS_ANY_full.dat.fits"
# File was acquired from https://data.desi.lbl.gov/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.2/BGS_BRIGHT_0_full.ran.fits
BGS_RAND_FILE = DATA_FOLDER + "BGS_BRIGHT_0_full.ran.fits"
# File was acquired from https://data.desi.lbl.gov/desi/survey/catalogs/Y1/LSS/iron/LSScats/mainbw-bright-allTiles_v1.fits
BGS_PROB_OBS_FILE = DATA_FOLDER + "mainbw-bright-allTiles_v1.fits"
# File was acquired from Index of /public/dr1/vac/dr1/fastspecfit/iron/v2.1/catalogs/fastspec-iron-main-bright.fits
BGS_FASTSPEC_FILE = DATA_FOLDER + "fastspec-iron-main-bright.fits"
# File was acquired from https://data.desi.lbl.gov/desi/spectro/redux/iron/tiles-iron.csv
BGS_TILES_FILE = DATA_FOLDER + "tiles-iron.csv"

# This file is BUILT by running code in the BGS_study.ipynb notebook
# It is a joined BGS file, with a filtered down set of rows and columns
IAN_BGS_MERGED_FILE = DATA_FOLDER + "ian_BGS_merged.fits"

# This file is BUILT by running code in the MXXL_study.ipynb notebook
IAN_MXXL_LOST_APP_TO_Z_FILE = DATA_FOLDER + "mxxl_lost_appmag_to_z_map.dat"
# This file is BUILT by running code in the MXXL_study.ipynb notebook
MXXL_PROB_OBS_FILE = BIN_FOLDER + "prob_obs.npy"