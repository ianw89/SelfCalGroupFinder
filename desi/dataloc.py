###################################################
# Information on where files are located
###################################################

# This is the root folder of this repository; update it to your own path
BASE_FOLDER = '/home/users/imw2293/SelfCalGroupFinder/'
BASE2_FOLDER = '/mount/sirocco1/imw2293/GROUP_CAT/'

# Subfolders
BIN_FOLDER = BASE_FOLDER + 'bin/'
OUTPUT_FOLDER = BASE2_FOLDER + 'OUTPUT/'
DATA_FOLDER = BASE2_FOLDER + 'DATA/'
SDSS_FOLDER = BASE_FOLDER + 'data/' + 'SDSS/'
BGS_FOLDER = DATA_FOLDER + 'BGS_IRON/'
BGS_Y3_FOLDER = DATA_FOLDER + 'BGS_JURA/'
MXXL_DATA_DIR="/export/sirocco2/tinker/DESI/MXXL_MOCKS/"
UCHUU_FILES_FOLDER="/export/sirocco2/tinker/DESI/UCHUU_MOCKS/"


# SDSS Data Files
#################
# This file was acquried from https://cosmo.nyu.edu/~tinker/GROUP_FINDER/SELFCAL_GROUPS/sdss_fluxlim_v1.0.dat
SDSS_v1_DAT_FILE = SDSS_FOLDER + 'sdss_fluxlim_v1.0.dat'
SDSS_v1_GALPROPS_FILE = SDSS_FOLDER + "sdss_galprops_v1.0.dat"

# Simulation Data Files
#######################
MXXL_FILE = MXXL_DATA_DIR + "weights_3pass.hdf5"
UCHUU_FILE = UCHUU_FILES_FOLDER + "BGS_LC_Uchuu.fits"


# BGS Y1 DATA FILES
###################
# File was acquired from https://data.desi.lbl.gov/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.2/BGS_ANY_full.dat.fits
BGS_ANY_FULL_FILE = BGS_FOLDER + "BGS_ANY_full.dat.fits"
# File was acquired from https://data.desi.lbl.gov/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.2/BGS_BRIGHT_0_full.ran.fits
BGS_RAND_FILE = BGS_FOLDER + "BGS_BRIGHT_0_full.ran.fits"
# File was acquired from https://data.desi.lbl.gov/desi/survey/catalogs/Y1/LSS/iron/LSScats/mainbw-bright-allTiles_v1.fits
BGS_PROB_OBS_FILE = BGS_FOLDER + "mainbw-bright-allTiles_v1.fits"
# File was acquired from Index of /public/dr1/vac/dr1/fastspecfit/iron/v2.1/catalogs/fastspec-iron-main-bright.fits
BGS_FASTSPEC_FILE = BGS_FOLDER + "fastspec-iron-main-bright.fits"
# File was acquired from https://data.desi.lbl.gov/desi/spectro/redux/iron/tiles-iron.csv
BGS_TILES_FILE = BGS_FOLDER + "tiles-iron.csv"
# This file is BUILT by running code in the BGS_study.ipynb notebook
# It is a joined BGS file, with a filtered down set of rows and columns
IAN_BGS_MERGED_FILE = BGS_FOLDER + "ian_BGS_merged.fits"


# BGS Y3 DATA FILES
###################
# TODO Jura is unofficial Y3, replace later
# TODO need mainbw and fastspec files but they don't exist for Jura yet
# File was acquired from https://data.desi.lbl.gov/desi/survey/catalogs/DA2/LSS/jura-v1/LSScats/test/BGS_ANY_full_noveto.dat.fits    
BGS_Y3_ANY_FULL_FILE = BGS_Y3_FOLDER + "BGS_ANY_full_noveto.dat.fits"
# File was acquired from https://data.desi.lbl.gov/desi/survey/catalogs/DA2/LSS/jura-v1/LSScats/test/BGS_BRIGHT_0_full.ran.fits
BGS_Y3_RAND_FILE = BGS_Y3_FOLDER + "BGS_BRIGHT_0_full.ran.fits"
# File was acquired from https://data.desi.lbl.gov/desi/spectro/redux/jura/tiles-jura.csv
BGS_Y3_TILES_FILE = BGS_Y3_FOLDER + "tiles-jura.csv"
# This file is BUILT by running code in the BGS_study.ipynb notebook
# It is a joined BGS file, with a filtered down set of rows and columns
IAN_BGS_Y3_MERGED_FILE = BGS_Y3_FOLDER + "ian_BGS_Y3_merged.fits"


# MXXL DERIVED AUXILERY FILES
#############################

# This file is BUILT by running code in the MXXL_study.ipynb notebook
IAN_MXXL_LOST_APP_TO_Z_FILE = BGS_FOLDER + "mxxl_lost_appmag_to_z_map.dat"
# This file is BUILT by running code in the MXXL_study.ipynb notebook
MXXL_PROB_OBS_FILE = OUTPUT_FOLDER + "prob_obs.npy"