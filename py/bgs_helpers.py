import numpy as np
import pandas as pd
import astropy.coordinates as coord
import astropy.units as u
import astropy.io.fits as fits
from astropy.table import Table,join,vstack,unique,QTable
import sys
import pickle
import subprocess

if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from pyutils import *
from dataloc import *

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
sv3_poor_y3overlap = [0,1] # the first two regions from above have poor overlap wth Y3 footprint

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

# This corresponds to 8.33 square degrees and empircally makes sense by looking at the randoms
TILE_RADIUS = 5862.0 * u.arcsec # arcsec

# Sentinal value from legacy survey for no photo-z
NO_PHOTO_Z = -99.0

NTILE_MIN_TO_FIND = 10


def find_tiles_for_galaxies(tiles_df, gals_df, num_tiles_to_find):
    num_galaxies = len(gals_df.RA)
    num_tiles = len(tiles_df.RA)

    tiles_coord = coord.SkyCoord(ra=tiles_df.RA.to_numpy()*u.degree, dec=tiles_df['DEC'].to_numpy()*u.degree, frame='icrs')
    gals_coord = coord.SkyCoord(ra=gals_df.RA.to_numpy()*u.degree, dec=gals_df['DEC'].to_numpy()*u.degree, frame='icrs')

    # Structure for resultant data
    nearest_tile_ids = np.zeros((num_galaxies, num_tiles_to_find), dtype=int)
    ntiles_inside = np.zeros((num_galaxies), dtype=int)

    for n in range(num_tiles_to_find):
        idx, d2d, d3d = coord.match_coordinates_sky(gals_coord, tiles_coord, nthneighbor=n+1, storekdtree=None)
        nearest_tile_ids[:,n] = tiles_df.iloc[idx].TILEID
        ntiles_inside += (d2d < TILE_RADIUS).astype(int)

    
    return ntiles_inside, nearest_tile_ids

def add_mag_columns(table):
    print("Adding magnitude columns to table.")
    
    # if the table doesn't have DN4000_MODEL, print a warning
    if 'DN4000_MODEL' not in table.columns:
        print("Warning: DN4000_MODEL not in table. Color cut will only be based on g-r.")
        dn4000 = None
    else:
        dn4000 = table['DN4000_MODEL']

    app_mag_r = get_app_mag(table['FLUX_R'])
    app_mag_g = get_app_mag(table['FLUX_G'])
    g_r = app_mag_g - app_mag_r

    if np.ma.is_masked(table['Z']):
        z_obs = table['Z'].data.data
    else:
        z_obs = table['Z']

    # nans for lost galaxies will propagate through the calculations as desired
    abs_mag_R = app_mag_to_abs_mag(app_mag_r, z_obs)
    abs_mag_R_k = k_correct(abs_mag_R, z_obs, g_r, band='r')
    abs_mag_G = app_mag_to_abs_mag(app_mag_g, z_obs)
    abs_mag_G_k = k_correct(abs_mag_G, z_obs, g_r, band='g')
    log_L_gal = abs_mag_r_to_log_solar_L(abs_mag_R_k) 
    G_R_k = abs_mag_G_k - abs_mag_R_k # based on the polynomial k-corr
    G_R_k_fastspecfit = table['ABSMAG01_SDSS_G'] - table['ABSMAG01_SDSS_R'] # based on fastspecfit k-corr
    G_R_BEST = np.where(np.isnan(G_R_k_fastspecfit), G_R_k, G_R_k_fastspecfit)
    quiescent = is_quiescent_BGS_smart(log_L_gal, dn4000, G_R_BEST)

    table.add_column(app_mag_r, name='APP_MAG_R')
    table.add_column(app_mag_g, name='APP_MAG_G')
    table.add_column(abs_mag_R, name='ABS_MAG_R')
    table.add_column(abs_mag_R_k, name='ABS_MAG_R_K')
    table.add_column(abs_mag_G, name='ABS_MAG_G')
    table.add_column(abs_mag_G_k, name='ABS_MAG_G_K')
    table.add_column(log_L_gal, name='LOG_L_GAL')
    table.add_column(G_R_BEST, name='G_R_BEST')
    table.add_column(quiescent, name='QUIESCENT')


def add_photz_columns(table_file :str, phot_z_file):
    """
    Reads an astropy table and adds columns from the legacy survey file we built (photo-z, etc.).
    """
    print("Adding photo-z columns to table.")
    table = Table.read(table_file, format='fits')

    if 'Z_PHOT' in table.columns:
        print("Z_PHOT already in table, replacing it.")
        table.remove_columns(['Z_PHOT', 'RELEASE', 'BRICKID', 'OBJID', 'REF_CAT'])

    phot_z_table = pickle.load(open(phot_z_file, 'rb'))
    phot_z_table['TARGETID'] = phot_z_table.index # in the DataFrame TARGETID is the index, not a column, so copy it over so the conversion keeps it
    # Merge in the photo-z and whatever else info we took from Legacy Surveys sweeps

    percent_complete = (phot_z_table['Z_LEGACY_BEST'] != NO_PHOTO_Z).sum() / len(phot_z_table)
    print(f"Phot-z file has phot-z for {percent_complete:.2%} of targets.")

    final_table = join(table, QTable.from_pandas(phot_z_table), join_type='left', keys="TARGETID")

    print(len(table))
    print(len(phot_z_table))
    print(len(final_table))

    final_table.rename_column('Z_LEGACY_BEST', 'Z_PHOT')
    final_table.rename_column('RA_1', 'RA')
    final_table.rename_column('DEC_1', 'DEC')
    final_table.remove_columns(['RA_2', 'DEC_2'])
    print(final_table.columns)

    # TODO I should switch to having the merged file be pickle.dump of a DataFrame. 
    # Only thing is NEAREST_TILEIDS cannot be a lit
    final_table.write(table_file, format='fits', overwrite=True)



def table_to_df(table: Table):
    """
    This does not work for all purposes yet.
    """
    # TODO why not use to_pandas()?
    #df = table.to_pandas()
    
    dec = table['DEC'].astype("<f8") # Big endian vs little endian regression in pandas. Convert more of these fields like this
    ra = table['RA'].astype("<f8") # as needed if using pandas with this data
    df = pd.DataFrame({
        'DEC': dec,
        'RA': ra,
        })

    return df

def read_fastspecfit_sv3():
    hdul = fits.open(BGS_SV3_FASTSPEC_FILE, memmap=True)
    data = hdul[1].data
    fastspecfit_table = Table([
        data['TARGETID'], 
        data['DN4000'], 
        data['DN4000_MODEL'], 
        data['ABSMAG01_SDSS_G'], 
        data['ABSMAG01_SDSS_R'], 
        data['SFR'], 
        data['LOGMSTAR']
        ], 
        names=('TARGETID', 'DN4000', 'DN4000_MODEL', 'ABSMAG01_SDSS_G', 'ABSMAG01_SDSS_R', 'SFR', 'LOGMSTAR'))
    hdul.close()
    return fastspecfit_table

def read_fastspecfit_y1_reduced():
    hdul = fits.open(BGS_FASTSPEC_FILE, memmap=True)
    data = hdul[1].data
    fastspecfit_table = Table([
        data['TARGETID'], 
        data['DN4000'], 
        data['DN4000_MODEL'], 
        data['ABSMAG01_SDSS_G'], 
        data['ABSMAG01_SDSS_R'], 
        data['SFR'], 
        data['LOGMSTAR']
        ], 
        names=('TARGETID', 'DN4000', 'DN4000_MODEL', 'ABSMAG01_SDSS_G', 'ABSMAG01_SDSS_R', 'SFR', 'LOGMSTAR'))
    hdul.close()
    return fastspecfit_table

def add_photometric_columns(existing_table, version: str):
    print(f"Adding extra photometric columns.")
    if version == 'sv3':
        photo_table = Table.read(BGS_SV3_COMBINED_PHOTOMETRIC_CATALOG, format='fits') # Contains SV3
        photo_table = photo_table[photo_table['SURVEY'] == 'sv3']
        jointype = 'left' # Just for SV3, because we have supplementary Y3 galaxies without rows in this...
    elif version == '1':  
        photo_table = Table.read(BGS_Y1_COMBINED_PHOTOMETRIC_CATALOG, format='fits') 
        jointype = 'inner'
    elif version == '3':
        photo_table = Table.read(BGS_Y3_COMBINED_PHOTOMETRIC_CATALOG, format='fits')
        jointype = 'inner'
    else:
        print("No other photometric tables available yet")
    
    final_table = join(existing_table, photo_table, join_type=jointype, keys="TARGETID")

    # Check if each rows that have some TARGETID have the same values in the columns 
    #cols_to_check = ['TARGETID', 'FRACFLUX_G', 'FRACFLUX_R', 'FRACFLUX_Z', 'SHAPE_E1', 'SHAPE_E2', 'SHAPE_R_IVAR', 'SHAPE_E1_IVAR', 'SHAPE_E2_IVAR', 'SERSIC', 'SERSIC_IVAR']
    #test_set = final_table[0:100000]
    #test_set = test_set[test_set['PROGRAM'] == 'bright']
    #test_set.keep_columns(cols_to_check)
    #results = test_set.group_by('TARGETID').groups.aggregate(lambda x: np.all(np.isclose(x, x[0])))
    #for c in cols_to_check[1:]:
    #    print(f"Checking column {c}")
    #    assert np.all(results[c])

    # There are duplicates in photometric table as it combined all surveys and potential / observed tables. 
    final_table = unique(final_table, 'TARGETID')
    
    print(f"  Original len={len(existing_table):,}, photometric len={len(photo_table):,}, final len={len(final_table):,}.")

    return final_table

def add_fastspecfit_columns(main_table, version:str):
    if version == 'sv3':
        fastspecfit_table = read_fastspecfit_sv3()
    else:
        fastspecfit_table = read_fastspecfit_y1_reduced()
    final_table = join(main_table, fastspecfit_table, join_type='left', keys="TARGETID")
    return final_table

def read_tiles_Y1_main():
    tiles_table = Table.read(BGS_TILES_FILE, format='csv')
    tiles_table.keep_columns(['TILEID', 'FAFLAVOR', 'TILERA', 'TILEDEC'])
    tiles_df = pd.DataFrame({'RA': tiles_table['TILERA'].astype("<f8"), 'DEC': tiles_table['TILEDEC'].astype("<f8"), 'FAFLAVOR': tiles_table['FAFLAVOR'], 'TILEID': tiles_table['TILEID']})
    tiles_df = tiles_df[tiles_df.FAFLAVOR == 'mainbright']
    tiles_df.reset_index(drop=True, inplace=True)
    return tiles_df

def read_tiles_Y3_sv3():
    tiles_table = Table.read(BGS_Y3_TILES_FILE, format='csv')
    tiles_table.keep_columns(['TILEID', 'FAFLAVOR', 'TILERA', 'TILEDEC'])
    tiles_df = pd.DataFrame({'RA': tiles_table['TILERA'].astype("<f8"), 'DEC': tiles_table['TILEDEC'].astype("<f8"), 'FAFLAVOR': tiles_table['FAFLAVOR'], 'TILEID': tiles_table['TILEID']})
    tiles_df = tiles_df[tiles_df.FAFLAVOR == 'sv3bright']
    tiles_df.reset_index(drop=True, inplace=True)
    return tiles_df

def read_tiles_Y3_main():
    tiles_table = Table.read(BGS_Y3_TILES_FILE, format='csv')
    tiles_table.keep_columns(['TILEID', 'FAFLAVOR', 'TILERA', 'TILEDEC'])
    tiles_df = pd.DataFrame({'RA': tiles_table['TILERA'].astype("<f8"), 'DEC': tiles_table['TILEDEC'].astype("<f8"), 'FAFLAVOR': tiles_table['FAFLAVOR'], 'TILEID': tiles_table['TILEID']})
    tiles_df = tiles_df[tiles_df.FAFLAVOR == 'mainbright']
    tiles_df.reset_index(drop=True, inplace=True)
    return tiles_df

def add_NTILE_MINE_to_table(table_file :str|Table, year: str):
    print("Adding NTILE_MINE to table.")
    if year == "sv3":
        tiles_df = read_tiles_Y3_sv3()
    elif year == "1":
        tiles_df = read_tiles_Y1_main()
    elif year == "3":
        tiles_df = read_tiles_Y3_main()
    else:
        raise ValueError("Year must be 1 or 3")
    
    if table_file is str:
        table = Table.read(table_file, format='fits')
    elif isinstance(table_file, Table):
        table = table_file
    else:
        raise ValueError("table_file must be a string or astropy Table")
    
    galaxies_df = table_to_df(table)
    
    ntiles_inside, nearest_tile_ids = find_tiles_for_galaxies(tiles_df, galaxies_df, 15)
    if 'NTILE_MINE' in table.columns:
        table.remove_columns(['NTILE_MINE', 'NEAREST_TILEIDS'])
    table.add_column(ntiles_inside, name="NTILE_MINE")
    table.add_column(nearest_tile_ids, name="NEAREST_TILEIDS")

    return table


def create_merged_file(orig_table_file : str, merged_file : str, year : str):
    table = Table.read(orig_table_file, format='fits')
    print(f"Read {len(table)} galaxies from {orig_table_file}")

    # The lost galaxies will not have fastspecfit rows as they have no spectra
    table = add_fastspecfit_columns(table, year)

    # Filter to needed columns only and save
    table.keep_columns(['TARGETID', 'SPECTYPE', 'DEC', 'RA', 'Z_not4clus', 'FLUX_R', 'FLUX_G', 'PROB_OBS', 'ZWARN', 'DELTACHI2', 'NTILE', 'TILES', 'DN4000', 'DN4000_MODEL', 'ABSMAG01_SDSS_G', 'ABSMAG01_SDSS_R', 'MASKBITS'])
    table.rename_column('Z_not4clus', 'Z')
    table.write(merged_file, format='fits', overwrite='True')

    table = add_photometric_columns(table, year)
    table.write(merged_file, format='fits', overwrite='True')

    add_mag_columns(table)
    table.write(merged_file, format='fits', overwrite='True')

    add_NTILE_MINE_to_table(table, year)
    table.write(merged_file, format='fits', overwrite='True')

    add_photz_columns(merged_file, IAN_PHOT_Z_FILE_WSPEC)


def fix_columns_in_phot_z_file(f):
    phot_z_table = pickle.load(open(f, 'rb'))
    phot_z_table['REF_CAT_NEW'] = phot_z_table['REF_CAT'].astype('S2')
    phot_z_table.loc[(phot_z_table.REF_CAT_NEW == b'  '), 'REF_CAT_NEW'] = b''
    phot_z_table.loc[(phot_z_table.REF_CAT_NEW == b'na'), 'REF_CAT_NEW'] = b''
    phot_z_table.drop('REF_CAT', inplace=True, axis=1)
    phot_z_table.rename(columns={'REF_CAT_NEW':'REF_CAT'}, inplace=True)
    pickle.dump(phot_z_table, open(f, 'wb'))

# Run this after building photo-z file
#fix_columns_in_phot_z_file(IAN_PHOT_Z_FILE_NOSPEC)

def read_randoms(base, n):
    rtable = Table.read(base.replace("X", str(n)), format='fits')
    r_dec = rtable['DEC'].astype("<f8")
    r_ra = rtable['RA'].astype("<f8")
    r_ntiles = rtable['NTILE'].astype("<i8")

    randoms_df = pd.DataFrame({'RA': r_ra, 'DEC': r_dec, 'NTILE': r_ntiles})

    if 'WEIGHT' in rtable.columns:
        randoms_df['WEIGHT'] = rtable['WEIGHT'].astype("<f8")

    return randoms_df

def read_randoms_with_addons(base, n, tiles_df, apply_10p_cut):
    randoms_df = read_randoms(base, n)
    ntiles_inside, nearest_tile_ids = find_tiles_for_galaxies(tiles_df, randoms_df, NTILE_MIN_TO_FIND)
    randoms_df['NTILE_MINE'] = ntiles_inside
    randoms_df['REGION'] = tile_to_region(nearest_tile_ids[:, 0])
    if apply_10p_cut:
        randoms_df = randoms_df[randoms_df['NTILE_MINE'] >= 10]
    return randoms_df

def get_sv3_randoms_inner():
    r = pickle.load(open(MY_RANDOMS_SV3, "rb"))
    return r[r['NTILE_MINE'] >= 10]
        
def get_sv3_randoms_raw():
    return pickle.load(open(MY_RANDOMS_SV3, "rb"))

def get_sv3_randoms_inner_mini():
    r = pickle.load(open(MY_RANDOMS_SV3_MINI, "rb"))
    return r[r['NTILE_MINE'] >= 10]

def get_sv3_randoms_raw_mini():
    return pickle.load(open(MY_RANDOMS_SV3_MINI, "rb"))
    
def build_sv3footprint_randoms_files(randoms_file, mini_output_file, mini_output_file_filtered, big_output_file, big_output_file_filtered, has_NS_split=False, apply_10p_cut=False):
    tiles_sv3 = read_tiles_Y3_sv3() # Gets the SV3 bright tiles from the Y3 master tile list

    # Make a mini version from one randoms file
    randoms_df0 = read_randoms_with_addons(randoms_file, 0, tiles_sv3, apply_10p_cut)
    pickle.dump(randoms_df0, open(mini_output_file, "wb"))
    to_remove = np.isin(randoms_df0['REGION'], sv3_poor_y3overlap)
    randoms_df0 = randoms_df0.loc[~to_remove]
    pickle.dump(randoms_df0, open(mini_output_file_filtered, "wb"))

    # Make a big version combining all randoms files
    big_rand_df = read_randoms_with_addons(randoms_file, 0, tiles_sv3, apply_10p_cut)
    for i in range(1, 18):
        rand_df = read_randoms_with_addons(randoms_file, i, tiles_sv3, apply_10p_cut)
        big_rand_df = pd.concat([big_rand_df, rand_df])

    if has_NS_split:
        for i in range(0, 18):
            rand_df = read_randoms_with_addons(randoms_file.replace("_N_", "_S_"), i, tiles_sv3, apply_10p_cut)
            big_rand_df = pd.concat([big_rand_df, rand_df])

    pickle.dump(big_rand_df, open(big_output_file, "wb"))
    
    to_remove = np.isin(big_rand_df['REGION'], sv3_poor_y3overlap)
    big_rand_df = big_rand_df.loc[~to_remove]
    pickle.dump(big_rand_df, open(big_output_file_filtered, "wb"))

def build_sv3_full_randoms_files():
    build_sv3footprint_randoms_files(
        BGS_SV3_RAND_FILE,
        MY_RANDOMS_SV3_MINI_20,
        MY_RANDOMS_SV3_MINI,
        MY_RANDOMS_SV3_20,
        MY_RANDOMS_SV3
    )

def build_sv3_clustering_randoms_files():
    build_sv3footprint_randoms_files(
        BGS_SV3_CLUSTERING_RAND_FILE,
        MY_RANDOMS_SV3_CLUSTERING_MINI_20,
        MY_RANDOMS_SV3_CLUSTERING_MINI,
        MY_RANDOMS_SV3_CLUSTERING_20,
        MY_RANDOMS_SV3_CLUSTERING,
        has_NS_split=True
    )

def build_y3_likesv3_clustering_randoms_files():
    build_sv3footprint_randoms_files(
        BGS_Y3_CLUSTERING_RAND_FILE,
        MY_RANDOMS_Y3_LIKESV3_CLUSTERING_MINI_20,
        MY_RANDOMS_Y3_LIKESV3_CLUSTERING_MINI,
        MY_RANDOMS_Y3_LIKESV3_CLUSTERING_20,
        MY_RANDOMS_Y3_LIKESV3_CLUSTERING,
        apply_10p_cut=True # Do this here instead of later to reduce file
    )

def build_y1_randoms_files():
    tiles = read_tiles_Y1_main()

    # Make a mini version from one randoms file
    randoms_df0 = read_randoms(BGS_Y1_RAND_FILE, 0)
    ntiles_inside, nearest_tile_ids = find_tiles_for_galaxies(tiles, randoms_df0, NTILE_MIN_TO_FIND)
    randoms_df0['NTILE_MINE'] = ntiles_inside
    pickle.dump(randoms_df0, open(MY_RANDOMS_Y1_MINI, "wb"))

    # Full version may be needed later but not for now

def build_y3_randoms_files():
    tiles = read_tiles_Y3_main()

    # Make a mini version from one randoms file
    randoms_df0 = read_randoms(BGS_Y3_RAND_FILE, 0)
    ntiles_inside, nearest_tile_ids = find_tiles_for_galaxies(tiles, randoms_df0, NTILE_MIN_TO_FIND)
    randoms_df0['NTILE_MINE'] = ntiles_inside
    pickle.dump(randoms_df0, open(MY_RANDOMS_Y3_MINI, "wb"))

    # Full version may be needed later but not for now
