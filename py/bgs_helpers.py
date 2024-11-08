import numpy as np
import pandas as pd
import astropy.coordinates as coord
import astropy.units as u
import astropy.io.fits as fits
from astropy.table import Table,join,vstack,unique,QTable
import sys

if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from pyutils import *
from dataloc import *
from photoz import *

# This corresponds to 8.33 square degrees and empircally makes sense by looking at the randoms
TILE_RADIUS = 5862.0 * u.arcsec # arcsec

def find_tiles_for_galaxies(tiles_df, gals_df, num_tiles_to_find):
    num_galaxies = len(gals_df.RA)
    num_tiles = len(tiles_df.RA)

    tiles_coord = coord.SkyCoord(ra=tiles_df.RA.to_numpy()*u.degree, dec=tiles_df.Dec.to_numpy()*u.degree, frame='icrs')
    gals_coord = coord.SkyCoord(ra=gals_df.RA.to_numpy()*u.degree, dec=gals_df.Dec.to_numpy()*u.degree, frame='icrs')

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
    app_mag_r = get_app_mag(table['FLUX_R'])
    app_mag_g = get_app_mag(table['FLUX_G'])
    g_r = app_mag_g - app_mag_r

    if np.ma.is_masked(table['Z']):
        z_obs = table['Z'].data.data
    else:
        z_obs = table['Z']
    
    abs_mag_R = app_mag_to_abs_mag(app_mag_r, z_obs)
    abs_mag_R_k = k_correct(abs_mag_R, z_obs, g_r, band='r')
    abs_mag_G = app_mag_to_abs_mag(app_mag_g, z_obs)
    abs_mag_G_k = k_correct(abs_mag_G, z_obs, g_r, band='g')
    log_L_gal = abs_mag_r_to_log_solar_L(abs_mag_R_k) 
    G_R_k = abs_mag_G_k - abs_mag_R_k
    quiescent = is_quiescent_BGS_gmr(log_L_gal, G_R_k)

    table.add_column(app_mag_r, name='APP_MAG_R')
    table.add_column(app_mag_g, name='APP_MAG_G')
    table.add_column(abs_mag_R, name='ABS_MAG_R')
    table.add_column(abs_mag_R_k, name='ABS_MAG_R_K')
    table.add_column(abs_mag_G, name='ABS_MAG_G')
    table.add_column(abs_mag_G_k, name='ABS_MAG_G_K')
    table.add_column(log_L_gal, name='LOG_L_GAL')
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
        'Dec': dec,
        'RA': ra,
        })

    return df

def read_fastspecfit_y1_reduced():
    hdul = fits.open(BGS_FASTSPEC_FILE, memmap=True)
    #print(hdul[1].columns)
    data = hdul[1].data
    fastspecfit_id = data['TARGETID']
    DN4000 = data['DN4000'] # TODO there is also DN4000_OBS and DN4000_MODEL (and inverse variance)
    FSF_G = data['ABSMAG01_SDSS_G']
    FSF_R = data['ABSMAG01_SDSS_R']
    hdul.close()

    fastspecfit_table = Table([fastspecfit_id, DN4000, FSF_G, FSF_R], names=('TARGETID', 'DN4000', 'ABSMAG01_SDSS_G', 'ABSMAG01_SDSS_R'))
    return fastspecfit_table

def add_fastspecfit_columns(main_table):
    fastspecfit_table = read_fastspecfit_y1_reduced()
    final_table = join(main_table, fastspecfit_table, join_type='left', keys="TARGETID")
    return final_table

def read_tiles_Y1_main():
    tiles_table = Table.read(BGS_TILES_FILE, format='csv')
    tiles_table.keep_columns(['TILEID', 'FAFLAVOR', 'TILERA', 'TILEDEC'])
    tiles_df = pd.DataFrame({'RA': tiles_table['TILERA'].astype("<f8"), 'Dec': tiles_table['TILEDEC'].astype("<f8"), 'FAFLAVOR': tiles_table['FAFLAVOR'], 'TILEID': tiles_table['TILEID']})
    tiles_df = tiles_df[tiles_df.FAFLAVOR == 'mainbright']
    tiles_df.reset_index(drop=True, inplace=True)
    return tiles_df

def read_tiles_Y3_sv3():
    tiles_table = Table.read(BGS_Y3_TILES_FILE, format='csv')
    tiles_table.keep_columns(['TILEID', 'FAFLAVOR', 'TILERA', 'TILEDEC'])
    tiles_df = pd.DataFrame({'RA': tiles_table['TILERA'].astype("<f8"), 'Dec': tiles_table['TILEDEC'].astype("<f8"), 'FAFLAVOR': tiles_table['FAFLAVOR'], 'TILEID': tiles_table['TILEID']})
    tiles_df = tiles_df[tiles_df.FAFLAVOR == 'sv3bright']
    tiles_df.reset_index(drop=True, inplace=True)
    return tiles_df

def read_tiles_Y3_main():
    tiles_table = Table.read(BGS_Y3_TILES_FILE, format='csv')
    tiles_table.keep_columns(['TILEID', 'FAFLAVOR', 'TILERA', 'TILEDEC'])
    tiles_df = pd.DataFrame({'RA': tiles_table['TILERA'].astype("<f8"), 'Dec': tiles_table['TILEDEC'].astype("<f8"), 'FAFLAVOR': tiles_table['FAFLAVOR'], 'TILEID': tiles_table['TILEID']})
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
    orig_table = Table.read(orig_table_file, format='fits')
    print(f"Read {len(orig_table)} galaxies from {orig_table_file}")

    # The lost galaxies will not have fastspecfit rows I think
    table = add_fastspecfit_columns(orig_table)

    del(orig_table)

    # Filter to needed columns only and save
    table.keep_columns(['TARGETID', 'SPECTYPE', 'DEC', 'RA', 'Z_not4clus', 'FLUX_R', 'FLUX_G', 'PROB_OBS', 'ZWARN', 'DELTACHI2', 'NTILE', 'TILES', 'DN4000', 'ABSMAG01_SDSS_G', 'ABSMAG01_SDSS_R', 'MASKBITS'])
    table.rename_column('Z_not4clus', 'Z')
    table.write(merged_file, format='fits', overwrite='True')

    add_mag_columns(table)
    table.write(merged_file, format='fits', overwrite='True')

    add_NTILE_MINE_to_table(table, year)
    table.write(merged_file, format='fits', overwrite='True')

    add_photz_columns(merged_file, IAN_PHOT_Z_FILE_NOSPEC)


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