import numpy as np
import pandas as pd
import astropy.coordinates as coord
import astropy.units as u
import astropy.io.fits as fits
from astropy.table import Table,join,vstack,unique,QTable
import sys
import pickle
import fitsio

if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from pyutils import *
from dataloc import *
from scripts.fetch_build_photometric_catalog import prepare_photo_vac

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

def determine_unobserved_from_z(column):
    """
    Determine if a galaxy is unobserved based on the Z column.
    """
    if np.ma.is_masked(column):
        unobserved = column.mask # the masked values are what is unobserved
    else:
        unobserved = np.zeros(len(column), dtype=bool)

    # Anything marked nan is unobserved
    unobserved |= np.isnan(column)

    # Some versions have sentinal values (99999)
    # Mark anythin obviously too high as unobserved
    unobserved |= column.astype("<f8") > 50

    return unobserved

def get_tbl_column(tbl, colname, required=False):
    if colname in tbl.columns:
        if np.ma.is_masked(tbl[colname]):
            return tbl[colname].data.data
        return tbl[colname]
    else:
        if required:
            raise ValueError(f"Required column {colname} not found in table.")
        return None

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
    print("Adding magnitude and quiescent classification columns to table.")
    # if the table doesn't have HALPHA or SFR, print a warning
    #if 'HALPHA_EW' not in table.columns or 'SFR' not in table.columns or 'LOGMSTAR' not in table.columns:
    #    print("WARNING: Missing HALPHA_EW, SFR, or LOGMSTAR columns in table.")
    #    halpha = None
    #    ssfr = None
    #else:
    #    halpha = table['HALPHA_EW']
    #    ssfr = table['SFR'] / np.power(10, table['LOGMSTAR'])

    # if the table doesn't have DN4000_MODEL, print a warning
    if 'DN4000_MODEL' not in table.columns:
        print("WARNING: Missing DN4000_MODEL column in table..")
        dn4000 = None
    else:
        dn4000 = table['DN4000_MODEL']

    # If photo-z are not present, print a warning
    if 'Z_PHOT' not in table.columns:
        print("ERROR: No photo-z present in table.")
        return

    app_mag_r = get_app_mag(table['FLUX_R'])
    app_mag_g = get_app_mag(table['FLUX_G'])
    g_r = app_mag_g - app_mag_r

    z_obs = get_tbl_column(table, 'Z', required=True).astype("<f8")
    no_spectra = determine_unobserved_from_z(z_obs)# | np.isnan(table['ABSMAG01_SDSS_R']) | np.isnan(table['ABSMAG01_SDSS_G'])

    # Where z_obs is nan, use the photo-z for absolute magnitude conversions
    speczcount = (~no_spectra).sum()
    z_obs[no_spectra] = table['Z_PHOT'][no_spectra].astype("<f8")
    stillmissing = np.isnan(z_obs).sum()
    photozcount = (~np.isnan(z_obs)).sum() - speczcount
    print(f"For absolute magnitude conversions, we have {speczcount:,} using spec-z, {photozcount:,} using photo-z, and {stillmissing:,} with neither.", flush=True)

    # nans for lost galaxies will propagate through the calculations as desired
    abs_mag_R = app_mag_to_abs_mag(app_mag_r, z_obs)
    abs_mag_G = app_mag_to_abs_mag(app_mag_g, z_obs)
    abs_mag_R_k, abs_mag_G_k = k_correct_fromlookup(abs_mag_R, abs_mag_G, z_obs)
    # Old polynomial way
    #abs_mag_R_k = k_correct_gama(abs_mag_R, z_obs, g_r, band='r')
    #abs_mag_G_k = k_correct_gama(abs_mag_G, z_obs, g_r, band='g')
    abs_mag_R_k_BEST = np.where(np.isnan(table['ABSMAG01_SDSS_R']), abs_mag_R_k, table['ABSMAG01_SDSS_R'])
    abs_mag_G_k_BEST = np.where(np.isnan(table['ABSMAG01_SDSS_G']), abs_mag_G_k, table['ABSMAG01_SDSS_G'])

    log_L_gal = abs_mag_r_to_log_solar_L(abs_mag_R_k_BEST) 
    G_R_BEST = abs_mag_G_k_BEST - abs_mag_R_k_BEST
    #x, y, z, zz, quiescent_kmeans, missing = is_quiescent_BGS_kmeans(log_L_gal, dn4000, halpha, ssfr, G_R_BEST, model=QUIESCENT_MODEL_V2)
    quiescent = is_quiescent_BGS_dn4000(log_L_gal, dn4000, G_R_BEST)
    table.add_column(app_mag_r, name='APP_MAG_R') 
    table.add_column(app_mag_g, name='APP_MAG_G') 
    table.add_column(abs_mag_R, name='ABS_MAG_R') 
    table.add_column(abs_mag_R_k, name='ABS_MAG_R_K')
    table.add_column(abs_mag_G, name='ABS_MAG_G')
    table.add_column(abs_mag_G_k, name='ABS_MAG_G_K')
    table.add_column(abs_mag_R_k_BEST, name='ABS_MAG_R_K_BEST') 
    table.add_column(abs_mag_G_k_BEST, name='ABS_MAG_G_K_BEST')
    table.add_column(log_L_gal, name='LOG_L_GAL')
    table.add_column(G_R_BEST, name='G_R_BEST')
    table.add_column(quiescent, name='QUIESCENT') 

    #table.add_column(quiescent_alt, name='QUIESCENT_KMEANS')
    return table
    


def add_photz_columns(table, phot_z_file):
    """
    Reads an astropy table and adds columns from the legacy survey file we built (photo-z, etc.).
    """
    print("Adding photo-z columns to table.")

    if 'Z_PHOT' in table.columns:
        print("Z_PHOT already in table, replacing it.")
        table.remove_columns(['Z_PHOT', 'RELEASE', 'BRICKID', 'OBJID', 'REF_CAT'])

    phot_z_table = pickle.load(open(phot_z_file, 'rb'))
    phot_z_table['TARGETID'] = phot_z_table.index # in the DataFrame TARGETID is the index, not a column, so copy it over so the conversion keeps it
    # Merge in the photo-z and whatever else info we took from Legacy Surveys sweeps

    percent_complete = (phot_z_table['Z_LEGACY_BEST'] != NO_PHOTO_Z).sum() / len(phot_z_table)
    print(f"Phot-z file has phot-z for {percent_complete:.2%} of targets.")

    final_table = join(table, QTable.from_pandas(phot_z_table), join_type='left', keys="TARGETID")

    print(f"  Original len={len(table):,}, photometric len={len(phot_z_table):,}, final len={len(final_table):,}.")

    final_table.rename_column('Z_LEGACY_BEST', 'Z_PHOT')
    final_table.rename_column('RA_1', 'RA')
    final_table.rename_column('DEC_1', 'DEC')
    final_table.remove_columns(['RA_2', 'DEC_2'])

    return final_table


def get_radec_df(table: Table):
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
    # Small data, no need to reduce it a priori
    hdul = fits.open(BGS_SV3_FASTSPEC_FILE, memmap=True)
    fastspecfit_table = Table([
        hdul[1].data['Z'],
        hdul[1].data['TARGETID'], 
        hdul[1].data['DN4000'], 
        hdul[1].data['DN4000_MODEL'], 
        hdul[1].data['ABSMAG01_SDSS_G'], 
        hdul[1].data['ABSMAG01_IVAR_SDSS_G'], 
        hdul[1].data['ABSMAG01_SDSS_R'], 
        hdul[1].data['ABSMAG01_IVAR_SDSS_R'], 
        hdul[1].data['SFR'], 
        hdul[1].data['LOGMSTAR'],
        hdul[1].data['HALPHA_EW'],
        hdul[1].data['HALPHA_EW_IVAR'],
        hdul[1].data['HBETA_EW'],
        hdul[1].data['HBETA_EW_IVAR'],
        ], 
        names=('Z_FSF', 'TARGETID', 'DN4000','DN4000_MODEL','ABSMAG01_SDSS_G', 'ABSMAG01_SDSS_G_IVAR', 'ABSMAG01_SDSS_R', 'ABSMAG01_SDSS_R_IVAR', 'SFR', 'LOGMSTAR', 'HALPHA_EW', 'HALPHA_EW_IVAR', 'HBETA_EW', 'HBETA_EW_IVAR'))
    hdul.close()
    return fastspecfit_table

def read_fastspecfit_y1():
    if not os.path.exists(BGS_Y1_FASTSPEC_FILE):
        print(f"BGS_Y1_FASTSPEC_FILE does not exist. Building from '{NERSC_BGS_IRON_FASTSPECFIT_DIR}'")
        hp = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
        filepattern = os.path.join(NERSC_BGS_IRON_FASTSPECFIT_DIR, "fastspec-iron-main-bright-nside1-hpXX.fits")
        for h in hp:
            filename = filepattern.replace("XX", h)
            if os.path.exists(filename):
                print(f"  Found {filename}")
                hdul = fits.open(filename, memmap=True)
                fastspecfit_table = Table([
                    hdul[1].data['Z'],
                    hdul[2].data['TARGETID'], 
                    hdul[2].data['DN4000'], 
                    hdul[2].data['DN4000_IVAR'], 
                    hdul[2].data['DN4000_MODEL'], 
                    hdul[2].data['DN4000_MODEL_IVAR'], 
                    hdul[2].data['ABSMAG01_SDSS_G'], 
                    hdul[2].data['ABSMAG01_IVAR_SDSS_G'], 
                    hdul[2].data['ABSMAG01_SDSS_R'], 
                    hdul[2].data['ABSMAG01_IVAR_SDSS_R'], 
                    hdul[2].data['SFR'], 
                    hdul[2].data['SFR_IVAR'], 
                    hdul[2].data['LOGMSTAR'],
                    hdul[2].data['LOGMSTAR_IVAR'],
                    hdul[3].data['HALPHA_EW'],
                    hdul[3].data['HALPHA_EW_IVAR'],
                    hdul[3].data['HBETA_EW'],
                    hdul[3].data['HBETA_EW_IVAR'],
                    ], 
                    names=('Z_FSF', 'TARGETID', 'DN4000', 'DN4000_IVAR', 'DN4000_MODEL', 'DN4000_MODEL_IVAR', 'ABSMAG01_SDSS_G', 'ABSMAG01_IVAR_SDSS_G', 'ABSMAG01_SDSS_R', 'ABSMAG01_IVAR_SDSS_R', 'SFR', 'SFR_IVAR', 'LOGMSTAR', 'LOGMSTAR_IVAR', 'HALPHA_EW', 'HALPHA_EW_IVAR', 'HBETA_EW', 'HBETA_EW_IVAR'))
                hdul.close()
                if h == hp[0]:
                    all_fastspecfit_table = fastspecfit_table
                else:
                    all_fastspecfit_table = vstack([all_fastspecfit_table, fastspecfit_table])
        
        all_fastspecfit_table.write(BGS_Y1_FASTSPEC_FILE, format='fits', overwrite=True)
        print(f"  Saved {len(all_fastspecfit_table)} rows to {BGS_Y1_FASTSPEC_FILE}")
        return all_fastspecfit_table

    return Table.read(BGS_Y1_FASTSPEC_FILE, format='fits')


def read_fastspecfit_y3():
    if not os.path.exists(BGS_Y3_FASTSPEC_FILE):
        print(f"BGS_Y3_FASTSPEC_FILE does not exist. Building from '{NERSC_BGS_LOA_FASTSPECFIT_DIR}'")
        hp = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
        filepattern = os.path.join(NERSC_BGS_LOA_FASTSPECFIT_DIR, "fastspec-loa-main-bright-nside1-hpXX.fits")
        for h in hp:
            filename = filepattern.replace("XX", h)
            if os.path.exists(filename):
                print(f"  Found {filename}")
                hdul = fits.open(filename, memmap=True)
                fastspecfit_table = Table([
                    hdul[1].data['Z'],
                    hdul[2].data['TARGETID'], 
                    hdul[2].data['DN4000'], 
                    hdul[2].data['DN4000_IVAR'], 
                    hdul[2].data['DN4000_MODEL'], 
                    hdul[2].data['DN4000_MODEL_IVAR'], 
                    hdul[2].data['ABSMAG01_SDSS_G'], 
                    hdul[2].data['ABSMAG01_IVAR_SDSS_G'], 
                    hdul[2].data['ABSMAG01_SDSS_R'], 
                    hdul[2].data['ABSMAG01_IVAR_SDSS_R'], 
                    hdul[2].data['SFR'], 
                    hdul[2].data['SFR_IVAR'], 
                    hdul[2].data['LOGMSTAR'],
                    hdul[2].data['LOGMSTAR_IVAR'],
                    hdul[3].data['HALPHA_EW'],
                    hdul[3].data['HALPHA_EW_IVAR'],
                    hdul[3].data['HBETA_EW'],
                    hdul[3].data['HBETA_EW_IVAR'],
                    ], 
                    names=('Z_FSF', 'TARGETID', 'DN4000', 'DN4000_IVAR', 'DN4000_MODEL', 'DN4000_MODEL_IVAR', 'ABSMAG01_SDSS_G', 'ABSMAG01_SDSS_G_IVAR', 'ABSMAG01_SDSS_R', 'ABSMAG01_SDSS_R_IVAR', 'SFR', 'SFR_IVAR', 'LOGMSTAR', 'LOGMSTAR_IVAR', 'HALPHA_EW', 'HALPHA_EW_IVAR', 'HBETA_EW', 'HBETA_EW_IVAR'))
                hdul.close()
                if h == hp[0]:
                    all_fastspecfit_table = fastspecfit_table
                else:
                    all_fastspecfit_table = vstack([all_fastspecfit_table, fastspecfit_table])
        
        all_fastspecfit_table.write(BGS_Y3_FASTSPEC_FILE, format='fits', overwrite=True)
        print(f"  Saved {len(all_fastspecfit_table)} rows to {BGS_Y3_FASTSPEC_FILE}")
        return all_fastspecfit_table

    return Table.read(BGS_Y3_FASTSPEC_FILE, format='fits')

def add_photometric_columns(existing_table, version: str):
    print(f"Adding extra photometric columns.", flush=True)
    if version == 'sv3':
        if os.path.exists(BGS_SV3_COMBINED_PHOTOMETRIC_CATALOG):
            photo_table = Table.read(BGS_SV3_COMBINED_PHOTOMETRIC_CATALOG, format='fits') # Contains SV3
        else:
            print(f"BGS_SV3_COMBINED_PHOTOMETRIC_CATALOG does not exist. Building...")
            photo_table = prepare_photo_vac('fuji')
        photo_table = photo_table[photo_table['SURVEY'] == 'sv3']
        jointype = 'left' # Just for SV3, because we have supplementary Y3 galaxies without rows in this...
    elif version == '1':  
        if os.path.exists(BGS_Y1_COMBINED_PHOTOMETRIC_CATALOG):
            photo_table = Table.read(BGS_Y1_COMBINED_PHOTOMETRIC_CATALOG, format='fits')
        else:
            print(f"BGS_Y1_COMBINED_PHOTOMETRIC_CATALOG does not exist. Building...")
            photo_table = prepare_photo_vac('iron')
        photo_table = Table.read(BGS_Y1_COMBINED_PHOTOMETRIC_CATALOG, format='fits') 
        jointype = 'inner'
    elif version == '3':
        if os.path.exists(BGS_Y3_COMBINED_PHOTOMETRIC_CATALOG):
            photo_table = Table.read(BGS_Y3_COMBINED_PHOTOMETRIC_CATALOG, format='fits')
        else: 
            print(f"BGS_Y3_COMBINED_PHOTOMETRIC_CATALOG does not exist. Building...")
            photo_table = prepare_photo_vac('loa')
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
    
    print(f"  Original len={len(existing_table):,}, photometric len={len(photo_table):,}, final len={len(final_table):,}.", flush=True)

    return final_table

def add_fastspecfit_columns(main_table, version:str):
    print("Adding fastspecfit columns to table.", flush=True)
    if version.lower() == 'sv3':
        fastspecfit_table = read_fastspecfit_sv3()
    elif version == '1':
        fastspecfit_table = read_fastspecfit_y1()
    elif version == '3':
        fastspecfit_table = read_fastspecfit_y3()
    final_table = join(main_table, fastspecfit_table, join_type='left', keys="TARGETID")
    final_table = final_table.filled(np.nan) # convert masked to nan

    no_spectra = determine_unobserved_from_z(main_table['Z'])

    # If FSF and LSScats don't agree on the Z, LSScats wins (there are more than one way to decide on final Z). 
    # In these cases, nan out the FSF columns that use the Z.
    bad_fsf = np.abs(final_table['Z'][~no_spectra] - final_table['Z_FSF'][~no_spectra]) > 0.001
    if bad_fsf.any():
        print(f"Warning: Found {bad_fsf.sum()} galaxies with large (>0.001) difference between observed Z and fastspecfit Z. Setting fastspecfit-based columns to nan for these.")
        for c in fastspecfit_table.colnames: 
            if c != 'TARGETID' and c != 'Z_FSF':
                final_table[c][~no_spectra][bad_fsf] = np.nan
        # Unclear if we need to only wipe out ABS MAGS or everything. Assuming everything to be safe.
        #final_table['ABSMAG01_SDSS_G'][~no_spectra][bad_fsf] = np.nan
        #final_table['ABSMAG01_SDSS_R'][~no_spectra][bad_fsf] = np.nan


    return final_table

def read_tiles_Y1_main():
    tiles_table = Table.read(BGS_Y1_TILES_FILE, format='csv')
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
    print("Adding NTILE_MINE to table.", flush=True)
    if year == "sv3" or year == "SV3":
        tiles_df = read_tiles_Y3_sv3()
    elif year == "1" or year == "Y1":
        tiles_df = read_tiles_Y1_main()
    elif year == "3" or year == "Y3" or year == "DA2":
        tiles_df = read_tiles_Y3_main()
    else:
        raise ValueError("Year must be 1 or 3")
    
    if isinstance(table_file, str):
        table = Table.read(table_file, format='fits')
    elif isinstance(table_file, Table):
        table = table_file
    else:
        raise ValueError("table_file must be a string or astropy Table")
    
    galaxies_df = get_radec_df(table)
    
    ntiles_inside, nearest_tile_ids = find_tiles_for_galaxies(tiles_df, galaxies_df, 15)
    if 'NTILE_MINE' in table.columns:
        table.remove_columns(['NTILE_MINE', 'NEAREST_TILEIDS'])
    table.add_column(ntiles_inside, name="NTILE_MINE")
    table.add_column(nearest_tile_ids, name="NEAREST_TILEIDS")

    return table

def add_physical_halflight_radius(table):
    print("Adding physical half-light radius to table.", flush=True)
    if 'SHAPE_R' not in table.columns:
        print("No SHAPE_R column found in table. Skipping.")
        return table

    if 'Z' not in table.columns:
        print("No Z column found in table. Skipping.")
        return table

    # Convert SHAPE_R from arcsec to kpc using the redshift
    shape_r_arcsec = table['SHAPE_R'].astype("<f8")  # Ensure it's float64
    z = table['Z'].astype("<f8")
    shape_r_kpc = get_physical_angular_diameter_distance(shape_r_arcsec, z)
    table.add_column(shape_r_kpc, name='SHAPE_R_KPC')

    return table


def create_merged_file(orig_tbl_fn : str, merged_fn : str, year : str, photoz_wspec=True):
    print(f"CREATING MERGED FILE {merged_fn} for year {year}.", flush=True)
    # 'DCHISQ' not in SV3
    columns = ['TARGETID', 'SPECTYPE', 'DEC', 'RA', 'Z_not4clus', 'FLUX_R', 'FLUX_G', 'PROB_OBS', 'ZWARN', 'DELTACHI2', 'NTILE', 'TILES', 'MASKBITS', 'SHAPE_R', 'PHOTSYS']
    if ON_NERSC:
        import fitsio
        table = Table(fitsio.read(orig_tbl_fn, columns=columns))
    else:
        table = Table.read(orig_tbl_fn, format='fits')
        table.keep_columns(columns)

    print(f"Read {len(table)} galaxies from {orig_tbl_fn}", flush=True)
    table.rename_column('Z_not4clus', 'Z')

    # Fill masked values with NaN everywhere and remove stars
    table = table.filled(np.nan)  
    table = table[table['SPECTYPE'] != 'STAR']
    print(f"Removed stars, now {len(table)} objects.", flush=True)
    table = table[table['SPECTYPE'] != 'QSO']
    print(f"Removed quasars, now {len(table)} objects.", flush=True)

    # Add additional derived columns from fastspecfit
    # The lost galaxies will get nans for the fsf columns
    table = add_fastspecfit_columns(table, year)
    print("FSF Joined", flush=True)

    # Add extra columns that were cut from LSS Catalogs from the photometric VAC
    table = add_photometric_columns(table, year)
    print("Photometric VAC Joined", flush=True)

    # Add photo-zs
    if photoz_wspec:
        table = add_photz_columns(table, IAN_PHOT_Z_FILE_WSPEC)
    else:
        table = add_photz_columns(table, IAN_PHOT_Z_FILE_NOSPEC)
    print("Photo-z Joined", flush=True)

    # Derive some luminosity / color related properties
    table = add_mag_columns(table)
    print("Mag Calculations Joined", flush=True)

    # Add information on the nearest tiles to each target for Npass filtering later
    table = add_NTILE_MINE_to_table(table, year)
    print("NTILE_MINE Joined", flush=True)
    
    table.write(merged_fn, format='fits', overwrite='True')
    print("Merged file written.")

    return table

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

def supplement_sv3_merged_file_with_y3(orig_path, supplemental_path, combined_path):
    sv3_table: Table = Table.read(orig_path, format='fits')
    y3_table: Table = Table.read(supplemental_path, format='fits')

    # Let's cut Y3 data down to targets somewhat close to SV3 regions
    # No point in keeping rest and slowing down code
    gals_coord = coord.SkyCoord(ra=y3_table['RA']*u.degree, dec=y3_table['DEC']*u.degree, frame='icrs')
    close_array = get_objects_near_sv3_regions(gals_coord, 2.5) # 2.5 deg radius is generously around the center of each SV3 region
    y3_table = y3_table[close_array]

    print(f"{len(y3_table)} galaxies will be added for SV3 NN catalog")
    assert (y3_table['NTILE_MINE'] > 9).sum() == 0, "Y3 shouldn't add any galaxies that will go into the catalog"

    # Find targets in Y3 that are already in SV3
    y3_in_sv3 = np.isin(y3_table['TARGETID'], sv3_table['TARGETID'])
    sv3_in_y3 = np.isin(sv3_table['TARGETID'], y3_table['TARGETID'])
    assert y3_in_sv3.sum() == sv3_in_y3.sum()
    print(f"Common targets between Y3 and SV3: {y3_in_sv3.sum()}; {y3_in_sv3.sum() / len(y3_table):.2%} of Y3 cut and {sv3_in_y3.sum() / len(sv3_table):.2%} of SV3")

    # Steal p_obs from SV3 for Y3 galaxies that are already in SV3
    print(f"Stealing {y3_in_sv3.sum()} p_obs values from  Y3 galaxies for SV3")
    sv3_table['PROB_OBS'][sv3_in_y3] = y3_table['PROB_OBS'][y3_in_sv3]
    sv3_table['PROB_OBS'][~sv3_in_y3] = np.nan #0.689

    # Remove rows from y3_table that have TARGETID already in sv3_table
    print(f"Removing {y3_in_sv3.sum()} (of {len(y3_table)}) Y3 galaxies that are already in SV3 catalog")
    y3_table.remove_rows(y3_in_sv3)

    #colname = 'NEAREST_TILEIDS'
    #print(sv3_table[colname].shape)
    #print(y3_table[colname].shape)

    sv3_table.remove_column('TILES')
    y3_table.remove_column('TILES')

    combined = vstack([sv3_table, y3_table], join_type='outer')

    # Ensure resultant data is what we want
    print(sv3_table.columns)
    print(y3_table.columns)
    print(combined.columns)

    combined.write(combined_path, format='fits', overwrite=True)


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

def read_randoms(base: str, n: int) -> pd.DataFrame:
    rtable = fitsio.read(base.replace("X", str(n)), columns=['RA', 'DEC', 'NTILE', 'PHOTSYS'])

    randoms_df = pd.DataFrame({'RA': rtable['RA'].astype("<f8"), 'DEC': rtable['DEC'].astype("<f8"), 'NTILE': rtable['NTILE'].astype("<i8"), 'PHOTSYS': rtable['PHOTSYS']})

    #if 'WEIGHT' in rtable.columns:
    #    randoms_df['WEIGHT'] = rtable['WEIGHT'].astype("<f8")

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
    pickle.dump(randoms_df0, open(RANDOMS_Y1_0_WITHMYNTILE, "wb"))

    # Full version may be needed later but not for now

def build_y3_randoms_files():
    tiles = read_tiles_Y3_main()

    # Make a mini version from one randoms file
    randoms_df0 = read_randoms(BGS_Y3_RAND_FILE, 0)
    ntiles_inside, nearest_tile_ids = find_tiles_for_galaxies(tiles, randoms_df0, NTILE_MIN_TO_FIND)
    randoms_df0['NTILE_MINE'] = ntiles_inside
    pickle.dump(randoms_df0, open(RANDOMS_Y3_0_WITHMYNTILE, "wb"))

    # Full version may be needed later but not for now
