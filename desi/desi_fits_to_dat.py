import numpy as np
import sys
from pyutils import *
from astropy.table import Table, join

def usage():
    print("Usage: python3 desi_fits_to_dat.py [mode] [APP_MAG_CUT] [CATALOG_APP_MAG_CUT] [input_filename].hdf5 [output_filename] [COLORS_ON]")
    print("  Mode is 1 for OBSERVED 1+ PASSES, 2 for OBSERVED 3+ PASSES, and 5 for SIMPLE ")
    print("  Will generate [output_filename].dat for use with kdGroupFinder and [output_filename]_galprops.dat with additional galaxy properties.")
    print("  These two files will have galaxies indexed in the same way (line-by-line matched).")

# TODO ensure this is right
def get_app_mag(FLUX_R):
    return 22.5 - 2.5*np.log10(FLUX_R)

def main():
    """
    INPUT FORMAT: FITS FILE 

    ['TARGETID', 'BGS_TARGET', 'TARGET_STATE', 'LOCATION', 'SPECTYPE', 'Z_not4clus', 'ZERR', 'ZWARN', 'ZWARN_MTL', 'FLUX_R', 'FLUX_G', 'RA', 'DEC', 'BITWEIGHTS', 'PROB_OBS', 'MW_TRANSMISSION_R', 'MW_TRANSMISSION_G']


     OUTPUT FORMAT FOR GROUP FINDER: space seperated values. Columns, in order, are:
    
     RA [degrees; float] - right ascension of the galaxy
     Dec [degrees; float] - decliration of the galaxy
     z [float] - redshift of the galaxy
     log L_gal [L_sol/h^-2; float] - r-band luminosity of the galaxy in solar units, assuming M_r,sol=4.65 and h=1.
     V_max [(Mpc/h)^3; float] - maximum volume at which each galaxy can be observed in SDSS. Taken from NYU VAGC
     color_flag [int]- 1=quiescent, 0=star-forming. This is based on the GMM cut of the Dn4000 data in Tinker 2020.
     chi [dimensionless] - THis is the normalized galaxy concentration. See details in both papers.

    OUTPUT FORMAT FOR GALPROPS: space separated values not needed by the group finder code. See bottom of file.
    """
    
    ################
    # ERROR CHECKING
    ################
    if len(sys.argv) != 7:
        print("Error 1")
        usage()
        exit(1)

    if int(sys.argv[1]) not in [member.value for member in Mode]:
        print("Error 2")
        usage()
        exit(2)

    if not sys.argv[4].endswith('.fits'):
        print("Error 3")
        usage()
        exit(3)

    ################
    # MAIN CODE
    ################

    mode = int(sys.argv[1])
    if mode == Mode.ALL.value:
        print("\nMode ALL - FIBER ASSIGNED ONLY WITH ANY NUMBER OF PASSES")
    elif mode == Mode.FIBER_ASSIGNED_ONLY.value:
        print("\nMode FIBER_ASSIGNED_ONLY")
    elif mode == Mode.NEAREST_NEIGHBOR.value:
        print("\nMode NEAREST_NEIGHBOR NOT SUPPORTED")
        exit(2)
    elif mode == Mode.FANCY.value:
        print("\nMode FANCY NOT SUPPORTED")
        exit(2)
    elif mode == Mode.SIMPLE.value:
        print("\nMode SIMPLE")

    APP_MAG_CUT = float(sys.argv[2])
    CATALOG_APP_MAG_CUT = float(sys.argv[3])
    Z_MIN = 0.01
    Z_MAX = 0.8
    FOOTPRINT_FRAC_1pass = 0.187906 # As calculated from the randoms with 1-pass coverage
    FOOTPRINT_FRAC = 0.0317538 # As calculated from the randoms with 3-pass coverage
    # TODO update footprint with new calculation from ANY. It shouldn't change.
    frac_area = FOOTPRINT_FRAC
    if mode == Mode.ALL.value:
        frac_area = FOOTPRINT_FRAC_1pass

    COLORS_ON = sys.argv[6] == "1"
    print(f"Color classificaiton sent to group finder: {COLORS_ON}")

    print("Reading FITS data from ", sys.argv[4])
    # Unobserved galaxies have masked rows in appropriate columns of the table
    u_table = Table.read(sys.argv[4], format='fits')

    # Temp hack to get p_obs as ANY doesn't have it, unlike BRIGHT
    p_table = Table.read('bin/mainbw-bright-allTiles_v1.fits')
    u_table = join(u_table, p_table, keys='TARGETID')
    
    outname_base = sys.argv[5] 

    # astropy's Table used masked arrays, so we have to use .data.data to get the actual data
    # The masked rows are unobserved targets
    obj_type = u_table['SPECTYPE'].data.data
    dec = u_table['DEC']
    ra = u_table['RA']
    z_obs = u_table['Z_not4clus'].data.data
    target_id = u_table['TARGETID']
    app_mag_r = get_app_mag(u_table['FLUX_R'])
    app_mag_g = get_app_mag(u_table['FLUX_G'])
    g_r = app_mag_g - app_mag_r
    p_obs = u_table['PROB_OBS']
    unobserved = u_table['ZWARN'] == 999999
    deltachi2 = u_table['DELTACHI2'].data.data  

    orig_count = len(dec)
    print(orig_count, "objects in FITS file")


    # If an observation was made, some automated system will evaluate the spectra and auto classify the SPECTYPE
    # as GALAXY, QSO, STAR. It is null (and masked) for non-observed targets.
    # NTILE tracks how many DESI pointings could have observed the target
    # null values (masked rows) are unobserved targets; not all columns are masked though

    # Make filter arrays (True/False values)
    three_pass_filter = u_table['NTILE'] >= 3 # 3pass coverage
    galaxy_observed_filter = obj_type == b'GALAXY'
    app_mag_filter = app_mag_r < APP_MAG_CUT
    redshift_filter = z_obs > Z_MIN
    redshift_hi_filter = z_obs < Z_MAX
    deltachi2_filter = deltachi2 > 40 # Ensures that there wasn't another z with similar likelihood from the z fitting code
    observed_requirements = np.all([galaxy_observed_filter, app_mag_filter, redshift_filter, redshift_hi_filter, deltachi2_filter], axis=0)
    
    # treat low deltachi2 as unobserved
    treat_as_unobserved = np.all([galaxy_observed_filter, app_mag_filter, np.invert(deltachi2_filter)], axis=0)
    #print(f"We have {np.count_nonzero(treat_as_unobserved)} observed galaxies with deltachi2 < 40 to add to the unobserved pool")
    unobserved = np.all([app_mag_filter, np.logical_or(unobserved, treat_as_unobserved)], axis=0)
    #print(f"We have {np.count_nonzero(unobserved)} observed galaxies with deltachi2 < 40 to add to the unobserved pool")

    if mode == Mode.ALL.value: # ALL is misnomer here it means 1pass or more
        keep = np.all([observed_requirements], axis=0)

    if mode == Mode.FIBER_ASSIGNED_ONLY.value: # means 3pass 
        keep = np.all([three_pass_filter, observed_requirements], axis=0)

    if mode == Mode.SIMPLE.value:
        keep = np.all([three_pass_filter, np.logical_or(observed_requirements, unobserved)], axis=0)

        # Filter down inputs to the ones we want in the catalog for NN and similar calculations
        # TODO why bother with this for the real data? Use all we got, right? 
        # I upped the cut to 21 so it doesn't do anything
        catalog_bright_filter = app_mag_r < CATALOG_APP_MAG_CUT 
        # TODO Shouldn't 3pass filter be in here too? I guess it doesn't matter
        catalog_keep = np.all([galaxy_observed_filter, catalog_bright_filter, redshift_filter, redshift_hi_filter, deltachi2_filter], axis=0)
        catalog_ra = ra[catalog_keep]
        catalog_dec = dec[catalog_keep]
        z_obs_catalog = z_obs[catalog_keep]

        print(len(z_obs_catalog), "galaxies in the NN catalog.")

    # Apply filters
    obj_type = obj_type[keep]
    dec = dec[keep]
    ra = ra[keep]
    z_obs = z_obs[keep]
    target_id = target_id[keep]
    app_mag_r = app_mag_r[keep]
    app_mag_g = app_mag_g[keep]
    p_obs = p_obs[keep]
    unobserved = unobserved[keep]
    observed = np.invert(unobserved)
    indexes_not_assigned = np.argwhere(unobserved)
    deltachi2 = deltachi2[keep]
    g_r = g_r[keep]

    count = len(dec)
    print(count, "galaxies left after filters.")
    print(f'{unobserved.sum() } remaining galaxies that need redshifts')
    print(f'{100*unobserved.sum() / len(unobserved) :.1f}% of remaining galaxies need redshifts')
    #print(f'Min z: {min(z_obs):f}, Max z: {max(z_obs):f}')

    z_eff = np.copy(z_obs)

    if mode == Mode.SIMPLE.value:
        with SimpleRedshiftGuesser(app_mag_r[observed], z_obs[observed]) as scorer:

            catalog = coord.SkyCoord(ra=catalog_ra*u.degree, dec=catalog_dec*u.degree, frame='icrs')
            to_match = coord.SkyCoord(ra=ra[unobserved]*u.degree, dec=dec[unobserved]*u.degree, frame='icrs')

            neighbor_indexes, d2d, d3d = coord.match_coordinates_sky(to_match, catalog, storekdtree=False)
            ang_distances = d2d.to(u.arcsec).value

            print(f"Assinging missing redshifts... ")   
            j = 0 # j counts the number of unobserved galaxies in the catalog that have been assigned a redshift thus far
            for i in indexes_not_assigned: # i is the index of the unobserved galaxy in the main arrays
                if j%10000==0:
                    print(f"{j}/{len(to_match)} complete", end='\r')

                chosen_z, isNN = scorer.choose_redshift(z_obs_catalog[neighbor_indexes[j]], ang_distances[j], p_obs[i], app_mag_r[i])
                z_eff[i] = chosen_z
                j = j + 1 

            print(f"{j}/{len(to_match)} complete")

    abs_mag_R = app_mag_to_abs_mag(app_mag_r, z_eff)
    abs_mag_R_k = k_correct(abs_mag_R, z_eff, g_r)

    # the luminosities sent to the group finder will be k-corrected to z=0.1
    log_L_gal = abs_mag_r_to_log_solar_L(abs_mag_R_k) 

    # the vmax should be calculated from un-k-corrected magnitudes
    V_max = get_max_observable_volume(abs_mag_R, z_eff, APP_MAG_CUT, frac_area)

    abs_mag_G = app_mag_to_abs_mag(app_mag_g, z_eff)
    abs_mag_G_k = k_correct(abs_mag_G, z_eff, g_r, band='g')
    
    G_R_k = abs_mag_G_k - abs_mag_R_k
    
    if COLORS_ON:
        # Use the k-corrected abs mags to define galaxies as quiescent or star-forming
        # TODO make the cut per logLgal bin instead. Fit the results and make a formula like SDSS Dn4000 did
        quiescent = is_quiescent_BGS_gmr(log_L_gal, G_R_k).astype(int) 
        print(f"{quiescent.sum()} quiescent galaxies, {len(quiescent) - quiescent.sum()} star-forming galaxies")
    else:
        quiescent = np.zeros(count, dtype=np.int8)
    
    # TODO get galaxy concentration from somewhere
    chi = np.zeros(count, dtype=np.int8) 

    # Output files
    galprops = np.column_stack([
        np.array(app_mag_r, dtype='str'), 
        np.array(target_id, dtype='str'), 
        np.array(unobserved, dtype='str'),
        np.array(G_R_k, dtype='str'),
        ])
    write_dat_files(ra, dec, z_eff, log_L_gal, V_max, quiescent, chi, outname_base, frac_area, galprops)

if __name__ == "__main__":
    main()