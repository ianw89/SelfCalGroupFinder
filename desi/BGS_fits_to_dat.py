import numpy as np
import sys
from pyutils import *
from astropy.table import Table

def usage():
    print("Usage: python3 BGS_fits_to_dat.py [mode] [APP_MAG_CUT] [CATALOG_APP_MAG_CUT] [input_filename].hdf5 [output_filename] [COLORS_ON]")
    print("  Mode is 1 for OBSERVED 1+ PASSES, 2 for OBSERVED 3+ PASSES, 5 for SIMPLE v2, and 6 for Simple v4 ")
    print("  Will generate [output_filename].dat for use with kdGroupFinder and [output_filename]_galprops.dat with additional galaxy properties.")
    print("  These two files will have galaxies indexed in the same way (line-by-line matched).")

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

    mode = int(sys.argv[1])
    app_mag_cut = float(sys.argv[2])
    catalog_app_mag_cut = float(sys.argv[3])
    colors_on = sys.argv[6] == "1"

    print("Reading FITS data from ", sys.argv[4])
    # Unobserved galaxies have masked rows in appropriate columns of the table
    table = Table.read(sys.argv[4], format='fits')
    
    outname_base = sys.argv[5] 

    pre_process_BGS(table, mode, outname_base, app_mag_cut, catalog_app_mag_cut, colors_on)



    ################
    # MAIN CODE
    ################

def pre_process_BGS(table, mode, outname_base, APP_MAG_CUT, CATALOG_APP_MAG_CUT, COLORS_ON):
    """
    Pre-processes the BGS data for use with the group finder.
    """
    Z_MIN = 0.001
    Z_MAX = 0.8
    
    FOOTPRINT_FRAC_1pass = 0.187906 # As calculated from the randoms with 1-pass coverage
    FOOTPRINT_FRAC = 0.0649945 # As calculated from the randoms with 3-pass coverage. 1310 degrees
    # TODO update footprint with new calculation from ANY. It shouldn't change.
    frac_area = FOOTPRINT_FRAC
    if mode == Mode.ALL.value:
        frac_area = FOOTPRINT_FRAC_1pass

    if mode == Mode.ALL.value:
        print("\nMode FIBER ASSIGNED ONLY 1+ PASSES")
    elif mode == Mode.FIBER_ASSIGNED_ONLY.value:
        print("\nMode FIBER ASSIGNED ONLY 3+ PASSES")
    elif mode == Mode.NEAREST_NEIGHBOR.value:
        print("\nMode NEAREST_NEIGHBOR")
    elif mode == Mode.FANCY.value:
        print("\nMode FANCY NOT SUPPORTED")
        exit(2)
    elif mode == Mode.SIMPLE.value:
        print("\nMode SIMPLE v2")
    elif mode == Mode.SIMPLE_v4.value:
        print("\nMode SIMPLE v4")

    print(f"Color classification sent to group finder: {COLORS_ON}")

    # astropy's Table used masked arrays, so we have to use .data.data to get the actual data
    # The masked rows are unobserved targets
    obj_type = table['SPECTYPE'].data.data
    dec = table['DEC']
    ra = table['RA']
    z_obs = table['Z_not4clus'].data.data
    target_id = table['TARGETID']
    app_mag_r = get_app_mag(table['FLUX_R'])
    app_mag_g = get_app_mag(table['FLUX_G'])
    g_r = app_mag_g - app_mag_r
    p_obs = table['PROB_OBS']
    unobserved = table['Z_not4clus'].mask # the masked values are what is unobserved
    deltachi2 = table['DELTACHI2'].data.data  
    dn4000 = table['DN4000'].data.data
    
    orig_count = len(dec)
    print(orig_count, "objects in FITS file")

    # If an observation was made, some automated system will evaluate the spectra and auto classify the SPECTYPE
    # as GALAXY, QSO, STAR. It is null (and masked) for non-observed targets.
    # NTILE tracks how many DESI pointings could have observed the target (at fiber level)
    # NTILE_MINE gives how many tiles include just from inclusion in circles drawn around tile centers
    # null values (masked rows) are unobserved targets; not all columns are masked though

    # Make filter arrays (True/False values)
    three_pass_filter = table['NTILE_MINE'] >= 3 # 3pass coverage
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

    if mode == Mode.ALL.value: # ALL is misnomer here it means 1pass or more
        keep = np.all([observed_requirements], axis=0)

    if mode == Mode.FIBER_ASSIGNED_ONLY.value: # means 3pass 
        keep = np.all([three_pass_filter, observed_requirements], axis=0)

    if mode == Mode.NEAREST_NEIGHBOR.value or mode == Mode.SIMPLE.value or mode == Mode.SIMPLE_v4.value:
        keep = np.all([three_pass_filter, np.logical_or(observed_requirements, unobserved)], axis=0)

        # Filter down inputs to the ones we want in the catalog for NN and similar calculations
        # TODO why bother with this for the real data? Use all we got, right? 
        # I upped the cut to 21 so it doesn't do anything
        catalog_bright_filter = app_mag_r < CATALOG_APP_MAG_CUT 
        catalog_keep = np.all([galaxy_observed_filter, catalog_bright_filter, redshift_filter, redshift_hi_filter, deltachi2_filter], axis=0)
        catalog_ra = ra[catalog_keep]
        catalog_dec = dec[catalog_keep]
        z_obs_catalog = z_obs[catalog_keep]
        catalog_gmr = app_mag_g[catalog_keep] - app_mag_r[catalog_keep]
        catalog_G_k = app_mag_to_abs_mag_k(app_mag_g[catalog_keep], z_obs_catalog, catalog_gmr, band='g')
        catalog_R_k = app_mag_to_abs_mag_k(app_mag_r[catalog_keep], z_obs_catalog, catalog_gmr, band='r')
        catalog_G_R_k = catalog_G_k - catalog_R_k
        catalog_quiescent = is_quiescent_BGS_gmr(abs_mag_r_to_log_solar_L(catalog_R_k), catalog_G_R_k)

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
    dn4000 = dn4000[keep]

    count = len(dec)
    print(count, "galaxies left after filters.")
    print(f'{unobserved.sum() } remaining galaxies that need redshifts')
    print(f'{100*unobserved.sum() / len(unobserved) :.1f}% of remaining galaxies need redshifts')
    #print(f'Min z: {min(z_obs):f}, Max z: {max(z_obs):f}')


    z_eff = np.copy(z_obs)

    if mode == Mode.NEAREST_NEIGHBOR.value:

        catalog = coord.SkyCoord(ra=catalog_ra*u.degree, dec=catalog_dec*u.degree, frame='icrs')
        to_match = coord.SkyCoord(ra=ra[unobserved]*u.degree, dec=dec[unobserved]*u.degree, frame='icrs')

        idx, d2d, d3d = coord.match_coordinates_sky(to_match, catalog, storekdtree=False)

        # i is the index of the full sized array that needed a NN z value
        # j is the index along the to_match list corresponding to that
        # idx are the indexes of the NN from the catalog
        assert len(indexes_not_assigned) == len(idx)

        print("Copying over NN properties... ", end='\r')
        j = 0
        for i in indexes_not_assigned:  
            z_eff[i] = z_obs_catalog[idx[j]]
            j = j + 1 
        print("Copying over NN properties... done")


    if mode == Mode.SIMPLE.value or mode == Mode.SIMPLE_v4.value:
        if mode == Mode.SIMPLE.value:
            ver = '2.0'
        elif mode == Mode.SIMPLE_v4.value:
            ver = '4.0'
        with SimpleRedshiftGuesser(app_mag_r[observed], z_obs[observed], ver) as scorer:

            catalog = coord.SkyCoord(ra=catalog_ra*u.degree, dec=catalog_dec*u.degree, frame='icrs')
            to_match = coord.SkyCoord(ra=ra[unobserved]*u.degree, dec=dec[unobserved]*u.degree, frame='icrs')

            # neighbor_indexes is the index of the nearest galaxy in the catalog arrays
            neighbor_indexes, d2d, d3d = coord.match_coordinates_sky(to_match, catalog, storekdtree=False)
            ang_distances = d2d.to(u.arcsec).value

            # We need to guess a color for the unobserved galaxies to help the redshift guesser
            # Multiple possible ideas

            # 1) Use the NN's redshift to k-correct the lost galaxies
            #abs_mag_R = app_mag_to_abs_mag(app_mag_r[unobserved], z_obs_catalog[neighbor_indexes])
            #abs_mag_R_k = k_correct(abs_mag_R, z_obs_catalog[neighbor_indexes], app_mag_g[unobserved] - app_mag_r[unobserved])
            #abs_mag_G = app_mag_to_abs_mag(app_mag_g[unobserved], z_obs_catalog[neighbor_indexes])
            #abs_mag_G_k = k_correct(abs_mag_G, z_obs_catalog[neighbor_indexes], app_mag_g[unobserved] - app_mag_r[unobserved], band='g')
            #log_L_gal = abs_mag_r_to_log_solar_L(abs_mag_R_k)
            #G_R_k = abs_mag_G_k - abs_mag_R_k
            #quiescent_gmr = is_quiescent_BGS_gmr(log_L_gal, G_R_k)

            # 2) Use an uncorrected apparent g-r color cut to guess if the galaxy is quiescent or not
            quiescent_gmr = is_quiescent_lost_gal_guess(app_mag_g[unobserved] - app_mag_r[unobserved]).astype(int)
            
            assert len(quiescent_gmr) == len(ang_distances)


            print(f"Assigning missing redshifts... ")   
            j = 0 # j counts the number of unobserved galaxies in the catalog that have been assigned a redshift thus far
            for i in indexes_not_assigned: # i is the index of the unobserved galaxy in the main arrays
                if j%10000==0:
                    print(f"{j}/{len(to_match)} complete", end='\r')

                catalog_idx = neighbor_indexes[j]
                chosen_z, isNN = scorer.choose_redshift(z_obs_catalog[catalog_idx], ang_distances[j], p_obs[i], app_mag_r[i], quiescent_gmr[j], catalog_quiescent[catalog_idx])
                
                z_eff[i] = chosen_z

                j = j + 1 

            print(f"{j}/{len(to_match)} complete")

    assert np.all(z_eff > 0.0)

    # Some of this is redudant with catalog calculations but oh well
    abs_mag_R = app_mag_to_abs_mag(app_mag_r, z_eff)
    abs_mag_R_k = k_correct(abs_mag_R, z_eff, g_r, band='r')
    abs_mag_G = app_mag_to_abs_mag(app_mag_g, z_eff)
    abs_mag_G_k = k_correct(abs_mag_G, z_eff, g_r, band='g')
    log_L_gal = abs_mag_r_to_log_solar_L(abs_mag_R_k) 
    G_R_k = abs_mag_G_k - abs_mag_R_k
    quiescent = is_quiescent_BGS_gmr(log_L_gal, G_R_k)
    print(f"{quiescent.sum()} quiescent galaxies, {len(quiescent) - quiescent.sum()} star-forming galaxies")
     #print(f"Quiescent agreement between g-r and Dn4000 for observed galaxies: {np.sum(quiescent_gmr[observed] == quiescent[observed]) / np.sum(observed)}")



    # the vmax should be calculated from un-k-corrected magnitudes
    V_max = get_max_observable_volume(abs_mag_R, z_eff, APP_MAG_CUT, frac_area)

    if not COLORS_ON:
        quiescent = np.zeros(count, dtype=np.int8)
    
    # TODO get galaxy concentration from somewhere
    chi = np.zeros(count, dtype=np.int8) 

    # Output files
    galprops = np.column_stack([
        np.array(app_mag_r, dtype='str'), 
        np.array(target_id, dtype='str'), 
        np.array(unobserved, dtype='str'),
        np.array(G_R_k, dtype='str'),
        np.array(dn4000, dtype='str'),
        ])
    write_dat_files(ra, dec, z_eff, log_L_gal, V_max, quiescent, chi, outname_base, frac_area, galprops)

if __name__ == "__main__":
    main()