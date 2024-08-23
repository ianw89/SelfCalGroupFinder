import numpy as np
import h5py
import sys
if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from pyutils import *
from dataloc import *

# Chooses 1 of the 2048 fiber assignment realizations with this bitstring and BITWORD as 'bitweight[0-31]
BITWORD = 'bitweight0'
BIT_CHOICE = 0
FIBER_ASSIGNED_SELECTOR = 2**BIT_CHOICE


def usage():
    print("Usage: python3 hdf5_to_dat.py [mode] [APP_MAG_CUT] [CATALOG_APP_MAG_CUT] [input_filename].hdf5 [output_filename] [colors]")
    print("  Mode is 1 for ALL, 2 for FIBER_ASSIGNED_ONLY, and 3 for NEAREST_NEIGHBOR, 4 for FANCY, 5 for SIMPLE ")
    print("  Will generate [output_filename].dat for use with kdGroupFinder and [output_filename]_galprops.dat with additional galaxy properties.")
    print("  These two files will have galaxies indexed in the same way (line-by-line matched).")


def main():
    """
    INPUT FORMAT: HDF5 FILE 

    Data/
    'abs_mag',
    'app_mag',
    'dec',
    'g_r',
    'galaxy_type',
    'halo_mass',
    'mxxl_id',
    'ra',
    'snap',
    'z_cos',
    'z_obs'

     OUTPUT FORMAT FOR GROUP FINDER: space seperated values. Columns, in order, are:
    
     RA [degrees; float] - right ascension of the galaxy
     Dec [degrees; float] - decliration of the galaxy
     z [float] - redshift of the galaxy
     log L_gal [L_sol/h^-2; float] - r-band luminosity of the galaxy in solar units, assuming M_r,sol=4.65 and h=1.
     V_max [(Mpc/h)^3; float] - maximum volume at which each galaxy can be observed in SDSS. Taken from NYU VAGC
     color_flag [int]- 1=quiescent, 0=star-forming. This is based on the GMM cut of the Dn4000 data in Tinker 2020.
     chi [dimensionless] - THis is the normalized galaxy concentration. See details in both papers.

    OUTPUT FORMAT FOR GALPROPS: space separated values not needed by the group finder code. 
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

    if not sys.argv[4].endswith('.hdf5'):
        print("Error 3")
        usage()
        exit(3)

    ################
    # MAIN CODE
    ################

    mode = int(sys.argv[1])
    if mode == Mode.ALL.value:
        print("\nMode ALL")
    elif mode == Mode.FIBER_ASSIGNED_ONLY.value:
        print("\nMode FIBER_ASSIGNED_ONLY")
    elif mode == Mode.NEAREST_NEIGHBOR.value:
        print("\nMode NEAREST_NEIGHBOR")
    elif mode == Mode.FANCY.value:
        print("\nMode FANCY")
    elif mode == Mode.SIMPLE.value:
        print("\nMode SIMPLE v2")
    elif mode == Mode.SIMPLE_v4.value:
        print("\nMode SIMPLE v4")
    else:
        print("Unknown Mode")
        exit(4)

    app_mag_cut = float(sys.argv[2])
    catalog_app_mag_cut = float(sys.argv[3])
    colors_on = sys.argv[6] == "1"

    pre_process_mxxl(sys.argv[4], mode, sys.argv[5], app_mag_cut, catalog_app_mag_cut, colors_on)

# TODO this is absurdly slow now somehow
def pre_process_mxxl(in_filepath: str, mode: int, outname_base: str, APP_MAG_CUT: float, CATALOG_APP_MAG_CUT: float, COLORS_ON: bool):

    print("Reading HDF5 data from ", in_filepath)
    infile = h5py.File(in_filepath, 'r')
    print(list(infile['Data']))

    FOOTPRINT_FRAC = 14800 / 41253
    print(f"Color classification sent to group finder: {COLORS_ON}")

    # read everything we need into memory
    dec = infile['Data/dec'][:]
    ra = infile['Data/ra'][:]
    z_obs = infile['Data/z_obs'][:]
    app_mag = infile['Data/app_mag'][:]
    g_r = infile['Data/g_r'][:]
    #abs_mag = infile['Data/abs_mag'][:] # We aren't using these; computing ourselves. 
    galaxy_type = infile['Data/galaxy_type'][:]
    mxxl_halo_mass = infile['Data/halo_mass'][:]
    mxxl_halo_id = infile['Data/mxxl_id'][:]
    observed = infile['Weight/'+BITWORD][:] & FIBER_ASSIGNED_SELECTOR 
    observed = observed.astype(bool)

    orig_count = len(dec)
    print(orig_count, "galaxies in HDF5 file")

    redshift_filter = z_obs > 0 # makes a filter array (True/False values)

    # Filter down inputs to the ones we want in the catalog for NN and similar calculations
    catalog_bright_filter = app_mag < CATALOG_APP_MAG_CUT 
    catalog_keep = np.all([catalog_bright_filter, redshift_filter, observed], axis=0)
    catalog_ra = ra[catalog_keep]
    catalog_dec = dec[catalog_keep]
    z_obs_catalog = z_obs[catalog_keep]
    halo_mass_catalog = mxxl_halo_mass[catalog_keep]
    halo_id_catalog = mxxl_halo_id[catalog_keep]
    catalog_gmr = g_r[catalog_keep]
    catalog_R_k = app_mag_to_abs_mag_k(app_mag[catalog_keep], z_obs_catalog, catalog_gmr, band='r')
    catalog_quiescent = is_quiescent_BGS_gmr(abs_mag_r_to_log_solar_L(catalog_R_k), catalog_gmr)

    # Filter down inputs we want to actually process and keep
    bright_filter = app_mag < APP_MAG_CUT # makes a filter array (True/False values)
    keep = np.all([bright_filter, redshift_filter], axis=0)
    dec = dec[keep]
    ra = ra[keep]
    z_obs = z_obs[keep]
    app_mag = app_mag[keep]
    g_r = g_r[keep]
    #abs_mag = abs_mag[keep]
    
    galaxy_type = galaxy_type[keep]
    mxxl_halo_mass = mxxl_halo_mass[keep]
    mxxl_halo_id = mxxl_halo_id[keep]
    assigned_halo_mass = np.copy(mxxl_halo_mass)
    assigned_halo_id = np.copy(mxxl_halo_id)
    observed = observed[keep]
    unobserved = np.invert(observed)
    indexes_not_assigned = np.argwhere(unobserved)

    # Want 0 for observed (DESI OR SDSS fill in), 1 for NN-assigned, 2 for other assigned
    z_assigned_flag = np.zeros(len(z_obs), dtype=np.int8)


    count = len(dec)
    print(count, "galaxies left after apparent mag cut at {0}".format(APP_MAG_CUT))
    print(np.sum(observed), "galaxies were assigned a fiber")
    print(f"Catalog for nearest neighbor calculations is of size {len(catalog_ra)}")


    with open(MXXL_PROB_OBS_FILE, 'rb') as f:
       prob_obs = np.load(f)
    assert len(prob_obs) == orig_count
    prob_obs = prob_obs[keep]

    # z_eff: same as z_obs if a fiber was assigned and thus a real redshift measurement was made
    # otherwise, it is an assigned value.
    # nearest neighbor will find the nearest (measured) galaxy and use its redshift.
    z_eff = np.copy(z_obs)


    if mode == Mode.FIBER_ASSIGNED_ONLY.value:
        # Filter it all down to just the ones with fiber's assigned
        dec = dec[observed]
        ra = ra[observed]
        z_obs = z_obs[observed]
        app_mag = app_mag[observed]
        g_r = g_r[observed]
        #abs_mag = abs_mag[observed]
        galaxy_type = galaxy_type[observed]
        mxxl_halo_mass = mxxl_halo_mass[observed]
        mxxl_halo_id = mxxl_halo_id[observed]
        assigned_halo_mass = assigned_halo_mass[observed]
        assigned_halo_id = assigned_halo_id[observed]
        z_eff = z_eff[observed]
        prob_obs = prob_obs[observed]
        observed = observed[observed]
        unobserved = np.invert(observed)
        assert np.all(observed)
        count = len(dec)

    
    elif mode == Mode.NEAREST_NEIGHBOR.value:

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
            assigned_halo_mass[i] = halo_mass_catalog[idx[j]]
            assigned_halo_id[i] = halo_id_catalog[idx[j]]
            j = j + 1 
        print("Copying over NN properties... done")

        z_assigned_flag[unobserved] = 1


    elif mode == Mode.FANCY.value:

        NUM_NEIGHBORS = 10
        with FancyRedshiftGuesser(NUM_NEIGHBORS) as scorer:

            catalog = coord.SkyCoord(ra=catalog_ra*u.degree, dec=catalog_dec*u.degree, frame='icrs')
            to_match = coord.SkyCoord(ra=ra[unobserved]*u.degree, dec=dec[unobserved]*u.degree, frame='icrs')
            
            neighbor_indexes = np.zeros(shape=(NUM_NEIGHBORS, len(to_match)), dtype=np.int32) # indexes point to CATALOG locations
            ang_distances = np.zeros(shape=(NUM_NEIGHBORS, len(to_match)))

            print(f"Finding nearest {NUM_NEIGHBORS} neighbors... ", end='\r')   
            for n in range(0, NUM_NEIGHBORS):
                idx, d2d, d3d = coord.match_coordinates_sky(to_match, catalog, nthneighbor=n+1, storekdtree='mxxl_fiber_assigned_tree')
                neighbor_indexes[n] = idx
                ang_distances[n] = d2d.to(u.arcsec).value
            print(f"Finding nearest {NUM_NEIGHBORS} neighbors... done")   


            print(f"Assinging missing redshifts... ")   
            j = 0
            for i in indexes_not_assigned:    
                if j%10000==0:
                    print(f"{j}/{len(to_match)} complete", end='\r')

                neighbors = neighbor_indexes[:,j]
                neighbors_z = z_obs_catalog[neighbors]
                neighbors_ang_dist = ang_distances[:,j]
                my_prob_obs = prob_obs[i]
                my_app_mag = app_mag[i]

                winning_num = scorer.choose_winner(neighbors_z, neighbors_ang_dist, my_prob_obs, my_app_mag, z_obs[i])
                winner_index = neighbors[winning_num]

                z_eff[i] = z_obs_catalog[winner_index] 
                assigned_halo_mass[i] = halo_mass_catalog[winner_index]
                assigned_halo_id[i] = halo_id_catalog[winner_index]
                j = j + 1 

                z_assigned_flag[i] = 1 # TODO


            print(f"{j}/{len(to_match)} complete")

        
    elif mode == Mode.SIMPLE.value or mode == Mode.SIMPLE_v4.value:
        if mode == Mode.SIMPLE.value:
            ver = '2.0'
        elif mode == Mode.SIMPLE_v4.value:
            ver = '4.0'


        # We need to guess a color for the unobserved galaxies to help the redshift guesser
        # For MXXL we have 0.1^G-R even for lost galaxies so this isn't quite like real BGS situation
        # TODO change if needed
        quiescent_gmr = np.zeros(count, dtype=int)
        np.put(quiescent_gmr, indexes_not_assigned, is_quiescent_BGS_gmr(None, g_r[unobserved]).astype(int))

        with SimpleRedshiftGuesser(app_mag[observed], z_obs[observed], ver) as scorer:

            catalog = coord.SkyCoord(ra=catalog_ra*u.degree, dec=catalog_dec*u.degree, frame='icrs')
            to_match = coord.SkyCoord(ra=ra[unobserved]*u.degree, dec=dec[unobserved]*u.degree, frame='icrs')
            
            neighbor_indexes, d2d, d3d = coord.match_coordinates_sky(to_match, catalog, storekdtree=False)
            ang_distances = d2d.to(u.arcsec).value

            print(f"Assigning missing redshifts... ")   
            chosen_z, isNN = scorer.choose_redshift(z_obs_catalog[neighbor_indexes], ang_distances, prob_obs[unobserved], app_mag[unobserved], quiescent_gmr, catalog_quiescent[neighbor_indexes])
            z_eff[unobserved] = chosen_z
            z_assigned_flag[unobserved] = np.where(isNN, 1, 2)
            assigned_halo_mass = np.where(isNN, halo_mass_catalog[neighbor_indexes], -1)
            assigned_halo_id = np.where(isNN, halo_id_catalog[neighbor_indexes], -1)
            print(f"Assigning missing redshifts complete.")   
    

    abs_mag = app_mag_to_abs_mag(app_mag, z_eff)
    abs_mag_k = k_correct(abs_mag, z_eff, g_r)

    # the luminosities sent to the group finder will be k-corrected to z=0.1
    log_L_gal = abs_mag_r_to_log_solar_L(abs_mag_k) 

    # the vmax should be calculated from un-k-corrected magnitudes
    V_max = get_max_observable_volume(abs_mag, z_eff, APP_MAG_CUT, FOOTPRINT_FRAC)

    if COLORS_ON:
        quiescent = is_quiescent_BGS_gmr(log_L_gal, g_r).astype(int) 
        print(f"{quiescent.sum()} quiescent galaxies, {len(quiescent) - quiescent.sum()} star-forming galaxies")
        colors = quiescent
    else:
        colors = np.zeros(count, dtype=np.int8) 
    
    chi = np.zeros(count, dtype=np.int8) # TODO compute chi
    
    # Output files
    t1 = time.time()
    galprops= pd.DataFrame({
        'app_mag': app_mag, 
        'g_r': g_r, 
        'galaxy_type': galaxy_type, 
        'mxxl_halo_mass': mxxl_halo_mass,
        'z_assigned_flag': z_assigned_flag,
        'assigned_halo_mass': assigned_halo_mass,
        'z_obs': z_obs,
        'mxxl_halo_id': mxxl_halo_id,
        'assigned_halo_id': assigned_halo_id
    })
    galprops.to_pickle(outname_base + "_galprops.pkl")
    t2 = time.time()
    print(f"Galprops pickling took {t2-t1:.4f} seconds")
    
    write_dat_files(ra, dec, z_eff, log_L_gal, V_max, colors, chi, outname_base, FOOTPRINT_FRAC)
        
    return outname_base + ".dat", {'zmin': np.min(z_eff), 'zmax': np.max(z_eff), 'frac_area': FOOTPRINT_FRAC }



if __name__ == "__main__":
    main()