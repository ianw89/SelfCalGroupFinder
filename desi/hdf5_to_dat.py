import numpy as np
import h5py
import sys
from pyutils import *
from enum import Enum
import pandas as pd

# Chooses 1 of the 2048 fiber assignment realizations with this bitstring and BITWORD as 'bitweight[0-31]
BITWORD = 'bitweight0'
BIT_CHOICE = 0
FIBER_ASSIGNED_SELECTOR = 2**BIT_CHOICE


def usage():
    print("Usage: python3 hdf5_to_dat.py [mode] [APP_MAG_CUT] [CATALOG_APP_MAG_CUT] [input_filename].hdf5 [output_filename]")
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
    Weight/
    'bitweight0',
    etc

     OUTPUT FORMAT FOR GROUP FINDER: space seperated values. Columns, in order, are:
    
     RA [degrees; float] - right ascension of the galaxy
     Dec [degrees; float] - decliration of the galaxy
     z [float] - redshift of the galaxy
     log L_gal [L_sol/h^-2; float] - r-band luminosity of the galaxy in solar units, assuming M_r,sol=4.65 and h=1.
     V_max [(Mpc/h)^3; float] - maximum volume at which each galaxy can be observed in SDSS. Taken from NYU VAGC
     color_flag [int]- 1=quiescent, 0=star-forming. This is based on the GMM cut of the Dn4000 data in Tinker 2020.
     chi [dimensionless] - THis is the normalized galaxy concentration. See details in both papers.

    OUTPUT FORMAT FOR GALPROPS: see bottom of this file
    """
    
    ################
    # ERROR CHECKING
    ################
    if len(sys.argv) != 6:
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
        print("\nMode SIMPLE")
    else:
        print("Unknown Mode")
        exit(4)

    APP_MAG_CUT = float(sys.argv[2])
    CATALOG_APP_MAG_CUT = float(sys.argv[3])
    FOOTPRINT_FRAC = 14800 / 41253

    print("Reading MXXL HDF5 data from ", sys.argv[4])
    infile = h5py.File(sys.argv[4], 'r')
    print(list(infile['Data']))

    outname_base = sys.argv[5]

    df = pd.DataFrame(data={
    'dec': infile['Data/dec'][:], 
    'ra': infile['Data/ra'][:],
    'z_obs': infile['Data/z_obs'][:],
    'app_mag': infile['Data/app_mag'][:],
    'abs_mag_mxxl': infile['Data/abs_mag'][:], # We aren't using these; computing ourselves.
    'g_r': infile['Data/g_r'][:],
    'galaxy_type': infile['Data/galaxy_type'][:],
    'mxxl_halo_mass': infile['Data/halo_mass'][:],
    'mxxl_halo_id': infile['Data/mxxl_id'][:],
    'observed': (infile['Weight/'+BITWORD][:] & FIBER_ASSIGNED_SELECTOR ).astype(bool)
    })

    orig_count = len(df)
    print(orig_count, "galaxies in HDF5 file")

    # filter arrays
    redshift_filter = df.z_obs > 0 # makes a filter array (True/False values)
    catalog_bright_filter = df.app_mag < CATALOG_APP_MAG_CUT 
    bright_filter = df.app_mag < APP_MAG_CUT # makes a filter array (True/False values)

    # Filter down DataFrame to the ones we want in the nearest-neighbor catalog 
    catalog_df = df[np.all([catalog_bright_filter, redshift_filter, df.observed], axis=0)].reset_index(drop=True)

    # Filter down DataFrame to the ones we want to actually process and keep
    keep = np.all([bright_filter, redshift_filter], axis=0)
    df = df[keep].reset_index(drop=True)
    df['assigned_halo_mass'] = np.copy(df.mxxl_halo_mass) # assigned for unobserved galaxies
    df['assigned_halo_id'] = np.copy(df.mxxl_halo_id) # assigned for unobserved galaxies
    
    unobserved = np.invert(df.observed)
    unobserved_df = df[unobserved]
    indexes_not_assigned = np.argwhere(unobserved)

    count = len(df)
    print(count, "galaxies left after apparent mag cut at {0}".format(APP_MAG_CUT))
    print(np.sum(df.observed), "galaxies were assigned a fiber")
    print(f"Catalog for nearest neighbor calculations is of size {len(catalog_df)}")

    with open('bin/prob_obs.npy', 'rb') as f:
       df['prob_obs'] = np.load(f)[keep]

    # z_eff: same as z_obs if a fiber was assigned and thus a real redshift measurement was made
    # otherwise, it is an assigned value.
    # nearest neighbor will find the nearest (measured) galaxy and use its redshift.
    df['z_eff'] = np.copy(df.z_obs)


    if mode == Mode.FIBER_ASSIGNED_ONLY.value:
        # Filter it all down to just the ones with fiber's assigned
        df = df[df.observed].reset_index(drop=True)
        assert np.all(df.observed)
        count = len(df)
    
    elif mode == Mode.NEAREST_NEIGHBOR.value:
        catalog = coord.SkyCoord(ra=catalog_df['ra']*u.degree, dec=catalog_df['dec']*u.degree, frame='icrs')
        to_match = coord.SkyCoord(ra=unobserved_df['ra']*u.degree, dec=unobserved_df['dec']*u.degree, frame='icrs')

        idx, d2d, d3d = coord.match_coordinates_sky(to_match, catalog, storekdtree=False)

        # i is the index of the full sized array that needed a NN z value
        # j is the index along the to_match list corresponding to that
        # idx are the indexes of the NN from the catalog
        assert len(indexes_not_assigned) == len(idx)

        print("Copying over NN properties... ", end='\r')
        j = 0
        for i in indexes_not_assigned:  
            df.z_eff[i] = catalog_df.z_obs[idx[j]]
            df.assigned_halo_mass[i] = catalog_df.mxxl_halo_mass[idx[j]]
            df.assigned_halo_id[i] = catalog_df.mxxl_halo_id[idx[j]]
            j = j + 1 
        print("Copying over NN properties... done")

    elif mode == Mode.FANCY.value:

        NUM_NEIGHBORS = 10
        with FancyRedshiftGuesser(NUM_NEIGHBORS) as scorer:

            catalog = coord.SkyCoord(ra=catalog_df['ra']*u.degree, dec=catalog_df['dec']*u.degree, frame='icrs')
            to_match = coord.SkyCoord(ra=unobserved_df['ra']*u.degree, dec=unobserved_df['dec']*u.degree, frame='icrs')
            
            neighbor_indexes = np.zeros(shape=(NUM_NEIGHBORS, len(to_match)), dtype=np.int32) # indexes point to CATALOG locations
            ang_distances = np.zeros(shape=(NUM_NEIGHBORS, len(to_match)))

            print(f"Finding nearest {NUM_NEIGHBORS} neighbors... ", end='\r')   
            for n in range(0, NUM_NEIGHBORS):
                idx, d2d, d3d = coord.match_coordinates_sky(to_match, catalog, nthneighbor=n+1, storekdtree='mxxl_fiber_assigned_tree')
                neighbor_indexes[n] = idx
                ang_distances[n] = d2d.to(u.arcsec).value
            print(f"Finding nearest {NUM_NEIGHBORS} neighbors... done")   


            print(f"Assinging missing redshifts... ")   
            # TODO don't loop?
            j = 0
            for i in indexes_not_assigned:    
                if j%10000==0:
                    print(f"{j}/{len(to_match)} complete", end='\r')

                neighbors = neighbor_indexes[:,j]
                neighbors_z = catalog_df.z_obs[neighbors]
                neighbors_ang_dist = ang_distances[:,j]
                my_prob_obs = df.prob_obs[i]
                my_app_mag = df.app_mag[i]

                winning_num = scorer.choose_winner(neighbors_z, neighbors_ang_dist, my_prob_obs, my_app_mag, df.z_obs[i])
                winner_index = neighbors[winning_num]

                df.z_eff[i] = catalog_df.z_obs[winner_index] 
                df.assigned_halo_mass[i] = catalog_df.mxxl_halo_mass[winner_index]
                df.assigned_halo_id[i] = catalog_df.mxxl_halo_id[winner_index]
                j = j + 1 

            print(f"{j}/{len(to_match)} complete")

        
    elif mode == Mode.SIMPLE.value:

        with SimpleRedshiftGuesser(df.app_mag[df.observed], df.z_obs[df.observed]) as scorer:

            catalog = coord.SkyCoord(ra=catalog_df['ra']*u.degree, dec=catalog_df['dec']*u.degree, frame='icrs')
            to_match = coord.SkyCoord(ra=unobserved_df['ra']*u.degree, dec=unobserved_df['dec']*u.degree, frame='icrs')
            
            neighbor_indexes, d2d, d3d = coord.match_coordinates_sky(to_match, catalog, storekdtree=False)
            ang_distances = d2d.to(u.arcsec).value

            print(f"Assigning missing redshifts... ")   
            j = 0 # j counts the number of unobserved galaxies in the catalog that have been assigned a redshift thus far
            for i in indexes_not_assigned:  # i is the index of the unobserved galaxy in the main arrays
                if j%10000==0:
                    print(f"{j}/{len(to_match)} complete", end='\r')

                chosen_z, isNN = scorer.choose_redshift(catalog_df.z_obs[neighbor_indexes[j]], ang_distances[j], df.prob_obs[i], df.app_mag[i], df.z_obs[i])

                df.z_eff[i] = chosen_z
                if isNN:
                    df.assigned_halo_mass[i] = catalog_df.mxxl_halo_mass[neighbor_indexes[j]]
                    df.assigned_halo_id[i] = catalog_df.mxxl_halo_id[neighbor_indexes[j]]
                else:
                    df.assigned_halo_mass[i] = -1 
                    df.assigned_halo_id[i] = -1
                j = j + 1 

        print(f"{j}/{len(to_match)} complete")
    

    df['abs_mag'] = app_mag_to_abs_mag(df.app_mag.to_numpy(), df.z_eff.to_numpy())
    df['abs_mag_k'] = k_correct(df.abs_mag.to_numpy(), df.z_eff.to_numpy(), df.g_r.to_numpy())

    # the luminosities sent to the group finder will be k-corrected to z=0.1
    df['log_L_gal'] = abs_mag_r_to_log_solar_L(df.abs_mag_k) 

    # the vmax should be calculated from un-k-corrected magnitudes
    df['V_max'] = get_max_observable_volume(df.abs_mag, df.z_eff, APP_MAG_CUT, FOOTPRINT_FRAC)

    """
    sanity_filter = V_max < np.max(V_max) * 0.9999
    df = df.loc[sanity_filter].reset_index(drop=True)
    """

    df['colors'] = np.zeros(count, dtype=np.int8) # TODO compute colors. Use color cut as per Alex's paper.
    df['chi'] = np.zeros(count, dtype=np.int8) # TODO compute chi
    
    # Output files
    galprops = np.column_stack([
        np.array(df.app_mag, dtype='str'), 
        np.array(df.g_r, dtype='str'), 
        np.array(df.galaxy_type, dtype='str'), 
        np.array(df.mxxl_halo_mass, dtype='str'),
        np.array(np.invert(df.observed), dtype='str'),
        np.array(df.assigned_halo_mass, dtype='str'),
        np.array(df.z_obs, dtype='str'),
        np.array(df.mxxl_halo_id, dtype='str'),
        np.array(df.assigned_halo_id, dtype='str')
        ])
    write_dat_files(df.ra, df.dec, df.z_eff, df.log_L_gal, df.V_max, df.colors, df.chi, outname_base, FOOTPRINT_FRAC, galprops)
        
if __name__ == "__main__":
    main()