import numpy as np
import h5py
import sys
from pyutils import *
from enum import Enum

# Chooses 1 of the 2048 fiber assignment realizations with this bitstring and BITWORD as 'bitweight[0-31]
BITWORD = 'bitweight0'
BIT_CHOICE = 0
FIBER_ASSIGNED_SELECTOR = 2**BIT_CHOICE


class Mode(Enum):
    ALL = 1 # include all galaxies
    FIBER_ASSIGNED_ONLY = 2 # include only galaxies that were assigned a fiber for FIBER_ASSIGNED_REALIZATION_BITSTRING
    NEAREST_NEIGHBOR = 3 # include all galaxies by assigned galaxies redshifts from their nearest neighbor
    FANCY = 4 
    SIMPLE = 5

def usage():
    print("Usage: python3 hdf5_to_dat.py [mode] [APP_MAG_CUT] [input_filename].hdf5 [output_filename]")
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

    OUTPUT FORMAT FOR GALPROPS: space separated values not needed by the group finder code. Columns, in order, are:

    app_mag
    g_r
    galaxy_type
    mxxl_halo_mass
    fiber_assigned_0
    """
    
    ################
    # ERROR CHECKING
    ################
    if len(sys.argv) != 5:
        print("Error 1")
        usage()
        exit(1)

    if int(sys.argv[1]) not in [member.value for member in Mode]:
        print("Error 2")
        usage()
        exit(2)

    if not sys.argv[3].endswith('.hdf5'):
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

    APP_MAG_CUT = float(sys.argv[2])

    print("Reading HDF5 data from ", sys.argv[3])
    infile = h5py.File(sys.argv[3], 'r')
    print(list(infile['Data']))

    outname_1 = sys.argv[4]+ ".dat"
    outname_2 = sys.argv[4] + "_galprops.dat"
    print("Output files will be {0} and {1}".format(outname_1, outname_2))

    # read everything we need into memory
    dec = infile['Data/dec'][:]
    ra = infile['Data/ra'][:]
    z_obs = infile['Data/z_obs'][:]
    app_mag = infile['Data/app_mag'][:]
    g_r = infile['Data/g_r'][:]
    galaxy_type = infile['Data/galaxy_type'][:]
    mxxl_halo_mass = infile['Data/halo_mass'][:]
    mxxl_halo_id = infile['Data/mxxl_id'][:]
    fiber_assigned_0 = infile['Weight/'+BITWORD][:] & FIBER_ASSIGNED_SELECTOR 
    fiber_assigned_0 = fiber_assigned_0.astype(bool)
    # TODO close file here to keep this pattern going

    orig_count = len(dec)
    print(orig_count, "galaxies in HDF5 file")

    # Filter it all down with the mag cut (and remove blueshifted ones)
    bright_filter = app_mag < APP_MAG_CUT # makes a filter array (True/False values) # TODO
    redshift_filter = z_obs > 0 # makes a filter array (True/False values)
    keep = np.all([bright_filter, redshift_filter], axis=0)
    dec = dec[keep]
    ra = ra[keep]
    z_obs = z_obs[keep]
    app_mag = app_mag[keep]
    g_r = g_r[keep]
    galaxy_type = galaxy_type[keep]
    mxxl_halo_mass = mxxl_halo_mass[keep]
    mxxl_halo_id = mxxl_halo_id[keep]
    assigned_halo_mass = np.copy(mxxl_halo_mass)
    assigned_halo_id = np.copy(mxxl_halo_id)
    fiber_assigned_0 = fiber_assigned_0[keep]
    fiber_not_assigned_0 = np.invert(fiber_assigned_0)
    indexes_not_assigned = np.argwhere(fiber_not_assigned_0)


    count = len(dec)
    print(count, "galaxies left after apparent mag cut at {0}".format(APP_MAG_CUT))
    print(np.sum(fiber_assigned_0), "galaxies were assigned a fiber")

    fiber_assigned_ra = ra[fiber_assigned_0]
    fiber_assigned_dec = dec[fiber_assigned_0]
    fiber_assigned_z_obs_catalog = z_obs[fiber_assigned_0]
    fiber_assigned_halo_mass_catalog = mxxl_halo_mass[fiber_assigned_0]
    fiber_assigned_halo_id_catalog = mxxl_halo_id[fiber_assigned_0]

    with open('bin/prob_obs.npy', 'rb') as f:
       prob_obs = np.load(f)
    prob_obs = prob_obs[keep]

    # z_eff: same as z_obs if a fiber was assigned and thus a real redshift measurement was made
    # otherwise, it is an assigned value.
    # nearest neighbor will find the nearest (measured) galaxy and use its redshift.
    z_eff = np.copy(z_obs)
    
    if mode == Mode.NEAREST_NEIGHBOR.value:
        # Astropy NN Search with kdtrees
        catalog = coord.SkyCoord(ra=fiber_assigned_ra*u.degree, dec=fiber_assigned_dec*u.degree, frame='icrs')
        to_match = coord.SkyCoord(ra=ra[fiber_not_assigned_0]*u.degree, dec=dec[fiber_not_assigned_0]*u.degree, frame='icrs')

        idx, d2d, d3d = coord.match_coordinates_sky(to_match, catalog, storekdtree=False)

        # i is the index of the full sized array that needed a NN z value
        # j is the index along the to_match list corresponding to that
        # idx are the indexes of the NN from the catalog
        assert len(indexes_not_assigned) == len(idx)

        print("Copying over NN properties... ", end='\r')
        j = 0
        for i in indexes_not_assigned:  
            z_eff[i] = fiber_assigned_z_obs_catalog[idx[j]]
            assigned_halo_mass[i] = fiber_assigned_halo_mass_catalog[idx[j]]
            assigned_halo_id[i] = fiber_assigned_halo_id_catalog[idx[j]]
            j = j + 1 
        print("Copying over NN properties... done")

    elif mode == Mode.FANCY.value:

        NUM_NEIGHBORS = 10
        with FancyRedshiftGuesser(NUM_NEIGHBORS) as scorer:

            catalog = coord.SkyCoord(ra=fiber_assigned_ra*u.degree, dec=fiber_assigned_dec*u.degree, frame='icrs')
            to_match = coord.SkyCoord(ra=ra[fiber_not_assigned_0]*u.degree, dec=dec[fiber_not_assigned_0]*u.degree, frame='icrs')
            
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
                neighbors_z = fiber_assigned_z_obs_catalog[neighbors]
                neighbors_ang_dist = ang_distances[:,j]
                my_prob_obs = prob_obs[i]
                my_app_mag = app_mag[i]

                winning_num = scorer.choose_winner(neighbors_z, neighbors_ang_dist, my_prob_obs, my_app_mag, z_obs[i])
                winner_index = neighbors[winning_num]

                z_eff[i] = fiber_assigned_z_obs_catalog[winner_index] 
                assigned_halo_mass[i] = fiber_assigned_halo_mass_catalog[winner_index]
                assigned_halo_id[i] = fiber_assigned_halo_id_catalog[winner_index]
                j = j + 1 

            print(f"{j}/{len(to_match)} complete")

        
    elif mode == Mode.SIMPLE.value:

        with SimpleRedshiftGuesser(app_mag[fiber_assigned_0], z_obs[fiber_assigned_0]) as scorer:

            catalog = coord.SkyCoord(ra=fiber_assigned_ra*u.degree, dec=fiber_assigned_dec*u.degree, frame='icrs')
            to_match = coord.SkyCoord(ra=ra[fiber_not_assigned_0]*u.degree, dec=dec[fiber_not_assigned_0]*u.degree, frame='icrs')
            
            neighbor_indexes, d2d, d3d = coord.match_coordinates_sky(to_match, catalog, storekdtree=False)
            ang_distances = d2d.to(u.arcsec).value

            print(f"Assinging missing redshifts... ")   
            j = 0
            for i in indexes_not_assigned:    
                if j%10000==0:
                    print(f"{j}/{len(to_match)} complete", end='\r')

                chosen_z, isNN = scorer.choose_redshift(fiber_assigned_z_obs_catalog[neighbor_indexes[j]], ang_distances[j], prob_obs[i], app_mag[i], z_obs[i])

                z_eff[i] = chosen_z
                if isNN:
                    assigned_halo_mass[i] = fiber_assigned_halo_mass_catalog[neighbor_indexes[j]]
                    assigned_halo_id[i] = fiber_assigned_halo_id_catalog[neighbor_indexes[j]]
                else:
                    assigned_halo_mass[i] = -1 
                    assigned_halo_id[i] = -1
                j = j + 1 

        print(f"{j}/{len(to_match)} complete")
        
    #abs_mag = infile['Data/abs_mag'][:] # We aren't using these; computing ourselves. 
    # TODO Mine are missing k-corrections
    my_abs_mag = app_mag_to_abs_mag(app_mag, z_eff)
    log_L_gal = abs_mag_r_to_log_solar_L(my_abs_mag)

    V_max = get_max_observable_volume(my_abs_mag, z_eff, APP_MAG_CUT)

    colors = np.zeros(count, dtype=np.int8) # TODO compute colors. Use color cut as per Alex's paper.
    chi = np.zeros(count, dtype=np.int8) # TODO compute chi


    # To output turn the data into rows, 1 per galaxy (for each of the two files) 
    # and then build up a large string to write in one go.
    
    # Note this copies the data from what was read in from the file
    # TODO ID's are being written as float not int for some reason, fix

    print("Building output file string... ", end='\r')
    output_1 = np.column_stack((ra, dec, z_eff, log_L_gal, V_max, colors, chi))
    output_2 = np.column_stack((app_mag, g_r, galaxy_type, mxxl_halo_mass, fiber_assigned_0, assigned_halo_mass, z_obs, mxxl_halo_id, assigned_halo_id))
    lines_1 = []
    lines_2 = []

    for i in range(0, count):
        if mode == Mode.FIBER_ASSIGNED_ONLY.value:
            if fiber_assigned_0[i]:
                lines_1.append(' '.join(map(str, output_1[i])))
                lines_2.append(' '.join(map(str, output_2[i])))
            else:
                pass
        else:
            lines_1.append(' '.join(map(str, output_1[i])))
            lines_2.append(' '.join(map(str, output_2[i])))

    outstr_1 = "\n".join(lines_1)
    outstr_2 = "\n".join(lines_2)    
    print("Building output file string... done")

    print("Writing output files... ",end='\r')
    open(outname_1, 'w').write(outstr_1)
    open(outname_2, 'w').write(outstr_2)
    print("Writing output files... done")

        


        
if __name__ == "__main__":
    main()