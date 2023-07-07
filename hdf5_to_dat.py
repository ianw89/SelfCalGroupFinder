import numpy as np
import h5py
import sys
from pyutils import *
from enum import Enum

APP_MAG_CUT = 19.5

FIBER_ASSIGNED_REALIZATION_BITSTRING = 1
 
class Mode(Enum):
    ALL = 1 # include all galaxies
    FIBER_ASSIGNED_ONLY = 2 # include only galaxies that were assigned a fiber for FIBER_ASSIGNED_REALIZATION_BITSTRING
    NEAREST_NEIGHBOR = 3 # include all galaxies by assigned galaxies redshifts from their nearest neighbor

def usage():
    print("Usage: python3 hdf5_to_dat.py [mode] [input_filename].hdf5 [output_filename]")
    print("  Mode is 1 for ALL, 2 for FIBER_ASSIGNED_ONLY, and 3 for NEAREST_NEIGHBOR")
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
    if len(sys.argv) != 4:
        print("Error 1")
        usage()
        exit(1)

    if int(sys.argv[1]) not in [member.value for member in Mode]:
        print("Error 2")
        usage()
        exit(2)

    if not sys.argv[2].endswith('.hdf5'):
        print("Error 3")
        usage()
        exit(3)

    ################
    # MAIN CODE
    ################

    mode = int(sys.argv[1])
    if mode == Mode.ALL.value:
        print("Mode ALL")
    elif mode == Mode.FIBER_ASSIGNED_ONLY.value:
        print("Mode FIBER_ASSIGNED_ONLY")
    elif mode == Mode.NEAREST_NEIGHBOR.value:
        print("Mode NEAREST_NEIGHBOR")

    print("Reading HDF5 data from ", sys.argv[2])
    input = h5py.File(sys.argv[2], 'r')
    print(list(input['Data']))

    outname_1 = sys.argv[3]+ ".dat"
    outname_2 = sys.argv[3] + "_galprops.dat"
    print("Output files will be {0} and {1}".format(outname_1, outname_2))

    dec = input['Data/dec'][:]
    ra = input['Data/ra'][:]
    z = input['Data/z_obs'][:]

    count = len(dec)
    print(count, "galaxies in HDF5 file")

    #abs_mag = input['Data/abs_mag'][:] # We aren't using these; computing ourselves. Not sure what the difference is
    app_mag = input['Data/app_mag'][:]
    my_abs_mag = app_mag_to_abs_mag(app_mag, z)
    log_L_gal = abs_mag_r_to_log_solar_L(my_abs_mag)

    V_max = get_max_observable_volume(my_abs_mag, z, APP_MAG_CUT)

    colors = np.zeros(count) # TODO compute colors. Use color cut as per Alex's paper.
    chi = np.zeros(count) # TODO compute chi

    g_r = input['Data/g_r'][:]
    galaxy_type = input['Data/galaxy_type'][:]
    mxxl_halo_mass = input['Data/halo_mass'][:]

    # choose 1 of the 2048 fiber assignment realizations with this bitstring
    fiber_assigned_0 = input['Weight/bitweight0'][:] & FIBER_ASSIGNED_REALIZATION_BITSTRING 
    print(np.sum(fiber_assigned_0 == 1), "galaxies were assigned a fiber")
    print(np.sum(fiber_assigned_0 == 0), "galaxies were NOT assigned a fiber")


    # To output turn the data into rows, 1 per galaxy (for each of the two files) 
    # and then build up a large string to write in one go.

    output_1 = np.column_stack((ra, dec, z, log_L_gal, V_max, colors, chi))
    output_2 = np.column_stack((app_mag, g_r, galaxy_type, mxxl_halo_mass, fiber_assigned_0))
    lines_1 = []
    lines_2 = []

    for i in range(0, count):
        # Drop galaxies with apparent mags over the limit we're using 
        # Also drop ones with NaN from a negative z value that propagated through
        if ( (output_2[i][0] <= APP_MAG_CUT) and not np.isnan(output_1[i][4]) ): 

            if mode == Mode.ALL.value or (mode == Mode.FIBER_ASSIGNED_ONLY.value and output_2[i][4]):
                lines_1.append(' '.join(map(str, output_1[i])))
                lines_2.append(' '.join(map(str, output_2[i])))
            elif mode == Mode.NEAREST_NEIGHBOR:
                print("NEAREST NEIGHBOR NOT IMPLEMENTED YET")
                exit(4)
            else:
                pass

    print(len(lines_1), "galaxies used")
    # Oddly doing it this way and not preloading all the ra/decs changes the numbers by a bit
    # It also takes double as long to do it this way.
    #for i in range(0, count):
    #    lines.append(' '.join(map(str, (ra[i], dec[i], z[i], log_L_gal[i], V_max[i], 0, 0 ) )))

    outstr_1 = "\n".join(lines_1)
    outstr_2 = "\n".join(lines_2)

    open(outname_1, 'w').write(outstr_1)
    open(outname_2, 'w').write(outstr_2)

        


if __name__ == "__main__":
    main()