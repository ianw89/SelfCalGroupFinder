import numpy as np
import h5py
import sys
from pyutils import *
from enum import Enum
from astropy.table import Table

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
    print("Usage: python3 uchuu_to_dat.py [mode] [APP_MAG_CUT] [CATALOG_APP_MAG_CUT] [input_filename].fits [output_filename]")
    print("  Mode is 1 for ALL, 2 for FIBER_ASSIGNED_ONLY, and 3 for NEAREST_NEIGHBOR, 4 for FANCY, 5 for SIMPLE ")
    print("  Will generate [output_filename].dat for use with kdGroupFinder and [output_filename]_galprops.dat with additional galaxy properties.")
    print("  These two files will have galaxies indexed in the same way (line-by-line matched).")


def main():
    """
    INPUT FORMAT: FITS FILE 

    [('R_MAG_APP', '>f4'), ('R_MAG_ABS', '>f4'), ('G_R_REST', '>f4'), ('G_R_OBS', '>f4'), ('DEC', '>f8'), ('HALO_MASS', '>f4'), ('CEN', '>i4'), ('RES', '>i4'), ('RA', '>f8'), ('Z_COSMO', '>f4'), ('Z', '>f4'), ('STATUS', '>i4'), ('FIRST_ACC_SCALE', '>f4'), ('M_ACC', '>f4'), ('M_VIR_ALL', '>f4'), ('R_VIR', '>f4'), ('V_PEAK', '>f4'), ('R_S', '>f4'), ('V_RMS', '>f4'), ('NGC', '>f4'), ('SGC', '>f4'), ('HALO_ID', '>i8'), ('PID', '>i8')]))


     OUTPUT FORMAT FOR GROUP FINDER: space seperated values. Columns, in order, are:
    
     RA [degrees; float] - right ascension of the galaxy
     Dec [degrees; float] - decliration of the galaxy
     z [float] - redshift of the galaxy
     log L_gal [L_sol/h^-2; float] - r-band luminosity of the galaxy in solar units, assuming M_r,sol=4.65 and h=1.
     V_max [(Mpc/h)^3; float] - maximum volume at which each galaxy can be observed in SDSS. Taken from NYU VAGC
     color_flag [int]- 1=quiescent, 0=star-forming. This is based on the GMM cut of the Dn4000 data in Tinker 2020.
     chi [dimensionless] - THis is the normalized galaxy concentration. See details in both papers.

    OUTPUT FORMAT FOR GALPROPS: space separated values not needed by the group finder code. Columns, in order, are:

    app_mag, g_r, central (boolean value), uchuu_halo_mass, uchuu_halo_id
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

    if not sys.argv[4].endswith('.fits'):
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
        print("\nMode FIBER_ASSIGNED_ONLY not possible on UCHUU (no fiber assignment data)")
        exit(2)
    elif mode == Mode.NEAREST_NEIGHBOR.value:
        print("\nMode NEAREST_NEIGHBOR not possible on UCHUU (no fiber assignment data)")
        exit(2)
    elif mode == Mode.FANCY.value:
        print("\nMode FANCY not possible on UCHUU (no fiber assignment data)")
        exit(2)
    elif mode == Mode.SIMPLE.value:
        print("\nMode SIMPLE not possible on UCHUU (no fiber assignment data)")
        exit(2)

    APP_MAG_CUT = float(sys.argv[2])
    CATALOG_APP_MAG_CUT = float(sys.argv[3])

    print("Reading FITS data from ", sys.argv[4])
    u_table = Table.read(sys.argv[4], format='fits')

    outname_1 = sys.argv[5]+ ".dat"
    outname_2 = sys.argv[5] + "_galprops.dat"
    print("Output files will be {0} and {1}".format(outname_1, outname_2))

    # read everything we need into memory
    dec = u_table['DEC']
    ra = u_table['RA']
    z_obs = u_table['Z']
    app_mag = u_table['R_MAG_APP']
    g_r = u_table['G_R_REST'] # TODO before using ensure it should be rest and not observed
    central = u_table['CEN']
    uchuu_halo_mass = u_table['HALO_MASS']
    uchuu_halo_id = u_table['HALO_ID']

    # set up filters on the galaxies
    bright_filter = app_mag < APP_MAG_CUT 
    redshift_filter = z_obs > 0 
    keep = np.all([bright_filter, redshift_filter], axis=0)

    orig_count = len(dec)
    print(orig_count, "galaxies in FITS file")

    # Filter down inputs we want to actually process and keep
    dec = dec[keep]
    ra = ra[keep]
    z_obs = z_obs[keep]
    app_mag = app_mag[keep]
    g_r = g_r[keep]
    central = central[keep]
    uchuu_halo_mass = uchuu_halo_mass[keep]
    uchuu_halo_id = uchuu_halo_id[keep]

    count = len(dec)
    print(count, "galaxies left after apparent mag cut at {0}".format(APP_MAG_CUT))

    #with open('bin/prob_obs.npy', 'rb') as f:
    #   prob_obs = np.load(f)
    #prob_obs = prob_obs[keep]

    # z_eff: same as z_obs if a fiber was assigned and thus a real redshift measurement was made
    # otherwise, it is an assigned value.
    # nearest neighbor will find the nearest (measured) galaxy and use its redshift.
    #z_eff = np.copy(z_obs)
    z_eff = z_obs # TODO go back to copying if UCHUU gets fiber assignment and we run other modes!
        
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
    output_2 = np.column_stack((app_mag, g_r, central, uchuu_halo_mass, uchuu_halo_id))
    lines_1 = []
    lines_2 = []

    for i in range(0, count):
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