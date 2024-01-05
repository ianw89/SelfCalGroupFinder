import numpy as np
import sys
from pyutils import *
from astropy.table import Table


def usage():
    print("Usage: python3 desi_fits_to_dat.py [mode] [APP_MAG_CUT] [CATALOG_APP_MAG_CUT] [input_filename].hdf5 [output_filename]")
    print("  Mode is 1 for ALL, 2 for FIBER_ASSIGNED_ONLY, and 3 for NEAREST_NEIGHBOR, 4 for FANCY, 5 for SIMPLE ")
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

    OUTPUT FORMAT FOR GALPROPS: space separated values not needed by the group finder code. Columns, in order, are:

    app_mag
    target_id
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
        print("\nMode FIBER_ASSIGNED_ONLY")
        exit(2)
    elif mode == Mode.NEAREST_NEIGHBOR.value:
        print("\nMode NEAREST_NEIGHBOR")
        exit(2)
    elif mode == Mode.FANCY.value:
        print("\nMode FANCY")
        exit(2)
    elif mode == Mode.SIMPLE.value:
        print("\nMode SIMPLE")
        exit(2)

    APP_MAG_CUT = float(sys.argv[2])
    CATGALOG_APP_MAG_CUT = float(sys.argv[3])

    print("Reading FITS data from ", sys.argv[4])
    u_table = Table.read(sys.argv[4], format='fits')

    outname_1 = sys.argv[5]+ ".dat"
    outname_2 = sys.argv[5] + "_galprops.dat"
    print("Output files will be {0} and {1}".format(outname_1, outname_2))

    obj_type = u_table['SPECTYPE']
    dec = u_table['DEC']
    ra = u_table['RA']
    z_obs = u_table['Z_not4clus']
    target_id = u_table['TARGETID']
    flux_r = u_table['FLUX_R']
    app_mag = get_app_mag(u_table['FLUX_R'])
    p_obs = u_table['PROB_OBS']

    orig_count = len(dec)
    print(orig_count, "objects in FITS file")

    # Make filter array (True/False values)
    galaxy_filter = obj_type == 'GALAXY'
    app_mag_filter = app_mag < APP_MAG_CUT
    redshift_filter = z_obs > 0 
    redshift_hi_filter = z_obs < 0.8 # TODO doesn't fix the issue...
    #three_pass_filter = TODO
    keep = np.all([galaxy_filter, app_mag_filter, redshift_filter, redshift_hi_filter], axis=0)

    dec = dec[keep]
    ra = ra[keep]
    z_obs = z_obs[keep]
    target_id = target_id[keep]
    app_mag = app_mag[keep]
    p_obs = p_obs[keep]


    count = len(dec)
    print(count, "galaxies left after apparent mag cut at {0}".format(APP_MAG_CUT))
    print(min(z_obs), max(z_obs), "min and max redshifts")

    z_eff = np.copy(z_obs)

    # TODO put fancy logic to fill in z_eff here



    # TODO Missing k-corrections
    abs_mag = app_mag_to_abs_mag(app_mag, z_eff)
    log_L_gal = abs_mag_r_to_log_solar_L(abs_mag)

    V_max = get_max_observable_volume(abs_mag, z_eff, APP_MAG_CUT, ra, dec)

    colors = np.zeros(count, dtype=np.int8) # TODO compute colors. Use color cut as per Alex's paper.
    chi = np.zeros(count, dtype=np.int8) # TODO compute chi

    # TODO g_r

    # To output turn the data into rows, 1 per galaxy (for each of the two files) 
    # and then build up a large string to write in one go.
    
    # Note this copies the data from what was read in from the file
    # TODO ID's are being written as float not int for some reason, fix

    print("Building output file string... ", end='\r')
    output_1 = np.column_stack((ra, dec, z_eff, log_L_gal, V_max, colors, chi))
    output_2 = np.column_stack((app_mag, target_id))
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