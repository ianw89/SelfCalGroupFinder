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
        exit(2)
    elif mode == Mode.FIBER_ASSIGNED_ONLY.value:
        print("\nMode FIBER_ASSIGNED_ONLY")
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
    Z_MIN = 0.001
    Z_MAX = 0.8

    print("Reading FITS data from ", sys.argv[4])
    # Unobserved galaxies have masked rows in appropriate columns of the table
    u_table = Table.read(sys.argv[4], format='fits')

    outname_1 = sys.argv[5]+ ".dat"
    outname_2 = sys.argv[5] + "_galprops.dat"
    print("Output files will be {0} and {1}".format(outname_1, outname_2))

    # astropy's Table used masked arrays, so we have to use .data.data to get the actual data
    # The masked rows are unobserved targets
    obj_type = u_table['SPECTYPE'].data.data
    dec = u_table['DEC']
    ra = u_table['RA']
    z_obs = u_table['Z_not4clus'].data.data
    target_id = u_table['TARGETID']
    flux_r = u_table['FLUX_R']
    app_mag = get_app_mag(u_table['FLUX_R'])
    p_obs = u_table['PROB_OBS']
    unobserved = u_table['ZWARN'] == 999999
    deltachi2 = u_table['DELTACHI2'].data.data  


    orig_count = len(dec)
    print(orig_count, "objects in FITS file")

    # If an observation was made, some automated system will evaluate the spectra and auto classify as GALAXY, QSO, STAR (or nothing?)
    # NTILE tracks how many DESI pointings could observe the target
    # null values (masked rows) are unobserved targets; not all columns are masked though
    # SPECTYPE gives the spectral type; it is empty for unoobserved targets

    # Make filter array (True/False values)
    three_pass_filter = u_table['NTILE'] >= 3 # 3pass coverage. Some have 4, not sure why
    galaxy_filter = np.logical_or(obj_type == b'GALAXY', obj_type == b'')
    app_mag_filter = app_mag < APP_MAG_CUT
    redshift_filter = z_obs > Z_MIN
    redshift_hi_filter = z_obs < Z_MAX
    deltachi2_filter = deltachi2 > 40

    if mode == Mode.FIBER_ASSIGNED_ONLY.value:
        keep = np.all([three_pass_filter, galaxy_filter, app_mag_filter, redshift_filter, redshift_hi_filter, deltachi2_filter], axis=0)

        obj_type = obj_type[keep]
        dec = dec[keep]
        ra = ra[keep]
        z_obs = z_obs[keep]
        target_id = target_id[keep]
        flux_r = flux_r[keep]
        app_mag = app_mag[keep]
        p_obs = p_obs[keep]
        observed = observed[keep]
        deltachi2 = deltachi2[keep]


    count = len(dec)
    print(count, "galaxies left after filters.")
    print(f'{unobserved.sum() } remaining galaxies are unobserved')
    print(f'{100*unobserved.sum() / len(unobserved) :.1f}% of remaininggalaxies are unobserved')
    #print(f'Min z: {min(z_obs):f}, Max z: {max(z_obs):f}')

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

    # Print the output files built up from ra, dec, z_eff, log_L_gal, V_max, colors, chi


    print("Building output file string... ", end='\r')
   #output_1 = np.column_stack((ra, dec, z_eff, log_L_gal, V_max, colors, chi))
    #output_2 = np.column_stack((app_mag, target_id))
    lines_1 = []
    lines_2 = []

    for i in range(0, count):
        lines_1.append(f'{ra[i]:f} {dec[i]:f} {z_eff[i]:f} {log_L_gal[i]:f} {V_max[i]:f} {colors[i]} {chi[i]}')
        lines_2.append(f'{app_mag[i]:f} {target_id[i]:f}')
        
        #lines_1.append(' '.join(map(str, output_1[i])))
        #lines_2.append(' '.join(map(str, output_2[i])))

    outstr_1 = "\n".join(lines_1)
    outstr_2 = "\n".join(lines_2)    
    print("Building output file string... done")

    print("Writing output files... ",end='\r')
    open(outname_1, 'w').write(outstr_1)
    open(outname_2, 'w').write(outstr_2)
    print("Writing output files... done")

        
if __name__ == "__main__":
    main()