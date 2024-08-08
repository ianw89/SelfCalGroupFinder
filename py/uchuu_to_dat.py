import numpy as np
import h5py
import sys
from SelfCalGroupFinder.py.pyutils import *
from astropy.table import Table


def usage():
    print("Usage: python3 uchuu_to_dat.py [mode] [APP_MAG_CUT] [CATALOG_APP_MAG_CUT] [input_filename].fits [output_filename] [COLORS_ON]")
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


    APP_MAG_CUT = float(sys.argv[2])
    CATALOG_APP_MAG_CUT = float(sys.argv[3])
    
    COLORS_ON = sys.argv[6]
    if COLORS_ON:
        print("\nColors not supported yet. Exiting...")
        exit(5)

    fpath = sys.argv[4]
    outname_base = sys.argv[5]

    pre_process_uchuu(fpath, mode, outname_base, APP_MAG_CUT, CATALOG_APP_MAG_CUT, COLORS_ON)

def pre_process_uchuu(in_filepath: str, mode: int, outname_base: str, app_mag_cut: float, catalog_app_mag_cut: float, colors_on: bool):

    FOOTPRINT_FRAC = 1.0 # UCHUU is whole sky

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

    print("Reading FITS data from ", in_filepath)
    u_table = Table.read(in_filepath, format='fits')


    # read everything we need into memory
    dec = u_table['DEC']
    ra = u_table['RA']
    z_obs = u_table['Z']
    app_mag = u_table['R_MAG_APP']
    abs_mag = u_table['R_MAG_ABS']
    #g_r = u_table['G_R_REST'] 
    g_r = u_table['G_R_OBS']
    central = u_table['CEN']
    uchuu_halo_mass = u_table['HALO_MASS']
    uchuu_halo_id = u_table['HALO_ID']

    # set up filters on the galaxies
    bright_filter = app_mag < app_mag_cut 
    redshift_filter = z_obs > 0 
    keep = np.all([bright_filter, redshift_filter], axis=0)

    orig_count = len(dec)
    print(orig_count, "galaxies in FITS file")

    # Filter down inputs we want to actually process and keep
    dec = dec[keep]
    ra = ra[keep]
    z_obs = z_obs[keep]
    app_mag = app_mag[keep]
    abs_mag = abs_mag[keep]
    g_r = g_r[keep]
    central = central[keep]
    uchuu_halo_mass = uchuu_halo_mass[keep]
    uchuu_halo_id = uchuu_halo_id[keep]

    count = len(dec)
    print(count, "galaxies left after apparent mag cut at {0}".format(app_mag_cut))

    #with open('bin/prob_obs.npy', 'rb') as f:
    #   prob_obs = np.load(f)
    #prob_obs = prob_obs[keep]

    # z_eff: same as z_obs if a fiber was assigned and thus a real redshift measurement was made
    # otherwise, it is an assigned value.
    # nearest neighbor will find the nearest (measured) galaxy and use its redshift.
    #z_eff = np.copy(z_obs)
    z_eff = z_obs # TODO go back to copying if UCHUU gets fiber assignment and we run other modes!
          
    abs_mag_me = app_mag_to_abs_mag(app_mag, z_eff)
    abs_mag_me_k = k_correct(abs_mag_me, z_eff, g_r)

    # the luminosities sent to the group finder will be k-corrected to z=0.1
    log_L_gal = abs_mag_r_to_log_solar_L(abs_mag_me_k) 

    # the vmax should be calculated from un-k-corrected magnitudes
    V_max = get_max_observable_volume(abs_mag_me, z_eff, app_mag_cut, FOOTPRINT_FRAC)

    colors = np.zeros(count, dtype=np.int8) # TODO compute colors. Use color cut as per Alex's paper.
    chi = np.zeros(count, dtype=np.int8) # TODO compute chi

    # Output files
    t1 = time.time()
    galprops= pd.DataFrame({
        'app_mag': app_mag, 
        'g_r': g_r, 
        'central': central, 
        'uchuu_halo_mass': uchuu_halo_mass,
        'uchuu_halo_id': uchuu_halo_id,
    })
    galprops.to_pickle(outname_base + "_galprops.pkl")
    t2 = time.time()
    print(f"Galprops pickling took {t2-t1:.4f} seconds")
    
    write_dat_files(ra, dec, z_eff, log_L_gal, V_max, colors, chi, outname_base, FOOTPRINT_FRAC, galprops)

    return outname_base + ".dat", {'zmin': np.min(z_eff), 'zmax': np.max(z_eff), 'frac_area': FOOTPRINT_FRAC }


if __name__ == "__main__":
    main()