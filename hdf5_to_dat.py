import numpy as np
import h5py
import sys


SOLAR_L_R_BAND = 4.65
def abs_mag_r_to_log_solar_L(arr):
    """Converts an absolute magnitude to log solar luminosities using the sun's r-band magnitude."""
    # This just comes from the definitions of magnitudes. 2.5 is 0.39794 dex
    return 0.39794 * (SOLAR_L_R_BAND - arr)


# TODO  I'm using 19.5 cut the sample out to 20 above, not 19.5.

def get_max_observable_volume(abs_mags, z_obs, m_cut=19.5):
    """
    Calculate the max volume at which the galaxy could be seen in comoving coords.

    Takes in an array of absolute magnitudes and an array of redshifts.
    """

    # Use distance modulus
    d_l = (10 ** ((m_cut - abs_mags + 5) / 5)) / 1e6 # luminosity distance in Mpc
    d_cm = d_l / (1 + z_obs)
    v_max = (d_cm**3) * (4*np.pi/3) # in comoving Mpc^3
    return v_max


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

    """

    if (len(sys.argv) > 1 and sys.argv[1].endswith('.hdf5')):

        outname = sys.argv[1][0:-5] + ".dat"
        print(outname)

        input = h5py.File(sys.argv[1], 'r')

        print(list(input['Data']))


        dec = input['Data/dec'][:]
        ra = input['Data/ra'][:]
        z = input['Data/z_obs'][:]

        count = len(dec)
        print(count, "galaxies")

        abs_mag = input['Data/abs_mag'][:]
        log_L_gal = abs_mag_r_to_log_solar_L(abs_mag)

        V_max = get_max_observable_volume(abs_mag, z)


        colors = np.zeros(count) # TODO compute colors
        chi = np.zeros(count) # TODO compute chi

        output = np.column_stack((ra, dec, z, log_L_gal, V_max, colors, chi))

        lines = []

        for row in output:
            lines.append(' '.join(map(str, row)))

        # Oddly doing it this way and not preloading all the ra/decs changes the numbers by a bit
        # It also takes double as long to do it this way.
        #for i in range(0, count):
        #    lines.append(' '.join(map(str, (ra[i], dec[i], z[i], log_L_gal[i], V_max[i], 0, 0 ) )))

        outstr = "\n".join(lines)

        open(outname, 'w').write(outstr)

        

    else:
        print("Usage: hdf5_to_dat.py [input_filename.hdf5]")


if __name__ == "__main__":
    main()