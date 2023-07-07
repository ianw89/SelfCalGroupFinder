import numpy as np
from astropy.cosmology import FlatLambdaCDM
_cosmo = FlatLambdaCDM(H0=73, Om0=0.25, Ob0=0.045, Tcmb0=2.725, Neff=3.04) 

def app_mag_to_abs_mag(app_mag, z_obs):
    """
    Converts apparent mags to absolute mags using MXXL cosmology and provided observed redshifts.

    TODO this runs slow with astropy units.
    """
    with np.errstate(divide='ignore'): # will be NaN for blueshifted galaxies
        return app_mag - 5*(np.log10(_cosmo.luminosity_distance(z_obs).value * 1E6) - 1)


SOLAR_L_R_BAND = 4.65
def abs_mag_r_to_log_solar_L(arr):
    """
    Converts an absolute magnitude to log solar luminosities using the sun's r-band magnitude.

    This just comes from the definitions of magnitudes. The scalar 2.5 is 0.39794 dex.
    """
    return 0.39794 * (SOLAR_L_R_BAND - arr)


def get_max_observable_volume(abs_mags, z_obs, m_cut):
    """
    Calculate the max volume at which the galaxy could be seen in comoving coords.

    Takes in an array of absolute magnitudes and an array of redshifts.
    """

    # Use distance modulus
    d_l = (10 ** ((m_cut - abs_mags + 5) / 5)) / 1e6 # luminosity distance in Mpc
    d_cm = d_l / (1 + z_obs)
    v_max = (d_cm**3) * (4*np.pi/3) # in comoving Mpc^3
    frac_area = 0.35876178702 # 14800 / 41253 which is DESI BGS footprint (see Alex DESI BGS Incompleteness paper) 
    # TODO calculate exactly for MXXL
    # can see this from a simple ra dec plot
    return v_max * frac_area