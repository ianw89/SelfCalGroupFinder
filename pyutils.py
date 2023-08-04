import numpy as np
import math
import sys
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import astropy.coordinates as coord

_cosmo = FlatLambdaCDM(H0=73, Om0=0.25, Ob0=0.045, Tcmb0=2.725, Neff=3.04) 

# The magic numbers in this right now are the ~0.5 NN same-halo points read-off of plots from the MXXL data
def use_nn(neighbor_z, neighbor_ang_dist, target_prob_obs):
    adjust = 0
    if target_prob_obs < 0.60:
        adjust = 5

    if neighbor_z < 0.1:
        threshold = 10
    elif neighbor_z < 0.2:
        threshold = 20
    elif neighbor_z < 0.3:
        threshold = 18
    else: # z > 0.3
        threshold = 8
    
    if neighbor_ang_dist < (threshold + adjust):
        return True

    return False


def z_to_ldist(z):
    """
    Gets the luminosity distance of the provided redshifts in Mpc using MXXL cosmology.
    """
    with np.errstate(divide='ignore'): # will be NaN for blueshifted galaxies
        return _cosmo.luminosity_distance(z).value
    
def app_mag_to_abs_mag(app_mag, z_obs):
    """
    Converts apparent mags to absolute mags using MXXL cosmology and provided observed redshifts.

    TODO this runs slowish with astropy units, workaround
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
    # TODO calculate exactly for MXXL? But should be this because footprints match
    # can see this from a simple ra dec plot
    return v_max * frac_area

class NearestNeighbor():

    def __init__(self, ra_arr, dec_arr, z_arr):
        self.ra_angles = coord.Angle(ra_arr * u.degree)
        self.ra_angles = self.ra_angles.wrap_at(360 * u.degree) # 0 to 2 pi
        self.ra_angles = self.ra_angles.radian
        #self.ra_sin = np.sin(self.ra_angles)
        #self.ra_cos = np.cos(self.ra_angles)

        self.dec_angles = coord.Angle(dec_arr * u.degree)
        self.dec_angles = self.dec_angles.radian
        self.dec_sin = np.sin(self.dec_angles)
        self.dec_cos = np.cos(self.dec_angles)

        self.z_arr = z_arr


    def get_closest_index(self, ra, dec):
        """
        Finds the nearest neighbor (in term of angular distance) to the provided coordinate, and returns the z value associated with nearest point.

        ra, dec must be in radians.
        """
        
        # Takes ~2 days for 1M galaxies with 100,000 needing a nn assignment

        # This could be made faster if we stored the positions in a more intelligent data structure 
        # so we didn't have to evaluate this for all of them. 
        # Also if we only had nearby ones to check then small angle version of this would also work.
    
        # Compute the angular distance in radians on our database of things to check in one vectorized go
        
        dist = np.arccos( np.sin(dec)*self.dec_sin + np.cos(dec)*self.dec_cos*np.cos(ra - self.ra_angles) )
        index = np.argmin(dist)
        

         
        # This below is def wrong
        
        # TODO maybe could do this because minimizing a monotonicly decreasing function 
        # is the same as maximizing its argument
        # TODO spherical polar vs other ra/dec 
        #measure = np.sin(dec)*self.dec_sin + np.cos(dec)*self.dec_cos*np.cos(ra - self.ra_angles)
        #index = np.argmax(measure)        
        
        return index