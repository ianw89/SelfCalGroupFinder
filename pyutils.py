import numpy as np
import math
import sys
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import astropy.coordinates as coord
from datetime import datetime
import time
import random

_cosmo = FlatLambdaCDM(H0=73, Om0=0.25, Ob0=0.045, Tcmb0=2.725, Neff=3.04) 



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


# The magic numbers in this right now are the ~0.5 NN same-halo points read-off of plots from the MXXL data
def use_nn(neighbor_z, neighbor_ang_dist, target_prob_obs, target_app_mag):
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
        implied_abs_mag = app_mag_to_abs_mag(target_app_mag, neighbor_z)
        if -24 < implied_abs_mag and implied_abs_mag < -14:
            return True

    return False


class RedshiftGuesser():

    def __init__(self, num_neighbors):

        self.rng = np.random.default_rng()

        self.nn_used = np.zeros(num_neighbors, dtype=np.int32)

        # These are from MXXL 19.5 cut
        # TODO use these abs_mag corrections on weights
        with open('bin/abs_mag_weight.npy', 'rb') as f:
            # pad the density array with a duplicate of the first value for < first bin, and 0 > last bin
            self.abs_mag_density = np.load(f)
            self.abs_mag_density = np.insert(self.abs_mag_density, [0,len(self.abs_mag_density)], [self.abs_mag_density[0],0])
            self.abs_mag_bins = np.load(f)

    def __del__(self):
        t = time.time()
        with open(f"bin/redshift_guesser_{t}.npy", 'wb') as f:
            np.save(f, self.nn_used, allow_pickle=False)

        print("NN used: ", self.nn_used)


    def score_neighbors(self, neighbors_z, neighbors_ang_dist, target_prob_obs, target_app_mag):
        """
        neighbors_z should be their spectroscopic redshifts
        neighbors_ang_dist should be in arsec
        target_prob_obs is the probability a fiber was assigned to the target (lost) galaxy
        target_app_mag is the DESI r-band apparent mag of the target (lost) galaxy
        """

        # TODO use z bins and p_obs
        # TODO consider how me using z from target messes up plots I got this from

        # this is rough line fitting important region of my plots of MXXL Truth data
        fsuccess = 1.0 - 0.45*np.log10(neighbors_ang_dist)
        # Ensure all neighbors have a nonzero base score (0.01)so other factors can weigh in
        MIN_ANGULAR_SCORE = 0.0001 # effectively sets downweighted far away neighbors are
        ang_dist_scores = np.max(np.stack((fsuccess, np.full(len(fsuccess), MIN_ANGULAR_SCORE)), axis=1), axis=1)

        # TODO grouping effects?

        # This will overweight this. Probably will never assign outside -23 to -19.5 Mag
        implied_abs_mag = app_mag_to_abs_mag(target_app_mag, neighbors_z)
        abs_mag_bin = np.digitize(implied_abs_mag, self.abs_mag_bins)
        abs_mag_scores = self.abs_mag_density[abs_mag_bin]

        final_scores = ang_dist_scores * abs_mag_scores

        return final_scores
    
    def choose_winner(self, neighbors_z, neighbors_ang_dist, target_prob_obs, target_app_mag):

        # TODO try with this on but do need some abs mag limit
        #k = 0
        #if use_nn(neighbors_z[k], neighbors_ang_dist[k], target_prob_obs):
        #    return 0

        scores = self.score_neighbors(neighbors_z, neighbors_ang_dist, target_prob_obs, target_app_mag)

        # randomly choose a winner uses the scores as a PDF to draw from
        #i = self.rng.choice(len(neighbors_z), p=scores)
        i = random.choices(range(len(neighbors_z)), weights=scores)
        self.nn_used[i] = self.nn_used[i] + 1

        return i

