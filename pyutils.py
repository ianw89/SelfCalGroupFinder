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

SIM_Z_THRESH = 0.003
# TODO smarter
def close_enough(target_z, z_arr, threshold=SIM_Z_THRESH):
    return np.abs(z_arr - target_z) < threshold

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

    def __init__(self, num_neighbors, debug=False):
        print("Initializing v6 of RedshiftGuesser")
        self.debug = debug
        self.rng = np.random.default_rng()

        self.nn_used = np.zeros(num_neighbors, dtype=np.int32)
        self.nn_correct = np.zeros(num_neighbors, dtype=np.int32)
        self.quick_nn = 0
        self.quick_correct = 0

        # These are from MXXL 19.5 cut
        with open('bin/abs_mag_weight.npy', 'rb') as f:
            # pad the density array with a duplicate of the first value for < first bin, and 0 > last bin
            self.abs_mag_density = np.load(f)
            self.abs_mag_density = np.insert(self.abs_mag_density, [0,len(self.abs_mag_density)], [self.abs_mag_density[0],0])
            self.abs_mag_bins = np.load(f)

    def __enter__(self):
        np.set_printoptions(precision=4, suppress=True)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        t = time.time()
        with open(f"bin/redshift_guesser_{t}.npy", 'wb') as f:
            np.save(f, self.quick_nn, allow_pickle=False)
            np.save(f, self.quick_correct, allow_pickle=False)
            np.save(f, self.nn_used, allow_pickle=False)
            np.save(f, self.nn_correct, allow_pickle=False)
        

        # TODO adding 1 to denominator hack
        print(f"Quick NN uses: {self.quick_nn}. Success: {self.quick_correct / (self.quick_nn+1)}")
        print(f"NN bin uses: {self.nn_used}. Success: {self.nn_correct / (self.nn_used+1)}")
        
        np.set_printoptions(precision=8, suppress=False)


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
        if self.debug:
            print(f"Angular Distance scores: {ang_dist_scores}")

        # Form groups 
        # TODO grouping effects?
        # Scale the nearest neighbor in the group's score by the number of neighbors in that group
        # Remove the other ones in a group from the pool by setting the score to 0
        grouped_scores = np.copy(ang_dist_scores)
        #group_z = np.copy(neighbors_z)
        for i in range(0, len(neighbors_z)):
            for j in range(len(neighbors_z)-1, 0, -1):
                if grouped_scores[j] > 0 and grouped_scores[i] > 0 and i!=j:
                    if close_enough(neighbors_z[i], neighbors_z[j]):
                        grouped_scores[j] = 0 # remove from the pool
                        grouped_scores[i] += ang_dist_scores[i] 

        if self.debug:
            print(f"Grouped scores: {grouped_scores}")                

        # Let implied abs mag weigh group. 
        # TODO Issue: Overweighting? see plot of results vs truth
        implied_abs_mag = app_mag_to_abs_mag(target_app_mag, neighbors_z)
        abs_mag_bin = np.digitize(implied_abs_mag, self.abs_mag_bins)
        abs_mag_scores = self.abs_mag_density[abs_mag_bin]
        if self.debug:
            print(f"abs_mag_scores scores: {abs_mag_scores}")   

        final_scores = grouped_scores * abs_mag_scores

        return final_scores
    
    def choose_winner(self, neighbors_z, neighbors_ang_dist, target_prob_obs, target_app_mag, target_z_true):
        if self.debug:
            print(f"\nNew call to choose_winner")
            
        k = 0
        if use_nn(neighbors_z[k], neighbors_ang_dist[k], target_prob_obs, target_app_mag):
            if close_enough(target_z_true, neighbors_z[k]):
                self.quick_correct += 1
            self.quick_nn += 1
            if self.debug:
                print(f"Used quick NN. True z={target_z_true}. NN: z={neighbors_z[k]}, ang dist={neighbors_ang_dist[k]}")
            return 0

        scores = self.score_neighbors(neighbors_z, neighbors_ang_dist, target_prob_obs, target_app_mag)

        if self.debug:
            print(f"Scores: {scores}")

        # Randomly choose a winner uses the scores as a PDF to draw from
        #i = self.rng.choice(len(neighbors_z), p=scores)
        #i = random.choices(range(len(neighbors_z)), weights=scores)

        # ... or choose maximal score
        i = np.argmax(scores)


        if close_enough(target_z_true, neighbors_z[i]):
            self.nn_correct[i] += 1
        self.nn_used[i] = self.nn_used[i] + 1
        if self.debug:
            print(f"Used scoring. True z={target_z_true}. NN {i}: z={neighbors_z[i]}, ang dist={neighbors_ang_dist[i]}")

        return i

