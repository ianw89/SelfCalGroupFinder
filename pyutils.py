import numpy as np
import math
import sys
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import astropy.coordinates as coord
from datetime import datetime
import time
import random
from scipy import special
#sys.path.append("/Users/ianw89/Documents/GitHub/hodpy")
#from hodpy.cosmology import CosmologyMXXL
#from hodpy.k_correction import GAMA_KCorrection


_cosmo = FlatLambdaCDM(H0=73, Om0=0.25, Ob0=0.045, Tcmb0=2.725, Neff=3.04) 
def get_MXXL_cosmology():
    return _cosmo

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


def app_mag_to_abs_mag_k(app_mag, z_obs, colour):
    """
    Converts apparent mags to absolute mags using MXXL cosmology and provided observed redshifts.
    """
    pass
    #cosmo = CosmologyMXXL()
    #kcorr = GAMA_KCorrection(cosmo)
    #return kcorr.absolute_magnitude(app_mag, z_obs, colour)

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


def build_app_mag_to_z_map(app_mag, z_obs):
    _NBINS = 100
    app_mag_bins = np.linspace(min(app_mag), max(app_mag), _NBINS)

    app_mag_indices = np.digitize(app_mag, app_mag_bins)

    the_map = {}
    for bin_i in range(1,len(app_mag_bins)+1):
        this_bin_redshifts = z_obs[app_mag_indices == bin_i]
        the_map[bin_i] = this_bin_redshifts

    # for app mags smaller than the smallest we have, use the z distribution of the one right above it
    the_map[0] = the_map[1]

    #print(f"App Mag Building Complete: {the_map}")

    return app_mag_bins, the_map




class RedshiftGuesser():

    def __enter__(self):
        np.set_printoptions(precision=4, suppress=True)
        return self
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        np.set_printoptions(precision=8, suppress=False)

    def choose_winner(self, neighbors_z, neighbors_ang_dist, target_prob_obs, target_app_mag, target_z_true):
        pass

def get_NN_30_line(z, t_Pobs):
    """
    Gets the angular distance at which, according to MXXL, a target with the given Pobs will be in the same halo
    as a nearest neighbor at reshift z 30% of the time.
    """
    FIT_SHIFT_RIGHT = [37,30,31,30,39,39,63,72]
    FIT_SCALE = [10,10,10,10,10,10,10,10]
    FIT_SHIFT_UP = [0.8,0.8,0.8,0.8,0.8,0.8,0.6,0.5]
    FIT_SQUEEZE = [1.4,1.3,1.3,1.3,1.4,1.4,1.5,1.5]
    base = [1.10,1.14,1.14,1.14,1.10,1.10,1.06,1.04]
    zb = np.digitize(z, SimpleRedshiftGuesser.z_bins)

    erf_in = FIT_SQUEEZE[zb]*(t_Pobs - FIT_SHIFT_UP[zb])

    # for middle ones use exponentiated inverse erf to get the curve 
    exponent = FIT_SHIFT_RIGHT[zb] - FIT_SCALE[zb]*special.erfinv(erf_in)
    arcsecs = base[zb]**exponent
    return arcsecs

def get_NN_40_line(z, t_Pobs):
    """
    Gets the angular distance at which, according to MXXL, a target with the given Pobs will be in the same halo
    as a nearest neighbor at reshift z 40% of the time.
    """
    FIT_SHIFT_RIGHT = [25,25,26,27,34,34,53,60]
    FIT_SCALE = [10,10,10,10,10,10,10,10]
    FIT_SHIFT_UP = [0.8,0.8,0.8,0.8,0.8,0.8,0.6,0.5]
    FIT_SQUEEZE = [1.4,1.3,1.3,1.3,1.4,1.4,1.5,1.5]
    base = [1.10,1.14,1.14,1.14,1.10,1.10,1.06,1.04]
    zb = np.digitize(z, SimpleRedshiftGuesser.z_bins)

    erf_in = FIT_SQUEEZE[zb]*(t_Pobs - FIT_SHIFT_UP[zb])

    # for middle ones use exponentiated inverse erf to get the curve 
    exponent = FIT_SHIFT_RIGHT[zb] - FIT_SCALE[zb]*special.erfinv(erf_in)
    arcsecs = base[zb]**exponent
    return arcsecs

def get_NN_50_line(z, t_Pobs):
    """
    Gets the angular distance at which, according to MXXL, a target with the given Pobs will be in the same halo
    as a nearest neighbor at reshift z 50% of the time.
    """
    FIT_SHIFT_RIGHT = [15,20,20,20,25,25,35,40]
    FIT_SCALE = [10,10,10,10,10,10,10,10]
    FIT_SHIFT_UP = [0.8,0.8,0.8,0.8,0.8,0.8,0.6,0.5]
    FIT_SQUEEZE = [1.3,1.3,1.3,1.3,1.4,1.4,1.5,1.5]
    base = [1.16,1.16,1.16,1.16,1.12,1.12,1.08,1.08]
    zb = np.digitize(z, SimpleRedshiftGuesser.z_bins)

    # for higher and lower z bit just use simple cut
    if zb in [0,6,7]:
        return np.full(t_Pobs.shape, 10)
    #if zb == 7:
    #    return np.full(t_Pobs.shape, 10)

    erf_in = FIT_SQUEEZE[zb]*(t_Pobs - FIT_SHIFT_UP[zb])

    # for middle ones use exponentiated inverse erf to get the curve 
    exponent = FIT_SHIFT_RIGHT[zb] - FIT_SCALE[zb]*special.erfinv(erf_in)
    arcsecs = base[zb]**exponent
    return arcsecs



class SimpleRedshiftGuesser(RedshiftGuesser):

    z_bins = [0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.36, 1.0]     
    LOW_ABS_MAG_LIMIT = -23.5
    HIGH_ABS_MAG_LIMIT = -13.0

    def __init__(self, app_mags, z_obs, debug=False):
        print("Initializing v2.1 of SimpleRedshiftGuesser")
        self.debug = debug
        self.rng = np.random.default_rng()
        self.quick_nn = 0
        self.quick_correct = 0
        self.quick_nn_bailed = 0
        self.random_choice = 0
        self.random_correct = 0

        self.app_mag_bins, self.app_mag_map = build_app_mag_to_z_map(app_mags, z_obs)

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, exc_tb):
        t = time.time()
        with open(f"bin/simple_redshift_guesser_{t}.npy", 'wb') as f:
            np.save(f, self.quick_nn, allow_pickle=False)
            np.save(f, self.quick_correct, allow_pickle=False)
            np.save(f, self.quick_nn_bailed, allow_pickle=False)
        
        # TODO adding 1 to denominator hack
        print(f"Quick NN uses: {self.quick_nn}. Success: {self.quick_correct / (self.quick_nn+1)}")
        print(f"Random draw uses: {self.random_choice}. Success: {self.random_correct / (self.random_choice+1)}")
        print(f"Quick NN bailed: {self.quick_nn_bailed}. Affected: {self.quick_nn_bailed / (self.quick_nn+self.random_choice)}")
        
        super().__exit__(exc_type,exc_value,exc_tb)


    def use_nn(self, neighbor_z, neighbor_ang_dist, target_pobs, target_app_mag):
        angular_threshold = get_NN_40_line(neighbor_z, target_pobs)

        if self.debug:
            print(f"Threshold {angular_threshold}\". Nearest neighbor is {neighbor_z}\".")

        close_enough = neighbor_ang_dist < angular_threshold

        if close_enough:
            implied_abs_mag = app_mag_to_abs_mag(target_app_mag, neighbor_z)

            if implied_abs_mag < SimpleRedshiftGuesser.LOW_ABS_MAG_LIMIT or implied_abs_mag > SimpleRedshiftGuesser.HIGH_ABS_MAG_LIMIT:
                self.quick_nn_bailed += 1
                return False
            else:
                return True

    def choose_redshift(self, neighbor_z, neighbor_ang_dist, target_prob_obs, target_app_mag, target_z_true):
        if self.debug:
            print(f"\nNew call to choose_winner")

        # Determine if we should use NN    
        if self.use_nn(neighbor_z, neighbor_ang_dist, target_prob_obs, target_app_mag):
            if close_enough(target_z_true, neighbor_z):
                self.quick_correct += 1
            self.quick_nn += 1
            if self.debug:
                print(f"Used quick NN. True z={target_z_true}. NN: z={neighbor_z}, ang dist={neighbor_ang_dist}")
            return neighbor_z, True

        # Otherwise draw a random redshift from the apparent mag bin similar to the target
        bin_i = np.digitize(target_app_mag, self.app_mag_bins)
        z_chosen = self.rng.choice(self.app_mag_map[bin_i[0]])
        #z_chosen = random.choice(self.app_mag_map.get(bin_i[0]))
        self.random_choice += 1

        if close_enough(target_z_true, z_chosen):
            self.random_correct += 1
        return z_chosen, False






class FancyRedshiftGuesser(RedshiftGuesser):

    def __init__(self, num_neighbors, debug=False):
        print("Initializing v6 of FancyRedshiftGuesser")
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
        return super().__enter__()

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
        
        super().__exit__(exc_type,exc_value,exc_tb)


    # The magic numbers in this right now are the ~0.5 NN same-halo points read-off of plots from the MXXL data
    def use_nn(self, neighbor_z, neighbor_ang_dist, target_prob_obs, target_app_mag):
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
        if self.use_nn(neighbors_z[k], neighbors_ang_dist[k], target_prob_obs, target_app_mag):
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

