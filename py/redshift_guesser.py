import numpy as np
import sys
import pickle

if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from dataloc import *
from pyutils import *
from nnanalysis import NNAnalyzer_cic



class SimpleRedshiftGuesser(RedshiftGuesser):

    LOW_ABS_MAG_LIMIT = -23.0
    HIGH_ABS_MAG_LIMIT = -13.5

    def __init__(self, app_mags, z_obs, ver, debug=False, use_saved_map=True):
        if ver == '2.0':
            print("Initializing v2.0 of SimpleRedshiftGuesser")
        elif ver == '4.0':
            print("Initializing v4.0 of SimpleRedshiftGuesser")
        elif ver == '5.0':
            print("Initializing v5.0 of SimpleRedshiftGuesser")
        else:
            raise("Invalid version of SimpleRedshiftGuesser")
        self.debug = debug
        self.version = ver
        self.rng = np.random.default_rng()
        self.quick_nn = 0
        self.quick_correct = 0
        self.quick_nn_bailed = 0
        self.random_choice = 0
        self.random_correct = 0

        if z_obs is not None:
            if np.max(z_obs) > 10.0:
                print(f"Warning: in SimpleRedshiftGuesser, z_obs values are very high. z_max={z_obs.max()}")
            if np.isnan(z_obs).any(): 
                print("Warning: in SimpleRedshiftGuesser, z_obs has NaNs")

        if use_saved_map:
            print(f"Warning: using MXXL saved app mag -> z map")
            with open(IAN_MXXL_LOST_APP_TO_Z_FILE, 'rb') as f:
                self.app_mag_bins, self.app_mag_map = pickle.load(f)
        else:
            self.app_mag_bins, self.app_mag_map = build_app_mag_to_z_map(app_mags, z_obs)

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, exc_tb):
        t = time.time()
        if self.debug:
            with open(f"bin/simple_redshift_guesser_{t}.npy", 'wb') as f:
                np.save(f, self.quick_nn, allow_pickle=False)
                np.save(f, self.quick_correct, allow_pickle=False)
                np.save(f, self.quick_nn_bailed, allow_pickle=False)
        
        if self.quick_nn > 0 or self.random_choice > 0:

            if self.quick_correct > 0 or self.random_correct > 0:
                print(f"Quick NN uses: {self.quick_nn}. Success: {self.quick_correct / (self.quick_nn+1)}")
                print(f"Random draw uses: {self.random_choice}. Success: {self.random_correct / (self.random_choice+1)}")
                print(f"Quick NN bailed: {self.quick_nn_bailed}. Affected: {self.quick_nn_bailed / (self.quick_nn+self.random_choice)}")
            else:
                print(f"Quick NN uses: {self.quick_nn}.")
                print(f"Random draw uses: {self.random_choice}.")
                print(f"Quick NN bailed: {self.quick_nn_bailed}. Affected: {self.quick_nn_bailed / (self.quick_nn+self.random_choice)}")
            
            
        super().__exit__(exc_type,exc_value,exc_tb)


    def use_nn(self, neighbor_z, neighbor_ang_dist, target_pobs, target_app_mag, target_quiescent, nn_quiescent):
        if self.version == '5.0':
            angular_threshold = get_NN_40_line_v5(neighbor_z, target_app_mag, target_quiescent, nn_quiescent)
        elif self.version == '4.0':
            angular_threshold = get_NN_40_line_v4(neighbor_z, target_pobs, target_quiescent, nn_quiescent)
        else:
            angular_threshold = get_NN_40_line_v2(neighbor_z, target_pobs)

        close_enough = neighbor_ang_dist < angular_threshold
        close_candidiates = np.sum(close_enough)
        close_idx = np.flatnonzero(close_enough)
        
        if len(close_idx) > 0:
            implied_abs_mag = app_mag_to_abs_mag(target_app_mag[close_idx], neighbor_z[close_idx])
            bad_abs_mag = np.logical_or(implied_abs_mag < SimpleRedshiftGuesser.LOW_ABS_MAG_LIMIT, implied_abs_mag > SimpleRedshiftGuesser.HIGH_ABS_MAG_LIMIT)
            self.quick_nn_bailed += np.sum(bad_abs_mag)

            close_enough[close_idx] = ~bad_abs_mag
            assert close_candidiates == (np.sum(close_enough) + np.sum(bad_abs_mag))

        return close_enough

    def choose_redshift(self, neighbor_z, neighbor_ang_dist, target_prob_obs, target_app_mag, target_quiescent, nn_quiescent, target_z_true=False):

        # Determine if we should use NN    
        use_nn = self.use_nn(neighbor_z, neighbor_ang_dist, target_prob_obs, target_app_mag, target_quiescent, nn_quiescent)
        
        if (target_z_true):
            self.quick_correct += np.sum(close_enough(target_z_true, neighbor_z))

        nn_uses = np.sum(use_nn)
        self.quick_nn += nn_uses
        self.random_choice += len(use_nn) - nn_uses

        # Otherwise draw a random redshift from the apparent mag bin similar to the target
        bin_i = np.digitize(target_app_mag[~use_nn], self.app_mag_bins)

        #if len(self.app_mag_map[bin_i[0]]) == 0:
        #    if bin_i[0] < len(self.app_mag_map) - 2:
        #    bin_i[0] = bin_i[0]+1
        #    print(f"Trouble with app mag {target_app_mag}, which goes in bin {bin_i}")

        random_z = np.copy(neighbor_z)
        idx_for_random = np.flatnonzero(~use_nn)

        #z_chosen = self.rng.choice(self.app_mag_map[bin_i[0]])
        for i in np.arange(len(bin_i)):
            random_z[idx_for_random[i]] = np.random.choice(self.app_mag_map[bin_i[i]])
        
        return_z = np.where(use_nn, neighbor_z, random_z)
        return return_z, use_nn



class PhotometricRedshiftGuesser(RedshiftGuesser):

    def __init__(self, debug=False):
        self.debug = debug
        self.nna: NNAnalyzer_cic = None
        self.app_mag_bins = None
        self.app_mag_map = None
        self.mode: Mode = None
        self.score_b_cache = None
        self.expected_galcount = None
    
    @classmethod
    def from_files(cls, app_mag_to_z_file, nna_file, mode: Mode):
        obj = cls()
        obj.mode = mode
        print(f"Initializing PhotometricRedshiftGuesser for {mode}")
        print(f"Warning: using saved app mag -> z map")
        with open(app_mag_to_z_file, 'rb') as f:
            obj.app_mag_bins, obj.app_mag_map = pickle.load(f)
        obj.nna = NNAnalyzer_cic.from_results_file(nna_file)
        return obj
    
    @classmethod
    def from_data(cls, app_mags, z_obs, nna_file, mode: Mode):        
        obj = cls()
        obj.mode = mode
        print(f"Initializing PhotometricRedshiftGuesser for {mode}")
        obj.app_mag_bins, obj.app_mag_map = build_app_mag_to_z_map(app_mags, z_obs)
        obj.nna = NNAnalyzer_cic.from_results_file(nna_file)
        return obj
    
    def use_score_cache_for_mcmc(self, galcount):
        self.score_b_cache = {}
        self.expected_galcount = galcount

    def choose_redshift(self, neighbor_z, neighbor_ang_dist, target_z_phot, target_prob_obs, target_app_mag, target_quiescent, nn_quiescent, params) -> tuple[np.ndarray[np.float64], np.ndarray[np.int64]]:
        """
        Choose a redshift for targets based on its own photometric and its neighbors' properties.

        Parameters:
        -----------
        neighbor_z : np.ndarray
            Array of shape (N, COUNT) containing the redshifts of the neighbors.
        neighbor_ang_dist : np.ndarray
            Array of shape (N, COUNT) containing the angular distances to the neighbors.
        target_z_phot : np.ndarray
            Array of shape (COUNT,) containing the photometric redshifts of the targets.
        target_prob_obs : np.ndarray
            Array of shape (COUNT,) containing the observational probabilities of the targets.
        target_app_mag : np.ndarray
            Array of shape (COUNT,) containing the apparent magnitudes of the targets.
        target_quiescent : np.ndarray
            Array of shape (COUNT,) containing boolean values indicating if the targets are quiescent.
        nn_quiescent : np.ndarray
            Array of shape (N, COUNT) containing boolean values indicating if the neighbors are quiescent.
        params : tuple
            A tuple containing four arrays (bb, rb, br, rr) each of length 3, representing parameters for scoring.

        Returns:
        --------
        tuple[np.ndarray[np.float64], np.ndarray[np.int64]]
            A tuple containing two arrays:
            - z_chosen: Array of shape (COUNT,) containing the chosen redshifts for the targets.
            - flag_value: Array of shape (COUNT,) containing the AssignedRedshiftFlag values. This is indices of the neighbors used for the chosen redshifts or other values for different choices.
        """         

        num_neighbors = neighbor_z.shape[0]
        gal_count = neighbor_z.shape[1]
        assert gal_count == len(target_prob_obs)
        assert gal_count == len(target_app_mag)
        assert gal_count == len(target_quiescent)
        assert gal_count == len(target_z_phot)

        # This is a cache of the scores; useful to speedup MCMC 
        if self.score_b_cache is not None:
            assert self.expected_galcount == gal_count

        if target_quiescent.dtype == bool:
            target_quiescent = target_quiescent.astype(float)
        if nn_quiescent.dtype == bool:
            nn_quiescent = nn_quiescent.astype(float)

        #print(f"{N} neighbors will be considered.")

        # target color then neighbor color. So rb means red target, blue neighbor.   
        # 0,1,2,3 indxes are a,b,zmatch_sigma,zmatch_pow
        bb, rb, br, rr = params

        if len(bb) == 3:
            assert len(bb) == 3 and len(rb) == 3 and len(br) == 3 and len(rr) == 3
            zmatch_pow = 2.0 # letting this float was too many freedoms
        elif len(bb) == 4: 
            assert len(bb) == 4 and len(rb) == 4 and len(br) == 4 and len(rr) == 4
            zmatch_pow = np.where(target_quiescent,
                    np.where(nn_quiescent, rr[3], rb[3]),
                    np.where(nn_quiescent, br[3], bb[3]))
        else:
            raise ValueError("Invalid length of parameters")
            
        a = np.where(target_quiescent,
                np.where(nn_quiescent, rr[0], rb[0]),
                np.where(nn_quiescent, br[0], bb[0]))
        b = np.where(target_quiescent,
                np.where(nn_quiescent, rr[1], rb[1]),
                np.where(nn_quiescent, br[1], bb[1]))
        zmatch_sigma = np.where(target_quiescent,
                np.where(nn_quiescent, rr[2], rb[2]),
                np.where(nn_quiescent, br[2], bb[2]))

        # PHOTO-Z scoring of neighbor
        delta_z = np.abs(neighbor_z - target_z_phot)
        dzp = np.power(delta_z, zmatch_pow)
        score_a = np.exp(- dzp / (np.power(10.0, -zmatch_sigma)))

        # Other properties scoring of neighbor
        score_b = np.zeros((num_neighbors, gal_count))
        for i in range(num_neighbors):
            # If we have a cache, use it
            if self.score_b_cache is not None:
                if i in self.score_b_cache:
                    score_b[i, :] = self.score_b_cache[i]
                    continue

            # Compute the scores
            score_b[i, :] = self.nna.get_score(None, target_app_mag, target_quiescent, neighbor_z[i], neighbor_ang_dist[i], nn_quiescent[i])
                    
            # If using cache, save the scores for future iterations
            if self.score_b_cache is not None:
                self.score_b_cache[i] = score_b[i, :]

        score = a*score_a + b*score_b
        #print(f"Total score: \n{score}")

        THRESH = 1.0 # redundant with other scores params so no need to tune
        max_neighbor_index = np.argmax(score, axis=0)
        max_scores = np.max(score, axis=0)
        #print(max_neighbor_index)

        # If the max score is over the threshold, use that neighbor's redshift
        z_chosen = np.where(max_scores > THRESH, neighbor_z[max_neighbor_index, np.arange(gal_count)], np.nan)
        flag_value = np.where(max_scores > THRESH, max_neighbor_index + 1, AssignedRedshiftFlag.PSEUDO_RANDOM.value)
        used_neighbors = ~(flag_value == AssignedRedshiftFlag.PSEUDO_RANDOM.value)
        idx_remaining = np.flatnonzero(~used_neighbors)

        # For the other ones, what we do is different for each mode
        if self.mode == Mode.PHOTOZ_PLUS_v1:
            # Otherwise draw a random redshift from the apparent mag bin similar to the target
            bin_i = np.digitize(target_app_mag[~used_neighbors], self.app_mag_bins)
            #otherway = z_chosen.copy()

            #t1 = time.time()
            for i in np.arange(len(bin_i)):
                z_chosen[idx_remaining[i]] = np.random.choice(self.app_mag_map[bin_i[i]])
            #t2 = time.time()

            # TODO dictionary isn't a vectorized object... this doesn't work at all
            #t3= time.time()
            #toset = np.random.choice(self.app_mag_map[bin_i])
            #otherway[idx_remaining] = toset
            #t4 = time.time()

            #assert np.isclose(z_chosen, otherway, equal_nan=True).all()
            #print(f"Time for loop: {t2-t1:.3f}. Time for vectorized: {t4-t3:.3f}")
                
        elif self.mode == Mode.PHOTOZ_PLUS_v2:
            # Just use the photo-z directly
            z_chosen[idx_remaining] = target_z_phot[idx_remaining]
            flag_value[idx_remaining] = AssignedRedshiftFlag.PHOTO_Z.value

            # For bad ones (no photo-z)
            idx_bad = np.flatnonzero(np.logical_or(np.isnan(target_z_phot), target_z_phot < 0.0))

            bin_i = np.digitize(target_app_mag[idx_bad], self.app_mag_bins)
            for i in np.arange(len(bin_i)):
                z_chosen[idx_bad[i]] = np.random.choice(self.app_mag_map[bin_i[i]])
                flag_value[idx_bad] = AssignedRedshiftFlag.PSEUDO_RANDOM.value

        elif self.mode == Mode.PHOTOZ_PLUS_v3:

            # use the bins of photo-z and app-mag as keys
            raise ValueError("Not implemented yet")

        assert np.isnan(z_chosen).sum() == 0, f"Some redshifts were not chosen. {z_chosen}"

        return z_chosen, flag_value
