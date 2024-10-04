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
        close_idx = np.argwhere(close_enough)
        
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
        idx_for_random = np.argwhere(~use_nn)

        #z_chosen = self.rng.choice(self.app_mag_map[bin_i[0]])
        for i in np.arange(len(bin_i)):
            random_z[idx_for_random[i]] = np.random.choice(self.app_mag_map[bin_i[i]])
        
        return_z = np.where(use_nn, neighbor_z, random_z)
        return return_z, use_nn



class PhotometricRedshiftGuesser(RedshiftGuesser):

    def __init__(self, debug=False):
        print("Initializing v1 of PhotometricRedshiftGuesser")
        self.debug = debug
        self.nna = None
        self.app_mag_bins = None
        self.app_mag_map = None
        #self.score_threshold = 0.9 
        self.photoz_weight = 0.8 # TODO tune these
        self.other_weight = 1.0 # TODO tune these
        self.neighbors_considered = 10 # TODO tune these
        self.zmatch_sigma = 0.004 # TODO tune these
        self.zmatch_pow = 4.0 # TODO tune these

    
    @classmethod
    def from_files(cls, app_mag_to_z_file, nna_file):
        obj = cls()
        print(f"Warning: using saved app mag -> z map")
        with open(app_mag_to_z_file, 'rb') as f:
            obj.app_mag_bins, obj.app_mag_map = pickle.load(f)
        with open(nna_file, 'rb') as f:
            obj.nna = NNAnalyzer_cic.from_results_file(NEIGHBOR_ANALYSIS_SV3_BINS_FILE)
        return obj
    
    @classmethod
    def from_data(cls, app_mags, z_obs, nna_file):        
        obj = cls()
        obj.app_mag_bins, obj.app_mag_map = build_app_mag_to_z_map(app_mags, z_obs)
        with open(nna_file, 'rb') as f:
            obj.nna = NNAnalyzer_cic.from_results_file(NEIGHBOR_ANALYSIS_SV3_BINS_FILE)
        return obj

    def choose_redshift(self, neighbor_z, neighbor_ang_dist, target_z_phot, target_prob_obs, target_app_mag, target_quiescent, nn_quiescent) -> tuple[np.ndarray[np.float64], np.ndarray[np.int64]]:
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

        Returns:
        --------
        tuple[np.ndarray[np.float64], np.ndarray[np.int64]]
            A tuple containing two arrays:
            - z_chosen: Array of shape (COUNT,) containing the chosen redshifts for the targets.
            - neighbor_used: Array of shape (COUNT,) containing the indices of the neighbors used for the chosen redshifts.
        """
        with np.printoptions(precision=4, suppress=True):

            num_neighbors = neighbor_z.shape[0]
            if self.neighbors_considered > num_neighbors:
                print(f"Warning: only {num_neighbors} neighbors available, but {self.neighbors_considered} requested.")
                self.neighbors_considered = num_neighbors
            gal_count = neighbor_z.shape[1]
            assert gal_count == len(target_prob_obs)
            assert gal_count == len(target_app_mag)
            assert gal_count == len(target_quiescent)
            assert gal_count == len(target_z_phot)

            if target_quiescent.dtype == bool:
                target_quiescent = target_quiescent.astype(float)
            if nn_quiescent.dtype == bool:
                nn_quiescent = nn_quiescent.astype(float)

            #print(f"{N} neighbors will be considered.")

            # PHOTO-Z scoring of neighbor
            delta_z = np.abs(neighbor_z - target_z_phot)
            dzp = np.power(delta_z, self.zmatch_pow)
            score_a = np.exp(- dzp / (2*self.zmatch_sigma**2))
            #print(f"Photo-z Scores {score_a}; shape {score_a.shape}")

            # Other properties scoring of neighbor
            score_b = np.zeros((self.neighbors_considered, gal_count))
            for i in range(self.neighbors_considered):
                # TODO right now our results file has crap for P_obs, so always ignore it
                s = self.nna.get_score(None, target_app_mag, target_quiescent, neighbor_z[i], neighbor_ang_dist[i], nn_quiescent[i])
                #print(f"Other Scores {s}; shape {s.shape}")
                score_b[i, :] = s

            #print(f"Other Scores {score_a}; shape {score_a.shape}")

            score = self.photoz_weight*score_a + self.other_weight*score_b
            #print(f"Total score: \n{score}")

            THRESH = 1.0 # redundant with other scores params so no need to tune
            max_neighbor_index = np.argmax(score, axis=0)
            max_scores = np.max(score, axis=0)
            #print(max_neighbor_index)

            # If the max score is over the threshold, use that neighbor's redshift
            z_chosen = np.where(max_scores > THRESH, neighbor_z[max_neighbor_index, np.arange(gal_count)], np.nan)
            neighbor_used = np.where(max_scores > THRESH, max_neighbor_index, np.nan) + 1

            # Otherwise draw a random redshift from the apparent mag bin similar to the target
            use_nn = ~np.isnan(z_chosen)
            idx = np.argwhere(~use_nn)
            bin_i = np.digitize(target_app_mag[~use_nn], self.app_mag_bins)

            #z_chosen = self.rng.choice(self.app_mag_map[bin_i[0]])
            for i in np.arange(len(bin_i)):
                z_chosen[idx[i]] = np.random.choice(self.app_mag_map[bin_i[i]])
            

        return z_chosen, neighbor_used