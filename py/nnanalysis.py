import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c
import astropy.coordinates as coord
import astropy.units as u
import numpy.ma as ma
from random import randint
import pickle
import sys
from scipy.interpolate import interpn
import sys
from pandas.api.types import CategoricalDtype
from scipy.ndimage import gaussian_filter

if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from pyutils import *
#from plotting import *
from dataloc import *
#import k_correction as kcorr
#import kcorr.k_corrections as desikc


def getlabel(index, z_bins):
    if index==0:
        label = "< {0:.2f}".format(z_bins[index])
    else:
        label = "{0:.2f} - {1:.2f}".format(z_bins[index-1], z_bins[index])
    return label

def get_color_label(nn_q, target_q):

    if nn_q and target_q:
        title = "Red NN, Red Target"
    elif not nn_q and target_q:
        title = "Blue NN, Red Target"
    elif nn_q and not target_q:
        title = "Red NN, Blue Target"
    elif not nn_q and not target_q:
        title = "Blue NN, Blue Target"
    else:
        raise Exception("color combination error")

    return title

def cic_binning(data, bin_edges, logscale=None, weights=None):
    num_data_points = data.shape[0]
    data_c = data.copy()

    if len(data_c.shape) == 1:
        data_c = data_c[:, np.newaxis]
    num_dimensions = data_c.shape[1]

    # When logscale is provided, convert the data to linear scale so that binning can be done correctly
    if logscale is not None:
        assert len(logscale) == num_dimensions
        for dim in range(num_dimensions):
            if logscale[dim]:
                data_c[:, dim] = np.power(logscale[dim], data_c[:, dim])
                bin_edges[dim] = np.power(logscale[dim], bin_edges[dim])

    if weights is not None:
        assert len(weights) == num_data_points
        if isinstance(weights, list):
            weights = np.array(weights)

    #print(f"Num dimensions: {num_dimensions}, Num data points: {num_data_points}")
    assert len(bin_edges) == num_dimensions

    shape = [len(edges) for edges in bin_edges]
    #print(f"Shape: {shape}")

    # Calculate the bin indices and weights for each dimension
    bin_indices = np.zeros(np.shape(data_c), dtype='int32') # array of left bin edge indicies for each data point in each dimension
    weights_left = np.zeros(np.shape(data_c), dtype='float64')
    weights_right = np.zeros(np.shape(data_c), dtype='float64')

    all_bincounts = np.zeros(shape, dtype='float64')

    # Force data values lower than the lowest bin edge to be the first bin value and similar for high bin values for each dimension
    for dim in range(num_dimensions):
        data_c[:, dim] = np.clip(data_c[:, dim], bin_edges[dim][0], bin_edges[dim][-1])
    #print(f"Data clipped: \n{data_clipped}")  

    for dim in range(num_dimensions):
        #print(f"Dimension: {dim}")
        # Calculate the bin index for the current dimension - left
        bin_index = np.digitize(data_c[:, dim], bin_edges[dim]) - 1
        bin_index = np.clip(bin_index, 0, len(bin_edges[dim]) - 1) # I think this is redudant now
        bin_indices[:, dim] = bin_index
        
        # Calculate the distance to the left and right bin edges
        left_edge = bin_edges[dim][bin_index]
        right_edge = bin_edges[dim][np.minimum(bin_index + 1, len(bin_edges[dim]) - 1)]
        dist_left = (data_c[:, dim] - left_edge)
        dist_right = (right_edge - data_c[:, dim])
        width = dist_left + dist_right
        
        # Handle the case where data is exactly on the right edge
        on_right_edge = (data_c[:, dim] == right_edge)
        dist_left[on_right_edge] = 0.0
        dist_right[on_right_edge] = 1.0
        width[on_right_edge] = 1.0
        
        weights_left[:, dim] = dist_right / width
        weights_right[:, dim] = dist_left / width

        #assert np.isclose(weights_left[:, dim] + weights_right[:, dim], 1.0).all()

        #print(f"Values        :\n {data_clipped[:, dim]}")
        #print(f"Bin Index     :\n {bin_index}")
        #print(f"Bin Value     :\n {bin_edges[dim][bin_index]}")
        #print(f"Left Weights  :\n {weights_left[:, dim]}")
        #print(f"Right Weights :\n {weights_right[:, dim]}\n")

    # Generate all combinations of bin_indices and bin_indices + 1
    x = [0, 1]
    cloud_idx = np.array(np.meshgrid(*[x]*num_dimensions)).T.reshape(-1, num_dimensions)
    # Calculate the weights for all combinations
    subtotal_weights = np.prod(np.where(cloud_idx == 0, weights_left[:, np.newaxis, :], weights_right[:, np.newaxis, :]), axis=2)
    #print(f"Subtotal Weights: \n{subtotal_weights}")
    #print(f"Shape of subtotal weights: {np.shape(subtotal_weights)}")

    # If user weights are provided, multiply them with the subtotal weights
    if weights is not None:
        #user_weights = np.broadcast_to(weights[:, np.newaxis], (num_data_points, 2**num_dimensions))
        subtotal_weights *= weights[:, np.newaxis]

    # Calculate the indices for all combinations
    indices = bin_indices[:, np.newaxis, :] + cloud_idx

    # Flatten the weights and indices for easy accumulation
    flat_weights = subtotal_weights.flatten()
    flat_indices = indices.reshape(-1, num_dimensions)
    
    # Remove any indices that are outside the bin range on right
    outside_range = np.any(flat_indices >= shape, axis=1)
    flat_indices = flat_indices[~outside_range]
    flat_weights = flat_weights[~outside_range]
    #print(f"Flat Indices: \n{flat_indices}")
    #print(f"Flat Weights: \n{flat_weights}")

    # Accumulate the weights into the bin counts
    np.add.at(all_bincounts, tuple(flat_indices.T), flat_weights)

    assert np.isclose(np.sum(all_bincounts), num_data_points * (np.average(weights) if weights is not None else 1.0)), f"Sum of bin counts: {np.sum(all_bincounts)}, Num data points: {num_data_points}"

    return all_bincounts


class NNAnalyzer_cic():

    def __init__(self):

        self.row_locator = None
        self.df = None
        self.data_cache = {}
        
        # Now bin so that things with ang distances higher than the max we care about are thrown out
        #print("Angular Distance Bin Markers", ANGULAR_BINS)
        #print("Redshift Bin Markers", Z_BINS)
        #print("Abs mag Bin Markers", ABS_MAG_BINS)
        #print("Pobs Bin Markers", POBS_BINS)
        #print("App mag bin markers", APP_MAG_BINS)

        # T_POBS N_Q T_Q N_Z T_APPMAG N_ANG_DIST N_ABSMAG
        # TODO have neighbor number be one of these and then I can integrate over it to 
        # have more overall pairs to evaluate.
        # Then it also lets me analyze affect of increasing neighbor number
        self.reset_bins()

        #print(self.all_ang_bincounts.shape)

    def reset_bins(self):
        self.all_ang_bincounts = np.zeros((len(POBS_BINS), 2, 2, len(Z_BINS), len(APP_MAG_BINS), len(ANGULAR_BINS), len(ABS_MAG_BINS)))
        self.all_sim_z_bincounts = np.zeros((len(POBS_BINS), 2, 2, len(Z_BINS), len(APP_MAG_BINS), len(ANGULAR_BINS), len(ABS_MAG_BINS)))

    @classmethod
    def from_data(cls, dec, ra, z_obs, app_mag, abs_mag, g_r, quiescent, observed, prob_obs):

        assert len(dec) == len(ra)
        assert len(dec) == len(z_obs)
        assert len(dec) == len(app_mag)
        assert len(dec) == len(abs_mag)
        assert len(dec) == len(g_r)
        assert len(dec) == len(quiescent)
        assert len(dec) == len(observed)
        assert len(dec) == len(prob_obs)

        df = pd.DataFrame(data={
            'dec': dec, 
            'ra': ra,
            'z_obs': z_obs,
            'app_mag': app_mag,
            'abs_mag': abs_mag,
            'g_r': g_r,
            'quiescent': quiescent.astype(float),
            'observed': observed,
            'prob_obs': prob_obs
        })

        assert not np.any(np.isnan(z_obs)), "Some z_obs are nan; need to update code to handle data where some z are unknown"
        assert not np.any(np.isnan(abs_mag)), "Some abs_mag are nan; need to update code to handle data where some z are unknown"

        obj = cls()
        obj.df = df

        return obj

    @classmethod
    def from_results_file(cls, filename):
        with open(filename, 'rb') as f:
            all_counts, simz_counts = pickle.load(f)
        
        obj = cls()
        obj.all_ang_bincounts = all_counts
        obj.all_sim_z_bincounts = simz_counts

        return obj
    
    def apply_gaussian_smoothing(self, sigma):
        axes = [0,3,4,5,6] # skip the quiescent ones since we want to keep hard cut there
        self.all_ang_bincounts = gaussian_filter(self.all_ang_bincounts, sigma=sigma, axes=axes)
        self.all_sim_z_bincounts = gaussian_filter(self.all_sim_z_bincounts, sigma=sigma, axes=axes)
 
    def set_row_locator(self, row_locator):
        assert len(row_locator) == len(self.df)
        self.row_locator = row_locator

    def find_nn_properties(self, LOST_GALAXIES_ONLY):
        if self.df is None:
            raise ValueError("Need to create using from_data() first.")

        df = self.df
        
        if LOST_GALAXIES_ONLY:
            catalog = coord.SkyCoord(ra=df.loc[df.observed, 'ra'].to_numpy()*u.degree, dec=df.loc[df.observed, 'dec'].to_numpy()*u.degree, frame='icrs')
            z_obs_catalog = df.loc[df.observed, 'z_obs'].to_numpy()
            color_catalog = df.loc[df.observed, 'quiescent'].to_numpy()
            abs_mag_catalog = df.abs_mag.to_numpy()

        else:
            catalog = coord.SkyCoord(ra=df.ra.to_numpy()*u.degree, dec=df.dec.to_numpy()*u.degree, frame='icrs')
            z_obs_catalog = df.z_obs.to_numpy()
            color_catalog = df['QUIESCENT'].to_numpy()
            abs_mag_catalog = df.abs_mag.to_numpy()

        if LOST_GALAXIES_ONLY: 
            nthneighbor = 1
            if self.row_locator is None:
                self.row_locator = np.invert(df['observed'])
            else:
                self.row_locator = np.logical_and(np.invert(df['observed']), self.row_locator)
        else:
            nthneighbor = 2 # since catalog includes the targets in this case
            if self.row_locator is None:
                self.row_locator = np.repeat(True, len(df))

        rl = self.row_locator

        to_match = coord.SkyCoord(ra=df.loc[rl,'ra'].to_numpy()*u.degree, dec=df.loc[rl,'dec'].to_numpy()*u.degree, frame='icrs')
        idx, d2d, d3d = coord.match_coordinates_sky(to_match, catalog, nthneighbor=nthneighbor, storekdtree=False)
        ang_dist = d2d.to(u.arcsec).value
        sim_z = rounded_tophat_score(df.loc[rl,'z_obs'], z_obs_catalog[idx])

        df['nn1_ang_dist'] = np.nan
        df.loc[rl,'nn1_ang_dist'] = ang_dist
        df['nn1_abs_mag'] = np.nan
        df.loc[rl, 'nn1_abs_mag'] = abs_mag_catalog[idx]
        df['nn1_z'] = np.nan
        df.loc[rl, 'nn1_z'] = z_obs_catalog[idx]
        df['nn1_quiescent'] = np.nan
        df.loc[rl, 'nn1_quiescent'] = color_catalog[idx].astype(float)
        df['nn1_sim_z'] = np.nan
        df.loc[rl, 'nn1_sim_z'] = sim_z

        #print(df.dtypes)
        
        print("Nearest Neighbor properties set")


    def make_bins(self):
        if self.df is None:
            raise ValueError("Need to create using from_data() first.")
        
        df = self.df
        row_locator = self.row_locator
        gal = df.loc[row_locator]
        gal.reset_index(drop=True, inplace=True)
        #print(f"Length of df: {len(df)}")
        #print(f"Length of gal: {len(gal)}")

        self.reset_bins()

        # extract from the gal DataFrame the data we need as a numpy array
        #np.zeros((len(POBS_BINS), 2, 2, len(Z_BINS), len(APP_MAG_BINS), len(ANGULAR_BINS), len(ABS_MAG_BINS)))
        data = gal[['prob_obs', 'nn1_quiescent', 'quiescent', 'nn1_z', 'app_mag', 'nn1_ang_dist', 'nn1_abs_mag']].to_numpy()
        assert not np.any(np.isnan(data)), "Some data is nan"

        #print(f"Data shape: {np.shape(data)}, Bin shape: {np.shape(self.all_ang_bincounts)}")
        self.all_ang_bincounts = cic_binning(
            data, 
            [POBS_BINS, np.array([0,1]), np.array([0,1]), Z_BINS, APP_MAG_BINS, ANGULAR_BINS, ABS_MAG_BINS],
            logscale=[False, False, False, False, 2.5, False, 2.5],)
        print(f"All Bincounts complete. Overall shape: {np.shape(self.all_ang_bincounts)}")
        
        self.all_sim_z_bincounts = cic_binning(
            data, 
            [POBS_BINS, np.array([0,1]), np.array([0,1]), Z_BINS, APP_MAG_BINS, ANGULAR_BINS, ABS_MAG_BINS], 
            logscale=[False, False, False, False, 2.5, False, 2.5],
            weights=gal['nn1_sim_z'].to_numpy())
        print(f"SimZ Bincounts complete. Overall shape: {np.shape(self.all_sim_z_bincounts)}")
        

    def binary_split(self, data):
         # Make rough bins of just over a threshold or not
        nn_success_thresh = 0.4 # change fit lines below if you change this!
        success_bins = [0, nn_success_thresh, 1.01]
        return np.digitize(data, bins=success_bins)
    
    def integrate_out_dimension(self, axis):
        # pivot table could make this easier

        if axis in self.data_cache:
            return self.data_cache[axis]

        all_counts = np.sum(self.all_ang_bincounts, axis=axis)
        simz_counts = np.sum(self.all_sim_z_bincounts, axis=axis)
        #assert simz_counts < all_counts, "SimZ counts should be less than all counts in each bin!"
        all_div = np.where(all_counts == 0.0, 1.0, all_counts)
        frac = simz_counts / all_div # empty bins become 0 TODO

        self.data_cache[axis] = frac, simz_counts, all_counts

        #print(f"Integrated out dimension {axis}. New shape: {np.shape(all_counts)}")
        return frac, simz_counts, all_counts

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.all_ang_bincounts, self.all_sim_z_bincounts), f)

    def get_score(self, target_prob_obs: float|np.ndarray|None, target_app_mag: float|np.ndarray, target_quiescent: float|np.ndarray, neighbor_z: float|np.ndarray, neighbor_ang_dist: float|np.ndarray, nn_quiescent: float|np.ndarray):
        
        # Ensure inputs are numpy arrays for consistency
        if target_prob_obs is not None:
            target_prob_obs = np.atleast_1d(target_prob_obs)
            target_prob_obs = np.clip(target_prob_obs, POBS_BINS[0], POBS_BINS[-1])
        target_app_mag = np.atleast_1d(target_app_mag)
        target_app_mag = np.clip(target_app_mag, APP_MAG_BINS[0], APP_MAG_BINS[-1])
        target_quiescent = np.atleast_1d(target_quiescent)
        target_quiescent = np.clip(target_quiescent, QUIESCENT_BINS[0], QUIESCENT_BINS[-1])
        neighbor_z = np.atleast_1d(neighbor_z)
        neighbor_z = np.clip(neighbor_z, Z_BINS[0], Z_BINS[-1])
        neighbor_ang_dist = np.atleast_1d(neighbor_ang_dist)
        neighbor_ang_dist = np.clip(neighbor_ang_dist, ANGULAR_BINS[0], ANGULAR_BINS[-1])
        nn_quiescent = np.atleast_1d(nn_quiescent)
        nn_quiescent = np.clip(nn_quiescent, QUIESCENT_BINS[0], QUIESCENT_BINS[-1])

        # Check that all input arrays have the same length
        #assert len(target_prob_obs) == len(target_app_mag) == len(target_quiescent) == len(neighbor_z) == len(neighbor_ang_dist) == len(nn_quiescent), "All input arrays must have the same length"
        
        score = np.zeros_like(target_app_mag, dtype=float)

        # Integrate out the dimensions we don't like 
        if target_prob_obs is None:
            # TODO store in a map of (dims integrated out) => frac array to avoid repitition
            frac_simz, simz_counts, all_counts = self.integrate_out_dimension((0, 6)) # PROB_OBS and N_ABS_MAG
            score = interpn(
                points=(QUIESCENT_BINS, QUIESCENT_BINS, Z_BINS, APP_MAG_BINS, ANGULAR_BINS),
                values=frac_simz,
                xi=np.array([nn_quiescent, target_quiescent, neighbor_z, target_app_mag, neighbor_ang_dist]).T,
                method='linear',
                bounds_error=True
            )
        else:
            frac_simz, simz_counts, all_counts = self.integrate_out_dimension((6)) # N_ABS_MAG
            score = interpn(
                points=(POBS_BINS, QUIESCENT_BINS, QUIESCENT_BINS, Z_BINS, APP_MAG_BINS, ANGULAR_BINS),
                values=frac_simz,
                xi=np.array([target_prob_obs, nn_quiescent, target_quiescent, neighbor_z, target_app_mag, neighbor_ang_dist]).T,
                method='linear',
                bounds_error=True
            )

        return score
 

    def plot_angdist_absmag_per_zbin_cc(self):
        frac, same_counts, all_counts = self.integrate_out_dimension((0,4))
        print(np.min(self.frac), np.max(self.frac))
        
        for nn_color_idx in [0,1]:
            for target_color_idx in [0,1]:
                title = get_color_label(nn_color_idx, target_color_idx)
                print(title)
                z_bin_numbers_to_plot = range(frac.shape[2])

                fig, axes = plt.subplots(nrows=len(z_bin_numbers_to_plot), ncols=3, figsize=(6*3, 4*len(z_bin_numbers_to_plot)))

                row=-1
                for zb in z_bin_numbers_to_plot:
                    row+=1

                    if len(z_bin_numbers_to_plot) == 1:
                        axrow = axes
                    else:
                        axrow = axes[row]
                    phrase = 'Similar Z'
                    
                    cplot = axrow[0].pcolor(ANGULAR_BINS, ABS_MAG_BINS, np.swapaxes(frac[nn_color_idx,target_color_idx,zb,:,:], 0, 1), shading='auto', cmap='RdYlGn', norm=c.Normalize(vmin=0, vmax=0.8))
                    fig.colorbar(cplot, ax=axrow[0])
                    axrow[0].set_title(f"NN {phrase} Fraction (NN z {getlabel(zb, Z_BINS)})")
                    axrow[0].set_ylabel("NN abs R-mag")
                    axrow[0].set_xlabel("Angular Distance (arcsec) to NN")
                    axrow[0].set_xscale('log')
                    axrow[0].set_xlim(2.0, 1000)
                    
                    cplot = axrow[1].pcolor(ANGULAR_BINS, ABS_MAG_BINS, np.swapaxes(self.binary_split(frac)[nn_color_idx,target_color_idx,zb,:,:], 0, 1), shading='auto', cmap='RdYlGn')
                    fig.colorbar(cplot, ax=axrow[1])
                    axrow[1].set_title(f"NN {phrase} Over 40% (NN z {getlabel(zb, Z_BINS)})")
                    axrow[1].set_ylabel("NN abs R-mag")    
                    axrow[1].set_xlabel("Angular Distance (arcsec) to NN")
                    axrow[1].set_xscale('log')
                    axrow[1].set_xlim(2.0, 1000)
                    
                    cplot = axrow[2].pcolor(ANGULAR_BINS, ABS_MAG_BINS, np.swapaxes(all_counts[nn_color_idx,target_color_idx,zb,:,:],0,1), shading='auto', cmap='YlGn', norm=c.LogNorm(vmin=1, vmax=5000))
                    fig.colorbar(cplot, ax=axrow[2])
                    axrow[2].set_title(f"Counts (NN z {getlabel(zb, Z_BINS)})")
                    axrow[2].set_ylabel("NN abs R-mag")    
                    axrow[2].set_xlabel("Angular Distance (arcsec) to NN")
                    axrow[2].set_xscale('log')
                    axrow[2].set_xlim(2.0, 1000)
                    
                    #for i in range(len(axrow)):
                    #    axrow[i].scatter(get_NN_40_line_v5(np.repeat(Z_BINS[zb]-0.01, len(APP_MAG_BINS)), APP_MAG_BINS, target_color_idx, nn_color_idx), APP_MAG_BINS)
                        
                fig.suptitle(title)
                fig.tight_layout() 

    def plot_angdist_appmag_per_zbin_cc(self, t_c=[0,1], nn_c=[0,1], z_bin_numbers_to_plot=None):
        self.frac, same_counts, all_counts = self.integrate_out_dimension((0,6))
        print(np.min(self.frac), np.max(self.frac))
        
        for nn_q in nn_c:
            for t_q in t_c:
                title = get_color_label(nn_q, t_q)
                print(title)
                if z_bin_numbers_to_plot is None:
                    z_bin_numbers_to_plot = range(self.frac.shape[2])

                fig, axes = plt.subplots(nrows=len(z_bin_numbers_to_plot), ncols=3, figsize=(6*3, 4*len(z_bin_numbers_to_plot)))

                row=-1
                for zb in z_bin_numbers_to_plot:
                    row+=1

                    if len(z_bin_numbers_to_plot) == 1:
                        axrow = axes
                    else:
                        axrow = axes[row]
                    phrase = 'Similar Z'
                    
                    cplot = axrow[0].pcolor(ANGULAR_BINS, APP_MAG_BINS, self.frac[nn_q,t_q,zb,:,:], shading='auto', cmap='RdYlGn', norm=c.Normalize(vmin=0, vmax=0.8))
                    fig.colorbar(cplot, ax=axrow[0])
                    axrow[0].set_title(f"{phrase} Fraction (z {getlabel(zb, Z_BINS)})")
                    axrow[0].set_ylabel("Lost Galaxy app r-mag")
                    axrow[0].set_xlabel("Angular Distance (arcsec) to NN")
                    axrow[0].set_xscale('log')
                    axrow[0].set_xlim(2.0, 1000)

                    cplot = axrow[1].pcolor(ANGULAR_BINS, APP_MAG_BINS, self.binary_split(self.frac)[nn_q,t_q,zb,:,:], shading='auto', cmap='RdYlGn')
                    fig.colorbar(cplot, ax=axrow[1])
                    axrow[1].set_title(f"{phrase} Over 40% (z {getlabel(zb, Z_BINS)})")
                    axrow[1].set_ylabel("Lost Galaxy app r-mag")    
                    axrow[1].set_xlabel("Angular Distance (arcsec) to NN")
                    axrow[1].set_xscale('log')
                    axrow[1].set_xlim(2.0, 1000)

                    cplot = axrow[2].pcolor(ANGULAR_BINS, APP_MAG_BINS, all_counts[nn_q,t_q,zb,:,:], shading='auto', cmap='YlGn', norm=c.LogNorm(vmin=1, vmax=5000))
                    fig.colorbar(cplot, ax=axrow[2])
                    axrow[2].set_title(f"Counts (z {getlabel(zb, Z_BINS)})")
                    axrow[2].set_ylabel("Lost Galaxy app r-mag")    
                    axrow[2].set_xlabel("Angular Distance (arcsec) to NN")
                    axrow[2].set_xscale('log')
                    axrow[2].set_xlim(2.0, 1000)
                    
                    #for i in range(len(axrow)):
                    #    mags = np.linspace(np.min(APP_MAG_BINS), np.max(APP_MAG_BINS), 40)
                    #    axrow[i].scatter(get_NN_40_line_v5(np.repeat(Z_BINS_FOR_SIMPLE_MIDPOINTS[zb], len(mags)), mags, t_q, nn_q), mags)
                        
                fig.suptitle(title)
                fig.tight_layout() 

    def plot_angdist_pobs_per_zbin_cc(self, t_c=[0,1], nn_c=[0,1], z_bin_numbers_to_plot=None):
        frac, sim_z_counts, all_counts = self.integrate_out_dimension((4,6))  # (15, 2, 2, 8, 20)
        frac, sim_z_counts, all_counts = self.integrate_out_dimension((4,6))  # (15, 2, 2, 8, 20)
        print(np.min(self.frac), np.max(self.frac))

        for nn_quiescent in nn_c:
            for target_quiescent in t_c:
                title = get_color_label(nn_quiescent, target_quiescent)
                print(title)
                if z_bin_numbers_to_plot is None:
                    z_bin_numbers_to_plot = range(frac.shape[3])

                fig, axes = plt.subplots(nrows=len(z_bin_numbers_to_plot), ncols=3, figsize=(6*3, 4*len(z_bin_numbers_to_plot)))

                row=-1
                for zb in z_bin_numbers_to_plot:
                    row+=1
                    #print(f"Galaxies in this z-bin: {np.sum(density)}")

                    if len(z_bin_numbers_to_plot) == 1:
                        axrow = axes
                    else:
                        axrow = axes[row]
                    
                    ax=axrow[0]

                    phrase = 'Similar Z'

                    cplot = ax.pcolor(ANGULAR_BINS, POBS_BINS, frac[:,nn_quiescent,target_quiescent,zb,:], shading='auto', cmap='RdYlGn', norm=c.Normalize(vmin=0, vmax=0.8))
                    fig.colorbar(cplot, ax=ax)
                    ax.set_title(f"NN {phrase} Fraction (NN z {getlabel(zb, Z_BINS)})")
                    ax.set_ylabel("Lost Galaxy $P_{obs}$")
                    ax.set_xlabel("Angular Distance (arcsec) to NN")
                    ax.set_xscale('log')
                    ax.set_xlim(2.0, 1000)
                    
                    cplot = axrow[1].pcolor(ANGULAR_BINS, POBS_BINS, self.binary_split(frac)[:,nn_quiescent,target_quiescent,zb,:], shading='auto', cmap='RdYlGn')
                    fig.colorbar(cplot, ax=axrow[1])
                    axrow[1].set_title(f"NN {phrase} Over 40% (NN z {getlabel(zb, Z_BINS)})")
                    axrow[1].set_ylabel("Lost Galaxy $P_{obs}$")
                    axrow[1].set_xlabel("Angular Distance (arcsec) to NN")
                    axrow[1].set_xscale('log')
                    axrow[1].set_xlim(2.0, 1000)
                    
                    cplot = axrow[2].pcolor(ANGULAR_BINS, POBS_BINS, all_counts[:,nn_quiescent,target_quiescent,zb,:], shading='auto', cmap='YlGn', norm=c.LogNorm(vmin=1, vmax=5000))
                    fig.colorbar(cplot, ax=axrow[2])
                    axrow[2].set_title(f"Counts (NN z {getlabel(zb, Z_BINS)})")
                    axrow[2].set_ylabel("Lost Galaxy $P_{obs}$")
                    axrow[2].set_xlabel("Angular Distance (arcsec) to NN")
                    axrow[2].set_xscale('log')
                    axrow[2].set_xlim(2.0, 1000)

                    for i in range(len(axrow)):
                        pobs = np.linspace(np.min(POBS_BINS), np.max(POBS_BINS), 40)
                        axrow[i].scatter(get_NN_40_line_v4(np.repeat(Z_BINS_FOR_SIMPLE_MIDPOINTS[zb], len(pobs)), pobs, target_quiescent, nn_quiescent), pobs)
                        

                fig.suptitle(title)
                fig.tight_layout() 