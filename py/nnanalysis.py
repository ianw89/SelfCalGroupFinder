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
        label = "< {0:.3f}".format(z_bins[index])
    else:
        label = "{0:.3f} - {1:.3f}".format(z_bins[index-1], z_bins[index])
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

def cic_binning(data, bin_definitions, logscale=None, weights=None):
    """
    Cloud-in-cell bin the data using the provided bin definitions.
    Note that the bin definitions are not edges but rather the centers of the bins, unlike many numpy functions.
    """
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
                bin_definitions[dim] = np.power(logscale[dim], bin_definitions[dim])

    if weights is not None:
        assert len(weights) == num_data_points
        if isinstance(weights, list):
            weights = np.array(weights)

    #print(f"Num dimensions: {num_dimensions}, Num data points: {num_data_points}")
    assert len(bin_definitions) == num_dimensions

    shape = [len(edges) for edges in bin_definitions]
    #print(f"Shape: {shape}")

    # Calculate the bin indices and weights for each dimension
    bin_indices = np.zeros(np.shape(data_c), dtype='int32') # array of left bin edge indicies for each data point in each dimension
    weights_left = np.zeros(np.shape(data_c), dtype='float64')
    weights_right = np.zeros(np.shape(data_c), dtype='float64')

    all_bincounts = np.zeros(shape, dtype='float64')

    # Force data values lower than the lowest bin edge to be the first bin value and similar for high bin values for each dimension
    for dim in range(num_dimensions):
        data_c[:, dim] = np.clip(data_c[:, dim], bin_definitions[dim][0], bin_definitions[dim][-1])
    #print(f"Data clipped: \n{data_clipped}")  

    for dim in range(num_dimensions):
        #print(f"Dimension: {dim}")
        # Calculate the bin index for the current dimension - left
        bin_index = np.digitize(data_c[:, dim], bin_definitions[dim]) - 1
        bin_index = np.clip(bin_index, 0, len(bin_definitions[dim]) - 1) # I think this is redudant now
        bin_indices[:, dim] = bin_index
        
        # Calculate the distance to the left and right bin edges
        left_edge = bin_definitions[dim][bin_index]
        right_edge = bin_definitions[dim][np.minimum(bin_index + 1, len(bin_definitions[dim]) - 1)]
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
        self.reset_bins()

        #print(self.all_ang_bincounts.shape)

    def reset_bins(self):
        self.all_ang_bincounts = np.zeros((len(NEIGHBOR_BINS), 2, 2, len(Z_BINS), len(APP_MAG_BINS), len(ANGULAR_BINS), len(ABS_MAG_BINS)))
        self.all_sim_z_bincounts = np.zeros((len(NEIGHBOR_BINS), 2, 2, len(Z_BINS), len(APP_MAG_BINS), len(ANGULAR_BINS), len(ABS_MAG_BINS)))

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        # Check that the DataFrame has the required columns
        required_columns = ['DEC', 'RA', 'Z', 'APP_MAG_R', 'ABS_MAG_R_K', 'QUIESCENT', 'OBSERVED']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
        # Check that the DataFrame has no NaN values in the required columns
        if df[required_columns].isnull().values.any():
            raise ValueError("DataFrame contains null values in required columns.")
        
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
        #axes = [3,4,5,6] # skip the quiescent ones since we want to keep hard cut there
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
            catalog = coord.SkyCoord(ra=df.loc[df['OBSERVED'], 'RA'].to_numpy()*u.degree, dec=df.loc[df['OBSERVED'], 'DEC'].to_numpy()*u.degree, frame='icrs')
            z_obs_catalog = df.loc[df['OBSERVED'], 'Z'].to_numpy()
            color_catalog = df.loc[df['OBSERVED'], 'QUIESCENT'].to_numpy()
            abs_mag_catalog = df['ABS_MAG_R_K'].to_numpy()

        else:
            catalog = coord.SkyCoord(ra=df['RA'].to_numpy()*u.degree, dec=df['DEC'].to_numpy()*u.degree, frame='icrs')
            z_obs_catalog = df['Z'].to_numpy()
            color_catalog = df['QUIESCENT'].to_numpy()
            abs_mag_catalog = df['ABS_MAG_R_K'].to_numpy()

        if LOST_GALAXIES_ONLY: 
            offset = 0
            if self.row_locator is None:
                self.row_locator = np.invert(df['OBSERVED'])
            else:
                self.row_locator = np.logical_and(np.invert(df['OBSERVED']), self.row_locator)
        else:
            offset = 1 # since catalog includes the targets in this case
            if self.row_locator is None:
                self.row_locator = np.repeat(True, len(df))

        # get a random string to use as a key
        storename = f"nn_properties_{randint(100000, 999999)}.pkl"
        rl = self.row_locator
        to_match = coord.SkyCoord(ra=df.loc[rl, 'RA'].to_numpy() * u.degree, dec=df.loc[rl, 'DEC'].to_numpy() * u.degree, frame='icrs')
        
        # Initialize columns for N neighbors
        for n in NEIGHBOR_BINS:
            df[f'nn{n}_ang_dist'] = np.nan
            df[f'nn{n}_abs_mag'] = np.nan
            df[f'nn{n}_z'] = np.nan
            df[f'nn{n}_quiescent'] = np.nan
            df[f'nn{n}_sim_z'] = np.nan

        # Loop through each neighbor and assign values
        for n in NEIGHBOR_BINS:
            idx_n, d2d_n, _ = coord.match_coordinates_sky(to_match, catalog, nthneighbor=n+offset, storekdtree=storename)
            ang_dist_n = d2d_n.to(u.arcsec).value
            sim_z_n = rounded_tophat_score(df.loc[rl, 'Z'], z_obs_catalog[idx_n])

            df.loc[rl, f'nn{n}_ang_dist'] = ang_dist_n
            df.loc[rl, f'nn{n}_abs_mag'] = abs_mag_catalog[idx_n]
            df.loc[rl, f'nn{n}_z'] = z_obs_catalog[idx_n]
            df.loc[rl, f'nn{n}_quiescent'] = color_catalog[idx_n].astype(float)
            df.loc[rl, f'nn{n}_sim_z'] = sim_z_n

            # Ensure no nan left
            assert df.loc[rl, f'nn{n}_ang_dist'].isnull().sum() == 0, f"NaN values found in nn{n}_ang_dist"
            assert df.loc[rl, f'nn{n}_abs_mag'].isnull().sum() == 0, f"NaN values found in nn{n}_abs_mag"
            assert df.loc[rl, f'nn{n}_z'].isnull().sum() == 0, f"NaN values found in nn{n}_z"
            assert df.loc[rl, f'nn{n}_quiescent'].isnull().sum() == 0, f"NaN values found in nn{n}_quiescent"
            assert df.loc[rl, f'nn{n}_sim_z'].isnull().sum() == 0, f"NaN values found in nn{n}_sim_z"
        
        print("Nearest Neighbor properties set")


    def make_bins(self):
        if self.df is None:
            raise ValueError("Need to create using from_df() first.")
        
        df = self.df
        row_locator = self.row_locator
        gal = df.loc[row_locator]
        gal.reset_index(drop=True, inplace=True)
        print(f"Length of df: {len(df)}")
        print(f"Length of gal: {len(gal)}")

        self.reset_bins()

        # extract from the gal DataFrame the data we need as a numpy array
        data = np.zeros((len(gal)*len(NEIGHBOR_BINS), 7), dtype=float)
        weights = np.zeros(len(gal)*len(NEIGHBOR_BINS), dtype=float)

        offset = 0
        end = offset + len(gal)
        for n in NEIGHBOR_BINS:
            data[offset:end, 0] = n
            data[offset:end, 1] = gal[f'nn{n}_quiescent'].to_numpy()
            data[offset:end, 2] = gal[f'QUIESCENT'].to_numpy() 
            data[offset:end, 3] = gal[f'nn{n}_z'].to_numpy()
            data[offset:end, 4] = gal['APP_MAG_R'].to_numpy()
            data[offset:end, 5] = gal[f'nn{n}_ang_dist'].to_numpy()
            data[offset:end, 6] = gal[f'nn{n}_abs_mag'].to_numpy()

            weights[offset:end] = gal[f'nn{n}_sim_z'].to_numpy()
            offset += len(gal)
            end = offset + len(gal)

        #print(f"Data shape: {np.shape(data)}, Bin shape: {np.shape(self.all_ang_bincounts)}")
        #((len(NEIGHBOR_BINS), 2, 2, len(Z_BINS), len(APP_MAG_BINS), len(ANGULAR_BINS), len(ABS_MAG_BINS)))
        self.all_ang_bincounts = cic_binning(
            data, 
            [NEIGHBOR_BINS, np.array([0,1]), np.array([0,1]), Z_BINS, APP_MAG_BINS, ANGULAR_BINS, ABS_MAG_BINS],
            logscale=[False, False, False, False, 2.5, False, 2.5],)
        print(f"All Bincounts complete. Overall shape: {np.shape(self.all_ang_bincounts)}")
        
        self.all_sim_z_bincounts = cic_binning(
            data, 
            [NEIGHBOR_BINS, np.array([0,1]), np.array([0,1]), Z_BINS, APP_MAG_BINS, ANGULAR_BINS, ABS_MAG_BINS], 
            logscale=[False, False, False, False, 2.5, False, 2.5],
            weights=weights)
        print(f"SimZ Bincounts complete. Overall shape: {np.shape(self.all_sim_z_bincounts)}")

    def fill_bins(self):
        # Fill in bins without any data with neighboring values; don't touch bins with data
        mask = self.all_ang_bincounts == 0  # Identify empty bins
        while np.any(mask):  # Continue until no empty bins remain
            print(f"There are {np.sum(mask)} empty bins to fill.")
            self.all_ang_bincounts[mask] = gaussian_filter(self.all_ang_bincounts, sigma=1)[mask]
            self.all_sim_z_bincounts[mask] = gaussian_filter(self.all_sim_z_bincounts, sigma=1)[mask]
            mask =  self.all_ang_bincounts == 0  # Update mask for remaining empty bins


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
        frac = simz_counts / all_div 

        self.data_cache[axis] = frac, simz_counts, all_counts

        #print(f"Integrated out dimension {axis}. New shape: {np.shape(all_counts)}")
        return frac, simz_counts, all_counts

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.all_ang_bincounts, self.all_sim_z_bincounts), f)

    def get_score(self, neighbor_num: float|np.ndarray|None, target_app_mag: float|np.ndarray, target_quiescent: float|np.ndarray, neighbor_z: float|np.ndarray, neighbor_ang_dist: float|np.ndarray, nn_quiescent: float|np.ndarray):
        
        # Ensure inputs are numpy arrays for consistency
        if neighbor_num is not None:
            neighbor_num = np.atleast_1d(neighbor_num)
            neighbor_num = np.clip(neighbor_num, NEIGHBOR_BINS[0], NEIGHBOR_BINS[-1])
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
        if neighbor_num is None:
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
                points=(NEIGHBOR_BINS, QUIESCENT_BINS, QUIESCENT_BINS, Z_BINS, APP_MAG_BINS, ANGULAR_BINS),
                values=frac_simz,
                xi=np.array([neighbor_num, nn_quiescent, target_quiescent, neighbor_z, target_app_mag, neighbor_ang_dist]).T,
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

    def plot_angdist_appmag_per_zbin_cc(self, t_c=[0,1], nn_c=[0,1], z_bin_numbers_to_plot=None, neighbors=None):
        if neighbors is None:
            self.frac, same_counts, all_counts = self.integrate_out_dimension((0,6))
        else:
            self.frac, same_counts, all_counts = self.integrate_out_dimension((6))
            self.frac = self.frac[neighbors,:,:,:,:,:]
            all_counts = all_counts[neighbors,:,:,:,:,:]

        for nn_q in nn_c:
            for t_q in t_c:
                title = get_color_label(nn_q, t_q)
                print(title)
                if z_bin_numbers_to_plot is None:
                    z_bin_numbers_to_plot = range(self.frac.shape[2])

                fig, axes = plt.subplots(nrows=len(z_bin_numbers_to_plot), ncols=2, figsize=(6*2, 4*len(z_bin_numbers_to_plot)))

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

                    #cplot = axrow[1].pcolor(ANGULAR_BINS, APP_MAG_BINS, self.binary_split(self.frac)[nn_q,t_q,zb,:,:], shading='auto', cmap='RdYlGn')
                    #fig.colorbar(cplot, ax=axrow[1])
                    #axrow[1].set_title(f"{phrase} Over 40% (z {getlabel(zb, Z_BINS)})")
                    #xrow[1].set_ylabel("Lost Galaxy app r-mag")    
                    #axrow[1].set_xlabel("Angular Distance (arcsec) to NN")
                    #axrow[1].set_xscale('log')
                    #axrow[1].set_xlim(2.0, 1000)

                    cplot = axrow[1].pcolor(ANGULAR_BINS, APP_MAG_BINS, all_counts[nn_q,t_q,zb,:,:], shading='auto', cmap='YlGn', norm=c.LogNorm(vmin=1, vmax=500))
                    fig.colorbar(cplot, ax=axrow[1])
                    axrow[1].set_title(f"Counts (z {getlabel(zb, Z_BINS)})")
                    axrow[1].set_ylabel("Lost Galaxy app r-mag")    
                    axrow[1].set_xlabel("Angular Distance (arcsec) to NN")
                    axrow[1].set_xscale('log')
                    axrow[1].set_xlim(2.0, 1000)
                    
                    #for i in range(len(axrow)):
                    #    mags = np.linspace(np.min(APP_MAG_BINS), np.max(APP_MAG_BINS), 40)
                    #    axrow[i].scatter(get_NN_40_line_v5(np.repeat(Z_BINS_FOR_SIMPLE_MIDPOINTS[zb], len(mags)), mags, t_q, nn_q), mags)
                        
                fig.suptitle(title)
                fig.tight_layout() 

    def plot_angdist_neighbor_per_zbin_cc(self, t_c=[0,1], nn_c=[0,1], z_bin_numbers_to_plot=None):
        frac, sim_z_counts, all_counts = self.integrate_out_dimension((4,6))  # (15, 2, 2, 8, 20)

        for nn_quiescent in nn_c:
            for target_quiescent in t_c:
                title = get_color_label(nn_quiescent, target_quiescent)
                print(title)
                if z_bin_numbers_to_plot is None:
                    z_bin_numbers_to_plot = range(frac.shape[3])

                fig, axes = plt.subplots(nrows=len(z_bin_numbers_to_plot), ncols=2, figsize=(6*2, 4*len(z_bin_numbers_to_plot)))

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

                    cplot = ax.pcolor(ANGULAR_BINS, NEIGHBOR_BINS, frac[:,nn_quiescent,target_quiescent,zb,:], shading='auto', cmap='RdYlGn', norm=c.Normalize(vmin=0, vmax=0.8))
                    fig.colorbar(cplot, ax=ax)
                    ax.set_title(f"NN {phrase} Fraction (NN z {getlabel(zb, Z_BINS)})")
                    ax.set_ylabel("Neighbor #")
                    ax.set_xlabel("Angular Distance (arcsec) to NN")
                    ax.set_xscale('log')
                    ax.set_xlim(2.0, 1000)

                    cplot = axrow[1].pcolor(ANGULAR_BINS, NEIGHBOR_BINS, all_counts[:,nn_quiescent,target_quiescent,zb,:], shading='auto', cmap='YlGn', norm=c.LogNorm(vmin=1, vmax=5000))
                    fig.colorbar(cplot, ax=axrow[1])
                    axrow[1].set_title(f"Counts (NN z {getlabel(zb, Z_BINS)})")
                    axrow[1].set_ylabel("Neighbor #")
                    axrow[1].set_xlabel("Angular Distance (arcsec) to NN")
                    axrow[1].set_xscale('log')
                    axrow[1].set_xlim(2.0, 1000)

                    #for i in range(len(axrow)):
                    #    pobs = np.linspace(np.min(POBS_BINS), np.max(POBS_BINS), 40)
                    #    axrow[i].scatter(get_NN_40_line_v4(np.repeat(Z_BINS_FOR_SIMPLE_MIDPOINTS[zb], len(pobs)), pobs, target_quiescent, nn_quiescent), pobs)
                        

                fig.suptitle(title)
                fig.tight_layout() 