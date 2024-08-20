import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c
import astropy.coordinates as coord
import astropy.units as u
import numpy.ma as ma
from random import randint
import pickle
from astropy.table import Table
import sys
from pandas.api.types import CategoricalDtype

if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from pyutils import *
#from plotting import *
from dataloc import *
#import k_correction as kcorr
#import kcorr.k_corrections as desikc


def getlabel(index, z_bins):
    if index==0:
        label = "< {0}".format(z_bins[index])
    else:
        label = "{0} - {1}".format(z_bins[index-1], z_bins[index])
    return label

def get_color_label(target_q, nn_q):

    if target_q == 1 and nn_q == 1:
        title = "Red NN, Red Target"
    elif target_q == 0 and nn_q == 1:
        title = "Blue NN, Red Target"
    elif target_q == 1 and nn_q == 0:
        title = "Red NN, Blue Target"
    elif target_q == 0 and nn_q == 0:
        title = "Blue NN, Blue Target"

    return title




class NNAnalyzer():

    def __init__(self, dec, ra, z_obs, app_mag, halo_id, g_r, quiescent, observed, prob_obs):

        self.has_true_halo_id = (halo_id is not None)

        assert len(dec) == len(ra)
        assert len(dec) == len(z_obs)
        assert len(dec) == len(app_mag)
        assert len(dec) == len(g_r)
        assert len(dec) == len(quiescent)
        assert len(dec) == len(observed)
        assert len(dec) == len(prob_obs)

        self.df = pd.DataFrame(data={
            'dec': dec, 
            'ra': ra,
            'z_obs': z_obs,
            'app_mag': app_mag,
            'g_r': g_r,
            'quiescent': quiescent,
            'observed': observed,
            'prob_obs': prob_obs
        })
        if self.has_true_halo_id:
            assert len(dec) == len(halo_id)
            self.df['halo_id'] = halo_id

        # Now bin so that things with ang distances higher than the max we care about are thrown out
        self.ANG_DIST_BIN_COUNT = 20
        self.BIGGER_THAN_ANY_NN_DIST = 3600

        self.angular_bins = np.append(np.logspace(np.log10(3), np.log10(900), self.ANG_DIST_BIN_COUNT - 1), self.BIGGER_THAN_ANY_NN_DIST)
        print("Angular Distance Bin Markers", self.angular_bins)
        # Must determine NN and distance to them before binning this

        self.z_bins = np.array(SimpleRedshiftGuesser.z_bins)
        print("Redshift Bin Markers", self.z_bins)
        # Must determine NN and distance to them before binning this

        self.POBS_BIN_COUNT = 15
        self.POBS_bins = np.linspace(0.01, 1.01, self.POBS_BIN_COUNT)
        print("Pobs Bin Markers", self.POBS_bins)
        self.df['pobs_bin'] = np.digitize(self.df['prob_obs'], self.POBS_bins)
        #pd.cut(x=self.df['prob_obs'], bins=self.POBS_bins, include_lowest=True)

        self.APP_MAG_BIN_COUNT = 2
        self.app_mag_bins = np.linspace(15.0, 20.01, self.APP_MAG_BIN_COUNT)
        print("App mag bin markers", self.app_mag_bins)
        self.df['app_mag_bin'] = np.digitize(self.df['app_mag'], self.app_mag_bins)
        #pd.cut(x=self.df['app_mag'], bins=self.app_mag_bins, include_lowest=True)


    def find_nn_properties(self, LOST_GALAXIES_ONLY):
        df = self.df
        
        if LOST_GALAXIES_ONLY:
            catalog = coord.SkyCoord(ra=df.loc[df.observed, 'ra'].to_numpy()*u.degree, dec=df.loc[df.observed, 'dec'].to_numpy()*u.degree, frame='icrs')
            mxxl_halo_id_catalog = df.loc[df.observed, 'halo_id'].to_numpy()
            z_obs_catalog = df.loc[df.observed, 'z_obs'].to_numpy()
            color_catalog = df.loc[df.observed, 'quiescent'].to_numpy()
        else:
            catalog = coord.SkyCoord(ra=df.ra.to_numpy()*u.degree, dec=df.dec.to_numpy()*u.degree, frame='icrs')
            mxxl_halo_id_catalog = df.halo_id.to_numpy()
            z_obs_catalog = df.z_obs.to_numpy()
            color_catalog = df.quiescent.to_numpy()

        if LOST_GALAXIES_ONLY: 
            nthneighbor = 1
            row_locator = np.invert(df['observed'])
        else:
            nthneighbor = 2 # since catalog includes the targets in this case
            row_locator = np.repeat(True)

        self.row_locator = row_locator

        to_match = coord.SkyCoord(ra=df.loc[row_locator,'ra'].to_numpy()*u.degree, dec=df.loc[row_locator,'dec'].to_numpy()*u.degree, frame='icrs')
        idx, d2d, d3d = coord.match_coordinates_sky(to_match, catalog, nthneighbor=nthneighbor, storekdtree=None)
        ang_dist = d2d.to(u.arcsec).value

        sim_z = close_enough(df.loc[row_locator,'z_obs'], z_obs_catalog[idx])
        if self.has_true_halo_id:
            same_halo = df.loc[row_locator,'halo_id'] == mxxl_halo_id_catalog[idx]
            same_halo = np.where(df.loc[row_locator,'halo_id'] == 0, sim_z, df.loc[row_locator,'halo_id'] == mxxl_halo_id_catalog[idx])

        # Properties of the nearest neighbor galaxy we are binning
        nn_z_bin_ind =  np.digitize(z_obs_catalog[idx], self.z_bins)
        #pd.cut(x=z_obs_catalog[idx], bins=self.z_bins, include_lowest=True)
        angdist_bin_ind = np.digitize(ang_dist, self.angular_bins)
        print(angdist_bin_ind)

        df['nn_z_bin'] = np.nan
        df.loc[row_locator,'nn_z_bin'] = nn_z_bin_ind
        df['nn_ang_dist'] = np.nan
        df.loc[row_locator,'nn_ang_dist'] = ang_dist
        df['nn_ang_dist_bin'] = np.nan
        df.loc[row_locator,'nn_ang_dist_bin'] = angdist_bin_ind
        df['nn_quiescent'] = np.nan
        df.loc[row_locator, 'nn_quiescent'] = color_catalog[idx].astype(np.int8)
        df['nn_sim_z'] = np.nan
        df.loc[row_locator, 'nn_sim_z'] = sim_z.astype(np.int8)
        df['nn_same_halo'] = np.nan
        if self.has_true_halo_id:
            df.loc[row_locator, 'nn_same_halo'] = same_halo.astype(np.int8)

        print("Nearest Neighbor properties set")


    def make_bins(self):
        df = self.df
        row_locator = self.row_locator
        # SECOND we count how good NN is in each bin

        # Loop through properties of the lost galaxy
        """
        print("Starting loop...")
        for pobsb in range(len(self.POBS_bins)):
            for nncb in [0,1]: # 0 blue, 1 quiescent
                for tcb in [0,1]:
                    for zb in range(len(self.z_bins)):  
                        for amb in range(len(self.app_mag_bins)):
                            right_bin = np.all([
                                df.loc[row_locator, 'pobs_bin'] == pobsb,
                                df.loc[row_locator, 'nn_quiescent'] == nncb,
                                df.loc[row_locator, 'quiescent'] == tcb,
                                df.loc[row_locator, 'nn_z_bin'] == zb,
                                df.loc[row_locator, 'app_mag_bin'] == amb
                            ], axis=0)
                            
                            bincounts = np.bincount(df.loc[row_locator,'nn_ang_dist_bin'], minlength=len(self.angular_bins), weights=right_bin.astype(int))
                            self.all_ang_bincounts[pobsb][nncb][tcb][zb][amb] = bincounts

                            if self.has_true_halo_id:
                                bincounts2 = np.bincount(df.loc[row_locator,'nn_ang_dist_bin'], minlength=len(self.angular_bins), weights=np.all([df[row_locator, 'nn_same_halo'], right_bin], axis=0).astype(int))
                                self.all_same_halo_bincounts[pobsb][nncb][tcb][zb][amb] = bincounts2

                            bincounts3 = np.bincount(df.loc[row_locator,'nn_ang_dist_bin'], minlength=len(self.angular_bins), weights=np.all([df.loc[row_locator, 'nn_sim_z'], right_bin], axis=0).astype(int))
                            self.all_sim_z_bincounts[pobsb][nncb][tcb][zb][amb] = bincounts3
                      
        print("Loop complete")
        """ 

        ang_dist_bin_type = CategoricalDtype(categories=range(len(self.angular_bins)), ordered=True)

        df['pobs_bin'] = df['pobs_bin'].astype("category")
        df['nn_quiescent'] = df['nn_quiescent'].astype("category")
        df['quiescent'] = df['quiescent'].astype("category")
        df['nn_z_bin'] = df['nn_z_bin'].astype("category")
        df['app_mag_bin'] = df['app_mag_bin'].astype("category")
        df['nn_ang_dist_bin'] = df['nn_ang_dist_bin'].astype(ang_dist_bin_type)

        df_for_agg = df.loc[row_locator]

        all_counts = (df_for_agg.groupby(by=['pobs_bin', 'nn_quiescent', 'quiescent', 'nn_z_bin', 'app_mag_bin', 'nn_ang_dist_bin'], observed=False).dec.count()
                        .unstack(fill_value=0)
                        .stack()
                        .reset_index(name='count'))
        same_halo_counts = (df_for_agg.groupby(by=['pobs_bin', 'nn_quiescent', 'quiescent', 'nn_z_bin', 'app_mag_bin', 'nn_ang_dist_bin'], observed=False).nn_same_halo.sum()
                        .unstack(fill_value=0)
                        .stack()
                        .reset_index(name='count'))
        sim_z_count = (df_for_agg.groupby(by=['pobs_bin', 'nn_quiescent', 'quiescent', 'nn_z_bin', 'app_mag_bin', 'nn_ang_dist_bin'], observed=False).nn_sim_z.sum()
                        .unstack(fill_value=0)
                        .stack()
                        .reset_index(name='count'))
        
        self.all_counts = all_counts
        self.same_halo_counts = same_halo_counts
        self.sim_z_count = sim_z_count

        assert len(all_counts) == len(same_halo_counts)
        assert len(all_counts) == len(sim_z_count)

        self.all_ang_bincounts = all_counts['count'].to_numpy(dtype=np.int64).reshape(len(np.unique(all_counts.pobs_bin)), 2, 2, len(np.unique(all_counts.nn_z_bin)), len(np.unique(all_counts.app_mag_bin)), len(np.unique(all_counts.nn_ang_dist_bin)))
        self.all_same_halo_bincounts = same_halo_counts['count'].to_numpy(dtype=np.int64).reshape(len(np.unique(all_counts.pobs_bin)), 2, 2, len(np.unique(all_counts.nn_z_bin)), len(np.unique(all_counts.app_mag_bin)), len(np.unique(all_counts.nn_ang_dist_bin)))
        self.all_sim_z_bincounts = sim_z_count['count'].to_numpy(dtype=np.int64).reshape(len(np.unique(all_counts.pobs_bin)), 2, 2, len(np.unique(all_counts.nn_z_bin)), len(np.unique(all_counts.app_mag_bin)), len(np.unique(all_counts.nn_ang_dist_bin)))
        
        #self.all_ang_bincounts = np.ones((self.POBS_BIN_COUNT, 2, 2, len(self.z_bins), self.APP_MAG_BIN_COUNT, self.ANG_DIST_BIN_COUNT))
        #self.all_same_halo_bincounts = np.zeros((self.POBS_BIN_COUNT, 2, 2, len(self.z_bins), self.APP_MAG_BIN_COUNT, self.ANG_DIST_BIN_COUNT))
        #self.all_sim_z_bincounts = np.zeros((self.POBS_BIN_COUNT, 2, 2, len(self.z_bins), self.APP_MAG_BIN_COUNT, self.ANG_DIST_BIN_COUNT))

        # Calculate fractions
        # empty bins we call 0% TODO
        np.seterr(divide='ignore')
        if self.has_true_halo_id:
            self.frac_same_halo_full = np.nan_to_num(self.all_same_halo_bincounts / (self.all_ang_bincounts), copy=False) 
        self.frac_sim_z_full = np.nan_to_num(self.all_sim_z_bincounts / (self.all_ang_bincounts), copy=False)
        np.seterr(divide='warn')

        # To visualize things we need to reduce dimensionality. 
        # Aggregate of either Pobs or app mag, or choose a single value from them to examine.
        # The below always picks one z per plot.

        # axis 4 will sum over app mag. Axis 0 will sum over P_obs
        self.axis_to_sumover = 4

        # Use this to aggregate
        if self.has_true_halo_id:
            self.all_same_halo_bincounts_reduced = np.sum(self.all_same_halo_bincounts, axis=self.axis_to_sumover)
        self.all_ang_bincounts_reduced = np.sum(self.all_ang_bincounts, axis=self.axis_to_sumover)
        self.all_sim_z_bincounts_reduced = np.sum(self.all_sim_z_bincounts, axis=self.axis_to_sumover)

        # Use this instead to pick out a value
        index_to_use = 3
        #all_same_halo_bincounts_reduced = np.take(all_same_halo_bincounts_2, index_to_use, axis=axis_to_sumover)
        #all_ang_bincounts_reduced = np.take(all_ang_bincounts_2, index_to_use, axis=axis_to_sumover)
        #all_sim_z_bincounts_reduced = np.take(all_sim_z_bincounts_2, index_to_use, axis=axis_to_sumover)

        if self.axis_to_sumover == 0:

            if self.has_true_halo_id:
                self.all_same_halo_bincounts_reduced = np.swapaxes(self.all_same_halo_bincounts_reduced, 0,1)
                self.all_same_halo_bincounts_reduced = np.swapaxes(self.all_same_halo_bincounts_reduced, 0,2)
            self.all_ang_bincounts_reduced = np.swapaxes(self.all_ang_bincounts_reduced, 0,1)
            self.all_ang_bincounts_reduced = np.swapaxes(self.all_ang_bincounts_reduced, 0,2)
            self.all_sim_z_bincounts_reduced = np.swapaxes(self.all_sim_z_bincounts_reduced, 0,1)
            self.all_sim_z_bincounts_reduced = np.swapaxes(self.all_sim_z_bincounts_reduced, 0,2)

        if self.has_true_halo_id:
            self.frac_same = np.nan_to_num(self.all_same_halo_bincounts_reduced / (self.all_ang_bincounts_reduced), copy=False, nan=0.0)
        self.frac_sim_z = np.nan_to_num(self.all_sim_z_bincounts_reduced / (self.all_ang_bincounts_reduced), copy=False, nan=0.0)

        # Make rough bins of just over a threshold or not
        nn_success_thresh = 0.4 # change fit lines below if you change this!
        success_bins = [0, nn_success_thresh, 1.01]

        if self.has_true_halo_id:
            self.frac_same_binary = np.digitize(self.frac_same, bins=success_bins)
        self.frac_sim_z_binary = np.digitize(self.frac_sim_z, bins=success_bins)
        
        # Resultant shape must be consistent
        print(f"Bincounts complete. Overall shape: {np.shape(self.all_ang_bincounts_reduced)}")


    def color_plots(self, dataset, dataset_binary):

        #if use_sim_z:
        #   dataset = self.frac_sim_z
        #    dataset_binary = self.frac_sim_z_binary
        #else:
        #    dataset = self.frac_same
        #    dataset_binary = self.frac_same_binary

        for nn_color_idx in [0,1]:
            for target_color_idx in [0,1]:
                
                title = get_color_label(nn_color_idx, target_color_idx)
                print(title)

                ncols = 3 # there is code for 4 plots per row (z), but can make a subplot of it
                z_bin_numbers_to_plot = range(dataset.shape[3])
                #z_bin_numbers_to_plot = range(len(self.z_bins))
                #z_bin_numbers_to_plot = [2]

                fig, axes = plt.subplots(nrows=len(z_bin_numbers_to_plot), ncols=ncols, figsize=(6*ncols, 4*len(z_bin_numbers_to_plot)))

                if (self.axis_to_sumover == 4):
                    y_axis_bins = self.POBS_bins
                if (self.axis_to_sumover == 0):
                    y_axis_bins = self.app_mag_bins

                row=-1
                for zb in z_bin_numbers_to_plot:
                    
                    row+=1
                    density = self.all_ang_bincounts_reduced[:,nn_color_idx,target_color_idx,zb,:]
                    #print(f"Galaxies in this z-bin: {np.sum(density)}")

                    if len(z_bin_numbers_to_plot) == 1:
                        axrow = axes
                    else:
                        axrow = axes[row]
                    
                    if (ncols != 1):
                        ax=axrow[0]
                    else:
                        ax = axrow
                    

                    cplot = ax.pcolor(self.angular_bins, y_axis_bins, dataset[:,nn_color_idx,target_color_idx,zb,:], shading='auto', cmap='RdYlGn', norm=c.Normalize(vmin=0, vmax=0.8))
                    fig.colorbar(cplot, ax=ax)
                    ax.set_title(f"NN Same Halo Fraction (NN z {getlabel(zb, self.z_bins)})")
                    if (self.axis_to_sumover == 4):
                        ax.set_ylabel("Lost Galaxy $P_{obs}$")
                    if (self.axis_to_sumover == 0):
                        ax.set_ylabel("Lost Galaxy app r-mag")
                    ax.set_xlabel("Angular Distance (arcsec) to NN")
                    ax.set_xscale('log')
                    
                    
                    cplot = axrow[1].pcolor(self.angular_bins, y_axis_bins, dataset_binary[:,nn_color_idx,target_color_idx,zb,:], shading='auto', cmap='RdYlGn')
                    fig.colorbar(cplot, ax=axrow[1])
                    axrow[1].set_title(f"NN Same Halo Over 40% (NN z {getlabel(zb, self.z_bins)})")
                    if (self.axis_to_sumover == 4):
                        axrow[1].set_ylabel("Lost Galaxy $P_{obs}$")
                    if (self.axis_to_sumover == 0):
                        axrow[1].set_ylabel("Lost Galaxy app r-mag")    
                    axrow[1].set_xlabel("Angular Distance (arcsec) to NN")
                    axrow[1].set_xscale('log')
                    
                    cplot = axrow[2].pcolor(self.angular_bins, y_axis_bins, density, shading='auto', cmap='YlGn', norm=c.LogNorm(vmin=1, vmax=5000))
                    fig.colorbar(cplot, ax=axrow[2])
                    axrow[2].set_title(f"Counts (NN z {getlabel(zb, self.z_bins)})")
                    if (self.axis_to_sumover == 4):
                        axrow[2].set_ylabel("Lost Galaxy $P_{obs}$")
                    if (self.axis_to_sumover == 0):
                        axrow[2].set_ylabel("Lost Galaxy app r-mag")    
                    axrow[2].set_xlabel("Angular Distance (arcsec) to NN")
                    axrow[2].set_xscale('log')
                    
                    if self.axis_to_sumover == 4:
                        if ncols == 1 and len(z_bin_numbers_to_plot) == 1:
                            ax.scatter(get_NN_40_line(self.z_bins[zb]-0.01, self.POBS_bins, target_color_idx, nn_color_idx), self.POBS_bins)
                        else:
                            for i in range(len(axrow)):
                                axrow[i].scatter(get_NN_40_line(self.z_bins[zb]-0.01, self.POBS_bins, target_color_idx, nn_color_idx), self.POBS_bins)
                        

                fig.suptitle(title)
                fig.tight_layout() 