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

class NNAnalyzer():

    def __init__(self, dec, ra, z_obs, app_mag, abs_mag, halo_id, g_r, quiescent, observed, prob_obs):

        self.has_true_halo_id = (halo_id is not None)

        assert len(dec) == len(ra)
        assert len(dec) == len(z_obs)
        assert len(dec) == len(app_mag)
        assert len(dec) == len(abs_mag)
        assert len(dec) == len(g_r)
        assert len(dec) == len(quiescent)
        assert len(dec) == len(observed)
        assert len(dec) == len(prob_obs)

        self.df = pd.DataFrame(data={
            'dec': dec, 
            'ra': ra,
            'z_obs': z_obs,
            'app_mag': app_mag,
            'abs_mag': abs_mag,
            'g_r': g_r,
            'quiescent': quiescent,
            'observed': observed,
            'prob_obs': prob_obs
        })
        if self.has_true_halo_id:
            assert len(dec) == len(halo_id)
            self.df['halo_id'] = halo_id

        assert not np.any(np.isnan(z_obs)), "Some z_obs are nan; need to update code to handle data where some z are unknown"
        assert not np.any(np.isnan(abs_mag)), "Some abs_mag are nan; need to update code to handle data where some z are unknown"

        # Now bin so that things with ang distances higher than the max we care about are thrown out
        print("Angular Distance Bin Markers", ANGULAR_BINS)
        # Must determine NN and distance to them before binning this

        print("Redshift Bin Markers", Z_BINS)
        # Must determine NN and distance to them before binning this

        print("Abs mag Bin Markers", ABS_MAG_BINS)
        # Must determine NN and distance to them before binning this

        print("Pobs Bin Markers", POBS_BINS)
        self.df['pobs_bin'] = np.digitize(self.df['prob_obs'], POBS_BINS)
        #pd.cut(x=self.df['prob_obs'], bins=POBS_BINS, include_lowest=True)

        print("App mag bin markers", APP_MAG_BINS)
        self.df['app_mag_bin'] = np.digitize(self.df['app_mag'], APP_MAG_BINS)
        #pd.cut(x=self.df['app_mag'], bins=APP_MAG_BINS, include_lowest=True)

    def find_nn_properties(self, LOST_GALAXIES_ONLY, BRIGHT_ONLY=True):
        df = self.df
        
        if LOST_GALAXIES_ONLY:
            catalog = coord.SkyCoord(ra=df.loc[df.observed, 'ra'].to_numpy()*u.degree, dec=df.loc[df.observed, 'dec'].to_numpy()*u.degree, frame='icrs')
            mxxl_halo_id_catalog = df.loc[df.observed, 'halo_id'].to_numpy()
            z_obs_catalog = df.loc[df.observed, 'z_obs'].to_numpy()
            color_catalog = df.loc[df.observed, 'quiescent'].to_numpy()
            abs_mag_catalog = df.abs_mag.to_numpy()

        else:
            catalog = coord.SkyCoord(ra=df.ra.to_numpy()*u.degree, dec=df.dec.to_numpy()*u.degree, frame='icrs')
            mxxl_halo_id_catalog = df.halo_id.to_numpy()
            z_obs_catalog = df.z_obs.to_numpy()
            color_catalog = df.quiescent.to_numpy()
            abs_mag_catalog = df.abs_mag.to_numpy()

        if LOST_GALAXIES_ONLY: 
            nthneighbor = 1
            row_locator = np.invert(df['observed'])
        else:
            nthneighbor = 2 # since catalog includes the targets in this case
            row_locator = np.repeat(True, len(df))

        if BRIGHT_ONLY:
            row_locator = np.logical_and(row_locator, df['app_mag'] < 19.5)

        self.row_locator = row_locator

        to_match = coord.SkyCoord(ra=df.loc[row_locator,'ra'].to_numpy()*u.degree, dec=df.loc[row_locator,'dec'].to_numpy()*u.degree, frame='icrs')
        idx, d2d, d3d = coord.match_coordinates_sky(to_match, catalog, nthneighbor=nthneighbor, storekdtree=None)
        ang_dist = d2d.to(u.arcsec).value

        sim_z = sim_z_score(df.loc[row_locator,'z_obs'], z_obs_catalog[idx])
        if self.has_true_halo_id:
            same_halo = df.loc[row_locator,'halo_id'] == mxxl_halo_id_catalog[idx]
            # If no halo is exists then use sim z score
            same_halo = np.where(df.loc[row_locator,'halo_id'] == 0, sim_z > 0.8, df.loc[row_locator,'halo_id'] == mxxl_halo_id_catalog[idx])

        # Properties of the nearest neighbor galaxy we are binning
        nn_z_bin_ind =  np.digitize(z_obs_catalog[idx], Z_BINS)
        angdist_bin_ind = np.digitize(ang_dist, ANGULAR_BINS)
        nn_abs_mag_bin_in = np.digitize(abs_mag_catalog[idx], ABS_MAG_BINS)

        df['nn_z_bin'] = np.nan
        df.loc[row_locator,'nn_z_bin'] = nn_z_bin_ind
        df['nn_ang_dist'] = np.nan
        df.loc[row_locator,'nn_ang_dist'] = ang_dist
        df['nn_ang_dist_bin'] = np.nan
        df.loc[row_locator,'nn_ang_dist_bin'] = angdist_bin_ind
        df['nn_quiescent'] = np.nan
        df.loc[row_locator, 'nn_quiescent'] = color_catalog[idx].astype(bool)
        df['nn_sim_z'] = np.nan
        df.loc[row_locator, 'nn_sim_z'] = sim_z
        df['nn_same_halo'] = np.nan
        if self.has_true_halo_id:
            df.loc[row_locator, 'nn_same_halo'] = same_halo.astype(np.int8)
        df['nn_abs_mag_bin'] = np.nan
        df.loc[row_locator, 'nn_abs_mag_bin'] = nn_abs_mag_bin_in
        
        print("Nearest Neighbor properties set")


    def make_bins(self):
        df = self.df
        row_locator = self.row_locator

        ang_dist_bin_type = CategoricalDtype(categories=range(len(ANGULAR_BINS)), ordered=True)

        df['pobs_bin'] = df['pobs_bin'].astype("category")
        df['nn_quiescent'] = df['nn_quiescent'].astype("category")
        df['quiescent'] = df['quiescent'].astype("category")
        df['nn_z_bin'] = df['nn_z_bin'].astype("category")
        df['app_mag_bin'] = df['app_mag_bin'].astype("category")
        df['nn_ang_dist_bin'] = df['nn_ang_dist_bin'].astype(ang_dist_bin_type)
        df['nn_abs_mag_bin'] = df['nn_abs_mag_bin'].astype("category")

        df_for_agg = df.loc[row_locator]

        all_counts = (df_for_agg.groupby(by=['pobs_bin', 'nn_quiescent', 'quiescent', 'nn_z_bin', 'app_mag_bin', 'nn_ang_dist_bin', 'nn_abs_mag_bin'], observed=False).dec.count()
                        .unstack(fill_value=0)
                        .stack()
                        .reset_index(name='count'))
        same_halo_counts = (df_for_agg.groupby(by=['pobs_bin', 'nn_quiescent', 'quiescent', 'nn_z_bin', 'app_mag_bin', 'nn_ang_dist_bin', 'nn_abs_mag_bin'], observed=False).nn_same_halo.sum()
                        .unstack(fill_value=0)
                        .stack()
                        .reset_index(name='count'))
        sim_z_count = (df_for_agg.groupby(by=['pobs_bin', 'nn_quiescent', 'quiescent', 'nn_z_bin', 'app_mag_bin', 'nn_ang_dist_bin', 'nn_abs_mag_bin'], observed=False).nn_sim_z.sum()
                        .unstack(fill_value=0)
                        .stack()
                        .reset_index(name='count'))
        
        # Default agg func is mean
        self.pt = df_for_agg.pivot_table(columns=['pobs_bin', 'nn_quiescent', 'quiescent', 'nn_z_bin', 'app_mag_bin', 'nn_ang_dist_bin', 'nn_abs_mag_bin'], values=['nn_sim_z', 'nn_same_halo'], observed=False, margins=True, fill_value=0)
        
        # TODO add some assertions to check manually that this does what we expect

        self.all_counts = all_counts
        self.same_halo_counts = same_halo_counts
        self.sim_z_count = sim_z_count

        assert len(all_counts) == len(same_halo_counts)
        assert len(all_counts) == len(sim_z_count)

        self.all_ang_bincounts = all_counts['count'].to_numpy(dtype=np.int64).reshape(len(np.unique(all_counts.pobs_bin)), 2, 2, len(np.unique(all_counts.nn_z_bin)), len(np.unique(all_counts.app_mag_bin)), len(np.unique(all_counts.nn_ang_dist_bin)), len(np.unique(all_counts.nn_abs_mag_bin)))
        self.all_same_halo_bincounts = same_halo_counts['count'].to_numpy(dtype=np.int64).reshape(len(np.unique(all_counts.pobs_bin)), 2, 2, len(np.unique(all_counts.nn_z_bin)), len(np.unique(all_counts.app_mag_bin)), len(np.unique(all_counts.nn_ang_dist_bin)), len(np.unique(all_counts.nn_abs_mag_bin)))
        self.all_sim_z_bincounts = sim_z_count['count'].to_numpy(dtype=np.int64).reshape(len(np.unique(all_counts.pobs_bin)), 2, 2, len(np.unique(all_counts.nn_z_bin)), len(np.unique(all_counts.app_mag_bin)), len(np.unique(all_counts.nn_ang_dist_bin)), len(np.unique(all_counts.nn_abs_mag_bin)))
        
        # Calculate fractions
        # empty bins we call 0% TODO
        if self.has_true_halo_id:
            self.frac_same_halo_full = np.nan_to_num(self.all_same_halo_bincounts / (self.all_ang_bincounts), copy=False) 
        self.frac_sim_z_full = np.nan_to_num(self.all_sim_z_bincounts / (self.all_ang_bincounts), copy=False)
        
        # Resultant shape must be consistent
        print(f"Bincounts complete. Overall shape: {np.shape(self.all_ang_bincounts)}")

    def binary_split(self, data):
         # Make rough bins of just over a threshold or not
        nn_success_thresh = 0.4 # change fit lines below if you change this!
        success_bins = [0, nn_success_thresh, 1.01]
        return np.digitize(data, bins=success_bins)
    
    def integrate_out_dimension(self, axis, use_same_halo_instead=False):
        # pivot table could make this easier

        all_counts = np.sum(self.all_ang_bincounts, axis=axis)

        if use_same_halo_instead and self.has_true_halo_id:
            same_counts = np.sum(self.all_same_halo_bincounts, axis=axis)
            frac = np.nan_to_num(same_counts / (all_counts), copy=False, nan=0.0)
        else:
            same_counts = np.sum(self.all_sim_z_bincounts, axis=axis)
            frac = np.nan_to_num(same_counts / (all_counts), copy=False, nan=0.0)

        print(f"Integrated out dimension {axis}. New shape: {np.shape(all_counts)}")
        return frac, same_counts, all_counts


    def plot_angdist_absmag_per_zbin_cc(self, use_same_halo_instead=False):
        frac, same_counts, all_counts = self.integrate_out_dimension((0,4), use_same_halo_instead=use_same_halo_instead) # (2, 2, 8, 20, 16)
        print(frac.shape)
        
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
                    
                    cplot = axrow[0].pcolor(ANGULAR_BINS_MIDPOINTS, ABS_MAG_MIDPOINTS, np.swapaxes(frac[nn_color_idx,target_color_idx,zb,:,:], 0, 1), shading='auto', cmap='RdYlGn', norm=c.Normalize(vmin=0, vmax=0.8))
                    fig.colorbar(cplot, ax=axrow[0])
                    axrow[0].set_title(f"NN {phrase} Fraction (NN z {getlabel(zb, Z_BINS)})")
                    axrow[0].set_ylabel("NN abs R-mag")
                    axrow[0].set_xlabel("Angular Distance (arcsec) to NN")
                    axrow[0].set_xscale('log')
                    axrow[0].set_xlim(2.0, 1000)
                    
                    cplot = axrow[1].pcolor(ANGULAR_BINS_MIDPOINTS, ABS_MAG_MIDPOINTS, np.swapaxes(self.binary_split(frac)[nn_color_idx,target_color_idx,zb,:,:], 0, 1), shading='auto', cmap='RdYlGn')
                    fig.colorbar(cplot, ax=axrow[1])
                    axrow[1].set_title(f"NN {phrase} Over 40% (NN z {getlabel(zb, Z_BINS)})")
                    axrow[1].set_ylabel("NN abs R-mag")    
                    axrow[1].set_xlabel("Angular Distance (arcsec) to NN")
                    axrow[1].set_xscale('log')
                    axrow[1].set_xlim(2.0, 1000)
                    
                    cplot = axrow[2].pcolor(ANGULAR_BINS_MIDPOINTS, ABS_MAG_MIDPOINTS, np.swapaxes(all_counts[nn_color_idx,target_color_idx,zb,:,:],0,1), shading='auto', cmap='YlGn', norm=c.LogNorm(vmin=1, vmax=5000))
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

    def plot_angdist_appmag_per_zbin_cc(self, use_same_halo_instead=False):
        if not hasattr(self, 'frac_aa'):
            self.frac_aa, same_counts, self.all_counts = self.integrate_out_dimension((0,6), use_same_halo_instead=use_same_halo_instead) # (2, 2, 8, 10, 20)
        print(self.frac_aa.shape)
        
        for nn_quiescent in [0,1]:
            for target_quiescent in [0,1]:
                title = get_color_label(nn_quiescent, target_quiescent)
                print(title)
                z_bin_numbers_to_plot = range(self.frac_aa.shape[2])

                fig, axes = plt.subplots(nrows=len(z_bin_numbers_to_plot), ncols=3, figsize=(6*3, 4*len(z_bin_numbers_to_plot)))

                row=-1
                for zb in z_bin_numbers_to_plot:
                    row+=1

                    if len(z_bin_numbers_to_plot) == 1:
                        axrow = axes
                    else:
                        axrow = axes[row]
                    phrase = 'Similar Z'
                    
                    cplot = axrow[0].pcolor(ANGULAR_BINS_MIDPOINTS, APP_MAG_BINS_MIDPOINTS, self.frac_aa[nn_quiescent,target_quiescent,zb,:,:], shading='auto', cmap='RdYlGn', norm=c.Normalize(vmin=0, vmax=0.8))
                    fig.colorbar(cplot, ax=axrow[0])
                    axrow[0].set_title(f"NN {phrase} Fraction (NN z {getlabel(zb, Z_BINS)})")
                    axrow[0].set_ylabel("Lost Galaxy app r-mag")
                    axrow[0].set_xlabel("Angular Distance (arcsec) to NN")
                    axrow[0].set_xscale('log')
                    axrow[0].set_xlim(2.0, 1000)

                    cplot = axrow[1].pcolor(ANGULAR_BINS_MIDPOINTS, APP_MAG_BINS_MIDPOINTS, self.binary_split(self.frac_aa)[nn_quiescent,target_quiescent,zb,:,:], shading='auto', cmap='RdYlGn')
                    fig.colorbar(cplot, ax=axrow[1])
                    axrow[1].set_title(f"NN {phrase} Over 40% (NN z {getlabel(zb, Z_BINS)})")
                    axrow[1].set_ylabel("Lost Galaxy app r-mag")    
                    axrow[1].set_xlabel("Angular Distance (arcsec) to NN")
                    axrow[1].set_xscale('log')
                    axrow[1].set_xlim(2.0, 1000)

                    cplot = axrow[2].pcolor(ANGULAR_BINS_MIDPOINTS, APP_MAG_BINS_MIDPOINTS, self.all_counts[nn_quiescent,target_quiescent,zb,:,:], shading='auto', cmap='YlGn', norm=c.LogNorm(vmin=1, vmax=5000))
                    fig.colorbar(cplot, ax=axrow[2])
                    axrow[2].set_title(f"Counts (NN z {getlabel(zb, Z_BINS)})")
                    axrow[2].set_ylabel("Lost Galaxy app r-mag")    
                    axrow[2].set_xlabel("Angular Distance (arcsec) to NN")
                    axrow[2].set_xscale('log')
                    axrow[2].set_xlim(2.0, 1000)
                    
                    for i in range(len(axrow)):
                        mags = np.linspace(np.min(APP_MAG_BINS_MIDPOINTS), np.max(APP_MAG_BINS_MIDPOINTS), 40)
                        axrow[i].scatter(get_NN_40_line_v5(np.repeat(Z_BINS_MIDPOINTS[zb], len(mags)), mags, target_quiescent, nn_quiescent), mags)
                        
                fig.suptitle(title)
                fig.tight_layout() 

    def plot_angdist_pobs_per_zbin_cc(self, use_same_halo_instead=False):
        dataset, sim_z_counts, all_counts = self.integrate_out_dimension((4,6), use_same_halo_instead=use_same_halo_instead)  # (15, 2, 2, 8, 20)

        for nn_quiescent in [0,1]:
            for target_quiescent in [0,1]:
                title = get_color_label(nn_quiescent, target_quiescent)
                print(title)
                z_bin_numbers_to_plot = range(dataset.shape[3])

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
                    if use_same_halo_instead:
                        phrase = 'Same Halo'

                    cplot = ax.pcolor(ANGULAR_BINS_MIDPOINTS, POBS_BINS_MIDPOINTS, dataset[:,nn_quiescent,target_quiescent,zb,:], shading='auto', cmap='RdYlGn', norm=c.Normalize(vmin=0, vmax=0.8))
                    fig.colorbar(cplot, ax=ax)
                    ax.set_title(f"NN {phrase} Fraction (NN z {getlabel(zb, Z_BINS)})")
                    ax.set_ylabel("Lost Galaxy $P_{obs}$")
                    ax.set_xlabel("Angular Distance (arcsec) to NN")
                    ax.set_xscale('log')
                    ax.set_xlim(2.0, 1000)
                    
                    cplot = axrow[1].pcolor(ANGULAR_BINS_MIDPOINTS, POBS_BINS_MIDPOINTS, self.binary_split(dataset)[:,nn_quiescent,target_quiescent,zb,:], shading='auto', cmap='RdYlGn')
                    fig.colorbar(cplot, ax=axrow[1])
                    axrow[1].set_title(f"NN {phrase} Over 40% (NN z {getlabel(zb, Z_BINS)})")
                    axrow[1].set_ylabel("Lost Galaxy $P_{obs}$")
                    axrow[1].set_xlabel("Angular Distance (arcsec) to NN")
                    axrow[1].set_xscale('log')
                    axrow[1].set_xlim(2.0, 1000)
                    
                    cplot = axrow[2].pcolor(ANGULAR_BINS_MIDPOINTS, POBS_BINS_MIDPOINTS, all_counts[:,nn_quiescent,target_quiescent,zb,:], shading='auto', cmap='YlGn', norm=c.LogNorm(vmin=1, vmax=5000))
                    fig.colorbar(cplot, ax=axrow[2])
                    axrow[2].set_title(f"Counts (NN z {getlabel(zb, Z_BINS)})")
                    axrow[2].set_ylabel("Lost Galaxy $P_{obs}$")
                    axrow[2].set_xlabel("Angular Distance (arcsec) to NN")
                    axrow[2].set_xscale('log')
                    axrow[2].set_xlim(2.0, 1000)

                    for i in range(len(axrow)):
                        pobs = np.linspace(np.min(POBS_BINS_MIDPOINTS), np.max(POBS_BINS_MIDPOINTS), 40)
                        axrow[i].scatter(get_NN_40_line_v4(np.repeat(Z_BINS_MIDPOINTS[zb], len(pobs)), pobs, target_quiescent, nn_quiescent), pobs)
                        

                fig.suptitle(title)
                fig.tight_layout() 