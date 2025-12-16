import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as u
from matplotlib.patches import Circle
import nnanalysis as nn
import sys
if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from pyutils import *
from groupcatalog import *
from calibrationdata import *
import hod

# np.array(zip(*[line.split() for line in f])[1], dtype=float)

DPI_PAPER = 600
DPI = 250
FONT_SIZE_DEFAULT = 16

LGAL_XMINS = [1E8]

LGAL_MIN = 1E7
LGAL_MAX_TIGHT = 2E11
LOG_LGAL_MAX_TIGHT = np.log10(LGAL_MAX_TIGHT)
LGAL_MAX = 4E11
LOG_LGAL_MAX = np.log10(LGAL_MAX)

MSTAR_MIN = 1E7
MSTAR_MAX = 1E12

MHALO_MIN = 1E11
MHALO_MAX = 1E15

#plt.style.use('default')
plt.rcParams.update({'font.size': FONT_SIZE_DEFAULT})

def font_restore():
    plt.rcParams.update({'font.size': FONT_SIZE_DEFAULT})

def completeness_stats(cats: GroupCatalog|list[GroupCatalog]):
    if isinstance(cats, GroupCatalog):
        cats = [cats]
    for d in cats:
        name = d.name.replace("BGS sv3", "SV3")
        print(f"{name}")
        print(f"  Total galaxies: {len(d.all_data):,}")
        print(f"  Completeness: {spectroscopic_complete_percent(d.all_data['Z_ASSIGNED_FLAG'].to_numpy()):.1%}")
        print(f"  Lost gals - neighbor z used: {d.get_lostgal_neighbor_used():.1%}")
        total_f_sat(d)


##########################
# Plots
##########################
LEGENDS_ON = True
def legend(datasets):
    if len(datasets) > 1 and LEGENDS_ON:
        plt.legend()

def legend_ax(ax, datasets):
    if len(datasets) > 1 and LEGENDS_ON:
        ax.legend()


def get_dataset_display_name(d: GroupCatalog, keep_mag_limit=False):
    name = d.name.replace("Fiber Only", "Observed").replace(" Vanilla", "")
    if keep_mag_limit:
        return name
    else:
        return name.replace(" <19.5", "").replace(" <20.0", "")

def single_plots(d: GroupCatalog, truth_on=False):
    """
    Plots that are nice for just 1 group catalog at a time.
    """

    # LHMR Inverted
    plt.figure(dpi=DPI)
    means = np.log10(d.centrals[d.centrals['QUIESCENT']].groupby('LGAL_BIN', observed=False).apply(Mhalo_vmax_weighted))
    scatter = d.centrals.loc[d.centrals['QUIESCENT']].groupby('LGAL_BIN', observed=False).apply(Mhalo_std_vmax_weighted)
    plt.errorbar(np.log10(d.L_gal_labels), means, yerr=scatter, label=get_dataset_display_name(d), color='red', elinewidth=1, alpha=0.6)
    means = np.log10(d.centrals[~d.centrals['QUIESCENT']].groupby('LGAL_BIN', observed=False).apply(Mhalo_vmax_weighted))
    scatter = d.centrals.loc[~d.centrals['QUIESCENT']].groupby('LGAL_BIN', observed=False).apply(Mhalo_std_vmax_weighted)
    plt.errorbar(np.log10(d.L_gal_labels), means, yerr=scatter, label=get_dataset_display_name(d), color='blue', elinewidth=1, alpha=0.6)
    plt.ylabel('(log$(M_h)~[M_\\odot / h]$')
    plt.xlabel('log$(L_{cen})~[L_\odot / h^2]$')
    plt.title("Luminosity Halo Mass Relation (Centrals)")
    plt.ylim(10,15)
    plt.xlim(7,LOG_LGAL_MAX_TIGHT)
    plt.grid(True)
    plt.twiny()
    plt.xlim(log_solar_L_to_abs_mag_r(7), log_solar_L_to_abs_mag_r(LOG_LGAL_MAX_TIGHT))
    plt.xticks(np.arange(-23, -13, 1))
    plt.xlabel("$M_r$ - 5log(h)")
    plt.draw()

    # LHMR
    plt.figure(dpi=DPI)
    means = d.centrals[d.centrals['QUIESCENT']].groupby('Mh_bin', observed=False).apply(LogLgal_vmax_weighted)
    scatter = d.centrals.loc[d.centrals['QUIESCENT']].groupby('Mh_bin', observed=False).apply(LogLgal_lognormal_scatter_vmax_weighted)
    plt.errorbar(np.log10(d.labels), means, yerr=scatter, label=get_dataset_display_name(d), color='red', elinewidth=1)
    means = d.centrals[~d.centrals['QUIESCENT']].groupby('Mh_bin', observed=False).apply(LogLgal_vmax_weighted)
    scatter = d.centrals.loc[~d.centrals['QUIESCENT']].groupby('Mh_bin', observed=False).apply(LogLgal_lognormal_scatter_vmax_weighted)
    plt.errorbar(np.log10(d.labels), means, yerr=scatter, label=get_dataset_display_name(d), color='blue', elinewidth=1, alpha=0.6)
    plt.xlabel('(log$(M_h)~[M_\\odot / h]$')
    plt.ylabel('log$(L_{cen})~[L_\odot / h^2]$')
    plt.title("Luminosity Halo Mass Relation (Centrals, With Scatter)")
    plt.xlim(10,15)
    plt.ylim(7,np.log10(LGAL_MAX_TIGHT))
    plt.grid(True)
    plt.twinx()
    plt.ylim(log_solar_L_to_abs_mag_r(7), log_solar_L_to_abs_mag_r(np.log10(LGAL_MAX_TIGHT)))
    plt.yticks(np.arange(-23, -13, 1))
    plt.ylabel("$M_r$ - 5log(h)")
    plt.draw()
    

    plots_color_split(d, truth_on=truth_on, total_on=True)

    if 'Z_ASSIGNED_FLAG' in d.all_data.columns:
        plots_color_split_lost_split(d, 'LGAL_BIN')
    if d.has_truth and truth_on:
        q_gals = d.all_data[d.all_data['QUIESCENT']]
        sf_gals = d.all_data[np.invert(d.all_data['QUIESCENT'])]
        plots_color_split_lost_split_inner(d.name + " Truth", d.L_gal_labels, d.L_gal_bins, q_gals, sf_gals, fsat_truth_vmax_weighted,'LGAL_BIN_T')

    hod_bins_plot(d)

    #wp_rp(d)
    #wp_rp_magbins(d)

def completeness_comparison(*datasets):        
    fig,ax1=plt.subplots()
    fig.set_dpi(DPI)

    indexes = [9,12,15,18,21,24,27,30]
    for index in indexes:
        completeness = np.zeros(len(datasets))
        fsat = np.zeros(len(datasets))
        for i in range(0,len(datasets)):
            completeness[i] = (datasets[i].all_data['Z_ASSIGNED_FLAG'] == 0).mean()
            fsat[i] = datasets[i].f_sat.iloc[index]
        plt.plot(completeness, fsat, label=f'log(L)={np.log10(L_gal_labels[index]):.1f}', marker='o')
    ax1.set_xlabel("Completeness")
    ax1.set_ylabel("$f_{sat}$")
    #ax1.legend()
    ax1.set_ylim(0.0,0.6)
    fig.tight_layout()

def LHMR_withscatter(*catalogs):


    # LHMR
    plt.figure(dpi=DPI)
    for f in catalogs:        
        means = f.centrals.groupby('Mh_bin', observed=False).apply(LogLgal_vmax_weighted)
        scatter = f.centrals.groupby('Mh_bin', observed=False).apply(LogLgal_lognormal_scatter_vmax_weighted)
        plt.errorbar(np.log10(f.labels), means, yerr=scatter, label=get_dataset_display_name(f), color=f.color, elinewidth=1)
    plt.xlabel('(log$(M_h)~[M_\\odot h^{-1}])$')
    plt.ylabel('log$(L_{cen})~[L_\odot / h^2]$')
    plt.title("Central Luminosity vs. Halo Mass")
    legend(catalogs)
    plt.xlim(10,15)
    plt.ylim(7,LOG_LGAL_MAX_TIGHT)
    plt.draw()

    # RED LHMR
    plt.figure(dpi=DPI)
    for f in catalogs:  
        means = f.centrals[f.centrals['QUIESCENT']].groupby('Mh_bin', observed=False).apply(LogLgal_vmax_weighted)
        scatter = f.centrals.loc[f.centrals['QUIESCENT']].groupby('Mh_bin', observed=False).apply(LogLgal_lognormal_scatter_vmax_weighted)
        plt.errorbar(np.log10(f.labels), means, yerr=scatter, label=get_dataset_display_name(f), color=f.color, elinewidth=1)
    plt.xlabel('(log$(M_h)~[M_\\odot]$')
    plt.ylabel('log$(L_{cen})~[L_\odot / h^2]$')
    plt.title("Red Central Luminosity vs. Halo Mass")
    legend(catalogs)
    plt.xlim(10,15)
    plt.ylim(7,LOG_LGAL_MAX_TIGHT)
    plt.draw()

    # BLUE LHMR
    plt.figure(dpi=DPI)
    for f in catalogs:
        means = f.centrals[~f.centrals['QUIESCENT']].groupby('Mh_bin', observed=False).apply(LogLgal_vmax_weighted)
        scatter = f.centrals.loc[~f.centrals['QUIESCENT']].groupby('Mh_bin', observed=False).apply(LogLgal_lognormal_scatter_vmax_weighted)
        plt.errorbar(np.log10(f.labels), means, yerr=scatter, label=get_dataset_display_name(f), color=f.color, elinewidth=1)
    plt.xlabel('(log$(M_h)~[M_\\odot]$')
    plt.ylabel('log$(L_{cen})~[L_\odot / h^2]$')
    plt.title("Blue Central Luminosity vs. Halo Mass")
    legend(catalogs)
    plt.xlim(10,15)
    plt.ylim(7,np.log10(LGAL_MAX_TIGHT))
    plt.draw()

    # RED LHMR Inverted
    plt.figure(dpi=DPI)
    for f in catalogs:
        means = np.log10(f.centrals[f.centrals['QUIESCENT']].groupby('LGAL_BIN', observed=False).apply(Mhalo_vmax_weighted))
        scatter = f.centrals.loc[f.centrals['QUIESCENT']].groupby('LGAL_BIN', observed=False).apply(Mhalo_std_vmax_weighted)
        plt.errorbar(np.log10(f.L_gal_labels), means, yerr=scatter, label=get_dataset_display_name(f), color=f.color, elinewidth=1)
    plt.ylabel('(log$(M_h)~[M_\\odot]$')
    plt.xlabel('log$(L_{cen})~[L_\odot / h^2]$')
    plt.title("Red Central Luminosity vs. Halo Mass")
    legend(catalogs)
    plt.ylim(10,15)
    plt.xlim(7,LOG_LGAL_MAX_TIGHT)
    plt.draw()

    # BLUE LHMR Inverted
    plt.figure(dpi=DPI)
    for f in catalogs:
        means = np.log10(f.centrals[~f.centrals['QUIESCENT']].groupby('LGAL_BIN', observed=False).apply(Mhalo_vmax_weighted))
        scatter = f.centrals.loc[~f.centrals['QUIESCENT']].groupby('LGAL_BIN', observed=False).apply(Mhalo_std_vmax_weighted)
        plt.errorbar(np.log10(f.L_gal_labels), means, yerr=scatter, label=get_dataset_display_name(f), color=f.color, elinewidth=1)
    plt.ylabel('(log$(M_h)~[M_\\odot]$')
    plt.xlabel('log$(L_{cen})~[L_\odot / h^2]$')
    plt.title("Blue Central Luminosity vs. Halo Mass")
    legend(catalogs)
    plt.ylim(10,15)
    plt.xlim(7,LOG_LGAL_MAX_TIGHT)
    plt.draw()

def LHMR_from_logs():
    lhmr_r_mean, lhmr_r_std, lhmr_r_scatter_mean, lhmr_r_scatter_std, lhmr_b_mean, lhmr_b_std, lhmr_b_scatter_mean, lhmr_b_scatter_std, lhmr_all_mean, lhmr_all_std, lhmr_all_scatter_mean, lhmr_all_scatter_std = lhmr_variance_from_saved()

    plt.figure()
    plt.errorbar(Mhalo_labels, lhmr_all_mean, yerr=lhmr_all_std, fmt='.', color='k', label='All', capsize=3, alpha=0.7)
    plt.errorbar(Mhalo_labels, lhmr_b_mean, yerr=lhmr_b_std, fmt='.', color='b', label='Star-forming', capsize=3, alpha=0.7)
    plt.errorbar(Mhalo_labels, lhmr_r_mean, yerr=lhmr_r_std, fmt='.', color='r', label='Quiescent', capsize=3, alpha=0.7)
    plt.xlabel('log$(M_h~[M_\\odot]$')
    plt.ylabel(r'$\langle L_{\mathrm{cen}} \rangle$')
    plt.title("Mean Central Luminosity vs. Halo Mass")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1E10, 1E15)
    plt.ylim(1E7, 5E11)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.errorbar(Mhalo_labels, lhmr_all_scatter_mean, yerr=lhmr_all_scatter_std, fmt='.', color='k', label='All', capsize=3, alpha=0.7)
    plt.errorbar(Mhalo_labels, lhmr_b_scatter_mean, yerr=lhmr_b_scatter_std, fmt='.', color='b', label='Star-forming', capsize=3, alpha=0.7)
    plt.errorbar(Mhalo_labels, lhmr_r_scatter_mean, yerr=lhmr_r_scatter_std, fmt='.', color='r', label='Quiescent', capsize=3, alpha=0.7)
    plt.xlabel('log$(M_h~[M_\\odot]$')
    plt.ylabel(r'$\sigma_{{\mathrm{log}}(L_{\mathrm{cen}})}~$[dex]')
    plt.title("Central Luminosity Scatter vs. Halo Mass")
    plt.legend()
    plt.xscale('log')
    plt.xlim(1E10, 1E15)
    plt.tight_layout()
    plt.show()

# This should reproduce what plt does when you say .yscale('log') with errors.
def safe_log_err(logmean, lin_err):
    with np.errstate(divide='ignore', invalid='ignore'):
        # If lin_err is a single value:
        if np.isscalar(lin_err):
            lower = np.where(10**logmean - lin_err > 0, logmean - np.log10(10**logmean - lin_err), logmean)
            upper = np.log10(10**logmean + lin_err) - logmean
            return lower, upper
        else:
            assert len(lin_err) == 2 # Then it's the 16th and 84th percentiles
            # Convert to distance from mean in logspace
            lower = logmean - np.log10(lin_err[0])
            upper = np.log10(lin_err[1]) - logmean
            return lower, upper

def safe_log_err_vals(logmean, lin_err):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isscalar(lin_err):
            lower = np.where(10**logmean - lin_err > 0, np.log10(10**logmean - lin_err), -99)
            upper = np.log10(10**logmean + lin_err)
            return lower, upper
        else:
            assert len(lin_err) == 2 # Then it's the 16th and 84th percentiles
            lower = np.log10(lin_err[0])
            upper = np.log10(lin_err[1])
            return lower, upper

def LHMR_savederr(f: GroupCatalog, show_all=False, inset: GroupCatalog=None):

    log_mean_all = f.centrals.groupby('Mh_bin', observed=False).apply(LogLgal_vmax_weighted)
    log_means_r = f.centrals.loc[f.centrals['QUIESCENT']].groupby('Mh_bin', observed=False).apply(LogLgal_vmax_weighted)
    log_means_b = f.centrals.loc[~f.centrals['QUIESCENT']].groupby('Mh_bin', observed=False).apply(LogLgal_vmax_weighted)
    ratio = 10**log_means_r / 10**log_means_b

    lhmr_r_mean, lhmr_r_err, lhmr_r_scatter_mean, lhmr_r_scatter_err, lhmr_b_mean, lhmr_b_err, lhmr_b_scatter_mean, lhmr_b_scatter_err, lhmr_all_mean, lhmr_all_err, lhmr_all_scatter_mean, lhmr_all_scatter_err = lhmr_variance_from_saved()

    # Error bars for statistical error (bootstrapped)
    # And shaded for systematic (saved from MCMC)
    yerr_all_lower, yerr_all_upper = safe_log_err(log_mean_all, f.lhmr_bootstrap_err)
    yerr_b_lower, yerr_b_upper = safe_log_err(log_means_b, f.lhmr_sf_bootstrap_err)
    yerr_r_lower, yerr_r_upper = safe_log_err(log_means_r, f.lhmr_q_bootstrap_err)

    syserr_all_lower, syserr_all_upper = safe_log_err_vals(log_mean_all, lhmr_all_err)
    syserr_b_lower, syserr_b_upper = safe_log_err_vals(log_means_b, lhmr_b_err)
    syserr_r_lower, syserr_r_upper = safe_log_err_vals(log_means_r, lhmr_r_err)

    plt.figure(dpi=DPI)
    x_vals = np.log10(Mhalo_labels)
    x_vals2 = np.log10(Mhalo_labels2)

    if show_all:
        plt.errorbar(x_vals, log_mean_all, yerr=[yerr_all_lower, yerr_all_upper], fmt='.', color='k', label='All', capsize=4, alpha=0.7)
    plt.errorbar(x_vals, log_means_b, yerr=[yerr_b_lower, yerr_b_upper], fmt='.', color='b', label='SF Centrals', capsize=4, alpha=0.7)
    plt.errorbar(x_vals, log_means_r, yerr=[yerr_r_lower, yerr_r_upper], fmt='.', color='r', label='Q Centrals', capsize=4, alpha=0.7)

    if show_all:
        plt.fill_between(x_vals, syserr_all_lower, syserr_all_upper, color='k', alpha=0.2)
    plt.fill_between(x_vals, syserr_r_lower, syserr_r_upper, color='r', alpha=0.2)
    plt.fill_between(x_vals, syserr_b_lower, syserr_b_upper, color='b', alpha=0.2)
    
    plt.xlabel('log$(M_h~/~[M_\\odot h^{-1}]$)')
    plt.ylabel(r'log$(\langle L_{\mathrm{cen}} \rangle / [L_{\odot} h^{-2}])$')
    plt.xlim(10,15)
    plt.ylim(7,LOG_LGAL_MAX_TIGHT)
    plt.legend()
    plt.twinx()
    plt.ylim(log_solar_L_to_abs_mag_r(7), log_solar_L_to_abs_mag_r(LOG_LGAL_MAX_TIGHT))
    plt.yticks(np.arange(-23, -12, 2))
    plt.ylabel("$M_r$ - 5log(h)")

    if inset is not None:
        ax_inset = plt.gca().inset_axes([0.45, 0.1, 0.5, 0.5])
        ax_inset.tick_params(axis='both', which='major', labelsize=10)
        inset_log_means_r = inset.centrals.loc[inset.centrals['QUIESCENT']].groupby('Mh_bin2', observed=False).apply(LogLgal_vmax_weighted)
        inset_log_means_b = inset.centrals.loc[~inset.centrals['QUIESCENT']].groupby('Mh_bin2', observed=False).apply(LogLgal_vmax_weighted)
        ratio2 = 10**inset_log_means_r / 10**inset_log_means_b

        #inset_yerr_all_lower, inset_yerr_all_upper = safe_log_err(inset_log_mean_all, inset.lhmr_bootstrap_err)
        #inset_yerr_b_lower, inset_yerr_b_upper = safe_log_err(inset_log_means_b, inset.lhmr_sf_bootstrap_err)
        #inset_yerr_r_lower, inset_yerr_r_upper = safe_log_err(inset_log_means_r, inset.lhmr_q_bootstrap_err)
        
        #ax_inset.errorbar(x_vals, inset_log_means_b, yerr=[inset_yerr_b_lower, inset_yerr_b_upper], fmt='.', color='b', capsize=3, alpha=0.7)
        #ax_inset.errorbar(x_vals, inset_log_means_r, yerr=[inset_yerr_r_lower, inset_yerr_r_upper], fmt='.', color='r', capsize=3, alpha=0.7)
        #ax_inset.plot(x_vals2, inset_log_means_b, '-', color='b', alpha=0.7)
        #ax_inset.plot(x_vals2, inset_log_means_r, '-', color='r', alpha=0.7)
        ax_inset.plot(x_vals, ratio, '-', color='k', label='BGS')
        ax_inset.plot(x_vals2[:-6], ratio2[:-6], '-', color='purple', label='SDSS')
        ax_inset.set_xlim(10,15)
        ax_inset.legend(fontsize=8)
       #ax_inset.set_ylim(0.9, 1.1)
        #ax_inset.set_xlabel('log$(M_h~/~[M_\\odot h^{-1}]$)')
        ax_inset.set_ylabel('Q/SF', fontsize=10)
        # Dashed line at 1
        ax_inset.axhline(1.0, color='gray', linestyle='--', alpha=0.5)


def SHMR_savederr(f: GroupCatalog, show_all=False, inset: GroupCatalog=None):

    mean_all = f.centrals.groupby('Mh_bin', observed=False).apply(mstar_vmax_weighted)
    means_r = f.centrals.loc[f.centrals['QUIESCENT']].groupby('Mh_bin', observed=False).apply(mstar_vmax_weighted)
    means_b = f.centrals.loc[~f.centrals['QUIESCENT']].groupby('Mh_bin', observed=False).apply(mstar_vmax_weighted)
    #shmr_r_mean, shmr_r_err, shmr_r_scatter_mean, shmr_r_scatter_err, shmr_b_mean, shmr_b_err, shmr_b_scatter_mean, shmr_b_scatter_err, shmr_all_mean, shmr_all_err, shmr_all_scatter_mean, shmr_all_scatter_err = shmr_variance_from_saved()

    yerr_all_lower, yerr_all_upper = safe_log_err(np.log10(mean_all), f.shmr_bootstrap_err)
    yerr_b_lower, yerr_b_upper = safe_log_err(np.log10(means_b), f.shmr_sf_bootstrap_err)
    yerr_r_lower, yerr_r_upper = safe_log_err(np.log10(means_r), f.shmr_q_bootstrap_err)

    amask = mean_all > 0 & ~np.isnan(mean_all) & ~np.isnan(yerr_all_lower) & ~np.isnan(yerr_all_upper)
    rmask = means_r > 0 & ~np.isnan(means_r) & ~np.isnan(yerr_r_lower) & ~np.isnan(yerr_r_upper)
    bmask = means_b > 0 & ~np.isnan(means_b) & ~np.isnan(yerr_b_lower) & ~np.isnan(yerr_b_upper)
    ratio = means_r / means_b

    plt.figure(dpi=DPI)
    x_vals = np.log10(Mhalo_labels)
    x_vals2 = np.log10(Mhalo_labels2)

    if show_all:
        plt.errorbar(x_vals[amask], np.log10(mean_all[amask]), yerr=[yerr_all_lower[amask], yerr_all_upper[amask]], fmt='.', color='k', label='All', capsize=4, alpha=0.7)
    plt.errorbar(x_vals[bmask], np.log10(means_b[bmask]), yerr=[yerr_b_lower[bmask], yerr_b_upper[bmask]], fmt='.', color='b', label='SF Centrals', capsize=4, alpha=0.7)
    plt.errorbar(x_vals[rmask], np.log10(means_r[rmask]), yerr=[yerr_r_lower[rmask], yerr_r_upper[rmask]], fmt='.', color='r', label='Q Centrals', capsize=4, alpha=0.7)
    # No shaded systematic errors here as we didn't save it in MCMC chains.
    plt.xlabel('log$(M_h~/~[M_\\odot h^{-1}]$)')
    plt.ylabel(r'log$(\langle M_{\star} \rangle / [M_{\odot} h^{-2}])$')
    plt.xlim(10,15)
    plt.ylim(7,12)

    if inset is not None:
        ax_inset = plt.gca().inset_axes([0.45, 0.1, 0.5, 0.5])
        ax_inset.tick_params(axis='both', which='major', labelsize=10)
        inset_means_r = inset.centrals.loc[inset.centrals['QUIESCENT']].groupby('Mh_bin2', observed=False).apply(mstar_vmax_weighted)
        inset_means_b = inset.centrals.loc[~inset.centrals['QUIESCENT']].groupby('Mh_bin2', observed=False).apply(mstar_vmax_weighted)
        rimask = inset_means_r > 0 & ~np.isnan(inset_means_r)
        bimask = inset_means_b > 0 & ~np.isnan(inset_means_b)
        ratio2 = inset_means_r / inset_means_b

        ax_inset.plot(x_vals[rmask & bmask], ratio[rmask & bmask], '-', color='k', label='BGS')
        ax_inset.plot(x_vals2[rimask & bimask][:-3], ratio2[rimask & bimask][:-3], '-', color='purple', label='SDSS')
        ax_inset.set_xlim(10,15)
        ax_inset.legend(fontsize=8)
        ax_inset.set_ylabel('Q/SF', fontsize=10)
        # Dashed line at 1
        ax_inset.axhline(1.0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

def LHMR_scatter_savederr(f: GroupCatalog, show_all=False):

    scatter_all = f.centrals.groupby('Mh_bin', observed=False).apply(LogLgal_lognormal_scatter_vmax_weighted)
    scatter_r = f.centrals.loc[f.centrals['QUIESCENT']].groupby('Mh_bin', observed=False).apply(LogLgal_lognormal_scatter_vmax_weighted)
    scatter_b = f.centrals.loc[~f.centrals['QUIESCENT']].groupby('Mh_bin', observed=False).apply(LogLgal_lognormal_scatter_vmax_weighted)
    
    lhmr_r_mean, lhmr_r_err, lhmr_r_scatter_mean, lhmr_r_scatter_err, lhmr_b_mean, lhmr_b_err, lhmr_b_scatter_mean, lhmr_b_scatter_err, lhmr_all_mean, lhmr_all_err, lhmr_all_scatter_mean, lhmr_all_scatter_err = lhmr_variance_from_saved()

    # Error bars for statistical error (bootstrapped)
    # And shaded for systematic (saved from MCMC)
    # Scatter is computed as lognormal. That value is then considered a linear value (units of dex).
    # The error is the scatter is linear in that space.
    all_lower = scatter_all - f.lhmr_scatter_bootstrap_err[0]
    all_upper = f.lhmr_scatter_bootstrap_err[1] - scatter_all
    r_lower = scatter_r - f.lhmr_q_scatter_bootstrap_err[0]
    r_upper = f.lhmr_q_scatter_bootstrap_err[1] - scatter_r
    b_lower = scatter_b - f.lhmr_sf_scatter_bootstrap_err[0]
    b_upper = f.lhmr_sf_scatter_bootstrap_err[1] - scatter_b

    plt.figure(dpi=DPI)
    x_vals = np.log10(Mhalo_labels)

    if show_all:
        plt.errorbar(x_vals, scatter_all, yerr=[all_lower, all_upper], fmt='.', color='k', label='All', capsize=4, alpha=0.7)
    plt.errorbar(x_vals, scatter_r, yerr=[r_lower, r_upper], fmt='.', color='r', label='Q Centrals', capsize=4, alpha=0.7)
    plt.errorbar(x_vals, scatter_b, yerr=[b_lower, b_upper], fmt='.', color='b', label='SF Centrals', capsize=4, alpha=0.7)

    if show_all:
        plt.fill_between(x_vals, lhmr_all_scatter_err[0], lhmr_all_scatter_err[1], color='gray', alpha=0.2)
    plt.fill_between(x_vals, lhmr_r_scatter_err[0], lhmr_r_scatter_err[1], color='red', alpha=0.2)
    plt.fill_between(x_vals, lhmr_b_scatter_err[0], lhmr_b_scatter_err[1], color='blue', alpha=0.2)

    plt.xlabel('log$(M_h~/~[M_\\odot h^{-1}]$)')
    plt.ylabel(r'$\sigma_{{\mathrm{log}}(L_{\mathrm{cen}}~/~[L_{\odot} h^{-2}])}$')
    plt.legend()
    plt.xlim(10,15)
    plt.ylim(0.0, 0.4)
    plt.legend()
    plt.twinx()
    plt.ylim(0, np.abs(log_solar_L_to_abs_mag_r(9.4) - log_solar_L_to_abs_mag_r(9)))
    plt.ylabel(r'$\sigma_{M_r}$')
    plt.tight_layout()

def SHMR_scatter_savederr(f: GroupCatalog, show_all=False):

    scatter_all = f.centrals.groupby('Mh_bin', observed=False).apply(LogMstar_lognormal_scatter_vmax_weighted)
    scatter_r = f.centrals.loc[f.centrals['QUIESCENT']].groupby('Mh_bin', observed=False).apply(LogMstar_lognormal_scatter_vmax_weighted)
    scatter_b = f.centrals.loc[~f.centrals['QUIESCENT']].groupby('Mh_bin', observed=False).apply(LogMstar_lognormal_scatter_vmax_weighted)

    #shmr_r_mean, shmr_r_err, shmr_r_scatter_mean, shmr_r_scatter_err, shmr_b_mean, shmr_b_err, shmr_b_scatter_mean, shmr_b_scatter_err, shmr_all_mean, shmr_all_err, shmr_all_scatter_mean, shmr_all_scatter_err = shmr_variance_from_saved()

    all_lower = scatter_all - f.shmr_scatter_bootstrap_err[0]
    all_upper = f.shmr_scatter_bootstrap_err[1] - scatter_all
    r_lower = scatter_r - f.shmr_q_scatter_bootstrap_err[0]
    r_upper = f.shmr_q_scatter_bootstrap_err[1] - scatter_r
    b_lower = scatter_b - f.shmr_sf_scatter_bootstrap_err[0]
    b_upper = f.shmr_sf_scatter_bootstrap_err[1] - scatter_b

    plt.figure(dpi=DPI)
    x_vals = np.log10(Mhalo_labels)

    if show_all:
        plt.errorbar(x_vals, scatter_all, yerr=[all_lower, all_upper], fmt='.', color='k', label='All', capsize=4, alpha=0.7)
    plt.errorbar(x_vals, scatter_b, yerr=[b_lower, b_upper], fmt='.', color='b', label='SF Centrals', capsize=4, alpha=0.7)
    plt.errorbar(x_vals, scatter_r, yerr=[r_lower, r_upper], fmt='.', color='r', label='Q Centrals', capsize=4, alpha=0.7)

    # No shaded systematic errors here for now as we didn't save it in MCMC chains.

    plt.xlabel('log$(M_h~/~[M_\\odot h^{-1}]$)')
    plt.ylabel(r'$\sigma_{{\mathrm{log}}(M_{\star}~/~[M_{\odot} h^{-2}])}$')
    plt.xlim(10,15)
    plt.ylim(0.0, 0.7)
    plt.legend()
    plt.tight_layout()

def SHMR_scatter_litcompare(f: GroupCatalog):
    # TODO stuff to compare to from literature
    scatter_all = f.centrals.groupby('Mh_bin', observed=False).apply(LogMstar_lognormal_scatter_vmax_weighted)

    all_lower = scatter_all - f.shmr_scatter_bootstrap_err[0]
    all_upper = f.shmr_scatter_bootstrap_err[1] - scatter_all

    plt.figure(dpi=DPI)
    x_vals = np.log10(Mhalo_labels)
    plt.errorbar(x_vals, scatter_all, yerr=[all_lower, all_upper], fmt='.', color='k', label='This Work', capsize=4, alpha=0.7)
    plt.xlabel('log$(M_h~/~[M_\\odot h^{-1}]$)')
    plt.ylabel(r'$\sigma_{{\mathrm{log}}(M_{\star}~/~[M_{\odot} h^{-2}])}$')
    plt.xlim(10,15)
    plt.ylim(0.0, 0.7)
    plt.legend()
    plt.tight_layout()


def fsat_with_bootstrapped_err(gc: GroupCatalog):
    plt.figure()
    #plt.errorbar(L_gal_bins, gc.fsat, yerr=fsat_std, fmt='.', color='k', label='All', capsize=3, alpha=0.7)
    plt.errorbar(LogLgal_labels, gc.fsatr, yerr=gc.fsat_q_bootstrap_err, fmt='.', color='r', label='Quiescent', markersize=6, capsize=3, alpha=1.0)
    plt.errorbar(LogLgal_labels, gc.fsatb, yerr=gc.fsat_sf_bootstrap_err, fmt='.', color='b', label='Star-forming', markersize=6, capsize=3, alpha=1.0)
    plt.xlabel('log$(L_{\mathrm{gal}}~/~[L_{\odot}~h^{-2}])$')
    plt.ylabel('$f_{\mathrm{sat}}$')
    plt.legend()
    plt.xlim(8,LOG_LGAL_MAX_TIGHT)
    plt.ylim(0.0, 1.0)
    plt.twiny()
    plt.xticks(np.arange(-23, -10, 1))
    plt.xlim(log_solar_L_to_abs_mag_r(8), log_solar_L_to_abs_mag_r(LOG_LGAL_MAX_TIGHT))
    plt.xlabel("$M_r$ - 5log(h)")
    plt.tight_layout()

def fsat_with_err_from_saved(gc: GroupCatalog, show_all=False):
    fsat_err, fsatr_err, fsatb_err, fsat_mean, fsatr_mean, fsatb_mean = fsat_variance_from_saved()
    rcut = 11

    plt.figure(dpi=DPI)
    if show_all:
        plt.errorbar(LogLgal_labels, gc.fsat, yerr=gc.fsat_bootstrap_err, fmt='.', color='k', label='All', markersize=6, capsize=4, alpha=1.0)
    plt.errorbar(LogLgal_labels[rcut:], gc.fsatr[rcut:], yerr=gc.fsat_q_bootstrap_err[rcut:], fmt='.', color='r', label='Quiescent', markersize=6, capsize=4, alpha=1.0)
    plt.errorbar(LogLgal_labels, gc.fsatb, yerr=gc.fsat_sf_bootstrap_err, fmt='.', color='b', label='Star-forming', markersize=6, capsize=4, alpha=1.0)

    if show_all:
        plt.fill_between(LogLgal_labels, fsat_err[0], fsat_err[1], color='k', alpha=0.2)
    plt.fill_between(LogLgal_labels[rcut:], fsatr_err[0][rcut:], fsatr_err[1][rcut:], color='r', alpha=0.2) 
    plt.fill_between(LogLgal_labels, fsatb_err[0], fsatb_err[1], color='b', alpha=0.2)

    plt.xlabel('log$(L_{\mathrm{gal}}~/~[L_{\odot}~h^{-2}])$')
    plt.ylabel('$f_{\mathrm{sat}}$')
    plt.legend()
    plt.xlim(7,LOG_LGAL_MAX_TIGHT)
    plt.ylim(0.0, 1.0)
    plt.twiny()
    plt.xticks(np.arange(-23, -9, 2))
    plt.xlim(log_solar_L_to_abs_mag_r(7), log_solar_L_to_abs_mag_r(LOG_LGAL_MAX_TIGHT))
    plt.xlabel("$M_r$ - 5log(h)")
    plt.tight_layout()

def plots(*catalogs, show_err=None, truth_on=False):
    catalogs = list(catalogs)
    if show_err is not None and show_err not in catalogs:
        catalogs.append(show_err)
    elif not show_err == None:
        print("show_err must be a GroupCatalog")

    completeness_stats(catalogs)

    """

    # SHMR fractional
    fig,ax1=plt.subplots()
    fig.set_dpi(DPI)
    for f in datasets:
        mcen_means = f.centrals.groupby('Mh_bin', observed=False).apply(mstar_vmax_weighted)
        #mcen_scatter = f.centrals.groupby('Mh_bin', observed=False).apply(mstar_std_vmax_weighted)
        plt.plot(f.labels, mcen_means/f.labels, f.marker, label=get_dataset_display_name(f), color=f.color)
        #plt.errorbar(f.labels, mcen_means/f.labels, yerr=mcen_scatter/f.labels, label=get_dataset_display_name(f), color=f.color)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$M_h$')
    plt.ylabel('$M_{\\star}/M_h$')
    legend(datasets)
    plt.xlim(1E10,1E15)
    #plt.ylim(1E6,3E12)
    ax2 = ax1.twinx()
    idx = 0
    for f in datasets:
        widths = np.zeros(len(f.Mhalo_bins)-1)
        for i in range(0,len(f.Mhalo_bins)-1):
            widths[i]=(f.Mhalo_bins[i+1] - f.Mhalo_bins[i]) / len(datasets)
        
        # This version 1/vmax weights the counts
        #ax2.bar(f.L_gal_labels+(widths*idx), f.sats.groupby('LGAL_BIN', observed=False).apply(count_vmax_weighted), width=widths, color=f.color, align='edge', alpha=0.4)
        ax2.bar(f.labels+(widths*idx), f.all_data.groupby('Mh_bin', observed=False).size(), width=widths, color=f.color, align='edge', alpha=0.4)
        idx+=1
    ax2.set_ylabel('$N_{gal}$')
    ax2.set_yscale('log')
    plt.draw()
    """

    X_MAX = LGAL_MAX_TIGHT

    # ALL fsat vs Lgal 
    for xmin in LGAL_XMINS:
        fig,ax1=plt.subplots()
        fig.set_dpi(DPI)
        for f in catalogs:
            if f is show_err:
                plt.errorbar(f.L_gal_labels, f.f_sat, marker='.', linestyle='none', yerr=f.fsat_bootstrap_err, label=get_dataset_display_name(f), color=f.color)
            else:
                plt.plot(f.L_gal_labels, f.f_sat, f.marker, label=get_dataset_display_name(f), color=f.color)
        if truth_on:
            for f in catalogs:
                if 'IS_SAT_T' in f.all_data.columns:
                    plt.plot(f.L_gal_labels, f.truth_f_sat, 'v', label=f"{f.name} Truth", color=f.color)
        ax1.set_xscale('log')
        ax1.set_xlabel("$L_{\\mathrm{gal}}~[\\mathrm{L}_\\odot \\mathrm{h}^{-2} ]$")
        ax1.set_ylabel("$f_{\\mathrm{sat}}$")
        legend_ax(ax1, catalogs)
        ax1.set_xlim(xmin,X_MAX)
        ax1.set_ylim(0.0,0.6)
        ax1.grid(True)
        ax2=ax1.twiny()
        ax2.plot(catalogs[0].Mr_gal_labels, catalogs[0].f_sat, ls="")
        ax2.set_xlim(log_solar_L_to_abs_mag_r(np.log10(xmin)), log_solar_L_to_abs_mag_r(np.log10(X_MAX)))
        ax2.set_xlabel("$M_r$ - 5log(h)")
        fig.tight_layout()
        
    # Show % change in fsat from the show_err catalog
    if show_err is not None:
        fig,ax1=plt.subplots()
        fig.set_dpi(DPI)
        for f in catalogs:
            if f is not show_err:
                plt.plot(f.L_gal_labels, (f.f_sat - show_err.f_sat) / show_err.f_sat, f.marker, label=get_dataset_display_name(f), color=f.color)
        ax1.set_xscale('log')
        ax1.set_xlabel("$L_{\\mathrm{gal}}~[\\mathrm{L}_\\odot \\mathrm{h}^{-2} ]$")
        ax1.set_ylabel("$f_{\\mathrm{sat}}$ % Change")
        legend_ax(ax1, catalogs)
        ax1.set_xlim(LGAL_MIN,LGAL_MAX)
        ax1.set_ylim(-0.5,0.5)
        ax1.grid(True)
        fig.tight_layout()

    # Blue fsat
    for xmin in LGAL_XMINS:
        fig,ax1=plt.subplots()
        fig.set_dpi(DPI)
        for f in catalogs:
            if f is show_err:
                plt.errorbar(f.L_gal_labels, f.f_sat_sf, marker='.', linestyle='none', yerr=f.fsat_sf_bootstrap_err, label=get_dataset_display_name(f), color=f.color)
            else:
                plt.plot(f.L_gal_labels, f.f_sat_sf, f.marker, color=f.color, label=get_dataset_display_name(f))

        ax1.set_xscale('log')
        ax1.set_xlabel("$L_{\\mathrm{gal}}~[\\mathrm{L}_\\odot \\mathrm{h}^{-2} ]$")
        ax1.set_ylabel("Star-forming $f_{\\mathrm{sat}}$ ")
        legend_ax(ax1, catalogs)
        ax1.set_xlim(xmin,X_MAX)
        ax1.set_ylim(0.0,0.5)
        ax1.grid(True)
        ax2=ax1.twiny()
        ax2.plot(catalogs[0].Mr_gal_labels, catalogs[0].f_sat_sf, ls="")
        ax2.set_xlim(log_solar_L_to_abs_mag_r(np.log10(xmin)), log_solar_L_to_abs_mag_r(np.log10(X_MAX)))
        ax2.set_xlabel("$M_r$ - 5log(h)")
        fig.tight_layout()

    # Red fsat
    for xmin in LGAL_XMINS:
        fig,ax1=plt.subplots()
        fig.set_dpi(DPI)
        for f in catalogs:
            if f is show_err:
                plt.errorbar(f.L_gal_labels, f.f_sat_q, marker='.', linestyle='none', yerr=f.fsat_q_bootstrap_err, label=get_dataset_display_name(f), color=f.color)
            else:
                plt.plot(f.L_gal_labels, f.f_sat_q, f.marker, color=f.color, label=get_dataset_display_name(f))

        ax1.set_xscale('log')
        ax1.set_xlabel("$L_{\\mathrm{gal}}~[\\mathrm{L}_\\odot \\mathrm{h}^{-2} ]$")
        ax1.set_ylabel("Quiescent $f_{\\mathrm{sat}}$ ")
        legend_ax(ax1, catalogs)
        ax1.set_xlim(xmin,X_MAX)
        ax1.set_ylim(0.0,1.0)
        ax1.grid(True)
        ax2=ax1.twiny()
        ax2.plot(catalogs[0].Mr_gal_labels, catalogs[0].f_sat_q, ls="")
        ax2.set_xlim(log_solar_L_to_abs_mag_r(np.log10(xmin)), log_solar_L_to_abs_mag_r(np.log10(X_MAX)))
        ax2.set_xlabel("$M_r$ - 5log(h)")
        fig.tight_layout()

    #ax2 = ax1.twinx()
    #idx = 0
    #for f in datasets:
    #    widths = np.zeros(len(f.L_gal_bins)-1)
    #    for i in range(0,len(f.L_gal_bins)-1):
    #        widths[i]=(f.L_gal_bins[i+1] - f.L_gal_bins[i]) / (len(datasets))
    #    ax2.bar(f.L_gal_labels+(widths*idx), f.all_data[f.all_data['IS_SAT'] == True].groupby('LGAL_BIN', observed=False).size(), width=widths, color=f.color, align='edge', alpha=0.4)
    #    idx+=1
    #ax2.set_ylabel('$N_{sat}$')
    fig.tight_layout()

    qf_cen_plot(*catalogs)

    if len(catalogs) == 1:
        plots_color_split(*catalogs, total_on=True)


def wp_rp(catalog: GroupCatalog|tuple):
    colors = ['k', 'r', 'b']
    f = catalog
    plt.figure(figsize=(5, 5))
    to_use = f.get_best_wp_all() if isinstance(f, GroupCatalog) else f

    if to_use is None:
        return
    
    if hasattr(f, 'wp_err'):
        plt.errorbar(to_use[0][:-1], to_use[1], yerr=f.wp_err, marker='o', linestyle='-', label='All', color=colors[0])
        plt.errorbar(to_use[0][:-1], to_use[2], yerr=f.wp_r_err, marker='o', linestyle='-', label='Red', color=colors[1])
        plt.errorbar(to_use[0][:-1], to_use[3], yerr=f.wp_b_err, marker='o', linestyle='-', label='Blue', color=colors[2])
    else:
        plt.plot(to_use[0][:-1], to_use[1], marker='o', linestyle='-', label='All', color=colors[0])
        plt.plot(to_use[0][:-1], to_use[2], marker='o', linestyle='-', label='Red', color=colors[1])
        plt.plot(to_use[0][:-1], to_use[3], marker='o', linestyle='-', label='Blue', color=colors[2])
    plt.xscale('log')
    plt.ylim(3, 2000)
    plt.yscale('log')
    plt.xlabel(r'$r_p$ [Mpch^{-1}]')
    plt.ylabel(r'$w_p(r_p)$')
    plt.legend()
    if isinstance(catalog, BGSGroupCatalog):
        plt.title(f'{catalog.name} Full $w_p(r_p)$') 
    plt.grid(True)
    plt.show()

def wp_rp_magbins(c: GroupCatalog):
    colors = ['k', 'r', 'b']

    # Additional rows for each magnitude slice in wp_slices
    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
    if c.wp_slices_extra is not None and c.wp_slices_extra[0] is not None:
        to_use = c.wp_slices_extra
    else:
        to_use = c.wp_slices

    if to_use is None:
        return

    for i, mag_slice in enumerate(to_use):
        if i <= 1:
            continue
        if i == len(to_use) - 1:
            mag_range_label = f"{to_use[i][4]:.1f} > $M_r$ - 5log($h$)"
        else:
            mag_range_label = f"{to_use[i][4]:.1f} > $M_r$ - 5log($h$) > {to_use[i][5]:.1f}"

        row = (i-2) // 3
        col = (i-2) % 3

        axes[row, col].plot(to_use[i][0][:-1], to_use[i][1], marker='o', linestyle='-', label=c.name, color=colors[0])
        axes[row, col].plot(to_use[i][0][:-1], to_use[i][2], marker='o', linestyle='-', label=c.name, color=colors[1])
        axes[row, col].plot(to_use[i][0][:-1], to_use[i][3], marker='o', linestyle='-', label=c.name, color=colors[2])
        axes[row, col].set_xscale('log')
        axes[row, col].set_ylim(3, 2000)
        axes[row, col].set_yscale('log')
        axes[row, col].set_xlabel(r'$r_p$ [Mpch^{-1}]')
        axes[row, col].set_ylabel(r'$w_p(r_p)$')
        axes[row, col].set_title(f'{mag_range_label}')
        axes[row, col].grid(True)


    plt.tight_layout()
    plt.show()


def compare_wp_rp(d1: BGSGroupCatalog|tuple, d2_t: BGSGroupCatalog|tuple):
    """
    Compare two wp_rp functions (total, red, blue). The second is used as the 'truth' (in the denominator).
    """
    
    def plot_fractional_difference(ax, wp1, wp2, wp1_red, wp2_red, wp1_blue, wp2_blue, label, bins):
        percent_diff = 100 * (wp1 - wp2) / wp2
        percent_diff_r = 100 * (wp1_red - wp2_red) / wp2_red
        percent_diff_b = 100 * (wp1_blue - wp2_blue) / wp2_blue

        ax.plot(bins, percent_diff, marker='o', linestyle='-', color='black', label=f'Overall {label}')
        ax.plot(bins, percent_diff_r, marker='o', linestyle='-', color='red', label=f'Red {label}')
        ax.plot(bins, percent_diff_b, marker='o', linestyle='-', color='blue', label=f'Blue {label}')
        
        ax.set_xscale('log')
        lim = np.max([*np.abs(percent_diff), *np.abs(percent_diff_r), *np.abs(percent_diff_b)])
        ax.set_ylim(-lim, lim)
        ax.set_yscale('linear')
        ax.set_xlabel(r'$r_p$ [Mpch^{-1}]')
        ax.set_ylabel(r'$w_p(r_p)$ Difference (%)')
        ax.grid(True)

    fig, ax = plt.subplots(figsize=(5, 5))
    first = d1.get_best_wp_all() if isinstance(d1, GroupCatalog) else d1
    second = d2_t.get_best_wp_all() if isinstance(d2_t, GroupCatalog) else d2_t

    plot_fractional_difference(ax, first[1], second[1], first[2], second[2], first[3], second[3], 'Flux-limited', first[0][:-1])
    
    # Show a shaded region for the error in the d2 set. 
    if hasattr(d2_t, 'wp_r_err') and hasattr(d2_t, 'wp_b_err'):
        #ax.fill_between(d2_t.get_best_wp_all()[0][:-1], 100*d2_t.wp_err/d2_t.get_best_wp_all()[1], -100*d2_t.wp_err/d2_t.get_best_wp_all()[1], color='black', alpha=0.2)
        ax.fill_between(d2_t.get_best_wp_all()[0][:-1], 100*d2_t.wp_r_err/d2_t.get_best_wp_all()[2], -100*d2_t.wp_r_err/d2_t.get_best_wp_all()[2], color='red', alpha=0.25)
        ax.fill_between(d2_t.get_best_wp_all()[0][:-1], 100*d2_t.wp_b_err/d2_t.get_best_wp_all()[3], -100*d2_t.wp_b_err/d2_t.get_best_wp_all()[3], color='blue', alpha=0.25)
        
    if isinstance(d1, GroupCatalog) and isinstance(d2_t, GroupCatalog):
        ax.set_title(f'{d1.name} vs {d2_t.name}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    

    if isinstance(d1, BGSGroupCatalog) and isinstance(d2_t, BGSGroupCatalog):
        
        # Additional rows for each magnitude slice in wp_slices
        fig, axes = plt.subplots(2, 3, figsize=(12, 9))

        for i, mag_slice in enumerate(d1.wp_slices):
            if i <= 1:
                continue

            mag_range_label = f"{d1.wp_slices[i][4]:.1f} > $M_r$ - 5log($h$) > {d1.wp_slices[i][5]:.1f}"
            row = (i - 2) // 3
            col = (i - 2) % 3

            plot_fractional_difference(axes[row, col], d1.wp_slices[i][1], d2_t.wp_slices[i][1], d1.wp_slices[i][2], d2_t.wp_slices[i][2], d1.wp_slices[i][3], d2_t.wp_slices[i][3], f'\n{mag_range_label}', d1.wp_slices[i][0][:-1])
            axes[row, col].set_title(f'{mag_range_label}')

        plt.tight_layout()
        plt.show()


def plots_color_split_lost_split(f, grpby_col):
    q_gals = f.all_data[f.all_data['QUIESCENT']]
    sf_gals = f.all_data[np.invert(f.all_data['QUIESCENT'])]
    plots_color_split_lost_split_inner(get_dataset_display_name(f), f.L_gal_labels, f.L_gal_bins, q_gals, sf_gals, fsat_vmax_weighted, grpby_col)

def plots_color_split_lost_split_inner(name, L_gal_labels, L_gal_bins, q_gals, sf_gals, aggregation_func, grpby_col, show_plot=True):
    q_lost = q_gals[z_flag_is_not_spectro_z(q_gals['Z_ASSIGNED_FLAG'])].groupby([grpby_col], observed=False)
    q_obs = q_gals[z_flag_is_spectro_z(q_gals['Z_ASSIGNED_FLAG'])].groupby([grpby_col], observed=False)
    sf_lost = sf_gals[z_flag_is_not_spectro_z(sf_gals['Z_ASSIGNED_FLAG'])].groupby([grpby_col], observed=False)
    sf_obs = sf_gals[z_flag_is_spectro_z(sf_gals['Z_ASSIGNED_FLAG'])].groupby([grpby_col], observed=False)
    fsat_qlost = q_lost.apply(aggregation_func)
    fsat_qobs = q_obs.apply(aggregation_func)
    fsat_sflost = sf_lost.apply(aggregation_func)
    fsat_sfobs = sf_obs.apply(aggregation_func)
    fsat_qtot = q_gals.groupby([grpby_col], observed=False).apply(aggregation_func)
    fsat_sftot = sf_gals.groupby([grpby_col], observed=False).apply(aggregation_func)

    if show_plot:
        fig,axes=plt.subplots(nrows=1, ncols=2, figsize=(12,5))
        ax1=axes[0]
        ax2=axes[1]
        fig.set_dpi(DPI)
        ax1.plot(L_gal_labels, fsat_qlost, '>', label='Lost quiescent', color='r')
        ax1.plot(L_gal_labels, fsat_qobs, '.', label='Observed quiescent', color='r')
        ax1.plot(L_gal_labels, fsat_qtot, '-', label="Total quiescent", color='r')
        ax2.plot(L_gal_labels, fsat_sflost, '>', label='Lost star-forming', color='b')
        ax2.plot(L_gal_labels, fsat_sfobs, '.', label='Observed star-forming', color='b')
        ax2.plot(L_gal_labels, fsat_sftot, '-', label="Total star-forming", color='b')

        widths = np.zeros(len(L_gal_bins)-1)
        for i in range(0,len(L_gal_bins)-1):
            widths[i]=(L_gal_bins[i+1] - L_gal_bins[i]) / 2

        ax3 = ax1.twinx()
        idx = 0
        ax3.bar(L_gal_labels+(widths*idx), q_lost.size(), width=widths, color='orange', align='edge', alpha=0.4)
        idx+=1
        ax3.bar(L_gal_labels+(widths*idx), q_obs.size(), width=widths, color='k', align='edge', alpha=0.4)
        ax3.set_ylabel('$N_{gal}$')
        ax3.set_ylim(1, 1E5)
        ax3.set_yscale('log')

        ax4 = ax2.twinx()
        idx = 0
        ax4.bar(L_gal_labels+(widths*idx), sf_lost.size(), width=widths, color='orange', align='edge', alpha=0.4)
        idx+=1
        ax4.bar(L_gal_labels+(widths*idx), sf_obs.size(), width=widths, color='k', align='edge', alpha=0.4)
        ax4.set_ylim(1, 1E5)
        ax4.set_ylabel('$N_{gal}$')
        ax4.set_yscale('log')

        X_MIN = LGAL_MIN
        X_MAX = LGAL_MAX
        ax5=ax1.twiny()
        ax5.plot(Mr_gal_labels, fsat_qtot, ls="") # dummy plot
        ax5.set_xlim(log_solar_L_to_abs_mag_r(np.log10(X_MIN)), log_solar_L_to_abs_mag_r(np.log10(X_MAX)))
        ax5.set_xlabel("$M_r$ - 5log(h)")
        ax6=ax2.twiny()
        ax6.plot(Mr_gal_labels, fsat_qtot, ls="") # dummy plot
        ax6.set_xlim(log_solar_L_to_abs_mag_r(np.log10(X_MIN)), log_solar_L_to_abs_mag_r(np.log10(X_MAX)))
        ax6.set_xlabel("$M_r$ - 5log(h)")

        ax1.set_xscale('log')
        ax2.set_xscale('log')
        ax1.set_xlabel("$L_{\\mathrm{gal}}~[\\mathrm{L}_\\odot \\mathrm{h}^{-2} ]$")
        ax2.set_xlabel("$L_{\\mathrm{gal}}~[\\mathrm{L}_\\odot \\mathrm{h}^{-2} ]$")
        ax1.set_ylabel("$f_{\\mathrm{sat}}$ ")
        ax2.set_ylabel("$f_{\\mathrm{sat}}$ ")
        ax1.legend()
        ax2.legend()
        ax1.set_xlim(X_MIN,X_MAX)
        ax2.set_xlim(X_MIN,X_MAX)
        ax1.set_ylim(0.0,1.0)
        ax2.set_ylim(0.0,1.0)

        fig.suptitle(f"{name}")
        fig.tight_layout()

    return fsat_qlost.to_numpy(), fsat_qobs.to_numpy(), fsat_qtot.to_numpy(), fsat_sflost.to_numpy(), fsat_sfobs.to_numpy(), fsat_sftot.to_numpy()

def compare_fsat_color_split(dataone, datatwo, project_percent=None):
    temp1 = dataone.marker
    temp2 = datatwo.marker
    dataone.marker = '-'
    datatwo.marker = '--'
    plots_color_split(dataone, datatwo)
    dataone.marker = temp1
    datatwo.marker = temp2

    # Absolute change in fsat  plot
    red_difference = (dataone.f_sat_q - datatwo.f_sat_q)
    blue_difference = (dataone.f_sat_sf - datatwo.f_sat_sf)

    fig,ax1=plt.subplots()
    fig.set_dpi(DPI)
    plt.plot(dataone.L_gal_labels, red_difference, 'r', label="Quiescent")
    plt.plot(dataone.L_gal_labels, blue_difference, 'b', label="SF")
    if project_percent is not None:
        red_difference_projected = red_difference / project_percent
        blue_difference_projected = blue_difference / project_percent
        plt.plot(dataone.L_gal_labels, red_difference_projected, 'r--', label="Quiescent (projected full)")
        plt.plot(dataone.L_gal_labels, blue_difference_projected, 'b--', label="SF (projected full)")
    ax1.set_xscale('log')
    ax1.set_xlabel("$L_{\\mathrm{gal}}~[\\mathrm{L}_\\odot \\mathrm{h}^{-2} ]$")
    ax1.set_ylabel("$f_{\\mathrm{sat}}$ Absolute Difference")
    ax1.legend()
    X_MAX = LGAL_MAX
    ax1.set_xlim(LGAL_MIN,X_MAX)
    ax1.set_ylim(-0.15, 0.15)
    fig.tight_layout()

    # % Change Difference fsat plot
    red_difference_p = (dataone.f_sat_q - datatwo.f_sat_q) * 100
    blue_difference_p = (dataone.f_sat_sf - datatwo.f_sat_sf) * 100

    fig,ax1=plt.subplots()
    fig.set_dpi(DPI)
    plt.plot(dataone.L_gal_labels, red_difference_p, 'r', label="Quiescent")
    plt.plot(dataone.L_gal_labels, blue_difference_p, 'b', label="SF")
    if project_percent is not None:
        red_difference_projected = red_difference_p / project_percent
        blue_difference_projected = blue_difference_p / project_percent
        plt.plot(dataone.L_gal_labels, red_difference_projected, 'r--', label="Quiescent (projected full)")
        plt.plot(dataone.L_gal_labels, blue_difference_projected, 'b--', label="SF (projected full)")
    ax1.set_xscale('log')
    ax1.set_xlabel("$L_{\\mathrm{gal}}~[\\mathrm{L}_\\odot \\mathrm{h}^{-2} ]$")
    ax1.set_ylabel("$f_{\\mathrm{sat}}$ Difference (%)")
    ax1.legend()
    X_MAX = LGAL_MAX
    ax1.set_xlim(LGAL_MIN,X_MAX)
    ax1.set_ylim(-20,20)
    fig.tight_layout()



def plots_color_split(*datasets, truth_on=False, total_on=False):

    for xmin in LGAL_XMINS:

        fig,ax1=plt.subplots()
        fig.set_dpi(DPI)
        for f in datasets:
            #if not hasattr(f, 'f_sat_q'):
            #    f.f_sat_q = f.all_data[f.all_data['QUIESCENT']].groupby(['LGAL_BIN'], observed=False).apply(fsat_vmax_weighted)
            #if not hasattr(f, 'f_sat_sf'):
            #    f.f_sat_sf = f.all_data[np.invert(f.all_data['QUIESCENT'])].groupby(['LGAL_BIN'], observed=False).apply(fsat_vmax_weighted)
            
            if hasattr(f, 'fsat_q_bootstrap_err'):
                plt.errorbar(f.L_gal_labels, f.f_sat_q, yerr=f.fsat_q_bootstrap_err, label="Quiescent", color='r')
            else:
                plt.plot(f.L_gal_labels, f.f_sat_q, label="Quiescent", color='r')
                
            if hasattr(f, 'fsat_sf_bootstrap_err'):
                plt.errorbar(f.L_gal_labels, f.f_sat_sf, yerr=f.fsat_sf_bootstrap_err, label="Star-forming", color='b')
            else:
                plt.plot(f.L_gal_labels, f.f_sat_sf, f.marker, label="Star-forming", color='b')
            
            if total_on:
                if hasattr(f, 'fsat_bootstrap_err'):
                    plt.errorbar(f.L_gal_labels, f.f_sat, yerr=f.fsat_bootstrap_err, label="Total", color='k')
                else:
                    plt.plot(f.L_gal_labels, f.f_sat, label="Total", color='k')

        for f in datasets:
            if truth_on:
                if f.has_truth:
                    truth_on = False
                    f.f_sat_q_t = f.all_data[f.all_data['QUIESCENT']].groupby(['LGAL_BIN'], observed=False).apply(fsat_truth_vmax_weighted)
                    f.f_sat_sf_t = f.all_data[np.invert(f.all_data['QUIESCENT'])].groupby(['LGAL_BIN'], observed=False).apply(fsat_truth_vmax_weighted)
                    plt.plot(f.L_gal_labels, f.f_sat_q_t, 'x', label="Simulation's Truth", color='r')
                    plt.plot(f.L_gal_labels, f.f_sat_sf_t, 'x', label="Simulation's Truth", color='b')


        ax1.set_title(get_dataset_display_name(f))
        ax1.set_xscale('log')
        ax1.set_xlabel("$L_{\\mathrm{gal}}~[\\mathrm{L}_\\odot \\mathrm{h}^{-2} ]$")
        ax1.set_ylabel("$f_{\\mathrm{sat}}$ ")
        ax1.legend()
        X_MAX = LGAL_MAX
        ax1.set_xlim(xmin,X_MAX)
        ax1.set_ylim(0.0,1.0)

        ax2=ax1.twiny()
        ax2.plot(datasets[0].Mr_gal_labels, datasets[0].f_sat_q, ls="")
        ax2.set_xlim(log_solar_L_to_abs_mag_r(np.log10(xmin)), log_solar_L_to_abs_mag_r(np.log10(X_MAX)))
        ax2.set_xlabel("$M_r$ - 5log(h)")


        fig.tight_layout()



def total_f_sat(ds):
    """
    Prints out the total f_sat for a dataframe, 1/Vmax weighted and not. 
    """
    print(f"  fsat (no weight):  {ds.all_data['IS_SAT'].mean():.3f}")
    print(f"  fsat (1 / V_max):  {fsat_vmax_weighted(ds.all_data):.3f}")
    
    if ds.has_truth and 'IS_SAT_T' in ds.all_data.columns:
        print(f"  Truth (no weight):  {ds.all_data['IS_SAT_T'].mean():.3f}")
        print(f"  Truth (1 / V_max):  {fsat_truth_vmax_weighted(ds.all_data):.3f}")


def fsat_by_zbins(*datasets, z_bins=np.array([0.0, 0.2, 1.0])):
    for i in range(0, len(z_bins)-1):
        z_low = z_bins[i]
        z_high = z_bins[i+1]

        fig,ax1=plt.subplots()
        fig.set_dpi(DPI)

        for d in datasets:
            z_cut = np.all([d.all_data['Z'] > z_low, d.all_data['Z'] < z_high], axis=0)
            print(f"z: {z_low:.2} - {z_high:.2} ({np.sum(z_cut)} galaxies)")
            fsat_zcut = d.all_data[z_cut].groupby('LGAL_BIN').apply(fsat_vmax_weighted)
            plt.plot(d.L_gal_labels, fsat_zcut, d.marker, label=get_dataset_display_name(d), color=d.color)

        ax1.set_xscale('log')
        ax1.set_xlabel("$L_{\\mathrm{gal}}~[\\mathrm{L}_\\odot \\mathrm{h}^{-2} ]$")
        ax1.set_ylabel("$f_{\\mathrm{sat}}$")
        legend_ax(ax1, datasets)
        ax1.set_xlim(LGAL_MIN,LGAL_MAX)
        ax1.set_ylim(0.0,1.0)
        ax2=ax1.twiny()
        ax2.plot(datasets[0].Mr_gal_labels, datasets[0].f_sat, ls="")
        ax2.set_xlim(log_solar_L_to_abs_mag_r(np.log10(LGAL_MIN)), log_solar_L_to_abs_mag_r(np.log10(LGAL_MAX)))
        ax2.set_xlabel("$M_r$ - 5log(h)")
        ax1.set_title(f"z: {z_low:.2} - {z_high:.2}")
        fig.tight_layout()

def fsat_by_z_bins(dataset, z_bins=np.array([0.0, 0.05, 0.1, 0.2, 0.3, 1.0]), show_plots=True, aggregation=fsat_vmax_weighted):
    # Call plots_color_split_lost_split for a few z bins
    fsat_qlost_arr, fsat_qobs_arr, fsat_qtot_arr, fsat_sflost_arr, fsat_sfobs_arr, fsat_sftot_arr = [], [], [], [], [], []
    L_bin_number = 25

    for i in range(0, len(z_bins)-1):
        z_low = z_bins[i]
        z_high = z_bins[i+1]
        z_cut = np.all([dataset.all_data['Z'] > z_low, dataset.all_data['Z'] < z_high], axis=0)
        print(f"z: {z_low:.2} - {z_high:.2} ({np.sum(z_cut)} galaxies)")
        q_gals = dataset.all_data[np.all([dataset.all_data['QUIESCENT'], z_cut], axis=0)]
        #q_gals.reset_index(drop=True, inplace=True)
        sf_gals = dataset.all_data[np.all([np.invert(dataset.all_data['QUIESCENT']), z_cut], axis=0)]
        #sf_gals.reset_index(drop=True, inplace=True)
        fsat_qlost, fsat_qobs, fsat_qtot, fsat_sflost, fsat_sfobs, fsat_sftot = plots_color_split_lost_split_inner(f"{dataset.name} z: {z_low:.2} - {z_high:.2}", dataset.L_gal_labels, dataset.L_gal_bins, q_gals, sf_gals, aggregation, show_plot=show_plots)
        fsat_qlost_arr.append(fsat_qlost[L_bin_number])
        fsat_qobs_arr.append(fsat_qobs[L_bin_number])
        fsat_qtot_arr.append(fsat_qtot[L_bin_number])
        fsat_sflost_arr.append(fsat_sflost[L_bin_number])
        fsat_sfobs_arr.append(fsat_sfobs[L_bin_number])
        fsat_sftot_arr.append(fsat_sftot[L_bin_number])


    # Pick one luminosity bin, and then make 6 lines for fsat_qlost, fsat_qobs, fsat_qtot, fsat_sflost, fsat_sfobs, fsat_sftot
    # over over the z_bins
    fig,ax1=plt.subplots()
    fig.set_dpi(DPI)

    z_bins_midpoints = (z_bins[1:] + z_bins[:-1]) / 2

    plt.plot(z_bins_midpoints, fsat_qlost_arr, '>', label='Lost quiescent', color='r')
    plt.plot(z_bins_midpoints, fsat_qobs_arr, '.', label='Observed quiescent', color='r')
    plt.plot(z_bins_midpoints, fsat_qtot_arr, '-', label="Total quiescent", color='r')
    plt.plot(z_bins_midpoints, fsat_sflost_arr, '>', label='Lost star-forming', color='b')
    plt.plot(z_bins_midpoints, fsat_sfobs_arr, '.', label='Observed star-forming', color='b')
    plt.plot(z_bins_midpoints, fsat_sftot_arr, '-', label="Total star-forming", color='b')
    
    ax1.set_xscale('linear')
    ax1.set_xlabel("$z$")
    ax1.set_ylabel("$f_{\\mathrm{sat}}$ ")
    ax1.legend()
    ax1.set_xlim(0.0,1.0)
    ax1.set_ylim(0.0,1.0)
    plt.title(f"{dataset.name} - $L$~{dataset.L_gal_labels[L_bin_number]:.1E}")



def L_func_plot(datasets: list, values: list):
    L_MIN = L_gal_labels[0]
    L_MAX = 3E12
    fig,ax1=plt.subplots()
    fig.set_dpi(DPI)
    for i in range(len(datasets)):
        f = datasets[i]
        plt.plot(f.L_gal_labels, values[i], f.marker, label=get_dataset_display_name(f), color=f.color)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel("$L_{\\mathrm{gal}}~[\\mathrm{L}_\\odot \\mathrm{h}^{-2} ]$")
    ax1.set_ylabel("$N_{gal}$")
    #ax1.set_title("Galaxy Luminosity Counts")
    legend(datasets)
    ax1.set_xlim(L_MIN,L_MAX)
    #ax1.set_ylim(0.5,1E4)

    # Twin axis for Mr
    ax2=ax1.twiny()
    ax2.plot(datasets[0].Mr_gal_labels, values[0], ls="")
    ax2.set_xlim(log_solar_L_to_abs_mag_r(np.log10(L_MIN)), log_solar_L_to_abs_mag_r(np.log10(L_MAX)))
    ax2.set_xlabel("$M_r$ - 5log(h)")
    
def compare_L_funcs(one: pd.DataFrame, two: pd.DataFrame):
    one_counts = one.groupby('LGAL_BIN').RA.count()
    two_counts = two.groupby('LGAL_BIN').RA.count()
    L_func_plot([one, two], [one_counts, two_counts])


def quiescent_classification_dbl_plot(catalog: GroupCatalog, sdss: GroupCatalog):

    # See BGS_study Quiescent vs Star-Forming Analysis section
    mean_logLgal_per_bin = [8.04, 8.63, 8.97, 9.27, 9.57, 9.80, 10.00, 10.20, 10.46, 10.82]
    thresholds = [1.47, 1.5, 1.55, 1.57, 1.61, 1.63, 1.65, 1.67, 1.68, 1.69] 

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6), dpi=DPI)
    plt.sca(axes[0])
    plt.plot(mean_logLgal_per_bin, thresholds, 'o', label='Binned Thresholds Chosen', color='blue')
    x = np.linspace(6.5, 11.5, 1000)
    plt.plot(x, get_Dn4000_crit(x), label='Fitted Threshold Used Here', color='k', lw=3)
    plt.plot(x, get_SDSS_Dcrit(x), '--', label='Tinker 2021 SDSS Threshold', color='green', lw=2)
    plt.xlabel('log($L_{\\rm gal} / (L_\odot h^{-2}))$')
    plt.ylabel('$D_n4000$ Threshold')
    plt.xlim(np.log10(LGAL_MIN), LOG_LGAL_MAX_TIGHT)
    #plt.text(6.4, 1.675, f"Uses DN4000_MODEL from fastspecfit")
    #plt.text(6.4, 1.65, f"$g-r$ < 0.65 always Star-Forming")
    plt.legend()
    plt.twiny()
    plt.xlim(log_solar_L_to_abs_mag_r(6.5), log_solar_L_to_abs_mag_r(11.5))
    plt.xticks(np.arange(-23, -12, 2))
    plt.xlabel("$M_r$ - 5log($h$)")
    
    plt.sca(axes[1])
    df = catalog.all_data
    #df = df.loc[z_flag_is_spectro_z(df['Z_ASSIGNED_FLAG'])]
    #df = df.loc[df['IS_SAT'] == False]

    sdss_df = sdss.all_data
    sdss_df = sdss_df.loc[sdss_df['IS_SAT'] == False]
    qf_gmr = df.groupby('LGAL_BIN', observed=False).apply(qf_BGS_gmr_vmax_weighted)
    qf_dn4000model = df.groupby('LGAL_BIN', observed=False).apply(qf_Dn4000MODEL_smart_eq_vmax_weighted)
    plt.step(LogLgal_labels, qf_dn4000model, '-', label='Dn4000(L) Used Here', color='k', lw=3)
    #plt.plot(LogLgal_labels, df.groupby('LGAL_BIN', observed=False).apply(qf_vmax_weighted), '--', label='From Catalog', color='purple', lw=2)
    plt.step(LogLgal_labels, qf_gmr, '--', label=f'g-r(L) Alternative', color='red', lw=2)
    plt.step(LogLgal_labels, sdss_df.groupby('LGAL_BIN', observed=False).apply(qf_vmax_weighted), '--', label='Tinker 2021 SDSS', color='green', lw=2)
    plt.xlabel('log($L_{\\rm gal} / (L_\odot h^{-2}))$')
    plt.ylabel("$f_{\\mathrm{Q}}$ ")
    plt.xlim(np.log10(LGAL_MIN), LOG_LGAL_MAX_TIGHT)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.twiny()
    plt.xlim(log_solar_L_to_abs_mag_r(6.5), log_solar_L_to_abs_mag_r(11.5))
    plt.xticks(np.arange(-23, -12, 2))
    plt.xlabel("$M_r$ - 5log($h$)")


def qf_cen_plot(*datasets, test_methods=False, mstar=False):
    """
    Quiescent Fraction of Central Galaxies.
    """
    fig,ax1=plt.subplots()
    fig.set_dpi(DPI)
    groupby_property = 'Mstar_bin' if mstar else 'LGAL_BIN'
    label_property = 10**logmstar_labels if mstar else L_gal_labels
    for f in datasets:
        data = f.all_data.loc[np.all([~f.all_data['IS_SAT']], axis=0)]

        plt.plot(label_property[:-4], data.groupby(groupby_property, observed=False).apply(qf_vmax_weighted)[:-4], f.marker, label=get_dataset_display_name(f), color=f.color)

        if test_methods:
            qf_gmr = f.centrals.groupby(groupby_property, observed=False).apply(qf_BGS_gmr_vmax_weighted)
            #qf_dn4000 = f.centrals.groupby(groupby_property, observed=False).apply(qf_Dn4000_smart_eq_vmax_weighted)
            qf_dn4000model = f.centrals.groupby(groupby_property, observed=False).apply(qf_Dn4000MODEL_smart_eq_vmax_weighted)
            plt.plot(label_property, qf_gmr, '--', label=f'g-r(L)', color='g')
            #plt.plot(label_property, qf_dn4000, '-', label='Dn4000 Eq.1', color='g')
            plt.plot(label_property, qf_dn4000model, '--', label='Dn4000_M Eq. 1', color='purple')
       

    ax1.set_xscale('log')
    if not mstar:
        ax1.set_xlabel("$L_{\\mathrm{cen}}~[\\mathrm{L}_\\odot \\mathrm{h}^{-2} ]$")
        ax1.set_xlim(LGAL_MIN,LGAL_MAX)
        ax1.set_ylim(0.0,1.0)
    else:
        ax1.set_xlabel("$M_\star~[\\mathrm{M}_\\odot]$")
        ax1.set_xlim(MSTAR_MIN,MSTAR_MAX)
        #ax1.set_yscale('log')
        #ax1.set_ylim(1E-2, 1.0)
        ax1.set_ylim(0.0,1.0)
    
    ax1.set_ylabel("$f_{\\mathrm{Q}}$ ")
    #ax1.set_title("Satellite fraction vs Galaxy Luminosity")
    ax1.legend()


def hod_thresholds_plot(gc: GroupCatalog, color):
    maglims = gc.caldata.magbins[:-1]
    lumlims = abs_mag_r_to_solar_L(maglims)
    n_lcuts = len(maglims)
    ncols = int(np.ceil(n_lcuts / 2))
    nrows = 2 if n_lcuts > 1 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), dpi=DPI, sharey=True)
    axes = np.array(axes).reshape(-1)  # flatten in case axes is 2D
    logm = np.log10(gc.labels)

    for idx, lcut in enumerate(lumlims):
        ax = axes[idx]
        magcut = log_solar_L_to_abs_mag_r(np.log10(lcut))

        if color == 'r':
            ax.plot(logm, gc.hod_thresholds.central_q[idx, :], 'rd', markersize=9)
            ax.plot(logm, gc.hod_thresholds.satellite_q[idx, :], 'r.', markersize=9)
            #ax.plot(logm, gc.hod_thresholds.combined_q[idx, :], 'r-', markersize=9)
        elif color == 'b':
            ax.plot(logm, gc.hod_thresholds.central_sf[idx, :], 'bd', markersize=9)
            ax.plot(logm, gc.hod_thresholds.satellite_sf[idx, :], 'b.', markersize=9)
            #ax.plot(logm, gc.hod_thresholds.combined_sf[idx, :], 'b-', markersize=9)
        elif color == 'k':
            ax.plot(logm, gc.hod_thresholds.central_all[idx, :], color='purple', marker='d', markersize=9, linestyle='None')
            ax.plot(logm, gc.hod_thresholds.satellite_all[idx, :], color='purple', marker='.', markersize=9, linestyle='None')
            #ax.plot(logm, gc.hod_thresholds.combined_all[idx, :], 'k-', markersize=9)
        
        inset_fontsize = 13

        if hasattr(gc, 'hodt_cen_red_popt') and gc.hodt_cen_red_popt is not None:
            if color == 'r':
                ax.plot(logm, hod.hod_central_threshold_model(logm, *gc.hodt_cen_red_popt[idx]), 'k-', label='Q Central Fit', linewidth=3)
                ax.plot(logm, hod.hod_satellite_threshold_model(logm, *gc.hodt_sat_red_popt[idx]), 'k--', label='Q Satellite Fit', linewidth=3)
                ax.text(12.1, 2.7, f"$M_{{cut}}$: {gc.hodt_sat_red_popt[idx][0]:.2f}", fontsize=inset_fontsize)
                ax.text(12.1, 2.35, f"$M_\star$: {gc.hodt_sat_red_popt[idx][1]:.2f}", fontsize=inset_fontsize)
                ax.text(12.1, 2.0, f"α: {gc.hodt_sat_red_popt[idx][2]:.2f}", fontsize=inset_fontsize)
                ax.text(10.1, 2.65, f"$M_{{min}}$: {gc.hodt_cen_red_popt[idx][0]:.2f}", fontsize=inset_fontsize)
                ax.text(10.1, 2.3, f"$σ_{{min}}$: {gc.hodt_cen_red_popt[idx][1]:.2f}", fontsize=inset_fontsize)
            
            elif color == 'b':
                ax.plot(logm, hod.hod_central_threshold_blue_model(logm, *gc.hodt_cen_blue_popt[idx]), 'k-', label='SF Central Fit', linewidth=3)
                ax.plot(logm, hod.hod_satellite_threshold_model(logm, *gc.hodt_sat_blue_popt[idx]), 'k--', label='SF Satellite Fit', linewidth=3)
                ax.text(12.1, 2.7, f"$M_{{cut}}$: {gc.hodt_sat_blue_popt[idx][0]:.2f}", fontsize=inset_fontsize)
                ax.text(12.1, 2.35, f"$M_\star$: {gc.hodt_sat_blue_popt[idx][1]:.2f}", fontsize=inset_fontsize)
                ax.text(12.1, 2.0, f"α: {gc.hodt_sat_blue_popt[idx][2]:.2f}", fontsize=inset_fontsize)
                ax.text(10.1, 2.7, f"$M_{{min}}$: {gc.hodt_cen_blue_popt[idx][0]:.2f}", fontsize=inset_fontsize)
                ax.text(10.1, 2.35, f"$σ_{{min}}$: {gc.hodt_cen_blue_popt[idx][1]:.2f}", fontsize=inset_fontsize)
                ax.text(10.1, 2.0, f"$M_{{max}}$: {gc.hodt_cen_blue_popt[idx][2]:.2f}", fontsize=inset_fontsize)
                ax.text(10.1, 1.65, f"$σ_{{max}}$: {gc.hodt_cen_blue_popt[idx][3]:.2f}", fontsize=inset_fontsize)

            elif color == 'k':
                ax.plot(logm, hod.hod_central_threshold_model(logm, *gc.hodt_cen_all_popt[idx]), 'k-', label='All Central Fit', linewidth=3)
                ax.plot(logm, hod.hod_satellite_threshold_model(logm, *gc.hodt_sat_all_popt[idx]), 'k--', label='All Satellite Fit', linewidth=3)
                ax.text(12.1, 2.7, f"$M_{{cut}}$: {gc.hodt_sat_all_popt[idx][0]:.2f}", fontsize=inset_fontsize)
                ax.text(12.1, 2.35, f"$M_\star$: {gc.hodt_sat_all_popt[idx][1]:.2f}", fontsize=inset_fontsize)
                ax.text(12.1, 2.0, f"α: {gc.hodt_sat_all_popt[idx][2]:.2f}", fontsize=inset_fontsize)
                ax.text(10.1, 2.7, f"$M_{{min}}$: {gc.hodt_cen_all_popt[idx][0]:.2f}", fontsize=inset_fontsize)
                ax.text(10.1, 2.35, f"$σ_{{min}}$: {gc.hodt_cen_all_popt[idx][1]:.2f}", fontsize=inset_fontsize)
         
        if idx % ncols == 0:
            ax.set_ylabel("log$\\langle N \\rangle$")
        if nrows > 1 and idx >= ncols * (nrows - 1):
            ax.set_xlabel('log($M_h$ / [$M_\odot / h$])')

        ax.set_xlim(10,15)
        ax.set_ylim(-3, 3)
        ax.grid(True)
        ax.set_title(f"$M_r < {magcut:.1f}$")
        #ax.legend()

    # Hide unused axes if any
    #for i in range(n_lcuts, nrows * ncols):
    #    fig.delaxes(axes[i])

    #plt.tight_layout()
    #plt.draw()


def hod_bins_plot(gc: GroupCatalog, hodtab: hod.HODTabulated, model=False, seperate=False):
    """
    Plot the HOD from a file, overlaying with a histogram of the number of halos (nhalo).
    """
    data = hodtab
    n_lbins = len(data.mag_bin_centers)
    log_Mhalo = data.logM_bin_centers

    ncols = int(np.ceil(n_lbins / 2))
    nrows = 2 if n_lbins > 1 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), dpi=DPI, sharey=True)
    axes = np.array(axes).reshape(-1)  # flatten in case axes is 2D
    if seperate:
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), dpi=DPI, sharey=True)
        axes2 = np.array(axes2).reshape(-1)  # flatten in case axes is 2D

    for lbin in range(n_lbins):
        ax = axes[lbin]
        if seperate:
            ax2 = axes2[lbin]
        else:
            ax2 = ax

        alpha = 0.6
        if model:
            ax.plot(log_Mhalo, data.central_q[lbin], 'rd', label='Q Cen', markersize=9)
            ax.plot(log_Mhalo, data.satellite_q[lbin], 'r.', label='Q Sat', markersize=9)
            ax2.plot(log_Mhalo, data.central_sf[lbin], 'bd', label='SF Cen', markersize=9)
            ax2.plot(log_Mhalo, data.satellite_sf[lbin], 'b.', label='SF Sat', markersize=9)
        else:
            ax.plot(log_Mhalo, data.central_q[lbin], 'r-', label='Q Cen', alpha=alpha, linewidth=3)
            ax.plot(log_Mhalo, data.satellite_q[lbin], 'r:', label='Q Sat', alpha=alpha, linewidth=3)
            ax2.plot(log_Mhalo, data.central_sf[lbin], 'b-', label='SF Cen', alpha=alpha, linewidth=3)
            ax2.plot(log_Mhalo, data.satellite_sf[lbin], 'b:', label='SF Sat', alpha=alpha, linewidth=3)
        #else:
        #    ax.plot(log_halo_mass, log_all_cen_fraction, 'k-', label='All Cen', alpha=alpha, linewidth=3)
        #    ax.plot(log_halo_mass, log_all_sat_fraction, 'k--', label='All Sat', alpha=alpha, linewidth=3)

        if lbin / (ncols*2) >= 0.5:
            ax.set_xlabel('log($M_h$ / [$M_\odot / h$])')
            ax2.set_xlabel('log($M_h$ / [$M_\odot / h$])')
            
        ax.set_xlim(10.0, 15.0)
        ax.set_ylim(-3, 2)
        ax2.set_xlim(10.0, 15.0)
        ax2.set_ylim(-3, 2)
        ax.set_title(f'{gc.caldata.magbins[lbin]} > $M_r$ - 5log($h$) > {gc.caldata.magbins[lbin+1]}')
        ax2.set_title(f'{gc.caldata.magbins[lbin]} > $M_r$ - 5log($h$) > {gc.caldata.magbins[lbin+1]}')
        if lbin % ncols == 0:
            ax.set_ylabel('log(fraction)')
            ax2.set_ylabel('log(fraction)')

        #ax.legend(loc='upper left')
        ax.grid(True)
        ax2.grid(True)

        if model:
            ax.plot(log_Mhalo, hod.hod_central_model2(log_Mhalo, *gc.hod_cen_red_popt[lbin]), 'k-', label='Q Cen Model', linewidth=3)
            ax.plot(log_Mhalo, hod.hod_satellite_model(log_Mhalo, *gc.hod_sat_red_popt[lbin]), 'k--', label='Q Sat Model', linewidth=3)
            ax2.plot(log_Mhalo, hod.hod_central_model2(log_Mhalo, *gc.hod_cen_blue_popt[lbin]), 'k-', label='SF Cen Model', linewidth=3)
            ax2.plot(log_Mhalo, hod.hod_satellite_model(log_Mhalo, *gc.hod_sat_blue_popt[lbin]), 'k--', label='SF Sat Model', linewidth=3)

            # Print the parameters onto the plot
            ax.text(10.2, 1.7, f"α: {gc.hod_sat_red_popt[lbin][2]:.2f}")
            ax.text(10.2, 1.4, f"$M_\star$: {gc.hod_sat_red_popt[lbin][1]:.2f}")
            ax.text(10.2, 1.1, f"$M_{{cut}}$: {gc.hod_sat_red_popt[lbin][0]:.2f}")
            ax2.text(10.2, 1.7, f"α: {gc.hod_sat_blue_popt[lbin][2]:.2f}")
            ax2.text(10.2, 1.4, f"$M_\star$: {gc.hod_sat_blue_popt[lbin][1]:.2f}")
            ax2.text(10.2, 1.1, f"$M_{{cut}}$: {gc.hod_sat_blue_popt[lbin][0]:.2f}")

        #if not pretty:
        #    # Overlay nhalo histogram as a filled area on a secondary y-axis
        #    ax3 = ax.twinx()
        #    ax3.fill_between(log_Mhalo, nhalo, color='gray', alpha=0.2, step='mid', label='N_halo')
        #    if lbin % ncols == 0:
        #        ax3.set_ylabel('Number of halos')
        #    ax3.set_yscale('log')
        #    ax3.tick_params(axis='y', labelcolor='gray')

    # Hide unused axes if any
    for i in range(n_lbins, nrows * ncols):
        fig.delaxes(axes[i])
        if seperate:
            fig2.delaxes(axes2[i])

    return (fig, fig2) if seperate else fig


      

def hod_bins_plot_diff(gc1: GroupCatalog, gc2: GroupCatalog, color: str):
    """
    Plots the percentage difference in HODs between two catalogs for a given color.

    Args:
        gc1 (GroupCatalog): The first catalog (numerator).
        gc2 (GroupCatalog): The second catalog (denominator/baseline).
        color (str): The galaxy type to compare, either 'r' for quiescent or 'b' for star-forming.
    """
    if not hasattr(gc1, 'hodfit') or not hasattr(gc2, 'hodfit'):
        print("Error: Both catalogs must have a 'hodfit' attribute.")
        return
    if color not in ['r', 'b']:
        print("Error: color must be 'r' or 'b'.")
        return

    data1 = gc1.hodfit
    data2 = gc2.hodfit
    
    n_lbins = len(data1.mag_bin_centers)
    log_Mhalo = data1.logM_bin_centers

    if color == 'r':
        log_cen1, log_sat1 = data1.central_q, data1.satellite_q
        log_cen2, log_sat2 = data2.central_q, data2.satellite_q
        plot_color = 'red'
        color_label = 'Quiescent'
    else: # color == 'b'
        log_cen1, log_sat1 = data1.central_sf, data1.satellite_sf
        log_cen2, log_sat2 = data2.central_sf, data2.satellite_sf
        plot_color = 'blue'
        color_label = 'Star-forming'


    # --- BUG FIX: Convert from log-space to linear space before calculating difference ---
    cen1, sat1 = 10**log_cen1, 10**log_sat1
    cen2, sat2 = 10**log_cen2, 10**log_sat2

    # Calculate percentage difference in log-space
    with np.errstate(divide='ignore', invalid='ignore'):
        cen_diff = np.divide(cen1 - cen2, cen1) * 100
        sat_diff = np.divide(sat1 - sat2, sat1) * 100

    ncols = int(np.ceil(n_lbins / 2))
    nrows = 2 if n_lbins > 1 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), dpi=DPI, sharey=True)
    axes = np.array(axes).reshape(-1)

    for lbin in range(n_lbins):
        ax = axes[lbin]
        
        ax.plot(log_Mhalo, cen_diff[lbin], '-', color=plot_color, label='Centrals', linewidth=2)
        ax.plot(log_Mhalo, sat_diff[lbin], ':', color=plot_color, label='Satellites', linewidth=2)
        
        ax.axhline(0, color='k', linestyle='--', linewidth=1)

        if lbin >= ncols * (nrows - 1):
            ax.set_xlabel('log($M_h$ / [$M_\odot / h$])')
        
        ax.set_xlim(10.5, 15.0)
        ax.set_ylim(-200, 200)
        ax.set_title(f'{gc1.caldata.magbins[lbin]} > $M_r$ > {gc1.caldata.magbins[lbin+1]}')
        
        if lbin % ncols == 0:
            ax.set_ylabel('% Difference')

        #if lbin == 0:
            #ax.legend(loc='upper left')
        ax.grid(True)

    # Hide unused axes
    for i in range(n_lbins, nrows * ncols):
        fig.delaxes(axes[i])

    fig.suptitle(f'HOD % Difference: {gc1.name} vs {gc2.name}', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()






def proj_clustering_plot(gc: GroupCatalog):
    caldata = gc.caldata
    num = caldata.bincount
    mag_start = caldata.magbins[0]

    # Calculate number of columns for two rows
    nrows = 2
    ncols = int(np.ceil(num / nrows))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2 + 3 * ncols, 4 * nrows), dpi=DPI)
    axes = np.array(axes).reshape(-1)  # flatten for easy indexing

    overall, clust_r, clust_b, clust_nosep, lsat = gc.chisqr()

    for idx in range(len(caldata.magbins) - 1):
        i = abs(caldata.magbins[idx])

        ax = axes[idx]

        if caldata.color_separation[idx]:
            wp, wp_err, radius = caldata.get_wp_red(i)
            ax.errorbar(radius, wp, yerr=wp_err, fmt='.', color='darkred', capsize=3, ecolor='k')

            wp_mock, wp_mock_err = gc.get_mock_wp(i, 'red', wp_err)
            ax.errorbar(radius, wp_mock, yerr=wp_mock_err, fmt='-', capsize=3, color='r', alpha=0.6)

            wp, wp_err, radius = caldata.get_wp_blue(i)
            ax.errorbar(radius, wp, yerr=wp_err, fmt='.', color='darkblue', capsize=3, ecolor='k')

            wp_mock, wp_mock_err = gc.get_mock_wp(i, 'blue', wp_err)
            ax.errorbar(radius, wp_mock, yerr=wp_mock_err, fmt='-', capsize=3, color='b', alpha=0.6)

            # Put text of the chisqr value in plot
            ax.text(0.6, 0.9, f"$\chi^2_q$: {clust_r[i+mag_start]:.1f}", transform=ax.transAxes)
            ax.text(0.6, 0.78, f"$\chi^2_{{sf}}$: {clust_b[i+mag_start]:.1f}", transform=ax.transAxes)

        else:
            wp, wp_err, radius = caldata.get_wp_all(i)
            ax.errorbar(radius, wp, yerr=wp_err, fmt='.', color='k', capsize=3, ecolor='k')

            wp_mock, wp_mock_err = gc.get_mock_wp(i, 'all', wp_err)
            ax.errorbar(radius, wp_mock, yerr=wp_mock_err, fmt='-', capsize=3, color='k', alpha=0.6)

            # Put text of the chisqr value in plot
            ax.text(0.6, 0.9, f"$\chi^2$: {clust_nosep[i+mag_start]:.1f}", transform=ax.transAxes)

        # Plot config
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('$r_p$ [Mpc $h^{-1}$]')
        ax.set_ylabel('$w_p(r_p)$')
        ax.set_ylim(1, 4000)
        ax.set_title(f'[{-i}, {caldata.magbins[idx+1]}]')

    # Hide unused axes if any
    for j in range(len(caldata.magbins) - 1, nrows * ncols):
        fig.delaxes(axes[j])

    fig.tight_layout()

def lsat_data_compare_plot(gc: GroupCatalog):
    data = np.loadtxt(LSAT_OBSERVATIONS_SDSS_FILE, skiprows=0, dtype='float')
    lsat_compare_plot(data, gc.lsat_r, gc.lsat_b, None, None)

def lsat_data_compare_werr_plot(gc: GroupCatalog):
    data = np.loadtxt(LSAT_OBSERVATIONS_SDSS_FILE, skiprows=0, dtype='float')
    lsat_r_mean, lsat_r_std, lsat_b_mean, lsat_b_std = lsat_variance_from_saved()
    lsat_compare_plot(data, gc.lsat_r, gc.lsat_b, lsat_r_std, lsat_b_std)

def lsat_compare_plot(data, lsat_r, lsat_b, lsat_r_std, lsat_b_std):
    chisqr = compute_lsat_chisqr(data, lsat_r, lsat_b)

    ratio = lsat_r/lsat_b
    if lsat_r_std is not None and lsat_b_std is not None:
        ratio_err = ratio * ((lsat_r_std/lsat_r)**2 + (lsat_b_std/lsat_b)**2)**.5
        ratio_err_log = ratio_err/ratio/np.log(10)

    # Get Mean Lsat for r/b centrals from SDSS data
    obs_lcen = data[:,0] # log10 already
    obs_lsat_r = data[:,1]
    obs_err_r = data[:,2]
    obs_lsat_b = data[:,3]
    obs_err_b = data[:,4]
    obs_ratio = obs_lsat_r/obs_lsat_b
    obs_ratio_err = obs_ratio * ((obs_err_r/obs_lsat_r)**2 + (obs_err_b/obs_lsat_b)**2)**.5
    obs_ratio_err_log = obs_ratio_err/obs_ratio/np.log(10)

    fig,axes=plt.subplots(nrows=1, ncols=1, figsize=(5,5), dpi=DPI)

    axes.errorbar(obs_lcen, np.log10(obs_ratio), yerr=obs_ratio_err_log, fmt='o', color='k', markersize=3, capsize=3, ecolor='k', label='SDSS Data')
    
    if lsat_r_std is not None and lsat_b_std is not None:
        axes.errorbar(obs_lcen, np.log10(ratio), yerr=ratio_err_log, fmt='-', color='purple', markersize=3, capsize=3, ecolor='purple', label='Group Finder', alpha=0.8)
    else:
        axes.plot(obs_lcen, np.log10(ratio), '-', color='purple', label='Group Finder', alpha=0.8)
    axes.set_ylabel('log$(L_{sat}^{q}/L_{sat}^{sf})$')
    axes.set_xlabel('log$(L_{cen}~/~[L_\odot / h^2])$')
    axes.set_ylim(-0.2, 0.5)
    axes.legend()

    # Put text of the chisqr value in plot
    axes.text(.4,.93, f"$\chi^2$: {np.sum(chisqr):.1f}", transform=axes.transAxes)

    # Twin x for Mr
    ax2=axes.twiny()
    ax2.plot(obs_lcen, np.log10(obs_ratio), ls="")
    ax2.set_xlim(log_solar_L_to_abs_mag_r(obs_lcen[0]), log_solar_L_to_abs_mag_r(obs_lcen[-1]))
    ax2.set_xlabel("$M_r$ - 5log($h$)")
    

    #axes[1].plot(obs_lcen, np.log10(lsat_r), label='GF Quiescent', color='r')
    #axes[1].plot(obs_lcen, np.log10(lsat_b), label='GF Star-forming', color='b')
    #axes[1].errorbar(obs_lcen, np.log10(obs_lsat_r), yerr=obs_err_r/obs_lsat_r, label='SDSS Quiescent', fmt='o', color='r', markersize=3, capsize=2, ecolor='k')
    #axes[1].errorbar(obs_lcen, np.log10(obs_lsat_b), yerr=obs_err_b/obs_lsat_b, label='SDSS Star-Forming', fmt='o', color='b', markersize=3, capsize=2, ecolor='k')
    #axes[1].set_ylabel('log $L_{sat}~[L_\odot / h^2]$')
    #axes[1].set_xlabel('log $L_{cen}~[L_\odot / h^2]$')
    #axes[1].legend()

    #fig.suptitle(gc.name)
    fig.tight_layout()

    # TODO "lsat_groups_propx_red.out"


    
# Halo Masses (in group finder abundance matching)
def group_finder_centrals_halo_masses_plots(all_df, comparisons):
    
    count_to_use = np.max([len(c.all_data[c.all_data.index == c.all_data['IGRP']]) for c in comparisons])
    
    all_centrals = all_df.all_data[all_df.all_data.index == all_df.all_data['IGRP']]

    #if len(all_df.all_data) > count_to_use:
    #    # Randomly sample to match the largest comparison
    #    print(len(all_centrals))
    #    print(count_to_use)
    #    reduced_all_centrals_halos = np.random.choice(all_centrals['M_HALO'], count_to_use, replace=False)

    #angdist_bin_ind = np.digitize(reduced_all_centrals_halos, all_df.Mhalo_bins)
    angdist_bin_ind = np.digitize(all_centrals['M_HALO'], all_df.Mhalo_bins)
    all_bincounts = np.bincount(angdist_bin_ind)[0:len(all_df.Mhalo_bins)]
    all_density = all_bincounts / np.sum(all_bincounts)

    fig,axes=plt.subplots(nrows=1, ncols=1, figsize=(6,4))
    fig.set_dpi(DPI)
    axes.set_xscale('log')
    axes.set_ylim(-0.2, 0.2)
    axes.set_xlim(5E10,2E15)
    axes.set_xlabel('$M_h$')
    axes.set_ylabel('Change in log(M)')
    axes.axline((3E10,0), (3E15,0), linestyle='--', color='k')
    #axes.set_title("Group Finder Halo Masses of Centrals")

    #axes[1].plot(all_to_use.Mhalo_bins, all_density, label="All Galaxies") 
    #axes[1].set_xscale('log')
    #axes[1].set_yscale('log')
    #axes[1].set_xlim(5E10,2E15)
    #axes[1].set_xlabel('$M_h$')
    #axes[1].set_ylabel('Density of Galaxies')
    #axes[1].set_title("Group Finder Halo Masses of Centrals")

    for comparison in comparisons:

        centrals = comparison.all_data[comparison.all_data.index == comparison.all_data['IGRP']]
        angdist_bin_ind = np.digitize(centrals['M_HALO'], all_df.Mhalo_bins)
        bincounts = np.bincount(angdist_bin_ind)[0:len(all_df.Mhalo_bins)]
        density = bincounts / np.sum(bincounts)

        axes.plot(all_df.Mhalo_bins, np.log10(density / all_density), linestyle=comparison.marker, color=comparison.color, label=comparison.name) 
        #axes[1].plot(all_to_use.Mhalo_bins, density, linestyle=comparison.marker, color=comparison.color, label=comparison.name) 

    axes.legend()
    #axes[1].legend()

    # Look up the centrals from all in fiberonly
    for comparison in comparisons:

        centrals = comparison.all_data[comparison.all_data.index == comparison.all_data['IGRP']]
        catalog = coord.SkyCoord(ra=all_centrals.RA.to_numpy()*u.degree, dec=all_centrals['DEC'].to_numpy()*u.degree, frame='icrs')
        to_match = coord.SkyCoord(ra=centrals.RA.to_numpy()*u.degree, dec=centrals['DEC'].to_numpy()*u.degree, frame='icrs')
        idx, d2d, d3d = coord.match_coordinates_sky(to_match, catalog, nthneighbor=1, storekdtree=False)

        perfect_match = np.isclose(d2d.to(u.arcsec).value, 0, rtol=0.0, atol=0.0001) 
        # 0.0001 arcsec precision on matching doesn't hit floating point noise. You get same with 0.001
        print(f"What fraction of centrals in \'{comparison.name}\' are centrals in \'all\'? {np.sum(perfect_match) / len(d2d)}")



##################################
# Luminosity Funcions
##################################

def Lfunc_compare(cat1, cat2):
    """
    TODO BUG Incomplete.
    Compare the luminosity functions of two catalogs. Shows a % difference plot for total, red, and blue galaxies.
    The total count of galaxies is scaled to be the same in the two catalogs (for overall, red, blue seperately) so only the relative difference in Lfunc is shown.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.set_dpi(DPI)
    one = cat1.all_data
    two = cat2.all_data

    # Calculate the luminosity functions for both catalogs
    counts1 = one.groupby('LGAL_BIN').size()
    counts2 = two.groupby('LGAL_BIN').size()

    # Normalize the counts to the same total number of galaxies
    norm_counts1 = counts1 / counts1.sum()
    norm_counts2 = counts2 / counts2.sum()

    # Calculate the percentage difference
    percent_diff = 100 * (norm_counts1 - norm_counts2) / norm_counts2

    # Plot the percentage difference
    ax.plot(cat1.L_gal_labels, percent_diff, label='Total', color='black')

    # Repeat for red and blue galaxies
    counts1_red = one[one['QUIESCENT']].groupby('LGAL_BIN').size()
    counts2_red = two[two['QUIESCENT']].groupby('LGAL_BIN').size()
    norm_counts1_red = counts1_red / counts1_red.sum()
    norm_counts2_red = counts2_red / counts2_red.sum()
    percent_diff_red = 100 * (norm_counts1_red - norm_counts2_red) / norm_counts2_red
    ax.plot(cat1.L_gal_labels, percent_diff_red, label='Red', color='red')

    counts1_blue = one[~one['QUIESCENT']].groupby('LGAL_BIN').size()
    counts2_blue = two[~two['QUIESCENT']].groupby('LGAL_BIN').size()
    norm_counts1_blue = counts1_blue / counts1_blue.sum()
    norm_counts2_blue = counts2_blue / counts2_blue.sum()
    percent_diff_blue = 100 * (norm_counts1_blue - norm_counts2_blue) / norm_counts2_blue
    ax.plot(cat1.L_gal_labels, percent_diff_blue, label='Blue', color='blue')

    ax.set_xscale('log')
    ax.set_xlabel('$L_{\\mathrm{gal}}~[\\mathrm{L}_\\odot \\mathrm{h}^{-2} ]$')
    ax.set_ylabel('% Difference in counts')
    ax.legend()
    ax.set_xlim(LGAL_MIN, LGAL_MAX)
    ax.set_ylim(-35, 35)
    ax.axhline(0, color='black', lw=1)
    fig.tight_layout()

def lostgal_lum_func_paper_compare(*catalogs):
        
    #x = L_gal_labels # bins if ising cic code
    x_bins = np.logspace(8, 11.2, 12) # bins if using cic code
    x = (x_bins[:-1] + x_bins[1:]) / 2 # bin centers

    obs_counts = np.zeros((len(catalogs), len(x)))
    lost_truth_counts = np.zeros((len(catalogs), len(x)))
    lost_assumed_counts = np.zeros((len(catalogs), len(x)))
    obs_vs_losttruth = np.zeros((len(catalogs), len(x)))
    assumed_vs_truth = np.zeros((len(catalogs), len(x)))
    assumed_vs_truth_red = np.zeros((len(catalogs), len(x)))
    assumed_vs_truth_blue = np.zeros((len(catalogs), len(x)))
    total_counts = np.zeros((len(catalogs), len(x)))
    total_r_counts = np.zeros((len(catalogs), len(x)))
    total_b_counts = np.zeros((len(catalogs), len(x)))
    boost = []

    for i in range(len(catalogs)):
        assert len(catalogs[i].L_gal_labels) == len(L_gal_labels)
        catalog = catalogs[i]
        #data = catalog.all_data.convert_dtypes()
        data = catalog.all_data

        lostrows = z_flag_is_not_spectro_z(data['Z_ASSIGNED_FLAG'].to_numpy())
        lost_and_havetruth_rows = np.logical_and(z_flag_is_not_spectro_z(data['Z_ASSIGNED_FLAG'].to_numpy()), data['Z_T'].to_numpy() > 0)
        lost_withT_galaxies = data.loc[lost_and_havetruth_rows]
        obs_galaxies = data.loc[~lostrows]

        closeness = np.isclose(obs_galaxies['Z'], obs_galaxies['Z_T'], atol=SIM_Z_THRESH, rtol=0.0)
        assert closeness.sum() / len(closeness) > .98, f"{1 - (closeness.sum() / len(closeness))} of the galaxies have different z and Z_T despite being spectrocopically observed."

        if len(lost_withT_galaxies) == 0:
            boost.append(1)
            continue
        boost.append(len(obs_galaxies) / len(lost_withT_galaxies))

        total_counts[i] = np.histogram(data['L_GAL'].to_numpy(), bins=x_bins)[0]
        total_r_counts[i] = np.histogram(data.loc[data['QUIESCENT'], 'L_GAL'].to_numpy(), bins=x_bins)[0]
        total_b_counts[i] = np.histogram(data.loc[~data['QUIESCENT'], 'L_GAL'].to_numpy(), bins=x_bins)[0]
        obs_counts[i] = np.histogram(obs_galaxies['L_GAL'].to_numpy(), bins=x_bins)[0]
        lost_truth_counts[i] = np.histogram(lost_withT_galaxies['L_GAL_T'].to_numpy(), bins=x_bins)[0]
        lost_assumed_counts[i] = np.histogram(lost_withT_galaxies['L_GAL'].to_numpy(), bins=x_bins)[0]
        
        obs_vs_losttruth[i] = ((lost_truth_counts[i]*boost[i] - obs_counts[i]) / obs_counts[i]) * 100
        assumed_vs_truth[i] =  ((lost_assumed_counts[i] - lost_truth_counts[i]) / lost_truth_counts[i]) * 100

        obs_red_counts = np.histogram(obs_galaxies.loc[obs_galaxies['QUIESCENT'], 'L_GAL'].to_numpy(), bins=x_bins)[0]
        obs_blue_counts = np.histogram(obs_galaxies.loc[~obs_galaxies['QUIESCENT'], 'L_GAL'].to_numpy(), bins=x_bins)[0]
        lost_truth_red_counts = np.histogram(lost_withT_galaxies.loc[lost_withT_galaxies['QUIESCENT'], 'L_GAL_T'].to_numpy(), bins=x_bins)[0].astype(np.float64)
        lost_truth_blue_counts = np.histogram(lost_withT_galaxies.loc[~lost_withT_galaxies['QUIESCENT'], 'L_GAL_T'].to_numpy(), bins=x_bins)[0].astype(np.float64)
        lost_assumed_red_counts = np.histogram(lost_withT_galaxies.loc[lost_withT_galaxies['QUIESCENT'], 'L_GAL'].to_numpy(), bins=x_bins)[0].astype(np.float64)
        lost_assumed_blue_counts = np.histogram(lost_withT_galaxies.loc[~lost_withT_galaxies['QUIESCENT'], 'L_GAL'].to_numpy(), bins=x_bins)[0].astype(np.float64)

        # Set to nan if less than 10 galaxies in the bin to avoid noisy statistics
        lim = 20
        lost_truth_red_counts[lost_truth_red_counts < lim] = np.nan
        lost_truth_blue_counts[lost_truth_blue_counts < lim] = np.nan
        lost_assumed_red_counts[lost_assumed_red_counts < lim] = np.nan
        lost_assumed_blue_counts[lost_assumed_blue_counts < lim] = np.nan

        assumed_vs_truth_red[i] =  ((lost_assumed_red_counts - lost_truth_red_counts) / lost_truth_red_counts) * 100
        assumed_vs_truth_blue[i] =  ((lost_assumed_blue_counts - lost_truth_blue_counts) / lost_truth_blue_counts) * 100

    percent_lim = 100

    fig, axes = plt.subplots(nrows = 1, ncols=2, figsize=(12, 5))
    for i in range(len(catalogs)):
        axes[0].plot(x , assumed_vs_truth_red[i], label=f"{catalogs[i].name}", color=catalogs[i].color)
    #axes[0].set_title("Change in Red Lost Galaxy $\Phi_L(L)$")
    axes[0].set_xlabel('$L_{\\rm gal}~[L_{\\odot} h^{-2}]$')
    axes[0].set_ylabel("$\Delta \Phi_{L}^{q}(L)$ (%)")
    axes[0].set_xscale('log')
    axes[0].set_xlim(1e8, LGAL_MAX_TIGHT)
    axes[0].set_ylim(-percent_lim, percent_lim)
    axes[0].axhline(0, color='black', lw=1)
    axes[0].text(0.65, 0.05, "Quiescent", transform=axes[0].transAxes)
    ax2 = axes[0].twiny()
    ax2.set_xscale('linear')
    ax2.set_xlim(log_solar_L_to_abs_mag_r(8), log_solar_L_to_abs_mag_r(np.log10(LGAL_MAX_TIGHT)))
    ax2.set_xlabel('$M_r - 5log(h)$')

    for i in range(len(catalogs)):
        axes[1].plot(x , assumed_vs_truth_blue[i], label=f"{catalogs[i].name}", color=catalogs[i].color)
    #axes[1].set_title("Change in Blue Lost Galaxy $\Phi_L(L)$")
    axes[1].set_xlabel('$L_{\\rm gal}~[L_{\\odot} h^{-2}]$')
    axes[1].set_ylabel("$\Delta \Phi_{L}^{sf}(L)$ (%)")
    axes[1].set_xscale('log')
    axes[1].set_xlim(1e8, LGAL_MAX_TIGHT)
    axes[1].set_ylim(-percent_lim, percent_lim)
    axes[1].axhline(0, color='black', lw=1)
    axes[1].text(0.6, 0.05, "Star-forming", transform=axes[1].transAxes)
    ax3 = axes[1].twiny()
    ax3.set_xscale('linear')
    ax3.set_xlim(log_solar_L_to_abs_mag_r(8), log_solar_L_to_abs_mag_r(np.log10(LGAL_MAX_TIGHT)))
    ax3.set_xlabel('$M_r - 5log(h)$')

    axes[1].legend()
    fig.tight_layout()


def luminosity_function_plots(*catalogs):
    
    #x = L_gal_labels # bins if ising cic code
    x = np.logspace(8, 11, 12) # bins if using cic code

    obs_counts = np.zeros((len(catalogs), len(x)))
    lost_truth_counts = np.zeros((len(catalogs), len(x)))
    lost_assumed_counts = np.zeros((len(catalogs), len(x)))
    obs_vs_losttruth = np.zeros((len(catalogs), len(x)))
    assumed_vs_truth = np.zeros((len(catalogs), len(x)))
    assumed_vs_truth_red = np.zeros((len(catalogs), len(x)))
    assumed_vs_truth_blue = np.zeros((len(catalogs), len(x)))
    total_counts = np.zeros((len(catalogs), len(x)))
    total_r_counts = np.zeros((len(catalogs), len(x)))
    total_b_counts = np.zeros((len(catalogs), len(x)))
    boost = []

    for i in range(len(catalogs)):
        assert len(catalogs[i].L_gal_labels) == len(L_gal_labels)
        catalog = catalogs[i]
        #data = catalog.all_data.convert_dtypes()
        data = catalog.all_data

        lostrows = z_flag_is_not_spectro_z(data['Z_ASSIGNED_FLAG'].to_numpy())
        lost_and_havetruth_rows = np.logical_and(z_flag_is_not_spectro_z(data['Z_ASSIGNED_FLAG'].to_numpy()), data['Z_T'].to_numpy() > 0)
        lost_withT_galaxies = data.loc[lost_and_havetruth_rows]
        obs_galaxies = data.loc[~lostrows]

        closeness = np.isclose(obs_galaxies['Z'], obs_galaxies['Z_T'], atol=SIM_Z_THRESH, rtol=0.0)
        assert closeness.sum() / len(closeness) > .98, f"{1 - (closeness.sum() / len(closeness))} of the galaxies have different z and Z_T despite being spectrocopically observed."

        if len(lost_withT_galaxies) == 0:
            boost.append(1)
            continue
        boost.append(len(obs_galaxies) / len(lost_withT_galaxies))

        # Non CIC way looks quite different...
        #obs_counts[i] = obs_galaxies.groupby('LGAL_BIN_T', observed=False).size()
        #lost_truth_counts[i] = lost_withT_galaxies.groupby('LGAL_BIN_T', observed=False).size()
        #lost_assumed_counts[i] = lost_withT_galaxies.groupby('LGAL_BIN', observed=False).size()

        total_counts[i] = nn.cic_binning(data['L_GAL'].to_numpy(), [x])
        total_r_counts[i] = nn.cic_binning(data.loc[data['QUIESCENT'], 'L_GAL'].to_numpy(), [x])
        total_b_counts[i] = nn.cic_binning(data.loc[~data['QUIESCENT'], 'L_GAL'].to_numpy(), [x])
        obs_counts[i] = nn.cic_binning(obs_galaxies['L_GAL'].to_numpy(), [x])
        lost_truth_counts[i] = nn.cic_binning(lost_withT_galaxies['L_GAL_T'].to_numpy(), [x])
        lost_assumed_counts[i] = nn.cic_binning(lost_withT_galaxies['L_GAL'].to_numpy(), [x])
        
        obs_vs_losttruth[i] = ((lost_truth_counts[i]*boost[i] - obs_counts[i]) / obs_counts[i]) * 100
        assumed_vs_truth[i] =  ((lost_assumed_counts[i] - lost_truth_counts[i]) / lost_truth_counts[i]) * 100

        obs_red_counts = nn.cic_binning(obs_galaxies.loc[obs_galaxies['QUIESCENT'], 'L_GAL'].to_numpy(), [x])
        obs_blue_counts = nn.cic_binning(obs_galaxies.loc[~obs_galaxies['QUIESCENT'], 'L_GAL'].to_numpy(), [x])
        lost_truth_red_counts = nn.cic_binning(lost_withT_galaxies.loc[lost_withT_galaxies['QUIESCENT'], 'L_GAL_T'].to_numpy(), [x])
        lost_truth_blue_counts = nn.cic_binning(lost_withT_galaxies.loc[~lost_withT_galaxies['QUIESCENT'], 'L_GAL_T'].to_numpy(), [x])
        lost_assumed_red_counts = nn.cic_binning(lost_withT_galaxies.loc[lost_withT_galaxies['QUIESCENT'], 'L_GAL'].to_numpy(), [x])
        lost_assumed_blue_counts = nn.cic_binning(lost_withT_galaxies.loc[~lost_withT_galaxies['QUIESCENT'], 'L_GAL'].to_numpy(), [x])

        assumed_vs_truth_red[i] =  ((lost_assumed_red_counts - lost_truth_red_counts) / lost_truth_red_counts) * 100
        assumed_vs_truth_blue[i] =  ((lost_assumed_blue_counts - lost_truth_blue_counts) / lost_truth_blue_counts) * 100

        with np.printoptions(precision=0, suppress=True, linewidth=200):
            print(f"\n*** {catalogs[i].name} ***")
            print(f"Lost rows: {lostrows.sum()}")
            print(f"Lost with T rows: {len(lost_withT_galaxies)}")
            print(f"Observed rows: {len(obs_galaxies)}")
            print(f"Red:  \nObseved: {obs_red_counts} \nLost (truth): {lost_truth_red_counts} \nLost (assumed): {lost_assumed_red_counts}")
            print(f"Blue: \nObseved: {obs_blue_counts} \nLost (truth): {lost_truth_blue_counts} \nLost (assumed): {lost_assumed_blue_counts}")

        plt.figure(dpi=DPI)
        plt.plot(x, total_counts[i], color='k', label='Total galaxies')
        plt.plot(x, total_r_counts[i], color='r', label='Total quiescent')
        plt.plot(x, total_b_counts[i], color='b', label='Total star-forming')

        """
        plt.figure(dpi=DPI)
        plt.plot(x, obs_counts[i], color='b', label='Obs galaxies')
        plt.plot(x, lost_truth_counts[i]*boost[i], color='g', label='Lost gals (True)')
        plt.plot(x, lost_assumed_counts[i]*boost[i], color='orange', label='Lost gals (Assumed)')
        plt.legend()
        plt.title("$\Phi(L)$ - " + catalog.name)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim(LGAL_MIN, LGAL_MAX)
        plt.xlabel('$L_{gal}$')
        plt.draw()
        """
        # Split by quiescent and star-forming
        plt.figure(dpi=DPI)
        plt.plot(x, assumed_vs_truth_red[i], color='r', label=f'Quiescent ({catalog.name})')
        plt.plot(x, assumed_vs_truth_blue[i], color='b', label=f'Star-forming ({catalog.name})')
        plt.title("Lost $\Phi_{Truth}(L)$ -> $\Phi_{Assumed}(L)$ - " + catalog.name)
        plt.xlabel('$L_{gal}$')
        plt.ylabel("% Change in counts")
        plt.xscale('log')
        plt.xlim(LGAL_MIN, LGAL_MAX)
        plt.ylim(-100, 100)
        plt.legend()
        plt.axhline(0, color='black', lw=1)
        plt.draw()

    plt.figure(dpi=DPI)
    for i in range(len(catalogs)):
        plt.plot(x, obs_vs_losttruth[i], label=f"{catalogs[i].name}")
    plt.title("Obs => Lost (Truth) Luminosity Function")
    plt.xlabel('$L_{gal}$')
    plt.ylabel("% Change in counts")
    plt.xscale('log')
    plt.xlim(LGAL_MIN, LGAL_MAX)
    plt.ylim(-35, 35)
    plt.legend()
    plt.axhline(0, color='black', lw=1)
    plt.draw()

    plt.figure(dpi=DPI)
    for i in range(len(catalogs)):
        plt.plot(x , assumed_vs_truth[i], label=f"{catalogs[i].name}", color=catalogs[i].color)
    plt.title("Change in Lost Galaxy $\Phi_L(L)$")
    plt.xlabel('$L_{gal}$')
    plt.ylabel("$\Delta \Phi_L(L)$ (%)")
    plt.xscale('log')
    plt.xlim(LGAL_MIN, LGAL_MAX)
    plt.ylim(-60, 60)
    plt.legend()
    plt.axhline(0, color='black', lw=1)
    plt.draw()

    fig, axes = plt.subplots(nrows = 1, ncols=2, figsize=(10, 4), dpi=DPI)
    for i in range(len(catalogs)):
        axes[0].plot(x , assumed_vs_truth_red[i], label=f"{catalogs[i].name}", color=catalogs[i].color)
    #axes[0].set_title("Change in Red Lost Galaxy $\Phi_L(L)$")
    axes[0].set_xlabel('$L_{gal}$')
    axes[0].set_ylabel("$\Delta \Phi_{L,r}(L)$ (%)")
    axes[0].set_xscale('log')
    axes[0].set_xlim(LGAL_MIN, LGAL_MAX)
    axes[0].set_ylim(-60, 60)
    axes[0].axhline(0, color='black', lw=1)
    for i in range(len(catalogs)):
        axes[1].plot(x , assumed_vs_truth_blue[i], label=f"{catalogs[i].name}", color=catalogs[i].color)
    #axes[1].set_title("Change in Blue Lost Galaxy $\Phi_L(L)$")
    axes[1].set_xlabel('$L_{gal}$')
    axes[1].set_ylabel("$\Delta \Phi_{L,b}(L)$ (%)")
    axes[1].set_xscale('log')
    axes[1].set_xlim(LGAL_MIN, LGAL_MAX)
    axes[1].set_ylim(-60, 60)
    axes[1].axhline(0, color='black', lw=1)
    plt.legend()
    fig.tight_layout()
    plt.draw()



#####################################
# Correct Redshifts Assigned
#####################################
def correct_redshifts_assigned_plot(*sets: GroupCatalog):
    scores_all_lost = []
    scores_n_only = []
    rounded_tophat_scores = []
    scores_metric1 = []
    scores_metric2 = []
    scores_metric3 = []
    scores_metric4 = []

    for s in sets:
        print(f"*** Summarizing results for {s.name} ***")
        print(f"Total Galaxies: {len(s.all_data)}. Lost Galaxies: {z_flag_is_not_spectro_z(s.all_data['Z_ASSIGNED_FLAG']).sum()}")
        
        # Truth method 1, take from other catalog
        zt1 = (s.all_data['Z_T'] < -1 ) # Looking for sentinal values
        zt2 = (np.isnan(s.all_data['Z_T']))
        zt3 = (s.all_data['Z_T'] > 20.0) # Looking for sentinal values
        valid_idx = ~np.any([zt1, zt2, zt3], axis=0)
        idx = valid_idx & (z_flag_is_not_spectro_z(s.all_data['Z_ASSIGNED_FLAG']))
        print(f"There are {idx.sum()} lost galaxies with 'truth'. {zt1.sum()} {zt2.sum()} {zt3.sum()}")

        # Truth method 2 for SV3 where we dropped passes. Should be equivalent to the above... but not quite? TODO BUG
        #zt1a = (s.all_data['Z_OBS'] < -1 ) # Looking for sentinal values
        #zt2a = (np.isnan(s.all_data['Z_OBS']))
        #zt3a = (s.all_data['Z_OBS'] > 20.0) # Looking for sentinal values
        #valid_idx_alt = ~np.any([zt1a, zt2a, zt3a], axis=0)
        #idx_alt = valid_idx_alt & (z_flag_is_not_spectro_z(s.all_data['Z_ASSIGNED_FLAG']))
        #print(f"There are {idx_alt.sum()} lost galaxies with 'truth' (alt). {zt1a.sum()} {zt2a.sum()} {zt3a.sum()}")

        assigned_z = s.all_data.loc[idx, 'Z'] 
        truth_z = s.all_data.loc[idx, 'Z_T'] # was using z_obs before which is fine for SV3 with droped passes
        assignment_type = s.all_data.loc[idx, 'Z_ASSIGNED_FLAG']

        score = powerlaw_score_1(assigned_z, truth_z)
        #rtophat = rounded_tophat_score(assigned_z, truth_z)
        rtophat = close_enough(assigned_z, truth_z)
        metric_score1 = photoz_plus_metric_1(assigned_z, truth_z, assignment_type)
        metric_score2 = photoz_plus_metric_2(assigned_z, truth_z, assignment_type)
        metric_score3 = photoz_plus_metric_3(assigned_z, truth_z, assignment_type)
        metric_score4 = photoz_plus_metric_4(assigned_z, truth_z, assignment_type)
        scores_all_lost.append(score.mean())
        rounded_tophat_scores.append(rtophat.mean())
        scores_metric1.append(metric_score1 * -1)
        scores_metric2.append(metric_score2 * -1)
        scores_metric3.append(metric_score3 * -1)
        scores_metric4.append(metric_score4 * -1)
        #print(f" Galaxies to compare: {len(assigned_z)} ({len(assigned_z) / len(s.all_data):.1%})")
        print(f" Neighbor z used {s.get_lostgal_neighbor_used():.1%}")
        print(f" Score Mean: {rtophat.mean():.4f}")

        # Calculate and print off close_enough_smooth for each z_assigned_flag value seperately
        for flag in np.unique(assignment_type):
            assigned_z_flag = assigned_z[assignment_type == flag]
            truth_z_flag = truth_z[assignment_type == flag]
            rtophat_flag = rounded_tophat_score(assigned_z_flag, truth_z_flag)
            print(f"Flag {AssignedRedshiftFlag(flag)} (used {np.sum(assignment_type == flag)}) - Score Mean: {rtophat_flag.mean():.4f}")

        #print("Neighbor-assigned Only:")
        assigned_z2 = s.all_data.loc[valid_idx & z_flag_is_neighbor(s.all_data['Z_ASSIGNED_FLAG']), 'Z']
        observed_z2 = s.all_data.loc[valid_idx & z_flag_is_neighbor(s.all_data['Z_ASSIGNED_FLAG']), 'Z_T']
        score2 = rounded_tophat_score(assigned_z2, observed_z2)
        scores_n_only.append(score2.mean())
        #print(f" Galaxies to compare: {len(assigned_z2)} ({len(assigned_z2) / len(s.all_data):.1%})")
        #print(f" Score Mean: {score2.mean():.4f}")

    # Plotting the results
    labels = [s.name for s in sets]
    x = np.arange(len(labels))  # the label locations
    width = 1/3  # the width of the bars; change based on how many bars you want

    fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
    #rects1 = ax.bar(x - width, scores_all_lost, width, label='Score (Powerlaw kernel)', color=get_color(0))
    #rects5 = ax.bar(x + width, scores_metric3, width, label='Metric 3', color=get_color(3))
    
    #rects2 = ax.bar(x - width/2        , rounded_tophat_scores, width, label='Fraction Correct', color=get_color(2))
    #rects5 = ax.bar(x + width/2, scores_metric4, width, label='MCMC Metric Score', color=get_color(3))

    rects2 = ax.bar(x -width/2       , rounded_tophat_scores, width, label='Fraction Correct (all)', color=get_color(2))
    #rects3 = ax.bar(x + width/2  , scores_n_only, width, label='Fraction Correct (neighbor assigned)', color=get_color(1))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    #ax.set_xlabel('Catalogs')
    ax.set_title('Correct Redshift Assignment')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, ha='right')
    #ax.legend()
    ax.set_ylim(0, 0.8)
    ax.set_yticks(np.arange(0, 0.8, 0.1))
    ax.yaxis.grid(True)  # Add horizontal gridlines

    fig.tight_layout()

    plt.show()


#####################################
# Purity and Completeness Functions
#####################################


def build_interior_bin_labels(bin_edges):
    labels = []
    for i in range(0,len(bin_edges)-1):
        labels.append(f"{bin_edges[i]:.2e} - {bin_edges[i+1]:.2e}")
    return labels

def test_purity_and_completeness(*catalogs: GroupCatalog, truth_catalog: GroupCatalog, color=None, lost_only=False):
    
    for s in catalogs:
        s.get_true_z_from(truth_catalog.all_data)
        s.refresh_df_views()

        good_rows = s.all_data['Z_T'] != NO_TRUTH_Z
        #print(f"Missing truth on {len(s.all_data) - np.sum(good_rows)} galaxies")
        if lost_only:
            good_rows &= z_flag_is_not_spectro_z(s.all_data['Z_ASSIGNED_FLAG'])

        if color is not None:
            good_rows &= s.all_data['QUIESCENT'].astype(bool) == color

        data = s.all_data.loc[good_rows]
        print(f"-=={get_dataset_display_name(s)}==-", f"({len(data)})")

        #if not s.has_truth:
        #    data['IS_SAT_T'] = np.logical_or(data.galaxy_type == 1, data.galaxy_type == 3)

        # No 1/vmax for these sums as we're just trying to calculate the purity/completeness of our sample

        # Of all the assigned sats, how many are actually sats?
        assigned_sats = data.loc[data['IS_SAT']]
        s.overall_purity_sats = np.sum(assigned_sats.IS_SAT_T.astype(bool)) / len(assigned_sats.index)
        print(f"Purity of sats: {s.overall_purity_sats:.3f}")

        # Of all the sats in the truth catalog, how many were found?
        true_sats = data.loc[data['IS_SAT_T'].astype(bool)]
        true_sats_assigned_correctly = true_sats.loc[true_sats['IS_SAT']]
        s.overall_completeness_sats = len(true_sats_assigned_correctly.index) / len(true_sats.index)
        print(f"Completeness of sats: {s.overall_completeness_sats:.3f}")

        # Of all the assigned centrals, how many are actually centrals?
        assigned_centrals = data.loc[~data['IS_SAT']]
        s.overall_purity_centrals = np.sum(~assigned_centrals.IS_SAT_T.astype(bool)) / len(assigned_centrals.index)
        print(f"Purity of centrals: {s.overall_purity_centrals:.3f}")

        # Of all the centrals in the truth catalog, how many were found?
        true_centrals = data.loc[~data.IS_SAT_T.astype(bool)]
        true_centrals_assigned_correctly = true_centrals.loc[~true_centrals['IS_SAT']]
        s.overall_completeness_centrals = len(true_centrals_assigned_correctly.index) / len(true_centrals.index)
        print(f"Completeness of centrals: {s.overall_completeness_centrals:.3f}")

        aggregation_column = 'LGAL_BIN_T'
        #aggregation_column_t = 'LGAL_BIN'

        # TODO update below to use these instead
        #sats_in_truthcat_g = sats_in_truthcat.groupby(aggregation_column_t, observed=False).size().to_numpy()
        #centrals_in_truthcat_g = centrals_in_truthcat.groupby(aggregation_column_t, observed=False).size().to_numpy()
        true_sats_g = true_sats.groupby(aggregation_column, observed=False).size().to_numpy()
        true_centrals_g = true_centrals.groupby(aggregation_column, observed=False).size().to_numpy()

        assigned_true_sats = assigned_sats.loc[assigned_sats.IS_SAT_T.astype(bool)]
        assigned_sats_g = assigned_sats.groupby(aggregation_column, observed=False).size().to_numpy()

        assigned_sats_correct_g = assigned_true_sats.groupby(aggregation_column, observed=False).size().to_numpy()
        s.keep=np.nonzero(assigned_sats_g > 9)
        s.purity_g = assigned_sats_correct_g[s.keep] / assigned_sats_g[s.keep]

        true_sats_assigned = true_sats.loc[true_sats['IS_SAT']]
        true_sats_g = true_sats.groupby(aggregation_column, observed=False).size().to_numpy()
        true_sats_correct_g = true_sats_assigned.groupby(aggregation_column, observed=False).size().to_numpy()
        s.keep2=np.nonzero(true_sats_g > 9)
        s.completeness_g = true_sats_correct_g[s.keep2] / true_sats_g[s.keep2]

        assigned_true_centrals = assigned_centrals.loc[~(assigned_centrals.IS_SAT_T.astype(bool))]
        assigned_centrals_g = assigned_centrals.groupby(aggregation_column, observed=False).size().to_numpy()
        assigned_centrals_correct_g = assigned_true_centrals.groupby(aggregation_column, observed=False).size().to_numpy()
        s.keep3=np.nonzero(assigned_centrals_g > 9)
        s.purity_c_g = assigned_centrals_correct_g[s.keep3] / assigned_centrals_g[s.keep3]

        true_centrals_assigned = true_centrals.loc[~(true_centrals['IS_SAT'])]
        true_centrals_g = true_centrals.groupby(aggregation_column, observed=False).size().to_numpy()
        true_centrals_correct_g = true_centrals_assigned.groupby(aggregation_column, observed=False).size().to_numpy()
        s.keep4=np.nonzero(true_centrals_g > 9)
        s.completeness_c_g = true_centrals_correct_g[s.keep4] / true_centrals_g[s.keep4]


def purity_complete_plots(*sets, ymin=0.0):
    plt.rcParams.update({'font.size': 14})

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 8))
    #fig.set_dpi(DPI/2)

    XMIN = 7.5
    XMAX = 11
    YMIN = ymin

    axes[1][0].set_title('Satellite Purity')
    axes[1][0].set_xlabel('log$(L_{\\mathrm{gal}}~/~(\\mathrm{L}_\\odot \\mathrm{h}^{-2} ))$')
    axes[1][0].set_xlim(XMIN,XMAX)
    axes[1][0].set_ylim(YMIN,1.0)
    ax2=axes[1][0].twiny()
    ax2.plot(Mr_gal_bins[sets[0].keep], sets[0].purity_g, ls="") #just to get it to show up
    ax2.set_xlim(log_solar_L_to_abs_mag_r(XMIN), log_solar_L_to_abs_mag_r(XMAX))
    ax2.set_xlabel("$M_r$ - 5log(h)")

    axes[1][1].set_title('Satellite Completeness')
    axes[1][1].set_xlabel('log$(L_{\\mathrm{gal}}~/~(\\mathrm{L}_\\odot \\mathrm{h}^{-2} ))$')
    axes[1][1].set_xlim(XMIN,XMAX)
    axes[1][1].set_ylim(YMIN,1.0)
    ax2=axes[1][1].twiny()
    ax2.plot(Mr_gal_bins[sets[0].keep2], sets[0].completeness_g, ls="") #just to get it to show up
    ax2.set_xlim(log_solar_L_to_abs_mag_r(XMIN), log_solar_L_to_abs_mag_r(XMAX))
    ax2.set_xlabel("$M_r$ - 5log(h)")

    axes[0][0].set_title('Central Purity')
    axes[0][0].set_xlabel('log$(L_{\\mathrm{gal}}~/~(\\mathrm{L}_\\odot \\mathrm{h}^{-2} ))$')
    axes[0][0].set_xlim(XMIN,XMAX)
    axes[0][0].set_ylim(YMIN,1.0)
    ax2=axes[0][0].twiny()
    ax2.plot(Mr_gal_bins[sets[0].keep3], sets[0].purity_c_g, ls="") #just to get it to show up
    ax2.set_xlim(log_solar_L_to_abs_mag_r(XMIN), log_solar_L_to_abs_mag_r(XMAX))
    ax2.set_xlabel("$M_r$ - 5log(h)")

    axes[0][1].set_title('Central Completeness')
    axes[0][1].set_xlabel('log$(L_{\\mathrm{gal}}~/~(\\mathrm{L}_\\odot \\mathrm{h}^{-2} ))$')
    axes[0][1].set_xlim(XMIN,XMAX)
    axes[0][1].set_ylim(YMIN,1.0)
    ax2=axes[0][1].twiny()
    ax2.plot(Mr_gal_bins[sets[0].keep4], sets[0].completeness_c_g, ls="") #just to get it to show up
    ax2.set_xlim(log_solar_L_to_abs_mag_r(XMIN), log_solar_L_to_abs_mag_r(XMAX))
    ax2.set_xlabel("$M_r$ - 5log(h)")
                   
    for s in sets:
        axes[1][0].plot(np.log10(s.L_gal_bins[s.keep]), s.purity_g, s.marker, label=f"{get_dataset_display_name(s)}", color=s.color)
        axes[1][1].plot(np.log10(s.L_gal_bins[s.keep2]), s.completeness_g, s.marker, label=f"{get_dataset_display_name(s)}", color=s.color)
        axes[0][0].plot(np.log10(s.L_gal_bins[s.keep3]), s.purity_c_g, s.marker, label=f"{get_dataset_display_name(s)}", color=s.color)
        axes[0][1].plot(np.log10(s.L_gal_bins[s.keep4]), s.completeness_c_g, s.marker, label=f"{get_dataset_display_name(s)}", color=s.color)

    axes[0][0].legend()

    # Grids on
    for ax in axes.flatten():
        ax.grid(True)

    fig.tight_layout()

    font_restore()

    # Make just the satellite purity plot on its own
    #plt.figure(dpi=DPI)
    #for s in sets:
    #    plt.plot(s.L_gal_bins[s.keep], s.purity_g, s.marker, label=f"{get_dataset_display_name(s)}", color=s.color)

    #plt.xscale('log')
    #plt.xlabel('$L_{\\mathrm{gal}}~[\\mathrm{L}_\\odot \\mathrm{h}^{-2} ]$')
    #plt.ylabel('Satellite Purity')
    #plt.legend()
    #plt.xlim(XMIN,XMAX)
    #plt.ylim(0.0,1.0)
    #plt.tight_layout()




def assigned_halo_analysis(*sets):
    """
    Compares assigned halos to MXXL 'truth' halos.
    
    TODO: Only works on MXXL right now.
    """

    for data in sets:

        print(data.name)

        #same_halo_mass = np.isclose(data.all_data['assigned_halo_mass'], data.all_data['mxxl_halo_mass'], atol=0.0, rtol=1e-03)
        #same_mxxl_halo = data.all_data['assigned_halo_mass']
        #data.all_data['same_mxxl_halo'] = same_mxxl_halo

        lost_galaxies = data.all_data[data.all_data['Z_ASSIGNED_FLAG'] != 0]
        print(len(lost_galaxies), "lost galaxies")

        lost_galaxies = lost_galaxies[lost_galaxies['assigned_halo_id'] != 0]
        print(len(lost_galaxies), "lost galaxies after removing ones with no MXXL halo ID (these correspond to halos that were too small for the simulation and were added by hand)")

        lost_galaxies_same_halo = np.equal(lost_galaxies['assigned_halo_id'], lost_galaxies['mxxl_halo_id'])
        print("Fraction of time assigned halo ID is the same as the galaxy's actual halo ID: {0:.3f}".format(np.sum(lost_galaxies_same_halo) / len(lost_galaxies_same_halo)))
        
        lost_galaxies_same_halo_mass = np.isclose(lost_galaxies['assigned_halo_mass'], lost_galaxies['mxxl_halo_mass'], atol=0.0, rtol=1e-03)
        print("Fraction of time assigned halo mass is \'the same\' as the galaxy's actual halo mass: {0:.3f}".format(np.sum(lost_galaxies_same_halo_mass) / len(lost_galaxies_same_halo_mass)))
      
        z_thresh=0.01
        lost_galaxies_similar_z = np.isclose(lost_galaxies['Z'], lost_galaxies['Z_OBS'], atol=z_thresh, rtol=0.0)         
        print("Fraction of time assigned z is the target z +/- {0:.3f}:".format(z_thresh), np.sum(lost_galaxies_similar_z) / len(lost_galaxies_similar_z))

        z_thresh=0.005
        lost_galaxies_similar_z = np.isclose(lost_galaxies['Z'], lost_galaxies['Z_OBS'], atol=z_thresh, rtol=0.0)         
        print("Fraction of time assigned z is the target z +/- {0:.3f}:".format(z_thresh), np.sum(lost_galaxies_similar_z) / len(lost_galaxies_similar_z))

        z_thresh=0.003
        lost_galaxies_similar_z = np.isclose(lost_galaxies['Z'], lost_galaxies['Z_OBS'], atol=z_thresh, rtol=0.0)         
        print("Fraction of time assigned z is the target z +/- {0:.3f}:".format(z_thresh), np.sum(lost_galaxies_similar_z) / len(lost_galaxies_similar_z))

        z_thresh=0.001
        lost_galaxies_similar_z = np.isclose(lost_galaxies['Z'], lost_galaxies['Z_OBS'], atol=z_thresh, rtol=0.0)        
        print("Fraction of time assigned z is the target z +/- {0:.3f}:".format(z_thresh), np.sum(lost_galaxies_similar_z) / len(lost_galaxies_similar_z))
        
        # TODO as a function of reshift. But we essentially already have this from the direct MXXL data plots

        #z_bins = np.linspace(min(data.all_data['Z']), max(data.all_data['Z']), 20)
        #z_labels = z_bins[0:len(z_bins)-1] 
        #data.all_data['z_bin'] = pd.cut(x = data.all_data['Z'], bins = z_bins, labels = z_labels, include_lowest = True)

        #groupby_z = lost_galaxies.groupby('z_bin')['same_halo_mass'].sum() / lost_galaxies.groupby('z_bin')['same_halo_mass'].count()

        #plt.plot(z_labels, groupby_z)
        #plt.xlabel('$z_{eff}$ (effective/assigned redshift)')
        #plt.ylabel('Fraction Assigned Halo = True Host Halo')
        





###############################################
# Routines for making RA DEC maps
###############################################

# 36*18*640 / 41253 = 10.05 dots per sq degree
# so a dot size of 1 is 0.1 sq deg

# 0.00191 sq deg is area of a fiber's region
# that is the size we want to draw for each galaxy
#so s=0.01 is what we want

#def molleweide_map_for_catalog(catalog: GroupCatalog, alpha=0.1, dpi=150, fig=None, dotsize=0.01):
#    return make_map(catalog.all_data.RA.to_numpy(), catalog.all_data['DEC'].to_numpy(), alpha, dpi, fig, dotsize)

def make_map(ra, dec, alpha=0.1, dpi=150, fig=None, dotsize=0.01):
    """
    Give numpy array of ra and dec.
    """
    # If ra or dec is a pd.Series, convert to numpy array
    if isinstance(ra, pd.Series):
        ra = ra.to_numpy()
    if isinstance(dec, pd.Series):
        dec = dec.to_numpy()

    #if np.any(ra > 180.0): # if data given is 0 to 360
    assert np.all(ra > -0.1)
    ra = ra - 180
    if np.any(dec > 90.0): # if data is 0 to 180
        print(f"WARNING: Dec values are 0 to 180. Subtracting 90 to get -90 to 90.")
        assert np.all(dec > -0.1)
        dec = dec - 90

    # Build a map of the galaxies
    ra_angles = coord.Angle(ra*u.degree)
    ra_angles = ra_angles.wrap_at(180*u.degree)
    #ra_angles = ra_angles.wrap_at(360*u.degree)
    dec_angles = coord.Angle(dec*u.degree)

    if fig == None:
        fig = plt.figure(figsize=(12,12))
        fig.dpi=dpi
        ax = fig.add_subplot(111, projection="mollweide")
        plt.grid(visible=True, which='both')
        ax.set_xticklabels(['30°', '60°', '90°', '120°', '150°', '180°', '210°', '240°', '270°', '300°', '330°'])
    else:
        ax=fig.get_axes()[0]

    ax.scatter(ra_angles.radian, dec_angles.radian, alpha=alpha, s=dotsize)
    return fig



def plot_positions(*dataframes, tiles_df: pd.DataFrame = None, DEG_LONG=1, split=True, ra_min=30, dec_min = -5):

    ra_max = ra_min + DEG_LONG
    dec_max = dec_min + DEG_LONG

    fig,ax = plt.subplots(1)
    fig.set_size_inches(10*DEG_LONG + 1,10*DEG_LONG + 1) # the extra inch is because of the frame, rough correction
    dpi = 100
    fig.set_dpi(dpi)
    dots_per_sqdeg = 10 * 10 * dpi # so 10,000 dots in a square degree
    ax.set_aspect('equal')
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_xlim(ra_min, ra_max)
    ax.set_ylim(dec_min, dec_max)

    if tiles_df is not None:
        TILE_RADIUS = 5862.0 * u.arcsec # arcsec
        tile_radius = TILE_RADIUS.to(u.degree).value
        circle_ra_max = ra_max + 2*tile_radius
        circle_ra_min = ra_min - 2*tile_radius
        circle_dec_max = dec_max + 2*tile_radius
        circle_dec_min = dec_min - 2*tile_radius
        tiles_to_draw = tiles_df.query('RA < @circle_ra_max and RA > @circle_ra_min and DEC < @circle_dec_max and DEC > @circle_dec_min')
        
        for index, row in tiles_df.iterrows():
            circ = Circle((row.RA, row['DEC']), tile_radius, color='k', fill=False, lw=10)
            ax.add_patch(circ)

    alpha = 1 if len(dataframes) == 1 else 0.5
    for d in dataframes:
        # if d's type is GroupCatalog, extract the all_data property
        if hasattr(d, 'all_data'):
            d = d.all_data 
        plot_ra_dec_inner(d, ax, dots_per_sqdeg, ra_min, ra_max, dec_min, dec_max, split, alpha)


def plot_ra_dec_inner(dataframe: pd.DataFrame, ax, dots_per_sqdeg, ra_min, ra_max, dec_min, dec_max, split, alpha):

    if split:
        obs = dataframe[dataframe['Z_ASSIGNED_FLAG'] == 0]
        unobs = dataframe[dataframe['Z_ASSIGNED_FLAG'] != 0]
    else:
        obs = dataframe

    # 8 sq deg / 5000 fibers is 0.0016 sq deg per fiber
    # But in reality the paper says 0.0019 sq deg is area of a fiber's region (there is some overlap)
    # That is the size we want to draw for each galaxy here.
    # For 10,000 dots per sq degree, 0.0019 * 10000 = 
    fiber_patrol_area = 0.00191 # in sq deg
    ARBITRARY_SCALE_UP = 1 # my calculation didn't work so arbitrarilly sizing up with this
    size = fiber_patrol_area * dots_per_sqdeg * ARBITRARY_SCALE_UP 

    obs_selected = obs.query('RA < @ra_max and RA > @ra_min and DEC < @dec_max and DEC > @dec_min')
    ax.scatter(obs_selected.RA, obs_selected['DEC'], s=size, alpha=alpha)
    if split:
        unobs_selected = unobs.query('RA < @ra_max and RA > @ra_min and DEC < @dec_max and DEC > @dec_min')
        ax.scatter(unobs_selected.RA, unobs_selected['DEC'], marker='x', s=size, alpha=alpha)



def _getsize(z):
    if z < 0.05:
        return 300
    elif z < 0.1:
        return 200
    elif z < 0.2:
        return 120
    elif z < 0.2:
        return 75
    elif z < 0.3:
        return 45
    elif z < 0.4:
        return 25
    elif z < 0.5:
        return 15
    elif z < 0.6:
        return 8
    else:
        return 3

def examine_area(ra_min, ra_max, dec_min, dec_max, data: pd.DataFrame):
    length_arcmin = (ra_max - ra_min) * 60
    galaxies = data.query('RA < @ra_max and RA > @ra_min and DEC < @dec_max and DEC > @dec_min')
    centrals = galaxies.query('IS_SAT == False')
    sats = galaxies.query('IS_SAT == True')
    textsize = 9

    fig,ax = plt.subplots(1)
    fig.set_size_inches(10,10)
    ax.set_aspect('equal')

    plt.scatter(centrals.RA, centrals['DEC'], s=list(map(lambda x: _getsize(x)/(length_arcmin/10), centrals['Z'])), color='k')
    plt.scatter(sats.RA, sats['DEC'], s=list(map(lambda x: _getsize(x)/(length_arcmin/10), sats['Z'])), color='b')
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.title("Galaxies in Area")
    plt.xlim(ra_min, ra_max)
    plt.ylim(dec_min, dec_max)

    # Add patches for virial radius of halos of centrals
    for k in range(len(centrals)):
        current = centrals.iloc[k]
        radius = current.halo_radius_arcsec / 3600 # arcsec to degrees, like the plot
        circ = Circle((current.RA,current['DEC']), radius, color=get_color(0), alpha=0.10)
        ax.add_patch(circ)

    if len(galaxies) < 50:
        for k in range(len(galaxies)):
            plt.text(galaxies.iloc[k].RA, galaxies.iloc[k]['DEC'], "{0:.3f}".format(galaxies.iloc[k]['Z']), size=textsize)

    return galaxies


textsize = 9
def write_z(galaxy):
    if hasattr(galaxy, 'Z_T'):
        if close_enough(galaxy['Z'], galaxy['Z_T']):
            plt.text(galaxy.RA, galaxy['DEC'], "{0:.3f}".format(galaxy['Z']), size=textsize, color='green')
        else:
            plt.text(galaxy.RA, galaxy['DEC'], "{0:.3f}".format(galaxy['Z']), size=textsize, color='red')
            plt.text(galaxy.RA, galaxy['DEC']-0.0035, "{0:.3f}".format(galaxy['Z_T']), size=textsize, color='blue')
    else:
        plt.text(galaxy.RA, galaxy['DEC'], "{0:.3f}".format(galaxy['Z']), size=textsize, color='k')



def examine_groups_near(target, data: pd.DataFrame, nearby_angle: coord.Angle = coord.Angle('7m'), zfilt=None):

    buffer_angle = coord.Angle('1m')
    ra_map_max = (coord.Angle(target.RA*u.degree) + nearby_angle).value
    ra_map_min = (coord.Angle(target.RA*u.degree) - nearby_angle).value
    dec_map_max = (coord.Angle(target['DEC']*u.degree) + nearby_angle).value
    dec_map_min = (coord.Angle(target['DEC']*u.degree) - nearby_angle).value

    ra_max = (coord.Angle(target.RA*u.degree) + nearby_angle + buffer_angle).value
    ra_min = (coord.Angle(target.RA*u.degree) - nearby_angle - buffer_angle).value
    dec_max = (coord.Angle(target['DEC']*u.degree) + nearby_angle + buffer_angle).value
    dec_min = (coord.Angle(target['DEC']*u.degree) - nearby_angle - buffer_angle).value

    if zfilt:
        z_min = target['Z'] - zfilt
        z_max = target['Z'] + zfilt
        nearby = data.query('RA < @ra_max and RA > @ra_min and DEC < @dec_max and DEC > @dec_min and Z > @z_min and Z < @z_max')

        # Get rid of satellites whose central aren't in nearby
        centrals = nearby.loc[~nearby['IS_SAT']]    
        sats = nearby.loc[nearby['IS_SAT']]
        prev_nsats = len(sats)
        sats = sats.loc[sats['IGRP'].isin(centrals['IGRP'])]
        #print(f"Removed {prev_nsats - len(sats)} satellites whose central wasn't in the nearby group")
        nearby = pd.concat([centrals, sats])
    else:
        nearby = data.query('RA < @ra_max and RA > @ra_min and DEC < @dec_max and DEC > @dec_min')

    # Generate a unique color for each group
    unique_groups = nearby['IGRP'].unique()
    group_colors = {group: plt.cm.tab20(i / len(unique_groups)) for i, group in enumerate(unique_groups)}

    if len(nearby) <= 0:
        print(f"Skipping empty plot")
        return
    
    sats = nearby.loc[nearby['IS_SAT']]
    centrals = nearby.loc[~nearby['IS_SAT']]

    fig, ax = plt.subplots(1)
    fig.set_size_inches(10, 10)
    ax.set_aspect('equal')

    # Draw Halos
    for k in range(len(centrals)):
        current = centrals.iloc[k]
        radius = current.halo_radius_arcsec / 3600  # arcsec to degrees, like the plot
        circ = Circle((current['RA'], current['DEC']), radius, color=group_colors[current['IGRP']], alpha=0.16)
        ax.add_patch(circ)

    
    cenobs = centrals.loc[z_flag_is_spectro_z(centrals['Z_ASSIGNED_FLAG'])]
    cenunobs = centrals.loc[z_flag_is_not_spectro_z(centrals['Z_ASSIGNED_FLAG'])]
    satobs = sats.loc[z_flag_is_spectro_z(sats['Z_ASSIGNED_FLAG'])]
    satunobs = sats.loc[z_flag_is_not_spectro_z(sats['Z_ASSIGNED_FLAG'])]

    # Assign a color for each group and use it for both edge and face colors
    cenobs_colors = [group_colors[grp] for grp in cenobs['IGRP']]
    cenunobs_colors = [group_colors[grp] for grp in cenunobs['IGRP']]
    satobs_colors = [group_colors[grp] for grp in satobs['IGRP']]
    satunobs_colors = [group_colors[grp] for grp in satunobs['IGRP']]

    # Observed centrals: filled circles
    plt.scatter(
        cenobs['RA'], cenobs['DEC'],
        s=list(map(_getsize, cenobs['Z'])),
        color=cenobs_colors,
        edgecolor=cenobs_colors,
    )
    # Unobserved centrals: open circles
    plt.scatter(
        cenunobs['RA'], cenunobs['DEC'],
        s=list(map(_getsize, cenunobs['Z'])),
        facecolors='none',
        edgecolors=cenunobs_colors,
    )
    # Observed satellites: filled squares
    plt.scatter(
        satobs['RA'], satobs['DEC'],
        s=list(map(_getsize, satobs['Z'])),
        color=satobs_colors,
        edgecolor=satobs_colors,
        marker='s'
    )
    # Unobserved satellites: open squares
    plt.scatter(
        satunobs['RA'], satunobs['DEC'],
        s=list(map(_getsize, satunobs['Z'])),
        facecolors='none',
        edgecolors=satunobs_colors,
        marker='s'
    )

    # Add redshift labels and M_HALO for centrals
    for k in range(len(nearby)):
        plt.text(nearby.iloc[k].RA, nearby.iloc[k]['DEC'], "{0:.3f}".format(nearby.iloc[k]['Z']), size=textsize)
        #plt.text(nearby.iloc[k].RA, nearby.iloc[k]['DEC'] - 0.004, "$B_s$={0:.3f}".format(nearby.iloc[k]['BSAT']), size=textsize)
        if not nearby.iloc[k]['IS_SAT']:
            plt.text(nearby.iloc[k].RA, nearby.iloc[k]['DEC'] - 0.0080, "$M_h$={0:.1f}".format(np.log10(nearby.iloc[k]['M_HALO'])), size=textsize)
            #plt.text(nearby.iloc[k].RA, nearby.iloc[k]['DEC'], "{0:.3f}".format(nearby.iloc[k]['Z']), size=textsize)

    plt.xlim(ra_map_min, ra_map_max)
    plt.ylim(dec_map_min, dec_map_max)
    plt.xlabel('RA')
    plt.ylabel('DEC')
    ax.set_xticks(np.linspace(ra_min, ra_max, num=7))
    ax.set_xticklabels([f"{tick:.3f}°" for tick in ax.get_xticks()])
    ax.set_yticks(np.linspace(dec_min, dec_max, num=7))
    ax.set_yticklabels([f"{tick:.3f}°" for tick in ax.get_yticks()])
    ax.invert_xaxis()
    #plt.legend()
    plt.draw()


def examine_around(target, data: pd.DataFrame, nearby_angle: coord.Angle = coord.Angle('7m'), zfilt=False):

    z_eff = target['Z']
    #target_dist_true = z_to_ldist(target.z_obs)
    buffer_angle = coord.Angle('1m')
    ra_map_max = (coord.Angle(target.RA*u.degree) + nearby_angle).value
    ra_map_min = (coord.Angle(target.RA*u.degree) - nearby_angle).value
    dec_map_max = (coord.Angle(target['DEC']*u.degree) + nearby_angle).value
    dec_map_min = (coord.Angle(target['DEC']*u.degree) - nearby_angle).value

    ra_max = (coord.Angle(target.RA*u.degree) + nearby_angle + buffer_angle).value
    ra_min = (coord.Angle(target.RA*u.degree) - nearby_angle - buffer_angle).value
    dec_max = (coord.Angle(target['DEC']*u.degree) + nearby_angle + buffer_angle).value
    dec_min = (coord.Angle(target['DEC']*u.degree) - nearby_angle - buffer_angle).value

    if zfilt:
        z_min = target['Z_PHOT'] - 0.05
        z_max = target['Z_PHOT'] + 0.05
        nearby = data.query('RA < @ra_max and RA > @ra_min and DEC < @dec_max and DEC > @dec_min and z > @z_min and z < @z_max')
    else:
        nearby = data.query('RA < @ra_max and RA > @ra_min and DEC < @dec_max and DEC > @dec_min')

    if target.name in nearby.index:
        nearby = nearby.drop(target.name) # drop the target itself from this df

    target_observed = 'Z_ASSIGNED_FLAG' not in nearby.columns or target['Z_ASSIGNED_FLAG'] == 0

    # check if nearby has column z_assigned_flag
    if 'Z_ASSIGNED_FLAG' in nearby.columns:
        nearby_obs = nearby.loc[z_flag_is_spectro_z(nearby['Z_ASSIGNED_FLAG'])]
        nearby_unobs = nearby.loc[z_flag_is_not_spectro_z(nearby['Z_ASSIGNED_FLAG'])]
    else:
        nearby_obs = nearby
        nearby_unobs = False

    z_match = []
    # This doesn' work because the NN catalog has FAINT observed targets that aren't in the group catalog.
    #if z_flag_is_neighbor(target['Z_ASSIGNED_FLAG']):
    #    idx, d2d, _ = coord.match_coordinates_sky(
    #        coord.SkyCoord(ra=target.RA*u.degree, dec=target['DEC']*u.degree, frame='icrs'),
    #        coord.SkyCoord(ra=nearby_obs.RA.to_numpy()*u.degree, dec=nearby_obs['DEC'].to_numpy()*u.degree, frame='icrs'),
    #        nthneighbor=target['Z_ASSIGNED_FLAG'],
    #        storekdtree=False
    #    )
    #    z_match = nearby_obs.iloc[idx]
    #nearby_obs = nearby_obs.drop(z_match.name)

    good_obs_z_filter = list(map(lambda a: close_enough(target['Z'], a), nearby_obs['Z']))
    nearby_obs_good_z = nearby_obs.loc[good_obs_z_filter]
    nearby_obs_good_z_dim = nearby_obs_good_z.loc[nearby_obs_good_z['APP_MAG_R'] > 19.5]
    nearby_obs_good_z = nearby_obs_good_z.loc[np.invert(nearby_obs_good_z['APP_MAG_R'] > 19.5)]

    if len(good_obs_z_filter) > 0:
        nearby_obs_other = nearby_obs.loc[np.invert(good_obs_z_filter)]
    else:
        nearby_obs_other = nearby_obs
    nearby_obs_other_dim = nearby_obs_other.loc[nearby_obs_other['APP_MAG_R'] > 19.5]
    nearby_obs_other = nearby_obs_other.loc[np.invert(nearby_obs_other['APP_MAG_R'] > 19.5)]

    if nearby_unobs is not False:
        good_unobs_z_filter = list(map(lambda a: close_enough(target['Z'], a), nearby_unobs['Z']))

        nearby_unobs_good_z = nearby_unobs.loc[good_unobs_z_filter]
        if good_unobs_z_filter:
            nearby_unobs_other = nearby_unobs.loc[np.invert(good_unobs_z_filter)]
            nearby_unobs_other_dim = nearby_unobs_other.loc[nearby_unobs_other['APP_MAG_R'] > 19.5]
            nearby_unobs_other = nearby_unobs_other.loc[np.invert(nearby_unobs_other['APP_MAG_R'] > 19.5)]
        else:
            nearby_unobs_other = nearby_unobs_good_z # empty df
            nearby_unobs_other_dim = nearby_unobs_good_z

        nearby_unobs_good_z_dim = nearby_unobs_good_z.loc[nearby_unobs_good_z['APP_MAG_R'] > 19.5]
        nearby_unobs_good_z = nearby_unobs_good_z.loc[np.invert(nearby_unobs_good_z['APP_MAG_R'] > 19.5)]

    if target_observed:
        title = f"Observed Galaxy {target.name}: z={target['Z']:.3f}"
    else:
        if hasattr(target, 'Z_T'):
            title = f"Lost Galaxy {target.name}: A={target['Z_ASSIGNED_FLAG']} z={target['Z']:.4f} z_phot={target['Z_PHOT']:.4f} Truth={target['Z_T']:.4f}"
        else:
            title = f"Lost Galaxy {target.name}: A={target['Z_ASSIGNED_FLAG']} z={target['Z']:.4f} z_phot={target['Z_PHOT']:.4f}"

    if len(nearby) > 1:

        fig,ax = plt.subplots(1)
        fig.set_size_inches(10,10)
        ax.set_aspect('equal')

        # Add virial radii or MXXL Halos to the galaxies
        for k in range(len(nearby)):
            current = nearby.iloc[k]
            if current['IS_SAT']:
                continue
            radius = current.halo_radius_arcsec / 3600 # arcsec to degrees, like the plot
            #radius = current.mxxl_halo_vir_radius_guess_arcsec / 3600 # arcsec to degrees, like the plot
            if target['IGRP'] == current['IGRP']:
                circ = Circle((current.RA,current['DEC']), radius, color=get_color(1), alpha=0.20)
            else:
                circ = Circle((current.RA,current['DEC']), radius, color=get_color(0), alpha=0.10)
            ax.add_patch(circ)
        if not target['IS_SAT']:
            radius = target.halo_radius_arcsec / 3600 # arcsec to degrees, like the plot
            circ = Circle((target.RA,target['DEC']), radius, color=get_color(1), alpha=0.20)
            ax.add_patch(circ)

        dimalpha = 0.4

        plt.scatter(nearby_obs_other.RA, nearby_obs_other['DEC'], s=list(map(_getsize, nearby_obs_other['Z'])), color=get_color(0), label="Obs ({0})".format(len(nearby_obs_other)))
        if len(nearby_obs_other_dim) > 0:
            plt.scatter(nearby_obs_other_dim.RA, nearby_obs_other_dim['DEC'], s=list(map(_getsize, nearby_obs_other_dim['Z'])), color=get_color(2), alpha=dimalpha, label="Obs dim ({0})".format(len(nearby_obs_other_dim)))
        
        plt.scatter(nearby_obs_good_z.RA, nearby_obs_good_z['DEC'], s=list(map(_getsize, nearby_obs_good_z['Z'])), color=get_color(2), label="Obs good z ({0})".format(len(nearby_obs_good_z)))
        if len(nearby_obs_good_z_dim) > 0:
            plt.scatter(nearby_obs_good_z_dim.RA, nearby_obs_good_z_dim['DEC'], s=list(map(_getsize, nearby_obs_good_z_dim['Z'])), color=get_color(0), alpha=dimalpha, label="Obs good z dim ({0})".format(len(nearby_obs_good_z_dim)))

        if nearby_unobs is not False:
            plt.scatter(nearby_unobs_other.RA, nearby_unobs_other['DEC'], marker='x', s=list(map(_getsize, nearby_unobs_other['Z'])), color=get_color(0), label="Unobs ({0})".format(len(nearby_unobs_other)))
            if len(nearby_unobs_other_dim) > 0:
                plt.scatter(nearby_unobs_other_dim.RA, nearby_unobs_other_dim['DEC'], marker='x', s=list(map(_getsize, nearby_unobs_other_dim['Z'])), color=get_color(0), alpha=dimalpha, label="Unobs dim ({0})".format(len(nearby_unobs_other_dim)))
            
            plt.scatter(nearby_unobs_good_z.RA, nearby_unobs_good_z['DEC'], marker='x', s=list(map(_getsize, nearby_unobs_good_z['Z'])), color=get_color(2), label="Unobs good z ({0})".format(len(nearby_unobs_good_z)))
            if len(nearby_unobs_good_z_dim) > 0:
                plt.scatter(nearby_unobs_good_z_dim.RA, nearby_unobs_good_z_dim['DEC'], marker='x', s=list(map(_getsize, nearby_unobs_good_z_dim['Z'])), color=get_color(2), alpha=dimalpha, label="Unobs good z dim ({0})".format(len(nearby_unobs_good_z_dim)))
            
        # redshift data labels
        for k in range(len(nearby_obs)):
            plt.text(nearby_obs.iloc[k].RA, nearby_obs.iloc[k]['DEC'], "{0:.3f}".format(nearby_obs.iloc[k]['Z']), size=textsize)
        if nearby_unobs is not False:
            for k in range(len(nearby_unobs)):
                # Choose color for each based on if z_truth and z are close
                write_z(nearby_unobs.iloc[k])

        # Circle assigned one
        if len(z_match) > 0:
            plt.scatter(z_match.RA, z_match['DEC'], color=get_color(3), facecolors='none', s=_getsize(z_match['Z'])*2, label="Assigned")
            plt.text(z_match.RA, z_match['DEC'], "{0:.3f}".format(z_match['Z']), size=textsize)

        # Target galaxy
        if target_observed:
            plt.scatter(target.RA, target['DEC'], s=_getsize(target['Z']), color=get_color(1), label="Target")
            plt.text(target.RA, target['DEC'], "{0:.3f}".format(target['Z']), size=textsize)
        else:
            plt.scatter(target.RA, target['DEC'], s=_getsize(target['Z']), marker='X', color=get_color(1), label="Target")  
            write_z(target)

        plt.xlim(ra_map_min, ra_map_max)
        plt.ylim(dec_map_min, dec_map_max)
        plt.xlabel('RA')
        plt.ylabel('DEC')
        
        # Custom tick formatter for RA and Dec
        def deg_to_dms(deg):
            d = int(deg)
            m = int((deg - d) * 60)
            return f"{d}°{m}'"

        ax = plt.gca()
        ax.set_xticks(np.linspace(ra_min, ra_max, num=8))
        ax.set_xticklabels([deg_to_dms(tick) for tick in ax.get_xticks()])
        ax.set_yticks(np.linspace(dec_min, dec_max, num=8))
        ax.set_yticklabels([deg_to_dms(tick) for tick in ax.get_yticks()])
        plt.legend()
        plt.title(title)
        plt.draw()
    
    else:
        print("Skipping empty plot for {0}".format(title))



def plot_parameters(params):
    # Weights for each galaxy luminosity, when abundance matching
    # log w_cen,r = (ω_0,r / 2) (1 + erf[(log L_gal - ω_L,r) / σ_ω,r)] ) 
    # log w_cen,b = (ω_0,b / 2) (1 + erf[(log L_gal - ω_L,b) / σ_ω,b)] ) 
    # Bsat,r = β_0,r + β_L,r(log L_gal − 9.5)
    # Bsat,b = β_0,b + β_L,b(log L_gal − 9.5)

    def bsat(p0, p1, L):
        return np.maximum(p0 + p1 * (L - 9.5), 0.001)
    
    def cweight(w0, wl, s, L):
        return L * (w0 / 2) * (1 + special.erf((np.log10(L) - wl) / s))

    fig, axes = plt.subplots(1,2)
    fig.set_size_inches(8, 4)
    x = np.logspace(6, 12, 100)

    axes[0].set_title("Central Weights")
    axes[0].set_xlabel("log$(L_{\\mathrm{gal}}) [L_{\\odot} h^{-2}]$")
    axes[0].set_ylabel("Weight")
    axes[0].set_yscale('log')
    axes[0].set_xscale('log')

    axes[0].plot(x, x, color='k', linestyle='--')
    axes[0].plot(x, cweight(params[0], params[1], params[4], x), label='SF', color='b')
    axes[0].plot(x, cweight(params[2], params[3], params[5], x), label='Q', color='r')

    x = np.linspace(6, 12, 100)

    axes[1].set_title("Bsat")
    axes[1].set_xlabel("log$(L_{\\mathrm{gal}}) [L_{\\odot} h^{-2}]$")
    axes[1].set_ylabel("$B_{\\mathrm{sat}}$")
    #axes[1].set_yscale('log')
    axes[1].plot(x, bsat(params[6], params[7], x), label='Q', color='r')
    axes[1].plot(x, bsat(params[8], params[9], x), label='SF', color='b')

    axes[1].legend()

    plt.tight_layout()
    plt.show()


def gfparams_plots(gc: GroupCatalog, chains_flat):
    # ω_L_sf, σ_sf, ω_L_q, σ_q, ω_0_sf, ω_0_q, β_0q, β_Lq, β_0sf, β_Lsf
    def bsat(p0, p1, L):
        return np.maximum(p0 + p1 * (L - 9.5), 0.5)
    def cweight(w0, wl, s, L):
        return - (w0 / 2) * (1 + special.erf((L - wl) / s))
    def w_plot(q_w, sf_w):
        return np.log10(q_w / sf_w)

    params = np.array([gc.GF_props['omegaL_sf'], gc.GF_props['sigma_sf'], gc.GF_props['omegaL_q'], gc.GF_props['sigma_q'], gc.GF_props['omega0_sf'], gc.GF_props['omega0_q'], gc.GF_props['beta0q'], gc.GF_props['betaLq'], gc.GF_props['beta0sf'], gc.GF_props['betaLsf']])
    # Cannot use the 68 / 95 intervals because we're plotting a function of the parameters, not the parameters directly
    # And so the function of the median parameters may not fall inside the function of the 68 / 95 intervals

    LMIN = 6.5
    fig, axes = plt.subplots(2,1, figsize=(5,9))
    x = np.linspace(6, 11.5, 100)
    axes[0].set_xlabel("log$(L_{\\mathrm{cen}} / (L_{\\odot} h^{-2}) )$")
    axes[0].set_ylabel("log$(w_{\\rm cen}^q / w_{\\rm cen}^{sf})$")
    axes[0].set_ylim(-1,1)
    axes[0].set_xlim(LMIN, 11.1)
    axes[0].set_xticks(np.arange(7,12,1))
    ax0 = axes[0].twiny() # Mag axis on top
    ax0.set_xlim(log_solar_L_to_abs_mag_r(7), log_solar_L_to_abs_mag_r(LOG_LGAL_MAX_TIGHT))
    ax0.set_xticks(np.arange(-13, -25, -2))
    ax0.set_xlabel("$M_r$ - 5log(h)")

    axes[1].set_xlabel("log$(L_{\\mathrm{cen}}~/~(L_{\\odot} h^{-2}) )$")
    axes[1].set_ylabel("$B_{\\mathrm{sat}}$")
    axes[1].set_ylim(-1,40)
    axes[1].set_xlim(LMIN, 11.1)
    axes[1].set_xticks(np.arange(7,12,1))
    ax1 = axes[1].twiny() # Mag axis on top
    ax1.set_xlim(log_solar_L_to_abs_mag_r(7), log_solar_L_to_abs_mag_r(LOG_LGAL_MAX_TIGHT))
    ax1.set_xticks(np.arange(-13, -25, -2))
    ax1.set_xlabel("$M_r$ - 5log(h)")

    for i in range(1000):
        sample_idx = np.random.randint(0, len(chains_flat))
        sample_params = chains_flat[sample_idx]
        axes[0].plot(
            x,
            w_plot(
                cweight(sample_params[4], sample_params[0], sample_params[1], x),
                cweight(sample_params[5], sample_params[2], sample_params[3], x)
            ),
            color='purple',
            alpha=0.03
        )
        axes[1].plot(
            x,
            bsat(sample_params[6], sample_params[7], x),
            color='red',
            alpha=0.03
        )
        axes[1].plot(
            x,
            bsat(sample_params[8], sample_params[9], x),
            color='blue',
            alpha=0.03
        )
 
    # Horizontal line at y=0
    axes[0].axhline(0, color='k', linestyle='--', lw=1, alpha=0.5)

    axes[0].plot(x, w_plot(cweight(params[4], params[0], params[1], x), cweight(params[5], params[2], params[3], x)), color='k', lw=2)
    axes[1].plot(x, bsat(params[6], params[7], x), '-', label='Q', color='k', lw=2)
    axes[1].plot(x, bsat(params[8], params[9], x), '-',label='SF', color='k', lw=2)

    plt.tight_layout()
