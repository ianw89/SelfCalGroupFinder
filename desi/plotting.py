import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as u
from pyutils import *
import math
from groupcatalog import *

# np.array(zip(*[line.split() for line in f])[1], dtype=float)


DPI = 400
FONT_SIZE_DEFAULT = 12

LGAL_XMINS = [6E7, 3E8]

plt.style.use('default')
plt.rcParams.update({'font.size': FONT_SIZE_DEFAULT})

def font_restore():
    plt.rcParams.update({'font.size': FONT_SIZE_DEFAULT})





##########################
# Plots
##########################

def legend(datasets):
    if len(datasets) > 1:
        plt.legend()

def legend_ax(ax, datasets):
    #if len(datasets) > 1:
    ax.legend()

def do_hod_plot(df, centrals, sats, mass_bin_prop, mass_labels, color, name, SHOW_UNWEIGHTED=False):
    HOD_LGAL_CUT = 4E9

    vmax_cen_avg = np.average(centrals[centrals.L_gal > HOD_LGAL_CUT].V_max)
    vmax_sat_avg = np.average(sats[sats.L_gal > HOD_LGAL_CUT].V_max)
    vmax_gal_avg = np.average(df[df.L_gal > HOD_LGAL_CUT].V_max)
    halo_bin_sizes_weighted = centrals.groupby(mass_bin_prop).apply(count_vmax_weighted) * np.average(centrals.V_max)

    N_cen = centrals[centrals.L_gal > HOD_LGAL_CUT].groupby(mass_bin_prop).apply(count_vmax_weighted) * vmax_cen_avg / halo_bin_sizes_weighted
    N_sat = sats[sats.L_gal > HOD_LGAL_CUT].groupby(mass_bin_prop).apply(count_vmax_weighted) * vmax_sat_avg / halo_bin_sizes_weighted
    N_gal = df[df.L_gal > HOD_LGAL_CUT].groupby(mass_bin_prop).apply(count_vmax_weighted) * vmax_gal_avg / halo_bin_sizes_weighted
    

    plt.figure(dpi=DPI)
    plt.plot(mass_labels, N_cen, ".", label=f"Centrals", color=color)
    plt.plot(mass_labels, N_sat, "--", label=f"Satellites", color=color)
    plt.plot(mass_labels, N_gal, "-", label=f"All", color=color)

    if SHOW_UNWEIGHTED:
        halo_bin_sizes_weighted2 = centrals.groupby(mass_bin_prop).size()
        N_cen2 = centrals[centrals.L_gal > HOD_LGAL_CUT].groupby(mass_bin_prop).size()  / halo_bin_sizes_weighted2
        N_sat2 = sats[sats.L_gal > HOD_LGAL_CUT].groupby(mass_bin_prop).size()  / halo_bin_sizes_weighted2
        N_gal2 = df[df.L_gal > HOD_LGAL_CUT].groupby(mass_bin_prop).size()  / halo_bin_sizes_weighted2

        plt.plot(mass_labels, N_cen2, ".", label=f"Centrals Unweighted", color='red', alpha=0.5)
        plt.plot(mass_labels, N_sat2, "--", label=f"Satellites Unweighted", color='red', alpha=0.5)
        plt.plot(mass_labels, N_gal2, "-", label=f"All Unweighted", color='red', alpha=0.5)

    plt.loglog()    
    plt.ylabel("$<N_{gal}>$")    
    plt.xlabel('$M_{halo}$')
    plt.title(f"Halo Occupancy for \'{name}\' (L>$10^{{{np.log10(HOD_LGAL_CUT):.1f}}}$)")
    plt.legend()
    plt.xlim(1E11,1E15)
    plt.draw()


def get_dataset_display_name(d, keep_mag_limit=False):
    name = d.name.replace("Fiber Only", "Observed").replace("Simple v4", "Our Algorithm").replace(" Vanilla", "").replace("Nearest Neighbor", "NN")
    if keep_mag_limit:
        return name
    else:
        return name.replace(" <19.5", "").replace(" <20.0", "")

def hod_plots(*datasets, truth_on=True):

    for f in datasets:
        do_hod_plot(f.all_data, f.centrals, f.sats, 'Mh_bin', f.labels, f.color, get_dataset_display_name(f))

        if truth_on and f.has_truth:
            do_hod_plot(f.all_data, f.centrals_T, f.sats_T, 'Mh_bin_T', f.labels, 'k', get_dataset_display_name(f))



def plots(*datasets, truth_on=False):
    contains_20_data = False
    for f in datasets:
        if ('20' in f.name):
            contains_20_data = True
    
    # TODO: I believe that Mh and Mstar don't have any h factors, but should double check.
    # Probably depends on what was given to the group finder?
    # LHMR
    plt.figure(dpi=DPI)
    for f in datasets:
        if ('20' not in f.name):
            lcen_means = f.centrals.groupby('Mh_bin', observed=False).apply(Lgal_vmax_weighted)
            lcen_scatter = f.centrals.groupby('Mh_bin', observed=False).L_gal.std() # TODO not right?
            plt.errorbar(f.labels, lcen_means, yerr=lcen_scatter, label=get_dataset_display_name(f), color=f.color)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$M_{halo}$')
    plt.ylabel('$L_{cen}$')
    #plt.title("Central Luminosity vs. Halo Mass")
    legend(datasets)
    plt.xlim(1E10,1E15)
    plt.ylim(3E7,2E12)
    plt.draw()

    if contains_20_data:
        plt.figure(dpi=DPI)
        for f in datasets:
            if ('20' in f.name):
                lcen_means = f.centrals.groupby('Mh_bin', observed=False).apply(Lgal_vmax_weighted)
                lcen_scatter = f.centrals.groupby('Mh_bin', observed=False).L_gal.std()
                plt.errorbar(f.labels, lcen_means, yerr=lcen_scatter, label=get_dataset_display_name(f), color=f.color)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('$M_{halo}$')
        plt.ylabel('$log(L_{cen})$')
        plt.title("Central Luminosity vs. Halo Mass")
        legend(datasets)
        plt.xlim(1E10,1E15)
        plt.ylim(3E8,2E12)
        plt.draw()

    # SHMR
    plt.figure(dpi=DPI)
    for f in datasets:
        mcen_means = f.centrals.groupby('Mh_bin', observed=False).apply(mstar_vmax_weighted)
        #mcen_scatter = f.centrals.groupby('Mh_bin', observed=False).apply(mstar_std_vmax_weighted)
        plt.plot(f.labels, mcen_means, f.marker, label=get_dataset_display_name(f), color=f.color)
        #plt.errorbar(f.labels, mcen_means, yerr=mcen_scatter, label=get_dataset_display_name(f), color=f.color)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$M_{halo}$')
    plt.ylabel('$M_{\\star}$')
    legend(datasets)
    plt.xlim(1E10,1E15)
    #plt.ylim(1E6,3E12)
    plt.draw()

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
    plt.xlabel('$M_{halo}$')
    plt.ylabel('$M_{\\star}/M_{halo}$')
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
        #ax2.bar(f.L_gal_labels+(widths*idx), f.sats.groupby('Lgal_bin', observed=False).apply(count_vmax_weighted), width=widths, color=f.color, align='edge', alpha=0.4)
        ax2.bar(f.labels+(widths*idx), f.all_data.groupby('Mh_bin', observed=False).size(), width=widths, color=f.color, align='edge', alpha=0.4)
        idx+=1
    ax2.set_ylabel('$N_{gal}$')
    ax2.set_yscale('log')
    plt.draw()

    # fsat vs Lgal with Ngal
    fig,ax1=plt.subplots()
    fig.set_dpi(DPI)
    for f in datasets:
        plt.plot(f.L_gal_labels, f.f_sat, f.marker, label=get_dataset_display_name(f), color=f.color)
    if truth_on:
        for f in datasets:
            if f.has_truth:
                plt.plot(f.L_gal_labels, f.truth_f_sat, 'v', label=f"{get_dataset_display_name(f)} Truth", color=f.color)
    ax1.set_xscale('log')
    ax1.set_xlabel("$L_{\\mathrm{gal}}~[\\mathrm{h}^{-2} \\mathrm{L}_\\odot]$")
    ax1.set_ylabel("$f_{\\mathrm{sat}}$")
    #ax1.set_title("Satellite fraction vs Galaxy Luminosity")
    ax1.set_xlim(2E7,3E11)
    ax1.set_ylim(0.0,0.6)
    legend_ax(ax1, datasets)
    ax2 = ax1.twinx()
    idx = 0
    for f in datasets:
        widths = np.zeros(len(f.L_gal_bins)-1)
        for i in range(0,len(f.L_gal_bins)-1):
            widths[i]=(f.L_gal_bins[i+1] - f.L_gal_bins[i]) / len(datasets)
        # This version 1/vmax weights the counts
        #ax2.bar(f.L_gal_labels+(widths*idx), f.sats.groupby('Lgal_bin', observed=False).apply(count_vmax_weighted), width=widths, color=f.color, align='edge', alpha=0.4)
        ax2.bar(f.L_gal_labels+(widths*idx), f.all_data.groupby('Lgal_bin', observed=False).size(), width=widths, color=f.color, align='edge', alpha=0.4)
        idx+=1
    ax2.set_ylabel('$N_{gal}$')
    ax2.set_yscale('log')
    fig.tight_layout()

    # fsat vs Lgal 
    for xmin in LGAL_XMINS:
        fig,ax1=plt.subplots()
        fig.set_dpi(DPI)
        for f in datasets:
            plt.plot(f.L_gal_labels, f.f_sat, f.marker, label=get_dataset_display_name(f), color=f.color)
        if truth_on:
            for f in datasets:
                if 'is_sat_truth' in f.all_data.columns:
                    plt.plot(f.L_gal_labels, f.truth_f_sat, 'v', label=f"{f.name} Truth", color=f.color)
        ax1.set_xscale('log')
        ax1.set_xlabel("$L_{\\mathrm{gal}}~[\\mathrm{h}^{-2} \\mathrm{L}_\\odot]$")
        ax1.set_ylabel("$f_{\\mathrm{sat}}$")
        legend_ax(ax1, datasets)
        X_MAX = 1E11
        ax1.set_xlim(xmin,X_MAX)
        ax1.set_ylim(0.0,0.6)
        ax2=ax1.twiny()
        ax2.plot(Mr_gal_labels, datasets[0].f_sat, ls="")
        ax2.set_xlim(log_solar_L_to_abs_mag_r(np.log10(xmin)), log_solar_L_to_abs_mag_r(np.log10(X_MAX)))
        ax2.set_xlabel("$M_r$ - 5log(h)")

    #ax2 = ax1.twinx()
    #idx = 0
    #for f in datasets:
    #    widths = np.zeros(len(f.L_gal_bins)-1)
    #    for i in range(0,len(f.L_gal_bins)-1):
    #        widths[i]=(f.L_gal_bins[i+1] - f.L_gal_bins[i]) / (len(datasets))
    #    ax2.bar(f.L_gal_labels+(widths*idx), f.all_data[f.all_data.is_sat == True].groupby('Lgal_bin', observed=False).size(), width=widths, color=f.color, align='edge', alpha=0.4)
    #    idx+=1
    #ax2.set_ylabel('$N_{sat}$')
    fig.tight_layout()

    if len(datasets) == 1:
        plots_color_split(*datasets, total_on=True)

    for d in datasets:
        plots_color_split(d, truth_on=truth_on)
        if 'z_assigned_flag' in d.all_data.columns:
            plots_color_split_lost_split(d)
        if d.has_truth:
            q_gals = d.all_data[d.all_data.quiescent]
            sf_gals = d.all_data[np.invert(d.all_data.quiescent)]
            plots_color_split_lost_split_inner(d.name + " Sim. Truth", d.L_gal_labels, d.L_gal_bins, q_gals, sf_gals, fsat_truth_vmax_weighted)

    print("TOTAL f_sat - entire sample: ")
    for f in datasets:
        print(f.name)
        total_f_sat(f)

def plots_color_split_lost_split(f):
    q_gals = f.all_data[f.all_data.quiescent]
    sf_gals = f.all_data[np.invert(f.all_data.quiescent)]
    plots_color_split_lost_split_inner(get_dataset_display_name(f), f.L_gal_labels, f.L_gal_bins, q_gals, sf_gals, fsat_vmax_weighted)

def plots_color_split_lost_split_inner(name, L_gal_labels, L_gal_bins, q_gals, sf_gals, aggregation_func, show_plot=True):
    q_lost = q_gals[q_gals.z_assigned_flag].groupby(['Lgal_bin'], observed=False)
    q_obs = q_gals[~q_gals.z_assigned_flag].groupby(['Lgal_bin'], observed=False)
    sf_lost = sf_gals[sf_gals.z_assigned_flag].groupby(['Lgal_bin'], observed=False)
    sf_obs = sf_gals[~sf_gals.z_assigned_flag].groupby(['Lgal_bin'], observed=False)
    fsat_qlost = q_lost.apply(aggregation_func)
    fsat_qobs = q_obs.apply(aggregation_func)
    fsat_sflost = sf_lost.apply(aggregation_func)
    fsat_sfobs = sf_obs.apply(aggregation_func)
    fsat_qtot = q_gals.groupby(['Lgal_bin'], observed=False).apply(aggregation_func)
    fsat_sftot = sf_gals.groupby(['Lgal_bin'], observed=False).apply(aggregation_func)

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
        ax3.set_yscale('log')

        ax4 = ax2.twinx()
        idx = 0
        ax4.bar(L_gal_labels+(widths*idx), sf_lost.size(), width=widths, color='orange', align='edge', alpha=0.4)
        idx+=1
        ax4.bar(L_gal_labels+(widths*idx), sf_obs.size(), width=widths, color='k', align='edge', alpha=0.4)
        ax4.set_ylabel('$N_{gal}$')
        ax4.set_yscale('log')

        X_MIN = 3E7
        X_MAX = 1E11
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
        ax1.set_xlabel("$L_{\\mathrm{gal}}~[\\mathrm{h}^{-2} \\mathrm{L}_\\odot]$")
        ax2.set_xlabel("$L_{\\mathrm{gal}}~[\\mathrm{h}^{-2} \\mathrm{L}_\\odot]$")
        ax1.set_ylabel("$f_{\\mathrm{sat}}$ ")
        ax2.set_ylabel("$f_{\\mathrm{sat}}$ ")
        ax1.legend()
        ax2.legend()
        ax1.set_xlim(3E7,1E11)
        ax2.set_xlim(3E7,1E11)
        ax1.set_ylim(0.0,1.0)
        ax2.set_ylim(0.0,1.0)

        fig.suptitle(f"{name}")
        fig.tight_layout()

    return fsat_qlost.to_numpy(), fsat_qobs.to_numpy(), fsat_qtot.to_numpy(), fsat_sflost.to_numpy(), fsat_sfobs.to_numpy(), fsat_sftot.to_numpy()

def compare_fsat_color_split(dataone, datatwo):
    temp1 = dataone.marker
    temp2 = datatwo.marker
    dataone.marker = '-'
    datatwo.marker = '--'
    plots_color_split(dataone, datatwo)
    dataone.marker = temp1
    datatwo.marker = temp2

def plots_color_split(*datasets, truth_on=False, total_on=False):

    for xmin in LGAL_XMINS:

        fig,ax1=plt.subplots()
        fig.set_dpi(DPI)
        for f in datasets:
            if not hasattr(f, 'f_sat_q'):
                f.f_sat_q = f.all_data[f.all_data.quiescent].groupby(['Lgal_bin'], observed=False).apply(fsat_vmax_weighted)
            if not hasattr(f, 'f_sat_sf'):
                f.f_sat_sf = f.all_data[np.invert(f.all_data.quiescent)].groupby(['Lgal_bin'], observed=False).apply(fsat_vmax_weighted)
            plt.plot(f.L_gal_labels, f.f_sat_q, f.marker, label=get_dataset_display_name(f) + " Quiescent", color='r')
            plt.plot(f.L_gal_labels, f.f_sat_sf, f.marker, label=get_dataset_display_name(f) + " Star-forming", color='b')

            if total_on:
                plt.plot(f.L_gal_labels, f.f_sat, label=get_dataset_display_name(f) + " Total", color='k')

        for f in datasets:
            if truth_on:
                if f.has_truth:
                    truth_on = False
                    if not hasattr(f, 'f_sat_q_t'):
                        f.f_sat_q_t = f.all_data[f.all_data.quiescent].groupby(['Lgal_bin'], observed=False).apply(fsat_truth_vmax_weighted)
                    if not hasattr(f, 'f_sat_sf_t'):
                        f.f_sat_sf_t = f.all_data[np.invert(f.all_data.quiescent)].groupby(['Lgal_bin'], observed=False).apply(fsat_truth_vmax_weighted)
                    plt.plot(f.L_gal_labels, f.f_sat_q_t, 'x', label="Simulation's Truth", color='r')
                    plt.plot(f.L_gal_labels, f.f_sat_sf_t, 'x', label="Simulation's Truth", color='b')


        ax1.set_xscale('log')
        ax1.set_xlabel("$L_{\\mathrm{gal}}~[\\mathrm{h}^{-2} \\mathrm{L}_\\odot]$")
        ax1.set_ylabel("$f_{\\mathrm{sat}}$ ")
        ax1.legend()
        X_MAX = 1E11
        ax1.set_xlim(xmin,X_MAX)
        ax1.set_ylim(0.0,1.0)

        ax2=ax1.twiny()
        ax2.plot(Mr_gal_labels, datasets[0].f_sat_q, ls="")
        ax2.set_xlim(log_solar_L_to_abs_mag_r(np.log10(xmin)), log_solar_L_to_abs_mag_r(np.log10(X_MAX)))
        ax2.set_xlabel("$M_r$ - 5log(h)")


        fig.tight_layout()



def total_f_sat(ds):
    """
    Prints out the total f_sat for a dataframe, 1/Vmax weighted and not. 
    """
    print(f"  (no weight):  {ds.all_data['is_sat'].mean():.3f}")
    print(f"  (1 / V_max):  {fsat_vmax_weighted(ds.all_data):.3f}")
    
    if ds.has_truth:
        print(f"  Truth (no weight):  {ds.all_data['is_sat_truth'].mean():.3f}")
        print(f"  Truth (1 / V_max):  {fsat_truth_vmax_weighted(ds.all_data):.3f}")

def fsat_by_z_bins(dataset, z_bins=np.array([0.0, 0.05, 0.1, 0.2, 0.3, 1.0]), show_plots=True, aggregation=fsat_vmax_weighted):
    # Call plots_color_split_lost_split for a few z bins
    fsat_qlost_arr, fsat_qobs_arr, fsat_qtot_arr, fsat_sflost_arr, fsat_sfobs_arr, fsat_sftot_arr = [], [], [], [], [], []
    L_bin_number = 25

    for i in range(0, len(z_bins)-1):
        z_low = z_bins[i]
        z_high = z_bins[i+1]
        z_cut = np.all([dataset.all_data.z > z_low, dataset.all_data.z < z_high], axis=0)
        print(f"z: {z_low:.2} - {z_high:.2} ({np.sum(z_cut)} galaxies)")
        q_gals = dataset.all_data[np.all([dataset.all_data.quiescent, z_cut], axis=0)]
        #q_gals.reset_index(drop=True, inplace=True)
        sf_gals = dataset.all_data[np.all([np.invert(dataset.all_data.quiescent), z_cut], axis=0)]
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
    ax1.set_xlabel("$L_{\\mathrm{gal}}~[\\mathrm{h}^{-2} \\mathrm{L}_\\odot]$")
    ax1.set_ylabel("$N_{gal}$")
    #ax1.set_title("Galaxy Luminosity Counts")
    legend(datasets)
    ax1.set_xlim(L_MIN,L_MAX)
    #ax1.set_ylim(0.5,1E4)

    # Twin axis for Mr
    ax2=ax1.twiny()
    ax2.plot(Mr_gal_labels, values[0], ls="")
    ax2.set_xlim(log_solar_L_to_abs_mag_r(np.log10(L_MIN)), log_solar_L_to_abs_mag_r(np.log10(L_MAX)))
    ax2.set_xlabel("$M_r$ - 5log(h)")


def qf_cen_plot(*datasets):
    """
    Quiescent Fraction of Central Galaxies.
    """
    fig,ax1=plt.subplots()
    fig.set_dpi(DPI)
    for f in datasets:
        #if not hasattr(f, 'qf_gmr'):
        f.qf_gmr = f.centrals.groupby('Lgal_bin', observed=False).apply(qf_BGS_gmr_vmax_weighted)
        #if not hasattr(f, 'qf_dn4000'):
        f.qf_dn4000 = f.centrals.groupby('Lgal_bin', observed=False).apply(qf_Dn4000_smart_eq_vmax_weighted)
        f.qf_dn4000_hard = f.centrals.groupby('Lgal_bin', observed=False).apply(qf_Dn4000_1_6_vmax_weighted)
        plt.plot(f.L_gal_labels, f.qf_gmr, '.', label=f'0.1^(g-r) < {GLOBAL_RED_COLOR_CUT}', color='b')
        plt.plot(f.L_gal_labels, f.qf_dn4000, '-', label='Dn4000 Eq.1', color='g')
        plt.plot(f.L_gal_labels, f.qf_dn4000_hard, '-', label='Dn4000 > 1.6', color='r')

    ax1.set_xscale('log')
    ax1.set_xlabel("$L_{\\mathrm{cen}}~[\\mathrm{h}^{-2} \\mathrm{L}_\\odot]$")
    ax1.set_ylabel("$f_{\\mathrm{Q}}$ ")
    #ax1.set_title("Satellite fraction vs Galaxy Luminosity")
    ax1.legend()
    X_MIN = 3E8
    X_MAX = 1E11
    ax1.set_xlim(X_MIN,X_MAX)
    ax1.set_ylim(0.0,1.0)

# It gives same result as NFW version! Good
def get_vir_radius_mine(halo_mass):
    _cosmo = get_MXXL_cosmology()
    rho_m = (_cosmo.critical_density(0) * _cosmo.Om(0))
    return np.power(((3/(4*math.pi)) * halo_mass / (200*rho_m)), (1/3)).to(u.kpc).value

# TODO use this again if needed
def post_process(frame):
    """
    # TODO make work for UCHUU too; need refactoring regarding halo mass property names, etc
    """
    df: pd.DataFrame = frame.all_data
    
    # Calculate additional halo properties
    masses = df.loc[:, 'mxxl_halo_mass'].to_numpy() * 1E10 * u.solMass
    df.loc[:, 'mxxl_halo_vir_radius_guess'] = get_vir_radius_mine(masses)

    _cosmo = FlatLambdaCDM(H0=73, Om0=0.25, Ob0=0.045, Tcmb0=2.725, Neff=3.04) 
    # TODO comoving or proper?
    as_per_kpc = _cosmo.arcsec_per_kpc_proper(df.loc[:, 'z'].to_numpy())
    df.loc[:, 'mxxl_halo_vir_radius_guess_arcsec'] =  df.loc[:, 'mxxl_halo_vir_radius_guess'] * as_per_kpc.to(u.arcsec / u.kpc).value

    # Luminosity distance to z_obs
    df.loc[:, 'ldist_true'] = z_to_ldist(df.z_obs.to_numpy())



# Halo Masses (in group finder abundance matching)
def group_finder_centrals_halo_masses_plots(all_df, comparisons):
    
    count_to_use = np.max([len(c.all_data[c.all_data.index == c.all_data.igrp]) for c in comparisons])
    
    all_centrals = all_df.all_data[all_df.all_data.index == all_df.all_data.igrp]

    #if len(all_df.all_data) > count_to_use:
    #    # Randomly sample to match the largest comparison
    #    print(len(all_centrals))
    #    print(count_to_use)
    #    reduced_all_centrals_halos = np.random.choice(all_centrals.M_halo, count_to_use, replace=False)

    #angdist_bin_ind = np.digitize(reduced_all_centrals_halos, all_df.Mhalo_bins)
    angdist_bin_ind = np.digitize(all_centrals.M_halo, all_df.Mhalo_bins)
    all_bincounts = np.bincount(angdist_bin_ind)[0:len(all_df.Mhalo_bins)]
    all_density = all_bincounts / np.sum(all_bincounts)

    fig,axes=plt.subplots(nrows=1, ncols=1, figsize=(6,4))
    fig.set_dpi(DPI)
    axes.set_xscale('log')
    axes.set_ylim(-0.2, 0.2)
    axes.set_xlim(5E10,2E15)
    axes.set_xlabel('$M_{halo}$')
    axes.set_ylabel('Change in log(M)')
    axes.axline((3E10,0), (3E15,0), linestyle='--', color='k')
    #axes.set_title("Group Finder Halo Masses of Centrals")

    #axes[1].plot(all_to_use.Mhalo_bins, all_density, label="All Galaxies") 
    #axes[1].set_xscale('log')
    #axes[1].set_yscale('log')
    #axes[1].set_xlim(5E10,2E15)
    #axes[1].set_xlabel('$M_{halo}$')
    #axes[1].set_ylabel('Density of Galaxies')
    #axes[1].set_title("Group Finder Halo Masses of Centrals")

    for comparison in comparisons:

        centrals = comparison.all_data[comparison.all_data.index == comparison.all_data.igrp]
        angdist_bin_ind = np.digitize(centrals.M_halo, all_df.Mhalo_bins)
        bincounts = np.bincount(angdist_bin_ind)[0:len(all_df.Mhalo_bins)]
        density = bincounts / np.sum(bincounts)

        axes.plot(all_df.Mhalo_bins, np.log10(density / all_density), linestyle=comparison.marker, color=comparison.color, label=comparison.name) 
        #axes[1].plot(all_to_use.Mhalo_bins, density, linestyle=comparison.marker, color=comparison.color, label=comparison.name) 

    axes.legend()
    #axes[1].legend()

    # Look up the centrals from all in fiberonly
    for comparison in comparisons:

        centrals = comparison.all_data[comparison.all_data.index == comparison.all_data.igrp]
        catalog = coord.SkyCoord(ra=all_centrals.RA.to_numpy()*u.degree, dec=all_centrals.Dec.to_numpy()*u.degree, frame='icrs')
        to_match = coord.SkyCoord(ra=centrals.RA.to_numpy()*u.degree, dec=centrals.Dec.to_numpy()*u.degree, frame='icrs')
        idx, d2d, d3d = coord.match_coordinates_sky(to_match, catalog, nthneighbor=1, storekdtree=False)

        perfect_match = np.isclose(d2d.to(u.arcsec).value, 0, rtol=0.0, atol=0.0001) 
        # 0.0001 arcsec precision on matching doesn't hit floating point noise. You get same with 0.001
        print(f"What fraction of centrals in \'{comparison.name}\' are centrals in \'all\'? {np.sum(perfect_match) / len(d2d)}")



#####################################
# Purity and Completeness Functions
#####################################


def build_interior_bin_labels(bin_edges):
    labels = []
    for i in range(0,len(bin_edges)-1):
        labels.append(f"{bin_edges[i]:.2e} - {bin_edges[i+1]:.2e}")
    return labels

def test_purity_and_completeness(*sets):

    for s in sets:
        print(get_dataset_display_name(s))
        data = s.all_data

        if not s.has_truth:
            data['is_sat_truth'] = np.logical_or(data.galaxy_type == 1, data.galaxy_type == 3)

        # No 1/vmax weightings for these sums as we're just trying to calculate the purity/completeness
        # of our sample

        assigned_sats = data[data.is_sat == True]
        print(f"Purity of sats: {np.sum(assigned_sats.is_sat_truth) / len(assigned_sats.index):.3f}")

        true_sats = data[data.is_sat_truth == True]
        print(f"Completeness of sats: {np.sum(true_sats.is_sat) / len(true_sats.index):.3f}")

        assigned_centrals = data[data.is_sat == False]
        print(f"Purity of centrals: {1 - (np.sum(assigned_centrals.is_sat_truth) / len(assigned_centrals.index)):.3f}")

        true_centrals = data[data.is_sat_truth == False]
        print(f"Completeness of centrals: {1 - (np.sum(true_centrals.is_sat) / len(true_centrals.index)):.3f}")

        assigned_true_sats = assigned_sats[assigned_sats.is_sat_truth == True]
        assigned_sats_g = assigned_sats.groupby('Lgal_bin', observed=False).size().to_numpy()
        assigned_sats_correct_g = assigned_true_sats.groupby('Lgal_bin', observed=False).size().to_numpy()
        s.keep=np.nonzero(assigned_sats_g)
        s.purity_g = assigned_sats_correct_g[s.keep] / assigned_sats_g[s.keep]

        true_sats_assigned = true_sats[true_sats.is_sat == True]
        true_sats_g = true_sats.groupby('Lgal_bin', observed=False).size().to_numpy()
        true_sats_correct_g = true_sats_assigned.groupby('Lgal_bin', observed=False).size().to_numpy()
        s.keep2=np.nonzero(true_sats_g)
        s.completeness_g = true_sats_correct_g[s.keep2] / true_sats_g[s.keep2]

        assigned_true_centrals = assigned_centrals[assigned_centrals.is_sat_truth == False]
        assigned_centrals_g = assigned_centrals.groupby('Lgal_bin', observed=False).size().to_numpy()
        assigned_centrals_correct_g = assigned_true_centrals.groupby('Lgal_bin', observed=False).size().to_numpy()
        s.keep3=np.nonzero(assigned_centrals_g)
        s.purity_c_g = assigned_centrals_correct_g[s.keep3] / assigned_centrals_g[s.keep3]

        true_centrals_assigned = true_centrals[true_centrals.is_sat == False]
        true_centrals_g = true_centrals.groupby('Lgal_bin', observed=False).size().to_numpy()
        true_centrals_correct_g = true_centrals_assigned.groupby('Lgal_bin', observed=False).size().to_numpy()
        s.keep4=np.nonzero(true_centrals_g)
        s.completeness_c_g = true_centrals_correct_g[s.keep4] / true_centrals_g[s.keep4]


def purity_complete_plots(*sets):
    plt.rcParams.update({'font.size': 14})

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.set_dpi(DPI/2)

    XMIN = 6E7
    XMAX = 5E10

    axes[1][0].set_title('Satellite Purity')
    axes[1][0].set_xscale('log')
    axes[1][0].set_xlabel('$L_{\\mathrm{gal}}~[\\mathrm{h}^{-2} \\mathrm{L}_\\odot]$')
    axes[1][0].set_xlim(XMIN,XMAX)
    axes[1][0].set_ylim(0.4,1.0)

    axes[1][1].set_title('Satellite Completeness')
    axes[1][1].set_xscale('log')
    axes[1][1].set_xlabel('$L_{\\mathrm{gal}}~[\\mathrm{h}^{-2} \\mathrm{L}_\\odot]$')
    axes[1][1].set_xlim(XMIN,XMAX)
    axes[1][1].set_ylim(0.4,1.0)

    axes[0][0].set_title('Central Purity')
    axes[0][0].set_xscale('log')
    axes[0][0].set_xlabel('$L_{\\mathrm{gal}}~[\\mathrm{h}^{-2} \\mathrm{L}_\\odot]$')
    axes[0][0].set_xlim(XMIN,XMAX)
    axes[0][0].set_ylim(0.4,1.0)

    axes[0][1].set_title('Central Completeness')
    axes[0][1].set_xscale('log')
    axes[0][1].set_xlabel('$L_{\\mathrm{gal}}~[\\mathrm{h}^{-2} \\mathrm{L}_\\odot]$')
    axes[0][1].set_xlim(XMIN,XMAX)
    axes[0][1].set_ylim(0.4,1.0)

    for s in sets:
        axes[1][0].plot(s.L_gal_bins[s.keep], s.purity_g, s.marker, label=f"{get_dataset_display_name(s)}", color=s.color)
        axes[1][1].plot(s.L_gal_bins[s.keep2], s.completeness_g, s.marker, label=f"{get_dataset_display_name(s)}", color=s.color)
        axes[0][0].plot(s.L_gal_bins[s.keep3], s.purity_c_g, s.marker, label=f"{get_dataset_display_name(s)}", color=s.color)
        axes[0][1].plot(s.L_gal_bins[s.keep4], s.completeness_c_g, s.marker, label=f"{get_dataset_display_name(s)}", color=s.color)

    
    axes[0][0].legend()
    fig.tight_layout()


    font_restore()

    # Make just the satellite purity plot on its own
    plt.figure(dpi=DPI)
    for s in sets:
        plt.plot(s.L_gal_bins[s.keep], s.purity_g, s.marker, label=f"{get_dataset_display_name(s)}", color=s.color)

    plt.xscale('log')
    plt.xlabel('$L_{\\mathrm{gal}}~[\\mathrm{h}^{-2} \\mathrm{L}_\\odot]$')
    plt.ylabel('Satellite Purity')
    plt.legend()
    plt.xlim(XMIN,XMAX)
    plt.ylim(0.4,1.0)
    plt.tight_layout()




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

        lost_galaxies = data.all_data[data.all_data.z_assigned_flag]
        print(len(lost_galaxies), "lost galaxies")

        lost_galaxies = lost_galaxies[lost_galaxies['assigned_halo_id'] != 0]
        print(len(lost_galaxies), "lost galaxies after removing ones with no MXXL halo ID (these correspond to halos that were too small for the simulation and were added by hand)")

        lost_galaxies_same_halo = np.equal(lost_galaxies['assigned_halo_id'], lost_galaxies['mxxl_halo_id'])
        print("Fraction of time assigned halo ID is the same as the galaxy's actual halo ID: {0:.3f}".format(np.sum(lost_galaxies_same_halo) / len(lost_galaxies_same_halo)))
        
        lost_galaxies_same_halo_mass = np.isclose(lost_galaxies['assigned_halo_mass'], lost_galaxies['mxxl_halo_mass'], atol=0.0, rtol=1e-03)
        print("Fraction of time assigned halo mass is \'the same\' as the galaxy's actual halo mass: {0:.3f}".format(np.sum(lost_galaxies_same_halo_mass) / len(lost_galaxies_same_halo_mass)))
      
        z_thresh=0.01
        lost_galaxies_similar_z = np.isclose(lost_galaxies['z'], lost_galaxies['z_obs'], atol=z_thresh, rtol=0.0)         
        print("Fraction of time assigned z is the target z +/- {0:.3f}:".format(z_thresh), np.sum(lost_galaxies_similar_z) / len(lost_galaxies_similar_z))

        z_thresh=0.005
        lost_galaxies_similar_z = np.isclose(lost_galaxies['z'], lost_galaxies['z_obs'], atol=z_thresh, rtol=0.0)         
        print("Fraction of time assigned z is the target z +/- {0:.3f}:".format(z_thresh), np.sum(lost_galaxies_similar_z) / len(lost_galaxies_similar_z))

        z_thresh=0.003
        lost_galaxies_similar_z = np.isclose(lost_galaxies['z'], lost_galaxies['z_obs'], atol=z_thresh, rtol=0.0)         
        print("Fraction of time assigned z is the target z +/- {0:.3f}:".format(z_thresh), np.sum(lost_galaxies_similar_z) / len(lost_galaxies_similar_z))

        z_thresh=0.001
        lost_galaxies_similar_z = np.isclose(lost_galaxies['z'], lost_galaxies['z_obs'], atol=z_thresh, rtol=0.0)        
        print("Fraction of time assigned z is the target z +/- {0:.3f}:".format(z_thresh), np.sum(lost_galaxies_similar_z) / len(lost_galaxies_similar_z))
        
        # TODO as a function of reshift. But we essentially already have this from the direct MXXL data plots

        #z_bins = np.linspace(min(data.all_data.z), max(data.all_data.z), 20)
        #z_labels = z_bins[0:len(z_bins)-1] 
        #data.all_data['z_bin'] = pd.cut(x = data.all_data['z'], bins = z_bins, labels = z_labels, include_lowest = True)

        #groupby_z = lost_galaxies.groupby('z_bin')['same_halo_mass'].sum() / lost_galaxies.groupby('z_bin')['same_halo_mass'].count()

        #plt.plot(z_labels, groupby_z)
        #plt.xlabel('$z_{eff}$ (effective/assigned redshift)')
        #plt.ylabel('Fraction Assigned Halo = True Host Halo')
        