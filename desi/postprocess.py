import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as u
from pyutils import *
import types
import numpy.ma as ma
import math


# Common PLT helpers
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
def get_color(i):
    co = colors[i%len(colors)]
    return co

DPI = 1200
FONT_SIZE_DEFAULT = 12

plt.style.use('default')
plt.rcParams.update({'font.size': FONT_SIZE_DEFAULT})

def font_restore():
    plt.rcParams.update({'font.size': FONT_SIZE_DEFAULT})



# Shared bins for various purposes
Mhalo_bins = np.logspace(10, 15.5, 40)
Mhalo_labels = Mhalo_bins[0:len(Mhalo_bins)-1] 

L_gal_bins = np.logspace(6, 12.5, 40)
L_gal_labels = L_gal_bins[0:len(L_gal_bins)-1]


def process_uchuu(filename):
    filename_props = str.replace(filename, ".out", "_galprops.dat")

    df = pd.read_csv(filename, delimiter=' ', names=('RA', 'Dec', 'z', 'L_gal', 'V_max', 'P_sat', 'M_halo', 'N_sat', 'L_tot', 'igrp', 'unknown'))
    galprops = pd.read_csv(filename_props, delimiter=' ', names=('app_mag', 'g_r', 'central', 'uchuu_halo_mass', 'uchuu_halo_id'), dtype={'uchuu_halo_id': np.int64, 'central': np.bool_})
    all_data = pd.merge(df, galprops, left_index=True, right_index=True)

    return process_core(filename, all_data)

def process(filename):
    filename_props = str.replace(filename, ".out", "_galprops.dat")

    df = pd.read_csv(filename, delimiter=' ', names=('RA', 'Dec', 'z', 'L_gal', 'V_max', 'P_sat', 'M_halo', 'N_sat', 'L_tot', 'igrp', 'unknown'))
    galprops = pd.read_csv(filename_props, delimiter=' ', names=('app_mag', 'g_r', 'galaxy_type', 'mxxl_halo_mass', 'fiber_assigned_0', 'assigned_halo_mass', 'z_obs', 'mxxl_halo_id', 'assigned_halo_id'), dtype={'mxxl_halo_id': np.int32, 'assigned_halo_id': np.int32})
    all_data = pd.merge(df, galprops, left_index=True, right_index=True)


    return process_core(filename, all_data)

def process_BGS(filename):
    filename_props = str.replace(filename, ".out", "_galprops.dat")

    df = pd.read_csv(filename, delimiter=' ', names=('RA', 'Dec', 'z', 'L_gal', 'V_max', 'P_sat', 'M_halo', 'N_sat', 'L_tot', 'igrp', 'unknown'))
    galprops = pd.read_csv(filename_props, delimiter=' ', names=('app_mag', 'target_id', 'z_assigned_flag'), dtype={'target_id': np.int64, 'z_assigned_flag': np.bool_})
    all_data = pd.merge(df, galprops, left_index=True, right_index=True)

    return process_core(filename, all_data)


def process_core(filename, all_data):

    # Drop bad data, should have been cleaned up earlier though!
    orig_count = len(all_data)
    all_data = all_data[all_data.M_halo != 0]
    new_count = len(all_data)
    if (orig_count != new_count):
        print("Dropped {0} bad galaxies".format(orig_count - new_count))

    all_data['is_sat'] = (all_data.index != all_data.igrp).astype(int)
    if 'galaxy_type' in all_data.columns:
        all_data['is_sat_truth'] = np.logical_or(all_data.galaxy_type == 1, all_data.galaxy_type == 3).astype(int)
    elif 'central' in all_data.columns:
        all_data['is_sat_truth'] = np.invert(all_data.central)
    all_data['logLgal'] = np.log10(all_data.L_gal)

    #bins = np.logspace(np.log10(min(all_data.M_halo)), np.log10(max(all_data.M_halo)), 30)
    #labels = bins[0:len(bins)-1] # using bottom (or top?) value, not middle
    all_data['Mh_bin'] = pd.cut(x = all_data['M_halo'], bins = Mhalo_bins, labels = Mhalo_labels, include_lowest = True)
    
    centrals = all_data[all_data.index == all_data.igrp]
    #logmstar_means = centrals.groupby('Mh_bin').log_M_star.mean()
    #logmstar_scatter = centrals.groupby('Mh_bin').log_M_star.std()
    loglcen_means = centrals.groupby('Mh_bin').logLgal.mean()
    loglcen_scatter = centrals.groupby('Mh_bin').logLgal.std()

    # Compute f_sat(Lgal)
    #L_gal_bins = np.logspace(np.log10(min(all_data.L_gal)), np.log10(max(all_data.L_gal)), 30)
    #L_gal_labels = L_gal_bins[0:len(L_gal_bins)-1] # using bottom (or top?) value, not middle
    all_data['Lgal_bin'] = pd.cut(x = all_data['L_gal'], bins = L_gal_bins, labels = L_gal_labels, include_lowest = True)
    
    f_sat = all_data.groupby('Lgal_bin').is_sat.mean()
    Lgal_counts = all_data.groupby('Lgal_bin').RA.count()

    dataset = types.SimpleNamespace()
    dataset.filename = filename[filename.rfind('/')+1 : len(filename)-4]
    dataset.all_data = all_data
    dataset.Mhalo_bins = Mhalo_bins
    dataset.labels = Mhalo_labels
    dataset.centrals = centrals
    #dataset.logmstar_means = logmstar_means
    #dataset.logmstar_scatter = logmstar_scatter
    dataset.loglcen_means = loglcen_means
    dataset.loglcen_scatter = loglcen_scatter
    dataset.L_gal_bins = L_gal_bins
    dataset.L_gal_labels = L_gal_labels
    dataset.f_sat = f_sat
    dataset.Lgal_counts = Lgal_counts

    return dataset


def legend(datasets):
    if len(datasets) > 1:
        plt.legend()

def legend_ax(ax, datasets):
    if len(datasets) > 1:
        ax.legend()

def plots(*datasets, truth_on=False):
    contains_20_data = False
    for f in datasets:
        if ('20' in f.name):
            contains_20_data = True
    
    plt.figure(dpi=DPI)
    for f in datasets:
        if ('20' not in f.name):
            plt.errorbar(f.labels, f.loglcen_means, yerr=f.loglcen_scatter, label=f.name, color=f.color)
    plt.xscale('log')
    plt.xlabel('$M_{halo}$')
    plt.ylabel('$log(L_{cen})$')
    #plt.title("Central Luminosity vs. Halo Mass")
    legend(datasets)
    plt.xlim(2E11,1E15)
    plt.draw()

    if contains_20_data:
        plt.figure(dpi=DPI)
        for f in datasets:
            if ('20' in f.name):
                plt.errorbar(f.labels, f.loglcen_means, yerr=f.loglcen_scatter, label=f.name, color=f.color)
        plt.xscale('log')
        plt.xlabel('$M_{halo}$')
        plt.ylabel('$log(L_{cen})$')
        plt.title("Central Luminosity vs. Halo Mass")
        legend(datasets)
        plt.xlim(2E11,1E15)
        plt.draw()
    """
    plt.figure(dpi=DPI)    
    for f in datasets:
        if ('20' not in f.name):
            plt.plot(f.labels, f.loglcen_scatter, color=f.color, label=f.name)
    plt.xscale('log')
    plt.xlabel('$M_{halo}$')
    plt.ylabel('$\\sigma(\\log(L_{cen})$')
    plt.title("Central Luminosity Scatter vs. Halo Mass")
    legend(datasets)
    plt.xlim(2E11,1E15)
    plt.draw()

    if contains_20_data:
        plt.figure(dpi=DPI)    
        for f in datasets:
            if ('20' in f.name):
                plt.plot(f.labels, f.loglcen_scatter, color=f.color, label=f.name)
        plt.xscale('log')
        plt.xlabel('$M_{halo}$')
        plt.ylabel('$\\sigma(\\log(L_{cen})$')
        plt.title("Central Luminosity Scatter vs. Halo Mass")
        legend(datasets)
        plt.xlim(2E11,1E15)
        plt.draw()
    """
    """     
    plt.figure()
    for f in datasets:
        plt.scatter(f.centrals.M_halo, f.centrals.L_gal, alpha=0.002)
    plt.loglog()
    plt.xlabel('M_halo / h')
    plt.ylabel('L_gal / $h^2$)')
    plt.draw() 
    """

    make_N_sat_plot = False
    for f in datasets:
        if 'N_sat' in f.all_data.columns:
            make_N_sat_plot = True

    if make_N_sat_plot:
        plt.figure(dpi=DPI)
        for f in datasets:
            if 'N_sat' in f.all_data.columns:
                Nsat_means = f.all_data.groupby('Mh_bin').N_sat.mean()
                plt.plot(f.labels, Nsat_means, f.marker, label=f.name, color=f.color)
                #plt.hist(f.centrals.N_sat, np.arange(0,50,1), alpha=0.5)
        plt.loglog()    
        plt.ylabel("$<N_{sat}>$")    
        plt.xlabel('$M_{halo}$')
        plt.title("Mean Number of Satellites by Halo Mass")
        legend(datasets)
        plt.xlim(2E11,1E15)
        plt.draw()

    fig,ax1=plt.subplots()
    fig.set_dpi(DPI)
    for f in datasets:
        plt.plot(f.L_gal_labels, f.f_sat, f.marker, label=f.name, color=f.color)
    if truth_on:
        truth_f_sat = truth_on.all_data.groupby('Lgal_bin').is_sat_truth.mean()
        plt.plot(truth_on.L_gal_labels, truth_f_sat, label="Truth", color=truth_on.color)
    ax1.set_xscale('log')
    ax1.set_xlabel("$L_{gal}$")
    ax1.set_ylabel("$f_{sat}$")
    ax1.set_title("Satellite fraction vs Galaxy Luminosity")
    ax1.set_xlim(2E7,2E11)
    ax1.set_ylim(0.0,0.6)
    legend_ax(ax1, datasets)
    ax2 = ax1.twinx()
    idx = 0
    for f in datasets:
        widths = np.zeros(len(f.L_gal_bins)-1)
        for i in range(0,len(f.L_gal_bins)-1):
            widths[i]=(f.L_gal_bins[i+1] - f.L_gal_bins[i]) / len(datasets)
        ax2.bar(f.L_gal_labels+(widths*idx), f.all_data[f.all_data.is_sat == True].groupby('Lgal_bin').size(), width=widths, color=f.color, align='edge', alpha=0.4)
        idx+=1
    ax2.set_ylabel('$N_{sat}$')
    ax2.set_yscale('log')
    fig.tight_layout()

    fig,ax1=plt.subplots()
    fig.set_dpi(DPI)
    for f in datasets:
        plt.plot(f.L_gal_labels, f.f_sat, f.marker, label=f.name, color=f.color)
    if truth_on:
        truth_f_sat = truth_on.all_data.groupby('Lgal_bin').is_sat_truth.mean()
        plt.plot(truth_on.L_gal_labels, truth_f_sat, label="Truth", color=truth_on.color)
    ax1.set_xscale('log')
    ax1.set_xlabel("$L_{gal}$")
    ax1.set_ylabel("$f_{sat}$")
    #ax1.set_title("Satellite fraction vs Galaxy Luminosity")
    legend_ax(ax1, datasets)
    ax1.set_xlim(3E8,1E11)
    ax1.set_ylim(0.0,0.6)
    ax2 = ax1.twinx()
    idx = 0
    for f in datasets:
        widths = np.zeros(len(f.L_gal_bins)-1)
        for i in range(0,len(f.L_gal_bins)-1):
            widths[i]=(f.L_gal_bins[i+1] - f.L_gal_bins[i]) / (len(datasets))
        ax2.bar(f.L_gal_labels+(widths*idx), f.all_data[f.all_data.is_sat == True].groupby('Lgal_bin').size(), width=widths, color=f.color, align='edge', alpha=0.4)
        idx+=1
    ax2.set_ylabel('$N_{sat}$')
    fig.tight_layout()


    print("TOTAL f_sat: ")
    for f in datasets:
        print(f"  {f.name}:  {f.all_data['is_sat'].sum() / f.all_data['is_sat'].count():.3f}")
        if 'is_sat_truth' in f.all_data.columns:
            print(f"  Corresponding Simulation \'Truth\': {f.all_data['is_sat_truth'].sum() / f.all_data['is_sat_truth'].count():.3f}")

# It gives same result as NFW version! Good
def get_vir_radius_mine(halo_mass):
    _cosmo = get_MXXL_cosmology()
    rho_m = (_cosmo.critical_density(0) * _cosmo.Om(0))
    return np.power(((3/(4*math.pi)) * halo_mass / (200*rho_m)), (1/3)).to(u.kpc).value

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
        print(s.name)
        data = s.all_data

        assigned_sats = data[data.is_sat == True]
        print(f"Purity of sats: {np.sum(assigned_sats.is_sat_truth) / len(assigned_sats.index):.3f}")

        true_sats = data[data.is_sat_truth == True]
        print(f"Completeness of sats: {np.sum(true_sats.is_sat) / len(true_sats.index):.3f}")

        assigned_centrals = data[data.is_sat == False]
        print(f"Purity of centrals: {1 - (np.sum(assigned_centrals.is_sat_truth) / len(assigned_centrals.index)):.3f}")

        true_centrals = data[data.is_sat_truth == False]
        print(f"Completeness of centrals: {1 - (np.sum(true_centrals.is_sat) / len(true_centrals.index)):.3f}")

        assigned_true_sats = assigned_sats[assigned_sats.is_sat_truth == True]
        assigned_sats_g = assigned_sats.groupby('Lgal_bin').size().to_numpy()
        assigned_sats_correct_g = assigned_true_sats.groupby('Lgal_bin').size().to_numpy()
        s.keep=np.nonzero(assigned_sats_g)
        s.purity_g = assigned_sats_correct_g[s.keep] / assigned_sats_g[s.keep]

        true_sats_assigned = true_sats[true_sats.is_sat == True]
        true_sats_g = true_sats.groupby('Lgal_bin').size().to_numpy()
        true_sats_correct_g = true_sats_assigned.groupby('Lgal_bin').size().to_numpy()
        s.keep2=np.nonzero(true_sats_g)
        s.completeness_g = true_sats_correct_g[s.keep2] / true_sats_g[s.keep2]

        assigned_true_centrals = assigned_centrals[assigned_centrals.is_sat_truth == False]
        assigned_centrals_g = assigned_centrals.groupby('Lgal_bin').size().to_numpy()
        assigned_centrals_correct_g = assigned_true_centrals.groupby('Lgal_bin').size().to_numpy()
        s.keep3=np.nonzero(assigned_centrals_g)
        s.purity_c_g = assigned_centrals_correct_g[s.keep3] / assigned_centrals_g[s.keep3]

        true_centrals_assigned = true_centrals[true_centrals.is_sat == False]
        true_centrals_g = true_centrals.groupby('Lgal_bin').size().to_numpy()
        true_centrals_correct_g = true_centrals_assigned.groupby('Lgal_bin').size().to_numpy()
        s.keep4=np.nonzero(true_centrals_g)
        s.completeness_c_g = true_centrals_correct_g[s.keep4] / true_centrals_g[s.keep4]


def purity_complete_plots(*sets):
    plt.rcParams.update({'font.size': 14})

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.set_dpi(DPI/2)

    axes[1][0].set_title('Satellite Purity')
    axes[1][0].set_xscale('log')
    axes[1][0].set_xlabel('$L_{gal}$')
    axes[1][0].set_xlim(2E8,1E11)
    axes[1][0].set_ylim(0.4,1.0)

    axes[1][1].set_title('Satellite Completeness')
    axes[1][1].set_xscale('log')
    axes[1][1].set_xlabel('$L_{gal}$')
    axes[1][1].set_xlim(2E8,1E11)
    axes[1][1].set_ylim(0.4,1.0)

    axes[0][0].set_title('Central Purity')
    axes[0][0].set_xscale('log')
    axes[0][0].set_xlabel('$L_{gal}$')
    axes[0][0].set_xlim(2E8,1E11)
    axes[0][0].set_ylim(0.4,1.0)

    axes[0][1].set_title('Central Completeness')
    axes[0][1].set_xscale('log')
    axes[0][1].set_xlabel('$L_{gal}$')
    axes[0][1].set_xlim(2E8,1E11)
    axes[0][1].set_ylim(0.4,1.0)

    for s in sets:
        axes[1][0].plot(s.L_gal_bins[s.keep], s.purity_g, s.marker, label=f"{s.name}", color=s.color)
        axes[1][1].plot(s.L_gal_bins[s.keep2], s.completeness_g, s.marker, label=f"{s.name}", color=s.color)
        axes[0][0].plot(s.L_gal_bins[s.keep3], s.purity_c_g, s.marker, label=f"{s.name}", color=s.color)
        axes[0][1].plot(s.L_gal_bins[s.keep4], s.completeness_c_g, s.marker, label=f"{s.name}", color=s.color)

    
    axes[0][0].legend()
    fig.tight_layout()

    font_restore()








def resulting_halo_analysis(*sets):
    """
    Compares assigned halos to MXXL 'truth' halos.
    
    TODO: Only works on MXXL right now.
    """

    for data in sets:

        print(data.name)

        #same_halo_mass = np.isclose(data.all_data['assigned_halo_mass'], data.all_data['mxxl_halo_mass'], atol=0.0, rtol=1e-03)
        #same_mxxl_halo = data.all_data['assigned_halo_mass']
        #data.all_data['same_mxxl_halo'] = same_mxxl_halo

        lost_galaxies = data.all_data[data.all_data.fiber_assigned_0 == 0]
        print(len(lost_galaxies), "lost galaxies")

        # TODO understand this MXXL quirk
        lost_galaxies = lost_galaxies[lost_galaxies['assigned_halo_id'] != 0]
        print(len(lost_galaxies), "lost galaxies after removing ones with no MXXL halo ID (no idea why)")

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
        
