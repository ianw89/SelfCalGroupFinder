import numpy as np
import os
import re
from matplotlib import pyplot as plt
from pycorr import TwoPointEstimator
from matplotlib.lines import Line2D

def save_wp(savedir, red_results, blue_results, all_results, magbins):
    
     # Save the results to text files in the format we want, and also save the covariance matrix as numpy array
    for i in range(len(red_results)):
        red_wp, red_cov = red_results[i]
        blue_wp, blue_cov = blue_results[i]
        all_wp, all_cov = all_results[i]

        # Currently we're choosing not to use the full covariance matrix, just the diagonal for our chi squared
        # since the result of the jackknife tests was kinda weird correlation matrices.

        # Format is: rp wp wp_err
        if red_wp is not None:
            with open(os.path.join(savedir, f'wp_red_M{-magbins[i]:d}.dat'), 'w') as f:
                for j in range(len(red_wp)):
                    f.write(f'{red_wp[j,0]:.8f} {red_wp[j,2]:.8f} {red_wp[j,3]:.8f}\n')
            np.save(os.path.join(savedir, f'wp_red_M{-magbins[i]:d}_cov.npy'), red_cov)

        if blue_wp is not None:
            with open(os.path.join(savedir, f'wp_blue_M{-magbins[i]:d}.dat'), 'w') as f:
                for j in range(len(blue_wp)):
                    f.write(f'{blue_wp[j,0]:.8f} {blue_wp[j,2]:.8f} {blue_wp[j,3]:.8f}\n')
            np.save(os.path.join(savedir, f'wp_blue_M{-magbins[i]:d}_cov.npy'), blue_cov)
            
        if all_wp is not None:
            with open(os.path.join(savedir, f'wp_all_M{-magbins[i]:d}.dat'), 'w') as f:
                for j in range(len(all_wp)):
                    f.write(f'{all_wp[j,0]:.8f} {all_wp[j,2]:.8f} {all_wp[j,3]:.8f}\n')
            np.save(os.path.join(savedir, f'wp_all_M{-magbins[i]:d}_cov.npy'), all_cov)




def load_allcounts_from_disk(base_dir):
    """
    Recursively searches for and loads all 'allcounts*.npy' files from a base directory.

    Parses metadata from the filenames and returns a list of dictionaries, 
    each containing the loaded data and its associated parameters.

    Args:
        base_dir (str): The top-level directory to start the search from.

    Returns:
        list: A list of dictionaries. Each dictionary has two keys:
              'params': A dictionary of metadata parsed from the filename.
              'data': The loaded TwoPointEstimator object.
    """
    
    # Regex to parse the complex filename structure.
    # It captures named groups for each parameter.
    filename_pattern = re.compile(
        r"allcounts_BGS_BRIGHT"
        r"(?:_R-(?P<mag_thresh>[\d\.]+))?"           # Optional magnitude threshold
        r"(?:_R-(?P<mag_range>[\d\.]+-[\d\.]+))?"   # Optional magnitude range
        r"(?:_SERSIC-(?P<sersic>[\d\.-]+))?"  # Optional SERSIC cut
        r"(?:_(?P<sample_type>SF|Q|ALL))?"           # Optional sample type
        r"_(?P<region>GCcomb)"                # Region
        r"_(?P<zmin>[\d\.]+)"                 # zmin
        r"_(?P<zmax>[\d\.]+)"                 # zmax
        r"_(?P<weights>[\w_]+)"               # Weights
        r"_(?P<bin_type>\w+)"                 # Binning type
        r"_njack(?P<njack>\d+)"               # njack
        r"_nran(?P<nran>\d+)"                 # nran
        r"_split(?P<split>\d+)"               # split
        r"\.npy"
    )

    loaded_results = []
    print(f"Searching for allcounts files in: {base_dir}")

    if not os.path.isdir(base_dir):
        print(f"Error: Base directory not found: {base_dir}")
        return []

    for root, _, files in os.walk(base_dir):
        for file in files:
            match = filename_pattern.match(file)
            if match:
                full_path = os.path.join(root, file)
                params = match.groupdict()

                # Default sample_type to 'ALL' if not present in filename
                if params['sample_type'] is None:
                    params['sample_type'] = 'ALL'
                
                print(f"Found and loading: {file}")
                try:
                    # Load the TwoPointEstimator object
                    estimator = TwoPointEstimator.load(full_path)
                    loaded_results.append({
                        'params': params,
                        'data': estimator
                    })
                except Exception as e:
                    print(f"  -> Failed to load {full_path}: {e}")

    print(f"\nFinished search. Loaded {len(loaded_results)} files.")
    return loaded_results






def wp_thresholds(loaded_results, weight_type):
    """
    Plots wp(rp) for a list of loaded clustering results for a specific weight type.

    Creates a figure for each unique magnitude threshold.
    Choose ALL, SF, Q or any combination of them.

    Args:
        loaded_results (list): The list of dictionaries produced by
                               load_allcounts_from_disk.
        weight_type (str): The specific weight type to plot (e.g., 'WEIGHT_FKP_V1').
    """

    sdss_magbins = [19.5, 20.5, 21, 21.5, 22]
    colors = ['blue', 'limegreen', 'k', 'r', 'magenta']
    colors2 = ['blue', 'limegreen', 'k', 'r', 'magenta']

    # Filter results by the specified weight type, only mag_thresh entries, only ALL sample type
    filtered_results = [
        res for res in loaded_results
        if res['params']['weights'] == weight_type
        and res['params']['mag_thresh'] is not None
        and res['params']['sample_type'] == 'ALL'
    ]

    if not filtered_results:
        print(f"No threshold results found for weight_type='{weight_type}'.")
        return

    print(f"Plotting threshold results for weight_type='{weight_type}'")

    # Group results by magnitude threshold
    results_by_thresh = {}
    for result in filtered_results:
        thresh = result['params']['mag_thresh']
        if thresh not in results_by_thresh:
            results_by_thresh[thresh] = []
        results_by_thresh[thresh].append(result)

    # Sort thresholds numerically
    sorted_thresholds = sorted(results_by_thresh.keys(), key=float)

    fig, ax = plt.subplots(figsize=(5, 4))
    #fig.suptitle(f'wp(rp) by Mag Threshold', fontsize=16)

    for i, thresh in enumerate(sorted_thresholds):
        results = results_by_thresh[thresh]
        # Sort from low zmax to high zmax
        results.sort(key=lambda x: float(x['params']['zmax']))

        # There should typically be one result per threshold, but we loop just in case there are multiple (e.g., different rp bins, nran, zmax, etc)
        markers = ['o', '.']  # Different markers for multiple entries with the same threshold
        for j, item in enumerate(results):
            params = item['params']
            estimator = item['data']
            c = colors[i % len(colors)]
            marker = markers[j % len(markers)]

            if int(params['njack']) > 0:
                rp, wp, cov = estimator.get_corr(return_sep=True, return_cov=True, mode='wp')
                wp_err = np.sqrt(np.diag(cov))
            else:
                rp, wp = estimator.get_corr(return_sep=True, mode='wp')
                wp_err = None

            label = f"$M_r$ < -{thresh} (z<{params['zmax']})"
           # if marker == '.':
            ax.errorbar(rp, wp, yerr=wp_err, label=label, fmt=marker, markersize=3,capsize=3, alpha=0.8, color=c)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.1, 35)
    ax.set_xlabel(r'$r_p$ [Mpc/h]')
    ax.set_ylabel(r'$w_p(r_p)$')

    # Legend 1: colors = magnitude thresholds
    color_handles = [ ]
    for i, thresh in enumerate(sorted_thresholds):
        color_handles.append(Line2D([0], [0], color=colors[i % len(colors)], lw=2, label=f'$M_r$ < -{thresh}'))
    # Legend 2: symbols = BGS vs SDSS
    symbol_handles = [
        Line2D([0], [0], marker='o', color='gray', lw=0, label='BGS 17.6 fluxlim'),
        Line2D([0], [0], marker='.', color='gray', lw=0, label='BGS 19.5 fluxlim'),
    ]
    leg1 = ax.legend(handles=color_handles, fontsize=11, loc='lower left')
    ax.add_artist(leg1)  # keep first legend when adding second
    ax.legend(handles=symbol_handles, fontsize=11, loc='upper right')

    plt.tight_layout()




def compare_wp_thresholds_to_sdss(loaded_results, weight_type):
    """
    Plots wp(rp) for a list of loaded clustering results for a specific weight type.

    Creates a figure for each unique magnitude threshold.
    Choose ALL, SF, Q or any combination of them.

    Args:
        loaded_results (list): The list of dictionaries produced by
                               load_allcounts_from_disk.
        weight_type (str): The specific weight type to plot (e.g., 'WEIGHT_FKP_V1').
    """

    sdss_magbins = [19.5, 20.5, 21, 21.5, 22]
    colors = ['blue', 'limegreen', 'k', 'r', 'magenta']
    colors2 = ['blue', 'limegreen', 'k', 'r', 'magenta']

    # Filter results by the specified weight type, only mag_thresh entries, only ALL sample type
    filtered_results = [
        res for res in loaded_results
        if res['params']['weights'] == weight_type
        and res['params']['mag_thresh'] is not None
        and res['params']['sample_type'] == 'ALL'
    ]

    if not filtered_results:
        print(f"No threshold results found for weight_type='{weight_type}'.")
        return

    print(f"Plotting threshold results for weight_type='{weight_type}'")

    # Group results by magnitude threshold
    results_by_thresh = {}
    for result in filtered_results:
        thresh = result['params']['mag_thresh']
        if thresh not in results_by_thresh:
            results_by_thresh[thresh] = []
        results_by_thresh[thresh].append(result)

    # Sort thresholds numerically
    sorted_thresholds = sorted(results_by_thresh.keys(), key=float)

    fig, ax = plt.subplots(figsize=(5, 4))
    #fig.suptitle(f'wp(rp) by Mag Threshold', fontsize=16)

    for i, thresh in enumerate(sorted_thresholds):
        results = results_by_thresh[thresh]
        # Sort from low zmax to high zmax
        results.sort(key=lambda x: float(x['params']['zmax']))

        # There should typically be one result per threshold, but we loop just in case there are multiple (e.g., different rp bins, nran, zmax, etc)
        markers = ['v', '.', 'o']  # Different markers for multiple entries with the same threshold
        for j, item in enumerate(results):
            params = item['params']
            estimator = item['data']
            c = colors[i % len(colors)]
            marker = markers[j % len(markers)]

            if int(params['njack']) > 0:
                rp, wp, cov = estimator.get_corr(return_sep=True, return_cov=True, mode='wp')
                wp_err = np.sqrt(np.diag(cov))
            else:
                rp, wp = estimator.get_corr(return_sep=True, mode='wp')
                wp_err = None

            if marker == '.':
                print(params['mag_thresh'] )
                if params['mag_thresh'] == "19.52":
                    wp = wp * 0.398  # 0.4 dex shift down
                if params['mag_thresh'] == "20.64":
                    wp = wp * 0.630 # 0.2 dex shift down
                ax.errorbar(rp, wp, yerr=wp_err, fmt='o', markerfacecolor=c, markeredgecolor='k', markersize=4, capsize=5, ecolor='k', color=c)


    # SDSS Data
    from dataloc import PARAMS_SDSS_FOLDER
    #zehavi_bins = np.logspace(np.log10(0.13), np.log10(33), 13)
    zehavi_bins = [0.17, 0.27, 0.42, 0.67, 1.1, 1.7, 2.7, 4.2, 6.7, 10.6, 16.9, 26.8, 42.3]
    
    """
    for m in sdss_magbins:
        savedir = PARAMS_SDSS_FOLDER + f'sdss-thresh-{m:.1f}.csv'
        if not os.path.exists(savedir):
            print(f'File {savedir} not found')
            continue
        data = np.loadtxt(savedir, skiprows=1, dtype='float', delimiter=',')
        x = (zehavi_bins[:-1] + zehavi_bins[1:]) / 2
        print(np.shape(data))
        if np.shape(data)[0] == 11:
            x = x[1:]  # Skip first point for the faintest bin since it's not in the SDSS data I ripped
        plt.plot(x, data[:,1], '-', label=f'SDSS $M_r<-{m}$', color=colors2.pop(0), lw=2)
    """



    """
    0.17 2615  (491)  1028  (68)   586.2 (19.5) 455.7 (11.3) 366.1 (9.3) 307.0 (9.2) 322.5 (17.0) 313.3 (25.9) 294.3 (34.7)
    0.27 1189  (202)  731.7 (34.0) 402.9 (11.7) 296.9 (6.9)  264.3 (7.6) 228.5 (8.3) 231.1 (15.3) 230.2 (24.9) 221.5 (32.1)
    0.42 728.0 (96.3) 392.6 (17.1) 258.7 (6.7)  197.0 (5.1)  184.0 (6.6) 159.3 (7.2) 162.4 (12.8) 165.4 (21.1) 161.4 (27.6)
    0.67 491.4 (55.3) 228.6 (10.9) 163.2 (4.7)  134.1 (4.1)  128.6 (5.5) 110.4 (5.6) 114.6 (10.3) 118.3 (17.5) 114.7 (22.0)
    1.1 272.8  (23.2) 144.6 (6.4)  105.5 (3.0)  89.4  (3.3)  84.7  (4.3) 72.9 (4.2) 75.5 (7.7) 79.7 (13.2) 75.5 (16.5)
    1.7 154.4  (14.5) 94.3  (3.7)  68.9  (2.2)  61.1  (2.6)  59.4  (3.6) 49.8 (3.4) 50.6 (6.0) 53.8 (10.5) 48.6 (11.5)
    2.7 111.5  (10.4) 70.5  (2.7)  50.2  (2.1)  44.0  (2.3)  42.9  (3.3) 34.6 (2.9) 35.0 (4.7) 37.4 (7.8) 32.4 (7.7)
    4.2 94.5   (5.6)  48.6  (2.3)  35.5  (1.8)  31.2  (2.0)  30.9  (3.1) 24.6 (2.5) 24.2 (3.6) 25.9 (5.8) 19.7 (4.4)
    6.7 56.8   (3.8)  33.1  (1.8)  24.5  (1.6)  21.3  (1.8)  21.9  (2.7) 16.7 (2.4) 15.3 (2.9) 17.4 (4.5) 10.8 (2.8)
    10.6 35.1  (3.2)  20.9  (1.5)  15.3  (1.3)  13.7  (1.5)  14.6  (2.1) 10.7 (1.9) 9.20 (1.78) 10.6 (2.6) 6.35 (1.93)
    16.9 22.0  (2.2)  11.6  (1.2)  8.54  (0.94) 7.65  (1.07) 8.24  (1.32) 5.73 (1.28) 4.11 (1.29) 5.31 (1.42) 3.62 (1.34)
    26.8 11.4  (1.6)  6.04  (0.95) 4.11  (0.71) 4.09  (0.88) 4.88  (1.06) 2.82 (1.13) 1.81 (1.39) 3.56 (1.76) 2.14 (1.23)
    42.3 5.89  (1.21) 3.28  (0.64) 2.73  (0.54) 3.21  (0.70) 3.58  (0.85) 1.39 (0.91) 0.72 (1.24) 0.96 (1.02) 0.56 (1.26)
    """
    lw = 1
    zehavi_195 = np.array([307.0, 228.5, 159.3, 110.4, 72.9, 49.8, 34.6, 24.6, 16.7, 10.7, 5.73, 2.82, 1.39]) * 0.398
    zehavi_195_err = [9.2, 8.3, 7.2, 5.6, 4.2, 3.4, 2.9, 2.5, 2.4, 1.9, 1.28, 1.13, 0.91]
    plt.fill_between(zehavi_bins, np.array(zehavi_195) - np.array(zehavi_195_err), np.array(zehavi_195) + np.array(zehavi_195_err), color=colors2[0], alpha=0.2)
    plt.plot(zehavi_bins, zehavi_195, '-', color=colors2.pop(0), lw=lw)  
    zehavi_20 = [366.1, 264.3, 184.0, 128.6, 84.7, 59.4, 42.9, 30.9, 21.9, 14.6, 8.24, 4.88, 3.58]
    zehavi_20_err = [9.3, 7.6, 6.6, 5.5, 4.3, 3.6, 3.3, 3.1, 2.7, 2.1, 1.32, 1.06, 0.85]
    #plt.plot(zehavi_bins, zehavi_20, '-', color=colors2.pop(0), lw=lw)
    zehavi_205 = np.array([455.7, 296.9, 197.0, 134.1, 89.4, 61.1, 44.0, 31.2, 21.3, 13.7, 7.65, 4.09, 3.21]) * 0.630
    zehavi_205_err = [11.3, 6.9, 5.1, 4.1, 3.3, 2.6, 2.3, 2.0, 1.8, 1.5, 1.07, 0.88, 0.70]
    plt.fill_between(zehavi_bins, np.array(zehavi_205) - np.array(zehavi_205_err), np.array(zehavi_205) + np.array(zehavi_205_err), color=colors2[0], alpha=0.2)
    plt.plot(zehavi_bins, zehavi_205, '-', color=colors2.pop(0), lw=lw)
    zehavi_21 = [586.2, 402.9, 258.7, 163.2, 105.5, 68.9, 50.2, 35.5, 24.5, 15.3, 8.54, 4.11, 2.73]
    zehavi_21_err = [19.5, 11.7, 6.7, 4.7, 3.0, 2.2, 2.1, 1.8, 1.6, 1.3, 0.94, 0.71, 0.54]
    plt.fill_between(zehavi_bins, np.array(zehavi_21) - np.array(zehavi_21_err), np.array(zehavi_21) + np.array(zehavi_21_err), color=colors2[0], alpha=0.2)
    plt.plot(zehavi_bins, zehavi_21, '-', color=colors2.pop(0), lw=lw)
    zehavi_215 = [1028, 731.7, 392.6, 228.6, 144.6, 94.3, 70.5, 48.6, 33.1, 20.9, 11.6, 6.04, 3.28]
    zehavi_215_err = [68, 34.0, 17.1, 10.9, 6.4, 3.7, 2.7, 2.3, 1.8, 1.5, 1.2, 0.95, 0.64]
    plt.fill_between(zehavi_bins, np.array(zehavi_215) - np.array(zehavi_215_err), np.array(zehavi_215) + np.array(zehavi_215_err), color=colors2[0], alpha=0.2)
    plt.plot(zehavi_bins, zehavi_215, '-', color=colors2.pop(0), lw=lw)
    zehavi_22 = [2615, 1189, 728, 491, 272, 154, 111, 94.5, 56.8, 35.1, 22.0, 11.4, 5.89]
    zehavi_22_err = [491, 202, 96.3, 55.3, 23.2, 14.5, 10.4, 5.6, 3.8, 3.2, 2.2, 1.6, 1.21] 
    plt.fill_between(zehavi_bins, np.array(zehavi_22) - np.array(zehavi_22_err), np.array(zehavi_22) + np.array(zehavi_22_err), color=colors2[0], alpha=0.2)
    plt.plot(zehavi_bins, zehavi_22, '-', color=colors2.pop(0), lw=lw)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.1, 30)
    #ax.set_ylim(0.1, 3000)
    ax.set_xlabel(r'$r_p$ [Mpc/h]')
    ax.set_ylabel(r'$w_p(r_p)$')

    # Legend 1: colors = magnitude thresholds
    color_handles = [ ]
    for i, thresh in enumerate(sorted_thresholds):
        color_handles.append(Line2D([0], [0], color=colors[i % len(colors)], lw=2, label=f'$M_r$ < -{thresh} ({sdss_magbins[i]})'))
    # Legend 2: symbols = BGS vs SDSS
    symbol_handles = [
        #Line2D([0], [0], marker='v', color='gray', lw=0, label='BGS 17.6 fluxlim'),
        Line2D([0], [0], marker='o', color='gray', lw=0, label='BGS 19.5 fluxlim'),
        Line2D([0], [0], color='gray', lw=2, label='SDSS Zehavi+11'),
    ]
    leg1 = ax.legend(handles=color_handles, fontsize=10, loc='lower left')
    ax.add_artist(leg1)  # keep first legend when adding second
    ax.legend(handles=symbol_handles, fontsize=11, loc='upper right')

    # Add text above the 1st legend saying BGS and SDSS
    plt.text(0.22, 0.42, 'BGS (SDSS)', transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left')

    plt.tight_layout()







def plot_wp_QSF_bins(loaded_results, weight_type):
    """
    Plots wp(rp) for a list of loaded clustering results for a specific weight type.

    Creates a figure for each unique magnitude range, with two subplots:
    one for Star-Forming (SF) samples and one for Quiescent (Q) samples.
    Different SERSIC cuts are shown in shades of blue and red respectively.

    Args:
        loaded_results (list): The list of dictionaries produced by
                               load_allcounts_from_disk.
        weight_type (str): The specific weight type to plot (e.g., 'WEIGHT_FKP_V1').
    """
    # Filter results by the specified weight type
    filtered_results = [res for res in loaded_results if res['params']['weights'] == weight_type]

    if not filtered_results:
        print(f"No results found for weight_type='{weight_type}'.")
        return

    print(f"Plotting results for weight_type='{weight_type}'")

    # Group results by magnitude range
    results_by_mag = {}
    for result in filtered_results:
        # Filter out 'ALL' sample types as requested
        if result['params']['sample_type'] == 'ALL':
            continue
        mag_range = result['params'].get('mag_range', 'all_magnitudes')
        if mag_range not in results_by_mag:
            results_by_mag[mag_range] = []
        results_by_mag[mag_range].append(result)

    # Create one plot for each magnitude range
    for mag_range, results in results_by_mag.items():
        fig, (ax_sf, ax_q) = plt.subplots(1, 2, figsize=(16, 8), sharey=True, sharex=True)
        fig.suptitle(f'Projected Correlation Function (Weights: {weight_type})\nMagnitude Range: {mag_range}', fontsize=18)

        # Separate results into SF and Q
        sf_results = sorted([r for r in results if r['params']['sample_type'] == 'SF'],
                            key=lambda x: x['params']['sersic'] or '')
        q_results = sorted([r for r in results if r['params']['sample_type'] == 'Q'],
                           key=lambda x: x['params']['sersic'] or '')

        # --- Plot SF (Blue) results ---
        if sf_results:
            # Create a colormap for different SERSIC values
            n_sersic_sf = len(sf_results)
            blue_shades = plt.cm.viridis(np.linspace(0.0, 1, n_sersic_sf))

            for i, item in enumerate(sf_results):
                params = item['params']
                estimator = item['data']
                color = blue_shades[i]

                if int(params['njack']) > 0:
                    rp, wp, cov = estimator.get_corr(return_sep=True, return_cov=True, mode='wp')
                    wp_err = np.sqrt(np.diag(cov))
                else:
                    rp, wp = estimator.get_corr(return_sep=True, mode='wp')
                    wp_err = None

                label = f"Sersic: {params['sersic'] or 'None'}"
                ax_sf.errorbar(rp, wp, yerr=wp_err, label=label, fmt='o', color=color, capsize=3, alpha=0.8)

        ax_sf.set_title('Star-Forming (SF)')
        ax_sf.set_xscale('log')
        ax_sf.set_yscale('log')
        ax_sf.set_xlabel(r'$r_p$ [Mpc/h]')
        ax_sf.set_ylabel(r'$w_p(r_p)$')
        ax_sf.grid(True, which="both", ls="--", alpha=0.5)
        ax_sf.legend()

        # --- Plot Q (Red) results ---
        if q_results:
            # Create a colormap for different SERSIC values
            n_sersic_q = len(q_results)
            red_shades = plt.cm.viridis(np.linspace(0.0, 1, n_sersic_q))

            for i, item in enumerate(q_results):
                params = item['params']
                estimator = item['data']
                color = red_shades[i]

                if int(params['njack']) > 0:
                    rp, wp, cov = estimator.get_corr(return_sep=True, return_cov=True, mode='wp')
                    wp_err = np.sqrt(np.diag(cov))
                else:
                    rp, wp = estimator.get_corr(return_sep=True, mode='wp')
                    wp_err = None

                label = f"Sersic: {params['sersic'] or 'None'}"
                ax_q.errorbar(rp, wp, yerr=wp_err, label=label, fmt='s', color=color, capsize=3, alpha=0.8)

        ax_q.set_title('Quiescent (Q)')
        ax_q.set_xscale('log')
        ax_q.set_xlabel(r'$r_p$ [Mpc/h]')
        ax_q.grid(True, which="both", ls="--", alpha=0.5)
        ax_q.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
        plt.show()







def plot_weight_comparison(loaded_results):
    """
    Compares wp(rp) for different weight types, holding all other parameters constant.

    Generates a plot for each combination of parameters (e.g., mag_range, sersic cut,
    sample type) that has been measured with more than one weight type.

    Args:
        loaded_results (list): The list of dictionaries produced by
                               load_allcounts_from_disk.
    """
    # Group results by all parameters except for 'weights'
    results_by_params = {}
    for res in loaded_results:
        p = res['params'].copy()
        # The weight type will be used for labeling, not for grouping
        p.pop('weights', None)
        # Create a stable key from the remaining parameters
        key = tuple(sorted(p.items()))

        if key not in results_by_params:
            results_by_params[key] = []
        results_by_params[key].append(res)

    print(f"Found {len(results_by_params)} unique parameter combinations.")
    plots_made = 0

    # Iterate through the grouped results and plot comparisons
    for param_key, results_list in results_by_params.items():
        # Only make a plot if there's more than one weight type to compare
        if len(results_list) > 1:
            plots_made += 1
            fig, ax = plt.subplots(figsize=(8, 6))

            # Sort by weight name for consistent plotting
            results_list.sort(key=lambda x: x['params']['weights'])

            for i, item in enumerate(results_list):
                params = item['params']
                estimator = item['data']
                weight_type = params['weights']

                if int(params['njack']) > 0:
                    rp, wp, cov = estimator.get_corr(return_sep=True, return_cov=True, mode='wp')
                    wp_err = np.sqrt(np.diag(cov))
                else:
                    rp, wp = estimator.get_corr(return_sep=True, mode='wp')
                    wp_err = None

                ax.errorbar(rp, wp, yerr=wp_err, label=weight_type, fmt='-o', capsize=3, alpha=0.8)

            # Create a descriptive title from the parameters
            param_dict = dict(param_key)
            title_parts = [
                f"Mag: {param_dict.get('mag_range', 'N/A')}",
                f"Sersic: {param_dict.get('sersic', 'N/A')}",
                f"Type: {param_dict.get('sample_type', 'N/A')}",
                f"z: {param_dict.get('zmin', '?')}-{param_dict.get('zmax', '?')}"
            ]
            ax.set_title("Weight Comparison: " + ", ".join(title_parts))
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$r_p$ [Mpc/h]')
            ax.set_ylabel(r'$w_p(r_p)$')
            ax.grid(True, which="both", ls="--", alpha=0.5)
            ax.legend(title="Weight Type")
            plt.tight_layout()
            plt.show()

    if plots_made == 0:
        print("\nNo parameter sets found with more than one weight type. No comparison plots generated.")
    else:
        print(f"\nGenerated {plots_made} comparison plots.")