import re
import os
import numpy as np
from matplotlib import pyplot as plt
from pycorr import TwoPointEstimator


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

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(f'wp(rp) by Mag Threshold\n(Weights: {weight_type})', fontsize=16)

    for i, thresh in enumerate(sorted_thresholds):
        results = results_by_thresh[thresh]

        # There should typically be one result per threshold, but we loop just in case there are multiple (e.g., different rp bins, nran, zmax, etc)
        markers = ['v', '^', '.', 'o']  # Different markers for multiple entries with the same threshold
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
            ax.errorbar(rp, wp, yerr=wp_err, label=label, fmt='o', capsize=3, alpha=0.8, color=c)


    # SDSS Data
    from dataloc import PARAMS_SDSS_FOLDER
    for m in sdss_magbins:
        savedir = PARAMS_SDSS_FOLDER + f'sdss-thresh-{m:.1f}.csv'
        if not os.path.exists(savedir):
            print(f'File {savedir} not found')
            continue
        data = np.loadtxt(savedir, skiprows=1, dtype='float', delimiter=',')
        plt.plot(data[:,0], data[:,1], 'x', label=f'SDSS $M_r<-{m}$', color=colors.pop(0))
       # print(data[:,0])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$r_p$ [Mpc/h]')
    ax.set_ylabel(r'$w_p(r_p)$')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()









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