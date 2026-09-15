import numpy as np
import os
from scipy.optimize import minimize
from matplotlib import pyplot as plt

#######################################################################################
# Helper functions for interacting with clusting measurements once outside of the DESI/pycorr ecosystem.
#######################################################################################


def save_wp_dr2format(path, dat):
    # Save a tuple of (rp, wp, cov), each of which are numpy arrays, to a text file
    np.savetxt(path, np.column_stack(dat), header='rp wp cov', comments='')

def load_wp_dr2format(path):
    # Load a tuple of (rp, wp, cov), each of which are numpy arrays, from a text file
    data = np.loadtxt(path, skiprows=1)
    rp = data[:, 0]
    wp = data[:, 1]
    cov = data[:, 2:]
    return rp, wp, cov

def save_wp_dr1format(savedir, red_results, blue_results, all_results, magbins):
    
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


def get_bias(ref_wp, target_wp):
    """
    Similar to above, compares two wp(rp) measurements by calculating the bias of one with respect to the other.

    Bias is defined b^2 = target / reference, i.e., the ratio of the target wp to the reference wp.

    1. Ensure that the rp values are very close to each other, bin by bin (warn if not).
    2. Minimize a function to find the bias.

    Use the provided covariance matrices to compute the weighted average of the bias.

    Args:
        ref_wp (tuple): A tuple containing (rp_ref, wp_ref, wp_ref_cov)
        target_wp (tuple): A tuple containing (rp_target, wp_target, wp_target_cov) for the target measurement.

    Returns:
        tuple float: The best-fit bias value that minimizes the chi-squared function and an upper and lower uncertainty.
    """
    
    rp_ref, wp_ref, cov_ref = ref_wp
    rp_target, wp_target, cov_target = target_wp

    if not np.allclose(rp_ref, rp_target, rtol=0.1):
        print("WARNING: rp values of reference and target are not closely matched.")
        print("rp_ref:", rp_ref)
        print("rp_target:", rp_target)

    # Method 1: assume covariance matrices are independent and add them together
    def _chisqr_indepcov(bias):
        residual = wp_target - (bias**2 * wp_ref)
        C_tot = cov_target + cov_ref  
        reg = np.eye(C_tot.shape[0]) * 1e-12
        inv_cov = np.linalg.inv(C_tot + reg)
        return residual.T @ inv_cov @ residual

    # Method 2: assume the correlation matrix from the reference is correct for all subsamples
    # Use the diagonal elements from the subsample, but recompute the off-diagonal elements from the reference correlation matrix
    cov_target_modified = np.diag(np.diag(cov_target))  # Keep only the diagonal elements
    corr_ref = cov_ref / np.outer(np.sqrt(np.diag(cov_ref)), np.sqrt(np.diag(cov_ref)))  # Compute the correlation matrix from the reference
    cov_target_modified = np.outer(np.sqrt(np.diag(cov_target)), np.sqrt(np.diag(cov_target))) * corr_ref  # Reconstruct the covariance matrix using the reference correlation matrix
    #with np.printoptions(precision=3, suppress=True):
    #    print("Original target covariance matrix:\n", cov_target)
    #    print("Modified target covariance matrix using reference correlation matrix:\n", cov_target_modified)
    def _chisqr_use_ref_corr(bias):
        residual = wp_target - (bias**2 * wp_ref)
        C_tot = cov_target_modified + cov_ref  # Could instead compute this once outside for an assumed bias ~ 1 
        reg = np.eye(C_tot.shape[0]) * 1e-10
        inv_cov = np.linalg.inv(C_tot + reg)
        return residual.T @ inv_cov @ residual

    # Method 3: Use only the diagonal elements of the covariance matrices (i.e., ignore correlations)
    def _chisqr_diagonly(bias):
        residual = wp_target - (bias**2 * wp_ref)
        C_tot = np.diag(np.diag(cov_target)) + np.diag(np.diag(cov_ref))  # Only use diagonal elements
        inv_cov = np.linalg.inv(C_tot)
        return residual.T @ inv_cov @ residual

    # Method 4: Use target only, unmodified. Reference error bars are small anyway.
    def _chisqr_targetonly(bias):
        residual = wp_target - (bias**2 * wp_ref)
        C_tot = cov_target  # Only use target covariance
        reg = np.eye(C_tot.shape[0]) * 1e-10
        inv_cov = np.linalg.inv(C_tot + reg)
        return residual.T @ inv_cov @ residual

    _chisqr = _chisqr_indepcov  # Choose which method to use

    # Minimize the chi-squared function to find the best-fit bias
    result = minimize(_chisqr, x0=1.0, bounds=[(0.1, 10.0)])
    best_fit_bias = result.x[0]

    # Estimate the uncertainty on the best-fit bias by measuring the chisqr around the best-fit value
    # until we find where delta chi sqr is equal to 1 (allow asymmetric)
    delta_chi2 = 1.0
    step = 1e-4
    chi2_min = _chisqr(best_fit_bias)

    # Find upper error
    bias_up = best_fit_bias
    while _chisqr(bias_up) - chi2_min < delta_chi2:
        bias_up += step
    bias_err_up = bias_up - best_fit_bias

    # Find lower error
    bias_down = best_fit_bias
    while _chisqr(bias_down) - chi2_min < delta_chi2:
        bias_down -= step
    bias_err_down = best_fit_bias - bias_down

    return best_fit_bias, bias_err_up, bias_err_down
 




def bias_plots(results):
    """
    Results should be an iterable that has a params dictionary with keys 'mag_range', 'third_property', 'third_property_range', and 'sample_type' (which should be either 'Q' or 'SF').
    """

    # Group results by 3rd property name
    results_by_prop3 = {}
    for result in results:
        mag_range = result['params'].get('mag_range')
        prop3_name = result['params'].get('third_property')
        prop3_range = result['params'].get('third_property_range')

        if mag_range is None or prop3_name is None or prop3_range is None:
            continue

        if prop3_name not in results_by_prop3:
            results_by_prop3[prop3_name] = {}
        if mag_range not in results_by_prop3[prop3_name]:
            results_by_prop3[prop3_name][mag_range] = []

        results_by_prop3[prop3_name][mag_range].append(result)

    # Now go into each property, and order the magnitude ranges by the first number in the range
    for prop3_name in results_by_prop3:
        results_by_prop3[prop3_name] = dict(sorted(results_by_prop3[prop3_name].items(), key=lambda x: float(x[0].split('to')[0])))

        # For each third_property, make one of these 10 panel plots
    for third_property, mag_dict in results_by_prop3.items():
        fig, axes = plt.subplots(2, 5, figsize=(14, 6.5), sharex=True, sharey=True)
        axes = axes.flatten()
        i = 0
        for mag_range, measurements in mag_dict.items():
            ax = axes[i]
            i += 1

            x_q_values = []
            y_q_values = []
            y_q_err_up = []
            y_q_err_down = []
            x_sf_values = []
            y_sf_values = []
            y_sf_err_up = []
            y_sf_err_down = []

            for measurement in measurements:
                quiescent = measurement['params'].get('sample_type') == 'Q'
                prop_range = measurement['params'].get('third_property_range')
                prop_mean = get_bin_means_for(mag_range, quiescent, third_property, prop_range)
                #if prop_mean is None:
                    #print(f"Missing bin means for mag_range {mag_range}, quiescent {quiescent}, third_property {third_property}, prop_range {prop_range}")
                    #continue

                #x_value = np.mean([float(x) for x in measurement['params'].get('third_property_range', '0to0').split('to')])
                x_value = prop_mean
                y_value = measurement['bias']
                y_err_up_value = measurement['bias_err_up']
                y_err_down_value = measurement['bias_err_down']

                if quiescent:
                    x_q_values.append(x_value)
                    y_q_values.append(y_value)
                    y_q_err_up.append(y_err_up_value)
                    y_q_err_down.append(y_err_down_value)
                else:
                    x_sf_values.append(x_value)
                    y_sf_values.append(y_value)
                    y_sf_err_up.append(y_err_up_value)
                    y_sf_err_down.append(y_err_down_value)

            ax.errorbar(x_sf_values, y_sf_values, yerr=[y_sf_err_down, y_sf_err_up], fmt='o', capsize=5, markersize=6, markerfacecolor='b', markeredgewidth=1.5, markeredgecolor='k', label='Star-forming', ecolor='k', elinewidth=2)
            ax.errorbar(x_q_values, y_q_values, yerr=[y_q_err_down, y_q_err_up], fmt='o', capsize=5, markersize=6, markerfacecolor='r', markeredgewidth=1.5, markeredgecolor='k', label='Quiescent', ecolor='k', elinewidth=2)

            ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
            if i % 5 == 1:
                ax.set_ylabel('Bias $b$')
            ax.set_title(f'{mag_range.replace("to", " to ")}')

        plt.tight_layout()
        plt.suptitle(f'{third_property} Bias', fontsize=24)
        plt.subplots_adjust(top=0.9)  # Adjust top to make room for suptitle
        plt.show()    
