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


def get_bias_for_mag(magnitude: float|np.ndarray, quiescent: bool|np.ndarray) -> float|np.ndarray:
    import pickle
    spline_dir = '/global/cfs/cdirs/desi/users/ianw89/clustering712/DA2/LSS/loa-v1/LSScats/v1.1/v0.3/'
    spline_file = os.path.join(spline_dir, 'luminosity_bias_splines.pkl')

    # Ensure the spline file exists
    if not os.path.exists(spline_file):
        raise FileNotFoundError(f"Spline file not found: {spline_file}")

    with open(spline_file, 'rb') as f:
        splines = pickle.load(f)
        spline_Q = splines['spline_Q'] # cubic spline
        spline_SF = splines['spline_SF']
        linear_Q = splines['linear_Q'] # linear interpolation
        linear_SF = splines['linear_SF']

    if quiescent:
        # -22.53977930844298 to -18.042940914856334
        if magnitude < -22.6 or magnitude > -17.9:
            raise ValueError(f"Requested magnitude {magnitude} is outside the range of the quiescent spline data (-22.5397 to -18.0429).")
        
        return linear_Q(magnitude)
    else:
        # -22.53977930844298 to -16.66915776095745
        if magnitude < -22.6 or magnitude > -16.5:
            raise ValueError(f"Requested magnitude {magnitude} is outside the range of the star-forming spline data (-22.5397 to -16.6691).")
        
        return linear_SF(magnitude)


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

    if not np.allclose(rp_ref, rp_target, rtol=0.01):
        print("WARNING: rp values of reference and target are not closely matched.")
        print("rp_ref:", rp_ref)
        print("rp_target:", rp_target)

    # Method 1: assume covariance matrices are independent and add them together
    def _chisqr_indepcov(bias):
        residual = wp_target - (bias**2 * wp_ref)
        C_tot = cov_target + cov_ref  
        reg = np.eye(C_tot.shape[0]) * 1e-12
        inv_cov = np.linalg.inv(C_tot + reg)

        # Check if c_tot @ inv_cov is close to identity
        identity_check = C_tot @ inv_cov
        if not np.allclose(identity_check, np.eye(C_tot.shape[0]), rtol=1e-5):
            print("WARNING: C_tot @ inv_cov is not close to identity.")
            print("C_tot @ inv_cov:\n", identity_check)

        return residual.T @ inv_cov @ residual

    # Method 2: assume the correlation matrix from the reference is correct for all subsamples
    # Use the diagonal elements from the subsample, but recompute the off-diagonal elements from the reference correlation matrix
    # I have now been convinced that this is not a valid method.
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
    # This is reasonable because the error bars on the reference are tiny anyway and the off-diagonal elements of the target are quite noisy.
    def _chisqr_diagonly(bias):
        residual = wp_target - (bias**2 * wp_ref)
        C_tot = np.diag(np.diag(cov_target)) + np.diag(np.diag(cov_ref))  # Only use diagonal elements
        inv_cov = np.linalg.inv(C_tot)
        return residual.T @ inv_cov @ residual

    # Method 4: Use target only, unmodified. 
    # This is also reasonable because reference error bars are tiny by comparison.
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

    return best_fit_bias, bias_err_up, bias_err_down, chi2_min
 

def get_bias_closedform(ref_wp, target_wp):
    rp_ref, wp_ref, cov_ref = ref_wp
    rp_target, wp_target, cov_target = target_wp

    if not np.allclose(rp_ref, rp_target, rtol=0.01):
        print("WARNING: rp values of reference and target are not closely matched.")
        print("rp_ref:", rp_ref)
        print("rp_target:", rp_target)

    C_tot = cov_target + cov_ref
    reg = np.eye(C_tot.shape[0]) * 1e-12
    inv_cov = np.linalg.inv(C_tot + reg)

    # For A=b^2, chi2 is a quadratic function of A, which allows an exact closed-form solution for the best-fit amplitude.
    # chi2(A) = a*A^2 - 2*b*A + c 
    a = wp_ref @ inv_cov @ wp_ref
    b = wp_ref @ inv_cov @ wp_target
    c = wp_target @ inv_cov @ wp_target

    A_hat = b / a # best-fit amplitude
    sigma_A = 1 / np.sqrt(a) # exact 1-sigma width in A (from curvature of the quadratic)
    chi2_min = c - b**2 / a

    best_fit_bias = np.sqrt(A_hat)

    # TODO double check this math
    # Delta chi2 = 1 in A-space is exact: A = A_hat +/- sigma_A. Transform to bias via sqrt.
    bias_up = np.sqrt(A_hat + sigma_A) - best_fit_bias
    if A_hat - sigma_A > 0:
        bias_down = best_fit_bias - np.sqrt(A_hat - sigma_A)
    else:
        bias_down = best_fit_bias   # amplitude consistent with 0; one-sided/undefined lower bound

    return best_fit_bias, bias_up, bias_down, chi2_min




def save_biases(savedir, results):
    os.makedirs(savedir, exist_ok=True)

    columns = [
        ('magbin_fainter', 'f8'),
        ('magbin_brighter', 'f8'),
        ('magbin_mean', 'f8'),
        ('quiescent', 'f8'),
        ('third_property_name', 'U64'),
        ('third_property_lower', 'f8'),
        ('third_property_upper', 'f8'),
        ('third_property_mean', 'f8'),
        ('b', 'f8'),
        ('b_err_low', 'f8'),
        ('b_err_high', 'f8'),
        ('b_p', 'f8'),
        ('b_p_err_low', 'f8'),
        ('b_p_err_high', 'f8'),
        ('b_m', 'f8'),
    ]
    table = np.empty(len(results), dtype=columns)
    for name, _ in columns:
        if name == 'third_property_name':
            table[name] = 'nan'
        else:
            table[name] = np.nan

    def parse_range(value):
        if value is None or 'to' not in str(value):
            return np.nan, np.nan
        try:
            lower, upper = (float(bound) for bound in str(value).split('to', 1))
        except ValueError:
            return np.nan, np.nan
        return lower, upper

    def numeric_value(result, key):
        value = result.get(key)
        if value is None:
            return np.nan
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    for index, result in enumerate(results):
        params = result.get('params', {})
        mag_bright, mag_faint = parse_range(params.get('mag_range'))
        table['magbin_fainter'][index] = mag_faint
        table['magbin_brighter'][index] = mag_bright
        table['magbin_mean'][index] = numeric_value(result, 'sample_mean_mag')

        sample_type = params.get('sample_type')
        if sample_type == 'Q':
            table['quiescent'][index] = 1
        elif sample_type == 'SF':
            table['quiescent'][index] = 0

        prop_name = params.get('third_property')
        if prop_name is not None:
            table['third_property_name'][index] = str(prop_name)
            prop_lower, prop_upper = parse_range(params.get('third_property_range'))
            table['third_property_lower'][index] = prop_lower
            table['third_property_upper'][index] = prop_upper
            table['third_property_mean'][index] = numeric_value(result, 'third_property_mean')

        for column, key in (
            ('b', 'bias'),
            ('b_err_low', 'bias_err_down'),
            ('b_err_high', 'bias_err_up'),
            ('b_p', 'b_p'),
            ('b_p_err_low', 'b_p_err_down'),
            ('b_p_err_high', 'b_p_err_up'),
            ('b_m', 'b_m'),
        ):
            table[column][index] = numeric_value(result, key)

    np.save(os.path.join(savedir, 'biases_BGS_DR2_2-10Mpc_v0.3.npy'), table)
    return table
