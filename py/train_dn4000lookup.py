"""
Train the Dn4000 lookup table by optimizing metric parameters using MCMC.

This script:
1. Loads BGS Y1 data with Dn4000 measurements
2. Uses MCMC (via emcee) to find optimal metric scaling for (abs_mag_r, g-r) space
3. Builds and saves the final KDTree lookup table with optimal metrics
"""

import numpy as np
import sys
from scipy.spatial import KDTree
import pickle
import emcee

if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from pyutils import *
from dataloc import *
from bgs_helpers import *
from groupcatalog import BGS_Z_MAX, BGS_Z_MIN

def log_likelihood(params, magr_train, gmr_train, dn4000_train, logmstar_train,
                   magr_test, gmr_test, dn4000_test, logmstar_test):
    """
    Evaluate how well a given metric parameter performs for Dn4000 lookup accuracy.
    
    params: [metric_gmr]
    metric_magr is fixed to 1.0
    """
    metric_gmr = params[0]
    
    # Rebuild KDTree with new metrics
    magr_scaled = magr_train  # metric_magr = 1.0
    gmr_scaled = gmr_train * metric_gmr
    
    train_points = np.vstack((magr_scaled, gmr_scaled)).T
    kdtree_test = KDTree(train_points)
    
    # Query nearest neighbors
    test_points = np.vstack((magr_test, gmr_test * metric_gmr)).T
    distances, indices = kdtree_test.query(test_points)
    
    # Get predicted values
    pred_dn4000 = dn4000_train[indices]
    pred_logmstar = logmstar_train[indices]
    
    # Calculate errors
    dn4000_errors = np.abs(pred_dn4000 - dn4000_test)
    logmstar_errors = np.abs(pred_logmstar - logmstar_test)
    
    # Log likelihood: negative mean absolute error
    # Weight Dn4000 more heavily as it's the primary target
    log_like = -(2.0 * np.mean(dn4000_errors) + 1.0 * np.mean(logmstar_errors))
    
    return log_like

def log_prior(params):
    """Uniform prior on metric parameter"""
    metric_gmr = params[0]
    
    # Reasonable bounds for g-r metric
    if 0.3 < metric_gmr < 50.0:
        return 0.0
    return -np.inf

def log_probability(params, *args):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, *args)

def main():
    print("="*60)
    print("Training Dn4000 Lookup Table")
    print("="*60)
    
    # Load BGS Y1 data
    print("\nLoading BGS Y1 data...")
    table = Table.read(IAN_BGS_Y1_MERGED_FILE, format='fits')
    df = table_to_df(table, 20.175, BGS_Z_MIN, BGS_Z_MAX, True, 1)
    print(f"Loaded {len(df):,} galaxies")
    
    # Filter to galaxies with all required data
    goodidx = (~np.isnan(df['ABS_MAG_R']) & 
               ~np.isnan(df['ABS_MAG_G']) & 
               ~np.isnan(df['DN4000_MODEL']) &
               ~np.isnan(df['LOGMSTAR']))
    
    print(f"Number of galaxies with good data: {np.sum(goodidx):,}")
    
    # Extract data
    magr = df.loc[goodidx, 'ABS_MAG_R'].to_numpy()
    magg = df.loc[goodidx, 'ABS_MAG_G'].to_numpy()
    gmr = magg - magr
    dn4000 = df.loc[goodidx, 'DN4000_MODEL'].to_numpy()
    logmstar = df.loc[goodidx, 'LOGMSTAR'].to_numpy()
    
    # Split into training and test sets
    np.random.seed(9853)
    shuffled_indices = np.random.permutation(len(magr))
    train_size = int(0.8 * len(shuffled_indices))
    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]
    
    magr_train = magr[train_indices]
    gmr_train = gmr[train_indices]
    dn4000_train = dn4000[train_indices]
    logmstar_train = logmstar[train_indices]
    
    magr_test = magr[test_indices]
    gmr_test = gmr[test_indices]
    dn4000_test = dn4000[test_indices]
    logmstar_test = logmstar[test_indices]
    
    print(f"Training set: {len(train_indices):,} galaxies")
    print(f"Test set: {len(test_indices):,} galaxies")
    
    # Set up MCMC
    print("\nSetting up MCMC...")
    n_walkers = 8
    n_dim = 1
    
    # Starting positions for walkers
    pos = np.random.uniform(low=0.5, high=20.0, size=(n_walkers, n_dim))
    pos[0] = [4.0]
    pos[1] = [5.0]
    pos[2] = [6.0]
    pos[3] = [4.5]
    pos[4] = [5.5]
    
    # Create backend
    backend_file = "dn4000_lookup_optimization.h5"
    backend = emcee.backends.HDFBackend(backend_file)
    backend.reset(n_walkers, n_dim)
    
    # Set up sampler with training data as args
    args = (magr_train, gmr_train, dn4000_train, logmstar_train,
            magr_test, gmr_test, dn4000_test, logmstar_test)
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, 
                                     args=args, backend=backend)
    
    # Run MCMC
    print("Running MCMC...")
    n_steps = 500
    sampler.run_mcmc(pos, n_steps, progress=True)
    
    # Get results
    print("\nAnalyzing results...")
    samples = sampler.get_chain(discard=50, flat=True)
    log_prob = sampler.get_log_prob(discard=50, flat=True)
    
    # Find optimal parameters
    best_idx = np.argmax(log_prob)
    optimal_metric_gmr = samples[best_idx, 0]
    
    print("\nOptimal parameters:")
    print(f"  METRIC_GMR: {optimal_metric_gmr:.3f}")
    print(f"  METRIC_MAGR: 1.00 (fixed)")
    print(f"  Best log likelihood: {log_prob[best_idx]:.4f}")
    
    # Build final lookup table with optimal metrics
    print(f"\nBuilding final lookup table...")
    
    # Use all good galaxies for final table
    magr_scaled = magr  # metric_magr = 1.0
    gmr_scaled = gmr * optimal_metric_gmr
    
    lookup_points = np.vstack((magr_scaled, gmr_scaled)).T
    kdtree = KDTree(lookup_points)
    
    print(f"Built KDTree with {len(lookup_points):,} galaxies")
    
    # Save lookup table
    lookup_data = (kdtree, dn4000, logmstar, optimal_metric_gmr, 1.0)
    
    with open(BGS_Y3_DN4000_LOOKUP_FILE, 'wb') as f:
        pickle.dump(lookup_data, f)
    
    print(f"\nSaved lookup table to {BGS_Y3_DN4000_LOOKUP_FILE}")
    
    # Validate on test set
    print("\nValidating on test set...")
    test_points = np.vstack((magr_test, gmr_test * optimal_metric_gmr)).T
    distances, indices = kdtree.query(test_points)
    
    pred_dn4000 = dn4000[indices]
    pred_logmstar = logmstar[indices]
    
    dn4000_error = pred_dn4000 - dn4000_test
    logmstar_error = pred_logmstar - logmstar_test
    
    print(f"  Dn4000 MAE: {np.mean(np.abs(dn4000_error)):.4f}")
    print(f"  Dn4000 RMS: {np.sqrt(np.mean(dn4000_error**2)):.4f}")
    print(f"  LogMstar MAE: {np.mean(np.abs(logmstar_error)):.4f}")
    print(f"  LogMstar RMS: {np.sqrt(np.mean(logmstar_error**2)):.4f}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

if __name__ == "__main__":
    main()