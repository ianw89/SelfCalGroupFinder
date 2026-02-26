"""
Train the Dn4000 lookup table by optimizing metric parameters using MCMC.

This script:
1. Loads BGS Y1 data with Dn4000 measurements
2. Uses MCMC (via emcee) to find optimal metric, k value, and inner metric for distances within the k neighbors
3. After running this use spectroscopic_properties_lookup.ipynb to build and save the final KDTree lookup table with optimal metrics and k value
"""
import numpy as np
import sys
from pykdtree.kdtree import KDTree # We use this faster but unserializable KDTree for the MCMC here.
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
    
    params: [metric_gmr, metric_k]
    metric_magr is fixed to 1.0
    """
    metric_gmr = params[0]
    k = int(params[1])
    inner_metric_dn4000 = params[2]
    
    # Rebuild KDTree with new metrics
    magr_scaled = magr_train  # metric_magr = 1.0
    gmr_scaled = gmr_train * metric_gmr
    
    train_points = np.vstack((magr_scaled, gmr_scaled)).T
    train_points = np.ascontiguousarray(train_points, dtype=np.float32)
    kdtree_test = KDTree(train_points)
    
    # Query nearest neighbors
    test_points = np.vstack((magr_test, gmr_test * metric_gmr)).T
    test_points = np.ascontiguousarray(test_points, dtype=np.float32)
    distances, indices = kdtree_test.query(test_points, k=k)
    
    # Get predicted values for all k neighbors
    # Shape: (n_test, k)
    neighbor_dn4000 = dn4000_train[indices]
    neighbor_logmstar = logmstar_train[indices]
    
    if k > 1:
        # Calculate mean values across the k neighbors
        # Shape: (n_test,)
        mean_dn4000 = np.mean(neighbor_dn4000, axis=1)
        mean_logmstar = np.mean(neighbor_logmstar, axis=1)
        
        # Calculate distance from each neighbor's values to the mean; this is a choice of metric here again
        # Shape: (n_test, k)
        dist_to_mean = np.sqrt(inner_metric_dn4000 * (neighbor_dn4000 - mean_dn4000[:, np.newaxis])**2 + 
                               (neighbor_logmstar - mean_logmstar[:, np.newaxis])**2)
        
        # Find index of closest neighbor to mean for each test point
        # Shape: (n_test,)
        closest_to_mean_idx = np.argmin(dist_to_mean, axis=1)
        
        # Get the values from the neighbor closest to mean
        # Use advanced indexing
        pred_dn4000 = neighbor_dn4000[np.arange(len(closest_to_mean_idx)), closest_to_mean_idx]
        pred_logmstar = neighbor_logmstar[np.arange(len(closest_to_mean_idx)), closest_to_mean_idx]
    else:
        # If k=1, just take the single nearest neighbor
        pred_dn4000 = neighbor_dn4000.flatten()
        pred_logmstar = neighbor_logmstar.flatten()
    
    # Calculate errors
    dn4000_errors = np.abs(pred_dn4000 - dn4000_test)
    logmstar_errors = np.abs(pred_logmstar - logmstar_test)
    
    # Log likelihood: root mean square of each
    # Weight Dn4000 more heavily as it's the primary target and a narrower range
    log_like = -(2.0 * np.sqrt(np.mean(dn4000_errors**2)) + np.sqrt(np.mean(logmstar_errors**2)))
    
    # Debug prints
    print(f"Testing: k={k}, metric_gmr={metric_gmr:.2f}")
    print(f"  Mean distance: {np.mean(distances):.4f}")
    print(f"  RMSE Dn4000: {np.sqrt(np.mean(dn4000_errors**2)):.4f}")   
    print(f"  RMSE logmstar: {np.sqrt(np.mean(logmstar_errors**2)):.4f}")
    print(f"  Log likelihood: {log_like:.4f}")

    return log_like

def log_prior(params):
    """Uniform prior on metric parameter"""
    metric_gmr = params[0]
    k = params[1]
    inner_metric_dn4000 = params[2]
    
    # Reasonable bounds for g-r metric
    if 0.3 < metric_gmr < 50.0 and 1 <= k <= 75 and 0.1 <= inner_metric_dn4000 <= 20.0:
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
    goodidx = (~np.isnan(df['ABSMAG01_SDSS_R']) & 
               ~np.isnan(df['ABSMAG01_SDSS_G']) & 
               ~np.isnan(df['DN4000_MODEL']) &
               ~np.isnan(df['LOGMSTAR']))
    
    print(f"Number of galaxies with good data: {np.sum(goodidx):,}")
    
    # Extract data
    magr = df.loc[goodidx, 'ABSMAG01_SDSS_R'].to_numpy()
    magg = df.loc[goodidx, 'ABSMAG01_SDSS_G'].to_numpy()
    gmr = magg - magr
    dn4000 = df.loc[goodidx, 'DN4000_MODEL'].to_numpy()
    logmstar = df.loc[goodidx, 'LOGMSTAR'].to_numpy()
    
    # Split into training and test sets
    np.random.seed(83592)
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
    n_dim = 3
    
    # Starting positions for walkers
    pos = np.random.uniform(low=[0.5, 1, 0.5], high=[10.0, 50, 10.0], size=(n_walkers, n_dim))
    pos[0] = [4.0, 30, 3]
    pos[1] = [5.0, 20, 4]
    pos[2] = [6.0, 40, 5]
    pos[3] = [4.5, 15, 2]
    pos[4] = [5.5, 25, 1]
    
    # Create backend
    backend_file = OUTPUT_FOLDER + "dn4000_lookup_optimization.h5"
    backend = emcee.backends.HDFBackend(backend_file)
    backend.reset(n_walkers, n_dim)
    
    # Set up sampler with training data as args
    args = (magr_train, gmr_train, dn4000_train, logmstar_train,
            magr_test, gmr_test, dn4000_test, logmstar_test)
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, 
                                     args=args, backend=backend)
    
    # Run MCMC
    print("Running MCMC...")
    n_steps = 300
    sampler.run_mcmc(pos, n_steps, progress=True)

    
if __name__ == "__main__":
    main()