import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
import sys
from pykdtree.kdtree import KDTree
import emcee
import corner

if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from pyutils import *
from dataloc import *
from bgs_helpers import *
from groupcatalog import BGS_Z_MAX, BGS_Z_MIN

#####################################################
# Lesson: The metric does not matter almost at all. So long as the numerical range of the property values is roughly comparable,
# the KDTree lookup finds the same neighbors. 
# The k number of neighbors DOES matter. We get rapid gains from k=1 to k=10 or so, and very slow gains to k=100. 
# It's close enough to converged that I will not botehr looking beyond that.
# The mean distance is still close in property space at k=100 so it's defintely not TOO big. No turnover in performance yet.

# The feature choice probably matters more than the metric choice.
##################################################


pd.set_option('display.max_columns', None)

# Load the merged table
table = Table.read(IAN_BGS_Y1_MERGED_FILE, format='fits')
df = table_to_df(table, 20.175, BGS_Z_MIN, BGS_Z_MAX, True, 1)
print(f"Loaded {len(df):,} galaxies")

# Drop all but the columns we need
df = df[['TARGETID', 'Z', 'ABS_MAG_R', 'ABS_MAG_G', 'ABSMAG01_SDSS_R', 'ABSMAG01_SDSS_G', 'DN4000_MODEL', 'G_R_BEST', 'LOG_L_GAL', 'LOGMSTAR']]
df.reset_index(drop=True, inplace=True)

magr_k_gama = k_correct_gama(df['ABS_MAG_R'], df['Z'], df['ABS_MAG_G'] - df['ABS_MAG_R'], band='r')
magg_k_gama = k_correct_gama(df['ABS_MAG_G'], df['Z'], df['ABS_MAG_G'] - df['ABS_MAG_R'], band='g')
badmatch = (np.abs(magr_k_gama - df['ABSMAG01_SDSS_R']) > 1.0) | (np.abs(magg_k_gama - df['ABSMAG01_SDSS_G']) > 1.0)
goodidx = ~np.isnan(df['ABS_MAG_R']) & ~np.isnan(df['ABS_MAG_G']) & ~np.isnan(df['Z']) & ~np.isnan(df['ABSMAG01_SDSS_R']) & ~np.isnan(df['ABSMAG01_SDSS_G']) & ~badmatch
print(f"Number of galaxies with good data: {np.sum(goodidx):,}")

# Randomly split goodidx into a training and test set
np.random.seed(6884)
shuffled_indices = np.random.permutation(np.where(goodidx)[0])
train_size = int(0.8 * len(shuffled_indices))
train_indices = shuffled_indices[:train_size]
test_indices = shuffled_indices[train_size:]

# Prepare training and test data
z_train = df.loc[train_indices, 'Z'].to_numpy()
magr_train = df.loc[train_indices, 'ABS_MAG_R'].to_numpy() # These have no k-corr
magg_train = df.loc[train_indices, 'ABS_MAG_G'].to_numpy() # These have no k-corr
gmr_train = magg_train - magr_train

kcorr_r_train = magr_train - df.loc[train_indices, 'ABSMAG01_SDSS_R'].to_numpy() # These have the fastspecfit k-corr in them
kcorr_g_train = magg_train - df.loc[train_indices, 'ABSMAG01_SDSS_G'].to_numpy() # These have the fastspecfit k-corr in them

z_test = df.loc[test_indices, 'Z'].to_numpy()
magr_test = df.loc[test_indices, 'ABS_MAG_R'].to_numpy()
magg_test = df.loc[test_indices, 'ABS_MAG_G'].to_numpy()
gmr_test = magg_test - magr_test

kcorr_r_test = magr_test - df.loc[test_indices, 'ABSMAG01_SDSS_R'].to_numpy() # These have the fastspecfit k-corr in them
kcorr_g_test = magg_test - df.loc[test_indices, 'ABSMAG01_SDSS_G'].to_numpy() # These have the fastspecfit k-corr in them

# True k-corrected values
true_abs_mag_r = df.loc[test_indices, 'ABSMAG01_SDSS_R'].to_numpy()
true_abs_mag_g = df.loc[test_indices, 'ABSMAG01_SDSS_G'].to_numpy()    
true_gmr_kcorr = true_abs_mag_g - true_abs_mag_r

# Define the log likelihood function for optimizing k-correction lookup metrics
def log_likelihood(params):
    """
    Evaluate how well a given set of metric parameters performs
    for k-correction lookup accuracy.
    
    params: [metric_z, metric_gmr]
    metric_absmag_r is fixed to 1.0
    """
    if isinstance(params, int) or isinstance(params, float):
        metric_z = 20.0
        metric_gmr = 4.0
        metric_magr = 1.0
        k = int(params)
    else:
        assert len(params) == 2, "Expected 2 parameters: [metric_z, metric_gmr]"
        k = 100
        metric_z = params[0]
        metric_gmr = params[1]
        metric_magr = 1.0
    
    # Rebuild KDTree with new metrics
    z_scaled = z_train * metric_z
    gmr_scaled = gmr_train * metric_gmr
    abs_mag_r_scaled = magr_train * metric_magr
    
    train_points = np.vstack((z_scaled, gmr_scaled, abs_mag_r_scaled)).T
    train_points = np.ascontiguousarray(train_points, dtype=np.float32)
    kdtree_test = KDTree(train_points)
    
    # Query nearest neighbors
    test_points = np.vstack((z_test * metric_z, 
                             gmr_test * metric_gmr,
                             magr_test)).T
    test_points = np.ascontiguousarray(test_points, dtype=np.float32)
    distances, indices = kdtree_test.query(test_points, k=k)

    # indices shape: (n_test, 11), distances shape: (n_test, 11)
    
    # Get k-corrections for all 11 neighbors
    # Shape: (n_test, 11)
    neighbor_kcorr_r = kcorr_r_train[indices]
    neighbor_kcorr_g = kcorr_g_train[indices]

    if k > 1:
        
        # Calculate mean k-corrections across the 11 neighbors
        # Shape: (n_test,)
        mean_kcorr_r = np.mean(neighbor_kcorr_r, axis=1)
        mean_kcorr_g = np.mean(neighbor_kcorr_g, axis=1)
        
        # Find which neighbor is closest to the mean
        # Calculate distance from each neighbor's k-corr to the mean
        # Shape: (n_test, 11)
        dist_to_mean = np.sqrt((neighbor_kcorr_r - mean_kcorr_r[:, np.newaxis])**2 + 
                            (neighbor_kcorr_g - mean_kcorr_g[:, np.newaxis])**2)
        
        # Find index of closest neighbor to mean for each test point
        # Shape: (n_test,)
        closest_to_mean_idx = np.argmin(dist_to_mean, axis=1)

        # Get the k-corrections from the neighbor closest to mean
        # Use advanced indexing: for each test point i, get neighbor_kcorr_r[i, closest_to_mean_idx[i]]
        pred_kcorr_r = neighbor_kcorr_r[np.arange(len(closest_to_mean_idx)), closest_to_mean_idx]
        pred_kcorr_g = neighbor_kcorr_g[np.arange(len(closest_to_mean_idx)), closest_to_mean_idx]

    else:
        # If k=1, just take the single nearest neighbor's k-correction
        pred_kcorr_r = neighbor_kcorr_r
        pred_kcorr_g = neighbor_kcorr_g
    
    # CORRECT: Apply predicted k-corrections to TEST magnitudes
    pred_abs_mag_r_kcorr = magr_test - pred_kcorr_r
    pred_abs_mag_g_kcorr = magg_test - pred_kcorr_g
    pred_gmr_kcorr = pred_abs_mag_g_kcorr - pred_abs_mag_r_kcorr
    
    # Compare against TRUE k-corrected magnitudes from TEST set
    abs_mag_r_errors = np.abs(pred_abs_mag_r_kcorr - true_abs_mag_r)
    gmr_errors = np.abs(pred_gmr_kcorr - true_gmr_kcorr)
    
    # Weight absolute magnitude more heavily
    log_like = -(2.0 * np.sqrt(np.mean(abs_mag_r_errors**2)) + 
                 1.0 * np.sqrt(np.mean(gmr_errors**2)))
    
    # Debug prints
    print(f"Testing: metric_z={metric_z:.2f}, metric_gmr={metric_gmr:.2f}")
    print(f"  Mean distance: {np.mean(distances):.4f}")
    print(f"  RMSE abs_mag_r: {np.sqrt(np.mean(abs_mag_r_errors**2)):.4f}")
    print(f"  RMSE gmr: {np.sqrt(np.mean(gmr_errors**2)):.4f}")
    print(f"  Log likelihood: {log_like:.4f}")

    return log_like

def log_prior(params):
    """Uniform prior on metric parameters"""
    metric_z = params[0]
    metric_gmr = params[1]
    
    # Reasonable bounds for z and g-r metrics
    if 0.1 < metric_z < 200.0 and 0.1 < metric_gmr < 200.0:
        return 0.0
    return -np.inf

def log_probability(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params)


def optimize_metric():
        
    # Initialize walkers
    n_walkers = 10
    n_dim = 2

    # Set up sampler
    backend_file = "kcorr_lookup_optimization.h5"
    backend = emcee.backends.HDFBackend(backend_file)

    # Check if we should continue an existing run
    try:
        # Try to get the last position from the backend
        last_sample = backend.get_last_sample()
        print(f"Found existing MCMC run with {backend.iteration} steps")
        print("Continuing from last position...")
        pos = last_sample.coords
        n_steps = 25  # Additional steps to run
    except:
        # No existing run found, start fresh
        print("Starting new MCMC run...")
        backend.reset(n_walkers, n_dim)
        
        # Starting position: random numbers between 0.3 and 10
        pos = np.random.uniform(low=[2.0, 1.0], high=[100.0, 30.0], size=(n_walkers, n_dim))
        
        # Manually set some of the positions
        pos[0] = [50.0, 10.0] 
        pos[1] = [40.0, 8.0] 
        pos[2] = [30.0, 6.0] 
        pos[3] = [20.0, 4.0]
        
        n_steps = 25

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, backend=backend)

    # Run MCMC
    print("Running MCMC...")
    sampler.run_mcmc(pos, n_steps, progress=True)

    # Get chain data
    samples = sampler.get_chain(flat=True)
    log_prob = sampler.get_log_prob(flat=True)

    # Print params that minimix the loss function
    print("\nOptimal parameters:")
    argmax = np.argmax(log_prob)
    print(f"METRIC_Z: {samples[argmax, 0]}")
    print(f"METRIC_GMR: {samples[argmax, 1]}")
    print(f"METRIC_ABSMAG_R: 1.00 (fixed)")

    # Plot corner plot
    fig = corner.corner(samples, labels=["METRIC_Z", "METRIC_GMR"], truths=[samples[argmax, 0], samples[argmax, 1]])
    plt.show()

    # Save figure of corner plot
    fig.savefig("kcorr_lookup_optimization_corner.png")


def test_k_values():
    """
    Brute force test different values of k (number of neighbors) to find optimal value.
    Tests k from 1 to 50 and reports which gives best performance.
    """
    print("\n" + "="*60)
    print("BRUTE FORCE K OPTIMIZATION")
    print("="*60)
    
    # Test k values from 1 to 50
    k_values = range(51, 101, 5)
    results = []
    
    for k in k_values:
        # Use the existing log_likelihood function
        log_like = log_likelihood(k)
        loss = -log_like
        
        # Extract metrics from the print output or recalculate
        # For now, just store the log likelihood
        results.append({
            'k': k,
            'log_like': log_like
        })
        
        print(f"k={k:2d}: log_likelihood={log_like:.6f}")
    
    # Find best k
    results_df = pd.DataFrame(results)
    best_idx = results_df['log_like'].idxmax()
    best_k = results_df.loc[best_idx, 'k']
    best_log_like = results_df.loc[best_idx, 'log_like']
    
    print("\n" + "="*60)
    print(f"BEST K VALUE: {best_k}")
    print(f"Best log likelihood: {best_log_like:.6f}")
    print("="*60)
    
    # Plot results
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(results_df['k'], results_df['log_like'], 'o-')
    ax.axvline(best_k, color='r', linestyle='--', label=f'Best k={best_k}')
    ax.set_xlabel('k (number of neighbors)')
    ax.set_ylabel('Log Likelihood')
    ax.set_title('Log Likelihood vs k')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('OUTPUT/kcorr_k_optimization.png', dpi=150)
    print(f"\nSaved plot to OUTPUT/kcorr_k_optimization.png")


if __name__ == "__main__":
    optimize_metric()
    #test_k_values()