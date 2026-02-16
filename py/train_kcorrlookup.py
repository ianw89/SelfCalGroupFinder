import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
import sys
from scipy.spatial import KDTree
import pickle
import emcee
import corner

if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from pyutils import *
from dataloc import *
from bgs_helpers import *
from groupcatalog import BGS_Z_MAX, BGS_Z_MIN

pd.set_option('display.max_columns', None)

# Load the merged table
table = Table.read(IAN_BGS_Y1_MERGED_FILE, format='fits')
df = table_to_df(table, 20.175, BGS_Z_MIN, BGS_Z_MAX, True, 1)
print(f"Loaded {len(df):,} galaxies")

# Drop all but the columns we need
df = df[['TARGETID', 'Z', 'ABS_MAG_R', 'ABS_MAG_G', 'ABSMAG01_SDSS_R', 'ABSMAG01_SDSS_G', 'DN4000_MODEL', 'G_R_BEST', 'LOG_L_GAL', 'LOGMSTAR']]


magr_k_gama = k_correct_gama(df['ABS_MAG_R'], df['Z'], df['ABS_MAG_G'] - df['ABS_MAG_R'], band='r')
magg_k_gama = k_correct_gama(df['ABS_MAG_G'], df['Z'], df['ABS_MAG_G'] - df['ABS_MAG_R'], band='g')
badmatch = (np.abs(magr_k_gama - df['ABSMAG01_SDSS_R']) > 1.0) | (np.abs(magg_k_gama - df['ABSMAG01_SDSS_G']) > 1.0)
goodidx = ~np.isnan(df['ABS_MAG_R']) & ~np.isnan(df['ABS_MAG_G']) & ~np.isnan(df['Z']) & ~np.isnan(df['ABSMAG01_SDSS_R']) & ~np.isnan(df['ABSMAG01_SDSS_G']) & ~badmatch
gmr = df.loc[goodidx, 'ABS_MAG_G'] - df.loc[goodidx, 'ABS_MAG_R']
print(f"Number of galaxies with good data: {np.sum(goodidx):,}")

# Randomly split goodidx into a training and test set
np.random.seed(6884)
shuffled_indices = np.random.permutation(np.where(goodidx)[0])
train_size = int(0.8 * len(shuffled_indices))
train_indices = shuffled_indices[:train_size]
test_indices = shuffled_indices[train_size:]

df.reset_index(drop=True, inplace=True)

# Prepare training and test data
z_train = df.loc[train_indices, 'Z'].to_numpy()
magr_train = df.loc[train_indices, 'ABS_MAG_R'].to_numpy()
magg_train = df.loc[train_indices, 'ABS_MAG_G'].to_numpy()
gmr_train = magg_train - magr_train

kcorr_r_train = magr_train - df.loc[train_indices, 'ABSMAG01_SDSS_R'].to_numpy()
kcorr_g_train = magg_train - df.loc[train_indices, 'ABSMAG01_SDSS_G'].to_numpy()

z_test = df.loc[test_indices, 'Z'].to_numpy()
magr_test = df.loc[test_indices, 'ABS_MAG_R'].to_numpy()
magg_test = df.loc[test_indices, 'ABS_MAG_G'].to_numpy()
gmr_test = magg_test - magr_test

kcorr_r_test = magr_test - df.loc[test_indices, 'ABSMAG01_SDSS_R'].to_numpy()
kcorr_g_test = magg_test - df.loc[test_indices, 'ABSMAG01_SDSS_G'].to_numpy()

# Define the log likelihood function for optimizing k-correction lookup metrics
def log_likelihood(params):
    """
    Evaluate how well a given set of metric parameters performs
    for k-correction lookup accuracy.
    
    params: [metric_z, metric_gmr]
    metric_absmag_r is fixed to 1.0
    """
    metric_z, metric_gmr = params
    
    # Rebuild KDTree with new metrics
    z_scaled = z_train * metric_z
    gmr_scaled = gmr_train * metric_gmr
    abs_mag_r_scaled = magr_train
    
    train_points = np.vstack((z_scaled, gmr_scaled, abs_mag_r_scaled)).T
    kdtree_test = KDTree(train_points)
    
    # Query nearest neighbors
    test_points = np.vstack((z_test * metric_z, 
                             gmr_test * metric_gmr,
                             magr_test)).T
    distances, indices = kdtree_test.query(test_points)
    
    # Get predicted k-corrections
    pred_kcorr_r = kcorr_r_train[indices]
    pred_kcorr_g = kcorr_g_train[indices]

    # Evaluate goodness of fit by comparing to the true k-corrections
    kcorr_r_errors = np.abs(pred_kcorr_r - kcorr_r_test)
    kcorr_g_errors = np.abs(pred_kcorr_g - kcorr_g_test)
    log_like = -(np.mean(kcorr_r_errors) + np.mean(kcorr_g_errors))
    
    # This is an alternative loss function that focuses on the final k-corrected magnitudes rather than the k-corrections themselves.
    # Calculate predicted k-corrected magnitudes
    #pred_abs_mag_r_kcorr = magr_test - pred_kcorr_r
    #pred_abs_mag_g_kcorr = magg_test - pred_kcorr_g
    #pred_gmr_kcorr = pred_abs_mag_g_kcorr - pred_abs_mag_r_kcorr
    
    # Calculate errors
    #abs_mag_r_errors = np.abs(pred_abs_mag_r_kcorr - true_abs_mag_r)
    #gmr_errors = np.abs(pred_gmr_kcorr - true_gmr)
    
    # Log likelihood: negative weighted mean absolute error
    # Weight abs_mag_r error more heavily than g-r error
    # log_like = -(3.0 * np.mean(abs_mag_r_errors) + 1.0 * np.mean(gmr_errors))
    
    return log_like

def log_prior(params):
    """Uniform prior on metric parameters"""
    metric_z, metric_gmr = params
    
    # Reasonable bounds for metrics
    if 0.3 < metric_z < 100.0 and 0.3 < metric_gmr < 50.0:
        return 0.0
    return -np.inf

def log_probability(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params)


# Initialize walkers
n_walkers = 10
n_dim = 2 

# Starting position: random numbers between 0.3 and 10
pos = np.random.uniform(low=[2.0, 1.0], high=[100.0, 30.0], size=(n_walkers, n_dim))

# Manually set some of the positions. Remember that the abs mag metric is fixed to 1.0, so these are relative to that.
pos[0] = [50.0, 10.0] 
pos[1] = [40.0, 8.0] 
pos[2] = [30.0, 6.0] 
pos[3] = [20.0, 4.0] 

# Set up sampler
backend = emcee.backends.HDFBackend("kcorr_lookup_optimization.h5")
sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, backend=backend)

# Run MCMC
print("Running MCMC...")
n_steps = 500
sampler.run_mcmc(pos, n_steps, progress=True)


# Get chain data
chain = sampler.get_chain()
log_prob = sampler.get_log_prob()


# Get results
samples = chain.reshape(1000,2)

# Print params that minimix the loss function
print("\nOptimal parameters:")
argmax = np.argmax(log_prob).flatten()
print(f"METRIC_Z: {samples[argmax, 0]}")
print(f"METRIC_GMR: {samples[argmax, 1]}")
print(f"METRIC_ABSMAG_R: 1.00 (fixed)")

# Plot corner plot
fig = corner.corner(samples, labels=["METRIC_Z", "METRIC_GMR"], truths=samples[argmax, :][0])
plt.show()

# Save figure of corner plot
fig.savefig("kcorr_lookup_optimization_corner.png")



# Now for the best fit metric, build the tree and lookup to save off (using the full data now)

# Get the optimal parameters from MCMC
optimal_metric_z = samples[argmax, 0]
optimal_metric_gmr = samples[argmax, 1]

print(f"Building final lookup table with optimal metrics:")
print(f"  METRIC_Z: {optimal_metric_z}")
print(f"  METRIC_GMR: {optimal_metric_gmr}")

# Prepare full dataset (all good galaxies)
z_full = df.loc[goodidx, 'Z'].to_numpy()
magr_full = df.loc[goodidx, 'ABS_MAG_R'].to_numpy()
magg_full = df.loc[goodidx, 'ABS_MAG_G'].to_numpy()
gmr_full = magg_full - magr_full

# Calculate k-corrections for the full dataset
kcorr_r_full = magr_full - df.loc[goodidx, 'ABSMAG01_SDSS_R'].to_numpy()
kcorr_g_full = magg_full - df.loc[goodidx, 'ABSMAG01_SDSS_G'].to_numpy()

# Scale the features with optimal metrics
z_scaled = z_full * optimal_metric_z
gmr_scaled = gmr_full * optimal_metric_gmr
magr_scaled = magr_full  # metric_absmag_r = 1.0

# Build the KDTree
lookup_points = np.vstack((z_scaled, gmr_scaled, magr_scaled)).T
kdtree = KDTree(lookup_points)

# Store the k-corrections as lookup tables
kcorr_r_lookup = kcorr_r_full
kcorr_g_lookup = kcorr_g_full

print(f"Built KDTree with {len(lookup_points):,} galaxies")
print(f"K-correction lookup table shape: {kcorr_r_lookup.shape}")

# Save the lookup table and optimal metrics
lookup_data = (kdtree, kcorr_r_lookup, kcorr_g_lookup, optimal_metric_z, optimal_metric_gmr, 1.0)

with open(BGS_Y3_KCORR_LOOKUP_FILE, 'wb') as f:
    pickle.dump(lookup_data, f)

print(f"\nSaved lookup table to {BGS_Y3_KCORR_LOOKUP_FILE}")



