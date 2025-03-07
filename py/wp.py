import subprocess
import os
import sys
import pandas as pd
import numpy as np
from Corrfunc.utils import convert_rp_pi_counts_to_wp
from Corrfunc.mocks import DDrppi_mocks
import astropy.constants as const
import random
from astropy.cosmology import Planck13

if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from dataloc import *

# Constants
NTHREADS = 16
COSMOLOGY = 2  # 1->LasDamas, 2->Planck
RMIN = 0.1
RMAX = 40.0
NBINS = 10
PIMAX = 40.0  # TODO tune   
RBINS = np.logspace(np.log10(RMIN), np.log10(RMAX), NBINS + 1)



def calculate_wp_from_df(df: pd.DataFrame, randoms, data_weights=None, rand_weights=None):
    """
    Calculate the projected correlation function wp(rp) for the total, red, and blue
    galaxies for the provided dataframe and randoms using our canonical set of parameters.
    """
    #cz = df.z.to_numpy().copy() * const.c.to('km/s').value # corrfunc has a bug with cz
    #print(df.z.to_numpy())
    dist = Planck13.comoving_distance(df.z.to_numpy()) # Mpc
    
    # Overall sample
    rbins , wp_all = calculate_wp(
        df.RA.to_numpy(), 
        df['DEC'].to_numpy(),  
        dist.value, 
        randoms.RA.to_numpy(),
        randoms['DEC'].to_numpy(),
        data_weights=data_weights,
        rand_weights=rand_weights)
    rbins, wp_red = calculate_wp(
        df.loc[df['QUIESCENT']].RA.to_numpy(),
        df.loc[df['QUIESCENT']]['DEC'].to_numpy(),
        dist.value[df['QUIESCENT']],
        randoms.RA.to_numpy(),
        randoms['DEC'].to_numpy(),
        data_weights=data_weights,
        rand_weights=rand_weights)
    rbins, wp_blue = calculate_wp(
        df.loc[~df['QUIESCENT']].RA.to_numpy(),
        df.loc[~df['QUIESCENT']]['DEC'].to_numpy(),
        dist.value[~df['QUIESCENT']],
        randoms.RA.to_numpy(),
        randoms['DEC'].to_numpy(),
        data_weights=data_weights,
        rand_weights=rand_weights)

    return (rbins, wp_all, wp_red, wp_blue)

def calculate_wp(data_ra, data_dec, data_dist, rand_ra, rand_dec, rand_dist=None, data_weights=None, rand_weights=None):
    """
    Calculate the projected correlation function wp(rp) for a given data set and randoms using 
    our canonical set of parameters.
    """
    ratio = len(rand_ra) / len(data_ra)
    print(f"Calculate wp: {len(data_ra)} data points, {len(rand_ra)} random points. {ratio} ratio")
    g = np.random.Generator(np.random.PCG64())
    if ratio < 50:
        print("WARNING: The ratio of randoms to data points is less than 50.")

    if ratio > 500: 
        # Reduce the number of randoms to 500x the number of data points
        rand_idx = g.choice(len(rand_ra), size=500*len(data_ra), replace=False)
        rand_ra = rand_ra[rand_idx]
        rand_dec = rand_dec[rand_idx]
        if rand_dist is not None:
            rand_dist = rand_dist[rand_idx]

        print(f"Reduced randoms to {len(rand_ra)} points")
        
    # Random's redshifts should be just like the sample's
    if rand_dist is None:
        rand_dist = g.choice(data_dist, size=len(rand_ra), replace=True)

    DD_counts = DDrppi_mocks(1, COSMOLOGY, NTHREADS, PIMAX, RBINS, data_ra, data_dec, data_dist, weights1=data_weights, is_comoving_dist=True)
    DR_counts = DDrppi_mocks(0, COSMOLOGY, NTHREADS, PIMAX, RBINS, data_ra, data_dec, data_dist, RA2=rand_ra, DEC2=rand_dec, CZ2=rand_dist, weights1=data_weights, weights2=rand_weights, is_comoving_dist=True)
    RR_counts = DDrppi_mocks(1, COSMOLOGY, NTHREADS, PIMAX, RBINS, rand_ra, rand_dec, rand_dist, weights1=rand_weights, is_comoving_dist=True)

    # Convert pair counts to wp 
    wp = convert_rp_pi_counts_to_wp(len(data_ra), len(data_ra), len(rand_ra), len(rand_ra), DD_counts, DR_counts, DR_counts, RR_counts, NBINS, PIMAX)
    return RBINS, wp

