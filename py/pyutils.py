import numpy as np
import astropy.units as u
import matplotlib as plt
from astropy.cosmology import FlatLambdaCDM
import time
from enum import Enum
from scipy import special
import matplotlib.pyplot as plt
import math
import pandas as pd
import sys
from scipy.special import erf
import os
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import astropy.coordinates as coord
import pickle
from numpy.polynomial.polynomial import Polynomial

if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
import k_correction as gamakc
import kcorr.k_corrections as desikc
import k_corr_new.k_corrections as desikc2
from dataloc import *

#sys.path.append("/Users/ianw89/Documents/GitHub/hodpy")
#from hodpy.cosmology import CosmologyMXXL
#from hodpy.k_correction import GAMA_KCorrection

################################
# BINS USED FOR VARIOUS PURPOSES
#
# For the bin count to be right, whether data falls above or below the lowest and highest bin matters
# For plotting, it's better to use midpoints of the bins (or for the edges, a reasonable made up value)
################################

def linlogmixspace(start, end, count, blend=0.5):
    """
    Create values intermediate betwen linear space and logspace, where blend chooses the fraction of logspace (1=fully logspace).
    """
    if blend < 0 or blend > 1:
        raise ValueError("Blend must be between 0 and 1")
    log_vals = np.geomspace(start, end, count)
    lin_vals = np.linspace(start, end, count)
    return lin_vals * (1 - blend) + log_vals * blend


NEIGHBOR_BINS = np.arange(1,11) # 1 to 10 neighbors, inclusive

POBS_BIN_COUNT = 15
POBS_BINS = np.linspace(0.01, 1.01, POBS_BIN_COUNT) # upper bound is higher than any data, lower bound is not

APP_MAG_BIN_COUNT = 12
APP_MAG_BINS = np.linspace(15.5, 20.176, APP_MAG_BIN_COUNT) # upper bound is higher than any data, lower bound is not

ANG_DIST_BIN_COUNT = 15
ANGULAR_BINS = np.append(linlogmixspace(5, 900, ANG_DIST_BIN_COUNT - 1, blend=0.95), 3600) # upper bound is higher than any data, lower bound is not
#ANGULAR_BINS = np.append(np.geomspace(3, 900, ANG_DIST_BIN_COUNT - 1), 3600) # upper bound is higher than any data, lower bound is not

Z_BIN_COUNT = 12
#Z_BINS = np.linspace(0.05, 0.5, Z_BIN_COUNT) # upper bound is higher than any data, lower bound is not
Z_BINS = linlogmixspace(0.04, 0.5, Z_BIN_COUNT, blend=0.2)

DELTA_Z_COUNT = 15
DELTA_Z_BINS = np.linspace(-0.1, 0.1, DELTA_Z_COUNT-1) # upper bound is higher than any data, lower bound is not
DELTA_Z_BINS = np.append(DELTA_Z_BINS, 0.5) # upper bound is higher than any data, lower bound is not

#Z_BINS = np.array([0.094, 0.135, 0.169, 0.199, 0.231, 0.261, 0.291, 0.331, 0.39, 1.0]) # equal gals per bin over 10 bins

Z_BINS_FOR_SIMPLE = np.array([0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.36, 1.0]) 
Z_BINS_FOR_SIMPLE_MIDPOINTS = [0.04, 0.10, 0.14, 0.18, 0.22, 0.26, 0.33, 0.4]
#Z_BINS = Z_BINS_FOR_SIMPLE
#Z_BIN_COUNT = len(Z_BINS) 
#Z_BINS_MIDPOINTS = np.append([0.025], 0.5*(Z_BINS[1:] + Z_BINS[:-1]))

ABS_MAG_BIN_COUNT = 15
ABS_MAG_BINS = np.append(np.linspace(-22.5, -15, ABS_MAG_BIN_COUNT - 1), -5) # upper bound is higher than any data, lower bound is not
ABS_MAG_MIDPOINTS = np.append(np.append([-23], 0.5*(ABS_MAG_BINS[1:-1] + ABS_MAG_BINS[:-2])), -14.5)

QUIESCENT_BINS = np.array([0.0, 1.0]) 

L_gal_bins = np.logspace(6, 12.5, 40)
L_gal_labels = L_gal_bins[0:len(L_gal_bins)-1]

################################

DEGREES_ON_SPHERE = 41253

MASKBITS = dict(
    NPRIMARY   = 0x1,   # not PRIMARY
    BRIGHT     = 0x2,
    SATUR_G    = 0x4,
    SATUR_R    = 0x8,
    SATUR_Z    = 0x10,
    ALLMASK_G  = 0x20,
    ALLMASK_R  = 0x40,
    ALLMASK_Z  = 0x80,
    WISEM1     = 0x100, # WISE masked
    WISEM2     = 0x200,
    BAILOUT    = 0x400, # bailed out of processing
    MEDIUM     = 0x800, # medium-bright star                  NOTHING has this at this point anyway
    GALAXY     = 0x1000, # SGA large galaxy
    CLUSTER    = 0x2000, # Cluster catalog source
)

class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class Mode(Enum):
    ALL = 1 # include all galaxies
    FIBER_ASSIGNED_ONLY = 2 # include only galaxies that were assigned a fiber for FIBER_ASSIGNED_REALIZATION_BITSTRING
    NEAREST_NEIGHBOR = 3 # include all galaxies by assigned galaxies redshifts from their nearest neighbor
    FANCY = 4 
    SIMPLE = 5
    SIMPLE_v4 = 6
    SIMPLE_v5 = 7
    PHOTOZ_PLUS_v1 = 8
    PHOTOZ_PLUS_v2 = 9
    PHOTOZ_PLUS_v3 = 10
    PHOTOZ_PLUS_v4 = 11

    @classmethod
    def is_simple(cls, mode):
        if isinstance(mode, Mode):
            value = mode.value   
        value = mode
        return value == Mode.SIMPLE.value or value == Mode.SIMPLE_v4.value or value == Mode.SIMPLE_v5.value
    
    @classmethod
    def is_photoz_plus(cls, mode):
        if isinstance(mode, Mode):
            value = mode.value   
        value = mode
        return value == Mode.PHOTOZ_PLUS_v1.value or value == Mode.PHOTOZ_PLUS_v2.value or value == Mode.PHOTOZ_PLUS_v3.value or value == Mode.PHOTOZ_PLUS_v4.value
        
    @classmethod
    def is_all(cls, mode):
        if isinstance(mode, Mode):
            value = mode.value   
        value = mode
        return value == Mode.ALL.value

    @classmethod
    def is_fiber_assigned_only(cls, mode):
        if isinstance(mode, Mode):
            value = mode.value   
        value = mode
        return value == Mode.FIBER_ASSIGNED_ONLY.value

def mode_to_str(mode: Mode):
    if mode.value == Mode.ALL.value:
        return "All"
    elif mode.value == Mode.FIBER_ASSIGNED_ONLY.value:
        return "Observed"
    elif mode.value == Mode.NEAREST_NEIGHBOR.value:
        return "NN"
    elif mode.value == Mode.FANCY.value:
        return "Fancy"
    elif mode.value == Mode.SIMPLE.value:
        return "Simple v2"
    elif mode.value == Mode.SIMPLE_v4.value:
        return "Simple v4"
    elif mode.value == Mode.SIMPLE_v5.value:
        return "Simple v5"
    elif mode.value == Mode.PHOTOZ_PLUS_v1.value:
        return "PZP v1"
    elif mode.value == Mode.PHOTOZ_PLUS_v2.value:
        return "PZP v2"
    elif mode.value == Mode.PHOTOZ_PLUS_v3.value:
        return "PZP v3"
    else:
        return "Unknown"
    

class AssignedRedshiftFlag(Enum):
    # TODO switch to using this enum
    PHOTO_Z = -3 # photo-z from legacy catalog; see paper
    PSEUDO_RANDOM = -2 # pseudo-randomly assigned redshift using our methods; see paper
    SDSS_SPEC = -1 # spectroscopic redshfit taken from SDSS
    DESI_SPEC = 0 # spectroscopic redshift from DESI
    NEIGHBOR_ONE = 1 # redshift assigned from nearest neighbor
    NEIGHBOR_TWO = 2 # redshift assigned from second nearest neighbor
    NEIGHBOR_THREE = 3
    NEIGHBOR_FOUR = 4
    NEIGHBOR_FIVE = 5
    NEIGHBOR_SIX = 6
    NEIGHBOR_SEVEN = 7
    NEIGHBOR_EIGHT = 8
    NEIGHBOR_NINE = 9
    NEIGHBOR_TEN = 10
    NEIGHBOR_ELEVEN = 11
    NEIGHBOR_TWELVE = 12
    NEIGHBOR_THIRTEEN = 13
    NEIGHBOR_FOURTEEN = 14
    NEIGHBOR_FIFTEEN = 15

def spectroscopic_complete_percent(flags: np.ndarray):
    return np.logical_or(flags == AssignedRedshiftFlag.SDSS_SPEC.value, flags == AssignedRedshiftFlag.DESI_SPEC.value).sum() / len(flags)

def pseudo_random_redshift_used(flags: np.ndarray):
    return (flags == AssignedRedshiftFlag.PSEUDO_RANDOM.value).sum() / len(flags)

# Common PLT helpers
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
def get_color(i):
    co = colors[i%len(colors)]
    return co

def mode_to_color(mode: Mode):
    if mode == Mode.ALL:
        return get_color(0)
    elif mode == Mode.FIBER_ASSIGNED_ONLY:
        return get_color(1)
    elif mode == Mode.NEAREST_NEIGHBOR:
        return get_color(2)
    elif mode == Mode.FANCY:
        return get_color(3)
    elif mode == Mode.SIMPLE:
        return get_color(6)
    elif mode == Mode.SIMPLE_v4:
        return 'k'

# using _h one makes distances Mpc / h instead
_cosmo_h = FlatLambdaCDM(H0=100, Om0=0.25, Ob0=0.045, Tcmb0=2.725, Neff=3.04) 
_cosmo_h_m30 = FlatLambdaCDM(H0=100, Om0=0.30, Ob0=0.045, Tcmb0=2.725, Neff=3.04) 
_cosmo_mxxl = FlatLambdaCDM(H0=73, Om0=0.25, Ob0=0.045, Tcmb0=2.725, Neff=3.04) 

def get_cosmology():
    return _cosmo_h 

_rho_m = (get_cosmology().critical_density(0) * get_cosmology().Om(0))
def get_vir_radius_mine(halo_mass):
    """Gets virial radius given halo masses (in solMass/h) in kpc/h."""
    return np.power(((3/(4*math.pi)) * halo_mass / (200*_rho_m)), 1/3).to(u.kpc).value

SIM_Z_THRESH = 0.005
def close_enough(target_z, z_arr, threshold=SIM_Z_THRESH):
    """
    Determine if elements in an array are within a specified threshold of a target value.

    Parameters:
    target_z (float): The target value to compare against.
    z_arr (numpy.ndarray): Array of values to compare to the target value.
    threshold (float, optional): The threshold within which values are considered "close enough". Defaults to SIM_Z_THRESH.

    Returns:
    numpy.ndarray: A boolean array where True indicates the corresponding element in z_arr is within the threshold of target_z.
    """
    return np.abs(z_arr - target_z) < threshold

def rounded_tophat_score(target_z, z_arr):
    """
    Compares two arrays of redshifts and returns a smooth transition from 1.0 to 0.0 around a 0.005 difference.
    This is used by the nnanalysis object to evaluat the similar-z fraction that goes into the bins.

    Parameters:
    target_z (float): The target redshift value.
    z_arr (numpy.ndarray): An array of redshift values to compare against the target.

    Returns:
    numpy.ndarray: An array of values smoothly transitioning from 1.0 to 0.0 based on the difference between target_z and z_arr.
    """
    TUNABLE=6
    TURNING=0.0023
    return np.round(1.0 - erf((np.abs(z_arr-target_z) / (np.pi * TURNING))**TUNABLE), 3)

def powerlaw_score_1(target_z, z_arr):
    """
    Compares two arrays of redshifts. Instead of binary close enough evaluations like close_enough,
    this will give a 1.0 for close enough values and smoothly go to 0.0 for definintely far away targets.
    """
    #G_POW = 3
    #SIGMA = 0.0039 
    # try a gaussian instead, normalized so the peak is 1.0
    #gauss = np.exp(-((np.abs(z_arr-target_z) / SIGMA)**G_POW))

    # try a negative power law to get fatter tails
    POW = 3
    FAT = 0.023 #0.0039 # 0.0023
    power =  1.0 / (1 + (np.abs(z_arr-target_z) / FAT)**POW)

    return power

def powerlaw_score_2(target_z, z_arr):
    POW = 3
    FAT = 0.01 
    power =  1.0 / (1 + (np.abs(z_arr-target_z) / FAT)**POW)

    return power

def lorentzian(x, amp, mean, gamma, exp):
    return amp * gamma**2 / (np.power(np.abs(x - mean), exp) + gamma**2)

def zdelta_spectrum(target_z, z_arr):
    # From fitting the distribution of zphot vs zspec
    # amp=22.902018525656057, mean=8.151405674056218e-05, gamma=0.009227129158691942, exp=2.2857148452956757
    return lorentzian(np.abs(z_arr - target_z), 1.0, 8.151405674056218e-05, 0.009227129158691942, 2.2857148452956757)

def photoz_plus_metric_1(z_guessed: np.ndarray[float], z_truth: np.ndarray[float], guess_type: np.ndarray[int]):
    assert guess_type.shape == z_guessed.shape
    assert guess_type.shape == z_truth.shape
    score = powerlaw_score_1(z_guessed, z_truth)

    # pseudorandom guesses help preserve luminosity func so give them a little credit
    score = score + np.where(guess_type == AssignedRedshiftFlag.PSEUDO_RANDOM.value, 0.3, 0.0) 
    score = score + np.where(guess_type == AssignedRedshiftFlag.PHOTO_Z.value, 0.3, 0.0) 

    # in case pseduo-random guesses were spot on it can be > 1.0, so cap the score at 1.0
    score = np.where(score > 1.0, 1.0, score)
    
    # The mean of them all is a sufficient statistic
    return - np.mean(score)

def photoz_plus_metric_2(z_guessed: np.ndarray[float], z_truth: np.ndarray[float], guess_type: np.ndarray[int]):
    assert guess_type.shape == z_guessed.shape
    assert guess_type.shape == z_truth.shape
    score = powerlaw_score_1(z_guessed, z_truth)

    # pseudorandom guesses help preserve luminosity func so give them a little credit
    score = score + np.where(guess_type == AssignedRedshiftFlag.PSEUDO_RANDOM.value, 0.4, 0.0) 
    score = score + np.where(guess_type == AssignedRedshiftFlag.PHOTO_Z.value, 0.4, 0.0) 

    # in case pseduo-random guesses were spot on it can be > 1.0, so cap the score at 1.0
    score = np.where(score > 1.0, 1.0, score)
    
    # The mean of them all is a sufficient statistic
    return - np.mean(score)

def photoz_plus_metric_3(z_guessed: np.ndarray[float], z_truth: np.ndarray[float], guess_type: np.ndarray[int]):
    assert guess_type.shape == z_guessed.shape
    assert guess_type.shape == z_truth.shape
    score = powerlaw_score_1(z_guessed, z_truth)

    # pseudorandom guesses help preserve luminosity func so give them a little credit
    score = score + np.where(guess_type == AssignedRedshiftFlag.PSEUDO_RANDOM.value, 0.4, 0.0) 
    score = score + np.where(guess_type == AssignedRedshiftFlag.PHOTO_Z.value, 0.1, 0.0) 

    # in case pseduo-random guesses were spot on it can be > 1.0, so cap the score at 1.0
    score = np.where(score > 1.0, 1.0, score)
    
    # The mean of them all is a sufficient statistic
    return - np.mean(score)

def photoz_plus_metric_4(z_guessed: np.ndarray[float], z_truth: np.ndarray[float], guess_type: np.ndarray[int]):
    assert guess_type.shape == z_guessed.shape
    assert guess_type.shape == z_truth.shape
    score = powerlaw_score_2(z_guessed, z_truth)

    # pseudorandom guesses help preserve luminosity func so give them a little credit
    score = score + np.where(guess_type == AssignedRedshiftFlag.PSEUDO_RANDOM.value, 0.2, 0.0) 

    # in case pseduo-random guesses were spot on it can be > 1.0, so cap the score at 1.0
    score = np.where(score > 1.0, 1.0, score)
    
    # The mean of them all is a sufficient statistic
    return - np.mean(score)

def get_app_mag(FLUX):
    """This converts nanomaggies into Pogson magnitudes"""
    return 22.5 - 2.5*np.log10(FLUX)

    #g = 22.5 - 2.5*np.log10(gflux.clip(1e-16))
    #r = 22.5 - 2.5*np.log10(rflux.clip(1e-16))
    #z = 22.5 - 2.5*np.log10(zflux.clip(1e-16))
    #w1 = 22.5 - 2.5*np.log10(w1flux.clip(1e-16))
    #rfib = 22.5 - 2.5*np.log10(rfiberflux.clip(1e-16))

def z_to_ldist(zs):
    """
    Gets the luminosity distance of the provided redshifts in Mpc.
    """
    return _cosmo_h.luminosity_distance(zs).value
    
def distmod(zs):
    return 5 * (np.log10(_cosmo_h.luminosity_distance(zs).value * 1E6) - 1)

def bgs_mag_to_sdsslike_mag(mag, band='r'):
    """
    Converts BGS magnitudes to SDSS-like magnitudes using an emperical relation found from galaxies
    observed in both surveys. Assumes the BGS mag is already k-corrected to z=0.1.
    """
    if band == 'r':
        # [-5.09354483 -0.64446931 -0.01922993]
        # Fit in post_plots, see ## BGS and SDSS Target Overlap Analysis
        A = -5.09354483
        B = -0.64446931
        C = -0.01922993
        correction = np.where(mag > -17.0, 0.3, Polynomial([A, B, C]).__call__(mag))
        correction = np.where(mag < -23.75, -0.65, correction)
        return mag + correction
    else:
        raise NotImplementedError(f"Band {band} not implemented")

def app_mag_to_abs_mag(app_mag, zs):
    """
    Converts apparent mags to absolute mags using h=1 cosmology and provided observed redshifts.
    """
    return app_mag - distmod(zs)

def app_mag_to_abs_mag_k(app_mag, z_obs, gmr, band='r'):
    """
    Converts apparent mags to absolute mags using MXXL cosmology and provided observed redshifts,
    with GAMA k-corrections.
    """
    return k_correct(app_mag_to_abs_mag(app_mag, z_obs), z_obs, gmr, band=band)

def k_correct_gama(abs_mag, z_obs, gmr, band='r'):
    corrector = gamakc.GAMA_KCorrection(band=band)
    return abs_mag - corrector.k(z_obs, gmr)

def k_correct_bgs(abs_mag, z_obs, gmr, band='r'):
    corrector  = desikc.DESI_KCorrection(band=band, file='jmext', photsys='S')
    return abs_mag - corrector.k(z_obs, gmr)

def k_correct_bgs_v2(abs_mag, z_obs, gmr, band='r'):
    corrector  = desikc2.DESI_KCorrection(band=band, file='jmext', photsys='S')
    return abs_mag - corrector.k(z_obs, gmr)

# TODO switch to new DESI version - but it is worse at recovering the fastspecfit distrubtion of colors...
# This is what gets called in production code
def k_correct(abs_mag, z_obs, gmr, band='r'):
    return k_correct_gama(abs_mag, z_obs, gmr, band)

SOLAR_L_R_BAND = 4.65
def abs_mag_r_to_log_solar_L(arr):
    """
    Converts an absolute magnitude to log solar luminosities using the sun's r-band magnitude.

    This just comes from the definitions of magnitudes. The scalar 2.5 is 0.39794 dex.
    """
    return 0.39794 * (SOLAR_L_R_BAND - arr)

def log_solar_L_to_abs_mag_r(arr):
    return SOLAR_L_R_BAND - (arr / 0.39794)

from astropy.cosmology import z_at_value

def get_max_observable_z(abs_mags, m_cut):
    # Use distance modulus
    d_l = (10 ** ((m_cut - abs_mags + 5) / 5)) / 1e6 # luminosity distance in Mpc

    return z_at_value(_cosmo_h.luminosity_distance, d_l*u.Mpc) # TODO what cosmology to use?

def get_max_observable_z_m30(abs_mags, m_cut):
    d_l = (10 ** ((m_cut - abs_mags + 5) / 5)) / 1e6 # luminosity distance in Mpc
    return z_at_value(_cosmo_h_m30.luminosity_distance, d_l*u.Mpc) # TODO what cosmology to use?

def get_max_observable_z_mxxlcosmo(abs_mags, m_cut):
    # Use distance modulus
    d_l = (10 ** ((m_cut - abs_mags + 5) / 5)) / 1e6 # luminosity distance in Mpc

    return z_at_value(_cosmo_mxxl.luminosity_distance, d_l*u.Mpc) # TODO what cosmology to use?

def get_volume_at_z(z, frac_area):
    return (4/3*np.pi) * _cosmo_h.luminosity_distance(z).value**3 * frac_area


def get_max_observable_volume(abs_mags, z_obs, m_cut, frac_area):
    """
    Calculate the Vmax (max volume at which the galaxy could be seen) in comoving coords.
    """

    # Use distance modulus
    d_l = (10 ** ((m_cut - abs_mags + 5) / 5)) / 1e6 # luminosity distance in Mpc
    d_cm = d_l / (1 + z_obs) # comoving distance in Mpc

    v_max = (d_cm**3) * (4*np.pi/3) # in comoving Mpc^3 

    # TODO BUG is this supposed to have / h in it?
    # TODO BUG I'm not convinced that everywhere we use Vmax frac_area should be baked into it
    # Group finder seems to expect this but not 100% confident
    # My fsat calculations that have 1/Vmax weightings are not affected by this
    return v_max * frac_area

def get_max_observable_volume_est(abs_mags, z_obs, m_cut, ra, dec):
    """
    Calculate the max volume at which the galaxy could be seen in comoving coords. 

    This overload calculates 
    """
    frac_area = estimate_frac_area(ra, dec)
    print(f"Footprint not provided; estimated to be {frac_area:.3f} of the sky")

    return get_max_observable_volume(abs_mags, z_obs, m_cut, frac_area)


def mollweide_transform(ra, dec):
    """
    Transform ra, dec arrays into x, y arrays for plotting in Mollweide projection.
    
    Expects ra, dec in degrees. They must range -180 to 180 and -90 to 90.
    """
    assert np.all(ra < 180.01)
    assert np.all(dec < 90.01)
    assert len(ra) == len(dec)

    def d(theta):
        delta = (-(theta + np.sin(theta) - pi_sin_l)
                         / (1 + np.cos(theta)))
        return delta, np.abs(delta) > 0.001

    longitude = ra * np.pi / 180 # runs -pi to pi
    latitude = dec * np.pi / 180 # should run -pi/2 to pi/2

    # Mollweide projection
    clat = np.pi/2 - np.abs(latitude)
    ihigh = clat < 0.087  # within 5 degrees of the poles
    ilow = ~ihigh
    aux = np.empty(latitude.shape, dtype=float)

    if ilow.any():  # Newton-Raphson iteration
        pi_sin_l = np.pi * np.sin(latitude[ilow])
        theta = 2.0 * latitude[ilow]
        delta, large_delta = d(theta)
        while np.any(large_delta):
            theta[large_delta] += delta[large_delta]
            delta, large_delta = d(theta)
        aux[ilow] = theta / 2

    if ihigh.any():  # Taylor series-based approx. solution
        e = clat[ihigh]
        d = 0.5 * (3 * np.pi * e**2) ** (1.0/3)
        aux[ihigh] = (np.pi/2 - d) * np.sign(latitude[ihigh])

    x = np.empty(len(longitude), dtype=float)
    y = np.empty(len(latitude), dtype=float)

    x = (2.0 * np.sqrt(2.0) / np.pi) * longitude * np.cos(aux) * (180/np.pi)
    y = np.sqrt(2.0) * np.sin(aux) * (180/np.pi)

    return x, y

def estimate_frac_area(ra, dec):
    """
    Estimate the fraction of the sky covered in the survey so far based on the ra, dec 
    of galaxies that have been observed thus far.

    BUG This procedure does not work very well. Though it recovered somethign reasonable
    for MXXL and UCHUU, it failed to get what I expected for BGS based on the randoms.

    We haven't used the fact that the each pointing of the telescope covers 7.44 square 
    degrees of the sky in this yet.
    """

    # Reduce data if too large
    #_MAX_POINTS = 10000000 
    #if len(ra) > 2*_MAX_POINTS:
    #    reduce = len(ra) // _MAX_POINTS
    #    #print(f"Reducing data size by a factor of {reduce}")
    #    rnd_indices = np.random.choice(len(ra), len(ra)//reduce, replace=False)
    #    ra = ra[rnd_indices]
    #    dec = dec[rnd_indices]
    
    # Shift to -pi to pi and -pi/2 to pi/2 if needed
    if np.any(ra > 180.0): # if data given is 0 to 360
        assert np.all(ra > -0.1)
        ra = ra - 180
    if np.any(dec > 90.0): # if data is 0 to 180
        assert np.all(dec > -0.1)
        dec = dec - 90

    #print("ra", min(ra), max(ra))
    #print("dec", min(dec), max(dec))
    x, y = mollweide_transform(ra, dec)
    #print("x", min(x), max(x))
    #print("y", min(y), max(y))

    #plt.figure(figsize=(8,4))
    #plt.scatter(ra, dec, alpha=0.1)
    #plt.scatter(x, y, alpha=0.1)
    #plt.title("Blue: Original. Orange: Mollweide.")
    #plt.xlim(-180,180)
    #plt.ylim(-90,90)

    # Now we have the x and y coordinates of the points in the Mollweide projection
    # We can use these to make a 2D histogram of the points
    # But the projection makes some of the bins impossible to fill; ignore those bins
    # Also we may need to tune fineness
    fineness=4 # fineness^2 is how many bins per square degree
    accessible_bins = DEGREES_ON_SPHERE*fineness**2 # 41253 square degrees in the sky, this must be the max bins

    xbins = np.linspace(-180,180,360*fineness +1)
    ybins = np.linspace(-90,90,180*fineness +1)
    hist, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins])

    filled_bincount = np.count_nonzero(hist)

    #print(f"Filled bins: {filled_bincount}. Total bins: {accessible_bins}")
    # You can fill slightly bins than the max due to edge effects I think?
    frac_area = min(filled_bincount / accessible_bins, 1.0)
    
    return frac_area


def build_app_mag_to_z_map(app_mag, z_obs):
    _NBINS = 100
    _MIN_GALAXIES_PER_BIN = 100
    app_mag_bins = np.linspace(min(app_mag), max(app_mag), _NBINS)
    #app_mag_bins = np.quantile(app_mag, np.linspace(0, 1, _NBINS + 1)) 

    app_mag_indices = np.digitize(app_mag, app_mag_bins)

    the_map = {}
    for bin_i in range(1,len(app_mag_bins)+1):
        this_bin_redshifts = z_obs[app_mag_indices == bin_i]
        the_map[bin_i] = this_bin_redshifts

    # for app mags smaller than the smallest we have, use the z distribution of the one right above it
    if 0 in the_map:
        print("UNEXPECTED")
    the_map[0] = the_map[1]
    assert len(app_mag_bins) == (len(the_map)-1)

    to_check = list(the_map.keys())
    for k in to_check:
        if len(the_map[k]) < _MIN_GALAXIES_PER_BIN and k < len(the_map)-1:
            #print(f"App Mag Bin {k} has too few galaxies. Adding in galaxies from the next bin to this one.")
            the_map[k] = np.concatenate((the_map[k], the_map[k+1]))
            to_check.append(k) # recheck it to see if it's still too small

    assert len(app_mag_bins) == (len(the_map)-1)
    #print(f"App Mag Building Complete: {the_map}")

    return app_mag_bins, the_map

def build_app_mag_to_z_map_2(app_mag, z_obs):
    nbins = len(app_mag) // 30000
    app_mag_bins = np.quantile(np.array(app_mag), np.linspace(0, 1, nbins + 1)) 
    app_mag_bins_low = np.linspace(min(app_mag), app_mag_bins[3], 30)
    app_mag_bins = np.concatenate((app_mag_bins_low, app_mag_bins[4:]))

    app_mag_indices = np.digitize(app_mag, app_mag_bins)

    the_map = {}
    for bin_i in range(1,len(app_mag_bins)+1):
        this_bin_redshifts = z_obs[app_mag_indices == bin_i]
        the_map[bin_i] = this_bin_redshifts

    # for app mags smaller than the smallest we have, use the z distribution of the one right above it
    if 0 in the_map:
        print("UNEXPECTED")
    the_map[0] = the_map[1]
    assert len(app_mag_bins) == (len(the_map)-1)

    return app_mag_bins, the_map

def build_app_mag_to_z_map_3(app_mag, z_phot, z_obs):
    nbins = len(app_mag) // 250000
    _MIN_GALAXIES_PER_BIN = 100
    app_mag_bins = np.quantile(np.array(app_mag), np.linspace(0, 1, nbins + 1)) 
    app_mag_bins_low = np.linspace(min(app_mag), app_mag_bins[3], 12)
    app_mag_bins = np.concatenate((app_mag_bins_low, app_mag_bins[4:]))

    z_phot_bins = np.linspace(0.0, 0.6, 21)

    app_mag_indices = np.digitize(app_mag, app_mag_bins)
    z_phot_indices = np.digitize(z_phot, z_phot_bins)

    the_map = {}
    for app_bin in range(1, len(app_mag_bins) + 1):
        for z_bin in range(1, len(z_phot_bins) + 1):
            bin_key = (app_bin, z_bin)
            this_bin_redshifts = z_obs[(app_mag_indices == app_bin) & (z_phot_indices == z_bin)]
            the_map[bin_key] = this_bin_redshifts

    fixing = True
    bin_jump = 1
    while(fixing):
        print(f"Padding with bin_jump {bin_jump}")  
        fixing = False
        for bin_key in list(the_map.keys()):
            app_bin, z_bin = bin_key
            if len(the_map[bin_key]) < _MIN_GALAXIES_PER_BIN and app_bin+bin_jump < len(app_mag_bins):
                the_map[bin_key] = np.concatenate((the_map[bin_key], the_map[(app_bin+bin_jump, z_bin)]))
                fixing = True
            elif len(the_map[bin_key]) < _MIN_GALAXIES_PER_BIN and app_bin-bin_jump > 0:
                print("edge case")
                the_map[bin_key] = np.concatenate((the_map[bin_key], the_map[(app_bin-bin_jump, z_bin)]))
        bin_jump += 1

    # Add values for edge in case we see anything to left of bins
    for z_bin in range(1, len(z_phot_bins) + 1):
        the_map[0, z_bin] = the_map[1, z_bin]
    for app_bin in range(1, len(app_mag_bins) + 1):
        the_map[app_bin, 0] = the_map[app_bin, 1]

    return app_mag_bins, z_phot_bins, the_map


def build_app_mag_to_z_map_4(app_mag, z_phot, z_obs):
    nbins = len(app_mag) // 250000
    _MIN_GALAXIES_PER_BIN = 50
    app_mag_bins = np.quantile(np.array(app_mag), np.linspace(0, 1, nbins + 1)) 
    app_mag_bins_low = np.linspace(min(app_mag), app_mag_bins[3], 12)
    app_mag_bins = np.concatenate((app_mag_bins_low, app_mag_bins[4:]))

    z_phot_bins = np.linspace(0.0, 0.6, 21)

    app_mag_indices = np.digitize(app_mag, app_mag_bins)
    z_phot_indices = np.digitize(z_phot, z_phot_bins)

    the_map = {}
    for app_bin in range(1, len(app_mag_bins) + 1):
        for z_bin in range(1, len(z_phot_bins) + 1):
            bin_key = (app_bin, z_bin)
            this_bin_redshifts = z_obs[(app_mag_indices == app_bin) & (z_phot_indices == z_bin)]
            the_map[bin_key] = this_bin_redshifts

    # Remove keys that lead tp < min
    to_remove = []
    for bin_key in list(the_map.keys()):
        if len(the_map[bin_key]) < _MIN_GALAXIES_PER_BIN:
            to_remove.append(bin_key)
    for key in to_remove:
        del the_map[key]

    return app_mag_bins, z_phot_bins, the_map

class dn4000lookup:
    """
    This is a lookup table for dn4000 values based on absolute magnitude and g-r color.
    It uses a KDTree for fast nearest neighbor search.
    """
    def __init__(self, file = BGS_Y3_DN4000_LOOKUP_FILE):
        self.METRIC_MAG = 1
        self.METRIC_GMR = 5
        if file is None or os.path.isfile(file) is False:
            raise ValueError(f"File {file} does not exist. The Dn4000 lookup table must be built first; see BGS_study.ipynb.")
        self.tree, self.dn4000_lookup = pickle.load(open(file, 'rb'))

    def query(self, abs_mag_array, gmr_array, k=50):
        query_points = np.vstack((abs_mag_array * self.METRIC_MAG, gmr_array * self.METRIC_GMR)).T  # Scale the query points
        distances, indices = self.tree.query(query_points, k=k)  # Query the KDTree for multiple points
        print(np.shape(distances), np.shape(indices))

        # Pick random neighbors for each query point, weighted by distance
        values = []
        for i in range(len(query_points)):
            p = (1 / distances[i]) / np.sum(1 / distances[i])
            values.append(np.random.choice(self.dn4000_lookup[indices[i]], p=p, size=1)[0])
        
        return np.array(values)
    


###############################################
# Lost Galaxy Redshift Assignment
###############################################

class RedshiftGuesser():

    def __enter__(self):
        np.set_printoptions(precision=4, suppress=True)
        return self
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        np.set_printoptions(precision=8, suppress=False)


# TODO photo-z instead of (or in addition to) app mag. But cannot do for MXXL, so need a SV3 version of this analysis first.

def get_NN_40_line_v5(z, t_appmag, target_quiescent, nn_quiescent):
    assert len(z) == len(t_appmag)
    if not isinstance(target_quiescent,(list,pd.core.series.Series,np.ndarray)):
        target_quiescent = np.repeat(target_quiescent, len(z))
    if not isinstance(nn_quiescent,(list,pd.core.series.Series,np.ndarray)):
        nn_quiescent = np.repeat(nn_quiescent, len(z))
    assert len(z) == len(target_quiescent)
    assert len(z) == len(nn_quiescent)
    target_quiescent = target_quiescent.astype(bool)
    nn_quiescent = nn_quiescent.astype(bool)

    #zb = np.digitize(z, Z_BINS_FOR_SIMPLE)
    #t_appmag_b = np.digitize(t_appmag, APP_MAG_BINS)
    
    # Treat mags lower than 15.0 as 15.0
    r = t_appmag - 14.0 # so will be between 1 and 6 usually
    r = np.where(r < 1.0, 1.0, r)

    # Treat redshifts higher than 0.4 as 0.4
    z = np.where(z > 0.4, 0.4, z)

    # Sets a app magnitude cutoff via an erf.
    # Inside the erf, the multiplier squeezes it (transitions quicker),
    # and the number before r is the offset in app mag space, and the number
    # before z controls redshift dependence on the offset
    cutoff_dim = np.ones(len(z))
    cutoff_bright = np.ones(len(z))

    # Redshift dependence changes slope. z in 0.0 to 0.4 to slope of -0.3 to 0.3
    # Redshift dependence also lowers distance thrshold
    
    # m controls slope
    m = np.zeros(len(z))

    # increasing b by 1 shifts the line right by 1 order of magnitude in ang dist
    b = np.zeros(len(z))

    # Sets a hard upp limit on the distance threshold
    upper_lim = np.ones(len(z)) * 250
             
    idx = np.flatnonzero(nn_quiescent & target_quiescent)
    m[idx] = 0
    b[idx] = 2.2 - 2.0*z[idx]
    cutoff_dim[idx] = -0.5 * erf(1.5*(-3.5+r[idx] - 12.0*z[idx])) + 0.5
    cutoff_bright[idx] = 0.5 * erf(2*(1.2+r[idx] - 15.0*z[idx])) + 0.5
    #logangdist = 2 - 0.05*zb

    idx = np.flatnonzero(nn_quiescent & ~target_quiescent)
    m[idx] = (1.8*z[idx]) - 0.6
    b[idx] = 3.8 - 5.0*z[idx] - 11.0*z[idx]**2
    upper_lim[idx] = 250 -  380.0*np.sqrt(z[idx]) # that is 10 for z=0.4
    cutoff_bright[idx] = 0.5 * erf(2*(0.5+r[idx] - 14.0*z[idx])) + 0.5
    #logangdist = 2 - 0.2*zb

    idx = np.flatnonzero(~nn_quiescent & target_quiescent)
    m[idx] = (5*z[idx]) - 1.2
    b[idx] = 4.5 - 2.8*z[idx] - 40.0*z[idx]**2
    upper_lim[idx] = 250 - 1250.0*z[idx] + 1450*z[idx]**2 
    upper_lim[idx] = np.where(upper_lim[idx] < 10, 10, upper_lim[idx])
    cutoff_bright[idx] = 0.5 * erf(2*(0.5+r[idx] - 14.0*z[idx])) + 0.5
    #logangdist = 2 - 0.1*zb

    idx = np.flatnonzero(~nn_quiescent & ~target_quiescent)
    m[idx] = (1.9*z[idx]) - 0.5
    b[idx] = 3.3 - 5.5*z[idx] - 13.0*z[idx]**2
    cutoff_bright[idx] = 0.5 * erf(2*(1.0+r[idx] - 22.0*z[idx] + 15*z[idx]**2)) + 0.5
    #logangdist = 2 - 0.2*zb

    d = (m*r + b) * cutoff_dim * cutoff_bright

    results = 10**d
    results = np.where(10**(d) > upper_lim, upper_lim, results)
    results = np.where(10**(d) < 0.0, 0.0, results)
    return results


def get_NN_30_line(z, t_Pobs):
    """
    Gets the angular distance at which, according to MXXL, a target with the given Pobs will be in the same halo
    as a nearest neighbor at reshift z 30% of the time.
    """
    FIT_SHIFT_RIGHT = np.array([37,30,31,30,39,39,63,72])
    FIT_SCALE = np.array([10,10,10,10,10,10,10,10])
    FIT_SHIFT_UP = np.array([0.8,0.8,0.8,0.8,0.8,0.8,0.6,0.5])
    FIT_SQUEEZE = np.array([1.4,1.3,1.3,1.3,1.4,1.4,1.5,1.5])
    base = np.array([1.10,1.14,1.14,1.14,1.10,1.10,1.06,1.04])
    zb = np.digitize(z, Z_BINS_FOR_SIMPLE)

    erf_in = FIT_SQUEEZE[zb]*(t_Pobs - FIT_SHIFT_UP[zb])

    # for middle ones use exponentiated inverse erf to get the curve 
    exponent = FIT_SHIFT_RIGHT[zb] - FIT_SCALE[zb]*special.erfinv(erf_in)
    arcsecs = base[zb]**exponent
    return arcsecs

def get_NN_40_line_v2(z, t_Pobs):
    """
    Gets the angular distance at which, according to MXXL, a target with the given Pobs will be in the same halo
    as a nearest neighbor at reshift z 40% of the time.
    """
    FIT_SHIFT_RIGHT = np.array([25,25,26,27,34,34,53,60])
    FIT_SHIFT_UP = np.array([0.8,0.8,0.8,0.8,0.8,0.8,0.6,0.5])
    FIT_SQUEEZE = np.array([1.4,1.3,1.3,1.3,1.4,1.4,1.5,1.5])
    base = np.array([1.10,1.14,1.14,1.14,1.10,1.10,1.06,1.04])
    zb = np.digitize(z, Z_BINS_FOR_SIMPLE)

    erf_in = FIT_SQUEEZE[zb]*(t_Pobs - FIT_SHIFT_UP[zb])

    exponent = FIT_SHIFT_RIGHT[zb] - 10*special.erfinv(erf_in)

    arcsecs = base[zb]**exponent
    return arcsecs


def get_NN_40_line_v4(z, t_Pobs, target_quiescent, nn_quiescent):
    """
    Gets the angular distance at which, according to MXXL, a target with the given Pobs will be in the same halo
    as a nearest neighbor at reshift z 40% of the time.
    """    
    assert len(z) == len(t_Pobs)
    if not isinstance(target_quiescent,(list,pd.core.series.Series,np.ndarray)):
        target_quiescent = np.repeat(target_quiescent, len(z))
    if not isinstance(nn_quiescent,(list,pd.core.series.Series,np.ndarray)):
        nn_quiescent = np.repeat(nn_quiescent, len(z))
    assert len(z) == len(target_quiescent)
    assert len(z) == len(nn_quiescent)
    target_quiescent = target_quiescent.astype(bool)
    nn_quiescent = nn_quiescent.astype(bool)

    FIT_SHIFT_RIGHT = np.array([[30,32,33,34,43,40,63,75],[15,25,26,26,34,32,50,50],[40,30,26,27,30,30,40,40],[30,20,20,20,25,25,35,40]])
    FIT_SHIFT_UP = np.array([0.7,0.7,0.7,0.7,0.7,0.7,0.6,0.5])
    FIT_SQUEEZE = np.array([1.4,1.3,1.3,1.3,1.4,1.4,1.5,1.5])
    base = np.array([1.10,1.14,1.14,1.14,1.10,1.10,1.06,1.04])
    zb = np.digitize(z, Z_BINS_FOR_SIMPLE)

    erf_in = FIT_SQUEEZE[zb]*(t_Pobs - FIT_SHIFT_UP[zb])

    # The exponent is color-dependent
    exponent = np.ones(len(z))

    idx = np.flatnonzero(nn_quiescent & target_quiescent)
    exponent[idx] = FIT_SHIFT_RIGHT[0][zb[idx]] - 10*special.erfinv(erf_in[idx])

    idx = np.flatnonzero(~nn_quiescent & target_quiescent)
    exponent[idx] = FIT_SHIFT_RIGHT[1][zb[idx]] - 10*special.erfinv(erf_in[idx])

    idx = np.flatnonzero(nn_quiescent & ~target_quiescent)
    exponent[idx] = FIT_SHIFT_RIGHT[2][zb[idx]] - 10*special.erfinv(erf_in[idx])

    idx = np.flatnonzero(~nn_quiescent & ~target_quiescent)
    exponent[idx] = FIT_SHIFT_RIGHT[3][zb[idx]] - 10*special.erfinv(erf_in[idx])

    arcsecs = base[zb]**exponent
    return arcsecs

def get_NN_50_line(z, t_Pobs):
    """
    Gets the angular distance at which, according to MXXL, a target with the given Pobs will be in the same halo
    as a nearest neighbor at reshift z 50% of the time.
    """
    FIT_SHIFT_RIGHT = np.array([15,20,20,20,25,25,35,40])
    FIT_SCALE = np.array([10,10,10,10,10,10,10,10])
    FIT_SHIFT_UP = np.array([0.8,0.8,0.8,0.8,0.8,0.8,0.6,0.5])
    FIT_SQUEEZE = np.array([1.3,1.3,1.3,1.3,1.4,1.4,1.5,1.5])
    base = np.array([1.16,1.16,1.16,1.16,1.12,1.12,1.08,1.08])
    zb = np.digitize(z, Z_BINS_FOR_SIMPLE)

    # for higher and lower z bit just use simple cut
    if zb in [0,6,7]:
        return np.full(t_Pobs.shape, 10)
    #if zb == 7:
    #    return np.full(t_Pobs.shape, 10)

    erf_in = FIT_SQUEEZE[zb]*(t_Pobs - FIT_SHIFT_UP[zb])

    # for middle ones use exponentiated inverse erf to get the curve 
    exponent = FIT_SHIFT_RIGHT[zb] - FIT_SCALE[zb]*special.erfinv(erf_in)
    arcsecs = base[zb]**exponent
    return arcsecs



class FancyRedshiftGuesser(RedshiftGuesser):

    def __init__(self, num_neighbors, debug=False):
        print("Initializing v6 of FancyRedshiftGuesser")
        self.debug = debug
        self.rng = np.random.default_rng()

        self.nn_used = np.zeros(num_neighbors, dtype=np.int32)
        self.nn_correct = np.zeros(num_neighbors, dtype=np.int32)
        self.quick_nn = 0
        self.quick_correct = 0

        # These are from MXXL 19.5 cut
        with open('bin/abs_mag_weight.npy', 'rb') as f:
            # pad the density array with a duplicate of the first value for < first bin, and 0 > last bin
            self.abs_mag_density = np.load(f)
            self.abs_mag_density = np.insert(self.abs_mag_density, [0,len(self.abs_mag_density)], [self.abs_mag_density[0],0])
            self.abs_mag_bins = np.load(f)

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, exc_tb):
        t = time.time()
        with open(f"bin/redshift_guesser_{t}.npy", 'wb') as f:
            np.save(f, self.quick_nn, allow_pickle=False)
            np.save(f, self.quick_correct, allow_pickle=False)
            np.save(f, self.nn_used, allow_pickle=False)
            np.save(f, self.nn_correct, allow_pickle=False)
        

        # TODO adding 1 to denominator hack
        print(f"Quick NN uses: {self.quick_nn}. Success: {self.quick_correct / (self.quick_nn+1)}")
        print(f"NN bin uses: {self.nn_used}. Success: {self.nn_correct / (self.nn_used+1)}")
        
        super().__exit__(exc_type,exc_value,exc_tb)


    # The magic numbers in this right now are the ~0.5 NN same-halo points read-off of plots from the MXXL data
    def use_nn(self, neighbor_z, neighbor_ang_dist, target_prob_obs, target_app_mag):
        adjust = 0
        if target_prob_obs < 0.60:
            adjust = 5

        if neighbor_z < 0.1:
            threshold = 10
        elif neighbor_z < 0.2:
            threshold = 20
        elif neighbor_z < 0.3:
            threshold = 18
        else: # z > 0.3
            threshold = 8
        
        if neighbor_ang_dist < (threshold + adjust):
            implied_abs_mag = app_mag_to_abs_mag(target_app_mag, neighbor_z)
            if -24 < implied_abs_mag and implied_abs_mag < -14:
                return True

        return False

    def score_neighbors(self, neighbors_z, neighbors_ang_dist, target_prob_obs, target_app_mag):
        """
        neighbors_z should be their spectroscopic redshifts
        neighbors_ang_dist should be in arsec
        target_prob_obs is the probability a fiber was assigned to the target (lost) galaxy
        target_app_mag is the DESI r-band apparent mag of the target (lost) galaxy
        """

        # TODO use z bins and p_obs
        # TODO consider how me using z from target messes up plots I got this from

        # this is rough line fitting important region of my plots of MXXL Truth data
        fsuccess = 1.0 - 0.45*np.log10(neighbors_ang_dist)
        # Ensure all neighbors have a nonzero base score (0.01)so other factors can weigh in
        MIN_ANGULAR_SCORE = 0.0001 # effectively sets downweighted far away neighbors are
        ang_dist_scores = np.max(np.stack((fsuccess, np.full(len(fsuccess), MIN_ANGULAR_SCORE)), axis=1), axis=1)
        if self.debug:
            print(f"Angular Distance scores: {ang_dist_scores}")

        # Form groups 
        # TODO grouping effects?
        # Scale the nearest neighbor in the group's score by the number of neighbors in that group
        # Remove the other ones in a group from the pool by setting the score to 0
        grouped_scores = np.copy(ang_dist_scores)
        #group_z = np.copy(neighbors_z)
        for i in range(0, len(neighbors_z)):
            for j in range(len(neighbors_z)-1, 0, -1):
                if grouped_scores[j] > 0 and grouped_scores[i] > 0 and i!=j:
                    if close_enough(neighbors_z[i], neighbors_z[j]):
                        grouped_scores[j] = 0 # remove from the pool
                        grouped_scores[i] += ang_dist_scores[i] 

        if self.debug:
            print(f"Grouped scores: {grouped_scores}")                

        # Let implied abs mag weigh group. 
        # TODO Issue: Overweighting? see plot of results vs truth
        implied_abs_mag = app_mag_to_abs_mag(target_app_mag, neighbors_z)
        abs_mag_bin = np.digitize(implied_abs_mag, self.abs_mag_bins)
        abs_mag_scores = self.abs_mag_density[abs_mag_bin]
        if self.debug:
            print(f"abs_mag_scores scores: {abs_mag_scores}")   

        final_scores = grouped_scores * abs_mag_scores

        return final_scores
    
    def choose_winner(self, neighbors_z, neighbors_ang_dist, target_prob_obs, target_app_mag, target_z_true):
        if self.debug:
            print(f"\nNew call to choose_winner")
            
        k = 0
        if self.use_nn(neighbors_z[k], neighbors_ang_dist[k], target_prob_obs, target_app_mag):
            if close_enough(target_z_true, neighbors_z[k]):
                self.quick_correct += 1
            self.quick_nn += 1
            if self.debug:
                print(f"Used quick NN. True z={target_z_true}. NN: z={neighbors_z[k]}, ang dist={neighbors_ang_dist[k]}")
            return 0

        scores = self.score_neighbors(neighbors_z, neighbors_ang_dist, target_prob_obs, target_app_mag)

        if self.debug:
            print(f"Scores: {scores}")

        # Randomly choose a winner uses the scores as a PDF to draw from
        #i = self.rng.choice(len(neighbors_z), p=scores)
        #i = random.choices(range(len(neighbors_z)), weights=scores)

        # ... or choose maximal score
        i = np.argmax(scores)


        if close_enough(target_z_true, neighbors_z[i]):
            self.nn_correct[i] += 1
        self.nn_used[i] = self.nn_used[i] + 1
        if self.debug:
            print(f"Used scoring. True z={target_z_true}. NN {i}: z={neighbors_z[i]}, ang dist={neighbors_ang_dist[i]}")

        return i


def write_dat_files(ra, dec, z_eff, log_L_gal, V_max, colors, chi, outname_base):
    """
    Use np.column_stack with dtype='str' to convert your galprops arrays into an all-string
    array before passing it in.
    """

    count = len(ra)
    outname_1 = outname_base + ".dat"
    #outname_2 = outname_base + "_galprops.dat"

    print(f"Output file will be {outname_1}")

    # Time the two approaches
    t1 = time.time()
    lines_1 = []
    #lines_2 = []

    for i in range(0, count):
        lines_1.append(f'{ra[i]:.14f} {dec[i]:.14f} {z_eff[i]:.14f} {log_L_gal[i]:f} {V_max[i]:f} {colors[i]} {chi[i]}')  
        #lines_2.append(' '.join(map(str, galprops[i])))

    outstr_1 = "\n".join(lines_1)
    #outstr_2 = "\n".join(lines_2)    

    open(outname_1, 'w').write(outstr_1)
    #open(outname_2, 'w').write(outstr_2)
    t2 = time.time()

    # Experiment
    #t3 = time.time()
    #pd.DataFrame({'ra':ra, 'dec':dec, 'z_eff':z_eff, 'log_L_gal':log_L_gal, 'VMAX':V_max, 'colors':colors, 'chi':chi}).to_csv(outname_base + ".dat~", sep=' ', index=False, header=False)
    #pd.DataFrame(galprops).to_csv(outname_base + "_galprops.dat~", sep=' ', index=False, header=False)
    #t4 = time.time()

    print(f"Time for file writing: {t2-t1:.2f}")
    #print(f"Time for pandas-based file writing: {t4-t3}")

def write_dat_files_v2(ra, dec, z_eff, log_L_gal, V_max, colors, chi, outname_base):
    """
    Use np.column_stack with dtype='str' to convert your galprops arrays into an all-string
    array before passing it in.
    """
    outname_1 = outname_base + ".dat"

    print(f"Output file will be {outname_1}")

    # Time the optimized approach
    t1 = time.time()

    # Use numpy to build the output strings more efficiently
    data = np.column_stack((ra, dec, z_eff, log_L_gal, V_max, colors, chi))

    np.savetxt(outname_1, data, fmt=['%.14f', '%.14f', '%.14f', '%f', '%f', '%d', '%s'])

    t2 = time.time()

    print(f"Time for file writing: {t2-t1:.2f}")

    # Experiment with feather
    #import pyarrow.feather as feather

    #t3 = time.time()
    #df = pd.DataFrame(data, columns=['ra', 'dec', 'z_eff', 'log_L_gal', 'V_max', 'colors', 'chi'])
    #feather.write_feather(df, outname_base + ".feather")
    #t4 = time.time()
    #print(f"Time for feather writing: {t4-t3:.2f}")






######################################
# Color Cuts
######################################

# Function to fit a 2-component Gaussian Mixture Model to a property and plot the results
def fit_gmm_and_plot(values, logLgal_bin_idx, num_bins, min, max, name, means_init=None, manual_thresholds=None):
    from scipy.optimize import fsolve

    midpoints = []

    for i in range(1, num_bins + 1):
        galaxy_idx_for_this_bin = logLgal_bin_idx == i
        binned_values = values[galaxy_idx_for_this_bin]

        if len(binned_values) == 0:
            continue

        # Fit a Gaussian Mixture Model with 2 components
        gmm = GaussianMixture(n_components=2, tol=1e-5, max_iter=1500, means_init=means_init)
        binned_values_reshaped = binned_values.reshape(-1, 1)
        gmm.fit(binned_values_reshaped)

        # Get the parameters of the fitted Gaussians
        means = gmm.means_.flatten()
        covariances = gmm.covariances_.flatten()
        weights = gmm.weights_.flatten()

        # Sort the means, covariances, and weights from smallest to largest mean
        sorted_indices = np.argsort(means)
        means = means[sorted_indices]
        covariances = covariances[sorted_indices]
        weights = weights[sorted_indices]

        x = np.linspace(min, max, 400)
        def model1(_x):
            return weights[0] * np.exp(-0.5 * ((_x - means[0]) ** 2) / covariances[0]) / np.sqrt(2 * np.pi * covariances[0])
        def model2(_x):
            return weights[1] * np.exp(-0.5 * ((_x - means[1]) ** 2) / covariances[1]) / np.sqrt(2 * np.pi * covariances[1])
        #def model3(_x):
        #    return weights[2] * np.exp(-0.5 * ((_x - means[2]) ** 2) / covariances[2]) / np.sqrt(2 * np.pi * covariances[2])
        def equations(_x):
            return model1(_x) - model2(_x)
        intersection = fsolve(equations, (means[0]+means[1])/2)[0]

        # Plot the histogram and the fitted Gaussians
        plt.figure(dpi=80, figsize=(10, 5))
        bins = np.arange(min, max, (max-min)/200)
        plt.hist(binned_values, bins=bins, density=True, alpha=0.6, label=f"Values")
        plt.plot(x, model1(x) + model2(x), label='Gaussian Mixture Model')
        plt.plot(x, model1(x), label='Gaussian 1')
        plt.plot(x, model2(x), label='Gaussian 2')
        #plt.plot(x, model3(x), label='Gaussian 3')
        buffer = (max-min) * 0.15
        midpoints.append(intersection)
        if min+buffer <= intersection <= max-buffer:
            plt.axvline(intersection, color='r', linestyle='--', label=f'Intersection at {intersection:.2f}')
        if manual_thresholds is not None:
            plt.axvline(manual_thresholds[i-1], color='g', linestyle='--', label=f'Chosen Threshold {manual_thresholds[i-1]:.2f}')
        plt.legend()
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title(f'{name} GMM for L {BGS_LOGLGAL_BINS[i-1]} - {BGS_LOGLGAL_BINS[i]}')
        plt.show()

        print(f"Bin {i}:")
        print(f"Means: {means}")
        print(f"Covariances: {covariances}")
        print(f"Weights: {weights}")
        print(f"Intersection: {intersection}")

    if len(midpoints) != num_bins:
        print("WARNING: Not all bins had easily identifiable good intersections.")
    return midpoints

def get_SDSS_Dcrit(logLgal):
    # From Jeremy's Paper based on Sloan MGS
    return 1.42 + (0.35 / 2) * (1 + special.erf((logLgal - 9.9) / 0.8))

def get_Dn4000_crit(logLgal):
    # See BGS_Study.ipynb Dn4000 section
    # This is for DN4000_MODEL! Don't use Dn4000 column, it's noiser.
    # OLD 1: A=1.411 B=0.171 C=9.795 D=0.775
    # OLD 2: A=1.377 B=0.181 C=9.881 D=1.108
    # Old 3: A=1.433 B=0.150 C=9.639 D=1.070
    A=1.452
    B=0.123
    C=9.275
    D=1.186
    return A + B*(1 + special.erf((logLgal - C) / D))

def get_halpha_crit(logLgal):
    # See BGS_Study.ipynb Halpha section
    # This is for LOG10(HALPHA_EW)
    #Fitted parameters: A=0.959, B=-0.228, C=10.254, D=0.791
    # Old: A=0.959, B=-0.228, C=10.254, D=0.791
    #A=0.959
    #B=-0.228
    #C=10.254
    #D=0.791
    #return A + B*(1 + erf((logLgal - C) / D))
    return np.repeat(0.55, len(logLgal)) # ~ 3.5 Angstrom EW

def get_gmr_crit(logLgal):
    # See BGS_Study.ipynb g-r section
    # This is for k-corrected G-R
    # Old: A=0.760 B=0.052 C=9.707 D=0.342
    # Fitted parameters: A=0.718, B=0.066, C=9.461, D=0.628
    if logLgal is None:
        return GLOBAL_RED_COLOR_CUT
    A=0.718
    B=0.066
    C=9.461
    D=0.628
    missing = np.isnan(logLgal)
    return np.where(missing, GLOBAL_RED_COLOR_CUT, A + B*(1 + erf((logLgal - C) / D)))

def is_quiescent_SDSS_Dn4000(logLgal, Dn4000):
    """
    Takes in two arrays of log Lgal and Dn4000 and returns an array 
    indicating if the galaxies are quiescent using 2010.02946 eq 1
    """
    Dcrit = get_SDSS_Dcrit(logLgal)
    return Dn4000 > Dcrit

def is_quiescent_BGS_dn4000(logLgal, Dn4000, gmr):
    """
    Takes in two arrays of log Lgal and Dn4000 and returns an array 
    indicating if the galaxies are quiescent using Dn4000 < Dcrit,
    or using g-r color cut when Dn4000 is not available. Also includes
    a very blue cut.
    """
    if Dn4000 is None:
        return is_quiescent_BGS_gmr(logLgal, gmr)
    Dcrit = get_Dn4000_crit(logLgal)
    missing = np.isnan(Dn4000)
    results = np.where(missing, is_quiescent_BGS_gmr(logLgal, gmr), Dn4000 > Dcrit)
    print(f"Quiescent Fraction for Dn4000: {np.mean(results[~missing]):.2%} (N={np.sum(~missing)})")
    print(f"Quiescent Fraction for missing: {np.mean(results[missing]):.2%} (N={np.sum(missing)})")
    very_blue = gmr < EXTREMAL_BLUE_COLOR_CUT
    results = np.where(very_blue, False, results)
    print(f"Overall Quiescent Fraction after very blue cut: {np.mean(results):.2%}")
    return results


def is_quiescent_BGS_dn4000_hardvariant(logLgal, Dn4000, gmr):
    """
    Takes in two arrays of log Lgal and Dn4000 and returns an array 
    indicating if the galaxies are quiescent using Dn4000 < 1.6 
    when Dn4000 is available and using g-r color cut when it is not.
    """
    if Dn4000 is None:
        return is_quiescent_BGS_gmr(logLgal, gmr)
    Dcrit = 1.6
    print(f"Dn4000 missing for {np.mean(np.isnan(Dn4000)):.1%}")
    return np.where(np.isnan(Dn4000), is_quiescent_BGS_gmr(logLgal, gmr), Dn4000 > Dcrit)

def is_quiescent_BGS_kmeans(logLgal, Dn4000, halpha, ssfr, gmr, model=None):

    if halpha is None or ssfr is None:
        print("WARNING: Completely missing Halpha or SSFR; using the Dn4000 or G-R cuts instead of KMeans.")
        return is_quiescent_BGS_dn4000(logLgal, Dn4000, gmr)
    
    results = np.zeros(len(logLgal), dtype=bool)

    # For masked tables, go to inner data so the masked values are nan.
    if np.ma.is_masked(logLgal):
        logLgal = logLgal.data.data
    if np.ma.is_masked(Dn4000):
        Dn4000 = Dn4000.data.data
    if np.ma.is_masked(halpha):
        halpha = halpha.data.data
    if np.ma.is_masked(ssfr):
        ssfr = ssfr.data.data
    if np.ma.is_masked(gmr):
        gmr = gmr.data.data

    # Deal with missing spectroscopic data
    print(f"Missing LOGLGAL data for {np.sum(np.isnan(logLgal))} ({np.mean(np.isnan(logLgal)):.2%})")
    print(f"Missing Dn4000 data for {np.sum(np.isnan(Dn4000))} ({np.mean(np.isnan(Dn4000)):.2%})")
    print(f"Missing Halpha data for {np.sum(np.isnan(halpha))} ({np.mean(np.isnan(halpha)):.2%})")
    print(f"Missing SSFR data for {np.sum(np.isnan(ssfr))} ({np.mean(np.isnan(ssfr)):.2%})")
    missing = np.isnan(logLgal) | np.isnan(Dn4000) | np.isnan(halpha) | np.isnan(ssfr)
    results[missing] = is_quiescent_BGS_gmr(logLgal[missing], gmr[missing])
    print(f"Quiescent Fraction for missing: {np.mean(results[missing]):.2%}")

    # You'd think these extremal ones should be classified as quiescent, but it may just mean the spectra
    # was taken in a location where no Halpha / other SFR indicators were detected. Fall back on Dn4000 method.
    extremal =  (halpha < 1E-10) | (ssfr < 1E-15)
    print(f"Extremal SSFR or Halpha data for {np.mean(extremal):.2%}")
    #results[extremal] = is_quiescent_BGS_gmr(logLgal[extremal], gmr[extremal])
    results[extremal] = is_quiescent_BGS_dn4000(logLgal[extremal], Dn4000[extremal], gmr[extremal])
    print(f"Quiescent Fraction for extremal: {np.mean(results[extremal]):.2%}")

    # Very blue handling. If the g-r color is very blue, call it star-forming regardless of the other properties 
    # which are subject to variance from the location of the fiber.
    if model is not None:
        veryblue = gmr < EXTREMAL_BLUE_COLOR_CUT
        print(f"Very blue data for {np.mean(veryblue):.2%}")
        results[veryblue] = False
        missing  = missing | extremal | veryblue
    else:
        missing  = missing | extremal

    c_halpha = np.log10(halpha[~missing])
    c_halpha = np.where(c_halpha > 3, 3, c_halpha)
    c_halpha = np.where(c_halpha < -3, -3, c_halpha)
    c_halpha = c_halpha - get_halpha_crit(logLgal[~missing])

    c_dn4000 = Dn4000[~missing]
    c_dn4000 = np.where(c_dn4000 < 1.0, 1.0, c_dn4000)
    c_dn4000 = np.where(c_dn4000 > 2.5, 2.5, c_dn4000)
    c_dn4000 = c_dn4000 - get_Dn4000_crit(logLgal[~missing])

    c_ssfr = np.log10(ssfr[~missing])
    c_ssfr = np.where(c_ssfr < -13, -13, c_ssfr)
    c_ssfr = np.where(c_ssfr > -9, -9, c_ssfr)
    c_ssfr = c_ssfr + 11 # -11 is the cut

    c_gmr = gmr[~missing]
    c_gmr = np.where(c_gmr < 0.0, 0.0, c_gmr)
    c_gmr = np.where(c_gmr > 1.3, 1.3, c_gmr)
    c_gmr = c_gmr - get_gmr_crit(logLgal[~missing])

    x = c_dn4000 * 1.0
    y = c_halpha * 0.35 # Trained with 0.27 I believe
    z = c_ssfr * 0.1
    zz = c_gmr * 1.1 # Trained with 1.5 I believe
    data = list(zip(x, y, z, zz))

    # Print off 95% ranges of values, and the difference between the 2.5 and 97.5 percentiles
    print(f"DN4000: {np.percentile(x, 2.5):.3f} to {np.percentile(x, 97.5):.3f} ({np.percentile(x, 97.5) - np.percentile(x, 2.5):.3f})")
    print(f"Halpha: {np.percentile(y, 2.5):.3f} to {np.percentile(y, 97.5):.3f} ({np.percentile(y, 97.5) - np.percentile(y, 2.5):.3f})")
    print(f"SSFR: {np.percentile(z, 2.5):.3f} to {np.percentile(z, 97.5):.3f} ({np.percentile(z, 97.5) - np.percentile(z, 2.5):.3f})")
    print(f"G-R: {np.percentile(zz, 2.5):.3f} to {np.percentile(zz, 97.5):.3f} ({np.percentile(zz, 97.5) - np.percentile(zz, 2.5):.3f})")
    # We want these to all be similar to weight equally in the classification

    kmeans : KMeans = None
    classification : np.ndarray = None

    print(f"Data size: {len(data)}")

    if model is not None:
        # Load the model
        print(f"Loading KMeans model from {model}")
        with open(model, 'rb') as f:
            kmeans = pickle.load(f) 
        # Use the model
        classification = kmeans.predict(data)
    else:
        print(f"Fitting KMeans model")
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(data)
        classification = kmeans.labels_

    # Each run will give 0 or 1 a different meaning. Let's standardize it so 1 means red and 0 is blue
    if kmeans.cluster_centers_[0][0] > kmeans.cluster_centers_[1][0]:
        classification = 1 - classification

    print(f"Classifications: {len(classification)}")
    print(f"results[~missing]: {len(results[~missing])}")
        
    results[~missing] = classification
    print(f"Quiescent Fraction for Kmeans: {np.mean(results[~missing]):.2%}")
    print(f"Quiescent Fraction Overall: {np.mean(results):.2%}")

    # Save off the model
    if model is None:
        with open(QUIESCENT_MODEL, 'wb') as f:
            pickle.dump(kmeans, f)

    return x, y, z, zz, results, missing

def compare_df_quiescence_to_sdss(df: pd.DataFrame, quiescent_col: str):
    sdss = pd.read_csv(SDSS_v1_DAT_FILE, delimiter=' ', names=('RA', 'DEC', 'Z', 'LOGLGAL', 'VMAX', 'QUIESCENT', 'chi'), index_col=False)
    sdss_galprops = pd.read_csv(SDSS_v1_1_GALPROPS_FILE, delimiter=' ', names=('MAG_G', 'MAG_R', 'SIGMA_V', 'DN4000', 'CONCENTRATION', 'LOG_M_STAR', 'Z_ASSIGNED_FLAG'))
    sdss = pd.merge(sdss, sdss_galprops, left_index=True, right_index=True)

    sdss_catalog = coord.SkyCoord(ra=sdss['RA'].to_numpy()*u.degree, dec=sdss['DEC'].to_numpy()*u.degree, frame='icrs')
    BGS_catalog = coord.SkyCoord(ra=df['RA'].to_numpy()*u.degree, dec=df['DEC'].to_numpy()*u.degree, frame='icrs')

    neighbor_indexes, d2d, d3d = coord.match_coordinates_sky(BGS_catalog, sdss_catalog)
    ang_distances = d2d.to(u.arcsec).value

    matched = np.logical_and(ang_distances < 1.0, ~np.isnan(df['DN4000_MODEL']))
    bgs_matches = df.loc[matched,'DN4000_MODEL'].to_numpy()
    sdss_indexes = neighbor_indexes[matched]
    sdss_matches = sdss.iloc[sdss_indexes].DN4000.to_numpy()

    # DN4000 Comparison
    print(f"Dn4000 Value Comparison")
    print(f"{len(bgs_matches)} matches found (with Dn4000 available)")
    print(f"{np.isclose(bgs_matches, sdss_matches, atol=0.05).sum() / len(bgs_matches)} of the matches are within 0.05 of each other.")
    print(f"{np.isclose(bgs_matches, sdss_matches, atol=0.1).sum() / len(bgs_matches)} of the matches are within 0.1 of each other.")
    print(f"{np.isclose(bgs_matches, sdss_matches, atol=0.2).sum() / len(bgs_matches)} of the matches are within 0.2 of each other.")
    print(f"{np.isclose(bgs_matches, sdss_matches, atol=0.3).sum() / len(bgs_matches)} of the matches are within 0.3 of each other.")
    
    difference = np.array(bgs_matches) - np.array(sdss_matches)

    # bin by sdss_matches
    bins = np.linspace(0.8, 2.5, 30)
    digitized = np.digitize(sdss_matches, bins)
    bin_means = [difference[digitized == i].mean() for i in range(1, len(bins))]
    #bin_stds = np.array([difference[digitized == i].std() for i in range(1, len(bins))])
    bin_counts = [np.sum(digitized == i) for i in range(1, len(bins))]

    # Calculate 1 sigma error bars
    bin_intervals1 = [np.percentile(difference[digitized == i], [16, 84]) for i in range(1, len(bins))]
    bin_lows1 = np.array([interval[0] for interval in bin_intervals1])
    bin_highs1 = np.array([interval[1] for interval in bin_intervals1])
    # Calculate 2 sigma error bars
    bin_intervals = [np.percentile(difference[digitized == i], [2.5, 97.5]) for i in range(1, len(bins))]
    bin_lows = np.array([interval[0] for interval in bin_intervals])
    bin_highs = np.array([interval[1] for interval in bin_intervals])
    # Convert yerr to a format that plt.errorbar can handle
    yerr_1sigma = np.array([bin_means - bin_lows1, bin_highs1 - bin_means])
    yerr_2sigma = np.array([bin_means - bin_lows, bin_highs - bin_means])

    plt.errorbar(bins[1:], bin_means, yerr=yerr_1sigma, fmt='o', label='Mean Difference (1 sigma)', color='k')
    plt.errorbar(bins[1:], bin_means, yerr=yerr_2sigma, fmt='o', label='Mean Difference (2 sigma)', alpha=0.5)
    plt.xlabel('SDSS Dn4000')
    plt.ylabel('BGS - SDSS Dn4000')
    plt.legend()
    plt.title("Shared Targets: Dn4000 Comparison")
    #draw horizontal line at 0.0
    plt.axhline(0.0, color='r', linestyle='--')
    

    # Compare quescient evaluation to SDSS for matched ones
    bgs_quiescent_matched = df.loc[matched, quiescent_col].to_numpy()
    print(len(bgs_quiescent_matched))

    sdss_indexes_q = neighbor_indexes[matched]
    sdss_quiescent_matched = sdss.iloc[sdss_indexes_q]['QUIESCENT'].to_numpy()

    # Percent that agree on quiescent classification
    print(f"\nQuiescent Classification Comparison")
    print(f"{np.sum(bgs_quiescent_matched == sdss_quiescent_matched) / len(bgs_quiescent_matched):.1%} of the matched galaxies agree on quiescent classification using {quiescent_col}.")

    # Percent that agree as a function of L


# This is read off of a 0.1^G-R plot I made using GAMA polynomial k-corr
# This also works well for MXXL
# TODO check this value after switching to DESI k-corr
GLOBAL_RED_COLOR_CUT = 0.76 
GLOBAL_RED_COLOR_CUT_NO_KCORR = 1.008
EXTREMAL_BLUE_COLOR_CUT = 0.65

#BGS_LOGLGAL_BINS = [6.9, 9.0, 9.4, 9.7, 9.9, 10.1, 10.3, 10.7, 13.5]
#BGS_LOGLGAL_BINS = [6.9, 8.7, 9.1, 9.4, 9.7, 9.9, 10.1, 10.3, 10.7, 13.5]
BGS_LOGLGAL_BINS = [6.5, 8.4, 8.8, 9.1, 9.4, 9.7, 9.9, 10.1, 10.3, 10.7, 13.5]

def is_quiescent_lost_gal_guess(gmr):
    # This better midpoint for g-r without k-corr. 
    # It recovers the same red/blue fraction in BGS Y1 Observed gals as GLOBAL_RED_COLOR_CUT for k-corrected ones.
    return gmr > GLOBAL_RED_COLOR_CUT_NO_KCORR

def is_quiescent_BGS_gmr(logLgal, G_R_k):
    """
    Takes in two arrays of log Lgal and 0.1^(G-R) and returns an array
    indicating if the galaxies are quiescent using G-R color cut
    from the BGS Y1 data.

    True for quiescent, False for star-forming.
    """
    # g-r: both are in mags, so more negative values of each are greater fluxes in those bands
    # So a more positive g-r means a redder galaxy

    crit = get_gmr_crit(logLgal)
    return G_R_k > crit

    #return G_R_k > GLOBAL_RED_COLOR_CUT




###################################################################
# MCMC Log Processing
###################################################################


def fsat_variance_from_saved():
    """
    Load the fsat variance from the saved file.
    """
    if os.path.exists(FSAT_VALUES_FROM_LOGS):
        fsat_arr, fsatr_arr, fsatb_arr = np.load(FSAT_VALUES_FROM_LOGS)
        fsat_std = np.std(fsat_arr, axis=0)
        fsatr_std = np.std(fsatr_arr, axis=0)
        fsatb_std = np.std(fsatb_arr, axis=0)
        fsat_mean = np.mean(fsat_arr, axis=0)
        fsatr_mean = np.mean(fsatr_arr, axis=0)
        fsatb_mean = np.mean(fsatb_arr, axis=0)
        return fsat_std, fsatr_std, fsatb_std, fsat_mean, fsatr_mean, fsatb_mean
    else:
        print(f"WARNING: {FSAT_VALUES_FROM_LOGS} does not exist. Cannot load variance. Call extract_variance_from_log(path) to create it.")
        return None
    
def lhmr_variance_from_saved():
    """
    Load the lhmr variance from the saved file.
    """
    if os.path.exists(LHMR_VALUES_FROM_LOGS):
        r_arr, r_scatter_arr, b_arr, b_scatter_arr, all_arr, all_scatter_arr = np.load(LHMR_VALUES_FROM_LOGS)
        print(r_arr)
        r_std = np.std(r_arr, axis=0)
        r_scatter_std = np.std(r_scatter_arr, axis=0)
        b_std = np.std(b_arr, axis=0)
        b_scatter_std = np.std(b_scatter_arr, axis=0)
        all_std = np.std(all_arr, axis=0)
        all_scatter_std = np.std(all_scatter_arr, axis=0)
        r_mean = np.mean(r_arr, axis=0)
        r_scatter_mean = np.mean(r_scatter_arr, axis=0)
        b_mean = np.mean(b_arr, axis=0)   
        b_scatter_mean = np.mean(b_scatter_arr, axis=0)
        all_mean = np.mean(all_arr, axis=0)
        all_scatter_mean = np.mean(all_scatter_arr, axis=0)

        return (r_mean,r_std,r_scatter_mean,r_scatter_std,b_mean,b_std,b_scatter_mean,b_scatter_std,all_mean,all_std,all_scatter_mean,all_scatter_std)
    else:
        print(f"WARNING: {LHMR_VALUES_FROM_LOGS} does not exist. Cannot load variance. Call extract_variance_from_log(path) to create it.")
        return None

def save_from_log(logpath: str, overwrite=False):
    """
    Extracts f_sat and LHMR from a log file and uses them to compute means and stds of these quantities.
    Intended for use with a log file from an MCMC run.
    """
    FSAT_BINCOUNT = len(L_gal_bins) 
    fsat_len = 0

    LHMR_BINCOUNT = 155 - 90 # See fit_clustering_omp.c "hod> " 
    lhmr_len = 0

    with open(logpath) as fp:

        # First get total number
        for line in fp:
            if line.startswith("fsat> 0 "):
                fsat_len += 1
            if line.startswith("LHMR> 9.0"):
                lhmr_len += 1
        print(f"fsat_len: {fsat_len}, lhmr_len: {lhmr_len}")

        # Prep arrays
        fsat_arr = np.zeros((fsat_len, FSAT_BINCOUNT))
        fsatr_arr = np.zeros((fsat_len, FSAT_BINCOUNT))
        fsatb_arr = np.zeros((fsat_len, FSAT_BINCOUNT))

        lhmr_r_arr = np.zeros((lhmr_len, LHMR_BINCOUNT))
        lhmr_r_scatter_arr = np.zeros((lhmr_len, LHMR_BINCOUNT))
        lhmr_b_arr = np.zeros((lhmr_len, LHMR_BINCOUNT))
        lhmr_b_scatter_arr = np.zeros((lhmr_len, LHMR_BINCOUNT))
        lhmr_all_arr = np.zeros((lhmr_len, LHMR_BINCOUNT))
        lhmr_all_scatter_arr = np.zeros((lhmr_len, LHMR_BINCOUNT))

        # Then go back to the beginning and read the values
        fp.seek(0)
        fsat_row = 0  # Track the current row in the arrays
        lhmr_row = 0  # Track the current row in the arrays

        for line in fp:
            if line.startswith("fsat> "):
                if fsat_row < fsat_len:
                    values = line.split(' ')
                    bin_i = int(values[1].strip())
                    fsat_arr[fsat_row, bin_i] = float(values[2].strip())
                    fsatr_arr[fsat_row, bin_i] = float(values[3].strip())
                    fsatb_arr[fsat_row, bin_i] = float(values[4].strip())

                    # Increment the row counter when the last bin is processed
                    if bin_i == FSAT_BINCOUNT - 1:
                        fsat_row += 1
            
            # Format is: <log10(M_h)> <mean_cenr> <std_cenr> <mean_cenb> <std_cenb> <mean_cen> <std_cen>
            if line.startswith("LHMR> "):
                if lhmr_row < lhmr_len:
                    values = line.split(' ')
                    bin_i = int(float(values[1].strip())*10) - 90 # 9.0 becomes 0
                    lhmr_r_arr[lhmr_row, bin_i] = float(values[2].strip())
                    lhmr_r_scatter_arr[lhmr_row, bin_i] = float(values[3].strip())
                    lhmr_b_arr[lhmr_row, bin_i] = float(values[4].strip())
                    lhmr_b_scatter_arr[lhmr_row, bin_i] = float(values[5].strip())
                    lhmr_all_arr[lhmr_row, bin_i] = float(values[6].strip())
                    lhmr_all_scatter_arr[lhmr_row, bin_i] = float(values[7].strip())

                    # Increment the row counter when the last bin is processed
                    if bin_i == LHMR_BINCOUNT - 1:
                        lhmr_row += 1

    # Save off the arrays of values
    if os.path.exists(FSAT_VALUES_FROM_LOGS):
        if overwrite:
            print(f"WARNING: {FSAT_VALUES_FROM_LOGS} already exists. Overwriting.")
            np.save(FSAT_VALUES_FROM_LOGS, (fsat_arr, fsatr_arr, fsatb_arr))
        else:
            # Don't overwrite
            print(f"WARNING: {FSAT_VALUES_FROM_LOGS} already exists. Will not overwrite.")
    else:
        np.save(FSAT_VALUES_FROM_LOGS, (fsat_arr, fsatr_arr, fsatb_arr))

    if os.path.exists(LHMR_VALUES_FROM_LOGS):
        if overwrite:
            print(f"WARNING: {LHMR_VALUES_FROM_LOGS} already exists. Overwriting.")
            np.save(LHMR_VALUES_FROM_LOGS, (lhmr_r_arr, lhmr_r_scatter_arr, lhmr_b_arr, lhmr_b_scatter_arr, lhmr_all_arr, lhmr_all_scatter_arr))
        else:
            # Don't overwrite
            print(f"WARNING: {LHMR_VALUES_FROM_LOGS} already exists. Will not overwrite.")
    else:
        np.save(LHMR_VALUES_FROM_LOGS, (lhmr_r_arr, lhmr_r_scatter_arr, lhmr_b_arr, lhmr_b_scatter_arr, lhmr_all_arr, lhmr_all_scatter_arr))

    # Calculate the std, mean
    fsat_std = np.std(fsat_arr, axis=0)
    fsatr_std = np.std(fsatr_arr, axis=0)
    fsatb_std = np.std(fsatb_arr, axis=0)
    fsat_mean = np.mean(fsat_arr, axis=0)
    fsatr_mean = np.mean(fsatr_arr, axis=0)
    fsatb_mean = np.mean(fsatb_arr, axis=0)

    lhmr_r_std = np.std(lhmr_r_arr, axis=0)
    lhmr_r_scatter_std = np.std(lhmr_r_scatter_arr, axis=0)
    lhmr_b_std = np.std(lhmr_b_arr, axis=0)
    lhmr_b_scatter_std = np.std(lhmr_b_scatter_arr, axis=0)
    lhmr_all_std = np.std(lhmr_all_arr, axis=0)
    lhmr_all_scatter_std = np.std(lhmr_all_scatter_arr, axis=0)
    lhmr_r_mean = np.mean(lhmr_r_arr, axis=0)
    lhmr_r_scatter_mean = np.mean(lhmr_r_scatter_arr, axis=0)
    lhmr_b_mean = np.mean(lhmr_b_arr, axis=0)   
    lhmr_b_scatter_mean = np.mean(lhmr_b_scatter_arr, axis=0)
    lhmr_all_mean = np.mean(lhmr_all_arr, axis=0)
    lhmr_all_scatter_mean = np.mean(lhmr_all_scatter_arr, axis=0)

    # TODO Lsat?

    return (fsat_mean, 
        fsat_std, 
        fsatr_mean,
        fsatr_std,
        fsatb_mean,
        fsatb_std,
        lhmr_r_mean,
        lhmr_r_std,
        lhmr_r_scatter_mean,
        lhmr_r_scatter_std,
        lhmr_b_mean,
        lhmr_b_std,
        lhmr_b_scatter_mean,
        lhmr_b_scatter_std,
        lhmr_all_mean,
        lhmr_all_std,
        lhmr_all_scatter_mean,
        lhmr_all_scatter_std)

