import numpy as np
import astropy.units as u
import matplotlib as plt
from astropy.cosmology import FlatLambdaCDM
import astropy.coordinates as coord
import time
from enum import Enum
from scipy import special
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Circle
import pandas as pd
import sys
import multiprocessing as mp
from scipy.special import erf
import os

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

POBS_BIN_COUNT = 15
POBS_BINS = np.linspace(0.01, 1.01, POBS_BIN_COUNT) # upper bound is higher than any data, lower bound is not
POBS_BINS_MIDPOINTS = np.append([0.0], 0.5*(POBS_BINS[1:] + POBS_BINS[:-1]))

APP_MAG_BIN_COUNT = 12
APP_MAG_BINS = np.linspace(15.5, 20.176, APP_MAG_BIN_COUNT) # upper bound is higher than any data, lower bound is not
APP_MAG_BINS_MIDPOINTS = np.append([15.0], 0.5*(APP_MAG_BINS[1:] + APP_MAG_BINS[:-1]))

ANG_DIST_BIN_COUNT = 15
ANGULAR_BINS = np.append(np.logspace(np.log10(3), np.log10(900), ANG_DIST_BIN_COUNT - 1), 3600) # upper bound is higher than any data, lower bound is not
ANGULAR_BINS_MIDPOINTS = np.append([2], 0.5*(ANGULAR_BINS[1:] + ANGULAR_BINS[:-1]))

Z_BIN_COUNT = 16
Z_BINS = np.linspace(0.05, 0.5, Z_BIN_COUNT) # upper bound is higher than any data, lower bound is not

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

################################

DEGREES_ON_SPHERE = 41253

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
        return value == Mode.PHOTOZ_PLUS_v1.value or value == Mode.PHOTOZ_PLUS_v2.value or value == Mode.PHOTOZ_PLUS_v3.value
        
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

def z_to_ldist(zs):
    """
    Gets the luminosity distance of the provided redshifts in Mpc.
    """
    return _cosmo_h.luminosity_distance(zs).value
    
def distmod(zs):
    return 5 * (np.log10(_cosmo_h.luminosity_distance(zs).value * 1E6) - 1)

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





######################################
# Color Cuts
######################################

def get_SDSS_Dcrit(logLgal):
    return 1.42 + (0.35 / 2) * (1 + special.erf((logLgal - 9.9) / 0.8))

# A=1.411, B=0.171, C=9.795, D=0.777
# Fitted parameters: A=1.411, B=0.171, C=9.795, D=0.775
# Very similar fit comapred to SDSS, might as well keep it the same and just use SDSS_Dcrit
def get_ian_Dcrit(logLgal):
    return 1.411 + (0.171) * (1 + special.erf((logLgal - 9.795) / 0.775))

def is_quiescent_SDSS_Dn4000(logLgal, Dn4000):
    """
    Takes in two arrays of log Lgal and Dn4000 and returns an array 
    indicating if the galaxies are quiescent using 2010.02946 eq 1
    """
    Dcrit = get_SDSS_Dcrit(logLgal)
    return Dn4000 > Dcrit

def is_quiescent_BGS_smart(logLgal, Dn4000, gmr):
    """
    Takes in two arrays of log Lgal and Dn4000 and returns an array 
    indicating if the galaxies are quiescent using 2010.02946 eq 1
    when Dn4000 is available and using g-r color cut when it is not.
    """
    if Dn4000 is None:
        return is_quiescent_BGS_gmr(logLgal, gmr)
    Dcrit = get_SDSS_Dcrit(logLgal)
    print(f"Dn4000 missing for {np.mean(np.isnan(Dn4000)):.1%}") # Broken?
    return np.where(np.isnan(Dn4000), is_quiescent_BGS_gmr(logLgal, gmr), Dn4000 > Dcrit)

def is_quiescent_BGS_smart_hardvariant(logLgal, Dn4000, gmr):
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

# This is read off of a 0.1^G-R plot I made using GAMA polynomial k-corr
# This also works well for MXXL
# TODO check this value after switching to DESI k-corr
GLOBAL_RED_COLOR_CUT = 0.76 
GLOBAL_RED_COLOR_CUT_NO_KCORR = 1.008

# Turns out binning by logLGal doesn't change much
# TODO after swithcing to DESI k-corrections, ensure this is still true
BGS_LOGLGAL_BINS = [6.9, 9.0, 9.4, 9.7, 9.9, 10.1, 10.3, 10.7, 13.5]
BINWISE_RED_COLOR_CUT = [0.76, 0.76, 0.77, 0.79, 0.77, 0.76, 0.76, 0.76, 0.76, 0.76]

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

    # This is the alternative implementation for using bins
    #logLgal_idx = np.digitize(logLgal, BGS_LOGLGAL_BINS)
    #per_galaxy_red_cut = BINWISE_RED_COLOR_CUT[logLgal_idx]
    #return gmr < per_galaxy_red_cut

    # g-r: both are in mags, so more negative values of each are greater fluxes in those bands
    # So a more positive g-r means a redder galaxy
    return G_R_k > GLOBAL_RED_COLOR_CUT

