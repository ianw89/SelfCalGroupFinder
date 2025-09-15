import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf
import emcee

# --- Define Model Functions ---
#def hod_central_model(logM, logM_min, sigma_logM):
#    """ Standard HOD model for central galaxies. """
#    return np.log10(0.5 * (1 + erf((logM - logM_min) / sigma_logM)))

def hod_central_model2(logM, logM_min, sigma_logM1, logM_max, sigma_logM2):
    """ Standard HOD model for central galaxies. """
    result = (0.5 * (1 + erf((logM - logM_min) / sigma_logM1)) - 0.5 * (1 + erf((logM - logM_max) / sigma_logM2)))
    result = np.clip(result, 1e-6, None)  # Avoid log of zero or negative
    return np.log10(result)

#def hod_central_model_cutoff(logM, logM_min, sigma_logM, logM_cut, kappa):
#    """ Standard HOD model for central galaxies. """
#    M = 10**logM
#    M_cut = 10**logM_cut
#    return np.log10(0.5 * (1 + erf((logM - logM_min) / sigma_logM)) * np.exp(-(M / M_cut)**kappa))

def hod_satellite_model(logM, logM_cut, logM_1, alpha):
    """ Standard HOD model for satellite galaxies. """
    M = 10**logM
    M_cut = 10**logM_cut
    M_1 = 10**logM_1
    base = np.fmax((M - M_cut) / (M_1+1e-6), 1e-6) 
    return np.log10((base)**alpha)

def fit_hod_models(log_halo_mass, logncen, lognsat, use_mcmc=True):
    """
    Fits HOD data to standard models using either curve_fit or MCMC.
    """
    logncen = np.clip(logncen, -6, 2)
    lognsat = np.clip(lognsat, -6, 5)
    
    # --- Centrals ---
    cenmask = logncen > -3
    p0_cen = [log_halo_mass[cenmask][0], 0.2, log_halo_mass[cenmask][-1], 0.7]
    bounds_cen = ([8, 0.1, 9, 0.1], [17, 3.0, 20, 3.0])
    
    # --- Satellites ---
    satmask = lognsat > -3
    m1_guess = len(log_halo_mass[satmask]) // 2
    p0_sat = [log_halo_mass[satmask][0], log_halo_mass[satmask][m1_guess], 1.0]
    bounds_sat = ([9, 9.5, 0.1], [17, 20, 3.0])

    if use_mcmc:
        print("--- Fitting Centrals with MCMC ---")
        popt_cen = fit_hod_mcmc(log_halo_mass[cenmask], logncen[cenmask], hod_central_model2, p0_cen, bounds_cen)
        print("\n--- Fitting Satellites with MCMC ---")
        popt_sat = fit_hod_mcmc(log_halo_mass[satmask], lognsat[satmask], hod_satellite_model, p0_sat, bounds_sat)
    else:
        print("--- Fitting with curve_fit ---")
        popt_cen, _ = curve_fit(hod_central_model2, log_halo_mass[cenmask], logncen[cenmask], p0=p0_cen, bounds=bounds_cen)
        popt_sat, _ = curve_fit(hod_satellite_model, log_halo_mass[satmask], lognsat[satmask], p0=p0_sat, bounds=bounds_sat)

    return popt_cen, popt_sat

def _log_prior(theta, lower_bounds, upper_bounds):
    """Priors are defined by the bounds. Returns -inf if outside."""
    for i in range(len(theta)):
        if not (lower_bounds[i] < theta[i] < upper_bounds[i]):
            return -np.inf
    return 0.0

def _log_likelihood(theta, x, y, yerr, model_func):
    """Gaussian log-likelihood, assuming model and data are in log space."""
    model_y = model_func(x, *theta)
    sigma2 = yerr**2
    return -0.5 * np.sum((y - model_y)**2 / sigma2)

def _log_probability(theta, x, y, yerr, model_func, lower_bounds, upper_bounds):
    """The full log-probability function for emcee."""
    lp = _log_prior(theta, lower_bounds, upper_bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _log_likelihood(theta, x, y, yerr, model_func)


def fit_hod_mcmc(log_halo_mass, logn, model_func, p0, bounds, y_err=0.1, nwalkers=12, nsteps=10000, discard=100):
    """
    Fits a model to data using MCMC with emcee.

    Parameters
    ----------
    log_halo_mass : array
        The x-data (log10 halo mass).
    logn : array
        The y-data to be fit (log10 occupation).
    model_func : function
        The model function to fit (e.g., hod_central_model2).
    p0 : list
        Initial guess for the parameters.
    bounds : tuple
        A tuple of (lower_bounds, upper_bounds) for the parameters.
    y_err : float, optional
        The uncertainty on the y-data points.
    nwalkers : int, optional
        The number of MCMC walkers.
    nsteps : int, optional
        The number of steps for each walker.
    discard : int, optional
        The number of "burn-in" steps to discard.

    Returns
    -------
    array
        The median of the posterior distributions for each parameter.
    """
    ndim = len(p0)
    lower_bounds, upper_bounds = bounds

    # Initialize walkers in a small ball around the initial guess
    pos = np.array(p0) + 1e-4 * np.random.randn(nwalkers, ndim)

    # Pass the extra arguments needed by the global _log_probability function.
    sampler_args = (log_halo_mass, logn, y_err, model_func, lower_bounds, upper_bounds)

    # Set up and run the sampler
    #with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, _log_probability, args=sampler_args)
    sampler.run_mcmc(pos, nsteps, progress=True)

    # Get the results, discarding the burn-in phase
    flat_samples = sampler.get_chain(discard=discard, flat=True)
    
    # Return the best fit model, not the median
    best_idx = np.argmax(sampler.get_log_prob(discard=discard, flat=True))
    best_params = flat_samples[best_idx]
    return best_params



