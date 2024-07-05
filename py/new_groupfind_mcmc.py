# library
import numpy as np
import matplotlib.pyplot as plt
import csv as csv
import sys
import scipy
from scipy.optimize import minimize
import os
import pickle
from astropy.io import ascii
import emcee
import time
from multiprocessing import Pool
import subprocess as sp
import pathlib

#--------------------------
# stepping from line input
#--------------------------
galdata_file = sys.argv[1]
cwd = pathlib.Path().resolve()
print(cwd)

#-- globals --
# TODO these are copied from Jeremy's code, need to understand them
niter = 0
#volume = [ 3.181e+05, 1.209e+06, 4.486e+06, 1.609e+07, 5.517e+07 ] # if including -17
#volume = [ 1.209e+06, 4.486e+06, 1.609e+07, 5.517e+07 ] #if starting at -18
volume = [  1.721e+06, 6.385e+06, 2.291e+07, 7.852e+07 ] # actual SDSS
volume = np.array(volume)
vfac = (volume/250.0**3)**.5 # factor by which to multiply errors
efac = 0.1 # let's just add a constant fractional error bar
   
# if we're doing MXXL (14k)
#vfac = vfac*3**.5

def chisqr(params):
    GF_props = {
        'zmin':0,
        'zmax':1.0,
        'frac_area':0.179,
        'fluxlim':1,
        'color':1,
        'omegaL_sf':params[0],
        'sigma_sf':params[1],
        'omegaL_q':params[2],
        'sigma_q':params[3],
        'omega0_sf':params[4],  
        'omega0_q':params[5],    
        'beta0q':params[6],    
        'betaLq':params[7],
        'beta0sf':params[8],
        'betaLsf':params[9],
        #'omega_chi_0_sf':params[10],  
        #'omega_chi_0_q':params[11],
        #'omega_chi_L_sf':params[12],
        #'omega_chi_L_q':params[13],
    }
    global ncount
    ncount = ncount + 1

    os.system('mv outxx outyy')

    # Run the Group Finder (which also populates mock)
    # subprocess.run will wait for completion
    args = ['./kdGroupFinder_omp', galdata_file, *list(map(str,GF_props.values()))]
    args = ['./kdGroupFinder_omp', galdata_file]
    args.append(str(GF_props['zmin']))
    args.append(str(GF_props['zmax']))
    args.append(str(GF_props['frac_area']))
    if GF_props['fluxlim'] == 1:
        args.append("-f")
    if GF_props['color'] == 1:
        args.append("-c")
    args.append(f"--wcen={GF_props['omegaL_sf']},{GF_props['sigma_sf']},{GF_props['omegaL_q']},{GF_props['sigma_q']},{GF_props['omega0_sf']},{GF_props['omega0_q']}")
    args.append(f"--bsat={GF_props['beta0q']},{GF_props['betaLq']},{GF_props['beta0sf']},{GF_props['betaLsf']}")
    if GF_props.get('omega_chi_0_sf') is not None:
        args.append(f"--chi1={GF_props['omega_chi_0_sf']},{GF_props['omega_chi_0_q']},{GF_props['omega_chi_L_sf']},{GF_props['omega_chi_L_q']}")            

    print(args)
    f = open('outxx', 'w')
    sp.run(args, cwd=cwd, stdout=f)

    # run the clustering sript
    os.system('./sh.wp')
    # read in the wp values from the results
    imag = np.linspace(18,21,4,dtype='int')
    #imag = np.linspace(90,110,5,dtype='int')
    chi = 0
    chi_contributions = []
    ii = 0 # index for vfac

    for i in imag:
        fname='wp_red_M'+"{:d}".format(i)+'.dat'
        data = ascii.read(fname, delimiter='\s', format='no_header')
        xid = np.array(data['col2'][...], dtype='float')
        errd = np.array(data['col3'][...], dtype='float')
        rad = np.array(data['col1'][...], dtype='float')

        fname='wp_mock_red_M'+"{:2}".format(i)+'.dat'
        data = ascii.read(fname, delimiter='\s', format='no_header')
        xim = np.array(data['col5'][...], dtype='float')

        errm = vfac[ii]*errd + efac*xim
        chivec = (xim-xid)**2/(errd**2 + errm**2) 
        chi_contributions.append(np.sum(chivec))
        #print 'XX ', niter, i, np.sum(chivec)

        #j = 0
        #for rr in rad:
            #print 'WPR',niter,i, rr, xim[j], xid[j], errd[j]
        #    j = j + 1
        
        fname='wp_blue_M'+"{:2}".format(i)+'.dat'
        data = ascii.read(fname, delimiter='\s', format='no_header')
        xid = np.array(data['col2'][...], dtype='float')
        errd = np.array(data['col3'][...], dtype='float')

        fname='wp_mock_blue_M'+"{:2}".format(i)+'.dat'
        data = ascii.read(fname, delimiter='\s', format='no_header')
        xim = np.array(data['col5'][...], dtype='float')

        errm = vfac[ii]*errd + efac*xim
        ii = ii + 1
        chivec = (xim-xid)**2/(errd**2 + errm**2) 
        #print 'XX ', niter, i,  np.sum(chivec)
        chi_contributions.append(np.sum(chivec))
        # #print out the wp result
        #j = 0
        #for rr in rad:
            #print 'WPB',niter, i, rr, xim[j], xid[j], errd[j]
        #    j = j + 1
    
    # now get the mean lsat
    fname = "Lsat_SDSS_DnGMM.dat"
    data = ascii.read(fname, delimiter='\s', format='no_header')
    fr = np.array(data['col2'][...], dtype='float')
    er = np.array(data['col3'][...], dtype='float')
    fb = np.array(data['col4'][...], dtype='float')
    eb = np.array(data['col5'][...], dtype='float')
    
    df = np.log(fr/fb)
    ef = ((er/fr)**2 + (eb/fb)**2)**.5
    # add to the ef to include error from the popsim
    # ef = ef
        
    # get the model result
    fname = "lsat_groups.out"
    data = ascii.read(fname, delimiter='\s', format='no_header')
    m = np.array(data['col1'][...], dtype='float')
    fr = np.array(data['col2'][...], dtype='float')
    fb = np.array(data['col3'][...], dtype='float')

    dfm = (fr-fb)*np.log(10)
    chivec = (df-dfm)**2/ef**2
    chi_contributions.append(np.sum(chivec))

    #j = 0
    #for mm in m:
        #print 'LSAT',niter, mm, df[j], ef[j], dfm[j]
    #    j = j + 1

    # This is for the second parameter (galaxy concentration)    
    """
    # now do lsat vs second parameter BLUE
    fname = "lsat_sdss_con.dat"
    data = ascii.read(fname, delimiter='\s', format='no_header')
    y = np.array(data['col2'][...], dtype='float')
    e = np.array(data['col3'][...], dtype='float')

    fname = "lsat_groups_propx_blue.out"
    data = ascii.read(fname, delimiter='\s', format='no_header')
    m = np.array(data['col2'][...], dtype='float')
    
    em = m*(e/y)
    chivec = (y-m)**2/(e**2+em**2)
    chi = chi + np.sum(chivec)
        
    # now do lsat vs second parameter RED
    fname = "lsat_sdss_con.dat"
    data = ascii.read(fname, delimiter='\s', format='no_header')
    y = np.array(data['col4'][...], dtype='float')
    e = np.array(data['col5'][...], dtype='float')

    fname = "lsat_groups_propx_red.out"
    data = ascii.read(fname, delimiter='\s', format='no_header')
    m = np.array(data['col2'][...], dtype='float')

    em = m*(e/y)
    chivec = (y-m)**2/(e**2+em**2)
    chi = chi + np.sum(chivec)
    """

    chi = np.sum(chi_contributions)

    # Print off the chi squared value and model info and return it 
    print(f'MODEL {ncount}')
    print(GF_props)
    print(f'CHI {chi}')
    print(f'CONTRIBUTIONS {chi_contributions}')
    os.system('date')
    sys.stdout.flush()

    return chi


# --- log-likelihood
def lnlike(theta):
    LnLike = -0.5*chisqr(theta)
    return LnLike

# --- set the priors (no priors right now)
def lnprior(theta):
    if (1): 
        return 0.0
    else:
        return -np.inf

# -- combine the two above
def lnprob(theta):
    lp = lnprior(theta)
    if np.isinf(lp): #check if lp is infinite
        return -np.inf
    return lp + lnlike(theta) #recall if lp not -inf, its 0, so this just returns likelihood


# ---- get things set up
ncount = 1 # might work 
nwalkers = 30
niter = 40000
   
#initial = np.array([1.312225e+01, 2.425592e+00, 1.291072e+01, 4.857720e+00, 1.745350e+01, 2.670356e+00, -9.231342e-01, 1.028550e+01, 1.301696e+01, -8.029334e+00, 2.689616e+00, 1.102281e+00, 2.231206e+00, 4.823592e-01])
initial = np.array([1.312225e+01, 2.425592e+00, 1.291072e+01, 4.857720e+00, 1.745350e+01, 2.670356e+00, -9.231342e-01, 1.028550e+01, 1.301696e+01, -8.029334e+00])

# These are the values mentioned on the webpage
#prev_best = np.array([13.1,2.42,12.9,4.84,17.4,2.67,-0.92,10.25,12.993,-8.04])

spread_factor = 0.2
ndim = len(initial)
p0 = np.array(initial)*(1 + spread_factor*np.random.randn(ndim))

# -- set up the backend
fname = f'mcmc_{int(sys.argv[2])}.dat'
backend_exists = os.path.isfile(fname)
print(f'BACKEND ALREADY EXISTS: {backend_exists}')

backend = emcee.backends.HDFBackend(fname)
#backend.reset(nwalkers, ndim)

# -- run emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend)
    
if backend_exists:
    # Continue where we left off
    pos, prob, state = sampler.run_mcmc(None, niter, progress=True)
else:
    # Start fresh using the above ICs
    print(p0)
    pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)


print("================================")
print("END OF groupfind_mcmc3.py SCRIPT")
print("================================")
