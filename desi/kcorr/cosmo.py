import astropy
import numpy as np
import astropy.units as u

from   astropy.cosmology       import  FlatLambdaCDM

# setting cosmological parameters
# check that this is reasonable.
h     = 1.
cosmo = FlatLambdaCDM(H0=100*h * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0= 0.25)

cosmo = FlatLambdaCDM(H0=100, Om0=0.313, Tcmb0=2.725)   #Standard Planck Cosmology in Mpc/h units


# abacus parameters:
# Planck 2018 base_plikHM_TTTEEE_lowl_lowE_lensing mean
# see 2.17 in Baseline params table for Planck 2018.

#h = 0.6736
#h = 1
#cosmo = FlatLambdaCDM(H0=100*h * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0= 0.3153)
#print('WARNING: USING ABACUS COSMOLOGY')

def fsky(area_sqdeg):
    return area_sqdeg / (360.**2. / np.pi)

def negz_proof(func):
    '''
    Decorator to handle negative redshifts.
    '''
  
    def wrap(zs):
        if type(zs) in [np.ndarray, astropy.table.column.Column]:
            scalar = False

        else:
            try:
                scalar = len(zs) > 1
                scalar = ~scalar

            except:
                # print('Assuming scalar type for neg. z wrapping of type {}'.format(type(zs)))

                zs = np.atleast_1d(zs)
                scalar = True

        zs = np.copy(zs)
        zs = np.atleast_1d(zs)

        negz     = zs <= 0.0
        
        zs[negz] = 1.e-99

        result   = func(zs)
        result[negz] = np.nan
 
        if scalar:
            result = result[0]

        return result

    return wrap




@negz_proof
def distcom(zs):
    return cosmo.comoving_distance(zs).value * cosmo.h

@negz_proof
def distmod(zs):
    return 5. * np.log10((1.+zs)*distcom(zs)) + 25.

def volcom(zs, area):
    return (4./3.) * np.pi * fsky(area) * distcom(zs)**3.

    

if __name__ == '__main__':
    zs  = np.arange(-10., 10., 1.)
    mus = distmod(zs)

    for z, mu in zip(zs, mus):
        print('{:.6f}\t{:.6f}'.format(z, mu))
