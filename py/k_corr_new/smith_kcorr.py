import os
import numpy as np
import matplotlib.pyplot as plt
import os

from   scipy.interpolate import interp1d
from   pkg_resources     import resource_filename


#raw_dir = os.environ['PYTHONPATH']


class DESI_KCorrection(object):
    def __init__(self, band, kind="cubic", file='jmext', photsys=None, suffix=''):
        """
        Colour-dependent polynomial fit to the FSF DESI K-corrections, 
        used to convert between SDSS r-band Petrosian apparent magnitudes, and rest 
        frame absolute manigutues at z_ref = 0.1
        
        Args:
            k_corr_file: file of polynomial coefficients for each colour bin
            z0: reference redshift. Default value is z0=0.1
            kind: type of interpolation between colour bins,
                  e.g. "linear", "cubic". Default is "cubic"
        """
        
        import os
        os.environ['CODE_ROOT'] = '/global/u2/l/ldrm11/Cole/'
        raw_dir = os.environ['CODE_ROOT'] + '/data/'        
        #print(os.environ['CODE_ROOT'])
        
        # check pointing to right directory.
        # print('FILE:', file)
        # print('PHOTSYS:', photsys)
        
        if file == 'ajs':
            k_corr_file = raw_dir + '/ajs_kcorr_{}band_z01.dat'.format(band.lower())
                 
        elif file == 'jmext':
            k_corr_file = raw_dir + '/jmext_kcorr_{}_{}band_z01{}.dat'.format(photsys.upper(), band.lower(), suffix)
            
        elif file == 'jmextcol':
            # WARNING: These are the colour polynomials.
            # TODO: separate this to avoid ambiguity.
            k_corr_file = raw_dir + '/jmextcol_kcorr_{}_{}band_z01.dat'.format(photsys.upper(), band.lower())
            
            
        else:
            print('FILE NOT SUPPORTED.')
                    
        # read file of parameters of polynomial fit to k-correction
        # polynomial k-correction is of the form
        # A*(z-z0)^6 + B*(z-z0)^5 + C*(z-z0)^4 + D*(z-z0)^3 + ... + G
        col_min, col_max, A, B, C, D, E, F, G, col_med = \
            np.loadtxt(k_corr_file, unpack=True)
    
        self.z0 = 0.1            # reference redshift

        self.nbins = len(col_min) # number of colour bins in file
        self.colour_min = np.min(col_med)
        self.colour_max = np.max(col_med)
        self.colour_med = col_med

        # functions for interpolating polynomial coefficients in rest-frame color.
        self.__A_interpolator = self.__initialize_parameter_interpolator(A, col_med, kind=kind)
        self.__B_interpolator = self.__initialize_parameter_interpolator(B, col_med, kind=kind)
        self.__C_interpolator = self.__initialize_parameter_interpolator(C, col_med, kind=kind)
        self.__D_interpolator = self.__initialize_parameter_interpolator(D, col_med, kind=kind)
        self.__E_interpolator = self.__initialize_parameter_interpolator(E, col_med, kind=kind)
        self.__F_interpolator = self.__initialize_parameter_interpolator(F, col_med, kind=kind)
        self.__G_interpolator = self.__initialize_parameter_interpolator(G, col_med, kind=kind)

        # Linear extrapolation for z > 0.5
        self.__X_interpolator = lambda x: None
        self.__Y_interpolator = lambda x: None
        self.__X_interpolator, self.__Y_interpolator = self.__initialize_line_interpolators() 
   
    def __initialize_parameter_interpolator(self, parameter, median_colour, kind="linear"):
        # returns function for interpolating polynomial coefficients, as a function of colour
        return interp1d(median_colour, parameter, kind=kind, fill_value="extrapolate")
    
    def __initialize_line_interpolators(self):
        # linear coefficients for z>0.5
        X = np.zeros(self.nbins)
        Y = np.zeros(self.nbins)
        
        # find X, Y at each colour
        redshift = np.array([0.58,0.6])
        arr_ones = np.ones(len(redshift))
        for i in range(self.nbins):
            k = self.k(redshift, arr_ones*self.colour_med[i])
            X[i] = (k[1]-k[0]) / (redshift[1]-redshift[0])
            Y[i] = k[0] - X[i]*redshift[0]
        
        X_interpolator = interp1d(self.colour_med, X, kind='linear', fill_value="extrapolate")
        Y_interpolator = interp1d(self.colour_med, Y, kind='linear', fill_value="extrapolate")
        
        return X_interpolator, Y_interpolator

    def __A(self, colour):
        # coefficient of the z**6 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__A_interpolator(colour_clipped)

    def __B(self, colour):
        # coefficient of the z**5 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__B_interpolator(colour_clipped)

    def __C(self, colour):
        # coefficient of the z**4 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__C_interpolator(colour_clipped)

    def __D(self, colour):
        # coefficient of the z**3 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__D_interpolator(colour_clipped)
    
    def __E(self, colour):
        # coefficient of the z**2 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__E_interpolator(colour_clipped)
    
    def __F(self, colour):
        # coefficient of the z**1 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__F_interpolator(colour_clipped)
    
    def __G(self, colour):
        # coefficient of the z**0 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__G_interpolator(colour_clipped)

    def __X(self, colour):
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__X_interpolator(colour_clipped)

    def __Y(self, colour):
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__Y_interpolator(colour_clipped)


    def restgmr(self, redshift, obsframe_colour, median=False):
        """
        Polynomial fit to the FSF
        rest-frame colours for 0.01<z<0.6
        The rest-frame colour is extrapolated linearly for z>0.6

        Args:
            redshift: array of redshifts
            observer colour:   array of ^0.1(g-r)__0 colour
        Returns:
            array of rest-frame colours.
        """
        restgmr   = np.zeros(len(redshift))
        idx = redshift <= 0.6
        
        if median:
            obsframe_colour = np.copy(obsframe_colour)
            
            # Fig. 13 of https://arxiv.org/pdf/1701.06581.pdf
            obsframe_colour = 0.603 * np.ones_like(obsframe_colour)

        restgmr[idx] = self.__A(obsframe_colour[idx])*(redshift[idx]-self.z0)**6 + \
                 self.__B(obsframe_colour[idx])*(redshift[idx]-self.z0)**5 + \
                 self.__C(obsframe_colour[idx])*(redshift[idx]-self.z0)**4 + \
                 self.__D(obsframe_colour[idx])*(redshift[idx]-self.z0)**3 + \
                 self.__E(obsframe_colour[idx])*(redshift[idx]-self.z0)**2 + \
                 self.__F(obsframe_colour[idx])*(redshift[idx]-self.z0)**1 + \
                 self.__G(obsframe_colour[idx])

        idx = redshift > 0.6
        
        restgmr[idx] = self.__X(obsframe_colour[idx])*redshift[idx] + self.__Y(obsframe_colour[idx])
        
        return  restgmr    

        
    def k(self, redshift, restframe_colour, median=False):
        """
        Polynomial fit to the DESI
        K-correction for z<0.6
        The K-correction is extrapolated linearly for z>0.6

        Args:
            redshift: array of redshifts
            colour:   array of ^0.1(g-r) colour
        Returns:
            array of K-corrections
        """
        K   = np.zeros(len(redshift))
        idx = redshift <= 0.6
        
        if median:
            restframe_colour = np.copy(restframe_colour)
            
            # Fig. 13 of https://arxiv.org/pdf/1701.06581.pdf            
            restframe_colour = 0.603 * np.ones_like(restframe_colour)

        K[idx] = self.__A(restframe_colour[idx])*(redshift[idx]-self.z0)**6 + \
                 self.__B(restframe_colour[idx])*(redshift[idx]-self.z0)**5 + \
                 self.__C(restframe_colour[idx])*(redshift[idx]-self.z0)**4 + \
                 self.__D(restframe_colour[idx])*(redshift[idx]-self.z0)**3 + \
                 self.__E(restframe_colour[idx])*(redshift[idx]-self.z0)**2 + \
                 self.__F(restframe_colour[idx])*(redshift[idx]-self.z0)**1 + \
                 self.__G(restframe_colour[idx])

        idx = redshift > 0.6
        
        K[idx] = self.__X(restframe_colour[idx])*redshift[idx] + self.__Y(restframe_colour[idx])
        
        return  K    

    
    
    
    
    def k_nonnative_zref(self, refz, redshift, restframe_colour, median=False):
        refzs = refz * np.ones_like(redshift)
        
        return  self.k(redshift, restframe_colour, median=median) - self.k(refzs, restframe_colour, median=median) - 2.5 * np.log10(1. + refz)

    def rest_gmr_index(self, rest_gmr, photsys=None, file=None, kcoeff=False):
        if file=='gama' or file=='ajs':
            bins = np.array([-100., 0.18, 0.35, 0.52, 0.69, 0.86, 1.03, 100.])
        else:
            # file is DESI
            import os
            os.environ['CODE_ROOT'] = '/global/u2/l/ldrm11/Cole/'
            raw_dir = os.environ['CODE_ROOT'] + '/data/'    
            
            band = 'r'
            k_corr_file = raw_dir + '/jmext_kcorr_{}_{}band_z01.dat'.format(photsys.upper(), band.lower())
            col_min, col_max, A, B, C, D, E, F, G, col_med = np.loadtxt(k_corr_file, unpack=True)
            bins = col_med
            print(bins)
            
        idx  = np.digitize(rest_gmr, bins=bins)

        '''
        if kcoeff==True:
            for i in enumerate(rest_gmr):
                ddict = {i:{col_med, A[0], B[0], C[0], D[0]}}
        '''

        return idx

class DESI_KCorrection_color():
    def __init__(self, file='jmext', photsys=None):
        self.kRcorr = DESI_KCorrection(band='R', file=file, photsys=photsys)
        self.kGcorr = DESI_KCorrection(band='G', file=file, photsys=photsys)

    def obs_gmr(self, rest_gmr):        
        return  rest_gmr + self.kRcorr.k(z, rest_gmr) - self.kGcorr.k(z, rest_gmr)

    def rest_gmr_nonnative(self, native_rest_gmr):
        refzs = np.zeros_like(native_rest_gmr)
        
        return  native_rest_gmr + self.kGcorr.k(refzs, native_rest_gmr) - self.kRcorr.k(refzs, native_rest_gmr) 