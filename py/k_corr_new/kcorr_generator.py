import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from astropy.table import join,Table,Column,vstack

def kcorr_table(mins, maxs, polys, medians, split_num, opath, print_table=False):
    '''
    Function to write k-correction polynomial coefficients to file.
    '''
    
    print('TABLE OPATH:', opath)
    
    header = "# 'gmr_min', 'gmr_max', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'gmr_med'\n"
    print(header)
    
    if opath:
        f = open(opath, "w")
        f.writelines([header])
    
    results = []
    for idx in range(split_num):
        result = "{} {} {} {} {} {} {} {} {} {}\n".format(mins[idx], maxs[idx], polys[idx][0], polys[idx][1], polys[idx][2], polys[idx][3], polys[idx][4], polys[idx][5], polys[idx][6], medians[idx])
        if print:
            print(result)
        
        results.append(result)
        
        if opath:
            f.writelines([result])

    if opath:
        #result.write(opath, format='fits', overwrite=True)
        f.close()
    

def func(z, a0, a1, a2, a3, a4, a5, a6):
    '''
    Function to fit k-correction, in form of /Sum a_n (z-z_{ref})^n
    '''
    zref = 0.1
    x = z-zref
    return a0*x**6 + a1*x**5 + a2*x**4 + a3*x**3 + a4*x**2 + a5*x**1 + a6



def gen_kcorr(fsf, regions, colval='REST_GMR_0P1', nbins=10, write=False, rolling=False, adjuster=False, fill_between=False, plot=True, suffix=''):

    from scipy import interpolate
    from scipy import stats
    from scipy.optimize import curve_fit
    from scipy.interpolate import splev, splrep, UnivariateSpline
    
    dat_everything = fsf
    bands = ['G', 'R']# , 'Z', 'W1']

    print('gen_kcorr colval:', colval)
    
    if rolling:
        nbins += 2

    colours = plt.cm.jet(np.linspace(0,1,nbins+2))

    for photsys in regions:
        for band in bands:

            photmask = (dat_everything['PHOTSYS'] == photsys)
            dat_all = dat_everything[photmask]
            print('PHOTSYS={}, BAND={}, LENGTH={}'.format(photsys, band, len(dat_all)))

            percentiles = np.arange(0, 1.01, 1/nbins) 
            bin_edges = stats.mstats.mquantiles(dat_all[colval], percentiles)
            dat_all['COLOUR_BIN'] = np.digitize(dat_all[colval], bin_edges)

            if rolling:
                # rolling colour bins - redefine digitisation for finer initial binning.
                bin_edges = np.linspace(0, 1.25, nbins)
                dat_all['COLOUR_BIN'] = np.digitize(dat_all[colval], bin_edges)

            print(len(dat_all))
            
            col_var = 'COLOUR_BIN'
            col_min_all = []
            col_max_all = []

            mins = []
            maxs = []
            polys = []
            medians = []
            all_bins    = []
            all_medians = []
            
            start, end = 1, nbins

            if rolling:
                start, end = 1, nbins - 1

            for idx in np.arange(start,end):
                mask = (dat_all[col_var] == idx)

                # extend the masks to make 'rolling' fits
                if rolling:
                    mask = (dat_all[col_var] == idx-1) | (dat_all[col_var] == idx) | (dat_all[col_var] == idx+1)

                dat = dat_all[mask]
                print(idx, len(dat))

                col_min = min(dat[colval])
                col_max = max(dat[colval])
                col_med = np.median(dat[colval])

                yvar = 'K{}_DERIVED'.format(band.upper())

                x = dat['Z']
                y = dat[yvar]
                x = x[~np.isnan(y)]
                y = y[~np.isnan(y)]
                x = x[np.isfinite(y)]
                y = y[np.isfinite(y)]
                
                # NEW MEDIAN test
                total_bins = 100
                bin_medians, bin_edges, binnumber = stats.binned_statistic(x, y,statistic='median', bins=total_bins)
                                
                bin_width = (bin_edges[1] - bin_edges[0])
                bin_centres = bin_edges[1:] - bin_width/2
                bins = bin_centres        
                running_median = bin_medians

                bins = bins[np.isfinite(running_median)]
                running_median = running_median[np.isfinite(running_median)]
                
                p0 = [1.5,  0.6, -1.15,  0.1, -0.16]
                popt, pcov = curve_fit(func, bins, running_median, maxfev=5000)

                mins.append(col_min)
                maxs.append(col_max)
                polys.append(popt)
                medians.append(col_med)
                all_bins.append(bins)
                all_medians.append(running_median)
                

                label= None
                if plot:
                    plt.plot(bins,running_median, marker=None,fillstyle='none',markersize=5,alpha=0.5,color=colours[idx],label=label)


                    if fill_between:
                        if idx % 2 != 0:
                            try:
                                plt.fill_between(bins,plow,phigh,color='blue',alpha=0.25)            
                            except:
                                pass
            
            if plot:
                plt.legend()
                plt.xlabel('z')
                plt.ylabel('K(z)')
                plt.ylim(-1, 1)
                plt.title('FSF (main, v2.0), PHOTSYS={}, band={}'.format(photsys, band))
                plt.show()

            opath = '/global/homes/l/ldrm11/Cole/data/jmext_kcorr_{}_{}band_z01{}.dat'.format(photsys.upper(), band.lower(), suffix)
            
            print('in kcorr:', opath)

            #opath = 'gjmext_kcorr_{}_{}band_z01.dat'.format(photsys.upper(), band.lower())
            
            #TODO: Check this.
            split_num = end-start

            if rolling:
                split_num = nbins - 2

            if write:
                print('WRITING TO {}'.format(opath))
                kcorr_table(mins, maxs, polys, medians, split_num, opath)

    if plot:
        bins = np.arange(-0.5, 1.5, 0.01)
        plt.hist(dat_all[colval], bins=bins)

        for col_split in maxs[0:-1]:
            plt.axvline(col_split, ls='--', lw=0.25, color='r')

        plt.xlabel(r'$(g-r)_0$')
        plt.ylabel('N')
        plt.title('FSF (main, v2.0), PHOTSYS={}, band={}'.format(photsys, band))
        #plt.show()
    
    return all_bins, all_medians 