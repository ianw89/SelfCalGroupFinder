import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from   scipy             import stats, interpolate
from   astropy.table     import Table, join, Column, vstack
from   scipy.optimize    import curve_fit
from   scipy.interpolate import splev, splrep, UnivariateSpline
from   scipy.interpolate import interp1d

if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from dataloc import *
from   k_corr_new.cosmo2             import cosmo, distmod, volcom
from   k_corr_new.tmr_kcorr         import tmr_kcorr



'''
Generates a rest frame colour look-up table and corresponding set of r-band k-corrections
based on John Moustakas's FastSpec catalogue.

Can use pregenerated tables if fresh=False.
'''

def selection(reg, survey='Y1'):
    f_ran=1.0 #random sampling fraction for quick run throughs
    Qevol=0.67 #Assumed luminosity evolution parameter
    area_N_DA1=2393.4228/(4*np.pi*(180.0/np.pi)**2)
    area_S_DA1=5358.2728/(4*np.pi*(180.0/np.pi)**2)
    
    if survey == 'sv3':
        area_N_DA1=90.0/(4*np.pi*(180.0/np.pi)**2)
        area_S_DA1=90.0/(4*np.pi*(180.0/np.pi)**2)
    
    zmin = 0.01
    zmax = 0.5
    
    South={'zmin': zmin, 'zmax': zmax, 'bright': 10.0, 'faint': 19.5 ,\
          'area': area_S_DA1, 'col': 'red' , 'style': 'solid', 'f_ran': f_ran, 'Qevol': Qevol}
    North={'zmin': zmin, 'zmax': zmax, 'bright': 10.0, 'faint': 19.54,\
          'area': area_N_DA1, 'col': 'blue', 'style': 'dashed', 'f_ran': f_ran, 'Qevol': Qevol}
    if (reg=='N'):
        x=North
    elif (reg=='S'):
        x=South
    else:
        print('Selection(region): unknown region')  
    return x


# load in catalogues
def load_catalogues(fpathN,fpathS):
    datS = Table.read(fpathS)
    datS.add_column(Column(name='reg', data=["S" for x in range(datS['Z'].size)]))
    datN = Table.read(fpathN)
    datN.add_column(Column(name='reg', data=["N" for x in range(datN['Z'].size)]))

    #Combine into a single table with the two parts flagged by the value in 
    #the reg column and delete the original tables
    dat=vstack(([datS, datN]), join_type='exact', metadata_conflicts='warn')
    del datS,datN

    # Add derived columns

    #Apparent magnitudes
    dat.add_column(Column(name='gmag', data=22.5-2.5*np.log10(dat['flux_g_dered'])))
    dat.add_column(Column(name='rmag', data=22.5-2.5*np.log10(dat['flux_r_dered'])))
    dat.add_column(Column(name='zmag', data=22.5-2.5*np.log10(dat['flux_z_dered'])))

    #Observer frame colour
    dat.add_column(Column(name='gmr_obs', data=dat['gmag']-dat['rmag']))
    dat.add_column(Column(name='OBS_GMR', data=dat['gmag']-dat['rmag']))

    return dat

  
# Make North like South if wanted.
def modify_fsf_to_match_N_S(fsf):
    
    rmask=(fsf['reg']=='S')
    colhist_S,colbin=np.histogram(fsf['REST_GMR_0P1_original'][rmask], bins=800,  density=True)
    colbin_cen= 0.5*(colbin[1:]+colbin[:-1])
    plt.plot(colbin_cen, colhist_S, label='South', color='red', linestyle='solid')
    rmask=(fsf['reg']=='N')
    colhist_N,colbin=np.histogram(fsf['REST_GMR_0P1_original'][rmask], bins=colbin,  density=True)
    plt.plot(colbin_cen, colhist_N, label='North', color='blue', linestyle='solid')
    plt.xlabel('G-R')
    plt.legend()
    plt.show()
  
    #Form cumulative distributions
    colhist_N_cuml= np.cumsum(colhist_N)
    colhist_S_cuml= np.cumsum(colhist_S)

    #Assign a new colour and new r-band magnitue to the North data
    rmask=(fsf['reg']=='N')
    colhist_cuml=np.interp(fsf['REST_GMR_0P1_original'][rmask],colbin_cen,colhist_N_cuml) # look up cumulative value at the object's colour
    col=np.interp(colhist_cuml,colhist_S_cuml,colbin_cen) # find the colour in the South distribution that has the same cumulative value
    fsf['ABSMAG_SDSS_R'][rmask]=fsf['ABSMAG_SDSS_R'][rmask]+fsf['REST_GMR_0P1_original'][rmask]-col #assign new r-band ABSMAG
    fsf['REST_GMR_0P1_original'][rmask] = fsf['ABSMAG_SDSS_G'][rmask] - fsf['ABSMAG_SDSS_R'][rmask] #and compute new rest frame colour


    # Plot to check that N and S colour distributions now agree    
    rmask=(fsf['reg']=='S')
    colhist_S,colbin=np.histogram(fsf['REST_GMR_0P1_original'][rmask], bins=80,  density=True)
    colbin_cen= 0.5*(colbin[1:]+colbin[:-1])
    plt.plot(colbin_cen, colhist_S, label='South', color='red', linestyle='solid')
    rmask=(fsf['reg']=='N')
    colhist_N,colbin=np.histogram(fsf[rmask]['REST_GMR_0P1_original'], bins=colbin,  density=True)
    plt.plot(colbin_cen, colhist_N, label='North', color='blue', linestyle='solid')
    plt.ylim(0.0,3.5)
    plt.xlim(-0.1,1.3)
    plt.xlabel('G-R')
    plt.legend()
    plt.show()

    #Sub divide by redshift to see if there is agreement in different redshift bins

    for zmin in (np.linspace(0.0,0.5,5)):
        zmax=zmin+0.1
        zmask = (fsf['Z']>zmin) & (fsf['Z']<zmax)
        rmask=(fsf['reg']=='S')
        mask=rmask&zmask
        colhist_S,colbin=np.histogram(fsf['REST_GMR_0P1_original'][mask], bins=80,  density=True)
        colbin_cen= 0.5*(colbin[1:]+colbin[:-1])
        plt.plot(colbin_cen, colhist_S, label='South', color='red', linestyle='solid')
        rmask=(fsf['reg']=='N')
        mask=rmask&zmask         
        colhist_N,colbin=np.histogram(fsf['REST_GMR_0P1_original'][mask], bins=colbin,  density=True)
        plt.plot(colbin_cen, colhist_N, label='North', color='blue', linestyle='solid')
        plt.ylim(0.0,3.5)
        plt.xlim(-0.1,1.3)
        plt.xlabel('G-R')
        title='{}<z<{}'.format(zmin,zmax)
        plt.title(title)
        plt.legend()
        plt.show()
    
    return fsf

def pc10(x):  #returns 10th percentile (for use in binned_statistics)
    pc10=np.percentile(x,10.0)
    return pc10


def pc90(x): #returns 90th percentile (for use in binned_statistics)
    pc90=np.percentile(x,90.0)
    return pc90




# Reads in John Moustakas' FASTSPEC catalogue on which we base our k-corrections
def gen_JM_lookup():
    # read in JM data for lookup.
    fpath = './data/fastspec-iron-main-bright.fits'

    print('READING IN {}...'.format(fpath))

    fsf = Table.read(fpath)

    # rename some columns for convenience
    fsf.rename_column('ABSMAG01_SDSS_G', 'ABSMAG_SDSS_G')
    fsf.rename_column('ABSMAG01_SDSS_R', 'ABSMAG_SDSS_R')
    fsf.rename_column('PHOTSYS', 'reg')
    
    #Only keep objects with positive fluxes
    fmask = (fsf['FLUX_R']>0.0) & (fsf['FLUX_G']>0.0) 
    fsf = fsf[fmask]
    
    # Compute apparent magnitudes and observer frame colour
    fsf['rmag']  = 22.5 - 2.5*np.log10(fsf['FLUX_R'])
    fsf['gmag']  = 22.5 - 2.5*np.log10(fsf['FLUX_G'])
    fsf['gmr_obs']    = fsf['gmag'] - fsf['rmag']
    
    # Compute the k-corrections derived from John's data and the rest frame colour
    fsf['DISTMOD']    = distmod(fsf['Z'])
    fsf['KR_DERIVED']  = fsf['rmag'] - fsf['DISTMOD'] - fsf['ABSMAG_SDSS_R'] 
    fsf['KG_DERIVED']  = fsf['gmag'] - fsf['DISTMOD'] - fsf['ABSMAG_SDSS_G'] 
    fsf['REST_GMR_0P1_original'] = fsf['ABSMAG_SDSS_G'] - fsf['ABSMAG_SDSS_R']
    
    
    #  Addtional masking to limit redshift range and to good data
    band = 'r'
    photmask = (fsf['reg'] == 'N') | (fsf['reg'] == 'S')
    zmask = (fsf['Z'] > 0.01) & (fsf['Z'] < 0.6)
    cmask = (fsf['REST_GMR_0P1_original'] != 0) & (fsf['REST_GMR_0P1_original'] > -0.25) & (fsf['REST_GMR_0P1_original'] < 1.5)
    kmask = (fsf['K{}_DERIVED'.format(band.upper())] != 0) 
    mask = photmask&zmask&kmask&cmask
    fsf = fsf[mask]
    
    # Add additional columns for later use
    fsf['REST_GMR_0P1'] = -99.9  # Will be used to store the colour assigned from the look-up table
    fsf['ABSMAG_RP1'] = -99.9  # Will be used to store the absolute magnitude computed from the 
                               #fitted polynomial k-correction

    print('FASTSPEC DATA READ.')
    
    return fsf




#
# Colour look-up table based on John Moustakas FASTSPEC data
# Can read in previously generated look-up table or generate a fresh one
#
def colour_table_lookup(dat, regmask, reg, plot=True, fresh=False):
        
    dcolbin=0.025    
    dzbin=0.0025
    xbins = np.arange(0.01, 0.6001, dzbin)
    ybins = np.arange(-1.0, 4.001, dcolbin)
    ny=np.size(ybins)-2
    nx=np.size(xbins)-2
    print('nx,ny:',nx,ny)
    
    if fresh:
        print('GENERATING NEW LOOKUP TABLE.')
        dat_all = gen_JM_lookup()                    # read John's FSF catalogue
        #dat_all = modify_fsf_to_match_N_S(dat_all)  # modify FSF to make N colour distribution match S
        rmask = (dat_all['reg'] == reg)              # select just N or S data as each has to be modelled separately
        dat_all=dat_all[rmask]                       # restrict all data to just this region
        
        x = dat_all['Z']
        y = dat_all['gmr_obs']
        z = dat_all['REST_GMR_0P1_original']
        # Determine the median rest frame colour in each bin of redshift and observer frame colour
        H, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, values = z, statistic='median', bins = [xbins, ybins], expand_binnumbers=True)
        count, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, values = z, statistic='count', bins = [xbins, ybins], expand_binnumbers=True)
        XX, YY = np.meshgrid(xedges, yedges)
 
        sam = False
        if (sam):
          # replace noisy and empty pixel values by neighouring values at the same redshift
          for idx in range(len(H)):
              mask = np.isnan(H[idx])
              try:
                  H[idx][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), H[idx][~mask])
              except:
                  pass # i.e: if whole column is empty
                
        shaun= True        
        if (shaun):    
          colmid=0.8   # A mid range colour in the well populated band that needs no extrapolation 
          jmid=(colmid-ybins[0])/dcolbin # corresponding bin index

          #Extrapolate to bins with zero or low count be replacing with the value from the nearest bin in colour at the same redshift with count above a threshold
          for j in range(len(yedges)-1):  # loop over the bins
            for i in range(len(xedges)-1):                
                if (count[i][j]<2.5):  # For any bin with a count of 2 or less
                    if (j>jmid):       # If redder than colmid replace with the value from the reddest bin with count greater than 3
                        j1=j
                        while ((count[i][j1]<3.5) & (j1>jmid)):
                           j1=j1-1
                           H[i][j]=H[i][j1]    
                    if (j<=jmid):       # If bluer than colmid replace with the value from the bluest bin with count greater than 3
                        j1=j
                        j1=j
                        while ((count[i][j1]<3.5) & (j1<=jmid)):
                           j1=j1+1
                           H[i][j]=H[i][j1] 
        
        
        # save the look-up table for future use
        np.savetxt('./data/H_{}.txt'.format(reg), H)
        print('LOOK-UP TABLE GENERATED AND SAVED.')
        
        
        # Now apply to the FSF data to define the looked-up rest frame colour to use when binning the k-corrections
        regmask_all=(dat_all['reg']==reg) # this will always be true as we have already masked to this region
        colour_table_lookup(dat_all, regmask_all, reg, plot=True, fresh=False)
        
        
        if plot:  # Compare colours from the look-up table with John's original colours
           for zmin in np.linspace(0.0,0.5,num=5): #For redshift bins plot histograms of the restframe colour distributions
             zmask= (dat_all['Z']> zmin) &(dat_all['Z']< zmin+0.125)
             mask=zmask
             jhist,bins=np.histogram(dat_all['REST_GMR_0P1_original'][mask],bins=40,density=True)
             zlmask= (dat['Z']> zmin) &(dat['Z']< zmin+0.1)
             lhist,bins=np.histogram(dat_all['REST_GMR_0P1'][mask],bins=bins,density=True)
             binc= (bins[:-1]+bins[1:])/2.0   
             plt.plot(binc,jhist,'k-',label="John Moustakas")
             plt.plot(binc,lhist,'r-',label="Look-up Table")
             plt.xlabel(r'$g-r$')
             plt.ylabel(r'N')
             plt.legend()
             plt.title('Colour distribution '+str(zmin)+'<z<'+str(zmin+0.125))
             plt.show()

        # now create corresponding k-corrections in bins of the assigned colour in the region we have already restricted two
        all_bins, all_medians = gen_kcorr(dat_all, reg, colval='REST_GMR_0P1', nbins=10, write=fresh, fill_between=True)

        
        if plot == True:  #If requested make some plots to verify the k-corrections
              # k-correction difference plots.
              kcorr_r  = DESI_KCorrection(band='R', file='jmext', photsys=reg)

              prop_cycle = plt.rcParams['axes.prop_cycle']
              colors = prop_cycle.by_key()['color']
                
                
              k_corr_file = './data/jmext_kcorr_{}_rband_z01.dat'.format(reg)
              col_min, col_max, A, B, C, D, E, F, G, col_med = np.loadtxt(k_corr_file, unpack=True)
                
                
              #  Plot median k corrections and the polynomial fits to them
              fig,ax = plt.subplots(2,1, figsize=(5,10))
              prop_cycle = plt.rcParams['axes.prop_cycle']
              colours = prop_cycle.by_key()['color']
            
              for idx in range(len(all_bins)):
                    print('idx=',idx,len(all_bins))
                    z       = all_bins[idx]
                    medians = all_medians[idx]
                    rest_GMR     = col_med[idx]
                    GMR          = rest_GMR * np.ones_like(z)
                    k            = kcorr_r.k(z, GMR)

                    ax[0].plot(z, medians, color=colours[idx], label='{:.5f}'.format(rest_GMR))
                    ax[0].plot(z, k, color=colours[idx])

                    ax[1].plot(z, medians-k, color=colours[idx])
                    ax[1].axhline(0.0, ls='--', lw=0.5, color='black')

              ax[0].legend()
              ax[1].set_xlabel('Z')
              ax[0].set_ylabel(r'$k_r$')
              ax[1].set_ylabel(r'$k_{r,med} - k_{r, curve}$')
              plt.show()

             
          
              #Compare the absolute magnitudes computed using the polynomial k-corrections to John Moustakas' orginal ones  
              dat_all['ABSMAG_SDSS_R_SAM'] = -99.9 # create an array to store the assigned absolute magnitudes
              rmask = (dat_all['reg'] == reg)
              Qzero = 0.0 # no evolution correction to be consistent with John
              dat_all['ABSMAG_SDSS_R_SAM'][rmask]=ABSMAG(dat_all['rmag'][rmask],dat_all['Z'][rmask],dat_all['REST_GMR_0P1'][rmask],kcorr_r,Qzero)

              diff = dat_all['ABSMAG_SDSS_R_SAM'][rmask]-dat_all['ABSMAG_SDSS_R'][rmask]
              bin_medians, bin_edges, binnumber = stats.binned_statistic(dat_all['Z'][rmask], diff, statistic='median', bins=25)
              bin_10pc, bin_edges, binnumber = stats.binned_statistic(dat_all['Z'][rmask], diff, statistic=pc10, bins=25)
              bin_90pc, bin_edges, binnumber = stats.binned_statistic(dat_all['Z'][rmask], diff, statistic=pc90, bins=25)  
            
              plt.scatter(dat_all['Z'][rmask],diff, marker='.', color='red', linewidths=0,s=0.2,alpha=0.4)
              centres = (bin_edges[:-1]+bin_edges[1:])/2
              plt.plot(centres, bin_medians, label='median')
              plt.plot(centres, bin_10pc, label='10%')
              plt.plot(centres, bin_90pc, label='90%')
              plt.ylim(-0.15, 0.15)
              plt.axhline(0.0, ls='--', color='black')
              plt.xlabel('Z')
              plt.ylabel('SM-JM Mr')
              plt.legend(loc='upper left')
              plt.show()

        # now call again to apply the look-up with the already saved look-up table
        colour_table_lookup(dat, regmask, reg, plot=True, fresh=False)
        
    else:
        print('LOADING IN LOOKUP TABLE.')
        H = np.loadtxt('./data/H_{}.txt'.format(reg))

        if plot: # make a plot of the look-up table
            XX, YY = np.meshgrid(xbins, ybins)
            fig   = plt.figure(figsize = (13,7))
            ax1   = plt.subplot(111)
            plot1 = ax1.pcolormesh(XX,YY,H.T)
            label = r'$(g-r)_{0, med}$'
            cbar = plt.colorbar(plot1, ax=ax1, pad = .015, aspect=10, label=label)
            plt.ylim(-0.2, 2)
            plt.xlabel(r'$z$')
            plt.ylabel(r'$g-r$')
            plt.title('Lookup Table ({})'.format(reg))
            
            #spath = './graphs/lookup_{}.pdf'.format(reg)
            #plt.savefig(spath)
            plt.show()


        Harray=np.asarray(H) # convert to numpy array to enable indexing     
        # use to generate grid indices for the  input data.
        __, __, __, i2 = stats.binned_statistic_2d(dat['Z'][regmask], dat['gmr_obs'][regmask], values=dat['Z'][regmask], statistic='median', bins = [xbins, ybins], expand_binnumbers=True)
        j2=i2   #keep the original indices as useful for generating indices of CIC neighbouring cells
        i2=i2-1 #correct for offset in returned bin indices
        
        #NGP look-up
        #dat['REST_GMR_0P1'][regmask]=Harray[i2[0],i2[1]]
        
        #Cloud-in-Cell look-up
        #first compute the difference in the coordinate value of each data point and its nearest grid point in units of the bin spacing
        dx=0.5+i2[0]-(dat['Z'][regmask]-xbins[0])/dzbin           # these should satisfy -0.5<dx<0.5          
        dy=0.5+i2[1]-(dat['gmr_obs'][regmask]-ybins[0])/dcolbin   # these should satisfy -0.5<dy<0.5 
        #for each negative value we need to change of j2 to select the correct neighbouring cell 
        j2[0][(dx<0)]=j2[0][(dx<0)]-2
        j2[1][(dy<0)]=j2[1][(dy<0)]-2
        #Cloud-in-Cell weights  (these add to unity)
        wa=(1.0-np.absolute(dx))*(1.0-np.absolute(dy))
        wb=     np.absolute(dx) *(1.0-np.absolute(dy))
        wc=(1.0-np.absolute(dx))*     np.absolute(dy)
        wd=     np.absolute(dx) *     np.absolute(dy)
        #To avoid out of bounds edge effects replace out of bound cells indices by the NGP cell index
        j0mask = (j2[0]>nx) | (j2[0]<0)
        j2[0][j0mask]=i2[0][j0mask]
        j1mask = (j2[1]>ny) | (j2[1]<0)
        j2[1][j1mask]=i2[1][j1mask]
        # The Cloud-in-Cell weighted value
        dat['REST_GMR_0P1'][regmask] = wa*Harray[i2[0],i2[1]]+wb*Harray[j2[0],i2[1]]+wc*Harray[i2[0],j2[1]]+wd*Harray[j2[0],j2[1]]

        print('REST-FRAME COLOURS ASSIGNED.')
   
    return dat



# generate an output table of k-correction polynomial coefficients
def kcorr_table(mins, maxs, polys, medians, split_num, opath, print_table=False):
    
    print('TABLE OPATH:', opath)
    
    header = "# 'gmr_min', 'gmr_max', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'gmr_med'\n"
    print(header)
    
    if opath:
        f = open(opath, "w")
        f.writelines([header])
    
    results = []
    for idx in range(split_num):
        result = "{} {} {} {} {} {} {} {} {} {}\n".format(mins[idx], maxs[idx], polys[idx][0], polys[idx][1], polys[idx][2], polys[idx][3], polys[idx][4], polys[idx][5], polys[idx][6], medians[idx])
        if print_table:
            print(result)
        
        results.append(result)
        
        if opath:
            f.writelines([result])

    if opath:
        #result.write(opath, format='fits', overwrite=True)
        f.close()
        
        
# functional form of k-correction polynomials        
def func(z, a0, a1, a2, a3, a4, a5, a6):
    zref = 0.1
    x = z-zref
    return a0*x**6 + a1*x**5 + a2*x**4 + a3*x**3 + a4*x**2 + a5*x**1 + a6


# generate k-correction polynomials        
def gen_kcorr(fsf, regions, colval='REST_GMR_0P1', nbins=10, write=False, rolling=False, adjuster=False, fill_between=False):
    print('Fitting polynomial k-corrections to the colour binned data.')
    dat_everything = fsf
    bands = ['R', 'G', 'Z', 'W1', 'W2']

    if rolling:
        nbins += 2

    colours = plt.cm.jet(np.linspace(0,1,nbins+2))

    for photsys in regions:
        for band in bands:

            photmask = (dat_everything['reg'] == photsys)
            dat_all = dat_everything[photmask]
            print('PHOTSYS={}, BAND={}, LENGTH={}'.format(photsys, band, len(dat_all)))

            percentiles = np.arange(0, 1.01, 1/nbins) 
            bin_edges = stats.mstats.mquantiles(dat_all[colval], percentiles)
            dat_all['COLOUR_BIN'] = np.digitize(dat_all[colval], bin_edges)

            if rolling:
                # rolling colour bins - redefine digitisation for finer initial binning.
                bin_edges = np.linspace(0, 1.25, nbins)
                dat_all['COLOUR_BIN'] = np.digitize(dat_all[colval], bin_edges)

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
                start = 1
                end = nbins - 1
 
            for idx in np.arange(start,end):
                mask = (dat_all[col_var] == idx)

                # extend the masks to make 'rolling' fits
                if rolling:
                    mask = (dat_all[col_var] == idx-1) | (dat_all[col_var] == idx) | (dat_all[col_var] == idx+1)

                dat = dat_all[mask]
                print('processing colour bin:',idx,'which has ', len(dat),' entries')

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
                total_bins = 40
                bin_medians, bin_edges, binnumber = stats.binned_statistic(x, y,statistic='median', bins=total_bins)
                bin_width = (bin_edges[1] - bin_edges[0])
                bin_centres = bin_edges[1:] - bin_width/2
                bins = bin_centres
                running_median = bin_medians
                #mask empty bins
                bins= bins[~np.isnan(running_median)] 
                running_median= running_median[~np.isnan(running_median)]

                p0 = [1.5,  0.6, -1.15,  0.1, -0.16]
                popt, pcov = curve_fit(func, bins, running_median, maxfev=5000)

                mins.append(col_min)
                maxs.append(col_max)
                polys.append(popt)
                medians.append(col_med)
                all_bins.append(bins)
                all_medians.append(running_median)
         
                label= None
                plt.plot(bins,running_median, marker=None,fillstyle='none',markersize=5,alpha=0.5,color=colours[idx])
            
                if fill_between:
                    if idx % 2 != 0:
                        try:
                            plt.fill_between(bins,plow,phigh,color='blue',alpha=0.25)            
                        except:
                            pass

            
            plt.xlabel('z')
            plt.ylabel('K(z)')
            plt.ylim(-1, 1)
            plt.title('FSF (main, v2.0), PHOTSYS={}, band={}'.format(photsys, band))
            plt.show()

            opath = './data/jmext_kcorr_{}_{}band_z01.dat'.format(photsys.upper(), band.lower())

            split_num = end-start

            if rolling:
                split_num = nbins - 2

            if write:
                print('WRITING TO {}'.format(opath))
                kcorr_table(mins, maxs, polys, medians, split_num, opath)

    bins = np.arange(-0.5, 1.5, 0.01)
    plt.hist(dat_all[colval], bins=bins)

    for col_split in maxs[0:-1]:
        plt.axvline(col_split, ls='--', lw=0.25, color='r')

    plt.xlabel(r'$(g-r)_0$')
    plt.ylabel('N')
    plt.title('FSF (main, v2.0), PHOTSYS={}, band={}'.format(photsys, band))
    #plt.show()
    
    return all_bins, all_medians 



class DESI_KCorrection(object):
    def __init__(self, band, kind="cubic", file='jmext', photsys=None):
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
        
        #import os
        #os.environ['CODE_ROOT'] = './'
        #raw_dir = os.environ['CODE_ROOT'] + '/data/'        
        #print(os.environ['CODE_ROOT'])

        raw_dir = PY_SRC_FOLDER + 'k_corr_new/data' #'/home/users/imw2293/SelfCalGroupFinder/desi/kcorr/parameters'
        
        # check pointing to right directory.
        # print('FILE:', file)
        # print('PHOTSYS:', photsys)
        
        if file == 'ajs':
            k_corr_file = raw_dir + '/ajs_kcorr_{}band_z01.dat'.format(band.lower())
                 
        elif file == 'jmext':
            k_corr_file = raw_dir + '/jmext_kcorr_{}_{}band_z01.dat.mpeg'.format(photsys.upper(), band.lower())
            
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
            #import os
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

    
# ABSMAG_R= appmag -DMOD  -kcorr_r.k(z, rest_GMR) +Qevol*(z-0.1) 
def ABSMAG(appmag,z,rest_GMR,kcorr_r,Qevol):
        DMOD=25.0+5.0*np.log10(cosmo.luminosity_distance(z).value)  
        ABSMAG=appmag-DMOD-kcorr_r.k(z,rest_GMR)+Qevol*(z-0.1)
        return ABSMAG