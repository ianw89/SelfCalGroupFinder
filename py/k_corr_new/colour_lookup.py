import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from astropy.table import Table, join

#import os
#import sys
#os.environ['CODE_ROOT'] = os.environ['HOME'] + '/Cole/'
#print(os.environ['CODE_ROOT'])
#sys.path.append(os.environ['HOME'])
#sys.path.append(os.environ['CODE_ROOT'])

from SelfCalGroupFinder.py.k_corr_new.cosmo2 import cosmo, distmod, volcom
import SelfCalGroupFinder.py.k_corr_new.kcorr_generator as kg
import SelfCalGroupFinder.py.k_corr_new.catalogue_analysis as ca
from SelfCalGroupFinder.py.k_corr_new.smith_kcorr   import DESI_KCorrection 


def pc10(x):  #returns 10th percentile (for use in binned_statistics)
    pc10=np.percentile(x,10.0)
    return pc10

def pc90(x): #returns 90th percentile (for use in binned_statistics)
    pc90=np.percentile(x,90.0)
    return pc90


def gen_JM_lookup(band='R'):
    # read in JM data for lookup.
    area_N=2393.4228/(4*np.pi*(180.0/np.pi)**2)
    area_S=5358.2728/(4*np.pi*(180.0/np.pi)**2)

    fpath = '/pscratch/sd/l/ldrm11/FSF/fastspec-iron-main-bright.fits'
    #fpath = '/global/u2/s/smcole/DESI/NvsS/data/fastspec-iron-main-bright.fits'

    print('READING IN {}...'.format(fpath))

    fsf = Table.read(fpath)

    fsf.rename_column('ABSMAG01_SDSS_G', 'ABSMAG_SDSS_G')
    fsf.rename_column('ABSMAG01_SDSS_R', 'ABSMAG_SDSS_R')
    fsf.rename_column('ABSMAG01_SDSS_Z', 'ABSMAG_SDSS_Z')
    fsf.rename_column('ABSMAG01_W1', 'ABSMAG_W1')

    fsf.rename_column('KCORR01_SDSS_G', 'KCORR_SDSS_G')
    fsf.rename_column('KCORR01_SDSS_R', 'KCORR_SDSS_R')
    fsf.rename_column('KCORR01_SDSS_Z', 'KCORR_SDSS_Z')
    fsf.rename_column('KCORR01_W1', 'KCORR_W1')

    # added z_flux mask
    fmask = (fsf['FLUX_R']>0.0) & (fsf['FLUX_G']>0.0) 
    
    if band=='Z':
        fmask = fmask & (fsf['FLUX_Z']>0.0)
    elif band=='W1':
        fmask = fmask & (fsf['FLUX_W1']>0.0)
    else:
        print('No extra band mask.')
        
    fsf = fsf[fmask]
    
    fsf['DISTMOD']    = distmod(fsf['Z'])
    fsf['RMAG_DRED']  = 22.5 - 2.5*np.log10(fsf['FLUX_R'])
    fsf['GMAG_DRED']  = 22.5 - 2.5*np.log10(fsf['FLUX_G'])
    fsf['ZMAG_DRED']  = 22.5 - 2.5*np.log10(fsf['FLUX_Z'])
    fsf['W1MAG_DRED']  = 22.5 - 2.5*np.log10(fsf['FLUX_W1'])
    #fsf['W2MAG_DRED']  = 22.5 - 2.5*np.log10(fsf['FLUX_W2'])

    fsf['OBS_GMR']    = fsf['GMAG_DRED'] - fsf['RMAG_DRED']
    fsf['OBS_RMZ']    = fsf['RMAG_DRED'] - fsf['ZMAG_DRED']
    fsf['OBS_RMW1']   = fsf['RMAG_DRED'] - fsf['W1MAG_DRED']

    
    h = 1.0
    fsf['KR_DERIVED']  = fsf['RMAG_DRED'] - fsf['DISTMOD'] - (fsf['ABSMAG_SDSS_R'] - 5*np.log10(h))
    fsf['KG_DERIVED']  = fsf['GMAG_DRED'] - fsf['DISTMOD'] - (fsf['ABSMAG_SDSS_G'] - 5*np.log10(h))
    fsf['KZ_DERIVED']  = fsf['ZMAG_DRED'] - fsf['DISTMOD'] - (fsf['ABSMAG_SDSS_Z'] - 5*np.log10(h))
    fsf['KW1_DERIVED']  = fsf['W1MAG_DRED'] - fsf['DISTMOD'] - (fsf['ABSMAG_W1'] - 5*np.log10(h))
    
    #fsf['KW2_DERIVED']  = fsf['W2MAG_DRED'] - fsf['DISTMOD'] - (fsf['ABSMAG_SDSS_W2'] - 5*np.log10(h))
    
    fsf['REST_GMR']  = fsf['ABSMAG_SDSS_G'] - fsf['ABSMAG_SDSS_R']
    fsf['REST_RMZ']  = fsf['ABSMAG_SDSS_R'] - fsf['ABSMAG_SDSS_Z']
    fsf['REST_RMW1'] = fsf['ABSMAG_SDSS_R'] - fsf['ABSMAG_W1']

    
    band = 'r'
    photmask = (fsf['PHOTSYS'] == 'N') | (fsf['PHOTSYS'] == 'S')
    zmask = (fsf['Z'] > 0.002) & (fsf['Z'] < 0.6)
    kmask = (fsf['K{}_DERIVED'.format(band.upper())] != 0) 
    cmask = (fsf['REST_GMR'] != 0) & (fsf['REST_GMR'] > -0.25) & (fsf['REST_GMR'] < 1.5)
    
    mask = photmask&zmask&kmask&cmask
    
    
    # test - key idea to avoid infs:
    if band == 'Z':
        czmask = (fsf['REST_RMZ'] != 0) & (fsf['REST_RMZ'] > -10) & (fsf['REST_RMZ'] < 4.0)
        mask = mask&czmask
   
    elif band == 'W1':
        cw1mask = (fsf['REST_RMW1'] != 0) & (fsf['REST_RMW1'] > -30) & (fsf['REST_RMW1'] < 30.0)
        mask = mask&cw1mask
        
    else:
        pass
    
    fsf['REST_GMR_0P1_original'] = fsf['REST_GMR']
    fsf['REST_RMZ_0P1_original'] = fsf['REST_RMZ']
    fsf['REST_RMW1_0P1_original'] = fsf['REST_RMW1']

    fsf['gmr_obs'] = fsf['OBS_GMR']
    fsf['rmz_obs'] = fsf['OBS_RMZ']
    fsf['rmw1_obs'] = fsf['OBS_RMW1']

    fsf['XBIN'] = -99
    fsf['YBIN'] = -99
    fsf['REST_GMR_0P1'] = -99.9
    fsf['REST_RMZ_0P1'] = -99.9
    fsf['REST_RMW1_0P1'] = -99.9

    fsf['reg'] = fsf['PHOTSYS']
    fsf['rmag'] = fsf['RMAG_DRED']
    fsf['gmag'] = fsf['GMAG_DRED']
    fsf['zmag'] = fsf['ZMAG_DRED']
    fsf['w1mag'] = fsf['W1MAG_DRED']

    fsf['ABSMAG_RP1'] = -99.9  
    fsf['ABSMAG_GP1'] = -99.9  
    fsf['ABSMAG_ZP1'] = -99.9  
    fsf['ABSMAG_W1P1'] = -99.9  

    fsf = fsf[mask]
    
    if band=='Z':
        isnan = np.isnan(fsf['rmz_obs'])
        fsf = fsf[~isnan]['rmz_obs']
    
    if band=='W1':
        isnan = np.isnan(fsf['rmw1_obs'])
        fsf = fsf[~isnan]['rmw1_obs']
    
    print('DATA READ.')
    
    return fsf


def colour_table_lookup(dat, regmask, reg, band='R', band2='G', plot=False, fresh=False):
        
    print('band1:', band, 'band2:', band2)
    
    dcolbin=0.025    
    dzbin=0.0025
    xbins = np.arange(0.00, 0.6001, dzbin)
    
    if band2=='G':
        ybins = np.arange(-1.0, 4.001, dcolbin)
    
    if band2=='Z':
        ybins = np.arange(-2.0, 8.001, dcolbin)
        
    if band2=='W1':
        ybins = np.arange(-15.0, 25.001, dcolbin)

    ny=np.size(ybins)-2
    nx=np.size(xbins)-2
    print('nx,ny:',nx,ny)
    
    if (band=='R') & (band2 == 'G'):
        colvar = 'gmr_obs'
        zvar   = 'REST_GMR_0P1'
        suffix = ''
        index=0

    elif (band=='R') & (band2 == 'Z'):
        colvar = 'rmz_obs'
        dat[colvar] = dat['rmag'] - dat['zmag']
        zvar  = 'REST_RMZ_0P1'
        suffix = '_rmz'
        index=1

    elif (band=='R') & (band2 == 'W1'):
        colvar = 'rmw1_obs'
        dat[colvar] = dat['w1mag'] - dat['rmag']
        zvar = 'REST_RMW1_0P1'
        suffix = '_rmw1'
        index=2

    else:
        print('Bands not supported.')

    if fresh:
        print('GENERATING NEW LOOKUP TABLE.')
        dat_all = gen_JM_lookup(band=band2)  # read John's FSF catalogue
#        dat_all = modify_fsf_to_match_N_S(dat_all)         #Modify FSF to make N colour distribution match S
        rmask = (dat_all['reg'] == reg)   #select just N or S data as each has to be modelled separately
        dat_all=dat_all[rmask] # restrict all data to just this region
        
        x = dat_all['Z']
        y = dat_all[colvar]
        z = dat_all[zvar+'_original']
        
        # Determine the median rest frame colour in each bin of redshift and observer frame colour
        H, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, values = z, statistic='median', bins = [xbins, ybins], expand_binnumbers=True)
        count, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, values = z, statistic='count', bins = [xbins, ybins], expand_binnumbers=True)
        XX, YY = np.meshgrid(xedges, yedges)
 
        sam = True
        if (sam):
          # replace noisy and empty pixel values by neighouring values at the same redshift
          for idx in range(len(H)):
              mask = np.isnan(H[idx])
              try:
                  H[idx][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), H[idx][~mask])
              except:
                  pass # i.e: if whole column is empty
                # save the look-up table for future use
                
        opath='/global/homes/l/ldrm11/Cole/data/H_{}_noncloud{}.txt'.format(reg, suffix)
        #opath='./data/H_{}.txt'.format(reg)
        np.savetxt(opath, H)
        
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
        opath='/global/homes/l/ldrm11/Cole/data/H_{}{}.txt'.format(reg, suffix)
        print('save opath:', opath)
        #opath='./data/H_{}.txt'.format(reg)

        np.savetxt(opath, H)
        print('LOOK-UP TABLE GENERATED AND SAVED.')
        
        # Now apply to the FSF data to define the looked-up rest frame colour to use when binning the k-corrections
        regmask_all=(dat_all['reg']==reg) # this will always be true as we have already masked to this region
        colour_table_lookup(dat_all, regmask_all, reg, plot=plot, fresh=False, band=band, band2=band2)
                
        if plot:  # Compare colours from the look-up table with John's original colours
           for zmin in np.linspace(0.0,0.5,num=5): #For redshift bins plot histograms of the restframe colour distributions
             zmask= (dat_all['Z']> zmin) &(dat_all['Z']< zmin+0.125)
             mask=zmask
             jhist,bins=np.histogram(dat_all[zvar+'_original'][mask],bins=40,density=True)
             zlmask= (dat['Z']> zmin) &(dat['Z']< zmin+0.1)
             lhist,bins=np.histogram(dat_all[zvar][mask],bins=bins,density=True)
             binc= (bins[:-1]+bins[1:])/2.0   
             plt.plot(binc,jhist,'k-',label="John Moustakas")
             plt.plot(binc,lhist,'r-',label="Look-up Table")
            
             if index==0:
                 plt.xlabel(r'$g-r$')
             elif index==1:
                 plt.xlabel(r'$r-z$')
             else:
                 plt.xlabel(r'$r-w1$')

             plt.ylabel(r'N')
             plt.legend()
             plt.title('Colour distribution '+str(zmin)+'<z<'+str(zmin+0.125))
             plt.show()

        # now create corresponding k-corrections in bins of the assigned colour in the region we have already restricted two
                
        all_bins, all_medians = kg.gen_kcorr(dat_all, reg, colval=zvar, nbins=10, write=fresh, fill_between=True, plot=False, suffix=suffix)

        
        if plot == True:  #If requested make some plots to verify the k-corrections
              # k-correction difference plots.
              kcorr_r  = DESI_KCorrection(band='G', file='jmext', photsys=reg, suffix=suffix)
              prop_cycle = plt.rcParams['axes.prop_cycle']
              colors = prop_cycle.by_key()['color']
              
              #k_corr_file = './data/jmext_kcorr_{}_rband_z01.dat'.format(reg)
              k_corr_file = '/global/homes/l/ldrm11/Cole/data/jmext_kcorr_{}_rband_z01{}.dat'.format(reg,suffix)

              print('k_corr_file:', k_corr_file)
                
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
              if index==0:
                 ax[0].set_ylabel(r'$k_r(z, g-r)$')
              elif index==1:
                 ax[0].set_ylabel(r'$k_r(z, r-z)$')
              else:
                 ax[0].set_ylabel(r'$k_r(z, r-w1)$')

              ax[1].set_ylabel(r'$k_{r,med} - k_{r, curve}$')
              plt.show()

          
              #Compare the absolute magnitudes computed using the polynomial k-corrections to John Moustakas' orginal ones  
              dat_all['ABSMAG_SDSS_R_SAM'] = -99.9 # create an array to store the assigned absolute magnitudes
              rmask = (dat_all['reg'] == reg)
              Qzero = 0.0 # no evolution correction to be consistent with John
              dat_all['ABSMAG_SDSS_R_SAM'][rmask]=ca.ABSMAG(dat_all['rmag'][rmask],dat_all['Z'][rmask],dat_all['REST_GMR_0P1'][rmask],kcorr_r,Qzero)

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


        print('calling colour lookup table again.')

        # now call again to apply the look-up with the already saved look-up table
        colour_table_lookup(dat, regmask, reg, plot=plot, fresh=False, band=band, band2=band2)
        
    else:
        print('LOADING IN LOOKUP TABLE.')
        opath='/global/homes/l/ldrm11/Cole/data/H_{}{}.txt'.format(reg,suffix)
        print('opath:', opath)
        
        print('colvar:', colvar)
        print(min(dat[colvar]), max(dat[colvar]))
        
        if zvar not in dat.dtype.names:
            dat[zvar] = -99.9
            
        #print('zvar:', zvar)
        #print(min(dat[zvar]), max(dat[zvar]))
        
        #plt.hist(dat[colvar], bins=50)
        
        H = np.loadtxt(opath)

        if plot: # make a plot of the look-up table
            XX, YY = np.meshgrid(xbins, ybins)
            #fig   = plt.figure(figsize = (13,7))
            fig   = plt.figure(figsize = (10/3,3))
            ax1   = plt.subplot(111)
            plot1 = ax1.pcolormesh(XX,YY,H.T)
            if index==0:
                plt.ylabel(r'$g-r$', fontsize=9)
                label = r'$(g-r)_{0, med}$'
            elif index==1:
                plt.ylabel(r'$r-z$', fontsize=9)
                label = r'$(r-z)_{0, med}$'
            else:
                plt.ylabel(r'$r-w1$', fontsize=9)
                label = r'$(r-w1)_{0, med}$'
                
            cbar = plt.colorbar(plot1, ax=ax1, pad = .015, aspect=10, label=label)
            ax1.set_ylim(-0.2, 2)
            ax1.set_xlabel(r'$z$', fontsize=9)
            ax1.set_title('Lookup Table ({})'.format(reg))
            spath = '/global/homes/l/ldrm11/Cole/graphs/lookup_{}.pdf'.format(reg)
            plt.savefig(spath, bbox_inches='tight')
            plt.show() 

        Harray=np.asarray(H) # convert to numpy array to enable indexing     
        # use to generate grid indices for the  input data.
        
        
        print('colvar:', colvar)
        
        __, __, __, i2 = stats.binned_statistic_2d(dat['Z'][regmask], dat[colvar][regmask], values=dat['Z'][regmask], statistic='median', bins = [xbins, ybins], expand_binnumbers=True)
        j2=i2   #keep the original indices as useful for generating indices of CIC neighbouring cells
        i2=i2-1 #correct for offset in returned bin indices
        
        #Normal look-up
        #dat['REST_GMR_0P1'][regmask]=Harray[i2[0],i2[1]]
        
        #Cloud-in-Cell look-up
        #first compute the difference in the coordinate value of each data point and its nearest grid point in units of the bin spacing
        dx=-0.5-i2[0]+(dat['Z'][regmask]-xbins[0])/dzbin           # these should satisfy -0.5<dx<0.5          
        dy=-0.5-i2[1]+(dat[colvar][regmask]-ybins[0])/dcolbin   # these should satisfy -0.5<dy<0.5 
        #for each negative value we need to change of j2 to select the correct neighbouring cell 
        j2[0][(dx<0)]=j2[0][(dx<0)]-2
        j2[1][(dy<0)]=j2[1][(dy<0)]-2
        #CIC weights  (these add to unity)
        wa=(1.0-np.absolute(dx))*(1.0-np.absolute(dy))
        wb=     np.absolute(dx) *(1.0-np.absolute(dy))
        wc=(1.0-np.absolute(dx))*     np.absolute(dy)
        wd=     np.absolute(dx) *     np.absolute(dy)
        #To avoid out of bounds edge effects replace out of bound cells indices by the NGP cell index
        j0mask = (j2[0]>nx) | (j2[0]<0)
        j2[0][j0mask]=i2[0][j0mask]
        j1mask = (j2[1]>ny) | (j2[1]<0)
        j2[1][j1mask]=i2[1][j1mask]
        # Form the CIC weighted value
        dat[zvar][regmask] = wa*Harray[i2[0],i2[1]]+wb*Harray[j2[0],i2[1]]+wc*Harray[i2[0],j2[1]]+wd*Harray[j2[0],j2[1]]

        print('REST-FRAME COLOURS ASSIGNED.')
   
    return dat