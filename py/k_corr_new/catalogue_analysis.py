import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import colour_lookup as col
import numpy as np
from scipy import stats
from astropy.table import join,Table,Column,vstack, setdiff
from smith_kcorr   import DESI_KCorrection 
from rootfinders import root_itp,root_sec,root_itp2
from astropy.cosmology import FlatLambdaCDM
from scipy.ndimage import gaussian_filter

#from desiutil.plots import prepare_data, init_sky, plot_grid_map, plot_healpix_map, plot_sky_circles, plot_sky_binned

cosmo = FlatLambdaCDM(H0=100, Om0=0.313, Tcmb0=2.725)   #Standard Planck Cosmology in Mpc/h units
import kcorr_generator as kg
from   cosmo2              import volcom, distmod
from schechter import schechter, named_schechter


########################################################################################################################    
    
#Define Sample selection cuts and their display styles
#and other "global" variables so they can be defined in any routine
def selection(reg, Qevol=0.67, survey='Y1'):
    f_ran=1.0 #random sampling fraction for quick run throughs
    #Qevol=0.67 #Assumed luminosity evolution parameter
    
    area_N_DA1=2393.4228/(4*np.pi*(180.0/np.pi)**2)
    area_S_DA1=5358.2728/(4*np.pi*(180.0/np.pi)**2)

    zmin = 0.002
    zmax = 0.6
    
    # Alex tests
    #zmin, zmax = 0.05, 0.3
    # DDP1:
    #zmin, zmax= 0.019, 0.272
    
    if survey == 'sv3':
        area_N_DA1=90.0/(4*np.pi*(180.0/np.pi)**2)
        area_S_DA1=90.0/(4*np.pi*(180.0/np.pi)**2)

        zmin = 0.01
        zmax = 0.5
    #zmin = max(0.01, np.min(dat[zmask]['Z']))
    #zmax = max(0.5,  np.max(dat[zmask]['Z']))
    
    
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


########################################################################################################################


# ABSMAG_R= appmag -DMOD  -kcorr_r.k(z, rest_GMR) +Qevol*(z-0.1) 
def ABSMAG(appmag,z,rest_GMR,kcorr_r,Qevol):
        DMOD=25.0+5.0*np.log10(cosmo.luminosity_distance(z).value)  
        ABSMAG=appmag-DMOD-kcorr_r.k(z,rest_GMR)+Qevol*(z-0.1)
        return ABSMAG
    
# Make plots of the k-corrections to check they are sensible and smooth in both redshift and colour
########################################################################################################################
def plot_kcorr(regions):
    # extract the default colour sequence to have more control of line colors
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for reg in regions:
        # set up the k-corrections for this photometry system region
        kcorr_r  = DESI_KCorrection(band='R', file='jmext', photsys=reg)
        Sel=selection(reg) # define selection parameters for this region
        z=np.linspace(0.0,0.6,300)
        icol=-1
        #for rest_GMR in np.array([0.39861393, 0.53434181, 0.6534462 , 0.76661587, 0.86391068,
       #0.93073082, 0.9832058 ]): #
            
        for rest_GMR in np.linspace(0.0,1.1,6):   
            GMR=rest_GMR*np.ones(z.size)
            icol += 1
            k=kcorr_r.k(z, GMR)
            label=reg+': G-R='+np.array2string(rest_GMR)
            plt.plot(z,k,label=label,color=colors[icol],linestyle=Sel['style']) 
    plt.xlabel('$z$')    
    plt.ylabel('$k^r(z)$')  
    plt.legend(loc=(1.04,0))
    spath = '/global/homes/l/ldrm11/Cole/graphs/kcorr.png'
    plt.savefig(spath)
    spath = '/global/homes/l/ldrm11/Cole/graphs/kcorr.pdf'
    plt.savefig(spath)
    
    plt.show()


    return
########################################################################################################################

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

    #remove z_ref=0.0 columns to avoid mis-use
    #dat.remove_column('ABSMAG_RP0')
    #dat.remove_column('REST_GMR_0P0')
    #dat['REST_GMR_0P1']=0.0

    # Add derived columns

    #Apparent magnitudes
    dat.add_column(Column(name='gmag', data=22.5-2.5*np.log10(dat['flux_g_dered']) ))
    dat.add_column(Column(name='rmag', data=22.5-2.5*np.log10(dat['flux_r_dered']) ))
    dat.add_column(Column(name='zmag', data=22.5-2.5*np.log10(np.clip(dat['flux_z_dered'],1.0e-10,None)) ))
    dat.add_column(Column(name='w1mag', data=22.5-2.5*np.log10(np.clip(dat['flux_w1_dered'],1.0e-10,None)) ))
    dat.add_column(Column(name='w2mag', data=22.5-2.5*np.log10(np.clip(dat['flux_w2_dered'],1.0e-10,None)) ))

    #Observerframe colour
    dat.add_column(Column(name='gmr_obs', data=dat['gmag']-dat['rmag']))
    dat.add_column(Column(name='OBS_GMR', data=dat['gmag']-dat['rmag']))
    
    return dat
########################################################################################################################

#Plot how z_max depends on absolute magnitude and colour code by rest frame colour
def plot_zmax_absmag(dat):
    cmap= plt.get_cmap('jet')
    col=np.clip(((dat['REST_GMR_0P1']+0.5)/2.0),0.0,1.0)
    plt.scatter(dat['ABSMAG_RP1'],dat['zmax'], marker='.', c=col ,cmap=cmap, linewidths=0,s=0.25,alpha=0.2)
    plt.xlim([-12,-23])
    plt.ylim([0.0,0.6])
    plt.ylabel('$z_{max}$')
    plt.xlabel('$M_r - 5 \log h$')
    plt.show()
    return
########################################################################################################################
#Plot how z_min depends on absolute magnitude and colour code by rest frame colour
def plot_zmin_absmag(dat):
    cmap= plt.get_cmap('jet')
    col=np.clip(((dat['REST_GMR_0P1']+0.5)/2.0),0.0,1.0)
    plt.scatter(dat['ABSMAG_RP1'],dat['zmin'], marker='.', c=col ,cmap=cmap, linewidths=0,s=0.25,alpha=0.2)
    plt.xlim([-12,-23])
    #plt.ylim([0.0,0.1])
    plt.ylabel('$z_{min}$')
    plt.xlabel('$M_r - 5 \log h$')
    plt.show()
    return

########################################################################################################################

#Plot how z_max depends z colour code by absolute magnitude
def plot_zmax_z(dat):
    cmap= plt.get_cmap('jet')
    col=np.clip(((dat['ABSMAG_RP1']+22)/10.0),0.0,1.0)
    plt.scatter(dat['Z'],dat['zmax'], marker='.', c=col ,cmap=cmap, linewidths=0,s=0.25,alpha=0.2)
    plt.xlim([0.0,0.6])
    plt.ylim([0.0,0.6])
    plt.ylabel('$z_{max}$')
    plt.xlabel('$z$')
    plt.show()
    return

########################################################################################################################

#Plot how z_min depends z colour code by absolute magnitude
def plot_zmin_z(dat):
    cmap= plt.get_cmap('jet')
    col=np.clip(((dat['ABSMAG_RP1']+22)/10.0),0.0,1.0)
    plt.scatter(dat['Z'],dat['zmin'], marker='.', c=col ,cmap=cmap, linewidths=0,s=0.25,alpha=0.2)
    plt.xlim([0.0,0.6])
    #plt.ylim([0.0,0.6])
    plt.ylabel('$z_{min}$')
    plt.xlabel('$z$')
    plt.show()
    return



########################################################################################################################
def recompute_rest_col_mag(dat,regions, fresh=False, test_plots=True, band='R', band2='G', Qevol=0.67):
        
    if (band=='R') & (band2 == 'G'):
        colvar = 'gmr_obs'
        zvar   = 'REST_GMR_0P1'
        suffix = ''
        index  = 0
    elif (band=='R') & (band2 == 'Z'):
        colvar = 'rmz_obs'
        zvar  = 'REST_RMZ_0P1'
        suffix = '_rmz'
        index = 1
    elif (band=='R') & (band2 == 'W1'):
        colvar = 'rmw1_obs'
        zvar = 'REST_RMW1_0P1'
        suffix = '_rmw1'
        index=2
    else:
        print('Bands not supported.')
        
    if fresh:
        fsf = col.gen_JM_lookup(band=band2)

    for reg in regions:
      print('starting region ',reg)
      Sel=selection(reg, Qevol) # define selection parameters for this region
      regmask=(dat['reg']==reg)#mask to select objects in specified region

      if fresh == True:
          fsfmask=(fsf['reg']==reg)
          col.colour_table_lookup(fsf, fsfmask, reg, plot=False, fresh=fresh, band=band, band2=band2) 
            
          print('Entering k-corr generation in recompute-rest-col-mag.')
          all_bins, all_medians = kg.gen_kcorr(fsf[fsfmask], reg, colval=zvar, nbins=10, write=fresh, fill_between=True, plot=False, suffix=suffix)

        
          if test_plots==True:
                
              print('PLOTTING TEST PLOTS.')
              # k-correction difference plots.
              kcorr_r  = DESI_KCorrection(band='R', file='jmext', photsys=reg, suffix=suffix)

              prop_cycle = plt.rcParams['axes.prop_cycle']
              colors = prop_cycle.by_key()['color']
                
                
              k_corr_file = '/global/homes/l/ldrm11/Cole/data/jmext_kcorr_{}_rband_z01{}.dat'.format(reg,suffix)
            
              #k_corr_file = '/global/u2/s/smcole/DESI/NvsS/data/jmext_kcorr_{}_rband_z01.dat'.format(reg)
            
              col_min, col_max, A, B, C, D, E, F, G, col_med = np.loadtxt(k_corr_file, unpack=True)

              fig,ax = plt.subplots(2,1, figsize=(5,10))
              prop_cycle = plt.rcParams['axes.prop_cycle']
              colours = prop_cycle.by_key()['color']
            
              for idx in range(len(all_bins)):
                    z            = all_bins[idx]
                    medians      = all_medians[idx]
                    rest_GMR     = col_med[idx]
                    GMR          = rest_GMR * np.ones_like(z)
                    k            = kcorr_r.k(z, GMR)

                    ax[0].plot(z, medians, color=colours[idx], label='{:.4f}'.format(rest_GMR))
                    ax[0].plot(z, k, color=colours[idx])

                    ax[1].plot(z, medians-k, color=colours[idx])
                    ax[1].axhline(0.0, ls='--', lw=0.5, color='black')

              ax[0].legend()
              ax[1].set_xlabel('Z')
              if index==0:
                  ax[0].set_ylabel(r'$k_r(z,g-r)$')
              elif index==1:
                  ax[0].set_ylabel(r'$k_r(z,r-z)$')     
              else:
                  ax[0].set_ylabel(r'$k_r(z,r-w1)$')
                
              ax[1].set_ylabel(r'$k_{r,med} - k_{r, curve}$')
              plt.show()

              ###############################################
              # abs. mag difference plot.
              idx = 7
              print(col_min[idx], col_max[idx], col_med[idx])
              GMR_med = col_med[idx]
              GMR_min = col_min[idx]
              GMR_max = col_max[idx]
              fsf['ABSMAG_SDSS_R_SAM'] = -99.9
              rmask = (fsf['reg'] == reg)
              Qzero = 0.0 # no evolution correction to be consistent with John
              fsf['ABSMAG_SDSS_R_SAM'][rmask]=ABSMAG(fsf['RMAG_DRED'][rmask],fsf['Z'][rmask],fsf['REST_GMR_0P1'][rmask]*0.+GMR_med,kcorr_r,Qzero)

              rcmask=rmask & (fsf[zvar]>GMR_min)  & (fsf[zvar]<GMR_max)
              diff = fsf['ABSMAG_SDSS_R_SAM'][rcmask]-fsf['ABSMAG_SDSS_R'][rcmask]
              bin_medians, bin_edges, binnumber = stats.binned_statistic(fsf[rcmask]['Z'], diff, statistic='median', bins=25)
            
              plt.scatter(fsf['Z'][rcmask],diff, marker='.', color='red', linewidths=0,s=0.2,alpha=0.4)
              centres = (bin_edges[:-1]+bin_edges[1:])/2
              plt.plot(centres, bin_medians, label='median')
              plt.ylim(-0.025, 0.025)
              plt.axhline(0.0, ls='--', color='black')
              plt.axhline(np.mean(diff), ls='--', color='blue')
              plt.xlabel('Z')
              plt.ylabel('SM-JM Mr')
              plt.legend()
              plt.title('{:.3f}<(g-r)_0<{:.3f}'.format(GMR_min, GMR_max))
              plt.show()

                
      col.colour_table_lookup(dat, regmask, reg, plot=test_plots, fresh=False,band=band, band2=band2) 
      kcorr_rM  = DESI_KCorrection(band='R', file='jmext', photsys=reg, suffix=suffix) #set k-correction for region
      dat['ABSMAG_RP1'][regmask]=ABSMAG(dat['rmag'][regmask],dat['Z'][regmask],dat['REST_GMR_0P1'][regmask],kcorr_rM,Sel['Qevol'])
        
    return


#################################################### ##################################################################

# make a histogram comparing the redshift distributions to selection limits
def hist_nz(dat,regions):
    bins = np.arange(0, 0.6, 0.01)
    bin_cen=(bins[:-1] + bins[1:]) / 2.0
    for reg in regions:
        Sel=selection(reg) # define selection parameters for this region
        kcorr_r  = DESI_KCorrection(band='R', file='jmext', photsys=reg) #set k-correction for region
        regmask=(dat['reg']==reg)#mask to select objects in specified region    
        mask = (regmask) & (dat['Z'] > Sel['zmin']) & (dat['rmag'] < Sel['faint']) & (dat['Z'] < Sel['zmax']) & (dat['rmag'] > Sel['bright']) #sample selection
        wcount,binz=np.histogram(dat[mask]['Z'], bins=bins,  density=True, weights=dat[mask]['WEIGHT'])
        count,binz=np.histogram(dat[mask]['Z'], bins=bins,  density=True)
        plt.plot(bin_cen, wcount, label='Weighted '+reg, color=Sel['col'], linestyle=Sel['style'])
        
        plt.plot(bin_cen, count, label=reg, linewidth=0.5, color=Sel['col'], linestyle=Sel['style'])
    plt.xlabel('z')
    plt.ylabel('dP/dz')
    plt.xlim([0,0.6])
    plt.ylim([0,4.4])
    plt.legend()
    plt.show()
    return
#################################################################################################################

# magnitude-reshifts scatterplot
# to check data extends to the selection limits
def plot_mag_z(dat,regions,contours=False):
    # Plotting ranges
    range=[[0.0,0.65],[15.5,19.55]]
    sigma=1.0 # smoothing to apply before making contours

    def flatten(l):        # limits of data to use in histogram()
        return [item for sublist in l for item in sublist]

    for reg in regions:
        Sel=selection(reg) # define selection parameters for this region
        regmask=(dat['reg']==reg)#mask to select objects in specified region  
        # Selection to remove stars and apply faint magnitude
        mask = (regmask) & (dat['Z'] > Sel['zmin']) & (dat['Z'] < Sel['zmax']) &  (dat['rmag'] < Sel['faint']) #sample selection
        print('All:',reg,'z range:',dat[regmask]['Z'].min(),dat[regmask]['Z'].max())
        print('Selected:',reg,'z range:',dat[regmask]['Z'].min(),dat[regmask]['Z'].max())
        print('All:',reg,'rmag range:',dat[regmask]['rmag'].min(),dat[regmask]['rmag'].max())
        print('Selected:',reg,'rmag range:',dat[regmask]['rmag'].min(),dat[regmask]['rmag'].max())
        # scatter plot
        plt.scatter(dat[mask]['Z'], dat[mask]['rmag'], marker='.', linewidths=0, alpha=0.5, s=0.25, color=Sel['col'])
        # plot limits
        plt.plot(range[0],[Sel['faint'],Sel['faint']], color=Sel['col'],linestyle='solid',linewidth=0.2)
        plt.plot(range[0],[Sel['bright'],Sel['bright']], color=Sel['col'],linestyle='solid',linewidth=0.2)
        plt.plot([Sel['zmin'],Sel['zmin']],range[1], color=Sel['col'],linestyle='solid',linewidth=0.2)
        plt.plot([Sel['zmax'],Sel['zmax']],range[1], color=Sel['col'],linestyle='solid',linewidth=0.2)
        if contours:
            extent=flatten(range)  # flattened version to use in contour()
            counts,x,y=np.histogram2d(dat[mask]['Z'], dat[mask]['rmag'],bins=100, range=range)
            counts=gaussian_filter(counts,sigma)*1.0e+06/np.count_nonzero(mask) #contour units arbitrary!
            plt.contour(counts.transpose(),extent=extent,levels=[20,40,80,160,320,640],colors=Sel['col'])
    

    plt.xlim(range[0])
    plt.ylim(range[1])
    plt.xlabel('z')
    plt.ylabel('r')
    plt.show()
    return
######################################################################################################################
# Colour-Magnitude Scatter plot with marginal histograms
def plot_col_mag(dat,regions):
    def flatten(l):        # limits of data to use in histogram()
        return [item for sublist in l for item in sublist]

    range=[[-0.1,1.3],[-23.99,-14]] # range to use for colour and absolute magnitude limits (both selection and plotting limits)
    nbin=100                       # number of bins per dimension
    sigma=1                        # gaussian smoothing in units of bin width
    levels= [160,320,640,1280,2560,5120,10240,20480] # contour levels in points per bin

    # Start with a square Figure.
    fig = plt.figure(figsize=(6, 6))
    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.0, hspace=0.0)
    # Create the Axes.
    axcolmag = fig.add_subplot(gs[1, 0])
    axcolmag_histx = fig.add_subplot(gs[0, 0], sharex=axcolmag)
    axcolmag_histy = fig.add_subplot(gs[1, 1], sharey=axcolmag)

    axcolmag_histx.tick_params(axis="x", labelbottom=False) #suppress default axis labels
    axcolmag_histy.tick_params(axis="y", labelleft=False)

    

    #Marginal Colour distribution
    for reg in regions:
        Sel=selection(reg) # define selection parameters for this region
        #Object selection mask
        smask=(dat['Z'] > Sel['zmin']) & (dat['REST_GMR_0P1']>-0.1)  & (dat['REST_GMR_0P1']<1.4) & (dat['rmag']>Sel['bright']) & (dat['rmag'] < Sel['faint'])
        regmask = (dat['reg']==reg)
        #sample selection
        mask = (regmask) & (smask)
        hist,x=np.histogram(dat['REST_GMR_0P1'][mask],bins=nbin,range=range[0])
        hist=hist/Sel['area']
        bin_centres= 0.5*(x[1:]+x[:-1])
        axcolmag_histx.plot(bin_centres,hist,color=Sel['col'])
        ymax=1.1*hist.max()
        axcolmag_histx.set(ylim=[0,ymax])

    #Marginal absolute magnitude distribution
    for reg in regions:
        Sel=selection(reg) # define selection parameters for this region
        #Object selection mask
        smask=(dat['Z'] > Sel['zmin']) & (dat['REST_GMR_0P1']>-0.1)  & (dat['REST_GMR_0P1']<1.4) & (dat['rmag']>Sel['bright']) & (dat['rmag'] < Sel['faint'])
        regmask = (dat['reg']==reg)
        #sample selection
        mask = (regmask) & (smask)
        hist,x=np.histogram(dat['ABSMAG_RP1'][mask],bins=nbin,range=range[1])
        hist=hist/Sel['area']
        bin_centres= 0.5*(x[1:]+x[:-1])
        axcolmag_histy.plot(hist,bin_centres,color=Sel['col'])
        xmax=1.1*hist.max()
        axcolmag_histy.set(xlim=[0,xmax])
        
    #2D scatter plots
    for reg in regions:
        Sel=selection(reg) # define selection parameters for this region
        #Object selection mask
        smask=(dat['Z'] > Sel['zmin']) & (dat['REST_GMR_0P1']>-0.1)  & (dat['REST_GMR_0P1']<1.4) & (dat['rmag']>Sel['bright']) & (dat['rmag'] < Sel['faint'])
        regmask = (dat['reg']==reg)
        #sample selection
        mask = (regmask) & (smask)
        axcolmag.scatter(dat[mask]['REST_GMR_0P1'],dat[mask]['ABSMAG_RP1'], marker='.', linewidths=0,s=0.25,color=Sel['col'],alpha=0.1,label='South')
        #Contours of the point distribution in the scatter plots
        extent=flatten(range)  # flattened version to use in contour()
        counts,x,y=np.histogram2d(dat['REST_GMR_0P1'][mask],dat['ABSMAG_RP1'][mask],bins=nbin,range=range)
        counts=gaussian_filter(counts,sigma)/Sel['area']
        axcolmag.contour(counts.transpose(),extent=extent,levels=levels, colors=Sel['col'])



    range[1].reverse()  #flip the Absolute magnitude axes direction
    axcolmag.set(xlabel='$M_g - M_r$', ylabel='$M_r -5 \log h$',xlim=range[0],ylim=range[1])
    Sel=selection('N')
    colN=Sel['col']
    Sel=selection('S')
    colS=Sel['col']
    axcolmag.legend(['North','South'],loc='upper left',labelcolor=[colN,colS])
    
    
    spath = '/global/homes/l/ldrm11/Cole/graphs/colmag.png'
    plt.savefig(spath, bbox_inches='tight')
    
    spath = '/global/homes/l/ldrm11/Cole/graphs/colmag.pdf'
    plt.savefig(spath, bbox_inches='tight')
    
    del mask,smask,regmask # tidy up
    del axcolmag,axcolmag_histx,axcolmag_histy
    return
######################################################################################################################

####################################################################################################################### All-sky maps 
#  see https://notebook.community/desihub/desiutil/doc/nb/SkyMapExamples for examples of how to make plots like these
def sky_plot(dat,regions):

    # All-sky scatter plot with galactic plane and ecliptic marked
    ax= init_sky()
    for reg in regions:
        regmask = (dat['reg']==reg)
        Sel=selection(reg) # define selection parameters for this region
        p = ax.scatter(ax.projection_ra(dat['RA'][regmask]),ax.projection_dec(dat['DEC'][regmask]),color=Sel['col'],s=0.25, marker='.', linewidths=0)


    # healpix source density map in healpix's of less than max_bin_area sq degrees
    ax= plot_sky_binned(dat['RA'] ,dat['DEC'], plot_type='healpix', max_bin_area=4.0, verbose=True)
    return
######################################################################################################################
#Cone plot
def cone_plot(dat,regions):
    #This is hardwired to produce two particular cone plots but could be adapted
    for reg in regions:
        regmask = (dat['reg']==reg)
        Sel=selection(reg) # define selection parameters for this region
        # Selection to remove stars and impose chosen magnitude limit
        #mask = (regmask) & (dat['Z'] > Sel['zmin']) & (dat['rmag'] < Sel['faint']) &  (dat['DEC'] > 30.0)  &  (dat['DEC'] < 35.0)  &  (dat['RA'] > 180.0) &  (dat['RA'] < 280.0)
        
        mask = (regmask) & (dat['Z'] > Sel['zmin']) & (dat['rmag'] < Sel['faint']) &  (dat['DEC'] > -5)  &  (dat['DEC'] < 0)  &  (dat['RA'] > 180.0) &  (dat['RA'] < 280.0)
        
        print(np.count_nonzero(regmask),' points in the cone plot')
        offset=+1.5*np.pi/2.0
        x=dat[mask]['Z']*np.sin(np.radians(dat[mask]['RA'])+offset)
        y=dat[mask]['Z']*np.cos(np.radians(dat[mask]['RA'])+offset)
        plt.axis('equal')
        plt.axis('off')
        plt.scatter(x,y,color=Sel['col'],s=0.25, marker='.', linewidths=0)
        ra_min=dat[mask]['RA'].min()
        ra_max=dat[mask]['RA'].max()
        ra=np.linspace(ra_min,ra_max,100)
        z_max=dat[mask]['Z'].max()
        frame_x=np.concatenate((np.array([0]),z_max*np.sin(np.radians(ra)+offset),np.array([0])))
        frame_y=np.concatenate((np.array([0]),z_max*np.cos(np.radians(ra)+offset),np.array([0])))
        plt.plot(frame_x,frame_y,color='black')
        plt.show()
    del x,y,ra # tidy up
    return
######################################################################################################################

def colour_table_lookup(dat, regmask, reg, suffix='', plot=True):
        
    
    print('LOADING IN LOOKUP TABLE.')
    H = np.loadtxt('H_{}{}.txt'.format(reg, suffix))
    
        
    xbins = np.arange(0.0, 0.61, 0.0025)  #dangeroulsy hard wired!!!
    ybins = np.arange(-1.0, 4.001, 0.025)
    ybins = np.arange(-4.0, 8.001, 0.025)


    if plot:
        XX, YY = np.meshgrid(xbins, ybins) 
        fig   = plt.figure(figsize = (13,7))
        ax1   = plt.subplot(111)
        plot1 = ax1.pcolormesh(XX,YY,H.T)
        label = r'$(g-r)_{0, med}$'
        cbar = plt.colorbar(plot1, ax=ax1, pad = .015, aspect=10, label=label)
        plt.xlabel(r'$g-r$')
        plt.ylabel(r'$z$')
        plt.title('Test of Nearest Pixel Lookup.')
        plt.show()
        
        
        
    # use to generate grid on input data.
    __, __, __, binnumber2 = stats.binned_statistic_2d(dat[regmask]['Z'], dat[regmask]['gmr_obs'], values=dat[regmask]['Z'], statistic='median', bins = [xbins, ybins], expand_binnumbers=True)
    Harray=np.asarray(H) # convert to numpy array to enable indexing    
    dat['REST_GMR_0P1'][regmask] = Harray[binnumber2[0],binnumber2[1]]
  
    print('REST-FRAME COLOURS ASSIGNED.')
    
    return 
#####################################################################################################################################################################################################

def gen_weight_files():
    
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.table import join,Table,Column,vstack
    full = Table.read('/pscratch/sd/l/ldrm11/BGS_ANY_full.dat.fits')

    mask = (full['ZWARN']*0 == 0) & (full['ZWARN'] != 999999) & (full['GOODHARDLOC'] == 1) &(22.5-2.5*np.log10(full['FLUX_R']/full['MW_TRANSMISSION_R'])< 19.5) & (full['TSNR2_BGS'] >1000)
    observed=full[mask]
    mask_zgood =(observed['DELTACHI2'] > 40) & (observed['ZWARN'] == 0)
    mask_zbad=(observed['DELTACHI2'] < 40) | (observed['ZWARN'] != 0)
    zbad=observed[mask_zbad]
    zgood=observed[mask_zgood]

    observed['rmag']=22.5-2.5*np.log10(observed['FLUX_R']/observed['MW_TRANSMISSION_R'])
    observed['fib_rmag']=22.5-2.5*np.log10(observed['FIBERFLUX_R']/observed['MW_TRANSMISSION_R'])
    zbad['rmag']=22.5-2.5*np.log10(zbad['FLUX_R']/zbad['MW_TRANSMISSION_R'])
    zbad['fib_rmag']=22.5-2.5*np.log10(zbad['FIBERFLUX_R']/zbad['MW_TRANSMISSION_R'])
    zgood['rmag']=22.5-2.5*np.log10(zgood['FLUX_R']/zgood['MW_TRANSMISSION_R'])
    zgood['fib_rmag']=22.5-2.5*np.log10(zgood['FIBERFLUX_R']/zgood['MW_TRANSMISSION_R'])
    
    return zgood, zbad, observed

######################################################################

def load_weight_catalogues():

    survey = 'Y1'
    fpathN = '/pscratch/sd/a/amjsmith/Y1/LSS/iron/LSScats/v0.5/BGS_BRIGHT_N_clustering.dat.fits'
    fpathS = '/pscratch/sd/a/amjsmith/Y1/LSS/iron/LSScats/v0.5/BGS_BRIGHT_S_clustering.dat.fits'
    dat = load_catalogues(fpathN, fpathS)

    reg = 'S' # assume redshift limits same for N and S.
    min_z_val = selection(reg)['zmin'] # should be 0.01
    max_z_val = selection(reg)['zmax'] # should be 0.5

    zmask = (dat['Z'] > min_z_val) & (dat['Z'] < max_z_val)
    rmask = ()
    mask = zmask
    dat = dat[mask]

    zgood, zbad, observed = gen_weight_files()
    
    obs = observed['TARGETID', 'TSNR2_BGS', 'fib_rmag']

    dat = join(dat, obs, keys='TARGETID')
    
    return dat, zgood, zbad, observed

######################################################################

def weight_table_lookup(dat, plot=True, fresh=False):

    ___, zgood, zbad, observed = load_weight_catalogues()
    
    obs = observed['TARGETID', 'TSNR2_BGS', 'fib_rmag']
    dat = join(dat, obs, keys='TARGETID')
    
    dat['WEIGHT_ZLOOKUP_SIMPLE'] = 1.0000001
    dat['WEIGHT_ZLOOKUP']        = 1.0000001

    pixel_size = 2
    fresh = True
    plot  = True
    dxbin = 25 * pixel_size
    dybin = 0.025 * pixel_size    
    xbins = np.arange(0, 15000, dxbin)
    ybins = np.arange(14, 25, dybin)
    ny=np.size(ybins)-2
    nx=np.size(xbins)-2
    print('nx,ny:',nx,ny)

    regions = ['N', 'S']

    for reg in regions:

        zgoodmask = (zgood['PHOTSYS'] == reg)      & (zgood['TSNR2_BGS'] > xbins[0]) & (zgood['TSNR2_BGS'] < xbins[-2])
        zbadmask  = (zbad['PHOTSYS'] == reg)       & (zbad['TSNR2_BGS'] > xbins[0]) & (zbad['TSNR2_BGS'] < xbins[-2])
        obsmask   = (observed['PHOTSYS'] == reg)   & (observed['TSNR2_BGS'] > xbins[0]) & (observed['TSNR2_BGS'] < xbins[-2])
        regmask   = (dat['reg'] == reg)            & (dat['TSNR2_BGS'] > xbins[0]) & (dat['TSNR2_BGS'] < xbins[-2])

        if fresh:

            #zgood, zbad, observed = gen_weight_files()
            xg = zgood[zgoodmask]['TSNR2_BGS']
            yg = zgood[zgoodmask]['fib_rmag']
            xb = zbad[zbadmask]['TSNR2_BGS']
            yb = zbad[zbadmask]['fib_rmag']
            xall = observed[obsmask]['TSNR2_BGS']
            yall = observed[obsmask]['fib_rmag']

            binx, biny = xbins, ybins
            Hg, xedges_g, yedges_g, binnumber_g = stats.binned_statistic_2d(xg, yg, None, 'count', bins=[binx, biny], expand_binnumbers=True)
            Hb, xedges_b, yedges_b, binnumber_b = stats.binned_statistic_2d(xb, yb, None, 'count', bins=[binx, biny], expand_binnumbers=True)
            Hall, xedges_all, yedges_all, binnumber_all = stats.binned_statistic_2d(xall, yall, None, 'count', bins=[binx, biny], expand_binnumbers=True)
            
            H = Hall*0.0+1.0
            nonzero_mask = (Hall >0)
            H[nonzero_mask] = (Hg[nonzero_mask]/Hall[nonzero_mask])
            
            XXg, YYg = np.meshgrid(xedges_g, yedges_g)
            XXb, YYb = np.meshgrid(xedges_b, yedges_b)

            if plot:
                fig = plt.figure(figsize = (13,7))
                ax1=plt.subplot(111)
                plot1 = ax1.pcolormesh(XXg,YYg,H.T)
                cbar = plt.colorbar(plot1,ax=ax1, pad = .015, aspect=10)
                plt.xlabel('TSNR2_BGS')
                plt.ylabel('r_fibre')
                plt.show()

            # save the look-up table for future use
            opath='/global/homes/l/ldrm11/Cole/data/w_{}.txt'.format(reg)
            #opath='./data/H_{}.txt'.format(reg)

            np.savetxt(opath, H)
            print('LOOK-UP TABLE GENERATED AND SAVED.')

            xvar = 'TSNR2_BGS'
            yvar = 'fib_rmag'
            Harray=np.asarray(H) # convert to numpy array to enable indexing     
            # use to generate grid indices for the  input data.
            __, __, __, i2 = stats.binned_statistic_2d(dat[xvar][regmask], dat[yvar][regmask], values=dat[xvar][regmask], statistic='median', bins = [xbins, ybins], expand_binnumbers=True)
            j2=i2   #keep the original indices as useful for generating indices of CIC neighbouring cells
            i2=i2-1 #correct for offset in returned bin indices

            #Normal look-up
            dat['WEIGHT_ZLOOKUP_SIMPLE'][regmask]=Harray[i2[0],i2[1]]

            #Cloud-in-Cell look-up
            #first compute the difference in the coordinate value of each data point and its nearest grid point in units of the bin spacing
            dx=-0.5+i2[0]+(dat[xvar][regmask]-xbins[0])/dxbin           # these should satisfy -0.5<dx<0.5          
            dy=0.5+i2[1]-(dat[yvar][regmask]-ybins[0])/dybin   # these should satisfy -0.5<dy<0.5 
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
            dat['WEIGHT_ZLOOKUP'][regmask] = wa*Harray[i2[0],i2[1]]+wb*Harray[j2[0],i2[1]]+wc*Harray[i2[0],j2[1]]+wd*Harray[j2[0],j2[1]]
            print('ZWEIGHT ASSIGNED.')
            
            
    return dat
    

def load_full_catalogue_subsets():
    
    zmin = 0.0  # Should be the minimum redshift cut used in making the clustering catalogue and should probably be set to exlcude stars i.e. 0.00125 or so
    full = Table.read('/pscratch/sd/a/amjsmith/Y1/LSS/iron/LSScats/v0.5/BGS_BRIGHT_full.dat.fits')
    
    # Compute magnitudes and fibre magnitudes and add to the full table
    full.add_column(Column(name='rmag', data=22.5-2.5*np.log10(full['FLUX_R']/full['MW_TRANSMISSION_R']) ))
    full.add_column(Column(name='fib_rmag', data=22.5-2.5*np.log10(full['FIBERFLUX_R']/full['MW_TRANSMISSION_R']) ))
    print('size of Full catalogue',full['rmag'].size)

    #Throw away objects with fibre mag>23 as they should not be in BGS
    full = full[(full['fib_rmag']<24.0)] 
    print('size of Full catalogue after r_fib>23 cut',full['rmag'].size)
    print('Faintest Fibre mag:',np.amax(full['fib_rmag']))
    #full.info('stats')

    #The observed subset is defined by those reachable by a fibre (ZWARN not Nan) that isn't broken (ZWARN=999999) that achieves a GOODHARDLOC and is predicted to yield decent SNR
    mask = (full['ZWARN']*0 == 0) & (full['ZWARN'] != 999999) & (full['GOODHARDLOC'] == 1)  & (full['TSNR2_BGS']>1000.0)
    #From the observed subset we ideally want to remove all stars but settle for removing just the spectroscopically confirmed stars
    mask = ~mask  | (full['Z_not4clus']<zmin) #true for all the unobserved (~mask) and all the observed that are stars (z<zmin)
    mask = ~mask  # now false for the above, i.e. true for all observed that aren't spectroscopically confirmed to be z<zmin (stars)
    observed=full[mask]
    #The subset of galaxies with good redshift have no ZWARN have large DELTACHI2 and are above zmin (i.e. not stars)
    mask_zgood =(observed['DELTACHI2'] > 40) & (observed['ZWARN'] == 0) & (observed['Z_not4clus']>zmin)
    zgood=observed[mask_zgood]
    
    #zgood.info('stats')
    print('number of objects in the observed subset',  observed['TARGETID'].size)
    print('number of objects in the good redshift subset',zgood['TARGETID'].size)
    #observed.info('stats')
    print('Faintest Observed Fibre mag:',np.amax(observed['fib_rmag']))
    
    
    return zgood, observed



def load_weight_catalogues():

    #Read the corresponding clustering catalogue and check it agrees with the zgood catalogue defined in load_full_catalogue_subsets()
    survey = 'Y1'
    fpathN = '/pscratch/sd/a/amjsmith/Y1/LSS/iron/LSScats/v0.5/BGS_BRIGHT_N_clustering.dat.fits'
    fpathS = '/pscratch/sd/a/amjsmith/Y1/LSS/iron/LSScats/v0.5/BGS_BRIGHT_S_clustering.dat.fits'
    clus = load_catalogues(fpathN, fpathS)
    print('number of objects in the clustering file',clus['TARGETID'].size)

 
    zgood, observed = load_full_catalogue_subsets()
    
    #Compare zgood and the clus catalogues and check they contain exactly the same set of objects
    diff=setdiff(zgood,clus,keys=['TARGETID'])
    print('The number of objects in zgood that do not match with clus',diff['TARGETID'].size,' Should be zero!')
    diff=setdiff(clus,zgood,keys=['TARGETID'])
    print('The number of objects in clus that do not match with zgood',diff['TARGETID'].size,' Should be zero!')   

    return clus, zgood, observed



def weight_table_lookup(clus, plot=True, fresh=False, regions=None):

    ___, zgood, observed = load_weight_catalogues()
    
    clus['WEIGHT_ZLOOKUP_SIMPLE'] = 1.0   #Default weight to be used outside the grid and in cells in which the completeness is ill-defined
    clus['WEIGHT_ZLOOKUP']        = 1.0
    
    nx=60
    ny=44
    fresh = True
    plot  = True  
    xbins = np.linspace(1000, 4000, nx+1) #pixel bin edges  for TSNR2_BGS
    ybins = np.linspace(14, 23, ny+1)     #pixel bin edges for fib_rmag 
    dxbin=xbins[1]-xbins[0]
    dybin=ybins[1]-ybins[0]
    print('Number of bins nx,ny:',nx,ny)
   
    #From the observed table add TNSR2_BGS and fib_mag to the clus table
    obs = observed['TARGETID', 'TSNR2_BGS', 'fib_rmag']
    clus = join(clus, obs, keys='TARGETID')

    if regions==None:
        regions = ['N', 'S'] # Loop over regions so that we deal with each separately

    for reg in regions:

        # select only objects within the region and within the grid boundaries
        zgoodmask = (zgood['PHOTSYS'] == reg)      & (zgood['TSNR2_BGS'] > xbins[0]) & (zgood['TSNR2_BGS'] < xbins[-1])
        obsmask   = (observed['PHOTSYS'] == reg)   & (observed['TSNR2_BGS'] > xbins[0]) & (observed['TSNR2_BGS'] < xbins[-1])
        print('observed in region and grid', np.count_nonzero(obsmask))
            
            
        if fresh:

            #compute grid coordinates for zgood and  observed 
            xg = zgood[zgoodmask]['TSNR2_BGS']
            yg = zgood[zgoodmask]['fib_rmag']
            xall = observed[obsmask]['TSNR2_BGS']
            yall = observed[obsmask]['fib_rmag']

            # In each pixel count the number of galaxies with redshift and the number observed
            Hg, xedges_g, yedges_g, binnumber_g = stats.binned_statistic_2d(xg, yg, None, 'count', bins=[xbins, ybins], expand_binnumbers=True)
            Hall, xedges_all, yedges_all, binnumber_all = stats.binned_statistic_2d(xall, yall, None, 'count', bins=[xbins, ybins], expand_binnumbers=True)
            # Avoiding divide by zero, set default completeness and compute completeness in the occupied pixels
            H = Hall*0.0+1.0
            nonzero_mask = (Hall >0) & (Hg >0)
            missed_mask =(Hall >0) & (Hg == 0)  # these are galaxies that don't have their weight passed to others as observed galaxies in the pixel but none with redshifts
            print(reg,' missed and unrepresented',np.sum(Hall[missed_mask]))        
            H[nonzero_mask] = (Hg[nonzero_mask]/Hall[nonzero_mask])

            # Make a plot of the completeness look-up table with the catalogue points overlaid
            if plot:
                XXg, YYg = np.meshgrid(xbins, ybins)
                #fig = plt.figure(figsize = (13,7))
                fig = plt.figure(figsize = (10/3, 3))

                ax1=plt.subplot(111)
                plot1 = ax1.pcolormesh(XXg,YYg,H.T, vmin=0.2)
                cbar = plt.colorbar(plot1,ax=ax1, pad = .015, aspect=10)
                plt.scatter(zgood[zgoodmask]['TSNR2_BGS'],zgood[zgoodmask]['fib_rmag'], marker='.',linewidth=0,s=0.1,alpha=0.1,c='red')
                plt.xlabel('TSNR2_BGS', fontsize=9)
                plt.ylabel('r_fibre', fontsize=9)
                plt.xlim([1000.0,4000.0])
                plt.ylim([14.0,23.0])
                spath = '/global/homes/l/ldrm11/Cole/graphs/weight_table_{}.pdf'.format(reg)
                plt.savefig(spath, bbox_inches='tight')
                
                spath = '/global/homes/l/ldrm11/Cole/graphs/weight_table_{}.png'.format(reg)
                plt.savefig(spath, bbox_inches='tight')
                plt.show()
                
                
                # An attempt to visualise CiC lookup on a finer grid
                nm=20
                xfbins = np.linspace(1000, 4000, nm*nx+1) #pixel bin edges  for TSNR2_BGS
                yfbins = np.linspace(14, 23, nm*ny+1)     #pixel bin edges for fib_rmag 
                XXg, YYg = np.meshgrid(xfbins, yfbins)
                Harray=np.asarray(H) # convert to numpy array to enable indexing     
                # use to generate grid indices for the  input data.
                __, __, __, i2 = stats.binned_statistic_2d(XXg.flatten(), YYg.flatten(), values=XXg.flatten(), statistic='median', bins = [xbins, ybins], expand_binnumbers=True)
                j2=i2   #keep the original indices as useful for generating indices of CIC neighbouring cells
                i2=i2-1 #correct for offset in returned bin indices
                #Cloud-in-Cell look-up
                #first compute the difference in the coordinate value of each data point and its nearest grid point in units of the bin spacing
                dx=-0.5-i2[0]+(XXg.flatten()-xbins[0])/dxbin   # these should satisfy -0.5<dx<0.5          
                dy=-0.5-i2[1]+(YYg.flatten()-ybins[0])/dybin   # these should satisfy -0.5<dy<0.5
                #for each negative value we need to change of j2 to select the correct neighbouring cell (i.e. to left and below rather than above and to the right)
                j2[0][(dx<0)]=j2[0][(dx<0)]-2
                j2[1][(dy<0)]=j2[1][(dy<0)]-2
                #CIC weights  (these add to unity)
                wa=(1.0-np.absolute(dx))*(1.0-np.absolute(dy))
                wb=     np.absolute(dx) *(1.0-np.absolute(dy))
                wc=(1.0-np.absolute(dx))*     np.absolute(dy)
                wd=     np.absolute(dx) *     np.absolute(dy)
                #To avoid out of bounds edge effects replace out of bound cells indices by the NGP cell index
                j0mask = (j2[0]>nx-1) | (j2[0]<0)
                j2[0][j0mask]=i2[0][j0mask]
                j1mask = (j2[1]>ny-1) | (j2[1]<0)
                j2[1][j1mask]=i2[1][j1mask]
                # Form the CIC weighted value and make its inverse the weight
                FH =  wa*Harray[i2[0],i2[1]]+wb*Harray[j2[0],i2[1]]+wc*Harray[i2[0],j2[1]]+wd*Harray[j2[0],j2[1]] 
                FH=FH.reshape(nm*ny+1,nm*nx+1)
                #fig = plt.figure(figsize = (13,7))
                fig = plt.figure(figsize = (10/3, 3))

                ax1=plt.subplot(111)
                plot1 = ax1.pcolormesh(XXg,YYg,FH, vmin=0.2)
                cbar = plt.colorbar(plot1,ax=ax1, pad = .015, aspect=10)
                plt.scatter(zgood[zgoodmask]['TSNR2_BGS'],zgood[zgoodmask]['fib_rmag'], marker='.',linewidth=0,s=0.1,alpha=0.1,c='red')
                plt.xlabel('TSNR2_BGS')
                plt.ylabel(r'$r_\mathrm{fibre}$')
                plt.xlim([1000.0,4000.0])
                plt.ylim([14.0,23.0])
                
                spath = '/global/homes/l/ldrm11/Cole/graphs/weight_table_CIC_{}.pdf'.format(reg)
                spath = '/global/homes/l/ldrm11/Cole/graphs/weight_table_CIC_{}.png'.format(reg)
                plt.savefig(spath, bbox_inches='tight')
                plt.show()

            # save the look-up table for future use
            #opath='/global/u2/s/smcole/DESI/NvsS/data/w_{}.txt'.format(reg)
            opath='/global/u2/l/ldrm11/DESI/data/w_{}.txt'.format(reg)
            np.savetxt(opath, H)
            print('LOOK-UP TABLE GENERATED AND SAVED.')

            
            
            # For objects in the clustering catalogue within the grid boundaries lookup their completeness and set their new redshift failure weight
            regmask= (clus['reg']==reg)  & (clus['TSNR2_BGS'] > xbins[0]) & (clus['TSNR2_BGS'] < xbins[-1]) # identify objects with the grid in this region
    
            
            #Then use look up to assign the weight
            xvar = 'TSNR2_BGS'
            yvar = 'fib_rmag'
            Harray=np.asarray(H) # convert to numpy array to enable indexing     
            # use to generate grid indices for the  input data.
            __, __, __, i2 = stats.binned_statistic_2d(clus[xvar][regmask], clus[yvar][regmask], values=clus[xvar][regmask], statistic='median', bins = [xbins, ybins], expand_binnumbers=True)
            j2=i2   #keep the original indices as useful for generating indices of CIC neighbouring cells
            i2=i2-1 #correct for offset in returned bin indices
            
            # check for any points out of range (due to rounding?)
            yyvar=clus[yvar][regmask]
            print('number outside boundary',yyvar[i2[1]>ny-1].size,'values:',yyvar[i2[1]>ny-1])
            i2[1]=np.minimum(i2[1],ny-1) #force to be within grid points outside due to rounding

            #Normal NGP  look-up
            clus['WEIGHT_ZLOOKUP_SIMPLE'][regmask]=1.0/( Harray[i2[0],i2[1]] )

            #Cloud-in-Cell look-up
            #first compute the difference in the coordinate value of each data point and its nearest grid point in units of the bin spacing
            dx=-0.5-i2[0]+(clus[xvar][regmask]-xbins[0])/dxbin   # these should satisfy -0.5<dx<0.5          
            dy=-0.5-i2[1]+(clus[yvar][regmask]-ybins[0])/dybin   # these should satisfy -0.5<dy<0.5 
            #for each negative value we need to change of j2 to select the correct neighbouring cell (i.e. to left and below rather than above and to the right)
            j2[0][(dx<0)]=j2[0][(dx<0)]-2
            j2[1][(dy<0)]=j2[1][(dy<0)]-2
            #CIC weights  (these add to unity)
            wa=(1.0-np.absolute(dx))*(1.0-np.absolute(dy))
            wb=     np.absolute(dx) *(1.0-np.absolute(dy))
            wc=(1.0-np.absolute(dx))*     np.absolute(dy)
            wd=     np.absolute(dx) *     np.absolute(dy)
            #To avoid out of bounds edge effects replace out of bound cells indices by the NGP cell index
            j0mask = (j2[0]>nx-1) | (j2[0]<0)
            j2[0][j0mask]=i2[0][j0mask]
            j1mask = (j2[1]>ny-1) | (j2[1]<0)
            j2[1][j1mask]=i2[1][j1mask]
            # Form the CIC weighted value and make its inverse the weight
            clus['WEIGHT_ZLOOKUP'][regmask] = 1.0/( wa*Harray[i2[0],i2[1]]+wb*Harray[j2[0],i2[1]]+wc*Harray[i2[0],j2[1]]+wd*Harray[j2[0],j2[1]] )
            print('ZWEIGHT ASSIGNED. to ', np.count_nonzero(regmask))
    
    if 'WEIGHT_OLD' not in clus.dtype.names:
        clus['WEIGHT_OLD'] = clus['WEIGHT']
    
    clus['WEIGHT'] = clus['WEIGHT_COMP'] * clus['WEIGHT_ZLOOKUP']

    return clus





