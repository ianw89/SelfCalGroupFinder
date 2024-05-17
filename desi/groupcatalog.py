import numpy as np
import pandas as pd
import astropy.coordinates as coord
import astropy.units as u
from pyutils import *
import pickle
import subprocess as sp
from astropy.table import Table
from hdf5_to_dat import pre_process_mxxl
from uchuu_to_dat import pre_process_uchuu

# Shared bins for various purposes
Mhalo_bins = np.logspace(10, 15.5, 40)
Mhalo_labels = Mhalo_bins[0:len(Mhalo_bins)-1] 

L_gal_bins = np.logspace(6, 12.5, 40)
L_gal_labels = L_gal_bins[0:len(L_gal_bins)-1]

Mr_gal_labels = log_solar_L_to_abs_mag_r(np.log10(L_gal_labels))


class GroupCatalog:

    def __init__(self, name):
        self.name = name
        self.file_pattern = BIN_FOLDER + self.name
        self.GF_outfile = self.file_pattern + ".out"
        self.results_file = self.file_pattern + ".pickle"
        self.color = 'k' # plotting color; nothing to do with galaxies
        self.marker = '-'
        self.preprocess_file = None
        self.GF_props = {} # Properties that are sent as command-line arguments to the group finder executable

        self.has_truth = False
        self.Mhalo_bins = Mhalo_bins
        self.labels = Mhalo_labels
        self.all_data = None
        self.centrals = None
        self.sats = None
        self.L_gal_bins = L_gal_bins
        self.L_gal_labels = L_gal_labels

        self.f_sat = None # per Lgal bin 
        self.Lgal_counts = None # size of Lgal bins 

    def run_group_finder(self):

        if self.preprocess_file is None:
            print("Warning: no input file set. Cannot run group finder.")
            return

        with open(self.GF_outfile, "w") as f:

            #args = [BIN_FOLDER + "kdGroupFinder_omp", inname, self.GF_props['zmin'], self.GF_props['zmax'], self.GF_props['frac_area'], self.GF_props['fluxlim'], self.GF_props['color'], self.GF_props['omegaL_sf'], self.GF_props['sigma_sf'], self.GF_props['omegaL_q'], self.GF_props['sigma_q'], self.GF_props['omega0_sf'], self.GF_props['omega0_q'], self.GF_props['beta0q'], self.GF_props['betaLq'], self.GF_props['beta0sf'], self.GF_props['betaLsf'], self.GF_outname]
            #sp.Popen(args, shell=True, stdout=sp.PIPE)

            args = [BIN_FOLDER + "kdGroupFinder_omp", self.preprocess_file, *list(map(str,self.GF_props.values()))]
            self.results = sp.run(args, cwd=BASE_FOLDER, stdout=f)

    def postprocess(self):
        if self.all_data is not None:
            
            # Compute some common aggregations upfront here
            # TODO make these lazilly evaluated properties on the GroupCatalog object
            # Can put more of them into this pattern from elsewhere in plotting code then
            self.f_sat = self.all_data.groupby('Lgal_bin').apply(fsat_vmax_weighted)
            self.Lgal_counts = self.all_data.groupby('Lgal_bin').RA.count()

            # Setup some convenience subsets of the DataFrame
            # TODO check memory implications of this
            self.centrals = self.all_data[self.all_data.index == self.all_data.igrp]
            self.sats = self.all_data[self.all_data.index != self.all_data.igrp]
        else:
            print("Warning: postprocess called with all_data DataFrame is not set yet. Override postprocess() or after calling run_group_finder() set it.")

class SDSSGroupCatalog(GroupCatalog):
    
    def __init__(self, name):
        super().__init__(name)
        self.preprocess_file = SDSS_v1_DAT_FILE

    def postprocess(self):
        galprops = pd.read_csv(SDSS_v1_GALPROPS_FILE, delimiter=' ', names=('Mag_g', 'Mag_r', 'sigma_v', 'Dn4000', 'concentration', 'log_M_star'))
        galprops['g_r'] = galprops.Mag_g - galprops.Mag_r 
        self.all_data = read_and_combine_gf_output(self, galprops)
        self.all_data['quiescent'] = is_quiescent_SDSS_Dn4000(self.all_data.logLgal, self.all_data.Dn4000)
        super().postprocess()

class MXXLGroupCatalog(GroupCatalog):

    def __init__(self, name, mode: Mode, mag_cut: float, catalog_mag_cut: float, use_colors: bool):
        super().__init__(name)
        self.mode = mode
        self.mag_cut = mag_cut
        self.catalog_mag_cut = catalog_mag_cut
        self.use_colors = use_colors
        self.color = mode_to_color(mode)
        

    def preprocess(self):
        fname, props = pre_process_mxxl(MXXL_FILE, self.mode.value, self.file_pattern, self.mag_cut, self.catalog_mag_cut, self.use_colors)
        self.preprocess_file = fname
        for p in props:
            self.GF_props[p] = props[p]

    def run_group_finder(self):
        if self.preprocess_file is None:
            self.preprocess()
        super().run_group_finder()


    def postprocess(self):
        filename_props = str.replace(self.GF_outfile, ".out", "_galprops.dat")
        galprops = pd.read_csv(filename_props, delimiter=' ', names=('app_mag', 'g_r', 'galaxy_type', 'mxxl_halo_mass', 'z_assigned_flag', 'assigned_halo_mass', 'z_obs', 'mxxl_halo_id', 'assigned_halo_id'), dtype={'mxxl_halo_id': np.int32, 'assigned_halo_id': np.int32})
        self.all_data = read_and_combine_gf_output(self, galprops)
        df = self.all_data
        self.has_truth = self.mode.value == Mode.ALL.value
        df['is_sat_truth'] = np.logical_or(df.galaxy_type == 1, df.galaxy_type == 3)
        if self.has_truth:
            df['Mh_bin_T'] = pd.cut(x = df['mxxl_halo_mass']*10**10, bins = Mhalo_bins, labels = Mhalo_labels, include_lowest = True)
            df['L_gal_T'] = np.power(10, abs_mag_r_to_log_solar_L(app_mag_to_abs_mag_k(df.app_mag.to_numpy(), df.z_obs.to_numpy(), df.g_r.to_numpy())))
            df['Lgal_bin_T'] = pd.cut(x = df['L_gal_T'], bins = L_gal_bins, labels = L_gal_labels, include_lowest = True)
            self.truth_f_sat = df.groupby('Lgal_bin_T').apply(fsat_truth_vmax_weighted)
            self.centrals_T = df[np.invert(df.is_sat_truth)]
            self.sats_T = df[df.is_sat_truth]

        # TODO if we switch to using bins we need a Truth version of this
        df['quiescent'] = is_quiescent_BGS_gmr(df.logLgal, df.g_r)

        super().postprocess()


class UchuuGroupCatalog(GroupCatalog):
   
    def __init__(self, name, mode: Mode, mag_cut: float, catalog_mag_cut: float, use_colors: bool):
        super().__init__(name)
        self.mode = mode
        self.mag_cut = mag_cut
        self.catalog_mag_cut = catalog_mag_cut
        self.use_colors = use_colors
        self.color = get_color(9)

    def preprocess(self):
        fname, props = pre_process_uchuu(UCHUU_FILE, self.mode.value, self.file_pattern, self.mag_cut, self.catalog_mag_cut, self.use_colors)
        self.preprocess_file = fname
        for p in props:
            self.GF_props[p] = props[p]

    def run_group_finder(self):
        if self.preprocess_file is None:
            self.preprocess()
        super().run_group_finder()


    def postprocess(self):
        filename_props = str.replace(self.GF_outfile, ".out", "_galprops.dat")
        galprops = pd.read_csv(filename_props, delimiter=' ', names=('app_mag', 'g_r', 'central', 'uchuu_halo_mass', 'uchuu_halo_id'), dtype={'uchuu_halo_id': np.int64, 'central': np.bool_})
        df = read_and_combine_gf_output(self, galprops)
        self.all_data = df

        self.has_truth = True
        self['is_sat_truth'] = np.invert(df.central)
        self['Mh_bin_T'] = pd.cut(x = self['uchuu_halo_mass']*10**10, bins = Mhalo_bins, labels = Mhalo_labels, include_lowest = True)
        # TODO BUG Need L_gal_T, the below is wrong!
        truth_f_sat = df.groupby('Lgal_bin').apply(fsat_truth_vmax_weighted)
        self.truth_f_sat = truth_f_sat
        self.centrals_T = df[np.invert(df.is_sat_truth)]
        self.sats_T = df[df.is_sat_truth]

        # TODO add quiescent column

        super().postprocess()

class BGSGroupCatalog(GroupCatalog):
    
    def __init__(self, name, mode: Mode, mag_cut: float, catalog_mag_cut: float, use_colors: bool):
        super().__init__(name)
        self.mode = mode
        self.mag_cut = mag_cut
        self.catalog_mag_cut = catalog_mag_cut
        self.use_colors = use_colors
        self.color = mode_to_color(mode)

    def preprocess(self):
        fname, props = pre_process_BGS(IAN_BGS_MERGED_FILE, self.mode.value, self.file_pattern, self.mag_cut, self.catalog_mag_cut, self.use_colors)
        self.preprocess_file = fname
        for p in props:
            self.GF_props[p] = props[p]

    def run_group_finder(self):
        if self.preprocess_file is None:
            self.preprocess()
        super().run_group_finder()

    def postprocess(self):
        filename_props = str.replace(self.GF_outfile, ".out", "_galprops.dat")
        galprops = pd.read_csv(filename_props, delimiter=' ', names=('app_mag', 'target_id', 'z_assigned_flag', 'g_r', 'Dn4000'), dtype={'target_id': np.int64, 'z_assigned_flag': np.bool_})
        df = read_and_combine_gf_output(self, galprops)
        self.all_data = df
        df['quiescent'] = is_quiescent_BGS_gmr(df.logLgal, df.g_r)
        super().postprocess()




def serialize(gc: GroupCatalog):
    # TODO this is a hack to get around class redefinitions invalidating serialized objects
    # Mess up subclasses?!
    gc.__class__ = eval(gc.__class__.__name__) #reset __class__ attribute
    with open(gc.results_file, 'wb') as f:
        pickle.dump(gc, f)

def deserialize(gc: GroupCatalog):
    gc.__class__ = eval(gc.__class__.__name__) #reset __class__ attribute
    with open(gc.results_file, 'rb') as f:    
        return pickle.load(f)





def pre_process_BGS(fname, mode, outname_base, APP_MAG_CUT, CATALOG_APP_MAG_CUT, COLORS_ON):
    """
    Pre-processes the BGS data for use with the group finder.
    """
    Z_MIN = 0.001
    Z_MAX = 0.8

    print("Reading FITS data from ", fname)
    # Unobserved galaxies have masked rows in appropriate columns of the table
    table = Table.read(fname, format='fits')
    
    FOOTPRINT_FRAC_1pass = 0.187906 # As calculated from the randoms with 1-pass coverage
    FOOTPRINT_FRAC = 0.0649945 # As calculated from the randoms with 3-pass coverage. 1310 degrees
    # TODO update footprint with new calculation from ANY. It shouldn't change.
    frac_area = FOOTPRINT_FRAC
    if mode == Mode.ALL.value:
        frac_area = FOOTPRINT_FRAC_1pass

    if mode == Mode.ALL.value:
        print("\nMode FIBER ASSIGNED ONLY 1+ PASSES")
    elif mode == Mode.FIBER_ASSIGNED_ONLY.value:
        print("\nMode FIBER ASSIGNED ONLY 3+ PASSES")
    elif mode == Mode.NEAREST_NEIGHBOR.value:
        print("\nMode NEAREST_NEIGHBOR")
    elif mode == Mode.FANCY.value:
        print("\nMode FANCY NOT SUPPORTED")
        exit(2)
    elif mode == Mode.SIMPLE.value:
        print("\nMode SIMPLE v2")
    elif mode == Mode.SIMPLE_v4.value:
        print("\nMode SIMPLE v4")

    print(f"Color classification sent to group finder: {COLORS_ON}")

    # astropy's Table used masked arrays, so we have to use .data.data to get the actual data
    # The masked rows are unobserved targets
    obj_type = table['SPECTYPE'].data.data
    dec = table['DEC']
    ra = table['RA']
    z_obs = table['Z_not4clus'].data.data
    target_id = table['TARGETID']
    app_mag_r = get_app_mag(table['FLUX_R'])
    app_mag_g = get_app_mag(table['FLUX_G'])
    g_r = app_mag_g - app_mag_r
    p_obs = table['PROB_OBS']
    unobserved = table['Z_not4clus'].mask # the masked values are what is unobserved
    deltachi2 = table['DELTACHI2'].data.data  
    dn4000 = table['DN4000'].data.data
    
    orig_count = len(dec)
    print(orig_count, "objects in FITS file")

    # If an observation was made, some automated system will evaluate the spectra and auto classify the SPECTYPE
    # as GALAXY, QSO, STAR. It is null (and masked) for non-observed targets.
    # NTILE tracks how many DESI pointings could have observed the target (at fiber level)
    # NTILE_MINE gives how many tiles include just from inclusion in circles drawn around tile centers
    # null values (masked rows) are unobserved targets; not all columns are masked though

    # Make filter arrays (True/False values)
    three_pass_filter = table['NTILE_MINE'] >= 3 # 3pass coverage
    galaxy_observed_filter = obj_type == b'GALAXY'
    app_mag_filter = app_mag_r < APP_MAG_CUT
    redshift_filter = z_obs > Z_MIN
    redshift_hi_filter = z_obs < Z_MAX
    deltachi2_filter = deltachi2 > 40 # Ensures that there wasn't another z with similar likelihood from the z fitting code
    observed_requirements = np.all([galaxy_observed_filter, app_mag_filter, redshift_filter, redshift_hi_filter, deltachi2_filter], axis=0)
    
    # treat low deltachi2 as unobserved
    treat_as_unobserved = np.all([galaxy_observed_filter, app_mag_filter, np.invert(deltachi2_filter)], axis=0)
    #print(f"We have {np.count_nonzero(treat_as_unobserved)} observed galaxies with deltachi2 < 40 to add to the unobserved pool")
    unobserved = np.all([app_mag_filter, np.logical_or(unobserved, treat_as_unobserved)], axis=0)

    if mode == Mode.ALL.value: # ALL is misnomer here it means 1pass or more
        keep = np.all([observed_requirements], axis=0)

    if mode == Mode.FIBER_ASSIGNED_ONLY.value: # means 3pass 
        keep = np.all([three_pass_filter, observed_requirements], axis=0)

    if mode == Mode.NEAREST_NEIGHBOR.value or mode == Mode.SIMPLE.value or mode == Mode.SIMPLE_v4.value:
        keep = np.all([three_pass_filter, np.logical_or(observed_requirements, unobserved)], axis=0)

        # Filter down inputs to the ones we want in the catalog for NN and similar calculations
        # TODO why bother with this for the real data? Use all we got, right? 
        # I upped the cut to 21 so it doesn't do anything
        catalog_bright_filter = app_mag_r < CATALOG_APP_MAG_CUT 
        catalog_keep = np.all([galaxy_observed_filter, catalog_bright_filter, redshift_filter, redshift_hi_filter, deltachi2_filter], axis=0)
        catalog_ra = ra[catalog_keep]
        catalog_dec = dec[catalog_keep]
        z_obs_catalog = z_obs[catalog_keep]
        catalog_gmr = app_mag_g[catalog_keep] - app_mag_r[catalog_keep]
        catalog_G_k = app_mag_to_abs_mag_k(app_mag_g[catalog_keep], z_obs_catalog, catalog_gmr, band='g')
        catalog_R_k = app_mag_to_abs_mag_k(app_mag_r[catalog_keep], z_obs_catalog, catalog_gmr, band='r')
        catalog_G_R_k = catalog_G_k - catalog_R_k
        catalog_quiescent = is_quiescent_BGS_gmr(abs_mag_r_to_log_solar_L(catalog_R_k), catalog_G_R_k)

        print(len(z_obs_catalog), "galaxies in the NN catalog.")

    # Apply filters
    obj_type = obj_type[keep]
    dec = dec[keep]
    ra = ra[keep]
    z_obs = z_obs[keep]
    target_id = target_id[keep]
    app_mag_r = app_mag_r[keep]
    app_mag_g = app_mag_g[keep]
    p_obs = p_obs[keep]
    unobserved = unobserved[keep]
    observed = np.invert(unobserved)
    indexes_not_assigned = np.argwhere(unobserved)
    deltachi2 = deltachi2[keep]
    g_r = g_r[keep]
    dn4000 = dn4000[keep]

    count = len(dec)
    print(count, "galaxies left after filters.")
    first_need_redshift_count = unobserved.sum()
    print(f'{first_need_redshift_count} remaining galaxies that need redshifts')
    print(f'{100*first_need_redshift_count / len(unobserved) :.1f}% of remaining galaxies need redshifts')
    #print(f'Min z: {min(z_obs):f}, Max z: {max(z_obs):f}')

    z_eff = np.copy(z_obs)

    # If a lost galaxy matches the SDSS catalog, grab it's redshift and use that
    # TODO BUG replace this with SDSS source galaxies list doesn't have any NN-assigned galaxies in it
    if unobserved.sum() > 0:
        sdss_vanilla = deserialize(SDSSGroupCatalog("SDSS Vanilla"))
        if sdss_vanilla.all_data is not None:

            sdss_catalog = coord.SkyCoord(ra=sdss_vanilla.all_data.RA.to_numpy()*u.degree, dec=sdss_vanilla.all_data.Dec.to_numpy()*u.degree, frame='icrs')
            to_match = coord.SkyCoord(ra=ra[unobserved]*u.degree, dec=dec[unobserved]*u.degree, frame='icrs')
            print(f"Matching {len(to_match)} lost galaxies to {len(sdss_catalog)} SDSS galaxies")
            idx, d2d, d3d = coord.match_coordinates_sky(to_match, sdss_catalog, nthneighbor=1, storekdtree=False)
            ang_distances = d2d.to(u.arcsec).value
            sdss_z = sdss_vanilla.all_data.iloc[idx]['z'].to_numpy()

            # if angular distance is < 3", then we consider it a match to SDSS catalog and copy over it's z
            ANGULAR_DISTANCE_MATCH = 3
            matched = ang_distances < ANGULAR_DISTANCE_MATCH
            
            z_eff[unobserved] = np.where(matched, sdss_z, np.nan)            
            unobserved[unobserved] = np.where(matched, False, unobserved[unobserved])
            observed = np.invert(unobserved)
            indexes_not_assigned = np.argwhere(unobserved)

            print(f"{matched.sum()} of {first_need_redshift_count} redshifts taken from SDSS.")
            print(f"{unobserved.sum()} remaining galaxies need redshifts.")
        else:
            print("No SDSS catalog to match to. Skipping.")

    if mode == Mode.NEAREST_NEIGHBOR.value:

        catalog = coord.SkyCoord(ra=catalog_ra*u.degree, dec=catalog_dec*u.degree, frame='icrs')
        to_match = coord.SkyCoord(ra=ra[unobserved]*u.degree, dec=dec[unobserved]*u.degree, frame='icrs')

        idx, d2d, d3d = coord.match_coordinates_sky(to_match, catalog, storekdtree=False)

        # i is the index of the full sized array that needed a NN z value
        # j is the index along the to_match list corresponding to that
        # idx are the indexes of the NN from the catalog
        assert len(indexes_not_assigned) == len(idx)

        print("Copying over NN properties... ", end='\r')
        j = 0
        for i in indexes_not_assigned:  
            z_eff[i] = z_obs_catalog[idx[j]]
            j = j + 1 
        print("Copying over NN properties... done")


    if mode == Mode.SIMPLE.value or mode == Mode.SIMPLE_v4.value:
        if mode == Mode.SIMPLE.value:
            ver = '2.0'
        elif mode == Mode.SIMPLE_v4.value:
            ver = '4.0'
        with SimpleRedshiftGuesser(app_mag_r[observed], z_obs[observed], ver) as scorer:

            catalog = coord.SkyCoord(ra=catalog_ra*u.degree, dec=catalog_dec*u.degree, frame='icrs')
            to_match = coord.SkyCoord(ra=ra[unobserved]*u.degree, dec=dec[unobserved]*u.degree, frame='icrs')

            # neighbor_indexes is the index of the nearest galaxy in the catalog arrays
            neighbor_indexes, d2d, d3d = coord.match_coordinates_sky(to_match, catalog, storekdtree=False)
            ang_distances = d2d.to(u.arcsec).value

            # We need to guess a color for the unobserved galaxies to help the redshift guesser
            # Multiple possible ideas

            # 1) Use the NN's redshift to k-correct the lost galaxies
            #abs_mag_R = app_mag_to_abs_mag(app_mag_r[unobserved], z_obs_catalog[neighbor_indexes])
            #abs_mag_R_k = k_correct(abs_mag_R, z_obs_catalog[neighbor_indexes], app_mag_g[unobserved] - app_mag_r[unobserved])
            #abs_mag_G = app_mag_to_abs_mag(app_mag_g[unobserved], z_obs_catalog[neighbor_indexes])
            #abs_mag_G_k = k_correct(abs_mag_G, z_obs_catalog[neighbor_indexes], app_mag_g[unobserved] - app_mag_r[unobserved], band='g')
            #log_L_gal = abs_mag_r_to_log_solar_L(abs_mag_R_k)
            #G_R_k = abs_mag_G_k - abs_mag_R_k
            #quiescent_gmr = is_quiescent_BGS_gmr(log_L_gal, G_R_k)

            # 2) Use an uncorrected apparent g-r color cut to guess if the galaxy is quiescent or not
            quiescent_gmr = is_quiescent_lost_gal_guess(app_mag_g[unobserved] - app_mag_r[unobserved]).astype(int)
            
            assert len(quiescent_gmr) == len(ang_distances)


            print(f"Assigning missing redshifts... ")   
            j = 0 # j counts the number of unobserved galaxies in the catalog that have been assigned a redshift thus far
            for i in indexes_not_assigned: # i is the index of the unobserved galaxy in the main arrays
                if j%10000==0:
                    print(f"{j}/{len(to_match)} complete", end='\r')

                catalog_idx = neighbor_indexes[j]
                chosen_z, isNN = scorer.choose_redshift(z_obs_catalog[catalog_idx], ang_distances[j], p_obs[i], app_mag_r[i], quiescent_gmr[j], catalog_quiescent[catalog_idx])
                
                z_eff[i] = chosen_z

                j = j + 1 

            print(f"{j}/{len(to_match)} complete")

    assert np.all(z_eff > 0.0)

    # Some of this is redudant with catalog calculations but oh well
    abs_mag_R = app_mag_to_abs_mag(app_mag_r, z_eff)
    abs_mag_R_k = k_correct(abs_mag_R, z_eff, g_r, band='r')
    abs_mag_G = app_mag_to_abs_mag(app_mag_g, z_eff)
    abs_mag_G_k = k_correct(abs_mag_G, z_eff, g_r, band='g')
    log_L_gal = abs_mag_r_to_log_solar_L(abs_mag_R_k) 
    G_R_k = abs_mag_G_k - abs_mag_R_k
    quiescent = is_quiescent_BGS_gmr(log_L_gal, G_R_k)
    print(f"{quiescent.sum()} quiescent galaxies, {len(quiescent) - quiescent.sum()} star-forming galaxies")
     #print(f"Quiescent agreement between g-r and Dn4000 for observed galaxies: {np.sum(quiescent_gmr[observed] == quiescent[observed]) / np.sum(observed)}")



    # the vmax should be calculated from un-k-corrected magnitudes
    V_max = get_max_observable_volume(abs_mag_R, z_eff, APP_MAG_CUT, frac_area)

    if not COLORS_ON:
        quiescent = np.zeros(count, dtype=np.int8)
    
    # TODO get galaxy concentration from somewhere
    chi = np.zeros(count, dtype=np.int8) 

    # TODO What value should z_assigned_flag be for SDSS-assigned?
    # Should update it from binary to an enum I think

    # Output files
    galprops = np.column_stack([
        np.array(app_mag_r, dtype='str'), 
        np.array(target_id, dtype='str'), 
        np.array(unobserved, dtype='str'),
        np.array(G_R_k, dtype='str'),
        np.array(dn4000, dtype='str'),
        ])
    write_dat_files(ra, dec, z_eff, log_L_gal, V_max, quiescent, chi, outname_base, frac_area, galprops)

    return outname_base + ".dat", {'zmin': np.min(z_eff), 'zmax': np.max(z_eff), 'frac_area': FOOTPRINT_FRAC }







##########################
# Processing Group Finder Output File
##########################

def read_and_combine_gf_output(gc: GroupCatalog, galprops_df):
    # TODO instead of reading GF output from disk, have option to just keep in memory
    main_df = pd.read_csv(gc.GF_outfile, delimiter=' ', names=('RA', 'Dec', 'z', 'L_gal', 'V_max', 'P_sat', 'M_halo', 'N_sat', 'L_tot', 'igrp', 'weight'))
    df = pd.merge(main_df, galprops_df, left_index=True, right_index=True)

    # Drop bad data, should have been cleaned up earlier though!
    orig_count = len(df)
    df = df[df.M_halo != 0]
    new_count = len(df)
    if (orig_count != new_count):
        print("Dropped {0} bad galaxies".format(orig_count - new_count))

    # add columns indicating if galaxy is a satellite
    df['is_sat'] = (df.index != df.igrp).astype(int)
    df['logLgal'] = np.log10(df.L_gal)

    # add column for halo mass bins and Lgal bins
    df['Mh_bin'] = pd.cut(x = df['M_halo'], bins = Mhalo_bins, labels = Mhalo_labels, include_lowest = True)
    df['Lgal_bin'] = pd.cut(x = df['L_gal'], bins = L_gal_bins, labels = L_gal_labels, include_lowest = True)

    return df # TODO update callers



# TODO might be wise to double check that my manual calculation of q vs sf matches what the group finder was fed by
# checking the input data. I think for now it should be exactly the same, but maybe we want it to be different for
# apples to apples comparison between BGS and SDSS



##########################
# Aggregation Helpers
##########################

def count_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        return len(series) / np.average(series.V_max)

def fsat_truth_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        return np.average(series.is_sat_truth, weights=1/series.V_max)
    
def fsat_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        return np.average(series.is_sat, weights=1/series.V_max)

def Lgal_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        return np.average(series.L_gal, weights=1/series.V_max)

def qf_Dn4000_1_6_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        return np.average((series.Dn4000 >  1.6), weights=1/series.V_max)

def qf_Dn4000_smart_eq_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        return np.average(is_quiescent_BGS_smart(series.logLgal, series.Dn4000, series.g_r), weights=1/series.V_max)

def qf_BGS_gmr_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        return np.average(is_quiescent_BGS_gmr(series.logLgal, series.g_r), weights=1/series.V_max)
    
def nsat_vmax_weighted(series):
    if len(series) == 0:
        return 0
    else:
        print(series.N_sat)
        return np.average(series.N_sat, weights=1/series.V_max)
    

