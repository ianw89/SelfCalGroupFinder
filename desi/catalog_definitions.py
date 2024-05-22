from groupcatalog import GroupCatalog, MXXLGroupCatalog, SDSSGroupCatalog, UchuuGroupCatalog, BGSGroupCatalog
from pyutils import Mode, get_color

BSAT_DEFAULT = 10

sdss_vanilla = SDSSGroupCatalog("SDSS Vanilla")
_cat = sdss_vanilla
_cat.color = get_color(4)
_cat.marker = '-'
_cat.GF_props = {
    'zmin':0,
    'zmax':1.0, # BUG ?
    'frac_area':0.179,
    'fluxlim':1,
    'color':1,
    'omegaL_sf':13.1,
    'sigma_sf':2.42,
    'omegaL_q':12.9,
    'sigma_q':4.84,
    'omega0_sf':0,  
    'omega0_q':0,    
    'beta0q':BSAT_DEFAULT,    
    'betaLq':0,
    'beta0sf':BSAT_DEFAULT,
    'betaLsf':0,
}

sdss_colors = SDSSGroupCatalog("SDSS Colors")
_cat = sdss_colors
_cat.color = get_color(4)
_cat.marker = '--'
_cat.GF_props = {
    'zmin':0,
    'zmax':1.0, # BUG ?
    'frac_area':0.179,
    'fluxlim':1,
    'color':1,
    'omegaL_sf':13.1,
    'sigma_sf':2.42,
    'omegaL_q':12.9,
    'sigma_q':4.84,
    'omega0_sf':17.4,  
    'omega0_q':2.67,    
    'beta0q':-0.92,    
    'betaLq':10.25,
    'beta0sf':12.993,
    'betaLsf':-8.04,
}

sdss_colors_chi = SDSSGroupCatalog("SDSS Colors Chi")
_cat = sdss_colors_chi
_cat.color = get_color(4)
_cat.marker = '.'
_cat.GF_props = {
    'zmin':0,
    'zmax':1.0, # BUG ?
    'frac_area':0.179,
    'fluxlim':1,
    'color':1,
    'omegaL_sf':13.1,
    'sigma_sf':2.42,
    'omegaL_q':12.9,
    'sigma_q':4.84,
    'omega0_sf':17.4,  
    'omega0_q':2.67,    
    'beta0q':-0.92,    
    'betaLq':10.25,
    'beta0sf':12.993,
    'betaLsf':-8.04,
    'omega_chi_0_sf':2.68,  
    'omega_chi_0_q':1.10,
    'omega_chi_L_sf':2.23,
    'omega_chi_L_q':0.48,
}

GF_PROPS_VANILLA = {
    'zmin':0, 
    'zmax':0,
    'frac_area':0, # should be filled in
    'fluxlim':1,
    'color':1,
    'omegaL_sf':13.1,
    'sigma_sf':2.42,
    'omegaL_q':12.9,
    'sigma_q':4.84,
    'omega0_sf':0,  
    'omega0_q':0,    
    'beta0q':BSAT_DEFAULT,    
    'betaLq':0,
    'beta0sf':BSAT_DEFAULT,
    'betaLsf':0,
}
GF_PROPS_COLORS = {
    'zmin':0, 
    'zmax':0,
    'frac_area':0, # should be filled in
    'fluxlim':1,
    'color':1,
    'omegaL_sf':13.1,
    'sigma_sf':2.42,
    'omegaL_q':12.9,
    'sigma_q':4.84,
    'omega0_sf':17.4,  
    'omega0_q':2.67,    
    'beta0q':-0.92,    
    'betaLq':10.25,
    'beta0sf':12.993,
    'betaLsf':-8.04,
}

mxxl_all = MXXLGroupCatalog("All MXXL <19.5", Mode.ALL, 19.5, 20.0, False)
_cat = mxxl_all
_cat.marker = '-'
_cat.GF_props = GF_PROPS_VANILLA.copy()

mxxl_all_c = MXXLGroupCatalog("All MXXL <19.5 c", Mode.ALL, 17.0, 17.5, True)
_cat = mxxl_all_c
_cat.marker = '--'
_cat.GF_props = GF_PROPS_COLORS.copy()

mxxl_fiberonly = MXXLGroupCatalog("Fiber Only MXXL <19.5", Mode.FIBER_ASSIGNED_ONLY, 19.5, 20.0, False)
_cat = mxxl_fiberonly
_cat.marker = '-'
_cat.GF_props = GF_PROPS_VANILLA.copy()

mxxl_fiberonly_c = MXXLGroupCatalog("Fiber Only MXXL <19.5 c", Mode.FIBER_ASSIGNED_ONLY, 19.5, 20.0, True)
_cat = mxxl_fiberonly_c
_cat.marker = '--'
_cat.GF_props = GF_PROPS_COLORS.copy()

mxxl_nn = MXXLGroupCatalog("Nearest Neighbor MXXL <19.5", Mode.NEAREST_NEIGHBOR, 19.5, 20.0, False)
_cat = mxxl_nn
_cat.marker = '-'
_cat.GF_props = GF_PROPS_VANILLA.copy()

mxxl_nn_c = MXXLGroupCatalog("Nearest Neighbor MXXL <19.5 c", Mode.NEAREST_NEIGHBOR, 19.5, 20.0, True)
_cat = mxxl_nn_c
_cat.marker = '--'
_cat.GF_props = GF_PROPS_COLORS.copy()

mxxl_simple_2 = MXXLGroupCatalog("Simple v2 MXXL <19.5", Mode.SIMPLE, 19.5, 20.0, False)
_cat = mxxl_simple_2
_cat.marker = '-'
_cat.GF_props = GF_PROPS_VANILLA.copy()

mxxl_simple_2_c = MXXLGroupCatalog("Simple v2 MXXL <19.5 c", Mode.SIMPLE, 19.5, 20.0, True)
_cat = mxxl_simple_2_c
_cat.marker = '--'
_cat.GF_props = GF_PROPS_COLORS.copy()

mxxl_simple_4 = MXXLGroupCatalog("Simple v4 MXXL <19.5", Mode.SIMPLE_v4, 19.5, 20.0, False)
_cat = mxxl_simple_4
_cat.marker = '-'
_cat.GF_props = GF_PROPS_VANILLA.copy()

mxxl_simple_4_c = MXXLGroupCatalog("Simple v4 MXXL <19.5 c", Mode.SIMPLE_v4, 19.5, 20.0, True)
_cat = mxxl_simple_4_c
_cat.marker = '--'
_cat.GF_props = GF_PROPS_COLORS.copy()

uchuu_all = UchuuGroupCatalog("All UCHUU <19.5", Mode.ALL, 19.5, 20.0, False)
_cat = uchuu_all
_cat.marker = '-'
_cat.GF_props = GF_PROPS_VANILLA.copy()

bgs_fiberonly_1pass = BGSGroupCatalog("Observed 1pass+ BGS <19.5", Mode.ALL, 19.5, 21.0, False)
_cat = bgs_fiberonly_1pass
_cat.marker = '-'
_cat.GF_props = GF_PROPS_VANILLA.copy()

bgs_fiberonly = BGSGroupCatalog("Observed BGS <19.5", Mode.FIBER_ASSIGNED_ONLY, 19.5, 21.0, False)
_cat = bgs_fiberonly
_cat.marker = '-'
_cat.GF_props = GF_PROPS_VANILLA.copy()

bgs_nn = BGSGroupCatalog("Nearest Neighbor BGS <19.5", Mode.NEAREST_NEIGHBOR, 19.5, 21.0, False)
_cat = bgs_nn
_cat.marker = '-'
_cat.GF_props = GF_PROPS_VANILLA.copy()

bgs_nn_sdsslike = BGSGroupCatalog("Nearest Neighbor BGS <19.5 SDSS-like", Mode.NEAREST_NEIGHBOR, 17.7, 17.7, False)
_cat = bgs_nn_sdsslike
_cat.marker = '-'
_cat.GF_props = GF_PROPS_VANILLA.copy()

bgs_simple_2 = BGSGroupCatalog("Simple v2 BGS <19.5", Mode.SIMPLE, 19.5, 21.0, False)
_cat = bgs_simple_2
_cat.marker = '-'
_cat.GF_props = GF_PROPS_VANILLA.copy()

bgs_simple_2_c = BGSGroupCatalog("Simple v2 BGS <19.5 c", Mode.SIMPLE, 19.5, 21.0, True)
_cat = bgs_simple_2_c
_cat.marker = '--'
_cat.GF_props = GF_PROPS_COLORS.copy()

bgs_simple_4 = BGSGroupCatalog("Simple v4 BGS <19.5", Mode.SIMPLE_v4, 19.5, 21.0, False)
_cat = bgs_simple_4
_cat.marker = '-'
_cat.GF_props = GF_PROPS_VANILLA.copy()

bgs_simple_4_no_sdss = BGSGroupCatalog("Simple v4 BGS <19.5 no-sdss", Mode.SIMPLE_v4, 19.5, 21.0, False, sdss_fill=False)
_cat = bgs_simple_4_no_sdss
_cat.marker = '.'
_cat.GF_props = GF_PROPS_VANILLA.copy()

bgs_simple_4_4p = BGSGroupCatalog("Simple v4 BGS <19.5 4pass", Mode.SIMPLE_v4, 19.5, 21.0, False, num_passes=4)
_cat = bgs_simple_4_4p
_cat.marker = '.'
_cat.GF_props = GF_PROPS_VANILLA.copy()

bgs_simple_4_c = BGSGroupCatalog("Simple v4 BGS <19.5 c", Mode.SIMPLE_v4, 19.5, 21.0, True)
_cat = bgs_simple_4_c
_cat.marker = '--'
_cat.GF_props = GF_PROPS_COLORS.copy()