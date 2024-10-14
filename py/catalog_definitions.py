import sys
import numpy as np
if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from groupcatalog import GroupCatalog, MXXLGroupCatalog, SDSSGroupCatalog, UchuuGroupCatalog, BGSGroupCatalog, SDSSPublishedGroupCatalog
from pyutils import Mode, get_color
from dataloc import *

sdss_vanilla = SDSSGroupCatalog("SDSS Vanilla", SDSS_v1_DAT_FILE, SDSS_v1_1_GALPROPS_FILE)
_cat = sdss_vanilla
_cat.color = get_color(4)
_cat.marker = '-'
_cat.GF_props = {
    'zmin':0,
    'zmax':1.0, # BUG ?
    'frac_area':0.179,
    'fluxlim':1,
    'color':1,
}

sdss_colors = SDSSGroupCatalog("SDSS Colors", SDSS_v1_DAT_FILE, SDSS_v1_1_GALPROPS_FILE)
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

sdss_colors_chi = SDSSGroupCatalog("SDSS Colors Chi", SDSS_v1_DAT_FILE, SDSS_v1_1_GALPROPS_FILE)
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

sdss_vanilla_v2 = SDSSGroupCatalog("SDSS Vanilla v2", SDSS_v2_DAT_FILE, SDSS_v2_GALPROPS_FILE)
_cat = sdss_vanilla_v2
_cat.color = get_color(4)
_cat.marker = '-'
_cat.GF_props = {
    'zmin':0,
    'zmax':1.0, # BUG ?
    'frac_area':0.179,
    'fluxlim':1,
    'color':1,
}

sdss_bgscut = SDSSGroupCatalog("SDSS BGS Cut", SDSS_BGSCUT_DAT_FILE, SDSS_BGSCUT_GALPROPS_FILE)
_cat = sdss_bgscut
_cat.color = get_color(4)
_cat.marker = '--'
_cat.GF_props = {
    'zmin':0,
    'zmax':1.0, # BUG ?
    'frac_area':0.128,
    'fluxlim':1,
    'color':1,
}

sdss_colors_v2 = SDSSGroupCatalog("SDSS Colors v2", SDSS_v2_DAT_FILE, SDSS_v2_GALPROPS_FILE)
_cat = sdss_colors_v2
_cat.color = get_color(4)
_cat.marker = '--'
_cat.GF_props = sdss_colors.GF_props.copy()

sdss_colors_chi_v2 = SDSSGroupCatalog("SDSS Colors Chi v2", SDSS_v2_DAT_FILE, SDSS_v2_GALPROPS_FILE)
_cat = sdss_colors_chi_v2
_cat.color = get_color(4)
_cat.marker = '.'
_cat.GF_props = sdss_colors_chi.GF_props.copy()

sdss_published = SDSSPublishedGroupCatalog("SDSS Published")

GF_PROPS_VANILLA = {
    'zmin':0, 
    'zmax':0,
    'frac_area':0, # should be filled in
    'fluxlim':1,
    'color':1,
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

bgs_fiberonly_1pass = BGSGroupCatalog("Observed 1pass+ BGS <19.5", Mode.FIBER_ASSIGNED_ONLY, 19.5, 21.0, num_passes=1)
_cat = bgs_fiberonly_1pass
_cat.marker = '-'
_cat.GF_props = GF_PROPS_VANILLA.copy()

bgs_fiberonly = BGSGroupCatalog("Observed BGS <19.5", Mode.FIBER_ASSIGNED_ONLY, 19.5, 21.0)
_cat = bgs_fiberonly
_cat.marker = '-'
_cat.GF_props = GF_PROPS_VANILLA.copy()

bgs_nn = BGSGroupCatalog("Nearest Neighbor BGS <19.5", Mode.NEAREST_NEIGHBOR, 19.5, 21.0)
_cat = bgs_nn
_cat.marker = '-'
_cat.GF_props = GF_PROPS_VANILLA.copy()

bgs_nn_sdsslike = BGSGroupCatalog("Nearest Neighbor BGS <19.5 SDSS-like", Mode.NEAREST_NEIGHBOR, 17.7, 17.7)
_cat = bgs_nn_sdsslike
_cat.marker = '-'
_cat.GF_props = GF_PROPS_VANILLA.copy()

bgs_simple_2 = BGSGroupCatalog("Simple v2 BGS <19.5", Mode.SIMPLE, 19.5, 21.0)
_cat = bgs_simple_2
_cat.marker = '-'
_cat.GF_props = GF_PROPS_VANILLA.copy()

bgs_simple_2_c = BGSGroupCatalog("Simple v2 BGS <19.5 c", Mode.SIMPLE, 19.5, 21.0)
_cat = bgs_simple_2_c
_cat.marker = '--'
_cat.GF_props = GF_PROPS_COLORS.copy()

bgs_simple_4_old = BGSGroupCatalog("Simple v4 BGS <19.5 v1.2", Mode.SIMPLE_v4, 19.5, 21.0, data_cut='Y1-Iron-v1.2')
_cat = bgs_simple_4_old
_cat.marker = '-'
_cat.color = 'gray'
_cat.GF_props = GF_PROPS_VANILLA.copy()

bgs_simple_4 = BGSGroupCatalog("Simple v4 BGS <19.5", Mode.SIMPLE_v4, 19.5, 21.0, data_cut='Y1-Iron')
_cat = bgs_simple_4
_cat.marker = '-'
_cat.color = 'k'
_cat.GF_props = GF_PROPS_VANILLA.copy()

bgs_simple_4_1pass = BGSGroupCatalog("Simple v4 BGS <19.5 1pass", Mode.SIMPLE_v4, 19.5, 21.0, num_passes=1, data_cut='Y1-Iron')
_cat = bgs_simple_4_1pass
_cat.marker = '.'
_cat.color = 'k'
_cat.GF_props = GF_PROPS_VANILLA.copy()

bgs_simple_4_no_sdss = BGSGroupCatalog("Simple v4 BGS <19.5 no-sdss", Mode.SIMPLE_v4, 19.5, 21.0, sdss_fill=False, data_cut='Y1-Iron')
_cat = bgs_simple_4_no_sdss
_cat.marker = '.'
_cat.color = 'k'
_cat.GF_props = GF_PROPS_VANILLA.copy()

bgs_simple_4_4p = BGSGroupCatalog("Simple v4 BGS <19.5 4pass", Mode.SIMPLE_v4, 19.5, 21.0, num_passes=4, data_cut='Y1-Iron')
_cat = bgs_simple_4_4p
_cat.marker = '.'
_cat.color = 'k'
_cat.GF_props = GF_PROPS_VANILLA.copy()

bgs_simple_4_c = BGSGroupCatalog("Simple v4 BGS <19.5 c", Mode.SIMPLE_v4, 19.5, 21.0, data_cut='Y1-Iron')
_cat = bgs_simple_4_c
_cat.marker = '--'
_cat.GF_props = GF_PROPS_COLORS.copy()

bgs_simple_5 = BGSGroupCatalog("Simple v5 BGS <19.5", Mode.SIMPLE_v5, 19.5, 21.0, data_cut='Y1-Iron')
_cat = bgs_simple_5
_cat.marker = '-'
_cat.color = 'k'
_cat.GF_props = GF_PROPS_VANILLA.copy()

bgs_y3_pzp_1 = BGSGroupCatalog("Photo-z Plus v1.1 BGS Y3 <19.5", Mode.PHOTOZ_PLUS_v1, 19.5, 21.0, data_cut='Y3-Kibo', extra_params=(10, (0.8, 1.0, 0.004, 4.0)))
_cat = bgs_y3_pzp_1
_cat.color = 'darkorange'
_cat.marker = '-'
_cat.GF_props = GF_PROPS_VANILLA.copy()

bgs_y3_simple_4 = BGSGroupCatalog("Simple v4 BGS Y3 <19.5", Mode.SIMPLE_v4, 19.5, 21.0, data_cut='Y3-Kibo')
_cat = bgs_y3_simple_4
_cat.color = 'darkorange'
_cat.marker = '-'
_cat.GF_props = GF_PROPS_VANILLA.copy()

bgs_y3_simple_5 = BGSGroupCatalog("Simple v5 BGS Y3 <19.5", Mode.SIMPLE_v5, 19.5, 21.0, data_cut='Y3-Kibo')
_cat = bgs_y3_simple_5
_cat.color = 'darkorange'
_cat.marker = '-'
_cat.GF_props = GF_PROPS_VANILLA.copy()

bgs_y3_simple_4_4p = BGSGroupCatalog("Simple v4 BGS Y3 <19.5 4pass", Mode.SIMPLE_v4, 19.5, 21.0, num_passes=4, data_cut='Y3-Kibo')
_cat = bgs_y3_simple_4_4p
_cat.marker = '.'
_cat.color = 'darkorange'
_cat.GF_props = GF_PROPS_VANILLA.copy()

bgs_y3_fiberonly = BGSGroupCatalog("Observed BGS Y3 <19.5", Mode.FIBER_ASSIGNED_ONLY, 19.5, 21.0, data_cut='Y3-Kibo')
_cat = bgs_y3_fiberonly
_cat.marker = '-'
_cat.GF_props = GF_PROPS_VANILLA.copy()

bgs_y3_fiberonly_1pass = BGSGroupCatalog("Observed BGS Y3 <19.5 1pass", Mode.FIBER_ASSIGNED_ONLY, 19.5, 21.0, num_passes=1, data_cut='Y3-Kibo')
_cat = bgs_y3_fiberonly_1pass
_cat.marker = '.'
_cat.GF_props = GF_PROPS_VANILLA.copy()

#0.004 = 4.5
# To translate betwen old sigma and new one, use new = -log10(2*old^2)
bgs_sv3_pz_1_10p = BGSGroupCatalog("Photo-z Plus v1 BGS sv3 10pass", Mode.PHOTOZ_PLUS_v1, 19.5, 21.0, num_passes=10, data_cut='sv3', sdss_fill=False, extra_params=(5, (0.8, 1.0, 4.5, 4.0)))
bgs_sv3_pz_1_10p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_pz_1_10p.color = [0.0, 1.0, 1.0]

bgs_sv3_pz_1_0_7p = BGSGroupCatalog("Photo-z Plus v1 BGS sv3 7pass", Mode.PHOTOZ_PLUS_v1, 19.5, 21.0, num_passes=10, drop_passes=3, data_cut='sv3', sdss_fill=False, extra_params=(5, (0.8, 1.0, 4.5, 4.0)))
bgs_sv3_pz_1_0_7p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_pz_1_0_7p.color = 'g'

bgs_sv3_pz_1_1_7p = BGSGroupCatalog("Photo-z Plus v1.1 BGS sv3 7pass ", Mode.PHOTOZ_PLUS_v1, 19.5, 21.0, num_passes=10, drop_passes=3, data_cut='sv3', sdss_fill=False, extra_params=(10, (0.8, 1.0, 4.5, 4.0)))
bgs_sv3_pz_1_1_7p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_pz_1_1_7p.color = [1.0, 0.0, 0.0]

bgs_sv3_pz_1_2_7p = BGSGroupCatalog("Photo-z Plus v1.2 BGS sv3 7pass ", Mode.PHOTOZ_PLUS_v1, 19.5, 21.0, num_passes=10, drop_passes=3, data_cut='sv3', sdss_fill=False, extra_params=(10, (0.997, 1.349, 4.5, 4.0)))
bgs_sv3_pz_1_2_7p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_pz_1_2_7p.color = [0.7, 0.0, 0.0]

bgs_sv3_pz_1_3_7p = BGSGroupCatalog("Photo-z Plus v1.3 BGS sv3 7pass ", Mode.PHOTOZ_PLUS_v1, 19.5, 21.0, num_passes=10, drop_passes=3, data_cut='sv3', sdss_fill=False, extra_params=(10, (1.5484, 1.2898, 2.2746, 1.6668)))
bgs_sv3_pz_1_3_7p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_pz_1_3_7p.color = [0.4, 0.0, 0.0]

bgs_sv3_pz_1_5_7p = BGSGroupCatalog("Photo-z Plus v1.5 BGS sv3 7pass ", Mode.PHOTOZ_PLUS_v1, 19.5, 21.0, num_passes=10, drop_passes=3, data_cut='sv3', sdss_fill=False, extra_params=(18, [2.33351118, 3.15764279, 2.2579, 1.51392376]))
bgs_sv3_pz_1_5_7p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_pz_1_5_7p.color = [0.2, 0.0, 0.0]

bgs_sv3_pz_1_6_7p = BGSGroupCatalog("Photo-z Plus v1.6 BGS sv3 7pass ", Mode.PHOTOZ_PLUS_v1, 19.5, 21.0, num_passes=10, drop_passes=3, data_cut='sv3', sdss_fill=False, extra_params=
 (17, [1.1528, 2.5795, 2.8277], [1.1408, 3.0129, 2.9726], [1.6243, 3.6578, 3.2446], [1.9327, 1.5995, 3.775 ]))
bgs_sv3_pz_1_6_7p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_pz_1_6_7p.color = [0.2, 0.0, 0.2]

bgs_sv3_pz_2_0_7p = BGSGroupCatalog("Photo-z Plus v2.0 BGS sv3 7pass ", Mode.PHOTOZ_PLUS_v2, 19.5, 21.0, num_passes=10, drop_passes=3, data_cut='sv3', sdss_fill=False, extra_params=
 (18, [2.33351118, 3.15764279, 2.15354, 1.51392376],
 [0.5658714,  1.28889203, 0.05925339, 1.02472837,],
 [1.99883106, 0.57370031, 2.0977, 2.4052792, ],
 [0.4071051,  0.67088151, 1.8756, 2.4734777 ]))
bgs_sv3_pz_2_0_7p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_pz_2_0_7p.color = [0.7, 0.2, 0.0]

bgs_sv3_pz_2_1_7p = BGSGroupCatalog("Photo-z Plus v2.1 BGS sv3 7pass ", Mode.PHOTOZ_PLUS_v2, 19.5, 21.0, num_passes=10, drop_passes=3, data_cut='sv3', sdss_fill=False, extra_params=
 (4, [0.3469, 1.2433, 3.1864], [0.0392, 0.375, 1.5278], [0.6808, 0.7361, 4.9456], [0.0429, 0.462, 6.7101]))
bgs_sv3_pz_2_1_7p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_pz_2_1_7p.color = [0.8, 0.35, 0.0]

bgs_sv3_pz_2_2_7p = BGSGroupCatalog("Photo-z Plus v2.2 BGS sv3 7pass ", Mode.PHOTOZ_PLUS_v2, 19.5, 21.0, num_passes=10, drop_passes=3, data_cut='sv3', sdss_fill=False, extra_params=
 (1, [0.3498, 1.4654, 6.5728], [0.2179, 0.0204, 2.9563], [0.6217, 0.1704, 3.7362], [0.4573, 0.9539, 3.8091]))
bgs_sv3_pz_2_2_7p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_pz_2_2_7p.color = [0.8, 0.45, 0.0]

bgs_sv3_pz_3_0_7p = BGSGroupCatalog("Photo-z Plus v3.0 BGS sv3 7pass ", Mode.PHOTOZ_PLUS_v3, 19.5, 21.0, num_passes=10, drop_passes=3, data_cut='sv3', sdss_fill=False, extra_params=())
bgs_sv3_pz_3_0_7p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_pz_3_0_7p.color = [0.45, 0.40, 0.0]

bgs_sv3_fiberonly_10p = BGSGroupCatalog("Observed BGS sv3 10pass", Mode.FIBER_ASSIGNED_ONLY, 19.5, 21.0, num_passes=10, data_cut='sv3')
bgs_sv3_fiberonly_10p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_fiberonly_10p.color = 'orange'

bgs_sv3_simple_4_10p = BGSGroupCatalog("Simple v4 BGS sv3 10pass", Mode.SIMPLE_v4, 19.5, 21.0, num_passes=10, data_cut='sv3', sdss_fill=False)
bgs_sv3_simple_4_10p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_simple_4_10p.color = [0.0, 1.0, 0.0]

bgs_sv3_simple_4_9p = BGSGroupCatalog("Simple v4 BGS sv3 9pass", Mode.SIMPLE_v4, 19.5, 21.0, num_passes=10, drop_passes=1, data_cut='sv3', sdss_fill=False)
bgs_sv3_simple_4_9p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_simple_4_9p.color = [0.0, 0.9, 0.0]

bgs_sv3_simple_4_8p = BGSGroupCatalog("Simple v4 BGS sv3 8pass", Mode.SIMPLE_v4, 19.5, 21.0, num_passes=10, drop_passes=2, data_cut='sv3', sdss_fill=False)
bgs_sv3_simple_4_8p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_simple_4_8p.color = [0.0, 0.8, 0.0]

bgs_sv3_simple_4_7p = BGSGroupCatalog("Simple v4 BGS sv3 7pass", Mode.SIMPLE_v4, 19.5, 21.0, num_passes=10, drop_passes=3, data_cut='sv3', sdss_fill=False)
bgs_sv3_simple_4_7p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_simple_4_7p.color = [0.0, 0.7, 0.0]

bgs_sv3_simple_4_6p = BGSGroupCatalog("Simple v4 BGS sv3 6pass", Mode.SIMPLE_v4, 19.5, 21.0, num_passes=10, drop_passes=4, data_cut='sv3', sdss_fill=False)
bgs_sv3_simple_4_6p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_simple_4_6p.color = [0.0, 0.6, 0.0]

bgs_sv3_simple_4_5p = BGSGroupCatalog("Simple v4 BGS sv3 5pass", Mode.SIMPLE_v4, 19.5, 21.0, num_passes=10, drop_passes=5, data_cut='sv3', sdss_fill=False)
bgs_sv3_simple_4_5p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_simple_4_5p.color = [0.0, 0.5, 0.0]

bgs_sv3_simple_4_4p = BGSGroupCatalog("Simple v4 BGS sv3 4pass", Mode.SIMPLE_v4, 19.5, 21.0, num_passes=10, drop_passes=6, data_cut='sv3', sdss_fill=False)
bgs_sv3_simple_4_4p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_simple_4_4p.color = [0.0, 0.4, 0.0]

bgs_sv3_simple_4_3p = BGSGroupCatalog("Simple v4 BGS sv3 3pass", Mode.SIMPLE_v4, 19.5, 21.0, num_passes=10, drop_passes=7, data_cut='sv3', sdss_fill=False)
bgs_sv3_simple_4_3p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_simple_4_3p.color = [0.0, 0.3, 0.0]

bgs_sv3_simple_4_2p = BGSGroupCatalog("Simple v4 BGS sv3 2pass", Mode.SIMPLE_v4, 19.5, 21.0, num_passes=10, drop_passes=8, data_cut='sv3', sdss_fill=False)
bgs_sv3_simple_4_2p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_simple_4_2p.color = [0.0, 0.2, 0.0]

bgs_sv3_simple_4_1p = BGSGroupCatalog("Simple v4 BGS sv3 1pass", Mode.SIMPLE_v4, 19.5, 21.0, num_passes=10, drop_passes=9, data_cut='sv3', sdss_fill=False)
bgs_sv3_simple_4_1p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_simple_4_1p.color = [0.0, 0.1, 0.0]

bgs_sv3_nn_10p = BGSGroupCatalog("NN BGS sv3 10pass", Mode.NEAREST_NEIGHBOR, 19.5, 21.0, num_passes=10, drop_passes=0, data_cut='sv3', sdss_fill=False)
bgs_sv3_nn_10p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_nn_10p.color = [1.0, 1.0, 0.0]

bgs_sv3_nn_7p = BGSGroupCatalog("NN BGS sv3 7pass", Mode.NEAREST_NEIGHBOR, 19.5, 21.0, num_passes=10, drop_passes=3, data_cut='sv3', sdss_fill=False)
bgs_sv3_nn_7p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_nn_7p.color = [0.7, 0.7, 0.0]

bgs_sv3_nn_6p = BGSGroupCatalog("NN BGS sv3 6pass", Mode.NEAREST_NEIGHBOR, 19.5, 21.0, num_passes=10, drop_passes=4, data_cut='sv3', sdss_fill=False)
bgs_sv3_nn_6p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_nn_6p.color = [0.6, 0.6, 0.0]

bgs_sv3_fiberonly_10p = BGSGroupCatalog("Observed BGS sv3 10pass", Mode.FIBER_ASSIGNED_ONLY, 19.5, 21.0, num_passes=10, data_cut='sv3')
bgs_sv3_fiberonly_10p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_fiberonly_10p.color = 'r'

bgs_sv3_simple_5_10p = BGSGroupCatalog("Simple v5 BGS sv3 10pass", Mode.SIMPLE_v5, 19.5, 21.0, num_passes=10, data_cut='sv3', sdss_fill=False)
bgs_sv3_simple_5_10p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_simple_5_10p.color = [0.0, 1.0, 0.0]

bgs_sv3_simple_5_9p = BGSGroupCatalog("Simple v5 BGS sv3 9pass", Mode.SIMPLE_v5, 19.5, 21.0, num_passes=10, drop_passes=1, data_cut='sv3', sdss_fill=False)
bgs_sv3_simple_5_9p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_simple_5_9p.color = [0.0, 0.9, 0.0]

bgs_sv3_simple_5_8p = BGSGroupCatalog("Simple v5 BGS sv3 8pass", Mode.SIMPLE_v5, 19.5, 21.0, num_passes=10, drop_passes=2, data_cut='sv3', sdss_fill=False)
bgs_sv3_simple_5_8p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_simple_5_8p.color = [0.0, 0.8, 0.0]

bgs_sv3_simple_5_7p = BGSGroupCatalog("Simple v5 BGS sv3 7pass", Mode.SIMPLE_v5, 19.5, 21.0, num_passes=10, drop_passes=3, data_cut='sv3', sdss_fill=False)
bgs_sv3_simple_5_7p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_simple_5_7p.color = [0.0, 0.7, 0.0]

bgs_sv3_simple_5_6p = BGSGroupCatalog("Simple v5 BGS sv3 6pass", Mode.SIMPLE_v5, 19.5, 21.0, num_passes=10, drop_passes=4, data_cut='sv3', sdss_fill=False)
bgs_sv3_simple_5_6p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_simple_5_6p.color = [0.0, 0.6, 0.0]

bgs_sv3_simple_5_5p = BGSGroupCatalog("Simple v5 BGS sv3 5pass", Mode.SIMPLE_v5, 19.5, 21.0, num_passes=10, drop_passes=5, data_cut='sv3', sdss_fill=False)
bgs_sv3_simple_5_5p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_simple_5_5p.color = [0.0, 0.5, 0.0]

bgs_sv3_simple_5_4p = BGSGroupCatalog("Simple v5 BGS sv3 4pass", Mode.SIMPLE_v5, 19.5, 21.0, num_passes=10, drop_passes=6, data_cut='sv3', sdss_fill=False)
bgs_sv3_simple_5_4p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_simple_5_4p.color = [0.0, 0.4, 0.0]

bgs_sv3_simple_5_3p = BGSGroupCatalog("Simple v5 BGS sv3 3pass", Mode.SIMPLE_v5, 19.5, 21.0, num_passes=10, drop_passes=7, data_cut='sv3', sdss_fill=False)
bgs_sv3_simple_5_3p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_simple_5_3p.color = [0.0, 0.3, 0.0]

bgs_sv3_simple_5_2p = BGSGroupCatalog("Simple v5 BGS sv3 2pass", Mode.SIMPLE_v5, 19.5, 21.0, num_passes=10, drop_passes=8, data_cut='sv3', sdss_fill=False)
bgs_sv3_simple_5_2p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_simple_5_2p.color = [0.0, 0.2, 0.0]

bgs_sv3_simple_5_1p = BGSGroupCatalog("Simple v5 BGS sv3 1pass", Mode.SIMPLE_v5, 19.5, 21.0, num_passes=10, drop_passes=9, data_cut='sv3', sdss_fill=False)
bgs_sv3_simple_5_1p.GF_props = GF_PROPS_VANILLA.copy()
bgs_sv3_simple_5_1p.color = [0.0, 0.1, 0.0]