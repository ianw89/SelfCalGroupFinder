import sys
import numpy as np
if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from groupcatalog import *
from pyutils import Mode, get_color
from dataloc import *


#  target/neighbor bb rb br rr
PZP_PARAMS_V24 = (8, [1.2938, 1.5467, 3.0134], [1.2229, 0.8628, 2.5882], [0.8706, 0.6126, 2.4447], [1.1163, 1.2938, 3.1650])
#PZP_PARAMS_V26 = (11, [1.7448, 1.7456, 2.8354], [1.2543, 1.0822, 2.6076], [1.0907, 1.22  , 2.0313], [1.1925, 0.9663, 2.2485])
PZP_PARAMS_V26 = (8, [0.95, 1.1, 3.0], [1.0, 1.15, 3.1], [0.95, 1.1, 3.0], [1.0, 1.25, 3.1]) # Hand tuned set that matchs fsat better than v2.4 but gets less z right
#Parameters: (13, [0.9041, 1.4346, 2.1083], [1.385 , 0.638 , 2.7953], [1.097 , 0.4847, 1.9423], [1.2167, 1.5412, 3.1833])

PZP_PARAMS_V40 = (12, 1.25, 1.1, 1.2, 1.1, 1.3) # Neighbors, a, bb_b, rb_b, br_b, rr_b

sdss_vanilla = SDSSGroupCatalog(
    "SDSS Vanilla", 
    SDSS_v1_DAT_FILE, 
    SDSS_v1_1_GALPROPS_FILE, 
    gfprops={
        'zmin': 0,
        'zmax': 1.0,
        'frac_area': 0.179,
        'fluxlim': 1,
        'color': 1,
    }
)
_cat = sdss_vanilla
_cat.color = get_color(4)
_cat.marker = '-'

sdss_colors = SDSSGroupCatalog(
    "SDSS Colors", 
    SDSS_v1_DAT_FILE, 
    SDSS_v1_1_GALPROPS_FILE, 
    gfprops={
        'zmin': 0,
        'zmax': 1.0,
        'frac_area': 0.179,
        'fluxlim': 1,
        'color': 1,
        'omegaL_sf': 13.1,
        'sigma_sf': 2.42,
        'omegaL_q': 12.9,
        'sigma_q': 4.84,
        'omega0_sf': 17.4,  
        'omega0_q': 2.67,    
        'beta0q': -0.92,    
        'betaLq': 10.25,
        'beta0sf': 12.993,
        'betaLsf': -8.04,
    }
)
_cat = sdss_colors
_cat.color = get_color(4)
_cat.marker = '--'

sdss_colors_chi = SDSSGroupCatalog(
    "SDSS Colors Chi", 
    SDSS_v1_DAT_FILE, 
    SDSS_v1_1_GALPROPS_FILE, 
    gfprops={
        'zmin': 0,
        'zmax': 1.0,
        'frac_area': 0.179,
        'fluxlim': 1,
        'color': 1,
        'omegaL_sf': 13.1,
        'sigma_sf': 2.42,
        'omegaL_q': 12.9,
        'sigma_q': 4.84,
        'omega0_sf': 17.4,  
        'omega0_q': 2.67,    
        'beta0q': -0.92,    
        'betaLq': 10.25,
        'beta0sf': 12.993,
        'betaLsf': -8.04,
        'omega_chi_0_sf': 2.68,  
        'omega_chi_0_q': 1.10,
        'omega_chi_L_sf': 2.23,
        'omega_chi_L_q': 0.48,
    }
)
_cat = sdss_colors_chi
_cat.color = get_color(4)
_cat.marker = '.'

sdss_vanilla_v2 = SDSSGroupCatalog(
    "SDSS Vanilla v2", 
    SDSS_v2_DAT_FILE, 
    SDSS_v2_GALPROPS_FILE, 
    gfprops={
        'zmin': 0,
        'zmax': 1.0,
        'frac_area': 0.179,
        'fluxlim': 1,
        'color': 1,
    }
)
_cat = sdss_vanilla_v2
_cat.color = get_color(4)
_cat.marker = '-'

sdss_bgscut = SDSSGroupCatalog(
    "SDSS BGS Cut", 
    SDSS_BGSCUT_DAT_FILE, 
    SDSS_BGSCUT_GALPROPS_FILE, 
    gfprops={
        'zmin': 0,
        'zmax': 1.0, 
        'frac_area': 0.128,
        'fluxlim': 1,
        'color': 1,
    }
)
_cat = sdss_bgscut
_cat.color = get_color(4)
_cat.marker = '--'

sdss_colors_v2 = SDSSGroupCatalog("SDSS Colors v2", SDSS_v2_DAT_FILE, SDSS_v2_GALPROPS_FILE, gfprops=sdss_colors.GF_props.copy())
sdss_colors_v2.color = get_color(4)
sdss_colors_v2.marker = '--'

sdss_colors_chi_v2 = SDSSGroupCatalog("SDSS Colors Chi v2", SDSS_v2_DAT_FILE, SDSS_v2_GALPROPS_FILE, gfprops=sdss_colors_chi.GF_props.copy())
sdss_colors_chi_v2.color = get_color(4)
sdss_colors_chi_v2.marker = '.'

sdss_colors_v2_mcmc = SDSSGroupCatalog("SDSS Colors v2 MCMC", SDSS_v2_DAT_FILE, SDSS_v2_GALPROPS_FILE, gfprops={
    'zmin':0,
    'zmax':1.0,
    'frac_area':0.179,
    'fluxlim':1,
    'color':1,
})

props = GF_PROPS_BGS_COLORS_C1
props['zmin'] = 0
props['zmax'] = 1.0
props['frac_area'] = 0.179
sdss_colors_v2_desiparams_v1 = SDSSGroupCatalog("SDSS Colors v2 MCMC", SDSS_v2_DAT_FILE, SDSS_v2_GALPROPS_FILE, gfprops=props)

sdss_published = SDSSPublishedGroupCatalog("SDSS Published")

mxxl_all = MXXLGroupCatalog("All MXXL <19.5", Mode.ALL, 19.5, 20.0, False, gfprops=GF_PROPS_BGS_VANILLA.copy())
_cat = mxxl_all
_cat.marker = '-'
mxxl_all_c = MXXLGroupCatalog("All MXXL <19.5 c", Mode.ALL, 17.0, 17.5, True, gfprops=GF_PROPS_BGS_COLORS_C1.copy())
mxxl_all_c.marker = '--'

mxxl_fiberonly = MXXLGroupCatalog("Fiber Only MXXL <19.5", Mode.FIBER_ASSIGNED_ONLY, 19.5, 20.0, False, gfprops=GF_PROPS_BGS_VANILLA.copy())
mxxl_fiberonly.marker = '-'

mxxl_fiberonly_c = MXXLGroupCatalog("Fiber Only MXXL <19.5 c", Mode.FIBER_ASSIGNED_ONLY, 19.5, 20.0, True, gfprops=GF_PROPS_BGS_COLORS_C1.copy())
mxxl_fiberonly_c.marker = '--'

mxxl_nn = MXXLGroupCatalog("Nearest Neighbor MXXL <19.5", Mode.NEAREST_NEIGHBOR, 19.5, 20.0, False, gfprops=GF_PROPS_BGS_VANILLA.copy())
mxxl_nn.marker = '-'

mxxl_nn_c = MXXLGroupCatalog("Nearest Neighbor MXXL <19.5 c", Mode.NEAREST_NEIGHBOR, 19.5, 20.0, True, gfprops=GF_PROPS_BGS_COLORS_C1.copy())
mxxl_nn_c.marker = '--'

mxxl_simple_2 = MXXLGroupCatalog("Simple v2 MXXL <19.5", Mode.SIMPLE, 19.5, 20.0, False, gfprops=GF_PROPS_BGS_VANILLA.copy())
mxxl_simple_2.marker = '-'

mxxl_simple_2_c = MXXLGroupCatalog("Simple v2 MXXL <19.5 c", Mode.SIMPLE, 19.5, 20.0, True, gfprops=GF_PROPS_BGS_COLORS_C1.copy())
mxxl_simple_2_c.marker = '--'
mxxl_simple_4 = MXXLGroupCatalog("Simple v4 MXXL <19.5", Mode.SIMPLE_v4, 19.5, 20.0, False, gfprops=GF_PROPS_BGS_VANILLA.copy())
mxxl_simple_4.marker = '-'

mxxl_simple_4_c = MXXLGroupCatalog("Simple v4 MXXL <19.5 c", Mode.SIMPLE_v4, 19.5, 20.0, True, gfprops=GF_PROPS_BGS_COLORS_C1.copy())
mxxl_simple_4_c.marker = '--'

uchuu_all = UchuuGroupCatalog("All UCHUU <19.5", Mode.ALL, 19.5, 20.0, False, gfprops=GF_PROPS_BGS_VANILLA.copy())
uchuu_all.marker = '-'

bgs_y1_fiberonly_1pass = BGSGroupCatalog("Observed 1pass BGS Y1", Mode.FIBER_ASSIGNED_ONLY, 19.5, 21.0, num_passes=1, gfprops=GF_PROPS_BGS_VANILLA.copy())
bgs_y1_fiberonly_1pass.marker = '-'

bgs_y1_fiberonly = BGSGroupCatalog("Observed BGS Y1", Mode.FIBER_ASSIGNED_ONLY, 19.5, 21.0, gfprops=GF_PROPS_BGS_VANILLA.copy())
bgs_y1_fiberonly.marker = '-'

bgs_nn = BGSGroupCatalog("Nearest Neighbor BGS Y1", Mode.NEAREST_NEIGHBOR, 19.5, 21.0, gfprops=GF_PROPS_BGS_VANILLA.copy())
bgs_nn.marker = '-'

bgs_nn_sdsslike = BGSGroupCatalog("Nearest Neighbor BGS Y1 SDSS-like", Mode.NEAREST_NEIGHBOR, 17.7, 17.7, gfprops=GF_PROPS_BGS_VANILLA.copy())
bgs_nn_sdsslike.marker = '-'

bgs_y1mini_pzp_2_4_c1 = BGSGroupCatalog("BGS Y1 Mini PZP v2.4 C1", Mode.PHOTOZ_PLUS_v2, 19.5, 21.0, data_cut='Y1-Iron-Mini', extra_params=PZP_PARAMS_V24, gfprops=GF_PROPS_BGS_COLORS_C1.copy(), caldata_ctor=CalibrationData.BGS_Y1mini)
bgs_y1mini_pzp_2_4_c1.marker = '--'
bgs_y1mini_pzp_2_4_c1.color = 'darkgreen'

bgs_y1_pzp_2_4 = BGSGroupCatalog("BGS Y1 PZP v2.4 Vanilla", Mode.PHOTOZ_PLUS_v2, 19.5, 21.0, data_cut='Y1-Iron', extra_params=PZP_PARAMS_V24, gfprops=GF_PROPS_BGS_VANILLA.copy())
bgs_y1_pzp_2_4.marker = '-'
bgs_y1_pzp_2_4.color = 'darkgreen'

bgs_y1_pzp_2_4_c1 = BGSGroupCatalog("BGS Y1 PZP v2.4 C1", Mode.PHOTOZ_PLUS_v2, 19.5, 21.0, data_cut='Y1-Iron', extra_params=PZP_PARAMS_V24, gfprops=GF_PROPS_BGS_COLORS_C1.copy())
bgs_y1_pzp_2_4_c1.marker = '--'
bgs_y1_pzp_2_4_c1.color = 'darkgreen'

bgs_y1_pzp_2_4_c2 = BGSGroupCatalog("BGS Y1 C2", Mode.PHOTOZ_PLUS_v2, 19.5, 21.0, data_cut='Y1-Iron', extra_params=PZP_PARAMS_V24, gfprops=GF_PROPS_BGS_COLORS_C2.copy())
bgs_y1_pzp_2_4_c2.marker = '-'
bgs_y1_pzp_2_4_c2.GF_props['iterations'] = 10
bgs_y1_pzp_2_4_c2.color = 'darkgreen'

bgs_y1_pzp_2_4_c2_noffc = BGSGroupCatalog("BGS Y1 C2 No Cuts", Mode.PHOTOZ_PLUS_v2, 19.5, 21.0, data_cut='Y1-Iron', ffc=False, extra_params=PZP_PARAMS_V24, gfprops=GF_PROPS_BGS_COLORS_C2.copy())
bgs_y1_pzp_2_4_c2_noffc.marker = '-'
bgs_y1_pzp_2_4_c2_noffc.color = 'darkred'

bgs_y1mini_hybrid_mcmc = BGSGroupCatalog("BGS Y1 Mini Hybrid MCMC", Mode.PHOTOZ_PLUS_v2, 19.5, 21.0, data_cut='Y1-Iron-Mini', extra_params=PZP_PARAMS_V24, gfprops=GF_PROPS_BGS_COLORS_C1.copy(), caldata_ctor=CalibrationData.BGS_Y1mini)

bgs_y1_hybrid_mcmc = BGSGroupCatalog("BGS Y1 Hybrid MCMC", Mode.PHOTOZ_PLUS_v2, 19.5, 21.0, data_cut='Y1-Iron', extra_params=PZP_PARAMS_V24, gfprops=GF_PROPS_BGS_COLORS_C1.copy(), caldata_ctor=CalibrationData.BGS_Y1_6bin)
bgs_y1_hybrid_mcmc.marker = '.'
bgs_y1_hybrid_mcmc.color = 'darkgreen'

bgs_y1_hybrid8_mcmc = BGSGroupCatalog("BGS Y1 Hybrid8 MCMC", Mode.PHOTOZ_PLUS_v2, 19.5, 21.0, data_cut='Y1-Iron', extra_params=PZP_PARAMS_V24, gfprops=GF_PROPS_BGS_COLORS_C1.copy(), caldata_ctor=CalibrationData.BGS_Y1_8bin)
bgs_y1_hybrid8_mcmc.marker = '.'
bgs_y1_hybrid8_mcmc.color = 'darkgreen'

bgs_y3_pzp_2_4 = BGSGroupCatalog("BGS Y3 PZP v2.4 Vanilla", Mode.PHOTOZ_PLUS_v2, 19.5, 21.0, data_cut='Y3-Loa', extra_params=PZP_PARAMS_V24, gfprops=GF_PROPS_BGS_VANILLA.copy())
bgs_y3_pzp_2_4.color = 'darkorange'
bgs_y3_pzp_2_4.marker = '-'

bgs_y3_pzp_2_4_c1 = BGSGroupCatalog("BGS Y3 PZP v2.4 C1", Mode.PHOTOZ_PLUS_v2, 19.5, 21.0, data_cut='Y3-Loa', extra_params=PZP_PARAMS_V24, gfprops=GF_PROPS_BGS_COLORS_C1.copy())
bgs_y3_pzp_2_4_c1.color = 'darkorange'
bgs_y3_pzp_2_4_c1.marker = '--'

bgs_y3_pzp_2_4_c2 = BGSGroupCatalog("BGS Y3 C2", Mode.PHOTOZ_PLUS_v2, 19.5, 21.0, data_cut='Y3-Loa', extra_params=PZP_PARAMS_V24, gfprops=GF_PROPS_BGS_COLORS_C2.copy())
bgs_y3_pzp_2_4_c2.color = 'darkorange'
bgs_y3_pzp_2_4_c2.marker = '-'
bgs_y3_pzp_2_4_c2.GF_props['iterations'] = 10

bgs_y3_fiberonly = BGSGroupCatalog("Observed BGS Y3 <19.5", Mode.FIBER_ASSIGNED_ONLY, 19.5, 21.0, data_cut='Y3-Loa', gfprops=GF_PROPS_BGS_VANILLA.copy())
bgs_y3_fiberonly.marker = '-'

bgs_y3_fiberonly_1pass = BGSGroupCatalog("Observed BGS Y3 <19.5 1pass", Mode.FIBER_ASSIGNED_ONLY, 19.5, 21.0, num_passes=1, data_cut='Y3-Loa', gfprops=GF_PROPS_BGS_VANILLA.copy())
bgs_y3_fiberonly_1pass.marker = '.'
#0.004 = 4.5
# To translate between old sigma and new one, use new = -log10(2*old^2)
bgs_sv3_pz_1_10p = BGSGroupCatalog(
    "Photo-z Plus v1 BGS sv3 10pass", 
    Mode.PHOTOZ_PLUS_v1, 
    19.5, 
    21.0, 
    num_passes=10, 
    data_cut='sv3', 
    sdss_fill=True, 
    extra_params=(5, (0.8, 1.0, 4.5, 4.0)), 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_pz_1_10p.color = [0.0, 1.0, 1.0]

bgs_sv3_pz_2_4_10p = BGSGroupCatalog(
    "Photo-z Plus v2.4 BGS sv3 10pass", 
    Mode.PHOTOZ_PLUS_v2, 
    19.5, 
    21.0, 
    num_passes=10, 
    data_cut='sv3', 
    sdss_fill=True, 
    extra_params=PZP_PARAMS_V24, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_pz_2_4_10p.color = 'k'

bgs_sv3_pz_2_4_10p_c1 = BGSGroupCatalog(
    "Photo-z Plus v2.4 BGS sv3 10pass C1", 
    Mode.PHOTOZ_PLUS_v2, 
    19.5, 
    21.0, 
    num_passes=10, 
    data_cut='sv3', 
    sdss_fill=True, 
    extra_params=PZP_PARAMS_V24, 
    gfprops=GF_PROPS_BGS_COLORS_C1.copy()
)
bgs_sv3_pz_2_4_10p_c1.color = 'k'

bgs_sv3_pz_2_4_10p_c2 = BGSGroupCatalog(
    "Photo-z Plus v2.4 BGS sv3 10pass C2", 
    Mode.PHOTOZ_PLUS_v2, 
    19.5, 
    21.0, 
    num_passes=10, 
    data_cut='sv3', 
    sdss_fill=True, 
    extra_params=PZP_PARAMS_V24, 
    gfprops=GF_PROPS_BGS_COLORS_C2.copy()
)
bgs_sv3_pz_2_4_10p_c2.GF_props['iterations'] = 10
bgs_sv3_pz_2_4_10p_c2.color = 'k'

# This one was calibrated on SDSS data
bgs_sv3_10p_mcmc = BGSGroupCatalog(
    "BGS SV3 PZPv2.4 10pass MCMC", 
    Mode.PHOTOZ_PLUS_v2, 
    19.5, 
    21.0, 
    num_passes=10, 
    data_cut='sv3', 
    sdss_fill=True, 
    extra_params=PZP_PARAMS_V24, 
    gfprops=GF_PROPS_BGS_COLORS_C1.copy(),
    caldata_ctor=CalibrationData.SDSS_4bin
)
bgs_sv3_10p_mcmc.color = 'k'

bgs_sv3_hybrid_mcmc = BGSGroupCatalog(
    "BGS SV3 Hybrid MCMC", 
    Mode.PHOTOZ_PLUS_v2, 
    19.5, 
    21.0, 
    num_passes=10, 
    data_cut='sv3', 
    sdss_fill=True, 
    extra_params=PZP_PARAMS_V24, 
    gfprops=GF_PROPS_BGS_COLORS_C1.copy()
)

bgs_y3_like_sv3_hybrid_mcmc_new = BGSGroupCatalog(
    "BGS Y3-Like-SV3 Hybrid MCMC NEW", 
    Mode.PHOTOZ_PLUS_v2, 
    19.5, 
    21.0, 
    num_passes=1, 
    data_cut='Y3-Loa-SV3Cut', 
    sdss_fill=True, 
    extra_params=PZP_PARAMS_V24, 
    gfprops=GF_PROPS_BGS_COLORS_C1.copy()
)

bgs_sv3_pz_2_4_10p_old = BGSGroupCatalog(
    "Photo-z Plus v2.4 BGS sv3 10pass OLD", 
    Mode.PHOTOZ_PLUS_v2, 
    19.5, 
    21.0, 
    num_passes=10, 
    data_cut='sv3', 
    sdss_fill=True, 
    extra_params=PZP_PARAMS_V24, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_pz_2_4_10p_old.color = 'k'

bgs_sv3_pz_1_0_7p = BGSGroupCatalog(
    "Photo-z Plus v1 BGS sv3 7pass", 
    Mode.PHOTOZ_PLUS_v1, 
    19.5, 
    21.0, 
    num_passes=10, 
    drop_passes=3, 
    data_cut='sv3', 
    sdss_fill=False, 
    extra_params=(5, (0.8, 1.0, 4.5, 4.0)), 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_pz_1_0_7p.color = 'g'

bgs_sv3_pz_2_0_7p = BGSGroupCatalog(
    "Photo-z Plus v2.0 BGS sv3 7pass", 
    Mode.PHOTOZ_PLUS_v2, 
    19.5, 
    21.0, 
    num_passes=10, 
    drop_passes=3, 
    data_cut='sv3', 
    sdss_fill=False, 
    extra_params=(1, [0.0, 0.0, 3.0]), 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_pz_2_0_7p.color = 'red'

bgs_sv3_pz_2_4_8p = BGSGroupCatalog(
    "Photo-z Plus v2.4 BGS sv3 8pass", 
    Mode.PHOTOZ_PLUS_v2, 
    19.5, 
    21.0, 
    num_passes=10, 
    drop_passes=2, 
    data_cut='sv3', 
    sdss_fill=False, 
    extra_params=PZP_PARAMS_V24, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_pz_2_4_8p.color = 'orange'

bgs_sv3_pz_2_4_7p = BGSGroupCatalog(
    "Photo-z Plus v2.4 BGS sv3 7pass", 
    Mode.PHOTOZ_PLUS_v2, 
    19.5, 
    21.0, 
    num_passes=10, 
    drop_passes=3, 
    data_cut='sv3', 
    sdss_fill=False, 
    extra_params=PZP_PARAMS_V24, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_pz_2_4_7p.color = 'darkorange'

bgs_sv3_pz_2_4_6p = BGSGroupCatalog(
    "Photo-z Plus v2.4 BGS sv3 6pass", 
    Mode.PHOTOZ_PLUS_v2, 
    19.5, 
    21.0, 
    num_passes=10, 
    drop_passes=4, 
    data_cut='sv3', 
    sdss_fill=False, 
    extra_params=PZP_PARAMS_V24, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_pz_2_4_6p.color = 'darkorange'

bgs_sv3_pz_3_0_7p = BGSGroupCatalog(
    "Photo-z Plus v3.0 BGS sv3 7pass", 
    Mode.PHOTOZ_PLUS_v3, 
    19.5, 
    21.0, 
    num_passes=10, 
    drop_passes=3, 
    data_cut='sv3', 
    sdss_fill=False, 
    extra_params=(
        4, 
        [0.8104, 0.9215, 2.867], 
        [0.9102, 0.7376, 3.0275], 
        [0.8986, 1.0397, 2.6287], 
        [0.7488, 0.9489, 2.9319]
    ), 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_pz_3_0_7p.color = [0.45, 0.40, 0.0]

bgs_sv3_pz_3_1_7p = BGSGroupCatalog(
    "Photo-z Plus v3.1 BGS sv3 7pass", 
    Mode.PHOTOZ_PLUS_v3, 
    19.5, 
    21.0, 
    num_passes=10, 
    drop_passes=3, 
    data_cut='sv3', 
    sdss_fill=False, 
    extra_params=PZP_PARAMS_V24, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_pz_3_1_7p.color = 'purple'

bgs_y3_like_sv3_pz_2_4 = BGSGroupCatalog(
    "Photo-z Plus v2.4 BGS Y3 like-sv3", 
    Mode.PHOTOZ_PLUS_v2, 
    19.5, 
    21.0, 
    num_passes=1, 
    data_cut='Y3-Loa-SV3Cut', 
    sdss_fill=False, 
    extra_params=PZP_PARAMS_V24, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_y3_like_sv3_pz_2_4.color = 'slateblue'

bgs_y3_like_sv3_pz_2_6 = BGSGroupCatalog(
    "Photo-z Plus v2.6 BGS Y3 like-sv3", 
    Mode.PHOTOZ_PLUS_v2, 
    19.5, 
    21.0, 
    num_passes=1, 
    data_cut='Y3-Loa-SV3Cut', 
    sdss_fill=False, 
    extra_params=PZP_PARAMS_V26, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_y3_like_sv3_pz_2_6.color = 'gold'

bgs_y3_like_sv3_pz_4_0 = BGSGroupCatalog(
    "Photo-z Plus v4.0 BGS Y3 like-sv3", 
    Mode.PHOTOZ_PLUS_v4, 
    19.5, 
    21.0, 
    num_passes=1, 
    data_cut='Y3-Loa-SV3Cut', 
    sdss_fill=False, 
    extra_params=PZP_PARAMS_V40, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_y3_like_sv3_pz_4_0.color = 'gold'

bgs_y3_like_sv3_pz_2_4_c1 = BGSGroupCatalog(
    "Photo-z Plus v2.4 BGS Y3 like-sv3 C1", 
    Mode.PHOTOZ_PLUS_v2, 
    19.5, 
    21.0, 
    num_passes=1, 
    data_cut='Y3-Loa-SV3Cut', 
    sdss_fill=False, 
    extra_params=PZP_PARAMS_V24, 
    gfprops=GF_PROPS_BGS_COLORS_C1.copy()
)
bgs_y3_like_sv3_pz_2_4_c1.color = 'slateblue'

bgs_y3_like_sv3_pz_2_4_c2 = BGSGroupCatalog(
    "Photo-z Plus v2.4 BGS Y3 like-sv3 C2", 
    Mode.PHOTOZ_PLUS_v2, 
    19.5, 
    21.0, 
    num_passes=1, 
    data_cut='Y3-Loa-SV3Cut', 
    sdss_fill=False, 
    extra_params=PZP_PARAMS_V24, 
    gfprops=GF_PROPS_BGS_COLORS_C2.copy()
)
bgs_y3_like_sv3_pz_2_4_c2.GF_props['iterations'] = 10
bgs_y3_like_sv3_pz_2_4_c2.color = 'slateblue'

bgs_y3_like_sv3_pz_2_5 = BGSGroupCatalog(
    "Photo-z Plus v2.5 BGS Y3 like-sv3", 
    Mode.PHOTOZ_PLUS_v2, 
    19.5, 
    21.0, 
    num_passes=1, 
    data_cut='Y3-Loa-SV3Cut', 
    sdss_fill=False, 
    extra_params=(10, [1.10, 1.5467, 3.0134], [1.2229, 0.8628, 2.5882], [0.8706, 0.6126, 2.4447], [1.1163, 1.2938, 3.1650]), 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_y3_like_sv3_pz_2_5.color = 'darkorange'
bgs_y3_like_sv3_pz_2_5.marker = '--'

bgs_y3_like_sv3_fiberonly = BGSGroupCatalog(
    "Observed BGS Y3 like-sv3", 
    Mode.FIBER_ASSIGNED_ONLY, 
    19.5, 
    21.0, 
    num_passes=1, 
    data_cut='Y3-Loa-SV3Cut', 
    sdss_fill=False, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_y3_like_sv3_fiberonly.color = 'orange'

bgs_y3_like_sv3_pz_2_0 = BGSGroupCatalog(
    "Photo-z Plus v2.0 BGS Y3 like-sv3", 
    Mode.PHOTOZ_PLUS_v2, 
    19.5, 
    21.0, 
    num_passes=1, 
    data_cut='Y3-Loa-SV3Cut', 
    sdss_fill=False, 
    extra_params=(1, [0.0, 0.0, 3.0]), 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_y3_like_sv3_pz_2_0.color = 'red'

bgs_y3_like_sv3_nn = BGSGroupCatalog(
    "Nearest Neighbor BGS Y3 like-sv3", 
    Mode.NEAREST_NEIGHBOR, 
    19.5, 
    21.0, 
    num_passes=1, 
    data_cut='Y3-Loa-SV3Cut', 
    sdss_fill=False, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_y3_like_sv3_nn.color = 'green'

bgs_sv3_fiberonly_10p = BGSGroupCatalog(
    "Observed BGS sv3 10pass", 
    Mode.FIBER_ASSIGNED_ONLY, 
    19.5, 
    21.0, 
    num_passes=10, 
    data_cut='sv3', 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_fiberonly_10p.color = 'orange'

bgs_sv3_simple_4_10p = BGSGroupCatalog(
    "Simple v4 BGS sv3 10pass", 
    Mode.SIMPLE_v4, 
    19.5, 
    21.0, 
    num_passes=10, 
    data_cut='sv3', 
    sdss_fill=False, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_simple_4_10p.color = [0.0, 1.0, 0.0]

bgs_sv3_simple_4_9p = BGSGroupCatalog(
    "Simple v4 BGS sv3 9pass", 
    Mode.SIMPLE_v4, 
    19.5, 
    21.0, 
    num_passes=10, 
    drop_passes=1, 
    data_cut='sv3', 
    sdss_fill=False, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_simple_4_9p.color = [0.0, 0.9, 0.0]

bgs_sv3_simple_4_8p = BGSGroupCatalog(
    "Simple v4 BGS sv3 8pass", 
    Mode.SIMPLE_v4, 
    19.5, 
    21.0, 
    num_passes=10, 
    drop_passes=2, 
    data_cut='sv3', 
    sdss_fill=False, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_simple_4_8p.color = [0.0, 0.8, 0.0]

bgs_sv3_simple_4_7p = BGSGroupCatalog(
    "Simple v4 BGS sv3 7pass", 
    Mode.SIMPLE_v4, 
    19.5, 
    21.0, 
    num_passes=10, 
    drop_passes=3, 
    data_cut='sv3', 
    sdss_fill=False, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_simple_4_7p.color = [0.0, 0.7, 0.0]

bgs_sv3_simple_4_6p = BGSGroupCatalog(
    "Simple v4 BGS sv3 6pass", 
    Mode.SIMPLE_v4, 
    19.5, 
    21.0, 
    num_passes=10, 
    drop_passes=4, 
    data_cut='sv3', 
    sdss_fill=False, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_simple_4_6p.color = [0.0, 0.6, 0.0]

bgs_sv3_simple_4_5p = BGSGroupCatalog(
    "Simple v4 BGS sv3 5pass", 
    Mode.SIMPLE_v4, 
    19.5, 
    21.0, 
    num_passes=10, 
    drop_passes=5, 
    data_cut='sv3', 
    sdss_fill=False, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_simple_4_5p.color = [0.0, 0.5, 0.0]

bgs_sv3_simple_4_4p = BGSGroupCatalog(
    "Simple v4 BGS sv3 4pass", 
    Mode.SIMPLE_v4, 
    19.5, 
    21.0, 
    num_passes=10, 
    drop_passes=6, 
    data_cut='sv3', 
    sdss_fill=False, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_simple_4_4p.color = [0.0, 0.4, 0.0]

bgs_sv3_simple_4_3p = BGSGroupCatalog(
    "Simple v4 BGS sv3 3pass", 
    Mode.SIMPLE_v4, 
    19.5, 
    21.0, 
    num_passes=10, 
    drop_passes=7, 
    data_cut='sv3', 
    sdss_fill=False, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_simple_4_3p.color = [0.0, 0.3, 0.0]

bgs_sv3_simple_4_2p = BGSGroupCatalog(
    "Simple v4 BGS sv3 2pass", 
    Mode.SIMPLE_v4, 
    19.5, 
    21.0, 
    num_passes=10, 
    drop_passes=8, 
    data_cut='sv3', 
    sdss_fill=False, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_simple_4_2p.color = [0.0, 0.2, 0.0]

bgs_sv3_simple_4_1p = BGSGroupCatalog(
    "Simple v4 BGS sv3 1pass", 
    Mode.SIMPLE_v4, 
    19.5, 
    21.0, 
    num_passes=10, 
    drop_passes=9, 
    data_cut='sv3', 
    sdss_fill=False, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_simple_4_1p.color = [0.0, 0.1, 0.0]

bgs_sv3_nn_10p = BGSGroupCatalog(
    "NN BGS sv3 10pass", 
    Mode.NEAREST_NEIGHBOR, 
    19.5, 
    21.0, 
    num_passes=10, 
    drop_passes=0, 
    data_cut='sv3', 
    sdss_fill=True, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_nn_10p.color = [1.0, 1.0, 0.0]

bgs_sv3_nn_7p = BGSGroupCatalog(
    "NN BGS sv3 7pass", 
    Mode.NEAREST_NEIGHBOR, 
    19.5, 
    21.0, 
    num_passes=10, 
    drop_passes=3, 
    data_cut='sv3', 
    sdss_fill=False, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_nn_7p.color = [0.7, 0.7, 0.0]

bgs_sv3_nn_6p = BGSGroupCatalog(
    "NN BGS sv3 6pass", 
    Mode.NEAREST_NEIGHBOR, 
    19.5, 
    21.0, 
    num_passes=10, 
    drop_passes=4, 
    data_cut='sv3', 
    sdss_fill=False, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_nn_6p.color = [0.6, 0.6, 0.0]

bgs_sv3_fiberonly_10p = BGSGroupCatalog(
    "Observed BGS sv3 10pass", 
    Mode.FIBER_ASSIGNED_ONLY, 
    19.5, 
    21.0, 
    num_passes=10, 
    data_cut='sv3', 
    sdss_fill=True, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_fiberonly_10p.color = 'r'

bgs_sv3_simple_5_7p = BGSGroupCatalog(
    "Simple v5 BGS sv3 7pass", 
    Mode.SIMPLE_v5, 
    19.5, 
    21.0, 
    num_passes=10, 
    drop_passes=3, 
    data_cut='sv3', 
    sdss_fill=False, 
    gfprops=GF_PROPS_BGS_VANILLA.copy()
)
bgs_sv3_simple_5_7p.color = [0.0, 0.7, 0.0]








##############################################
# LISTS
##############################################3


sdss_list : list[GroupCatalog] = [
    sdss_vanilla,
    sdss_colors,
    sdss_colors_chi,
    sdss_vanilla_v2,
    sdss_colors_v2,
    sdss_colors_chi_v2,
    sdss_bgscut,
    sdss_colors_v2_desiparams_v1,
]
uchuu_list : list[GroupCatalog] = [
    uchuu_all,
]
mxxl_list : list[GroupCatalog] = [
    mxxl_all,
    #mxxl_all_c,
    mxxl_fiberonly,
    #mxxl_fiberonly_c,
    mxxl_nn,
    #mxxl_nn_c,
    mxxl_simple_2,
    #mxxl_simple_2_c,
    mxxl_simple_4,
    #mxxl_simple_4_c,
]
bgs_sv3_list : list[GroupCatalog] = [
    #bgs_sv3_nn_10p,
    #bgs_sv3_nn_7p,
    bgs_sv3_fiberonly_10p,
    #bgs_sv3_simple_4_10p,
    #bgs_sv3_simple_4_7p,
    #bgs_sv3_simple_5_7p,
    #bgs_sv3_pz_1_10p,
    #bgs_sv3_pz_2_0_7p,
    bgs_sv3_pz_2_4_10p, 
    bgs_sv3_pz_2_4_10p_c1, # Truthiest catalog
    bgs_sv3_pz_2_4_10p_c2, # Truthiest catalog
    #bgs_sv3_pz_1_0_7p,
    #bgs_sv3_pz_2_4_7p,
    #bgs_sv3_pz_2_5_7p,
    #bgs_sv3_pz_3_1_7p,
    bgs_y3_like_sv3_pz_2_4,
    bgs_y3_like_sv3_pz_2_4_c1,
    bgs_y3_like_sv3_pz_2_4_c2,
    bgs_y3_like_sv3_fiberonly,
    bgs_y3_like_sv3_pz_2_0,
    bgs_y3_like_sv3_nn,
]
bgs_aux_list : list[GroupCatalog] = [
    bgs_y1_fiberonly_1pass,
    bgs_y3_fiberonly_1pass,
    bgs_nn_sdsslike,
]
bgs_y1_list : list[GroupCatalog] = [
    bgs_y1_pzp_2_4_c2,
    bgs_y1mini_pzp_2_4_c1,
    bgs_y1_pzp_2_4,
]
bgs_y3_list : list[GroupCatalog] = [
    bgs_y3_pzp_2_4_c2,
    bgs_y3_pzp_2_4,

]