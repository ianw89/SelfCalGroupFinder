import os
import pickle
import sys
from urllib.parse import urljoin
import numpy as np
import time
from astropy.table import Table,join,vstack,unique
import asyncio
import concurrent.futures
from multiprocessing import Pool, TimeoutError
import tracemalloc
import astropy.coordinates as coord

if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from groupcatalog import GroupCatalog, BGSGroupCatalog, TestGroupCatalog, serialize, deserialize, SDSSGroupCatalog
import catalog_definitions as cat
from dataloc import *
from pyutils import *
import plotting as pp

sdss_list : list[GroupCatalog] = [
    cat.sdss_vanilla,
    cat.sdss_colors,
    cat.sdss_colors_chi,
    cat.sdss_vanilla_v2,
    cat.sdss_colors_v2,
    cat.sdss_colors_chi_v2,
    cat.sdss_bgscut,
]
uchuu_list : list[GroupCatalog] = [
    cat.uchuu_all,
]
mxxl_list : list[GroupCatalog] = [
    cat.mxxl_all,
    #cat.mxxl_all_c,
    cat.mxxl_fiberonly,
    #cat.mxxl_fiberonly_c,
    cat.mxxl_nn,
    #cat.mxxl_nn_c,
    cat.mxxl_simple_2,
    #cat.mxxl_simple_2_c,
    cat.mxxl_simple_4,
    #cat.mxxl_simple_4_c,
]
bgs_sv3_list : list[GroupCatalog] = [
    cat.bgs_sv3_nn_7p,
    cat.bgs_sv3_nn_6p,
    cat.bgs_sv3_fiberonly_10p,
    cat.bgs_sv3_simple_4_10p,
    cat.bgs_sv3_simple_4_9p,
    cat.bgs_sv3_simple_4_8p,
    cat.bgs_sv3_simple_4_7p,
    cat.bgs_sv3_simple_4_6p,
    cat.bgs_sv3_simple_4_5p,
    cat.bgs_sv3_simple_4_4p,
    cat.bgs_sv3_simple_4_3p,
    cat.bgs_sv3_simple_4_2p,
    cat.bgs_sv3_simple_4_1p,
    cat.bgs_sv3_simple_5_10p,
    cat.bgs_sv3_simple_5_9p,
    cat.bgs_sv3_simple_5_8p,
    cat.bgs_sv3_simple_5_7p,
    cat.bgs_sv3_pz_1_10p,
    cat.bgs_sv3_pz_1_0_7p,
    cat.bgs_sv3_pz_1_1_7p,
    cat.bgs_sv3_pz_1_2_7p,
    cat.bgs_sv3_pz_1_3_7p,
    cat.bgs_sv3_pz_2_0_7p,
]
bgs_y1_list : list[GroupCatalog] = [
    #cat.bgs_simple_4_old,
    cat.bgs_simple_4,
    #cat.bgs_simple_4_1pass,
    #cat.bgs_simple_4_no_sdss,
    cat.bgs_simple_4_4p,
    #cat.bgs_simple_4_c,
    ##cat.bgs_fiberonly,
    #cat.bgs_fiberonly_1pass,
    ##cat.bgs_nn,
    ##cat.bgs_nn_sdsslike,
    #cat.bgs_simple_2,
    #cat.bgs_simple_2_c,
    cat.bgs_simple_5,
]
bgs_y3_list : list[GroupCatalog] = [
    cat.bgs_y3_simple_4,
    cat.bgs_y3_simple_4_4p,
    cat.bgs_y3_fiberonly_1pass,
    cat.bgs_y3_fiberonly,
    cat.bgs_y3_simple_5,
]

datasets_to_run: list[GroupCatalog] = []
#datasets_to_run.extend(sdss_list)
#datasets_to_run.extend(uchuu_list)
#datasets_to_run.extend(mxxl_list)
#datasets_to_run.extend(bgs_sv3_list)  
#datasets_to_run.extend(bgs_y1_list)
#datasets_to_run.extend(bgs_y3_list)
#datasets_to_run.extend([cat.bgs_sv3_pz_2_1_7p])



mcmc = BGSGroupCatalog("Photo-z Plus MCMC BGS sv3 7pass ", Mode.PHOTOZ_PLUS_v1, 19.5, 21.0, num_passes=10, drop_passes=3, data_cut='sv3', sdss_fill=False)
mcmc.GF_props = cat.GF_PROPS_VANILLA.copy()

mcmc2 = BGSGroupCatalog("Photo-z Plus MCMC BGS sv3 7pass ", Mode.PHOTOZ_PLUS_v2, 19.5, 21.0, num_passes=10, drop_passes=3, data_cut='sv3', sdss_fill=False)
mcmc2.GF_props = cat.GF_PROPS_VANILLA.copy()

mcmc3 = BGSGroupCatalog("Photo-z Plus MCMC BGS sv3 7pass ", Mode.PHOTOZ_PLUS_v3, 19.5, 21.0, num_passes=10, drop_passes=3, data_cut='sv3', sdss_fill=False)
mcmc3.GF_props = cat.GF_PROPS_VANILLA.copy()

callable_list = [
    [mcmc], #0
    [mcmc2], #1
    [mcmc3], #2
    sdss_list, #3
    uchuu_list, #4
    mxxl_list, #5
    bgs_sv3_list, #6
    bgs_y1_list, #7
    bgs_y3_list, #8
]

def process_gc(gc: GroupCatalog):
    name = gc.name
    print(f"***** process_gc({name}) start *****")
    #gc = deserialize(d)
    gc.run_group_finder()
    gc.postprocess()
    #d.run_corrfunc()
    gc.dump()
    del(gc)
    print(f"+++++ process_gc({name}) done +++++")

async def process_wrapper(gc: GroupCatalog):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, process_gc, gc)

async def process_wrapper_mp(gc: GroupCatalog):
    loop = asyncio.get_running_loop()
    with concurrent.futures.ProcessPoolExecutor() as pool:
        await loop.run_in_executor(pool, process_gc, gc)

async def main_serial():
    t1 = time.time()
    for gc in datasets_to_run:
        await process_wrapper(gc)
    t2 = time.time()
    print(f"\n\nMAIN TOTAL TIME: {t2-t1}")
    return t2-t1

async def main_threaded_parallel():
    tasks = []
    t1 = time.time()
    for gc in datasets_to_run:
        tasks.append(asyncio.create_task(process_wrapper(gc)))
    await asyncio.gather(*tasks)
    t2 = time.time()
    print(f"\n\nMAIN TOTAL TIME: {t2-t1}")
    return t2-t1

async def main_multiproc_parallel():
    tasks = []
    t1 = time.time()
    for gc in datasets_to_run:
        tasks.append(asyncio.create_task(process_wrapper_mp(gc)))
    await asyncio.gather(*tasks)
    t2 = time.time()
    print(f"\n\nMAIN TOTAL TIME: {t2-t1}")
    return t2-t1

def main_multiproc_pool():
    t1 = time.time()
    with Pool(processes=3) as pool:
        pool.map(process_gc, datasets_to_run)
    t2 = time.time()
    print(f"\n\nMAIN TOTAL TIME: {t2-t1}")
    return t2-t1

def perf_test():
    #tracemalloc.start()
    t1=asyncio.run(main_serial())
    #print(tracemalloc.get_traced_memory())
    #tracemalloc.stop()
    
    #tracemalloc.start()
    t2=asyncio.run(main_threaded_parallel())
    #print(tracemalloc.get_traced_memory())
    #tracemalloc.stop()

    #tracemalloc.start()
    t3=asyncio.run(main_multiproc_parallel())
    #print(tracemalloc.get_traced_memory())
    #tracemalloc.stop()

    # Tracing doesn't go to subprocesses, oh well
    #tracemalloc.start()
    t4=main_multiproc_pool()
    #print(tracemalloc.get_traced_memory())
    #tracemalloc.stop()

    print(f"Serial: {t1:.2f}")
    print(f"Threaded: {t2:.2f}")
    print(f"Multiproc: {t3:.2f}")
    print(f"Pool: {t4:.2f}")

if __name__ == "__main__":
    #perf_test()

    if len(sys.argv[1]) > 0:
        to_add = int(sys.argv[1])
        datasets_to_run.extend(callable_list[to_add])

    # Winner
    asyncio.run(main_threaded_parallel())