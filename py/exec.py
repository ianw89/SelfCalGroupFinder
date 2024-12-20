import sys
from urllib.parse import urljoin
import time
import asyncio
import concurrent.futures
from multiprocessing import Pool

# EXAMPLE USAGE
# nohup python exec.py 6 8 &> exec.out &


if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from groupcatalog import *
import catalog_definitions as cat
from dataloc import *
from pyutils import *

datasets_to_run: list[GroupCatalog] = []

# These run MCMC on the fiber incompleteness handling parameters, not group finder parameters.
# TODO we did this on SV3 7pass, when Y3-likesv3 would have been better.
# If we revisit this, we should use the Y3-like sv3 data instead of this.

mcmc = BGSGroupCatalog("Photo-z Plus MCMC BGS sv3 7pass ", Mode.PHOTOZ_PLUS_v1, 19.5, 21.0, num_passes=10, drop_passes=3, data_cut='sv3', sdss_fill=False)
mcmc.GF_props = cat.GF_PROPS_VANILLA.copy()
# V2 was decided to be best
mcmc2 = BGSGroupCatalog("Photo-z Plus MCMC BGS sv3 7pass ", Mode.PHOTOZ_PLUS_v2, 19.5, 21.0, num_passes=10, drop_passes=3, data_cut='sv3', sdss_fill=False)
mcmc2.GF_props = cat.GF_PROPS_VANILLA.copy()

mcmc3 = BGSGroupCatalog("Photo-z Plus MCMC BGS sv3 7pass ", Mode.PHOTOZ_PLUS_v3, 19.5, 21.0, num_passes=10, drop_passes=3, data_cut='sv3', sdss_fill=False)
mcmc3.GF_props = cat.GF_PROPS_VANILLA.copy()

callable_list = [
    [mcmc], #0
    [mcmc2], #1
    [mcmc3], #2
    cat.sdss_list, #3
    cat.uchuu_list, #4
    cat.mxxl_list, #5
    cat.bgs_sv3_list, #6
    cat.bgs_aux_list, #7
    cat.bgs_main_list, #8
    [] #9
]

def process_gc(gc: GroupCatalog):
    name = gc.name
    print(f"***** process_gc({name}) start *****")
    #gc = deserialize(gc)
    gc.run_group_finder()
    gc.postprocess()
    #d.run_corrfunc()
    #gc.calculate_projected_clustering(with_extra_randoms=True) # 15m
    #gc.calculate_projected_clustering_in_magbins(with_extra_randoms=True) # 45m maybe?
    
    #if gc == cat.bgs_sv3_pz_2_4_10p:
    #    gc.add_jackknife_err_to_proj_clustering(with_extra_randoms=True, for_mag_bins=True)
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
        for arg in sys.argv[1:]:
            to_add = int(arg)
            datasets_to_run.extend(callable_list[to_add])

    asyncio.run(main_serial())
    #asyncio.run(main_threaded_parallel())