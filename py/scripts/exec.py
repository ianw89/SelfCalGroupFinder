import sys
import time
import asyncio
import concurrent.futures
from multiprocessing import Pool

# EXAMPLE USAGE
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# nohup python3 py/scripts/exec.py 6 7 8 9 &> variations.out &
# nohup python3 py/scripts/exec.py mcmc 12 x0 &> y1mini_mcmc0_2.out &
# nohup python3 py/scripts/exec.py mcmc 13 x5 &> y1full_mcmc5_0.out &
# nohup python3 py/scripts/exec.py mcmc 14 x7 &> y1full8_mcmc7_1.out &
# nohup python3 py/scripts/exec.py 0 &> pzp_mcmc.out &


execution_mode = 'once' # or 'clustering' or 'mcmc'
mcmcnum = None # Will make a new folder
mcmc_iter = 1000 # x 20 walkers

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

# New MCMC of the fiber incompleteness handling parameters using the Y3 cut
mcmc = BGSGroupCatalog(
    "BGS Y3 Like SV3 PZP MCMC", 
    Mode.PHOTOZ_PLUS_v4, 
    19.5, 
    21.0, 
    num_passes=1, 
    data_cut='Y3-Loa-SV3Cut', 
    sdss_fill=False, 
    gfprops=GF_PROPS_BGS_VANILLA.copy())
# This is the old MCMC that led to finding the nice v2.4 PZP parameter set. Actually it might have been v3.1 that I just changed mode to v2 instead of v3 after.
mcmc2 = BGSGroupCatalog("Photo-z Plus MCMC BGS sv3 7pass ", Mode.PHOTOZ_PLUS_v2, 19.5, 21.0, num_passes=10, drop_passes=3, data_cut='sv3', sdss_fill=False, gfprops=cat.GF_PROPS_BGS_VANILLA.copy())

callable_list = [
    [mcmc], #0
    [mcmc2], #1
    [cat.bgs_sv3_hybrid_mcmc], #2
    cat.sdss_list, #3
    cat.uchuu_list, #4
    cat.mxxl_list, #5
    cat.bgs_sv3_list, #6
    cat.bgs_aux_list, #7
    cat.bgs_y1_list, #8
    cat.bgs_y3_list, #9
    [cat.bgs_y3_like_sv3_hybrid_mcmc_new], #10
    [cat.bgs_sv3_10p_mcmc], #11 sloan one
    [cat.bgs_y1mini_hybrid_mcmc], # 12
    [cat.bgs_y1_hybrid_mcmc], # 13
    [cat.bgs_y1_hybrid8_mcmc], # 14
]

def process_gc(gc: GroupCatalog):
    name = gc.name
    print(f"***** process_gc({name}) start *****")

    if execution_mode == 'once':
        result = gc.run_group_finder(popmock=True)
        if result:
            gc.calc_wp_for_mock()
            gc.postprocess()
        else:
            print(f"Group finder failed for {name}, skipping further processing.")
            return

    elif execution_mode == 'clustering':
        gc.run_group_finder(popmock=True)
        gc.calc_wp_for_mock()
        gc.postprocess()
        gc.calculate_projected_clustering(with_extra_randoms=True)
        gc.calculate_projected_clustering_in_magbins(with_extra_randoms=True) 
        if gc == cat.bgs_sv3_pz_2_4_10p:
            gc.add_jackknife_err_to_proj_clustering(with_extra_randoms=True, for_mag_bins=True)

    elif execution_mode == 'mcmc':
        gc.setup_GF_mcmc(mcmc_num=mcmcnum)
        gc.run_GF_mcmc(mcmc_iter)


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
            if arg == 'mcmc':
                execution_mode = 'mcmc'
            elif arg == 'clustering':
                execution_mode = 'clustering'
            elif arg == 'once':
                execution_mode = 'once'
            elif arg.startswith('x'): # specify mcmc folder number
                arg = arg[1:]
                mcmcnum = int(arg)
            # Otherwise we expect an int, which is the index of the catalogs definitions to run
            else:
                to_add = int(arg)
                datasets_to_run.extend(callable_list[to_add])

    asyncio.run(main_serial())
    #asyncio.run(main_threaded_parallel())