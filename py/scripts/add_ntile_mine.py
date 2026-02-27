#!/usr/bin/env python
# coding: utf-8


# In[ ]:

import sys
from astropy.table import Table
#if './SelfCalGroupFinder/py/' not in sys.path:
#    sys.path.append('./SelfCalGroupFinder/py/')
from bgs_helpers import add_NTILE_MINE_to_table, tile_to_region, sv3_poor_y3overlap
from multiprocessing import Pool
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_tracer", help="tracer type that subsample will come from", default='BGS_BRIGHT')
parser.add_argument("--basedir", help="base directory for input, default is SCRATCH",default='/global/cfs/cdirs/desi/survey/catalogs/')
parser.add_argument("--outdir", help="directory for out, default is SCRATCH",default=os.environ['SCRATCH'])
parser.add_argument("--version", help="catalog version for input",default='3.1')
parser.add_argument("--survey", help="e.g., Y1, DA2",default='SV3')
parser.add_argument("--verspec",help="version for redshifts",default='fuji')
parser.add_argument("--use_map_veto", help="string to include in full file name denoting whether map veto was applied",default='')
parser.add_argument("--passfilter", help="cut tables to this number of passes based on NTILE_MINE",default='')
parser.add_argument("--f", help="junk",default='')

args = parser.parse_args()

# This script adds the NTILE_MINE column to the randoms files for a given tracer.
# For the DATA, modify mkCat_subsamp_mine.py to be aware of the relevant "MERGED" file which has that column in it.
fn_pattern = f'{args.input_tracer}_XX_full{args.use_map_veto}.ran.fits'
dirin = f"{args.basedir}{args.survey}/LSS/{args.verspec}/LSScats/{args.version}/"

# If outdir doesn't exist, make it
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

def _add2ran(rn):
    fname_in = os.path.join(dirin, fn_pattern.replace('XX', str(rn)))
    fname_out = os.path.join(args.outdir, fn_pattern.replace('XX', str(rn)))

    # If output file already exists, skip
    #if os.path.exists(fname_out):
    #    print(f"Output file {fname_out} already exists, skipping")
    #    return

    tbl = add_NTILE_MINE_to_table(fname_in, args.survey)

    # Remove the two regions for SV3
    if args.survey == 'SV3':
        ntid = tbl['NEAREST_TILEIDS'][:,0]
        region = tile_to_region(ntid)

        # Drop the bad two regions for equal comparison later
        to_remove = np.isin(region, sv3_poor_y3overlap)
        l1 = len(tbl)
        tbl = tbl[~to_remove]
        print(f"Length before/after SV3 region removal: {l1} {len(tbl)}", flush=True)
    
    # If passfilter is set, filter to only those passes
    if args.passfilter != '':
        passfilter = int(args.passfilter)
        print(f"Applying passfilter of {passfilter}", flush=True)
        to_keep = tbl['NTILE_MINE'] >= passfilter
        l1 = len(tbl)
        tbl = tbl[to_keep]
        print(f"Length before/after passfilter {passfilter}: {l1} {len(tbl)}", flush=True)

    tbl.write(fname_out, overwrite=True)
    print(f"Processed {rn}", flush=True)


# In[ ]:

with Pool() as pool:
    res = pool.map(_add2ran, np.arange(0,18))

# %%
