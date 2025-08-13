#!/usr/bin/env python
# coding: utf-8


# In[ ]:

import sys
from astropy.table import Table
#if './SelfCalGroupFinder/py/' not in sys.path:
#    sys.path.append('./SelfCalGroupFinder/py/')
from bgs_helpers import add_NTILE_MINE_to_table
from multiprocessing import Pool
import numpy as np
import os

fn_pattern = 'BGS_BRIGHT_XX_full_HPmapcut.ran.fits'
dirin = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5pip/'
dirout = '/global/cfs/cdirs/desi/users/ianw89/newclustering/Y1/LSS/iron/LSScats/v1.5pip/'

def _add2ran(rn):
    fname_in = os.path.join(dirin, fn_pattern.replace('XX', str(rn)))
    fname_out = os.path.join(dirout, fn_pattern.replace('XX', str(rn)))

    tbl = add_NTILE_MINE_to_table(fname_in, "1")

    tbl.write(fname_out, overwrite=True)
    print(f"Processed {rn}")


# In[ ]:

with Pool() as pool:
    res = pool.map(_add2ran, np.arange(3,18))
#for rn in np.arange(3,18):
#    _add2ran(rn)

# %%
