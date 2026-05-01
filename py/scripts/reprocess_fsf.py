#!/usr/bin/env python
# coding: utf-8

# Merged files are my concept of a single file that contains all the info needed for a BGS-like catalog.
# It uses an LSS Catalog, a photo-vac, fastspecfit, and a photo-z built from photo-z sweeps. 
# It contains all the galaxies we want in our catalog, including extra ones we will use in the neighbor catalog
# during preprocessing to deal with incompleteness. Most cuts are NOT made yet, but columns are 
# selected down to be generally what we need.

# In[ ]:


import sys
if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from dataloc import *
from bgs_helpers import *
from astropy.table import Table

# jupyter nbconvert --to script build-merged-file.ipynb


# In[ ]:

if os.path.exists(BGS_Y3_FASTSPEC_FILE):
    # Delete it
    os.remove(BGS_Y3_FASTSPEC_FILE)

fsf = read_fastspecfit_y3() # Will rebuild
# %%
