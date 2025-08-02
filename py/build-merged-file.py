#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from dataloc import *
from bgs_helpers import *
from astropy.table import Table

# jupyter nbconvert --to script build-merged-file.ipynb


# In[ ]:


# Creates SV3 FUJI merged file (no Y3 supplement yet)
create_merged_file(BGS_SV3_ANY_FULL_FILE, IAN_BGS_SV3_MERGED_NOY3_FILE, 'sv3', photoz_wspec=False)

# Now remove galaxies in the patches of SV3 that have poor overlap with Y3
sv3_table: Table = Table.read(IAN_BGS_SV3_MERGED_NOY3_FILE, format='fits')
print(len(sv3_table))

sv3_table['region'] = tile_to_region(sv3_table['NEAREST_TILEIDS'][:,0])
to_remove = np.isin(sv3_table['region'], sv3_poor_y3overlap)
sv3_table.remove_rows(to_remove)
print(len(sv3_table))

sv3_table.write(IAN_BGS_SV3_MERGED_NOY3_FILE, format='fits', overwrite='True')


# In[ ]:


# Creates Y1 IRON merged file
create_merged_file(BGS_Y1_ANY_FULL_FILE, IAN_BGS_Y1_MERGED_FILE, "1")


# In[ ]:


# Creates Y3 LOA merged file
create_merged_file(BGS_Y3_ANY_FULL_FILE, IAN_BGS_Y3_MERGED_FILE_LOA, "3")

# Add columns to allow the possibility to make a Y3 catalog that is cut to SV3 regions.
table = Table.read(IAN_BGS_Y3_MERGED_FILE_LOA, format='fits')
sv3tiles = read_tiles_Y3_sv3()
ntiles_inside, nearest_tile_ids = find_tiles_for_galaxies(sv3tiles, table_to_df(table), 10)
if 'NTILE_MINE_SV3' in table.columns:
    table.remove_columns(['NTILE_MINE_SV3'])
if 'NEAREST_TILEIDS_SV3' in table.columns:
    table.remove_columns(['NEAREST_TILEIDS_SV3'])
table.add_column(ntiles_inside, name="NTILE_MINE_SV3")
table.add_column(nearest_tile_ids, name="NEAREST_TILEIDS_SV3")

table.write(IAN_BGS_Y3_MERGED_FILE_LOA, format='fits', overwrite=True)
del table

# In[ ]:


# Now add Y3 galaxies to supplement the NN catalog, especialy valuable at the edges of the regions
# They won't go into the main catalog because their NTILE_MINE is < 10
supplement_sv3_merged_file_with_y3(IAN_BGS_SV3_MERGED_NOY3_FILE, IAN_BGS_Y3_MERGED_FILE_LOA, IAN_BGS_SV3_MERGED_FILE)


# In[ ]:


# Make another SV3 version with all 20 regions
create_merged_file(BGS_SV3_ANY_FULL_FILE, IAN_BGS_SV3_MERGED_FULL_FILE, 'sv3', photoz_wspec=False)
supplement_sv3_merged_file_with_y3(IAN_BGS_SV3_MERGED_FULL_FILE, IAN_BGS_Y3_MERGED_FILE_LOA, IAN_BGS_SV3_MERGED_FULL_FILE)


# In[ ]:


# Make a variant of Y3 Loa cut to SV3 regions, for the Fiber Incompleteness Study

# First replace photo-z column with one that has no spec-z 'contamination' 
table = Table.read(IAN_BGS_Y3_MERGED_FILE_LOA, format='fits')
table = add_photz_columns(table, IAN_PHOT_Z_FILE_NOSPEC)

# Remove galaxies that are far away from the SV3 regions
keep = table['NTILE_MINE_SV3'] >= 1 # Not going to 10+ here because we want to keep nearby galaxies for the NN catalog

print(f"Y3 Loa table cut from {len(table)} to {keep.sum()} rows for like-SV3 version")
table  = table[keep]

table.write(IAN_BGS_Y3_MERGED_FILE_LOA_SV3CUT, format='fits', overwrite=True)


# In[ ]:


# Update teh Y3 SV3 Cut to have an extra redshift columns with the redshifts from SV3
sv3_table = Table.read(IAN_BGS_SV3_MERGED_FILE, format='fits')
sv3_table.rename_column('Z', 'Z_SV3')

# Now add the SV3 redshifts to the Y3 table
y3_table = Table.read(IAN_BGS_Y3_MERGED_FILE_LOA_SV3CUT, format='fits')
if 'Z_SV3' in y3_table.columns:
    y3_table.remove_columns(['Z_SV3'])

# Use match_coorddinate_sky to match the two tables

y3c = coord.SkyCoord(ra=y3_table['RA']*u.degree, dec=y3_table['DEC']*u.degree, frame='icrs')
sv3c = coord.SkyCoord(ra=sv3_table['RA']*u.degree, dec=sv3_table['DEC']*u.degree, frame='icrs')
idx, d2d, d3d = coord.match_coordinates_sky(y3c, sv3c, nthneighbor=1)
close = 1 * u.arcsec
matched = d2d < close
print(f"Y3-SV3-Cut table matched {matched.sum()} of {len(y3_table)} rows")

# Add the SV3 redshift to the Y3 table
y3_table['Z_SV3'] = np.nan
y3_table['Z_SV3'][matched] = sv3_table['Z_SV3'][idx[matched]]

# Replace values > 999 with NaN
y3_table['Z_SV3'][y3_table['Z_SV3'] > 50] = np.nan

# Save the table
y3_table.write(IAN_BGS_Y3_MERGED_FILE_LOA_SV3CUT, format='fits', overwrite=True)


# In[ ]:




