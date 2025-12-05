import sys
import os
current_file_path = os.path.realpath(__file__)
current_directory = os.path.dirname(current_file_path)
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
from groupcatalog import *
import catalog_definitions as cat
from dataloc import *
from pyutils import *

#python3 py/scripts/chisqr_variance.py 

d = cat.bgs_y1_pzp_2_6_c2
runs = 50
results = []

for i in range(runs):
    success = d.run_group_finder(popmock=True, profile=False, silent=True, serial=False)
    if not success:
        print(f"Group finder failed for {d.name}")
        exit(1)
    d.calc_wp_for_mock()
    d.postprocess()
    r = d.chisqr()
    print(r)
    results.append(r)

