import sys
if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from pyutils import *
from bgs_helpers import *


build_y3_randoms_files()

build_y1_randoms_files()

build_sv3_full_randoms_files()

#build_sv3_clustering_randoms_files() # BUG broken

#build_y3_likesv3_clustering_randoms_files()