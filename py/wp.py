import subprocess
import os
import sys

if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from dataloc import *

# Define the path to the Corrfunc binary
corrfunc_path = "/export/sirocco1/tinker/src/Corrfunc/bin/"
if not os.path.exists(os.path.join(corrfunc_path, "wp")):
    corrfunc_path = "/mount/sirocco1/tinker/src/Corrfunc/bin/"

def run_corrfunc(cwd):

    processes = []
    mass_range = range(17, 22)

    # First part: Using CORRFUNC
    for m in mass_range:
        for col in ["red", "blue"]:
            cmd = f"{corrfunc_path}/wp 250 mock_{col}_M{m}.dat a {WP_RADIAL_BINS_FILE} 40 10 > wp_mock_{col}_M{m}.dat"
            processes.append(subprocess.Popen(cmd, cwd=OUTPUT_FOLDER, shell=True))

    for p in processes:
        p.wait()
    processes.clear()

    """

    # Repeat the pattern for the other parts
    # Second part: Mocks
    for m in mass_range:
        for col in ["red", "blue"]:
            cmd = f"wp_covar 0.1 10 10 250 0 250 0 sample_{col}_{m}.dat a 0 1 1 3 5 > wp_sample_{col}_M{m}.dat"
            processes.append(run_command_parallel(cmd))

    for p in processes:
        p.wait()
    processes.clear()

    # Third part: Populated from the group finder
    for m in mass_range:
        for col in ["red", "blue"]:
            cmd = f"wp_covar 0.1 10 10 250 0 250 1 mock_{col}_M{m}.dat a 0 1 1 1 5 > wp_mock_{col}_M{m}.dat"
            processes.append(run_command_parallel(cmd))

    for p in processes:
        p.wait()
    processes.clear()

    # Fourth part: Original mock - using CORRFUNC
    for m in mass_range:
        for col in ["red", "blue"]:
            cmd = f"{corrfunc}/wp 250 sample_{col}_{m}.dat a wp_rbins.dat 40 10 > wp_sample_{col}_{m}.dat"
            processes.append(run_command_parallel(cmd))

    for p in processes:
        p.wait()
    processes.clear()


    """