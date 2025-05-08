import subprocess

from dataloc import *
from astropy.table import Table

def export_extra_columns_for_clustering(merged_file, survey, verspec, ver):
    tbl = Table.read(merged_file, format='fits') 
    tbl.keep_columns(['TARGETID', 'QUIESCENT', 'QUIESCENT_SIMPLE', 'ABSMAG01_SDSS_R'])
    tbl.rename_column('ABSMAG01_SDSS_R', 'ABSMAG_R')
    frompath = BGS_Y1_FOLDER + "BGS_BRIGHT_extra.dat.fits"
    tbl.write(frompath, format='fits', overwrite=True)

    if not ON_NERSC:
        scp_command = f"scp -o ControlPath=bgconn {frompath} ianw89@perlmutter.nersc.gov:/global/cfs/cdirs/desi/users/ianw89/clustering/{survey}/LSS/{verspec}/LSScats/{ver}/"
        subprocess.run(scp_command, shell=True, check=True)
    else:
        scp_command = f"cp {frompath} /global/cfs/cdirs/desi/users/ianw89/clustering/{survey}/LSS/{verspec}/LSScats/{ver}/"
        subprocess.run(scp_command, shell=True, check=True)


def main():

    if not ON_NERSC:
        subprocess.run(f"ssh -fMNS bgconn -o ControlPersist=yes ianw89@perlmutter.nersc.gov", shell=True, check=True)

    export_extra_columns_for_clustering(IAN_BGS_Y1_MERGED_FILE, "Y1", "iron", "v1.5pip")



if __name__ == "__main__":
    main()