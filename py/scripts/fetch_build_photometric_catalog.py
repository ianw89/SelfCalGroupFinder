import subprocess
from dataloc import *
import os
import numpy as np
import astropy.table as t
import sys


def prepare_photo_vac(reduction):
    
    #################################
    # Setting up paths and variables #
    #################################

    if reduction == "fuji":
        root = FUJI_PHOTO_VAC_ROOT
        todir = BGS_FUJI_FOLDER
        combined_fname = BGS_SV3_COMBINED_PHOTOMETRIC_CATALOG
        filelist = {
            f"{root}observed-targets/targetphot-{reduction}.fits": f"{todir}targetphot-{reduction}.fits",
            f"{root}potential-targets/targetphot-potential-fuji.fits": f"{todir}targetphot-potential-fuji.fits"
        }

    elif reduction == "iron":
        root = IRON_PHOTO_VAC_ROOT
        todir = BGS_Y1_FOLDER
        combined_fname = BGS_Y1_COMBINED_PHOTOMETRIC_CATALOG
        filelist = {
            f"{root}observed-targets/targetphot-{reduction}.fits": f"{todir}targetphot-{reduction}.fits",
        }
        endings = np.arange(0, 48)
        endings = np.delete(endings, [1, 32, 34, 36, 37, 40, 41, 44, 45, 46])
        for e in endings:
            filelist[f"{root}potential-targets/targetphot-potential-nside2-hp{e:02d}-main-{reduction}.fits"] = f"{todir}targetphot-potential-nside2-hp{e:02d}-main-{reduction}.fits"
        #for i in [1, 2, 3]:
        #    filelist[f"{root}potential-targets/targetphot-potential-sv{i}-{reduction}.fits"] = f"{todir}targetphot-potential-sv{i}-{reduction}.fits"
    
    elif reduction == "loa":
        root = LOA_PHOTO_VAC_ROOT
        todir = BGS_Y3_FOLDER_LOA
        combined_fname = BGS_Y3_COMBINED_PHOTOMETRIC_CATALOG
        filelist = {
            f"{root}observed-targets/targetphot-nside1-hp00-main-bright-loa.fits": f"{todir}targetphot-nside1-hp00-main-bright-loa.fits",
            f"{root}observed-targets/targetphot-nside1-hp01-main-bright-loa.fits": f"{todir}targetphot-nside1-hp01-main-bright-loa.fits",
            f"{root}observed-targets/targetphot-nside1-hp02-main-bright-loa.fits": f"{todir}targetphot-nside1-hp02-main-bright-loa.fits",
            f"{root}observed-targets/targetphot-nside1-hp03-main-bright-loa.fits": f"{todir}targetphot-nside1-hp03-main-bright-loa.fits",
            f"{root}observed-targets/targetphot-nside1-hp04-main-bright-loa.fits": f"{todir}targetphot-nside1-hp04-main-bright-loa.fits",
            f"{root}observed-targets/targetphot-nside1-hp05-main-bright-loa.fits": f"{todir}targetphot-nside1-hp05-main-bright-loa.fits",
            f"{root}observed-targets/targetphot-nside1-hp06-main-bright-loa.fits": f"{todir}targetphot-nside1-hp06-main-bright-loa.fits",
            f"{root}observed-targets/targetphot-nside1-hp07-main-bright-loa.fits": f"{todir}targetphot-nside1-hp07-main-bright-loa.fits",
            f"{root}observed-targets/targetphot-nside1-hp08-main-bright-loa.fits": f"{todir}targetphot-nside1-hp08-main-bright-loa.fits",
            f"{root}observed-targets/targetphot-nside1-hp09-main-bright-loa.fits": f"{todir}targetphot-nside1-hp09-main-bright-loa.fits",
            f"{root}observed-targets/targetphot-nside1-hp10-main-bright-loa.fits": f"{todir}targetphot-nside1-hp10-main-bright-loa.fits",
            f"{root}observed-targets/targetphot-nside1-hp11-main-bright-loa.fits": f"{todir}targetphot-nside1-hp11-main-bright-loa.fits",    
        }
        endings = np.arange(0, 48)
        endings = np.delete(endings, [1, 3, 20, 23, 28, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45, 46])
        for e in endings:
            filelist[f"{root}potential-targets/targetphot-potential-nside2-hp{e:02d}-main-bright-loa.fits"] = f"{todir}targetphot-potential-nside2-hp{e:02d}-main-bright-loa.fits"
    else:
        print("Invalid reduction. Try 'iron', 'fuji', or 'loa'")
        exit(1)

    # subprocess.run(f"ssh -fMNS bgconn -o ControlPersist=yes ianw89@perlmutter.nersc.gov", shell=True, check=True)

    #################################
    # Fetching catalogs #
    #################################

    if not ON_NERSC:
        for f in filelist:
            frompath = f
            topath = filelist[f]
            if not os.path.exists(topath):
                print(f"Fetching {frompath}")
                try:
                    subprocess.run(f"scp -o ControlPath=bgconn ianw89@perlmutter.nersc.gov:{frompath} {topath}", shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error fetching {frompath}")
                    print(e)
                    exit(1)
            else:
                print(f"File already exists: {topath}")
    else:
        print("On NERSC, so building here instead of copying.")
        
    """
    ##### Get Observed Photometric Catalog #####
    fname = f"targetphot-{reduction}.fits"
    topath = todir + fname
    paths.append(topath)
    if not os.path.exists(topath):
        print(f"Fetching BGS {reduction} Observed photometric catalog")
        try:
            # This one is supposed to contain all the surveys together (main, sv1, sv2, sv3...)
            subprocess.run(f"scp -o ControlPath=bgconn ianw89@dtn01.nersc.gov:{observed_dir}{fname} {topath}", shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error fetching {fname}")
            print(e)
            exit(1)

    ##### Get Potential Photometric Catalogs #####
    if reduction == 'fuji':
        fname = 'targetphot-potential-fuji.fits'
        topath = todir + fname
        paths.append(topath)
        if not os.path.exists(topath):
            try:
                subprocess.run(f"scp -o ControlPath=bgconn ianw89@dtn01.nersc.gov:{potential_dir}{fname} {topath}", shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error fetching {fname}")
                print(e)
                exit(1)
    else: # For bigger reductions its broken up into many smaller files

        # skipping special and cmx here
        sv_num = [1,2,3]
        for i in sv_num:
            fname = f'targetphot-potential-sv{i}-{reduction}.fits'
            topath = todir + fname
            paths.append(topath)
            if not os.path.exists(topath):
                print(f"Fetching {fname}")

                subprocess.run(f"scp -o ControlPath=bgconn ianw89@perlmutter.nersc.gov:{potential_dir}{fname} {topath}", shell=True, check=True)

        for ending in endings:
            fname = f"targetphot-potential-nside2-hp{ending:02d}-main-{reduction}.fits"
            frompath = "ianw89@perlmutter.nersc.gov:" + potential_dir + fname
            topath = todir + fname
            paths.append(topath)

            if not os.path.exists(topath):
                print(f"Fetching BGS {reduction} potential targets: {frompath}")
                try:
                    subprocess.run(f"scp -o ControlPath=bgconn {frompath} {topath}", shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error fetching {frompath}")
                    print(e)
                    exit(1)

    #subprocess.run(f"ssh -S bgconn -o exit", shell=True, check=True)
    """

    #################################
    # Combine and reduce catalogs #
    #################################
    print("Combining BGS targets files")

    cols = ['TARGETID', 'SURVEY', 'PROGRAM', 'MORPHTYPE', 'FRACFLUX_G', 'FRACFLUX_R', 'FRACFLUX_Z', 
        'SHAPE_E1', 'SHAPE_E2', 'SERSIC', 'SHAPE_R_IVAR', 'SHAPE_E1_IVAR', 'SHAPE_E2_IVAR', 'SERSIC_IVAR']
    
    main_tbl = None
    lst = filelist.values() if not ON_NERSC else filelist.keys()
    for f in lst:
        print("Processing ", f)
        tbl = t.Table.read(f)
        tbl.keep_columns(cols)
        tbl = tbl[tbl['PROGRAM'] == 'bright']
        if main_tbl is None:
            main_tbl = tbl
        else:
            main_tbl = t.vstack([main_tbl, tbl], join_type='exact')
            del(tbl)

    print(f"Writing combined BGS targets file with {len(main_tbl)} entries")
    main_tbl.write(combined_fname, overwrite=True)
    
    return main_tbl




if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_build_photometric_catalog.py <reduction>")
        exit(1)
    reduction = sys.argv[1]

    prepare_photo_vac(reduction)
