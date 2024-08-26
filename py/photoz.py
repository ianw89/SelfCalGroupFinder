import os
import pickle
import re
import sys
from urllib.parse import urljoin
import numpy as np
import requests
import time
import astropy.units as u
import astropy.coordinates as coord
from astropy.table import Table,join,vstack,unique
from astropy.utils.exceptions import AstropyWarning
import warnings
from pathlib import Path
import asyncio
import aiohttp

if './SelfCalGroupFinder/py/' not in sys.path:
    sys.path.append('./SelfCalGroupFinder/py/')
from dataloc import *

# PHOTO-Z MERGING UTILS

START = 0
END = 1436
FILE_COUNT_LIMIT = 25
TASK_LIMIT = 10

url_base_pz = 'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/sweep/10.1-photo-z/'
url_base_main = 'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/sweep/10.1/'

def get_photoz_file_lists():
    # Download HTML from url_base_pz and then search for <a href="*.fits" and download those files

    if os.path.isfile(BGS_IMAGES_FOLDER + 'fits_links_pz.pkl') and os.path.isfile(BGS_IMAGES_FOLDER + 'fits_links_main.pkl'):
        fits_links_pz = pickle.load(open(BGS_IMAGES_FOLDER + 'fits_links_pz.pkl', 'rb'))
        fits_links_main = pickle.load(open(BGS_IMAGES_FOLDER + 'fits_links_main.pkl', 'rb'))
        return fits_links_pz, fits_links_main

    # Step 1: Download the HTML content
    response = requests.get(url_base_pz)
    response.raise_for_status()  # Check that the request was successful
    html_content = response.text

    response = requests.get(url_base_main)
    response.raise_for_status()  # Check that the request was successful
    html_content_main = response.text

    # Step 2: Use regex to find all <a> tags with href ending in .fits
    fits_links_pz   = re.findall(r'<a\s+(?:[^>]*?\s+)?href="([^"]*\.fits)"', html_content)
    fits_links_main = re.findall(r'<a\s+(?:[^>]*?\s+)?href="([^"]*\.fits)"', html_content_main) 

    # Step 3: Save the lists to disk
    pickle.dump(fits_links_pz, open(BGS_IMAGES_FOLDER + 'fits_links_pz.pkl', 'wb'))
    pickle.dump(fits_links_main, open(BGS_IMAGES_FOLDER + 'fits_links_main.pkl', 'wb'))

    return fits_links_pz, fits_links_main

async def download_file(session, url, filename, i):
    success = False

    while not success:
        try:
            response = await session.get(url, timeout=1200) 
            response.raise_for_status()
            print(f"Connected #{i}: {filename}", flush=True)
            content = await response.read()
            print(f"Read #{i}: {filename}", flush=True)
            # TODO write async
            with open(os.path.join(BGS_IMAGES_FOLDER, filename), 'wb') as file:
                file.write(content)
            print(f"Saved #{i}: {filename}", flush=True)
            success = True
            response.close()
        except Exception as e:
            print(f"Error downloading #{i} {filename}. Status: {response.status}. Exception {e}. Retry in 5s", flush=True)
            await asyncio.sleep(5)
    
    return success

def too_many_files():
    filecount = len([name for name in os.listdir(BGS_IMAGES_FOLDER) if os.path.isfile(os.path.join(BGS_IMAGES_FOLDER, name))])
    return filecount > FILE_COUNT_LIMIT

async def download_photoz_files_async():
    """
    Download the photo-z sweep files from the web and save them to disk. Don't let too many files build up. 
    Block until other program has processed them and then deleted.
    """
    fits_links_pz, fits_links_main = get_photoz_file_lists()

    async with aiohttp.ClientSession() as session:

        for i in range(START,END):
            # Make filenames, etc
            link_pz = fits_links_pz[i]
            fits_pz_url = urljoin(url_base_pz, link_pz)
            fits_pz_filename = os.path.basename(fits_pz_url)
            link_main = fits_links_main[i]
            fits_main_url = urljoin(url_base_main, link_main)
            fits_main_filename = os.path.basename(fits_main_url)
            # Ensure Bricks are matched
            assert fits_pz_filename[6:20] == fits_main_filename[6:20] 

            check_for_block = False
            

            # PHOTO Z SWEEP FILE
            f1 = Path(BGS_IMAGES_FOLDER + fits_pz_filename)
            file_done = os.path.isfile(f1) and f1.stat().st_size != 0
            if file_done:
                print(f"File #{i} {fits_pz_filename} already exists and is not empty. Skipping download.", flush=True)
            else:
                print(f"Requesting #{i} {fits_pz_filename}", flush=True)
                check_for_block = True
                asyncio.create_task(download_file(session, fits_pz_url, fits_pz_filename, i))

            # GENERAL SWEEP FILE    
            f2 = Path(BGS_IMAGES_FOLDER + fits_main_filename)
            file_done = os.path.isfile(f2) and f2.stat().st_size != 0
            if file_done:
                print(f"File #{i} {fits_main_filename} already exists and is not empty. Skipping download.", flush=True)
            else:
                print(f"Requesting #{i} {fits_main_filename}", flush=True)
                check_for_block = True
                asyncio.create_task(download_file(session, fits_main_url, fits_main_filename, i))

            # Block before continuing loop if too many files are present
            if check_for_block:
                ongoing = [task for task in asyncio.all_tasks() if not task.done()]
                if len(ongoing) > TASK_LIMIT:
                    # remove the main task, await the rest
                    ongoing.remove(asyncio.current_task())
                    await asyncio.gather(*ongoing)
                    print("Finished batch of tasks. Continuing...", flush=True)

                while too_many_files():
                    print(f"Waiting for more files to be processed...", flush=True)
                    await asyncio.sleep(5)
                
        ongoing = [task for task in asyncio.all_tasks() if not task.done()]
        # remove the main task, await the rest
        ongoing.remove(asyncio.current_task())
        await asyncio.gather(*ongoing)
        print("Finished all tasks.", flush=True)


async def process_photoz_files():

    # Read in the DESI TARGETS table
    desi_targets_table = pickle.load(open(IAN_PHOT_Z_FILE, 'rb'))
    desi_coords = coord.SkyCoord(ra=desi_targets_table['RA'].to_numpy()*u.degree, dec=desi_targets_table['DEC'].to_numpy()*u.degree, frame='icrs')
    fits_links_pz, fits_links_main = get_photoz_file_lists()

    #for i in range(len(fits_links_pz)):
    for i in range(START,END):

        print(f"Start processing brick #{i}", flush=True)

        try:
            link_pz = fits_links_pz[i]
            fits_pz_url = urljoin(url_base_pz, link_pz)
            fits_pz_filename = os.path.basename(fits_pz_url)

            link_main = fits_links_main[i]
            fits_main_url = urljoin(url_base_main, link_main)
            fits_main_filename = os.path.basename(fits_main_url)

            f1 = Path(BGS_IMAGES_FOLDER + fits_pz_filename)
            f2 = Path(BGS_IMAGES_FOLDER + fits_main_filename)
            file_ready = os.path.isfile(BGS_IMAGES_FOLDER + fits_pz_filename) and os.path.isfile(BGS_IMAGES_FOLDER + fits_main_filename) and f1.stat().st_size > 0 and f2.stat().st_size > 0
            while not file_ready:
                print(f"Waiting for {fits_main_filename}... ", end='\r', flush=True)
                time.sleep(5)
                file_ready = os.path.isfile(BGS_IMAGES_FOLDER + fits_pz_filename) and os.path.isfile(BGS_IMAGES_FOLDER + fits_main_filename) and f1.stat().st_size > 0 and f2.stat().st_size > 0
                if file_ready:
                    print(f"File {fits_main_filename} is ready. Continuing...", flush=True)

            # TODO speedup reading...
            table_pz = Table.read(BGS_IMAGES_FOLDER + fits_pz_filename, format='fits')
            table_pz.keep_columns(['RELEASE', 'BRICKID', 'OBJID', 'Z_PHOT_MEDIAN', 'Z_PHOT_MEDIAN_I', 'Z_SPEC'])
            df = table_pz.to_pandas()
            del(table_pz)

            table_main = Table.read(BGS_IMAGES_FOLDER + fits_main_filename, format='fits')
            table_main.keep_columns(['RELEASE', 'BRICKID', 'OBJID', 'TYPE', 'RA', 'DEC', 'REF_CAT']) # SHAPE_E1, SHAPE_E2, REF_ID, SERSIC
            df_main = table_main.to_pandas()
            del(table_main)

            # join on REALEASE, BRICKID, OBJID
            df_merged = df.join(df_main.set_index(['RELEASE', 'BRICKID', 'OBJID']), on=['RELEASE', 'BRICKID', 'OBJID'], how='inner')

            if len(df) != len(df_main) or len(df) != len(df_merged):
                print(f"Lengths do not match for {fits_pz_filename} ({len(df)}) and {fits_main_filename} ({len(df_main)}). Merged: {len(df_merged)}")

            del(df)
            del(df_main)
            df = df_merged

            # use Z_SPEC if available, otherwise use Z_PHOT_MEDIAN_I, otherwise use Z_PHOT_MEDIAN
            df['Z_LEGACY_BEST'] = df['Z_SPEC']
            #print(np.isclose(df['Z_LEGACY_BEST'], -99.0).sum())

            no_z = np.isclose(df['Z_LEGACY_BEST'], -99.0)
            df.loc[no_z, 'Z_LEGACY_BEST'] = df.loc[no_z, 'Z_PHOT_MEDIAN_I']
            #print(np.isclose(df['Z_LEGACY_BEST'], -99.0).sum())

            no_z = np.isclose(df['Z_LEGACY_BEST'], -99.0)
            df.loc[no_z, 'Z_LEGACY_BEST'] = df.loc[no_z, 'Z_PHOT_MEDIAN']
            #print(np.isclose(df['Z_LEGACY_BEST'], -99.0).sum())

            # Remove rows that still have no redshift
            df = df[np.invert(np.isclose(df['Z_LEGACY_BEST'], -99.0))]
            #print(np.isclose(df['Z_LEGACY_BEST'], -99.0).sum())

            # Remove rows that are point sources (stars, generally)
            df = df[df['TYPE'] != b'PSF']

            # Drop columns that are no longer needed
            df.drop(columns=['Z_PHOT_MEDIAN', 'Z_PHOT_MEDIAN_I', 'Z_SPEC'], inplace=True)
            df.reset_index(drop=True, inplace=True)

            # At this point df only contains galaxies with a redshift, and the best redshift from the legacy catalog.
            legacy_coords = coord.SkyCoord(ra=df.RA.to_numpy()*u.degree, dec=df.DEC.to_numpy()*u.degree, frame='icrs')

            # idx is the index of the DESI target (unchanging arrays) that is closest to the each legacy source
            idx, d2d, d3d = coord.match_coordinates_sky(legacy_coords, desi_coords, nthneighbor=1, storekdtree='BGS-GALS')
            ang_distances = d2d.to(u.arcsec).value

            # if angular distance is < 3", then we consider it a match
            matched = ang_distances < 3
            print(f"Matched {matched.sum()} out of {len(matched):,}", flush=True)

            # Reduce dataframe to galaxies matched to DESI targets only
            df = df.loc[matched]
            df.reset_index(drop=True, inplace=True)

            df['TARGETID'] = desi_targets_table.iloc[idx[matched]].index.to_numpy()
            df['MATCH_DIST'] = ang_distances[matched]

            # For each DESI target with a match, we must determine the legacy source that is closest.
            results = df.groupby('TARGETID').apply(lambda x: x.loc[x['MATCH_DIST'].idxmin()])
            ids_to_set = results['TARGETID'].to_numpy()

            # Finally update the DESI targets with these photo-z values (and remember which legacy source was used)
            desi_targets_table.loc[ids_to_set,'Z_LEGACY_BEST'] = results['Z_LEGACY_BEST'].to_numpy()
            desi_targets_table.loc[ids_to_set,'RELEASE'] = results['RELEASE'].to_numpy()
            desi_targets_table.loc[ids_to_set,'BRICKID'] = results['BRICKID'].to_numpy()
            desi_targets_table.loc[ids_to_set,'OBJID'] = results['OBJID'].to_numpy()

            print(f"Writing progress to disk...", flush=True)
            pickle.dump(desi_targets_table, open(IAN_PHOT_Z_FILE, 'wb'))

            # We made it through with no errors, so we can delete the files
            os.remove(BGS_IMAGES_FOLDER + fits_pz_filename)
            os.remove(BGS_IMAGES_FOLDER + fits_main_filename)

            #if i % 10 == 0:
            percent_complete = (desi_targets_table['Z_LEGACY_BEST'] != -99.0).sum() / len(desi_targets_table)
            print(f"Done with {i+1} bricks. {percent_complete*100:.4f}% of DESI targets have a photo-z.", flush=True)
    
        except Exception as e:
            print(f"Error processing brick {i}: {e}")       
    

async def main():
    """
    To use this program, call it once with argument 1 to download the photo-z files, 
    and simultaneously call it with argument 2 from another shell to process them.

    Set START and END as appropriate.
    """
    warnings.simplefilter('ignore', category=AstropyWarning)

    if len(sys.argv) != 2:
        print("Usage: python photoz.py <1 or 2>")
        sys.exit(1)

    if sys.argv[1] == '1':
        print("DOWNLOAD MODE")
        await download_photoz_files_async()

    if sys.argv[1] == '2':
        print("PROCESS MODE")
        await process_photoz_files()



if __name__ == "__main__":
    asyncio.run(main())