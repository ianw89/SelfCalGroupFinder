# SelfCalGroupFinder

This repository is a fork of Jeremy Tinker's SelfCalGroupFinder, a halo-based galxay group finder written in C. That project's webpage, which has not been updated since this fork began, is located at https://www.galaxygroupfinder.net/.

This ongoing fork converted the group finder to C++, and additionally contains various Python notebooks and a GroupCatalog class that wraps the C++ group finder and handles some pre-processing and post-processing. 

## C++ Group Finder
The C++ group finder can be built using the Makefile which may need to be updated on your system. The only external library needed is GSL; for my environment I found it convenient to copy the GSL shared libraries into a folder on a shared disk that all compute nodes have access to. Take a look at the Makefile.

For usage of the groupfinder, build it and then just run it from the commandline with --help. For formats of the input and outout files, see the file descriptions at the project's web page; I will try and document that better here soon.

The code can run for wither flux limited samples (a survey) or volume-limited samples (usually a sim) but only the flux-limited mode has been used recently and so the volume-limited version may be broken / have bugs. I will try and look at that soon.

The code expects a tabulated halo mass function is in the run directory, in a file called "halo_mass_function.dat." I have supplied one in the repo for the Bolshoi Planck cosmology using the Tinker et al (2008) halo mass function.

The C++ code can also populate a mock using the implied galaxy-halo connection it found by running the group finder. I use this to calibrate group finding parameters in an MCMC loop. You can use it with the --popmock option though a mock file must be provided. See my usage pattern in the python code.


## Python Wrapper

The python wrapper is mostly used by me for producing a BGS Group Catalog. It is still under development, and not really inteded to be shared yet. That said, if you want to try using my code, to setup a conda-based python environment that is known to work with this repository, do the following:

```
conda create -n "my-env"
conda activate my-env
conda install python=3.11 jupyter astropy scipy aiohttp pandas matplotlib h5py gsl 
python --version
python -m pip install --user emcee corner tqdm torch gpytorch corrfunc joblib scikit-learn seaborn
```

The dataloc.py file is for updating the paths to both the code and data files. Using the GroupCatalog class to make a BGS catalog requires a "merged file" which contains the source data: IAN_BGS_Y1_MERGED_FILE (or SV3, Y3...) as built by the build-merged-file.ipynb. Other requirements can be found in dataloc.py under "BGS DERIVED AUXILERY FILES" and they are built from in the BGS_Study.ipynb notebook from DESI LSS Catalogs. "SDSS Vanilla v2.pickle" is needed for SDSS fill-ins of targets with missing redshifts.


## Attribution:
This code is developed by Ian Williams and is a fork from the C-based SelfCalGroupFinder created by Jeremy Tinker.
Files in the py/kcorr folder are from Sam Moore.   
