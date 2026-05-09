# The Self Calibrated Group Finder

This repository is Ian Williams' Self Calibrated Group Finder, a fork of Jeremy Tinker's SelfCalGroupFinder. It is a halo-based galaxy group finder written in C++ with python code to wrap it and for analysis. 

Information on the original C Self Calibrated Group Finder this is forked can be obtained from https://www.galaxygroupfinder.net/.

## Published Group Catalogs

SDSS based group catalogs from Jeremy Tinker built using a slightly older version of this group finder can be found on https://www.galaxygroupfinder.net/

DESI BGS group catalogs built from this group finder will be available on https://data.desi.lbl.gov/doc/releases/dr1/#value-added-catalogs at some point in 2026.


## C++ Group Finder
The C++ group finder can be used standalone without any of the python code in this repository, and is the most standalone part of this project. It can be built using the Makefile which may need to be updated on your system. The only external library needed is GSL; for my environment I found it convenient to copy the GSL shared libraries into a folder on a shared disk that all compute nodes have access to. Take a look at the Makefile.

For usage of the groupfinder, build it and then just run it from the commandline with --help. For formats of the input and outout files, see the file descriptions at the project's web page; I will try and document better here soon.

The code can run for for flux limited samples (a survey) or volume-limited samples (usually a simulation) but only the flux-limited mode has been used recently and so the volume-limited version may be broken / have bugs. Please test if using for that purpose. I expect to revisit the volume-limited version in summer or fall of 2026.

The code needs a tabulated halo mass function. There are two checked into this repository, look for "hmf_t008_bolshoi.dat" and "hmf_t08_p18.dat" under the py/parameters folder. They both use the Tinker et al (2008) halo mass function, but slightly different cosmologies (Bolshoi Planck cosmology  vs Planck 2018), which barely affects things anyway. For format of that file is just two columns, the first is halo mass [Msol / h] and the second is the number density dn/dlogM [number / (Mpc/h)^3].

The C++ code can also builds mock using the implied galaxy-halo connection it found by running the group finder. I use this to calibrate group finding parameters in an MCMC loop. You can use it with the --popmock option though a mock file must be provided. See my usage pattern in the python code.


## Python Wrapper

The python wrapper is mostly used by me for producing a DESI BGS Group Catalog. It is still under development, and not really inteded to be shared. That said, if you want to try using my code, to setup a conda-based python environment that is known to work with this repository, do the following:

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
