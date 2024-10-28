# SelfCalGroupFinder

This repository is a fork of Jeremy Tinker's SelfCalGroupFinder, a halo-based galxay group finder written in C. That project's webpage, which has not been updated since this fork began, is located at https://www.galaxygroupfinder.net/.

This ongoing fork contains various Python notebooks and a GroupCatalog class that wraps the C-based group finder and handles some pre-processing and post-processing. This is still under development. To setup a conda-based python environment that is known to work with this repository, do the following:

```
conda create -n "my-env"
conda activate my-env
conda install python=3.11 jupyter astropy scipy aiohttp pandas matplotlib h5py gsl
pip install emcee corner tqdm torch gpytorch corrfunc
```

The dataloc.py file is for updating the paths to both the code and data files.

Using the GroupCatalog class to make a BGS catalog is requires the following files to be buitl from the notebooks:
- IAN_MXXL_LOST_APP_TO_Z_FILE, not part of repo
- SDSS Vanilla v2.pickle for SDSS fill-ins, not part of repo

The C-based group finder can be built using the makefile which may need to be updated on your system. It will require an additional library that can be cloned and built from here: https://github.com/jltinker/lib

For usage of the groupfinder, build it and then just run it from the commandline with --help. For formats of the input and outout files, see the file descriptions at the project's web page, which SDSS catalogs are available.

The code expects a tabulated halo mass function is in the run directory, in a file called "halo_mass_function.dat." I have supplied one in the repo for the Bolshoi Planck cosmology using the Tinker et al (2008) halo mass function.

The code will run for 5 iterations and then output the current state of the group catalog. Five is usually a reasonable number for convergence of the satellite fraction to a couple of percent.

Attribution:
This code is developed by Jeremy Tinker and Ian Williams.
Files in the py/kcorr folder are from Sam Moore.   
