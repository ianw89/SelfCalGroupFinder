#!/bin/bash
#Basic Usage: kdGroupFinder inputfile zmin zmax frac_area [fluxlim] [color] [wcenvalues 1-6] [Bsat_values 1-4] [wchi_values 1-4] > out

# If fluxlim = 1 I don't think zmin/zmax/frac_area matter at all
zmin=0 
zmax=0.8042979 
frac_area=0.35876178702 # TODO Doesn't matter if in flux-limited mode... right? I already multiplied in.
fluxlim=1 # is flux limited sample

color=1 # 0 is off but I don't think code works with colors off properly , so setting omega0 below to 0 turns off.
omegaL_sf=13.1
sigma_sf=2.42
omegaL_q=12.9
sigma_q=4.84
omega0_sf=0 #17.4    0 makes it effectively not do anything
omega0_q=0 #2.67    0 makes it effectively not do anything

beta0q=1 #-0.92    1 0 1 0 turns this off
betaLq=0  #10.25
beta0sf=1  #12.993
betaLsf=0  #-8.04

omega_chi_0_sf=2.68  # can just omit these 4 parameters to turn off
omega_chi_0_q=1.10
omega_chi_L_sf=2.23
omega_chi_L_q=0.48

# For reference only
#./kdGroupFinder_omp sdss_fluxlim_v1.0.dat $zmin $zmax $frac_area $fluxlim $color $omegaL_sf $sigma_sf $omegaL_q $sigma_q $omega0_sf $omega0_q $beta0q $betaLq $beta0sf $betaLsf $omega_chi_0_sf $omega_chi_0_q $omega_chi_L_sf $omega_chi_L_q > run_all_off_1.out

# ALL MXXL GALAXIES
name="/Volumes/Seagate Backup Plus Drive/galaxy-groups-data/mxxl_3pass_all"
if python3 hdf5_to_dat.py "/Volumes/Seagate Backup Plus Drive/galaxy-groups-data/weights_3pass.hdf5" "${name}" ; then
    bin/kdGroupFinder_omp "${name}.dat" $zmin $zmax $frac_area $fluxlim $color $omegaL_sf $sigma_sf $omegaL_q $sigma_q $omega0_sf $omega0_q $beta0q $betaLq $beta0sf $betaLsf > "${name}.out"
else
    echo "HDF5 to DAT conversion failed"
fi

# MXXL GALAXIES THAT HAD A FIBER ASSIGNED
#name="/Volumes/Seagate Backup Plus Drive/galaxy-groups-data/mxxl_3pass_fiberonly"
#bin/kdGroupFinder_omp "${name}.dat" $zmin $zmax $frac_area $fluxlim $color $omegaL_sf $sigma_sf $omegaL_q $sigma_q $omega0_sf $omega0_q $beta0q $betaLq $beta0sf $betaLsf > "${name}.out"
