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

ROOT_FOLDER="/Volumes/Seagate Backup Plus Drive/galaxy-groups-data/"
#ROOT_FOLDER="D:\\galaxy-groups-data\\"

function process_and_group_find () {
    name=$1
    rm "${name}_old.dat" "${name}_old_galprops.dat" "${name}_old.out" 2>bin/null
    mv "${name}.dat" "${name}_old.dat" 2>bin/null
    mv "${name}_galprops.dat" "${name}_old_galprops.dat" 2>bin/null
    mv "${name}.out" "${name}_old.out" 2>bin/null
    if python3 hdf5_to_dat.py $2 $3 $4 "${ROOT_FOLDER}weights_3pass.hdf5" "${name}" ; then
        bin/kdGroupFinder_omp "${name}.dat" $zmin $zmax $frac_area $fluxlim $color $omegaL_sf $sigma_sf $omegaL_q $sigma_q $omega0_sf $omega0_q $beta0q $betaLq $beta0sf $betaLsf > "${name}.out"
    else
        echo "HDF5 to DAT conversion failed"
    fi
}

# MXXL
run_all=false
run_all20=false
run_fiber_only=false
run_fiber_only20=false
run_nn_kd=false 
run_nn_kd20=false
run_fancy=false
run_fancy20=false
run_simple=true
run_simple20=true
run_simple_mix=false


if [ "$run_all" = true ] ; then
    process_and_group_find "${ROOT_FOLDER}mxxl_3pass_all" 1 19.5 19.5
fi

if [ "$run_all20" = true ] ; then
    process_and_group_find "${ROOT_FOLDER}mxxl_3pass_all20" 1 20.0 20.0
fi

if [ "$run_fiber_only" = true ] ; then
    process_and_group_find "${ROOT_FOLDER}mxxl_3pass_fiberonly" 2 19.5 19.5
fi

if [ "$run_fiber_only20" = true ] ; then
    process_and_group_find "${ROOT_FOLDER}mxxl_3pass_fiberonly20" 2 20.0 20.0
fi

if [ "$run_nn_kd" = true ] ; then
    process_and_group_find "${ROOT_FOLDER}mxxl_3pass_nn_kd" 3 19.5 19.5
fi

if [ "$run_nn_kd20" = true ] ; then
    process_and_group_find "${ROOT_FOLDER}mxxl_3pass_nn_kd20" 3 20.0 20.0
fi

if [ "$run_fancy" = true ] ; then
    process_and_group_find "${ROOT_FOLDER}mxxl_3pass_fancy_6" 4 19.5 19.5
fi

if [ "$run_fancy20" = true ] ; then
    process_and_group_find "${ROOT_FOLDER}mxxl_3pass_fancy_6_20" 4 20.0 20.0
fi

if [ "$run_simple" = true ] ; then
    process_and_group_find "${ROOT_FOLDER}mxxl_3pass_simple_3" 5 19.5 19.5
fi

if [ "$run_simple20" = true ] ; then
    process_and_group_find "${ROOT_FOLDER}mxxl_3pass_simple_3_20" 5 20.0 20.0
fi

if [ "$run_simple_mix" = true ] ; then
    process_and_group_find "${ROOT_FOLDER}mxxl_3pass_simple_3_mix" 5 19.5 20.0 
fi