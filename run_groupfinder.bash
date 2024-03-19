#!/bin/bash
#Basic Usage: kdGroupFinder inputfile zmin zmax frac_area [fluxlim] [color] [wcenvalues 1-6] [Bsat_values 1-4] [wchi_values 1-4] > out

# If fluxlim = 1 then zmin/zmax are ignored. frac_area still used in some places
#zmin=0 
#zmax=0.8042979 
#frac_area=0.35876178702
fluxlim=1

color=1 # 0 is off but I don't think code works with colors off properly , so setting omega0 below to 0 turns off.
omegaL_sf=13.1
sigma_sf=2.42
omegaL_q=12.9
sigma_q=4.84
omega0_sf=17.4    #0 makes it effectively not do anything
omega0_q=2.67    #0 makes it effectively not do anything

#1 0 1 0 turns this off
beta0q=-0.92    
betaLq=10.25
beta0sf=12.993
betaLsf=-8.04

# can just omit these 4 parameters to turn off
omega_chi_0_sf=2.68  
omega_chi_0_q=1.10
omega_chi_L_sf=2.23
omega_chi_L_q=0.48

# Ian Mac
#ROOT_FOLDER="/Volumes/Seagate Backup Plus Drive/galaxy-groups-data/"
#MXXL_FILES_FOLDER="bin/"

# Ian WSL
#ROOT_FOLDER="bin/"
#MXXL_FILES_FOLDER=$ROOT_FOLDER
#UCHUU_FILES_FOLDER=$ROOT_FOLDER

# Sirocco
ROOT_FOLDER="bin/"
MXXL_FILES_FOLDER="/export/sirocco2/tinker/DESI/MXXL_MOCKS/"
UCHUU_FILES_FOLDER="/export/sirocco2/tinker/DESI/UCHUU_MOCKS/"
PYTHON="/home/users/imw2293/.conda/envs/ian-conda311/bin/python3"

PYTHON_PROCESSING=true # whether to do python processing to create .DAT files before groupfinding
GROUP_FINDING=true # whether to do the group finding

# MXXL
run_all_alt=false
run_all=false
run_all20=false
run_fiber_only=false
run_fiber_only20=false
run_nn_kd=false 
run_nn_kd20=false
run_fancy=false
run_fancy20=false
run_simple=false
run_simple20=false
# UCHUU
run_uchuu_all=false
# DESI BGS
run_bgs_fiberonly_1passok=false
run_bgs_fiberonly=false 
run_bgs_simple=false
run_bgs_simple_sdsslike=false
run_bgs_simple_c=true
run_bgs_simple_sdsslike_c=true

function process_and_group_find () {
    run_groupfinder=$GROUP_FINDING
    name=$1
    colors_on=$7
    if $PYTHON_PROCESSING ; then
        echo "Calling python pre-processor on ${name}"
        rm "${name}_old.dat" "${name}_old_galprops.dat" "${name}_old.out" "${name}_meta_old.out" 2>bin/null
        mv "${name}.dat" "${name}_old.dat" 2>bin/null
        mv "${name}_meta.dat" "${name}_meta_old.dat" 2>bin/null
        mv "${name}_galprops.dat" "${name}_old_galprops.dat" 2>bin/null
        mv "${name}.out" "${name}_old.out" 2>bin/null
        if $PYTHON $6 $2 $3 $4 $5 "${name}" $colors_on; then
            echo "Conversion to DAT successful"
        else
            echo "Conversion to DAT failed"
            run_groupfinder=false
        fi
    fi
    if $run_groupfinder; then
        # read the file "${name}_meta.dat" into variables zmin zmax frac_area
        zmin=$(awk 'NR==1 {print $1}' "${name}_meta.dat")
        zmax=$(awk 'NR==1 {print $2}' "${name}_meta.dat")
        frac_area=$(awk 'NR==1 {print $3}' "${name}_meta.dat")
        echo $zmin $zmax $frac_area

        if [ "$colors_on" -eq 1 ]; then
            echo "Running kdGroupFinder_omp with colors on"
            bin/kdGroupFinder_omp "${name}.dat" $zmin $zmax $frac_area $fluxlim $color $omegaL_sf $sigma_sf $omegaL_q $sigma_q $omega0_sf $omega0_q $beta0q $betaLq $beta0sf $betaLsf > "${name}.out"
        else
            echo "Running kdGroupFinder_omp with colors off"
            bin/kdGroupFinder_omp "${name}.dat" $zmin $zmax $frac_area $fluxlim $color $omegaL_sf $sigma_sf $omegaL_q $sigma_q 0 0 1 0 1 0 > "${name}.out"
        fi
    fi

}

function process_and_group_find_mxxl () {
    process_and_group_find $1 $2 $3 $4 "${MXXL_FILES_FOLDER}weights_3pass.hdf5" desi/hdf5_to_dat.py $5
}

function process_and_group_find_uchuu () {
    process_and_group_find $1 $2 $3 $4 "${UCHUU_FILES_FOLDER}BGS_LC_Uchuu.fits" desi/uchuu_to_dat.py $5
}

function process_and_group_find_BGS () {
    process_and_group_find $1 $2 $3 $4 "${ROOT_FOLDER}BGS_ANY_full.dat.fits" desi/desi_fits_to_dat.py $5
}


if [ "$run_all_alt" = true ] ; then
    process_and_group_find_mxxl "${ROOT_FOLDER}mxxl_3pass_all_alt" 6 19.5 20.0 0
fi

if [ "$run_all" = true ] ; then
    process_and_group_find_mxxl "${ROOT_FOLDER}mxxl_3pass_all" 1 19.5 20.0 0
fi

if [ "$run_all20" = true ] ; then
    process_and_group_find_mxxl "${ROOT_FOLDER}mxxl_3pass_all20" 1 20.0 20.0 0
fi

if [ "$run_fiber_only" = true ] ; then
    process_and_group_find_mxxl "${ROOT_FOLDER}mxxl_3pass_fiberonly" 2 19.5 20.0 0
fi

if [ "$run_fiber_only20" = true ] ; then
    process_and_group_find_mxxl "${ROOT_FOLDER}mxxl_3pass_fiberonly20" 2 20.0 20.0 0
fi 

if [ "$run_nn_kd" = true ] ; then
    process_and_group_find_mxxl "${ROOT_FOLDER}mxxl_3pass_nn_kd" 3 19.5 20.0 0
fi

if [ "$run_nn_kd20" = true ] ; then
    process_and_group_find_mxxl "${ROOT_FOLDER}mxxl_3pass_nn_kd20" 3 20.0 20.0 0
fi

if [ "$run_fancy" = true ] ; then
    process_and_group_find_mxxl "${ROOT_FOLDER}mxxl_3pass_fancy_6" 4 19.5 20.0 0
fi

if [ "$run_fancy20" = true ] ; then
    process_and_group_find_mxxl "${ROOT_FOLDER}mxxl_3pass_fancy_6_20" 4 20.0 20.0 0
fi

if [ "$run_simple" = true ] ; then
    process_and_group_find_mxxl "${ROOT_FOLDER}mxxl_3pass_simple_2" 5 19.5 20.0 0
fi

if [ "$run_simple20" = true ] ; then
    process_and_group_find_mxxl "${ROOT_FOLDER}mxxl_3pass_simple_2_20" 5 20.0 20.0 0
fi



if [ "$run_uchuu_all" = true ] ; then
    process_and_group_find_uchuu "${ROOT_FOLDER}uchuu_all" 1 19.5 20.0 0
fi


if [ "$run_bgs_fiberonly_1passok" = true ] ; then
    process_and_group_find_BGS "${ROOT_FOLDER}BGS_fiberonly_1passok_1" 1 19.5 22.0 0
fi

if [ "$run_bgs_fiberonly" = true ] ; then
    process_and_group_find_BGS "${ROOT_FOLDER}BGS_fiberonly_1" 2 19.5 22.0 0
fi

if [ "$run_bgs_simple" = true ] ; then
    process_and_group_find_BGS "${ROOT_FOLDER}BGS_simple_2" 5 19.5 22.0 0
fi

if [ "$run_bgs_simple_sdsslike" = true ] ; then
    process_and_group_find_BGS "${ROOT_FOLDER}BGS_simple_2_sdsslike" 5 17.7 22.0 0
fi

if [ "$run_bgs_simple_c" = true ] ; then
    process_and_group_find_BGS "${ROOT_FOLDER}BGS_simple_2_c" 5 19.5 22.0 1
fi

if [ "$run_bgs_simple_sdsslike_c" = true ] ; then
    process_and_group_find_BGS "${ROOT_FOLDER}BGS_simple_2_sdsslike_c" 5 17.7 22.0 1
fi
