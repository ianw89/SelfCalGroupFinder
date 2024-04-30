#!/bin/bash
#Basic Usage: kdGroupFinder inputfile zmin zmax frac_area [fluxlim] [color] [wcenvalues 1-6] [Bsat_values 1-4] [wchi_values 1-4] > out
zmin=0
zmax=1
frac_area=0.179
fluxlim=1

color=1
omegaL_sf=13.1
sigma_sf=2.42
omegaL_q=12.9
sigma_q=4.84
omega0_sf=17.4    #0 makes it effectively not do anything
omega0_q=2.67    #0 makes it effectively not do anything

#1 0 1 0 turns this off
beta0q=-0.92    #Bsat = beta0 + betaLq(logLgal - 9.5)
betaLq=10.25
beta0sf=12.993
betaLsf=-8.04

# can just omit these 4 parameters to turn off
omega_chi_0_sf=2.68  
omega_chi_0_q=1.10
omega_chi_L_sf=2.23
omega_chi_L_q=0.48

# Colors/Chi values set to 0
#bin/kdGroupFinder_omp data/sdss_stripped.dat $zmin $zmax $frac_area $fluxlim $color $omegaL_sf $sigma_sf $omegaL_q $sigma_q 0 0 1 0 1 0 > bin/sdss_ian_stripped.out

# No colors or chi
bin/kdGroupFinder_omp data/sdss_fluxlim_v1.0.dat $zmin $zmax $frac_area $fluxlim $color $omegaL_sf $sigma_sf $omegaL_q $sigma_q 0 0 1 0 1 0 > bin/sdss_ian_1.out

# No chi
bin/kdGroupFinder_omp data/sdss_fluxlim_v1.0.dat $zmin $zmax $frac_area $fluxlim $color $omegaL_sf $sigma_sf $omegaL_q $sigma_q $omega0_sf $omega0_q $beta0q $betaLq $beta0sf $betaLsf > bin/sdss_ian_2.out

# All options on
bin/kdGroupFinder_omp data/sdss_fluxlim_v1.0.dat $zmin $zmax $frac_area $fluxlim $color $omegaL_sf $sigma_sf $omegaL_q $sigma_q $omega0_sf $omega0_q $beta0q $betaLq $beta0sf $betaLsf $omega_chi_0_sf $omega_chi_0_q $omega_chi_L_sf $omega_chi_L_q > bin/sdss_ian_3.out

