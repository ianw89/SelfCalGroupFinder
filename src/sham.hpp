#pragma once

// Min/Max for abundance matching.
// The AM goes from the most massive halos to the least using the number density in the halo mass function file.
// The exact choice of upper bound doesn't matter much as long as its above the biggest halos in the HMF.
// The HMF will have the density extremely low there so it won't get used in practice.
// The lower bound just needs to be low enough that the that the least massive halo you will assign to the galaxy samlpe
// is above it; otherwise it doesn't matter.
//
// Note that when populating mock we have a tighter range; this isn't used for that.
#define HALO_MAX 1.0E+16
#define HALO_MIN 1.0E+8

float density2host_halo_zbins3(float z, double vmax);
float density2host_halo(float galaxy_density);
float halo_abundance(float m);
float halo_abundance2(float m);
float func_match_nhost(float mass, float galaxy_density);
