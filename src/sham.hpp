#pragma once

// Min/Max for SHAMing. 
// Note that when populating mock we have a tighter range.
#define HALO_MAX 1.0E+16
#define HALO_MIN 1.0E+8

float density2host_halo_zbins3(float z, double vmax);
float density2host_halo(float galaxy_density);
float halo_abundance(float m);
float halo_abundance2(float m);
float func_match_nhost(float mass, float galaxy_density);
