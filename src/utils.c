#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "groups.h"

/*
 * Update the galaxy's derived halo properties using the mass, which must
 * already be set. Requries redshit and rco to be set as well.
*/
void update_galaxy_halo_props(struct galaxy *galaxy)
{
  galaxy->rad = pow(3 * galaxy->mass / (4. * PI * DELTA_HALO * RHO_CRIT * OMEGA_M), THIRD);
  galaxy->theta = galaxy->rad / galaxy->rco;
  galaxy->sigmav = sqrt(BIG_G * galaxy->mass / 2.0 / galaxy->rad * (1 + galaxy->redshift));
}

/* Distance-redshift relation
 */
float func_dr1(float z)
{
  return pow(OMEGA_M * (1 + z) * (1 + z) * (1 + z) + (1 - OMEGA_M), -0.5);
}

float distance_redshift(float z)
{
  float x;
  if (z <= 0)
    return 0;
  x = c_on_H0 * qromo(func_dr1, 0.0, z, midpnt);
  return x;
}

/* Angular separation between two points. Give the ra, dec in radians.
 */
float angular_separation(float a1, float d1, float a2, float d2)
{
  return atan((sqrt(cos(d2) * cos(d2) * sin(a2 - a1) * sin(a2 - a1) +
                    pow(cos(d1) * sin(d2) - sin(d1) * cos(d2) * cos(a2 - a1), 2.0))) /
              (sin(d1) * sin(d2) + cos(d1) * cos(d2) * cos(a2 - a1)));
}

float psat(struct galaxy *central, float dr, float dz, float bprob)
{
  float prob_ang, prob_rad;
  prob_ang = radial_probability_g(central, dr);
  prob_rad = compute_prob_rad(dz, central->sigmav);
  return (1 - 1 / (1 + prob_ang * prob_rad / bprob));
}

/* Probability assuming a projected NFW profile using the given galaxy's halo properties.
 */
float radial_probability_g(struct galaxy *gal, float dr)
{
  return radial_probability(gal->mass, dr, gal->rad, gal->theta);
}

/* Probability assuming a projected NFW profile
 */
float radial_probability(float mass, float dr, float rad, float ang_rad)
{
  float c, x, rs, delta, f;

  dr = dr * rad / ang_rad;

  c = 10.0 * pow(mass / 1.0E+14, -0.11);
  rs = rad / c;
  x = dr / rs;

  if (x < 1)
    f = 1 / (x * x - 1) * (1 - log((1 + sqrt(1 - x * x)) / x) / (sqrt(1 - x * x)));
  if (x == 1)
    f = 1.0 / 3.0;
  if (x > 1)
    f = 1 / (x * x - 1) * (1 - atan(sqrt(x * x - 1)) / sqrt(x * x - 1));

  delta = DELTA_HALO / 3.0 * c * c * c / (log(1 + c) - c / (1 + c));

  return 1.0 / c_on_H0 * 2 * rs * delta * f;
}
/* Computes p(delta z) assuming a gaussian as per Yang et al 2005 eq 9.
 * dz is the redshift difference between the galaxy and the group center times the speed of light
 * sigmav is the velocity dispersion of the group
 */
float compute_prob_rad(float dz, float sigmav)
{
  // presumably sigmav is comoving otherwise need (1+z_group) factor next to each one
  return exp(-dz * dz / (2 * sigmav * sigmav)) * SPEED_OF_LIGHT / (RT2PI * sigmav);
}

int search(int n, float *x, float val)
{
  int first, last, middle;
  first = 1;
   last = n ;
   middle = (first+last)/2;
   while (first <= last) {
      if (x[middle] < val)
         first = middle + 1;    
      else
         last = middle - 1;
      middle = (first + last)/2;
   }
   return first;
}
