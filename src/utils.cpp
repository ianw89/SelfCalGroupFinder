#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "groups.hpp"

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
float angular_separation_old(float a1, float d1, float a2, float d2)
{
  return atan((sqrt(cos(d2) * cos(d2) * sin(a2 - a1) * sin(a2 - a1) +
                    pow(cos(d1) * sin(d2) - sin(d1) * cos(d2) * cos(a2 - a1), 2.0))) /
              (sin(d1) * sin(d2) + cos(d1) * cos(d2) * cos(a2 - a1)));
}

/* Angular separation between two points using Vincenty formula. Give the ra, dec in radians.
 */
float angular_separation(float a1, float d1, float a2, float d2)
{
  float sin_d1 = sin(d1);
  float cos_d1 = cos(d1);
  float sin_d2 = sin(d2);
  float cos_d2 = cos(d2);
  float delta_a = a2 - a1;
  float cos_delta_a = cos(delta_a);
  float sin_delta_a = sin(delta_a);

  float numerator = sqrt(pow(cos_d2 * sin_delta_a, 2) +
                         pow(cos_d1 * sin_d2 - sin_d1 * cos_d2 * cos_delta_a, 2));
  float denominator = sin_d1 * sin_d2 + cos_d1 * cos_d2 * cos_delta_a;

  return atan2(numerator, denominator);
}

float psat(struct galaxy *central, float dr, float dz, float bprob)
{
  float prob_ang, prob_rad, result;
  prob_ang = compute_p_proj_g(central, dr);
  prob_rad = compute_p_z(dz, central->sigmav);
  result = (1 - 1 / (1 + prob_ang * prob_rad / bprob));

  #ifndef OPTIMIZE
  if (isnan(result))
  {
    fprintf(stderr, "Unexpected nan result in psat: dr=%f dz=%f bprob=%f prob_ang=%f prob_rad%f RESULT: %f\n", dr, dz, bprob, prob_ang, prob_rad, result);
    result = 0.0;
  }
  #endif

  return result;
}

/* Probability assuming a projected NFW profile using the given galaxy's halo properties.
 */
float compute_p_proj_g(struct galaxy *gal, float dr)
{
  return compute_p_proj(gal->mass, dr, gal->rad, gal->theta);
}

/* Probability assuming a projected NFW profile
 */
float compute_p_proj(float mass, float dr, float rad, float ang_rad)
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
float compute_p_z(float dz, float sigmav)
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

/* This takes a file and reads the number of lines in it,
 * rewinds the file and returns the lines.
 */

int filesize(FILE *fp)
{
  int i=-1;
  char a[1000];

  while(!feof(fp))
    {
      i++;
      fgets(a,1000,fp);
    }
  rewind(fp);
  return(i);
}

FILE *openfile(const char *ff)
{
  FILE *fp;
  if(!(fp=fopen(ff,"r")))
    {
      fprintf(stderr,"ERROR opening [%s]\n",ff);
      exit(0);
    }
  return(fp);
}
