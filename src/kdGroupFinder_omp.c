#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <sys/time.h>
#include <omp.h>
#include "nrutil.h"
#include "kdtree.h"
#include "groups.h"
#include "fit_clustering_omp.h"

/* Initializes global variables for running group finder.
 * Contains high level methods for group finding algorithm.
 */

struct galaxy *GAL;
int NGAL;

/* Local functions */
void find_satellites(int icen, void *kd);
float fluxlim_correction(float z);

/* Variables for determining if a galaxy is a satellite */
int USE_BSAT = 0; // off by default
const float BPROB_DEFAULT = 10.0;
float BPROB_RED, BPROB_BLUE, BPROB_XRED, BPROB_XBLUE= 0.0;

/* Variables for weighting the central galaxies of the blue galaxies */
int USE_WCEN = 0; // off by default
float WCEN_MASS, WCEN_SIG, WCEN_MASSR, WCEN_SIGR, WCEN_NORM, WCEN_NORMR = 0.0;

// TODO update these to new pattern
float PROPX_WEIGHT_RED = 1000.0,
      PROPX_WEIGHT_BLUE = 1000.0;
float PROPX_SLOPE_RED = 0,
      PROPX_SLOPE_BLUE = 0;
float PROPX2_WEIGHT_RED = 1000.0,
      PROPX2_WEIGHT_BLUE = 1000.0;

char *INPUTFILE;
float MINREDSHIFT;
float MAXREDSHIFT;
float FRAC_AREA;
float GALAXY_DENSITY;
int FLUXLIM = 0; // default is volume-limited
int COLOR = 0; // default is ignore color information (sometimes treating all as blue)
int STELLAR_MASS = 0; // defaulit is luminosities
int RECENTERING = 0; // this options appears to always be off right now and hasn't been tested since fork
int SECOND_PARAMETER = 0; // default is no extra per-galaxy parameters
int SILENT = 0; // TODO make this work
int VERBOSE = 0; // TODO make this work
int POPULATE_MOCK = 0; // default is do not populate mock

void groupfind()
{
  FILE *fp;
  char aa[1000];
  int i, i1, niter, MAX_ITER = 5, j, ngrp_prev, icen_new;
  float frac_area, zmin, zmax, nsat_tot, weight, wx;
  int colors = 1;
  double galden, pt[3], t0, t1, t3, t4;
  long IDUM1 = -555;

  static int *permanent_id, *itmp, *flag;
  static float volume, *xtmp, *lumshift;
  static void *kd;
  static int first_call = 1, ngrp;

  if (first_call)
  {
    colors = COLOR;
    first_call = 0;
    fp = openfile(INPUTFILE);
    NGAL = filesize(fp);
    if (!SILENT) fprintf(stderr, "Allocating space for [%d] galaxies\n", NGAL);
    GAL = calloc(NGAL, sizeof(struct galaxy));
    flag = ivector(0, NGAL - 1);

    zmin = MINREDSHIFT;
    zmax = MAXREDSHIFT;

    // For volume-limited samples, we calculate the volume and put that in the vmax
    // property of each galxaxy.
    if (!FLUXLIM)
    {
      volume = 4. / 3. * PI * (pow(distance_redshift(zmax), 3.0)) * FRAC_AREA;
      volume = volume - 4. / 3. * PI * (pow(distance_redshift(zmin), 3.0)) * FRAC_AREA;
    }

    // For flux-limited samples, we read in the vmax values from the file.
    // For that case, a factor of frac_area should already be included in the vmax.

    galden = 0;
    for (i = 0; i < NGAL; ++i)
    {
      // Thought called mstellar throughout the code, this galaxy property could be luminosity
      // instead (at least, that's what Ian has been doing). Unclear if that is an issue yet. TODO
      fscanf(fp, "%f %f %f %f", &GAL[i].ra, &GAL[i].dec, &GAL[i].redshift, &GAL[i].mstellar);
      GAL[i].ra *= PI / 180.;
      GAL[i].dec *= PI / 180.;
      //GAL[i].id = i;
      GAL[i].rco = distance_redshift(GAL[i].redshift);
      // check if the stellar mass (or luminosity) is in log
      if (GAL[i].mstellar < 100)
        GAL[i].mstellar = pow(10.0, GAL[i].mstellar);
      // I think we can just set the bsat here
      if (FLUXLIM)
        fscanf(fp, "%f", &GAL[i].vmax);
      else
        GAL[i].vmax = volume;
      if (colors)
        fscanf(fp, "%f", &GAL[i].color);
      if (SECOND_PARAMETER)
        fscanf(fp, "%f", &GAL[i].propx);
      if (SECOND_PARAMETER == 2)
        fscanf(fp, "%f", &GAL[i].propx2);
      fgets(aa, 1000, fp);
      galden += 1 / GAL[i].vmax;
    }
    fclose(fp);
    if (!SILENT) fprintf(stderr, "Done reading in from [%s]\n", INPUTFILE);

    if (!FLUXLIM)
    {
      if (!SILENT) fprintf(stderr, "Volume= %e L_box= %f\n", volume, pow(volume, THIRD));
      if (!SILENT) fprintf(stderr, "Number density= %e %e\n", NGAL / volume, galden);
      GALAXY_DENSITY = NGAL / volume;
    }

    // first sort by stellar mass
    xtmp = vector(1, NGAL);
    itmp = ivector(1, NGAL);
    permanent_id = ivector(1, NGAL);
    lumshift = vector(0, NGAL - 1);
    for (i = 1; i <= NGAL; ++i)
    {
      // just for kicks, give each galaxy a random luminosity
      lumshift[i - 1] = pow(10.0, gasdev(&IDUM1) * 0.0);

      xtmp[i] = -(GAL[i - 1].mstellar * lumshift[i - 1]);
      itmp[i] = i - 1;
    }
    if (!SILENT) fprintf(stderr, "sorting galaxies...\n");
    sort2(NGAL, xtmp, itmp);
    if (!SILENT) fprintf(stderr, "done sorting galaxies.\n");

    // do the inverse-abundance matching
    density2host_halo(0.01);
    if (!SILENT) fprintf(stderr, "Starting inverse-sham...\n");
    galden = 0;
    // reset the sham counters
    if (FLUXLIM)
      density2host_halo_zbins3(-1, 0);
    for (i1 = 1; i1 <= NGAL; ++i1)
    {
      i = itmp[i1];
      GAL[i].grp_rank = i1;
      // Set the galaxy's halo mass
      if (FLUXLIM)
        GAL[i].mass = density2host_halo_zbins3(GAL[i].redshift, GAL[i].vmax);
      else
      {
        galden += 1 / GAL[i].vmax;
        GAL[i].mass = density2host_halo(galden);
      }
      // Set other properties derived from that
      update_galaxy_halo_props(&GAL[i]);
      GAL[i].psat = 0;
      j = i;
      GAL[j].x = GAL[j].rco * cos(GAL[j].ra) * cos(GAL[j].dec);
      GAL[j].y = GAL[j].rco * sin(GAL[j].ra) * cos(GAL[j].dec);
      GAL[j].z = GAL[j].rco * sin(GAL[j].dec);
    }
    if (!SILENT) fprintf(stderr, "Done inverse-sham.\n");
    // assume that NGAL=NGROUP at first
    ngrp = NGAL;
  }

  // let's create a 3D KD tree
  if (!SILENT) fprintf(stderr, "Building KD-tree...\n");
  kd = kd_create(3);
  for (i = 1; i <= NGAL; ++i)
  {
    j = i;
    permanent_id[j] = j;
    pt[0] = GAL[j].x;
    pt[1] = GAL[j].y;
    pt[2] = GAL[j].z;
    assert(kd_insert(kd, pt, (void *)&permanent_id[j]) == 0);
  }
  if (!SILENT) fprintf(stderr, "Done building KD-tree. %d\n", ngrp);

  // test the FOF group finder
  // test_fof(kd);

  // now let's go to the center finder
  // test_centering(kd);

  // now start the group-finding iterations
  for (niter = 1; niter <= MAX_ITER; ++niter)
  {
    t3 = omp_get_wtime();
    // first, reset the psat values
    for (j = 0; j < NGAL; ++j)
    {
      GAL[j].igrp = -1;
      GAL[j].psat = 0;
      GAL[j].nsat = 0;
      GAL[j].mtot = GAL[j].mstellar;

      // Each galaxy can optionally have it's halo mass weighted by a property
      weight = 1.0;
      if (SECOND_PARAMETER)
      {
        if (GAL[j].color < 0.8)
        {
          wx = PROPX_WEIGHT_BLUE + PROPX_SLOPE_BLUE * (log10(GAL[j].mstellar) - 9.5);
          weight = exp(GAL[j].propx / wx);
        }
        if (GAL[j].color > 0.8)
        {
          wx = PROPX_WEIGHT_RED + PROPX_SLOPE_RED * (log10(GAL[j].mstellar) - 9.5);
          weight = exp(GAL[j].propx / wx);
        }
      }
      if (SECOND_PARAMETER == 2)
      {
        if (GAL[j].color < 0.8)
          weight *= exp(GAL[j].propx2 / PROPX2_WEIGHT_BLUE);
        if (GAL[j].color > 0.8)
          weight *= exp(GAL[j].propx2 / PROPX2_WEIGHT_RED);
      }
      GAL[j].mtot *= weight;

      // Color-dependent weighting of centrals masses
      weight = 1.0;
      if (USE_WCEN)
      {
        if (GAL[j].color < 0.8)
          // If colors not provided, this is what will be used
          weight = 1 / pow(10.0, 0.5 * (1 + erf((log10(GAL[j].mstellar) - WCEN_MASS) / WCEN_SIG)) * WCEN_NORM);
        else
          weight = 1 / pow(10.0, 0.5 * (1 + erf((log10(GAL[j].mstellar) - WCEN_MASSR) / WCEN_SIGR)) * WCEN_NORMR);
      }
      GAL[j].weight = weight;

      flag[j] = 1;
    }
    // find the satellites for each halo, in order of group mass
    ngrp_prev = ngrp;
    ngrp = 0;
    t0 = omp_get_wtime();
#pragma omp parallel for private(i1, i)
    for (i1 = 1; i1 <= ngrp_prev; ++i1)
    {
      i = itmp[i1];
      flag[i] = 0;
      find_satellites(i, kd);
    }
    for (i1 = 1; i1 <= ngrp_prev; ++i1)
    {
      i = itmp[i1];
      if (GAL[i].psat < 0.5)
      {
        GAL[i].igrp = i;
        ngrp++;
        GAL[i].mtot *= GAL[i].weight;
        xtmp[ngrp] = -GAL[i].mtot;
        itmp[ngrp] = i;
        GAL[i].listid = ngrp;
        if (FLUXLIM)
          xtmp[ngrp] *= fluxlim_correction(GAL[i].redshift);
      }
    }
    t1 = omp_get_wtime();

// go back and check objects are newly-exposed centrals
#pragma omp parallel for private(j)
    for (j = 0; j < NGAL; ++j)
    {
      if (flag[j] && GAL[j].psat < 0.5)
      {
        find_satellites(j, kd);
      }
    }
    for (j = 0; j < NGAL; ++j)
    {
      if (flag[j] && GAL[j].psat < 0.5)
      {
        ngrp++;
        GAL[j].igrp = j;
        GAL[j].mtot *= GAL[j].weight;
        xtmp[ngrp] = -GAL[j].mtot;
        itmp[ngrp] = j;
        GAL[j].listid = ngrp;
        if (FLUXLIM)
          xtmp[ngrp] *= fluxlim_correction(GAL[j].redshift);
      }
    }

    // Find new group centers if option enabled (its NOT)
    if (RECENTERING && niter != MAX_ITER)
    {
      for (j = 1; j <= ngrp; ++j)
      {
        i = itmp[j];
        if (GAL[i].mass > 5e12 && GAL[i].psat < 0.5)
        {
          icen_new = group_center(i, kd);
          if (icen_new == -1)
          {
            printf("ZERO %.1f %e %.3f\n", GAL[i].nsat, GAL[i].mass, GAL[i].psat);
            exit(0);
          }
          if (icen_new != i)
          {
            // transfer the halo values
            if (!SILENT && VERBOSE)
              fprintf(stderr, "REC %d %d %d %d\n",niter, i, icen_new, j);
            itmp[j] = icen_new;
            GAL[i].psat = 1;
            GAL[i].igrp = icen_new; // need to swap all of them, fyi...
            GAL[icen_new].psat = 0;
            GAL[icen_new].mtot = GAL[i].mtot;
            GAL[icen_new].nsat = GAL[i].nsat;
            GAL[i].nsat = 0;
          }
        }
      }
    }

    // sort groups by their total stellar mass
    sort2(ngrp, xtmp, itmp);

    // reassign the halo masses
    nsat_tot = galden = 0;
    // reset the sham counters
    if (FLUXLIM)
      density2host_halo_zbins3(-1, 0);
    for (j = 1; j <= ngrp; ++j)
    {
      GAL[i].grp_rank = j;
      i = itmp[j];
      galden += 1 / GAL[i].vmax;
      if (FLUXLIM == 1)
        GAL[i].mass = density2host_halo_zbins3(GAL[i].redshift, GAL[i].vmax);
      else
        GAL[i].mass = density2host_halo(galden);
      update_galaxy_halo_props(&GAL[i]);
      nsat_tot += GAL[i].nsat;
    }
    t4 = omp_get_wtime();

    // for the satellites, set their host halo mass
    for (j = 0; j < NGAL; ++j)
      if (GAL[j].psat > 0.5)
        GAL[j].mass = GAL[GAL[j].igrp].mass;

    if (!SILENT) 
      fprintf(stderr, "iter %d ngroups=%d fsat=%f (kdtime=%.2f %.2f)\n",
            niter, ngrp, nsat_tot / NGAL, t1 - t0, t4 - t3);
  } // end of main iteration loop

  /* Output to disk the final results
   */
  for (i = 0; i < NGAL; ++i)
  {
    // the weight printed off here is only the color-dependent centrals weight
    // not the chi properties affected weight
    // TODO: should we change that?
    printf("%d %f %f %f %e %e %f %e %e %e %d %e\n",
            i, GAL[i].ra * 180 / PI, GAL[i].dec * 180 / PI, GAL[i].redshift,
            GAL[i].mstellar, GAL[i].vmax, GAL[i].psat, GAL[i].mass,
            GAL[i].nsat, GAL[i].mtot, GAL[i].igrp, GAL[i].weight);
  }
  fflush(stdout);
  
  /* let's free up the memory of the kdtree
   */
  kd_free(kd);
}

/* Here is the main code to find satellites for a given central galaxy
 */
void find_satellites(int icen, void *kd)
{
  int j, k;
  float dx, dy, dz, theta, prob_ang, vol_corr, prob_rad, grp_lum, p0, range;
  float cenDist, bprob;
  void *set;
  int *pch;
  double cen[3];
  double sat[3];

  // check if this galaxy has already been given to a group
  if (GAL[icen].psat > 0.5)
    return;

  // Use the k-d tree kd to identify the nearest galaxies to the central.
  cen[0] = GAL[icen].x;
  cen[1] = GAL[icen].y;
  cen[2] = GAL[icen].z;

  // Nearest neighbour search should go out to about 4*sigma, the velocity dispersion of the SHAMed halo.
  // find all galaxies in 3D that are within 4sigma of the velocity dispersion
  range = 4 * GAL[icen].sigmav / 100.0 * (1 + GAL[icen].redshift) /
          sqrt(OMEGA_M * pow(1 + GAL[icen].redshift, 3.0) + 1 - OMEGA_M);
  set = kd_nearest_range(kd, cen, range);

  // Set now contains the nearest neighbours within a distance range. Grab their info.

  while (!kd_res_end(set))
  {

    // Get index value of the current neighbor
    pch = (int *)kd_res_item(set, sat);
    j = *pch;
    kd_res_next(set);
    // printf("%d %d %f %f %f %f\n",j,icen,GAL[icen].x, GAL[j].x, range,sat[0]);

    // Skip if target galaxy is the same as the central (obviously).
    if (j == icen)
      continue;

    // skip if the object is more massive than the icen
    if (GAL[j].mstellar >= GAL[icen].mstellar)
      continue;

    // Skip if already assigned to a central.
    // if(GAL[j].psat>0.5)continue;
    // UNLESS current group has priority
    if (GAL[j].psat > 0.5 && GAL[icen].grp_rank > GAL[GAL[j].igrp].grp_rank)
      continue;

    // check if the galaxy is outside the angular radius of the halo
    dz = fabs(GAL[icen].redshift - GAL[j].redshift) * SPEED_OF_LIGHT;
    theta = angular_separation(GAL[icen].ra, GAL[icen].dec, GAL[j].ra, GAL[j].dec);
    if (theta > GAL[icen].theta)
    {
      continue;
    }

    // Now determine the probability of being a satellite
    //(both projected onto the sky, and along the line of sight).

    // set the background level
    bprob = BPROB_DEFAULT;
    if (USE_BSAT)
    {
      if (GAL[j].color > 0.8)
        bprob = BPROB_RED + (log10(GAL[j].mstellar) - 9.5) * BPROB_XRED;
      else
        bprob = BPROB_BLUE + (log10(GAL[j].mstellar) - 9.5) * BPROB_XBLUE;
    }
    // let's put a lower limit of the prob
    if (bprob < 0.001)
      bprob = 0.001;

    p0 = psat(&GAL[icen], theta, dz, bprob);
    if (isnan(p0))
    {
      p0 = 1; //???
      fprintf(stderr, "Unexpected nan result for prob sat at index %i", j);
    }
    // Keep track of the highest psat so far
    if (p0 > GAL[j].psat)
      GAL[j].psat = p0;    
    if (p0 < 0.5)
      continue;

    // this is considered a member of the group
    // NB if this was previously a member of other (lower-rank)
    // group, remove it from that.
    if (GAL[j].igrp >= 0)
    {
      GAL[GAL[j].igrp].nsat--;
      GAL[GAL[j].igrp].mtot -= GAL[j].mstellar;
    }
    GAL[j].psat = p0;
    GAL[j].igrp = icen;
    GAL[icen].mtot += GAL[j].mstellar;
    GAL[icen].nsat++;
  }
  // exit(0);
  //  Correct for boundary conditions
  if (!FLUXLIM)
  {
    dz = SPEED_OF_LIGHT * fabs(GAL[icen].redshift - MINREDSHIFT);
    vol_corr = 1 - (0.5 * erfc(dz / (ROOT2 * GAL[icen].sigmav)));
    GAL[icen].nsat /= vol_corr;
    GAL[icen].mtot /= vol_corr;

    dz = SPEED_OF_LIGHT * fabs(GAL[icen].redshift - MAXREDSHIFT);
    vol_corr = 1 - (0.5 * erfc(dz / (ROOT2 * GAL[j].sigmav)));
    GAL[icen].nsat /= vol_corr;
    GAL[icen].mtot /= vol_corr;
  }
}

/* This is calibrated from the MXXL BGS mock,
 * from ratio of luminosity density in redshift
 * bins relative to total 1/Vmax-weighted luminosity
 * density. (SLightly different than Yang et al).
 *
 * luminosity_correction.py
 */
float fluxlim_correction(float z)
{
  return pow(10.0, pow(z / 0.18, 2.8) * 0.5); // rho_lum(z) for SDSS (r=17.77; MXXL)
  return 1;                                   // no correction
  return pow(10.0, pow(z / 0.16, 2.5) * 0.6); // SDSS (sham mock)
  return pow(10.0, pow(z / 0.40, 4.0) * 0.4); // from rho_lum(z) BGS
}
