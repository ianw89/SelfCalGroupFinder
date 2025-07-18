#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <sys/time.h>
#include <iostream>
#include <omp.h>
#include "libs/nanoflann.hpp"
#include "groups.hpp"
#include "fit_clustering_omp.hpp"
#include "nrutil.h"
using namespace nanoflann;

// Adaptor for nanoflann to access the GAL array
struct GalaxyCloud {
    inline unsigned int kdtree_get_point_count() const { return NGAL; }
    inline float kdtree_get_pt(const unsigned int idx, const unsigned int dim) const {
        if (dim == 0) return GAL[idx].x;
        if (dim == 1) return GAL[idx].y;
        return GAL[idx].z;
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

typedef KDTreeSingleIndexAdaptor<
    L2_Simple_Adaptor<float, GalaxyCloud>,
    GalaxyCloud,
    3 /* dim */
> GalaxyKDTree;

/* Local functions */
//void find_satellites(int icen, struct kdtree *kd);
void recalc_galprops();
void find_satellites(int icen, GalaxyKDTree *tree);
float fluxlim_correction(float z);
float get_wcen(int idx);
float get_chi_weight(int idx);
float get_bprob(int idx);
float lgrp_to_matching_rank(int idx);

galaxy *GAL = nullptr;
halo *HALO = nullptr;
int NGAL = 0;
int NHALO = 0;
const char *INPUTFILE = nullptr; 
const char *HALO_MASS_FUNC_FILE = "halo_mass_function.dat"; // Default value
const char *MOCK_FILE = nullptr;
const char *VOLUME_BINS_FILE = nullptr;

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

float MINREDSHIFT;
float MAXREDSHIFT;
float FRAC_AREA;
float GALAXY_DENSITY;
int INTERACTIVE = 0; // default is off
int FLUXLIM = 0; // default is volume-limited
float FLUXLIM_MAG = 0.0; 
int FLUXLIM_CORRECTION_MODEL = 0; // default is no correction, 1 for SDSS tuned, 2 for BGS tuned
int COLOR = 0; // default is ignore color information (sometimes treating all as blue)
int STELLAR_MASS = 0; // defaulit is luminosities
int RECENTERING = 0; // this options appears to always be off right now and hasn't been tested since fork
int SECOND_PARAMETER = 0; // default is no extra per-galaxy parameters
int SILENT = 0; // TODO make this work
int VERBOSE = 0; // TODO make this work
int POPULATE_MOCK = 0; // default is do not populate mock
int MAX_ITER = 5; // default is to go until fsat 0.002 convergence; can provide a number in parametrs instead
int ALLOW_EARLY_EXIT = 0; // default is to not allow early exit, but this is used in MCMC to speedups
FILE *MSG_PIPE = NULL; // default is no message pipe

void groupfind()
{
  FILE *fp;
  char aa[1000];
  float minvmax, maxvmax;
  int niter, ngrp_prev, icen_new;
  float frac_area, nsat_tot, weight;
  double galden, pt[3], t_start_findsats, t_end_findsats, t_start_iter, t_end_iter, t_alliter_s, t_alliter_e; // galden (galaxy density) only includes centrals, because only they get halos
  double *fsat_arr;


  static std::vector<int> flag;
  static float volume; 
  // xtmp stores the values of what we sort by and the index in the GAL array (first, second). It gets sorted and we find sats in that order.
  // Initially it gets setup with the lgal values; after it gets setup with the lgrp values (the effective length becomes ngrp then).
  static std::vector<std::pair<float,int>> xtmp;
  static int first_call = 1, ngrp;
  static GalaxyKDTree *tree = nullptr;

  fsat_arr = (double *) calloc(MAX_ITER, sizeof(double));

  if (first_call)
  {
    first_call = 0;
    fp = openfile(INPUTFILE);
    NGAL = filesize(fp);
    LOG_INFO("Allocating space for [%d] galaxies\n", NGAL);
    GAL = (struct galaxy*) calloc(NGAL, sizeof(struct galaxy));
    flag.resize(NGAL);

    // For volume-limited samples, we calculate the volume and put that in the vmax
    // property of each galxaxy.
    if (!FLUXLIM)
    {
      volume = 4. / 3. * PI * (pow(distance_redshift(MAXREDSHIFT), 3.0)) * FRAC_AREA;
      volume = volume - 4. / 3. * PI * (pow(distance_redshift(MINREDSHIFT), 3.0)) * FRAC_AREA;
    }

    // For flux-limited samples, we read in the vmax values from the file.
    // For that case, a factor of frac_area should already be included in the vmax.

    galden = 0;
    for (int i = 0; i < NGAL; ++i)
    {
      // Thought called lum throughout the code, this galaxy property could be mstellar too.
      fscanf(fp, "%f %f %f %f", &GAL[i].ra, &GAL[i].dec, &GAL[i].redshift, &GAL[i].lum);
      GAL[i].ra *= PI / 180.;
      GAL[i].dec *= PI / 180.;
      GAL[i].rco = distance_redshift(GAL[i].redshift);
      // check if the stellar mass (or luminosity) is in log
      if (GAL[i].lum < 100)
        GAL[i].lum = pow(10.0, GAL[i].lum);
      GAL[i].loglum = log10(GAL[i].lum);
      if (FLUXLIM)
        fscanf(fp, "%f", &GAL[i].vmax);
      else
        GAL[i].vmax = volume;
      if (COLOR)
        fscanf(fp, "%f", &GAL[i].color);
      if (SECOND_PARAMETER)
        fscanf(fp, "%f", &GAL[i].propx);
      if (SECOND_PARAMETER == 2)
        fscanf(fp, "%f", &GAL[i].propx2);
      GAL[i].weight = get_wcen(i);
      GAL[i].chiweight = get_chi_weight(i);
      GAL[i].bprob = get_bprob(i);
      GAL[i].x = GAL[i].rco * cos(GAL[i].ra) * cos(GAL[i].dec);
      GAL[i].y = GAL[i].rco * sin(GAL[i].ra) * cos(GAL[i].dec);
      GAL[i].z = GAL[i].rco * sin(GAL[i].dec);
      fgets(aa, 1000, fp);
      galden += 1 / GAL[i].vmax;
    }
    // print off largest and smallest vmax values
    minvmax = 1e10;
    maxvmax = -1e10;
    for (int i = 0; i < NGAL; ++i)
    {
      if (GAL[i].vmax < minvmax)
        minvmax = GAL[i].vmax;
      if (GAL[i].vmax > maxvmax)
        maxvmax = GAL[i].vmax;
    }
    LOG_INFO("min vmax= %e max vmax= %e\n", minvmax, maxvmax);

    fclose(fp);
    LOG_INFO("Done reading in from [%s]\n", INPUTFILE);

    if (!FLUXLIM)
    {
      LOG_INFO("Volume= %e L_box= %f\n", volume, pow(volume, THIRD));
      LOG_INFO("Number density= %e %e\n", NGAL / volume, galden);
      GALAXY_DENSITY = NGAL / volume;
    }

    xtmp.resize(NGAL);

  } // end of first call code
  else {
    LOG_INFO("Reusing existing GAL array. Recalculating properties.\n", NGAL);
    recalc_galprops();
  }

  // ************************************************************************************
  // For the first group finding iteration, sort by LGAL / stellar mass for SHAM
  // ************************************************************************************
  for (int i = 0; i < NGAL; ++i)
  {
    // Used to not multiply by chiweight here. (only in main iterations). But why not? Seems reasonable here too.
    xtmp[i] = std::make_pair(-(GAL[i].lum) * GAL[i].chiweight, i);
  }

  LOG_INFO("Sorting galaxies...\n");
  float tsort = omp_get_wtime();
  std::sort(xtmp.begin(), xtmp.end());
  float tsort2 = omp_get_wtime() - tsort;
  LOG_INFO("Done sorting galaxies in %.3f seconds.\n", tsort2);

  //for (int i = 0; i < NGAL; ++i)
  //  std::cerr << "xtmp[" << i << "] = (" << xtmp[i].first << ", " << xtmp[i].second << ")\n";

  // do the inverse-abundance matching
  density2host_halo(0.01); // TODO delete?
  LOG_INFO("Starting inverse-sham...\n");
  galden = 0;

  // reset the sham counters
  if (FLUXLIM)
    density2host_halo_zbins3(-1, 0);

  for (int i1 = 0; i1 < NGAL; ++i1)
  {
    int i = xtmp[i1].second;
    GAL[i].grp_rank = i1;
    // Set the galaxy's halo mass
    if (FLUXLIM)
      GAL[i].mass = density2host_halo_zbins3(GAL[i].redshift, GAL[i].vmax);
    else
    {
      galden += 1 / GAL[i].vmax;
      GAL[i].mass = density2host_halo(galden); // TODO is this right? we haven't counted galden fully yet?
    }
    // Set other properties derived from that
    update_galaxy_halo_props(&GAL[i]);
    GAL[i].psat = 0;
  }
  LOG_INFO("Done inverse-sham.\n");
  ngrp = NGAL;

  // Create the 3D KD tree for fast lookup of nearby galaxies
  if (tree == nullptr)
  {
    LOG_INFO("Building KD-tree...\n");
    static GalaxyCloud gal_cloud = GalaxyCloud();
    tree = new GalaxyKDTree(3, gal_cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    tree->buildIndex();
    LOG_INFO("Done building KD-tree. %d\n", ngrp);
  }

  // test the FOF group finder
  // test_fof(kd);

  // now let's go to the center finder
  // test_centering(kd);

  // Start the group-finding iterations
  t_alliter_s = omp_get_wtime();
  for (niter = 1; niter <= MAX_ITER; ++niter)
  {
    t_start_iter = omp_get_wtime();

    // Reset group properties except the halo mass
    // (We have the xtmp array from the last iteration with the previous LGRP values sorted already)
    for (int j = 0; j < NGAL; ++j)
    {
      GAL[j].igrp = -1;
      GAL[j].psat = 0;
      GAL[j].nsat = 0;
      GAL[j].lgrp = GAL[j].lum; //* GAL[j].chiweight;
      flag[j] = 1;
    }

    // Find the satellites for each halo, in order of group lum/mass
    // This is the where most CPU time is spent
    ngrp_prev = ngrp; // first iteration this is NGAL
    ngrp = 0;
    t_start_findsats = omp_get_wtime();
    int i1_par, i_par;
#pragma omp parallel for private(i1_par, i_par)
    for (i1_par = 0; i1_par < ngrp_prev; ++i1_par)
    {
      i_par = xtmp[i1_par].second;
      flag[i_par] = 0;
      find_satellites(i_par, tree);
    }
    t_end_findsats = omp_get_wtime();

    // After finding satellites, now set some properties on the centrals
    for (int i1 = 0; i1 < ngrp_prev; ++i1)
    {
      int i = xtmp[i1].second;
      if (GAL[i].psat <= 0.5)
      {
        GAL[i].igrp = i;
        xtmp[ngrp] = std::make_pair(lgrp_to_matching_rank(i), i);
        ngrp++;
        GAL[i].listid = ngrp;
      }
    }

// go back and check objects are newly-exposed centrals
    int j_par;
#pragma omp parallel for private(j_par)
    for (j_par = 0; j_par < NGAL; ++j_par)
    {
      if (flag[j_par] && GAL[j_par].psat <= 0.5)
      {
        //LOG_INFO("Newly exposed central: %d.\n", j);
        find_satellites(j_par, tree);
      }
    }
    for (j_par = 0; j_par < NGAL; ++j_par)
    {
      if (flag[j_par] && GAL[j_par].psat <= 0.5)
      {
        GAL[j_par].igrp = j_par;
        xtmp[ngrp] = std::make_pair(lgrp_to_matching_rank(j_par), j_par);
        ngrp++;
        GAL[j_par].listid = ngrp;
      }
    }

  // Fix up orphaned satellites (satellites of centrals that became satellites)
  for (int k = 0; k < NGAL; ++k) {
    if (GAL[k].psat > 0.5) {
      #ifndef OPTIMIZE
      if (GAL[k].igrp == k) { 
        LOG_ERROR("ERROR - psat>0.5 galaxy %d has igrp itself! (N_SAT=%d)\n", k, GAL[k].nsat);
      }
      #endif
      // Orphaned Satellite - it's central became a satellite in the final iteration. Need to reassign.
      if (GAL[GAL[k].igrp].igrp != GAL[k].igrp) { 
         // Consider this a central now, next loop will process it.
        //LOG_WARN("WARNING - psat>0.5 galaxy %d points to central %d which isn't a central!\n", k, GAL[k].igrp);
        GAL[k].igrp = k;
        GAL[k].psat = 0.0;
        GAL[k].nsat = 0;
        GAL[k].lgrp = GAL[k].lum;
        xtmp[ngrp] = std::make_pair(lgrp_to_matching_rank(k), k);
        ngrp++;  
        GAL[k].listid = ngrp;
      }
    }
    #ifndef OPTIMIZE
    else if (GAL[k].igrp != k) {
        LOG_ERROR("ERROR - psat<=0.5 galaxy %d not it's own central. igrp=%d, (N_SAT=%d)\n", k, GAL[k].igrp, GAL[k].nsat);
    }
    #endif
  }

    // Find new group centers if option enabled (its NOT)
    // Jeremy says this never worked anyway.
    // TODO: This might be broken, it's not been tested since the fork
    /*
    if (RECENTERING && niter != MAX_ITER)
    {
      for (j = 0; j < ngrp; ++j)
      {
        i = xtmp[j].second;
        if (GAL[i].mass > 5e12 && GAL[i].psat <= 0.5)
        {
          icen_new = group_center(i, kd);
          if (icen_new == -1)
          {
            printf("ZERO %d %e %.3f\n", GAL[i].nsat, GAL[i].mass, GAL[i].psat);
            exit(0);
          }
          if (icen_new != i)
          {
            // transfer the halo values
            LOG_VERBOSE("REC %d %d %d %d\n",niter, i, icen_new, j);
            xtmp[j].second = icen_new; // why not xtemp.first?
            GAL[i].psat = 1;
            GAL[i].igrp = icen_new; // need to swap all of them, fyi...
            GAL[icen_new].psat = 0;
            GAL[icen_new].lgrp = GAL[i].lgrp;
            GAL[icen_new].nsat = GAL[i].nsat;
            GAL[i].nsat = 0;
          }
        }
      }
    }
    */

    // sort groups by their total group luminosity / stellar mass for next time
    std::sort(xtmp.begin(), xtmp.begin() + ngrp);

    //for (int i = 0; i < NGAL; ++i)
    //  std::cerr << "xtmp[" << i << "] = (" << xtmp[i].first << ", " << xtmp[i].second << ")\n";

    // Re-assign the halo masses to each central
    nsat_tot = galden = 0;
    // reset the sham counters
    if (FLUXLIM)
      density2host_halo_zbins3(-1, 0);
    for (int j = 0; j < ngrp; ++j)
    {
      int i = xtmp[j].second; // Sorted index
      GAL[i].grp_rank = j;
      galden += 1 / GAL[i].vmax;
      if (FLUXLIM)
        GAL[i].mass = density2host_halo_zbins3(GAL[i].redshift, GAL[i].vmax);
      else
        GAL[i].mass = density2host_halo(galden); // BUG is this right, we haven't fully counted up galden yet?
      update_galaxy_halo_props(&GAL[i]);
      nsat_tot += GAL[i].nsat;
    }

    fsat_arr[niter-1] = nsat_tot / NGAL; // store the fraction of satellites in this iteration
    t_end_iter = omp_get_wtime();

    LOG_INFO("iter %d ngroups=%d fsat=%f (kdtime=%.2f %.2f)\n", niter, ngrp, fsat_arr[niter-1], t_end_findsats - t_start_findsats, t_end_iter - t_start_iter);

    // When allowing early exit, check if the change in fsat is small enough to stop
    if (ALLOW_EARLY_EXIT && niter > 1 && fabs(fsat_arr[niter-1] - fsat_arr[niter-2]) < 0.001)
    {
      LOG_INFO("Early abortion at iteration %d.\n", niter);
      break;
    }

  } // end of main iteration loop

  t_alliter_e = omp_get_wtime();
  LOG_PERF("Group finding complete. All iterations took %.2fs.\n", t_alliter_e - t_alliter_s);

  // **********************************
  // End of group finding
  // Copy group properties to each member
  // Perform other sanity checks
  // **********************************
  for (int j = 0; j < NGAL; ++j) {
    // Satellites
    if (GAL[j].psat > 0.5) {
      // It thinks it's a satellite of itself? Should not happen
      if (GAL[j].igrp == j) { 
        LOG_ERROR("FINAL ERROR - psat>0.5 galaxy %d has igrp itself! (N_SAT=%d)\n", j, GAL[j].nsat);
      }
      // Orphaned Satellite - it's central became a satellite in the final iteration. Need to reassign.
      if (GAL[GAL[j].igrp].igrp != GAL[j].igrp) { 
        LOG_ERROR("FINAL ERROR - psat>0.5 galaxy %d points to central %d which isn't a central!\n", j, GAL[j].igrp);
      }
      // Standard satellite - Copy group properties to this member
      else {
        GAL[j].lgrp = GAL[GAL[j].igrp].lgrp;
        GAL[j].mass = GAL[GAL[j].igrp].mass;
        GAL[j].nsat = GAL[GAL[j].igrp].nsat;
      }
    }
    // Centrals
    else {
      if (GAL[j].igrp != j) {
        LOG_ERROR("FINAL ERROR - psat<=0.5 galaxy %d not it's own central. igrp=%d, (N_SAT=%d)\n", j, GAL[j].igrp, GAL[j].nsat);
      }
    }
  }
    
  // **********************************
  // Output the group catalog to stdout
  // **********************************
  if (!SILENT) {
    for (int i = 0; i < NGAL; ++i) {
      printf("%d %f %f %f %e %e %f %e %d %e %d %f %f\n",
              i, GAL[i].ra * 180 / PI, GAL[i].dec * 180 / PI, GAL[i].redshift,
              GAL[i].lum, GAL[i].vmax, GAL[i].psat, GAL[i].mass,
              GAL[i].nsat, GAL[i].lgrp, GAL[i].igrp, GAL[i].weight, GAL[i].chiweight);
    }
  }
  fflush(stdout);
  
  // Free all dynamically allocated memory that we don't reuse in future groupfind() calls (ITERATIVE MODE)
  free(fsat_arr);
}


/**
 * Recalculate galaxy properties when the parameters have changed (interactive mode).
 */
void recalc_galprops() {
  for (int i=0; i<NGAL; ++i) {
    GAL[i].weight = get_wcen(i);
    GAL[i].chiweight = get_chi_weight(i);
    GAL[i].bprob = get_bprob(i);
    GAL[i].lgrp = GAL[i].lum; // reset lgrp to lum
  }
}

/* Here is the main code to find satellites for a given central galaxy
 */
void find_satellites(int icen, GalaxyKDTree *tree)
{
  int j, k;
  float dx, dy, dz, theta, prob_ang, vol_corr, prob_rad, grp_lum, p0;
  float bprob;
  std::vector<nanoflann::ResultItem<unsigned int, float>> ret_matches;
  nanoflann::SearchParameters params = nanoflann::SearchParameters();
  float sat[3];

  // check if this galaxy has already been given to a group
  if (GAL[icen].psat > 0.5)
    return;

  // Use the k-d tree kd to identify the nearest galaxies to the central.
  float query_pt[3] = { GAL[icen].x, GAL[icen].y, GAL[icen].z };

  // TODO This search could use this range for z, but something smaller for ra, dec. 
  // In Euclidean, what shape should the search be to take this into account?

  // Nearest neighbour search should go out to about 4*sigma, the velocity dispersion of the SHAMed halo.
  // find all galaxies in 3D that are within 4sigma of the velocity dispersion
  const float range = 4 * GAL[icen].sigmav / 100.0 * (1 + GAL[icen].redshift) /
          sqrt(OMEGA_M * pow(1 + GAL[icen].redshift, 3.0) + 1 - OMEGA_M);
  const float search_radius = range * range; // nanoflann expects squared radius
  ret_matches.reserve(20); 

  // TODO possible optimization is to store the set for each galaxy for a large sigmav value? Hmm
  // First time through (on a per-galaxy basis) increase sigmav to use by a factor of 2 (can tune)
  // Then store results and reuse them for the next iterations
  // Only redo if the a newly calculated search_radius is bigger than the stored one.
  ret_matches.clear();
  tree->radiusSearch(query_pt, search_radius, ret_matches, params);

  // Set now contains the nearest neighbours within a distance range. Grab their info.
  for (const auto& match : ret_matches)
  {

    // Get index value of the current neighbor
    j = match.first; // index of the galaxy in the GAL array

    // skip if the object is more massive than the icen
    if (GAL[j].lum >= GAL[icen].lum)
      continue;

    // Skip if target galaxy is the same as the central (obviously).
    if (j == icen)
      continue;

    // Skip if already assigned to a central, UNLESS current group has priority
    if (GAL[j].psat > 0.5 && GAL[icen].grp_rank > GAL[GAL[j].igrp].grp_rank)
      continue;

    // check if the galaxy is outside the angular radius of the halo
    theta = angular_separation(GAL[icen].ra, GAL[icen].dec, GAL[j].ra, GAL[j].dec);
    if (theta > GAL[icen].theta)
    {
      continue;
    }

    // Now determine the probability of being a satellite
    //(both projected onto the sky, and along the line of sight).
    dz = fabs(GAL[icen].redshift - GAL[j].redshift) * SPEED_OF_LIGHT;
    p0 = psat(&GAL[icen], theta, dz, GAL[j].bprob);

    // Keep track of the highest psat so far
    if (p0 > GAL[j].psat)
      GAL[j].psat = p0;    
    if (p0 <= 0.5)
      continue;

    // This is considered a member of the group

    // If this was previously a member of another (lower-rank) group, remove it from that.
    if (GAL[j].igrp >= 0)
    {
      // It was it's own central 
      // Not entirely sure how this happens given the ordering of the loop, but it can. I don't think it's a bug?
      // Perhaps it's because of parallelization?
      if (GAL[j].igrp == j) 
      {
        if (GAL[j].nsat > 0)
        {
          // It's its own central with satellites (as of this iteration!), but we are adding it to this central
          // BUG I think this special case leads to issue that we don't handle it right if its' the last iteration
          //fprintf(stderr, "Central %d with (N_SAT=%d) into CENTRAL %d\n", j, GAL[j].nsat, icen);
        }
      }
      else // Was just a sat of another group, update that group's properties
      {
       // BUG: Aren't there race conditions here?
        GAL[GAL[j].igrp].nsat--;
        GAL[GAL[j].igrp].lgrp -= GAL[j].lum;
      }
    }
    // Assign it to this group
    // BUG: Aren't there race conditions here? 
    // Multiple threads can try to assign the same galaxy to a group.
    GAL[j].psat = p0;
    GAL[j].nsat = 0;
    GAL[j].igrp = icen;
    GAL[icen].lgrp += GAL[j].lum;
    GAL[icen].nsat++;
  }
  
  //  Correct for boundary conditions
  if (!FLUXLIM)
  {
    // TODO BUG I switched nsat to be int, will this be busted?
    dz = SPEED_OF_LIGHT * fabs(GAL[icen].redshift - MINREDSHIFT);
    vol_corr = 1 - (0.5 * erfc(dz / (ROOT2 * GAL[icen].sigmav)));
    GAL[icen].nsat /= vol_corr;
    GAL[icen].lgrp /= vol_corr;

    dz = SPEED_OF_LIGHT * fabs(GAL[icen].redshift - MAXREDSHIFT);
    vol_corr = 1 - (0.5 * erfc(dz / (ROOT2 * GAL[j].sigmav)));
    GAL[icen].nsat /= vol_corr;
    GAL[icen].lgrp /= vol_corr;
  }

  //float t_done = omp_get_wtime();
  //if (VERBOSE)
  //  fprintf(stderr, "Thread %d: icen=%d start to set=%.5f full method=%.5f\n", thread_num, icen, t_nset_done - t_start, t_done - t_start);
}


/* 
 * This is a luminosity correction for flux-limited samples that 
 * is applied to the group luminosity for the purposes of the 
 * inverse-sham. For flux limited samples, we are already doing the 
 * SHAMing in narrow redshift bins, so this correction is really
 * a minor one for galaxies within the same redshift bin.
 * 
 * TODO: test that no correct vs model 2 for BGS shouldn't change a lot.
 * 
 * These comments are old and Jeremy does not remember them:
 * 
 * This is calibrated from the MXXL BGS mock,
 * from ratio of luminosity density in redshift
 * bins relative to total 1/Vmax-weighted luminosity
 * density. (SLightly different than Yang et al).
 *
 * luminosity_correction.py
 */
float fluxlim_correction(float z) {
  switch (FLUXLIM_CORRECTION_MODEL) {
  case 0:
    return 1; // no correction
  case 1:
    return pow(10.0, pow(z / 0.18, 2.8) * 0.5); // rho_lum(z) for SDSS (r=17.77; MXXL)
  case 2:
    return pow(10.0, pow(z / 0.40, 4.0) * 0.4); // from rho_lum(z) BGS
  }

  return 1;
  //return pow(10.0, pow(z / 0.16, 2.5) * 0.6); // SDSS (sham mock)
}

// Color-dependent weighting of centrals luminosities/stellar masses
float get_wcen(int idx) {
  float weight = 1.0;
  if (USE_WCEN)
  {
    if (GAL[idx].color < 0.8)
      // If colors not provided, this is what will be used
      weight = 1.0 / pow(10.0, 0.5 * (1 + erf((GAL[idx].loglum - WCEN_MASS) / WCEN_SIG)) * WCEN_NORM);
    else
      weight = 1.0 / pow(10.0, 0.5 * (1 + erf((GAL[idx].loglum - WCEN_MASSR) / WCEN_SIGR)) * WCEN_NORMR);
  }
  return weight;
}

float get_chi_weight(int idx) {
  float weight = 1.0;
  float wx;
  if (SECOND_PARAMETER) {
    if (GAL[idx].color < 0.8)
    {
      wx = PROPX_WEIGHT_BLUE + PROPX_SLOPE_BLUE * (GAL[idx].loglum - 9.5);
      weight = exp(GAL[idx].propx / wx);
    }
    else
    {
      wx = PROPX_WEIGHT_RED + PROPX_SLOPE_RED * (GAL[idx].loglum - 9.5);
      weight = exp(GAL[idx].propx / wx);
    }
  }
  if (SECOND_PARAMETER == 2) {
    if (GAL[idx].color < 0.8)
      weight *= exp(GAL[idx].propx2 / PROPX2_WEIGHT_BLUE);
    else
      weight *= exp(GAL[idx].propx2 / PROPX2_WEIGHT_RED);
  }
  return weight;
}

float get_bprob(int idx) {
  float bprob = BPROB_DEFAULT;
  if (USE_BSAT) {
    if (GAL[idx].color > 0.8)
      bprob = BPROB_RED + (GAL[idx].loglum - 9.5) * BPROB_XRED;
    else
      bprob = BPROB_BLUE + (GAL[idx].loglum - 9.5) * BPROB_XBLUE;
  }
  // let's put a lower limit of the prob
  if (bprob < 0.001)
    bprob = 0.001;

  return bprob;
}

float lgrp_to_matching_rank(int idx) {
  // Lgrp sums all unweighted luminosities (or stellar mass) in the group.
  // For the abundance matching, we want the chi weight to be applied to the central only.
  // The wcen weight is applied to the entire group luminosity.
  float value = - (GAL[idx].lgrp - GAL[idx].lum + GAL[idx].lum*GAL[idx].chiweight) * GAL[idx].weight;
  if (FLUXLIM)
    value *= fluxlim_correction(GAL[idx].redshift);
  return value;
}