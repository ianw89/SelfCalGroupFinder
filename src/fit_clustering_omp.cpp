#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>
#include <errno.h>
#include <stdint.h>
#include "groups.hpp"
#include "fit_clustering_omp.hpp"
#include "nrutil.h"

// Definitions
#define MAXBINS 10 // This is the max number of magnitude bins we use for the HOD. The actual amount we use is read in from VOLUME_BINS_FILE
#define MAX_SATELLITES 1500 // Print a warning and cap the number of satellites to this if it is exceeded. Likely bad parameters.
#define HALO_BINS 200 // log10(M_halo) / 0.1
#define MIN_HALO_IDX 90 // log10(M_halo) = 9.0
#define MAX_HALO_IDX 155 // log10(M_halo) = 9.0

/* Globals for the halos
 */
float BOX_SIZE = 250.0;
float BOX_EPSILON = 0.01;

/* Globals for the tabulated HODs */
int NVOLUME_BINS = 0;
// TODO replace these with dynamic sized arrays
// These are the HOD in bins
double ncenr[MAXBINS][HALO_BINS], nsatr[MAXBINS][HALO_BINS], ncenb[MAXBINS][HALO_BINS], nsatb[MAXBINS][HALO_BINS], ncen[MAXBINS][HALO_BINS], nsat[MAXBINS][HALO_BINS], nhalo[MAXBINS][HALO_BINS];
double nhalo_int[MAXBINS][HALO_BINS]; // integer version of nhalo (no vmax weight)
float maglim[MAXBINS]; // the fainter limit of the mag bin;the brighter limit is this - 1.0. 
int color_sep[MAXBINS]; // 1 means do red/blue seperately, 0 means all together
float maxz[MAXBINS];  // the max redshift of the mag bin, calculated from the fainter mag limit
float volume[MAXBINS]; // volume of the mag bin in [Mpc/h]^3

// These are the HOD in thresholds
//double ncenr_th[MAXBINS][HALO_BINS], nsatr_th[MAXBINS][HALO_BINS], ncenb_th[MAXBINS][HALO_BINS], nsatb_th[MAXBINS][HALO_BINS], ncen_th[MAXBINS][HALO_BINS], nsat_th[MAXBINS][HALO_BINS], nhalo_th[MAXBINS][HALO_BINS];
//double nhalo_int_th[MAXBINS][HALO_BINS]; // integer version of nhalo (no vmax weight)

/* LHMR. Scatter is in in log space. */
double mean_cen[HALO_BINS], mean_cenr[HALO_BINS], mean_cenb[HALO_BINS], std_cen[HALO_BINS], std_cenr[HALO_BINS], std_cenb[HALO_BINS];
double logmean_cen[HALO_BINS], logmean_cenr[HALO_BINS], logmean_cenb[HALO_BINS];
double nhalo_r_tot[HALO_BINS], nhalo_b_tot[HALO_BINS], nhalo_tot[HALO_BINS];

float REDSHIFT = 0.0,
      CVIR_FAC = 1.0;

int RANDOM_SEED = 753;

/* local functions
 */
void nsat_smooth(double arr[MAXBINS][HALO_BINS]);
void nsat_extrapolate(double arr[MAXBINS][HALO_BINS]);
float NFW_position(float mass, float x[], struct drand48_data *rng);
float NFW_velocity(float mass, float v[], struct drand48_data *rng);
float NFW_density(float r, float rs, float ps);
float halo_concentration(float mass);
float N_sat(float m, int imag, SampleType type);
float N_cen(float m, int imag, SampleType type);
void boxwrap_galaxy(float xh[], float xg[]);
float rand_gaussian(struct drand48_data *rng);
float rand_f(struct drand48_data *rng);
void write_hodinner(int type);
void write_hod();
void write_hodfit();

/**
 * @brief Generates a deterministic seed from for a luminosity bin, and sample type.
 *
 * This function creates a unique seed for each combination of inputs, ensuring that
 * each mock has its own reproducible random number sequence.
 * So long as the number of L bins does not change, you will get the same results for a mock.
 *
 * @param imag The index of the luminosity bin.
 * @param type The sample type (e.g., ALL, QUIESCENT, STARFORMING).
 * @return An unsigned long integer to be used as a seed.
 */
unsigned long generate_mock_seed(int imag, SampleType type)
{
  unsigned long hash = 17;
  hash = hash * 31 + imag;
  hash = hash * 31 + static_cast<int>(type);
  return hash;
}

void write_hod() {
  write_hodinner(MSG_HOD);
}

void write_hodfit() {
  write_hodinner(MSG_HODFIT);
}

void write_hodinner(int type) {
  if (MSG_PIPE != NULL) {
    if (type == MSG_HOD) LOG_VERBOSE("Writing HOD to pipe\n");
    else LOG_VERBOSE("Writing HOD FIT to pipe\n");
    uint8_t resp_msg_type = type;
    uint8_t resp_data_type = TYPE_DOUBLE;
    uint32_t rows = MAX_HALO_IDX - MIN_HALO_IDX;
    uint32_t cols = NVOLUME_BINS * 7 + 1;
    uint32_t total = rows * cols;
    double *buffer = (double *)malloc(sizeof(double) * total);
    size_t idx = 0;
    for (int i = MIN_HALO_IDX; i < MAX_HALO_IDX; ++i) {
      double mass = i / 10.0;
      buffer[idx++] = mass;
      for (int j = 0; j < NVOLUME_BINS; ++j) {
        buffer[idx++] = ncenr[j][i];
        buffer[idx++] = nsatr[j][i];
        buffer[idx++] = ncenb[j][i];
        buffer[idx++] = nsatb[j][i];
        buffer[idx++] = ncen[j][i];
        buffer[idx++] = nsat[j][i];
        buffer[idx++] = nhalo_int[j][i];
      }
    }
    
    fwrite(&resp_msg_type, 1, 1, MSG_PIPE);
    fwrite(&resp_data_type, 1, 1, MSG_PIPE);
    fwrite(&total, sizeof(uint32_t), 1, MSG_PIPE);
    fwrite(buffer, sizeof(double), idx, MSG_PIPE);
    fflush(MSG_PIPE);
    free(buffer);
  }
  else {
    if (type == MSG_HOD) print_hod("hod.out");
    else print_hod("hodfit.out");
  }
}

void print_hod(const char* filename)
{
  if (SILENT) return;
  int i, j;

  // Print out the tabulated hods. We don't use this file for anything directly.
  FILE *fp = fopen(filename, "w");
  fprintf(fp, "# HODs for volume limited samples\n");
  fprintf(fp, "# Volume bins: ");
  for (i = 0; i < NVOLUME_BINS; ++i)
    fprintf(fp, "%.1f ", maglim[i]);
  fprintf(fp, "\n");
  fprintf(fp, "# Redshift limits: ");
  for (i = 0; i < NVOLUME_BINS; ++i)
    fprintf(fp, "%f ", maxz[i]);
  fprintf(fp, "\n");
  fprintf(fp, "# Volume limits: ");
  for (i = 0; i < NVOLUME_BINS; ++i)
    fprintf(fp, "%.1f ", volume[i]);
  fprintf(fp, "\n");
  for (i = MIN_HALO_IDX; i < MAX_HALO_IDX; ++i)
  {
    // Print off halo occupancy fractions in narrow mass bins for each galaxy mag bin
    // Format is <M_h> [<ncenr_i> <nsatr_i> <ncenb_i> <nsatb_i> <ncen_i> <nsat_i>  <nhalo_i> for each i mag bin]
    fprintf(fp, "%.2f", i / 10.0);
    for (j = 0; j < NVOLUME_BINS; ++j) 
    { 
      fprintf(fp, " %e %e %e %e %e %e %d", 
        ncenr[j][i],
        nsatr[j][i], 
        ncenb[j][i],
        nsatb[j][i],
        ncen[j][i],
        nsat[j][i],
        nhalo_int[j][i]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);

  LOG_INFO("Tabulated HODs written to %s\n", filename);
}

/* 
 * Given the group results, use a lookup table to get the mean Lsat50 values
 * for each halo as a function of central luminosity (or mass). 
 * The code reads in a pre-tabulated list matching halo mass to Lsat50 (from the C250 box),
 * and makes a spline from it for geting actual values for each central.
 */
void lsat_model()
{
  // Indexes of lsatb, nhb, lsatr, nhr correspond to log(L or M*) / 0.1
  FILE *fp;
  int i, j, k, n, nt, nhb[150], nhr[150], im, ihm, ix;
  float *mx, *lx, lsatb[150], lsatr[150], *m2x, x;
  float ratio[150];
  int NBINS2;
  float dpropx, M0_PROPX = 8.75, DM_PROPX = 0.5;
  double lsat;
  int nsat;
  int **nhrx = imatrix(0, 5, -20, 20);
  int **nhbx = imatrix(0, 5, -20, 20);
  float **lsatbx = matrix(0, 5, -20, 20);
  float **lsatrx = matrix(0, 5, -20, 20);

  int **nhrx2 = imatrix(0, 5, -20, 20);
  int **nhbx2 = imatrix(0, 5, -20, 20);
  float **lsatbx2 = matrix(0, 5, -20, 20);
  float **lsatrx2 = matrix(0, 5, -20, 20);

  // this is binning for the mocks (rnage = -2,2)
  NBINS2 = 10;
  dpropx = 0.2;

  /* for the actual SDSS fitting, the
   * number of propx bins is 11, going from -2 to 2,
   * with binwidth of 0.4
   */
  NBINS2 = 5;
  dpropx = 0.4;

  /* This is the binning for FIT3,
   * where we're doing 1 mass bin in c50/90
   */
  NBINS2 = 12;
  dpropx = 0.2;
  M0_PROPX = 0;
  DM_PROPX = 100; // evertyhing should be in one mass bin

  LOG_INFO("lsat_model> Applying Lsat model...\n");

  for (i = 0; i <= 5; ++i)
    for (j = -20; j <= 20; ++j)
      nhbx[i][j] = nhrx[i][j] = lsatbx[i][j] = lsatrx[i][j] = 0;

  for (i = 0; i <= 5; ++i)
    for (j = -20; j <= 20; ++j)
      nhbx2[i][j] = nhrx2[i][j] = lsatbx2[i][j] = lsatrx2[i][j] = 0;

  fp = openfile("lsat_lookup.dat");
  nt = filesize(fp);
  mx = vector(1, nt); // log10 mass
  lx = vector(1, nt); // log10 Lsat50
  m2x = vector(1, nt); // spline interpolation
  for (i = 1; i <= nt; ++i)
    fscanf(fp, "%f %f", &mx[i], &lx[i]);
  fclose(fp);

  spline(mx, lx, nt, 1.0E+30, 1.0E+30, m2x);

  for (i = 0; i < 150; ++i)
    nhr[i] = lsatr[i] = nhb[i] = lsatb[i] = 0;

  // Loop over galaxies and calculate the Lsat values, filling up nhr, nhb, lsatr, lsatb arrays.
  for (i = 0; i < NGAL; ++i)
  {
    if (GAL[i].psat > 0.5)
      continue;
    // Bin the Lsat values according to the luminosity/stellar mass
    im = (int)(GAL[i].loglum / 0.1 + 0.5);
    // get x, the lsat value for this galaxy according to the lookup
    splint(mx, lx, m2x, nt, log10(GAL[i].mass), &x); 
    if (x > 10.4) { // This is unrealisticly high so something went wrong with the interpolation in this case
      LOG_WARN("lsat_model> WARNING: Gal %d, log10(M)=%e has log10(Lsat)=%f. Setting to 10.4 instead.\n", i, log10(GAL[i].mass), x);
      x = 10.4;
    }
    if (GAL[i].color > 0.8)
    {
      nhr[im]++;
      lsatr[im] += pow(10.0, x);
    }
    else
    {
      nhb[im]++;
      lsatb[im] += pow(10.0, x);
    }
    // let's print this out just for kick
    // printf("LSAT %d %d %f %f %f %f\n",i,im,log10(GAL[i].mass),x,GAL[i].loglum,GAL[i].color);

    if (!SECOND_PARAMETER)
      continue;

    // if FIT3 binning, only do z<0.15
    // Presumably the Lsat data is only valid in this regime?
    if (GAL[i].redshift > 0.15)
      continue;

    // binning for lsat-vs-propx at fixed lum
    im = (int)((GAL[i].loglum - M0_PROPX) / DM_PROPX);
    // printf("BINX %d %f %f\n",im,GAL[i].loglum,(GAL[i].loglum-M0_PROPX-DM_PROPX/2)/DM_PROPX);
    if (im < 0 || im >= 5)
      continue;
    ix = (int)floor((GAL[i].propx + dpropx / 2) / dpropx); // for the mocks
    // does this work for negative bins?
    if (ix < -NBINS2 || ix > NBINS2)
      goto NEXT_PROP;
    if (GAL[i].color > 0.8)
    {
      nhrx[im][ix]++;
      lsatrx[im][ix] += pow(10.0, x);
    }
    else
    {
      nhbx[im][ix]++;
      lsatbx[im][ix] += pow(10.0, x);
    }

  // TODO Not sure if this works
  NEXT_PROP: 
    if (SECOND_PARAMETER == 1)
      continue;
    ix = (int)((GAL[i].propx2 - 0.1) * 5);
    if (ix < -10 || ix > 10)
      continue;
    if (GAL[i].color > 0.8)
    {
      nhrx2[im][ix]++;
      lsatrx2[im][ix] += pow(10.0, x);
    }
    else
    {
      nhbx2[im][ix]++;
      lsatbx2[im][ix] += pow(10.0, x);
    }
  }
  
  // Turn the running sums into means
  for (i=0; i<150; ++i)
  {
    lsatr[i] = lsatr[i] / (nhr[i] + 1.0E-20);
    lsatb[i] = lsatb[i] / (nhb[i] + 1.0E-20);
  }

  // output this to a pre-specified file
  // (plus we know the limits of the data)
  // i=88 means 10^8.8 solar masses
  // 91 to 113 for UniverseMachine, STELLAR_MASS too
  // 88 to 107 for C250
  // 88 to 119 for TNG
  // 88 to 107 for SDSS, see the lsat_sdss_con.dat file that this will be compare to. 
  // 88 to 107 for BGS Data TODO
  int i_start = 88;
  int i_end = 107;
  int count = (i_end - i_start + 1); 
  if (MSG_PIPE != NULL)
  {
    LOG_VERBOSE("Writing LSAT to pipe\n");
    uint8_t resp_msg_type = MSG_LSAT;
    uint8_t resp_data_type = TYPE_FLOAT;
    uint32_t resp_count = count*2;
    fwrite(&resp_msg_type, 1, 1, MSG_PIPE);
    fwrite(&resp_data_type, 1, 1, MSG_PIPE);
    fwrite(&resp_count, sizeof(uint32_t), 1, MSG_PIPE);
    fwrite(&lsatr[i_start], sizeof(float), count, MSG_PIPE);
    fwrite(&lsatb[i_start], sizeof(float), count, MSG_PIPE);
    fflush(MSG_PIPE);
  }
  else 
  {
    fp = fopen("lsat_groups.out", "w");
    for (i = i_start; i <= i_end; ++i) 
      // Format: log(L or M*) <Lsat_r> <Lsat_b> nhr nhb
      fprintf(fp, "%f %e %e %d %d\n", i / 10.0, lsatr[i], lsatb[i], nhr[i], nhb[i]);
    fclose(fp);
    LOG_VERBOSE("lsat_model> lsat_groups.out written\n");
  }

  if (SECOND_PARAMETER == 0)
    goto CLEANUP;
  
  // check if we're doing the FIT3 single bin
  if (DM_PROPX > 10)
  {
    fp = fopen("lsat_groups_propx_red.out", "w");
    // get the mean lsat to do internal normalization
    lsat = 0;
    nsat = 0;
    for (i = -NBINS2; i <= NBINS2; ++i)
    {
      lsat += lsatrx[0][i];
      nsat += nhrx[0][i];
    }
    for (i = -NBINS2; i <= NBINS2; ++i)
      fprintf(fp, "%.1f %e\n", i * dpropx, lsatrx[0][i] / (nhrx[0][i] + 1.0E-10) * nsat / lsat);
    fclose(fp);

    fp = fopen("lsat_groups_propx_blue.out", "w");
    // get the mean lsat to do internal normalization
    lsat = 0;
    nsat = 0;
    for (i = -NBINS2; i <= NBINS2; ++i)
    {
      lsat += lsatbx[0][i];
      nsat += nhbx[0][i];
    }
    for (i = -NBINS2; i <= NBINS2; ++i)
      fprintf(fp, "%.1f %e\n", i * dpropx, lsatbx[0][i] / (nhbx[0][i] + 1.0E-10) * nsat / lsat);
    fclose(fp);
    goto CLEANUP;
  }
  // print out the correlations with propx
  // at fixed stellar mass
  fp = fopen("lsat_groups_propx_red.out", "w");
  for (i = -NBINS2; i <= NBINS2; ++i)
  {
    fprintf(fp, "%.1f %e %e %e %e %e\n", i * dpropx, lsatrx[0][i] / (nhrx[0][i] + 1.0E-10),
            lsatrx[1][i] / (nhrx[1][i] + 1.0E-10),
            lsatrx[2][i] / (nhrx[2][i] + 1.0E-10),
            lsatrx[3][i] / (nhrx[3][i] + 1.0E-10),
            lsatrx[4][i] / (nhrx[4][i] + 1.0E-10));
  }
  fclose(fp);

  fp = fopen("lsat_groups_propx_blue.out", "w");
  for (i = -NBINS2; i <= NBINS2; ++i)
  {
    fprintf(fp, "%.1f %e %e %e %e %e\n", i * dpropx, lsatbx[0][i] / (nhbx[0][i] + 1.0E-10),
            lsatbx[1][i] / (nhbx[1][i] + 1.0E-10),
            lsatbx[2][i] / (nhbx[2][i] + 1.0E-10),
            lsatbx[3][i] / (nhbx[3][i] + 1.0E-10),
            lsatbx[4][i] / (nhbx[4][i] + 1.0E-10));
  }
  fclose(fp);

  if (SECOND_PARAMETER == 1)
    goto CLEANUP;

  // print out the correlations with propx
  // at fixed stellar mass
  fp = fopen("lsat_groups_propx2_red.out", "w");
  for (i = -10; i <= 10; ++i)
  {
    fprintf(fp, "%.1f %e %e %e %e %e\n", i / 5.0, lsatrx2[0][i] / (nhrx2[0][i] + 1.0E-10),
            lsatrx2[1][i] / (nhrx2[1][i] + 1.0E-10),
            lsatrx2[2][i] / (nhrx2[2][i] + 1.0E-10),
            lsatrx2[3][i] / (nhrx2[3][i] + 1.0E-10),
            lsatrx2[4][i] / (nhrx2[4][i] + 1.0E-10));
  }
  fclose(fp);

  fp = fopen("lsat_groups_propx2_blue.out", "w");
  for (i = -10; i <= 10; ++i)
  {
    fprintf(fp, "%.1f %e %e %e %e %e\n", i / 5.0, lsatbx2[0][i] / (nhbx2[0][i] + 1.0E-10),
            lsatbx2[1][i] / (nhbx2[1][i] + 1.0E-10),
            lsatbx2[2][i] / (nhbx2[2][i] + 1.0E-10),
            lsatbx2[3][i] / (nhbx2[3][i] + 1.0E-10),
            lsatbx2[4][i] / (nhbx2[4][i] + 1.0E-10));
  }
  fclose(fp);


  CLEANUP:
  // Free the allocated memory
  free_vector(mx, 1, nt);
  free_vector(lx, 1, nt);
  free_vector(m2x, 1, nt);
  free_imatrix(nhrx, 0, 5, -20, 20);
  free_imatrix(nhbx, 0, 5, -20, 20);
  free_matrix(lsatbx, 0, 5, -20, 20);
  free_matrix(lsatrx, 0, 5, -20, 20);
  free_imatrix(nhrx2, 0, 5, -20, 20);
  free_imatrix(nhbx2, 0, 5, -20, 20);
  free_matrix(lsatbx2, 0, 5, -20, 20);
  free_matrix(lsatrx2, 0, 5, -20, 20);

  return;
}

/* tabluate the HODs for magnitude bins (specified below)
 * for red/blue subsamples. This code assumes a flux-limited
 * sample, so it partitions the data into volume-limited
 * samples appropriate for each mag bin (once again, the redshift
 * limits are specific below.
 *
 * The results are left in globals, which are then called to
 * populate the simulations for measuring clustering.
 */
void tabulate_hods()
{
  FILE *fp, *bins_fp;
  int i, j, im, igrp, ibin;
  float mag;
  // these are for MXXL-BGS
  // float maxz[MAXBINS] = { 0.0633186, 0.098004, 0.150207, 0.227501, 0.340158 };
  // these are for MXXL-SDSS
  // float maxz[MAXBINS] = { 0.0292367, 0.0458043, 0.0713047, 0.110097, 0.16823 };
  // these are for SHAM-SDSS (r=17.5)
  //float maxz[MAXBINS] = {0.02586, 0.0406, 0.06336, 0.0981, 0.1504}; 
  // these are for TNG300
  // float maxz[NBINS] = { 0.0633186, 0.098004, 0.150207, 0.227501, 0.340158, 0.5 };
  double w0 = 1.0;
  int nbins = 0;

  if (NVOLUME_BINS <= 0) {
    LOG_VERBOSE("Reading Volume Bins...\n");

    bins_fp = fopen(VOLUME_BINS_FILE, "r");
    while (fscanf(bins_fp, "%f %f %f %d", &maglim[nbins], &maxz[nbins], &volume[nbins], &color_sep[nbins]) == 4)
    {
      nbins++;
      if (nbins >= MAXBINS)
      {
        LOG_ERROR("ERROR: More lines in volume bins file than maximum of %d\n", MAXBINS);
        fclose(bins_fp);
        exit(EINVAL);
      }
    }
    fclose(bins_fp);
    NVOLUME_BINS = nbins;
  }

  LOG_INFO("Tabulating HODs...\n");

  for (i = 0; i < NVOLUME_BINS; ++i)
    for (j = 0; j < HALO_BINS; ++j)
    {
      ncenr[i][j] = nsatr[i][j] = nhalo[i][j] = ncenb[i][j] = nsatb[i][j] = ncen[i][j] = nsat[i][j] = nhalo_int[i][j]= 0.0;
    }
  
  for (j=0; j < HALO_BINS; ++j)
  {
    mean_cen[j] = mean_cenr[j] = mean_cenb[j] = std_cen[j] = std_cenr[j] = std_cenb[j] = nhalo_tot[j] = nhalo_b_tot[j] = nhalo_r_tot[j] = 0.0;
    logmean_cen[j] = logmean_cenr[j] = logmean_cenb[j] = 0.0;
  }
  
  for (i = 0; i < NGAL; ++i)
  {
    // what is host halo mass?
    // im is the mass bin index
    if (GAL[i].psat > 0.5)
    {
      igrp = GAL[i].igrp;
      im = log10(GAL[igrp].mass) / 0.1; 
    }
    else
    {
      // For centrals, count this halo in nhalo for the relevant redshift bins
      im = (int)(log10(GAL[i].mass) / 0.1);
      for (j = 0; j < NVOLUME_BINS; ++j)
        if (GAL[i].redshift < maxz[j])
        {
          w0 = 1 / volume[j];
          if (GAL[i].vmax < volume[j])
            // TODO Should this ever happen?
            //LOG_WARN("WARNING: vmax (%f) < volume (%f) for galaxy index %d. Using 1/vmax weight.\n", GAL[i].vmax, volume[j], i);
            w0 = 1 / GAL[i].vmax;
            if (!isfinite(w0)) {
              LOG_ERROR("ERROR: w0 is not finite for galaxy index %d with vmax=%f\n", i, GAL[i].vmax);
              assert(isfinite(w0));
            }
            if (isnan(w0)) {
              LOG_ERROR("ERROR: w0 is NaN for galaxy index %d with vmax=%f\n", i, GAL[i].vmax);
              assert(!isnan(w0));
            }
          nhalo[j][im] += w0; // 1/vmax weight the halo count
          nhalo_int[j][im]++; // integer version of nhalo (no vmax weight)
        }
    }

    // check the magnitude of the galaxy for the luminosity bins
    mag = -2.5 * GAL[i].loglum + 4.65;
    ibin = (int)(fabs(mag) + maglim[0]); // So if first bin is -17, mag=-17.5, ibin=0
    if (STELLAR_MASS)
    {
      mag = GAL[i].loglum * 2;
      ibin = (int)(mag - maglim[0]); // They will both be positive
    }

    if (ibin < 0 || ibin >= NVOLUME_BINS)
      continue; // skip if not in the magnitude range
    if (GAL[i].redshift > maxz[ibin])
      continue; // skip if not in the redshift range; peculiar velocities can push galaxies out of the bin...
    if (im < 0 || im >= HALO_BINS)
      LOG_ERROR("err> %d %e\n", im, GAL[i].mass);

    // vmax-weight everything
    w0 = 1 / volume[ibin];
    if (GAL[i].vmax < volume[ibin])
      w0 = 1 / GAL[i].vmax;

    if (GAL[i].psat > 0.5)
      nsat[ibin][im] += w0;
    else
      ncen[ibin][im] += w0;

    if (GAL[i].color > 0.8) // red
    {
      if (GAL[i].psat > 0.5)
        nsatr[ibin][im] += w0;
      else
        ncenr[ibin][im] += w0;
    }
    else // blue
    {
      if (GAL[i].psat > 0.5)
        nsatb[ibin][im] += w0;
      else
        ncenb[ibin][im] += w0;
    }
  }

  // If nsatr/b > 0 but nhalo = 0, it was an edge case for a rare halo. Just set nhalo to nsatr/b there.
  // This happens when the central (and thus the halo) didn't go into a volume bin, but a satellite did.
  // Peculiar velocities at the z boundary can do this I think.
  for (i = 0; i < NVOLUME_BINS; ++i)
    for (j = 0; j < HALO_BINS; ++j) {
      if (nsatr[i][j] > 0 && nhalo[i][j] == 0) {
        fprintf(stderr,"WARNING: nhalo[%d][%d] = 0, setting to nsatr[%d][%d] = %e\n", i, j, i, j, nsatr[i][j]);
        nhalo[i][j] = nsatr[i][j];
      }
      if (nsatb[i][j] > 0 && nhalo[i][j] == 0) {
        fprintf(stderr,"WARNING: nhalo[%d][%d] = 0, setting to nsatb[%d][%d] = %e\n", i, j, i, j, nsatb[i][j]);
        nhalo[i][j] = nsatb[i][j];
      }
      assert(!isnan(ncenr[i][j]));
      assert(!isnan(nsatr[i][j]));
      assert(!isnan(ncenb[i][j]));
      assert(!isnan(nsatb[i][j]));
      assert(!isnan(nsat[i][j]));
      assert(!isnan(ncen[i][j]));
      assert(!isnan(nhalo[i][j]));
      assert(isfinite(ncenr[i][j]));
      assert(isfinite(nsatr[i][j]));
      assert(isfinite(ncenb[i][j]));
      assert(isfinite(nsatb[i][j]));
      assert(isfinite(nsat[i][j]));
      assert(isfinite(ncen[i][j]));
      assert(isfinite(nhalo[i][j]));
    }

  // *****************************************
  // Now LHMR, which we want to print out
  // *****************************************
  // Calculate the means
  // Means in linear space, but scatter in log space as is stndard for this.
  double lum_weighted = 0.0;
  for (i=0; i<NGAL; ++i)
  {
    if (GAL[i].psat > 0.5)
      continue;

    im = log10(GAL[i].mass) / 0.1;
    lum_weighted = GAL[i].lum * 1.0/GAL[i].vmax;

    nhalo_tot[im] += 1.0/GAL[i].vmax;
    mean_cen[im] += lum_weighted;

    if (GAL[i].color > 0.8) { 
      mean_cenr[im] += lum_weighted;
      nhalo_r_tot[im] += 1.0/GAL[i].vmax;
    }
    else {
      mean_cenb[im] += lum_weighted;
      nhalo_b_tot[im] += 1.0/GAL[i].vmax;
    }
  }
  for (i=0; i<HALO_BINS; ++i)
  {
    if (nhalo_tot[i] > 0)
    {
      mean_cenr[i] /= (nhalo_r_tot[i] + 1.0E-20);
      mean_cenb[i] /= (nhalo_b_tot[i] + 1.0E-20);
      mean_cen[i] /= (nhalo_tot[i] + 1.0E-20);
    }
    logmean_cen[i] = log10(mean_cen[i]);
    logmean_cenr[i] = log10(mean_cenr[i]);
    logmean_cenb[i] = log10(mean_cenb[i]);
  }

  // Calculate log-normal scatter 
  for (i = 0; i < NGAL; ++i)
  {
    if (GAL[i].psat > 0.5)
      continue;

    im = log10(GAL[i].mass) / 0.1;

    std_cen[im] += pow(GAL[i].loglum - logmean_cen[im], 2) * 1.0 / GAL[i].vmax;
    if (GAL[i].color > 0.8) {
      std_cenr[im] += pow(GAL[i].loglum - logmean_cenr[im], 2) * 1.0 / GAL[i].vmax;
    } else {
      std_cenb[im] += pow(GAL[i].loglum - logmean_cenb[im], 2) * 1.0 / GAL[i].vmax;
    }
  }
  for (i = 0; i < HALO_BINS; ++i)
  {
    if (nhalo_tot[i] > 0)
    {
      std_cenr[i] = sqrt(std_cenr[i] / (nhalo_r_tot[i] + 1.0E-20));
      std_cenb[i] = sqrt(std_cenb[i] / (nhalo_b_tot[i] + 1.0E-20));
      std_cen[i]  = sqrt(std_cen[i]  / (nhalo_tot[i] + 1.0E-20));
    }
  }

  // Print off the LHMR
  if (MSG_PIPE != NULL)
  {
    LOG_VERBOSE("Writing LHMR to pipe\n");
    uint8_t resp_msg_type = MSG_LHMR;
    uint8_t resp_data_type = TYPE_DOUBLE;
    uint32_t resp_count = (MAX_HALO_IDX-MIN_HALO_IDX) * 3 * 2; // 65 bins, all/red/blue, mean/scatter
    fwrite(&resp_msg_type, 1, 1, MSG_PIPE);
    fwrite(&resp_data_type, 1, 1, MSG_PIPE);
    fwrite(&resp_count, sizeof(uint32_t), 1, MSG_PIPE);
    fwrite(&mean_cen[MIN_HALO_IDX], sizeof(double), 65, MSG_PIPE);
    fwrite(&std_cen[MIN_HALO_IDX], sizeof(double), 65, MSG_PIPE);
    fwrite(&mean_cenr[MIN_HALO_IDX], sizeof(double), 65, MSG_PIPE);
    fwrite(&std_cenr[MIN_HALO_IDX], sizeof(double), 65, MSG_PIPE);
    fwrite(&mean_cenb[MIN_HALO_IDX], sizeof(double), 65, MSG_PIPE);
    fwrite(&std_cenb[MIN_HALO_IDX], sizeof(double), 65, MSG_PIPE);
    fflush(MSG_PIPE);
  }
  else if (!SILENT)
  {
    for (i = MIN_HALO_IDX; i < MAX_HALO_IDX; ++i)
    {
      // Format is: <log10(M_h)> <mean_cenr> <std_cenr> <mean_cenb> <std_cenb> <mean_cen> <std_cen>
      LOG_VERBOSE("LHMR> %.2f %e %e %e %e %e %e\n", i / 10.0,
              mean_cenr[i], std_cenr[i], mean_cenb[i], std_cenb[i],
              mean_cen[i], std_cen[i]);
    }
  }

  // Back to HODs: switch to log10 of the fractions for the rest of the code later
  for (i = MIN_HALO_IDX; i < HALO_BINS; ++i)
  {
    for (j = 0; j < NVOLUME_BINS; ++j)
    {
      ncenr[j][i] = log10(ncenr[j][i] / (nhalo[j][i] + 1.0E-20) + 1.0E-20);
      nsatr[j][i] = log10(nsatr[j][i] / (nhalo[j][i] + 1.0E-20) + 1.0E-20);
      ncenb[j][i] = log10(ncenb[j][i] / (nhalo[j][i] + 1.0E-20) + 1.0E-20);
      nsatb[j][i] = log10(nsatb[j][i] / (nhalo[j][i] + 1.0E-20) + 1.0E-20);
      ncen[j][i] = log10(ncen[j][i] / (nhalo[j][i] + 1.0E-20) + 1.0E-20);
      nsat[j][i] = log10(nsat[j][i] / (nhalo[j][i] + 1.0E-20) + 1.0E-20);
    }
  }

  write_hod();

  // Smooth and extrapolate the satellite HODs to handle gaps in the data and the high-mass end where there is little data.
  //nsat_smooth(nsatr);
  //nsat_smooth(nsatb);
  //nsat_smooth(nsat);
  nsat_extrapolate(nsatr);
  nsat_extrapolate(nsatb);
  nsat_extrapolate(nsat);

  write_hodfit();
}

/* Do the same as above, but now giving
 * each halo an actual value of Lsat from the simulation.
 * Thus, we test if scatter (and fraction of Lsat=0's)
 * makes any difference.
 * --------------------------------
 * RESULTS: it makes no difference.
 */
void lsat_model_scatter()
{
  FILE *fp;
  int i, j, k, id, n, nt, nhb[150], nhr[150], im, ihm;
  float *mx, *lx, *m2x, x, lsat;
  double lsatb[150], lsatr[150];
  int *indx;
  float *mvir;

  LOG_INFO("lsat_model_scatter> start\n");

  indx = ivector(1, NHALO);
  mvir = vector(1, NHALO);
  for (i = 1; i <= NHALO; ++i)
  {
    indx[i] = i;
    mvir[i] = HALO[i].mass;
  }
  sort2(NHALO, mvir, indx);
  LOG_INFO("lsat_model_scatter> done sorting\n");

  for (i = 0; i < 150; ++i)
    nhr[i] = lsatr[i] = nhb[i] = lsatb[i] = 0;

  for (i = 0; i < NGAL; ++i)
  {
    if (GAL[i].psat > 0.5)
      continue;
    im = (int)(GAL[i].loglum / 0.1 + 0.5);
    id = search(NHALO, mvir, GAL[i].mass);
    lsat = HALO[indx[id]].lsat;
    // printf("LSAT %d %e %e %e\n",id,GAL[i].mass, HALO[indx[id]].mass, lsat);

    if (GAL[i].color > 0.8)
    {
      nhr[im]++;
      lsatr[im] += lsat;
    }
    else
    {
      nhb[im]++;
      lsatb[im] += lsat;
    }
  }
  LOG_INFO("lsat_model_scatter> done with calculations\n");

  // output this to a pre-specified file
  // (plus we knoe the limits of the data)
  // TODO this OVERWRITES the lsat_groups.out file the other method makes!
  fp = fopen("lsat_groups2.out", "w");
  for (i = 88; i <= 106; ++i) // i=88 means 10^8.8 solar masses
    fprintf(fp, "%e %e %e\n", i / 10.0, log10(lsatr[i] / nhr[i]), log10(lsatb[i] / nhb[i]));
  //for (i = 88; i <= 107; ++i) // C250
  //  fprintf(fp, "%e %e %e\n", i / 10.0, log10(lsatr[i] / nhr[i]), log10(lsatb[i] / nhb[i]));
  fclose(fp);

  LOG_INFO("lsat_model_scatter> lsat_groups2.out written\n");

  free_vector(mvir, 1, NHALO);
  free_ivector(indx, 1, NHALO);
  return;
}

void nsat_smooth(double arr[MAXBINS][HALO_BINS])
{
  // For places in the array where nhalo_int is < 5, smooth that value with the neighboring two.
  int imag, i;
  for (imag = 0; imag < NVOLUME_BINS; ++imag) {
    for (i = MIN_HALO_IDX + 1; i < MAX_HALO_IDX - 2; ++i) {
      if (nhalo_int[imag][i] < 5) {
        // Average with neighbors
        arr[imag][i] = (arr[imag][i - 1] + arr[imag][i] + arr[imag][i + 1]) / 3.0;
      }
    }
  }
}

void nsat_extrapolate(double arr[MAXBINS][HALO_BINS])
{  
  // Extend the satellite occupation function for high-mass halos
  // using a linear fit.
  
  int istart, iend, badpoints, idx;
  double bfit, mfit, mag;

  for (int imag = 0; imag < NVOLUME_BINS; ++imag)
  {
    iend = 0;
    mag = maglim[imag];

    // Start in a known good place with lots of data for each magbin.
    // Search up from there (in halo mass) until we run low on data.
    // Use the 10 data points leading up to that to fit the line.
    if (fabs(mag) < 16.1) {
      istart = 118;
    } else if (fabs(mag) < 17.1) {
      istart = 120;
    } else if (fabs(mag) < 18.1) {
      istart = 126;
    } else if (fabs(mag) < 19.1) {
      istart = 129;
    } else if (fabs(mag) < 20.1) {
      istart = 131;
    } else if (fabs(mag) < 21.1) {
      istart = 133;
    } else { // Brightest bin (-22 to -23)
      istart = 135;
    }
    for (int i = istart; i < MAX_HALO_IDX; ++i)
    {
      if (nhalo_int[imag][i] < 10) // need 10 halos to be considered sufficient data
      {
        iend++;
        if (iend == 3)
        {
          iend = i - 1;
          break;
        }
      }
    }

    if (iend < 3 || iend == MAX_HALO_IDX-1) {
      // We had plenty of data everywhere
      LOG_INFO("No HOD extrapolation needed for imag=%d\n", imag);
      continue;
    }

    istart = iend - 10;

    // Fit a line y = mfit * x + bfit to the data in arr[imag][i] for i = istart to iend
    double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
    int nfit = 0;
    for (int i = istart; i <= iend; ++i) {
      if (arr[imag][i] > -5) {
        double x = i / 10.0;
        double y = arr[imag][i];
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
        nfit++;
      }
    }
    if (nfit > 1) {
      double denom = nfit * sum_xx - sum_x * sum_x;
      if (fabs(denom) > 1e-10) {
        mfit = (nfit * sum_xy - sum_x * sum_y) / denom;
        bfit = (sum_y - mfit * sum_x) / nfit;
      } else {
        LOG_WARN("WARNING: Denominator too small for imag=%d, using default slope and intercept\n", imag);
        mfit = 1.0;
        bfit = -14.0;
      }
    } else {
        LOG_WARN("WARNING: Not enough points to fit for imag=%d, using default slope and intercept\n", imag);
        mfit = 1.0;
        bfit = -14.0;
    }

    //LOG_INFO("Extrapolate: imag=%d, istart=%d, iend=%d, mfit=%f, bfit=%f\n", imag, istart, iend, mfit, bfit);

    for (int i = iend; i < MAX_HALO_IDX; ++i) {
      arr[imag][i] = mfit * (i / 10.0) + bfit;
    }
  }
}

/*
 * Read in the mock halo data from a file.
 * The file should have the following format:
 * mass x y z vx vy vz lsat
 */
void prepare_halos() {
  if (HALO != nullptr) {
    return;
  }

  LOG_INFO("popsim> reading mock halo data...\n");
  FILE *fp = fopen(MOCK_FILE, "r");
  //fp = fopen("/export/sirocco1/tinker/SIMULATIONS/BOLSHOI/hosthalo_z0.0_M1e10.dat", "r");
  //fp = fopen("/export/sirocco2/tinker/SIMULATIONS/C250_2560/hosthalo_z0.0_M1e10_Lsat.dat", "r");
  if (!fp)
  {
    LOG_ERROR("popsim> could not open mock halo file\n");
    fflush(stderr);
    exit(ENOENT);
  }

  NHALO = filesize(fp);
  LOG_INFO("popsim> NHALO=%d\n", NHALO);
  HALO = (halo *) calloc(NHALO, sizeof(halo));
  for (int i = 0; i < NHALO; ++i)
  {
    fscanf(fp, "%f %f %f %f %f %f %f %f", &HALO[i].mass,
            &HALO[i].x, &HALO[i].y, &HALO[i].z,
            &HALO[i].vx, &HALO[i].vy, &HALO[i].vz, &HALO[i].lsat);
  }
  fclose(fp);
  LOG_INFO("popsim> done reading halo data [%d].\n", NHALO);
}

/*
 * Population a simulation's halo catalog using the tabulated HODs from tabulate_hods().
 * Uses linear interpolate in log-log space from the tabulated HOD values.
 */
void populate_simulation_omp(int imag, SampleType type)
{
  int i;
  FILE *outf;
  char fname[1000];
  int j, nsat_rand, imag_offset, imag_mult, istart, iend, mag;
  float nsat_calc, ncen_calc, mass, xg[3], vg[3], xh[3], logm, bfit;
  double r;
  int warned = 0;

  // If this combination of imag and type isn't needed, skip it
  assert (type == QUIESCENT || type == STARFORMING || type == ALL);
  assert (imag >= 0 || imag == -1);
  assert (imag < NVOLUME_BINS);
  if (color_sep[imag] == 0 && type != ALL)
    return;
  if (color_sep[imag] > 0 && type == ALL)
    return;

  // Setup a RNG at the same place for every time we want to build a mock
  // Thus if the group catalog state is the same, the mock will be the same
  struct drand48_data rng;
  srand48_r(generate_mock_seed(imag, type), &rng);

  imag_offset = (int)fabs(maglim[0]);
  imag_mult = 1;
  if (STELLAR_MASS)
  {
    imag_offset = MIN_HALO_IDX;
    imag_mult = 5;
  }
  mag = imag * imag_mult + imag_offset;

  LOG_VERBOSE("popsim> starting population for imag=%d, type=%d, mag=%d\n", imag, type, mag);

  switch (type) 
  {
    case QUIESCENT:
      sprintf(fname, "mock_red_M%d.dat", mag);
      break;
    case STARFORMING:
      sprintf(fname, "mock_blue_M%d.dat", mag);
      break;
    case ALL:
      sprintf(fname, "mock_all_M%d.dat", mag);
      break;
  }
  outf = fopen(fname, "w");

  // Loop through halos and populate with galaxies
  for (i = 0; i < NHALO; ++i)
  {
    mass = HALO[i].mass;
    logm = log10(mass);
    ncen_calc = N_cen(mass, imag, type);
    r = rand_f(&rng);
    if (r < ncen_calc)
    {
      fprintf(outf, "%.5f %.5f %.5f %f %f %f %d %f\n",
              HALO[i].x, HALO[i].y, HALO[i].z,
              HALO[i].vx, HALO[i].vy, HALO[i].vz, 0, logm);
    }
    // We can still add satellites even if there is no central becaues we're just doing this in a single luminosity bin

    // Assume poisson variance in the number of satellites
    // TODO could update this to use our measured variance?
    nsat_calc = N_sat(mass, imag, type);
    nsat_rand = poisson_deviate(nsat_calc, &rng);

    // For a really bad set of parameters, we can get a huge number of satellites for some halos.
    // Cap it so we don't print off a 10 Terabyte file! Any MCMC or whatever will move on hopefully.
    // TODO early abort program instead
    if (nsat_rand > MAX_SATELLITES) {
      if (!warned) {
        LOG_WARN("popsim> WARNING: giving %d sats for halo %d\n", nsat_rand, i);
        warned = 1;
      }
      nsat_rand = MAX_SATELLITES;
    }

    for (j = 1; j <= nsat_rand; ++j)
    {
      NFW_position(mass, xg, &rng);
      NFW_velocity(mass, vg, &rng);
      xh[0] = HALO[i].x;
      xh[1] = HALO[i].y;
      xh[2] = HALO[i].z;
      boxwrap_galaxy(xh, xg);
      if (isnan(xg[0] + xg[1] + xg[2]))
        continue;
      fprintf(outf, "%.5f %.5f %.5f %f %f %f %d %f\n",
              xg[0], xg[1], xg[2],
              HALO[i].vx + vg[0], HALO[i].vy + vg[1], HALO[i].vz + vg[2],
              1, logm);
    }
  }
  
  fclose(outf);
}

void boxwrap_galaxy(float xh[], float xg[])
{
  int i;
  for (i = 0; i < 3; ++i)
  {
    xg[i] += xh[i];
    if (xg[i] > BOX_SIZE)
      xg[i] -= BOX_SIZE;
    if (xg[i] < 0)
      xg[i] += BOX_SIZE;
    if (xg[i] > BOX_SIZE - BOX_EPSILON)
      xg[i] = BOX_SIZE - BOX_EPSILON;
  }
}

/**
 * Return the number of central galaxies for a halo of mass m
 * in the magnitude bin imag and color. Uses the tabulated HOD
 * and linearly interpolates in log10(M_h)-log10(N_cen) space.
 */
float N_cen(float m, int imag, SampleType type)
{
  int im;
  float x0, y0, y1, yp, logm;
  logm = log10(m) / 0.1;
  im = (int)logm;
  x0 = im;
  //x1 = im + 1;

  if (im < MIN_HALO_IDX || im >= HALO_BINS - 1)
    return 0;
  if (im >= MAX_HALO_IDX) {
    im = MAX_HALO_IDX - 1;
  }

  switch (type)
  {
    case QUIESCENT:
      y0 = ncenr[imag][im];
      y1 = ncenr[imag][im + 1];
      break;
    case STARFORMING:
      y0 = ncenb[imag][im];
      y1 = ncenb[imag][im + 1];
      break;
    case ALL:
      y0 = ncen[imag][im];
      y1 = ncen[imag][im + 1];
      break;
    default:
      LOG_ERROR("ERROR: Unknown sample type %d in N_cen\n", type);
      exit(EINVAL);
  }

  //yp = y0 + ((y1 - y0) / (x1 - x0)) * (logm - x0);
  yp = y0 + ((y1 - y0)) * (logm - x0);

  if (yp <= -10)
    return 0;
  return pow(10.0, yp);
}

/**
 * Return the number of satellites for a halo of mass m in the given 
 * magnitude bin and color. Uses the tabulated HOD.
 */
float N_sat(float m, int imag, SampleType type)
{
  int im;
  float x0, y0, y1, yp, logm;
  logm = log10(m) / 0.1;
  im = (int)logm;
  x0 = im;
  //x1 = im + 1;

  if (im < MIN_HALO_IDX || im >= HALO_BINS - 1)
    return 0;
  if (im >= MAX_HALO_IDX) {
    im = MAX_HALO_IDX - 1;
  }

  switch (type)
  {
    case QUIESCENT:
      y0 = nsatr[imag][im];
      y1 = nsatr[imag][im + 1];
      break;
    case STARFORMING:
      y0 = nsatb[imag][im];
      y1 = nsatb[imag][im + 1];
      break;
    case ALL:
      y0 = nsat[imag][im];
      y1 = nsat[imag][im + 1];
      break;
  }

  //yp = y0 + ((y1 - y0) / (x1 - x0)) * (logm - x0);
  yp = y0 + ((y1 - y0)) * (logm - x0);

  if (yp <= -10)
    return 0;
  return pow(10.0, yp);
}

//===========================================================================
//=  Function to generate Poisson distributed random variables              =
//=    - Input:  Mean value of distribution                                 =
//=    - Output: Returns with Poisson distributed random variable           =
//===========================================================================
int poisson_deviate(float mean, struct drand48_data *rng)
{
  // Efficient Poisson deviate using inversion for small mean, normal approx for large mean
  if (mean <= 0)
    return 0;
  if (mean < 30.0) {
    // Inversion method
    float L = expf(-mean);
    float p = 1.0f;
    int k = 0;
    do {
      p *= rand_f(rng);
      k++;
    } while (p > L);
    return k - 1;
  } else {
    // Normal approximation for large mean
    float g = sqrtf(mean);
    float val = mean + g * rand_gaussian(rng);
    if (val < 0) val = 0;
    return (int)(val + 0.5f);
  }
}


float rand_gaussian(struct drand48_data *rng)
{
  // Use Box-Muller transform with drand48_r for thread safety
  double u, v, s;

  do {
    drand48_r(rng, &u);
    drand48_r(rng, &v);
    u = 2.0 * u - 1.0;
    v = 2.0 * v - 1.0;
    s = u * u + v * v;
  } while (s >= 1.0 || s == 0.0);

  s = sqrt(-2.0 * log(s) / s);

  return (float)(u * s);
}


float rand_f(struct drand48_data *rng)
{
  int tnum = omp_get_thread_num();
  double r;
  drand48_r(rng, &r);
  return (float)r;
}

/* Randomy generates a position away from the origin with
 * a probability given by the NFW profile for a halo of the input
 * mass (and including the CVIR_FAC)
 */
float NFW_position(float mass, float x[], struct drand48_data *rng)
{
  float r, pr, max_p, costheta, sintheta, phi1, signs, rvir, rs, cvir, mfac = 1;
  double rr;
  cvir = halo_concentration(mass) * CVIR_FAC;
  rvir = pow(3 * mass / (4 * DELTA_HALO * PI * RHO_CRIT * OMEGA_M), 1.0 / 3.0);
  rs = rvir / cvir;
  max_p = NFW_density(rs, rs, 1.0) * rs * rs * 4.0 * PI;

  for (;;)
  {
    r = rand_f(rng) * rvir;
    pr = NFW_density(r, rs, 1.0) * r * r * 4.0 * PI / max_p;

    if (rand_f(rng) <= pr)
    {
      costheta = 2. * (rand_f(rng) - .5);
      sintheta = sqrt(1. - costheta * costheta);
      signs = 2. * (rand_f(rng) - .5);
      costheta = signs * costheta / fabs(signs);
      phi1 = 2.0 * PI * rand_f(rng);

      x[0] = r * sintheta * cos(phi1);
      x[1] = r * sintheta * sin(phi1);
      x[2] = r * costheta;
      return r;
    }
  }
}

/* This is the NFW density profile
 */
float NFW_density(float r, float rs, float ps)
{
  return (ps * rs / (r * (1 + r / rs) * (1 + r / rs)));
}

/* This sets the velocity to be isotropic Gaussian.
 */
float NFW_velocity(float mass, float v[], struct drand48_data *rng)
{
  // static float fac = -1;
  float sigv, vbias = 1, mfac = 1;
  int i;
  float fac;

  fac = sqrt(4.499E-48) * pow(4 * DELTA_HALO * PI * OMEGA_M * RHO_CRIT / 3, 1.0 / 6.0) * 3.09E19 * sqrt(1 + REDSHIFT);
  sigv = fac * pow(mass, 1.0 / 3.0) / ROOT2;
  for (i = 0; i < 3; ++i)
    v[i] = rand_gaussian(rng) * sigv;
  return (0);
}

float halo_concentration(float mass)
{
  return 10 * pow(mass / 1.0E14, -0.11);
}

