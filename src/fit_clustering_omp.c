// Initialization //

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <sys/time.h>
#include "nrutil.h"
#include "kdtree.h"
#include "groups.h"
#include <errno.h>

// Definitions
#define MAXBINS 10 // This is the number of magnitude bins we use. // TNG has 6 bins
#define NRANDOM 1000000
#define MAX_SATELLITES 1000

/* Global for the random numbers
 */
float UNIFORM_RANDOM[NRANDOM];
float GAUSSIAN_RANDOM[NRANDOM];
int IRAN_CURRENT[100];

/* Globals for the halos
 */
struct halo *HALO;
int NHALO;
float BOX_SIZE = 250.0;
float BOX_EPSILON = 0.01;

/* Globals for the tabulated HODs
 */
double ncenr[MAXBINS][200], nsatr[MAXBINS][200], ncenb[MAXBINS][200], nsatb[MAXBINS][200], nhalo[MAXBINS][200];
int NVOLUME_BINS;
float maglim[MAXBINS];

float REDSHIFT = 0.0,
      CVIR_FAC = 1.0;

/* local functions
 */
float NFW_position(float mass, float x[], int thisTask);
float NFW_velocity(float mass, float v[], int thisTask);
float NFW_density(float r, float rs, float ps);
int poisson_deviate(float nave, int thisTask);
float halo_concentration(float mass);
float N_sat(float m, int imag, int blue_flag);
float N_cen(float m, int imag, int blue_flag);
void boxwrap_galaxy(float xh[], float xg[]);
float tabulated_gaussian_random(int thisTask);
float tabulated_uniform_random(int thisTask);

/* Given the group results, calculate the mean
 * expected Lsat50 values as a function of central
 * luminosity (or mass). The code reads in a pre-tabulated
 * list matching halo mass to Lsat50 (from the C250 box).
 */
void lsat_model()
{
  // Indexes of lsatb, nhb, lsatr, nhr correspond to log(L or M*) / 0.1
  FILE *fp;
  int i, j, k, n, nt, nhb[150], nhr[150], im, ihm, ix;
  float *mx, *lx, lsatb[150], lsatr[150], *m2x, x;
  int **nhrx, **nhbx;
  float **lsatbx, **lsatrx;
  int **nhrx2, **nhbx2;
  float **lsatbx2, **lsatrx2;
  int NBINS2;
  float dpropx, M0_PROPX = 8.75, DM_PROPX = 0.5;
  double lsat;
  int nsat;
  nhrx = imatrix(0, 5, -20, 20);
  nhbx = imatrix(0, 5, -20, 20);
  lsatbx = matrix(0, 5, -20, 20);
  lsatrx = matrix(0, 5, -20, 20);

  nhrx2 = imatrix(0, 5, -20, 20);
  nhbx2 = imatrix(0, 5, -20, 20);
  lsatbx2 = matrix(0, 5, -20, 20);
  lsatrx2 = matrix(0, 5, -20, 20);

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

  if (!SILENT)
    fprintf(stderr, "lsat_model> Applying Lsat model...\n");

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

  for (i = 0; i < NGAL; ++i)
  {
    if (GAL[i].psat > 0.5)
      continue;
    // Bin the Lsat values according to the luminosity/stellar mass
    im = (int)(log10(GAL[i].lum) / 0.1 + 0.5);
    // get x, the lsat value for this galaxy according to the lookup
    splint(mx, lx, m2x, nt, log10(GAL[i].mass), &x); 
    if (x > 10.4) { // This is unrealisticly high so something went wrong with the interpolation in this case
      fprintf(stderr, "lsat_model> WARNING: Gal %d, log10(M)=%e has log10(Lsat)=%f\n", i, log10(GAL[i].mass), x);
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
    // printf("LSAT %d %d %f %f %f %f\n",i,im,log10(GAL[i].mass),x,log10(GAL[i].lum),GAL[i].color);

    if (!SECOND_PARAMETER)
      continue;

    // if FIT3 binning, only do z<0.15
    // Presumably the Lsat data is only valid in this regime?
    if (GAL[i].redshift > 0.15)
      continue;

    // binning for lsat-vs-propx at fixed lum
    im = (int)((log10(GAL[i].lum) - M0_PROPX) / DM_PROPX);
    // printf("BINX %d %f %f\n",im,log10(GAL[i].lum),(log10(GAL[i].lum)-M0_PROPX-DM_PROPX/2)/DM_PROPX);
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

  // output this to a pre-specified file
  // (plus we know the limits of the data)
  // Format: log(L or M*) log(<Lsat_r>)  log(<Lsat_b>)
  fp = fopen("lsat_groups.out", "w");
  if (STELLAR_MASS)
  {
    for (i = 91; i <= 113; ++i) // UniverseMachine ?
      fprintf(fp, "%e %e %e %e %d %e %d\n", i / 10.0, log10(lsatr[i] / nhr[i]), log10(lsatb[i] / nhb[i]), lsatr[i], nhr[i], lsatb[i], nhb[i]);
    fclose(fp);
  }
  else
  {
    for (i = 88; i <= 107; ++i) // C250  // i=88 means 10^8.8 solar masses
      // for(i=88;i<=119;++i) // TNG
      fprintf(fp, "%e %e %e %e %d %e %d\n", i / 10.0, log10(lsatr[i] / nhr[i]), log10(lsatb[i] / nhb[i]), lsatr[i], nhr[i], lsatb[i], nhb[i]);
    fclose(fp);
  }
  if (!SILENT) fprintf(stderr, "lsat_model> lsat_groups.out written\n");

  if (SECOND_PARAMETER == 0)
    return;

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
    return;
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
    return;

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
  // TODO switch to giving hte fluxlimit and calculate these.
  // these are for MXXL-BGS
  // float maxz[MAXBINS] = { 0.0633186, 0.098004, 0.150207, 0.227501, 0.340158 };
  // these are for MXXL-SDSS
  // float maxz[MAXBINS] = { 0.0292367, 0.0458043, 0.0713047, 0.110097, 0.16823 };
  // these are for SHAM-SDSS (r=17.5)
  //float maxz[MAXBINS] = {0.02586, 0.0406, 0.06336, 0.0981, 0.1504}; // TODO check these...
  float maxz[MAXBINS]; 
  float volume[MAXBINS]; // volume of the mag bin in [Mpc/h]^3
  // these are for TNG300
  // float maxz[NBINS] = { 0.0633186, 0.098004, 0.150207, 0.227501, 0.340158, 0.5 };
  float w0 = 1.0;
  int nbins = 0;

  if (!SILENT)
    fprintf(stderr, "Reading Volume Bins...\n");

  bins_fp = fopen(VOLUME_BINS_FILE, "r");
  while (fscanf(bins_fp, "%f %f %f", &maglim[nbins], &maxz[nbins], &volume[nbins]) == 3) 
  {
    nbins++;
    if (nbins >= MAXBINS)
    {
      fprintf(stderr, "ERROR: More lines in volume bins file than maximum of %d\n", MAXBINS);
      fclose(bins_fp);
      exit(EINVAL);
    }
  }
  fclose(bins_fp);
  NVOLUME_BINS = nbins;
  
  if (!SILENT)
    fprintf(stderr, "Tabulating HODs...\n");

  for (i = 0; i < NVOLUME_BINS; ++i)
  {
    // Initialize the arrays that hold the HODs to 0
    for (j = 0; j < 200; ++j)
      ncenr[i][j] = nsatr[i][j] = nhalo[i][j] = ncenb[i][j] = nsatb[i][j] = 0;
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
      im = log10(GAL[i].mass) / 0.1;
      for (j = 0; j < NVOLUME_BINS; ++j)
        if (GAL[i].redshift < maxz[j])
        {
          w0 = 1 / volume[j];
          if (GAL[i].vmax < volume[j])
            w0 = 1 / GAL[i].vmax;
            if (!isfinite(w0)) {
            fprintf(stderr, "ERROR: w0 is not finite for galaxy index %d with vmax=%f\n", i, GAL[i].vmax);
            assert(isfinite(w0));
            }
            if (isnan(w0)) {
            fprintf(stderr, "ERROR: w0 is NaN for galaxy index %d with vmax=%f\n", i, GAL[i].vmax);
            assert(!isnan(w0));
            }
          nhalo[j][im] += w0; // 1/vmax weight the halo count
        }
    }

    // check the magnitude of the galaxy
    mag = -2.5 * log10(GAL[i].lum) + 4.65;
    ibin = (int)(fabs(mag) + maglim[0]); // So if first bin is -17, mag=-17.5, ibin=0
    if (STELLAR_MASS)
    {
      mag = log10(GAL[i].lum) * 2;
      ibin = (int)(mag - maglim[0]); // They will both be positive
    }

    if (ibin < 0 || ibin >= NVOLUME_BINS)
      continue; // skip if not in the magnitude range
    if (GAL[i].redshift > maxz[ibin])
      continue; // skip if not in the redshift range; peculiar velocities can push galaxies out of the bin...
    if (im < 0 || im >= 200)
      fprintf(stderr, "err> %d %e\n", im, GAL[i].mass);

    // vmax-weight everything
    w0 = 1 / volume[ibin];
    if (GAL[i].vmax < volume[ibin])
      w0 = 1 / GAL[i].vmax;

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
    for (j = 0; j < 200; ++j) {
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
      assert(!isnan(nhalo[i][j]));
      assert(isfinite(ncenr[i][j]));
      assert(isfinite(nsatr[i][j]));
      assert(isfinite(ncenb[i][j]));
      assert(isfinite(nsatb[i][j]));
      assert(isfinite(nhalo[i][j]));
    }

  // Print out the tabulated hods
  fp = fopen("hod.out", "w");
  // Print a header with column names
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
  // 100, 155 wa previous
  for (i = 90; i < 155; ++i)
  {
    // Print off halo occupancy fractions in narrow mass bins for each galaxy mag bin
    // Format is <M_h> [<ncenr_i> <nsatr_i> <ncenb_i> <nsatb_i> <nhalo_i> for each i mag bin]
    fprintf(fp, "%.2f", i / 10.0);
    for (j = 0; j < NVOLUME_BINS; ++j) 
    { 
      fprintf(fp, " %e %e %e %e %e", 
        ncenr[j][i] * 1. / (nhalo[j][i] + 1.0E-20),
        nsatr[j][i] * 1. / (nhalo[j][i] + 1.0E-20), 
        ncenb[j][i] * 1. / (nhalo[j][i] + 1.0E-20),
        nsatb[j][i] * 1. / (nhalo[j][i] + 1.0E-20),
        nhalo[j][i]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
  
  // Switch to log10 of the fractions
  for (i = 90; i < 200; ++i)
  {
    for (j = 0; j < NVOLUME_BINS; ++j)
    {
      ncenr[j][i] = log10(ncenr[j][i] * 1. / (nhalo[j][i] + 1.0E-20) + 1.0E-10);
      nsatr[j][i] = log10(nsatr[j][i] * 1. / (nhalo[j][i] + 1.0E-20) + 1.0E-10);
      ncenb[j][i] = log10(ncenb[j][i] * 1. / (nhalo[j][i] + 1.0E-20) + 1.0E-10);
      nsatb[j][i] = log10(nsatb[j][i] * 1. / (nhalo[j][i] + 1.0E-20) + 1.0E-10);
    }
  }
  
  if (!SILENT)
    fprintf(stderr, "Tabulated HODs written to hod.out\n");
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

  if (!SILENT) fprintf(stderr, "lsat_model_scatter> start\n");

  indx = ivector(1, NHALO);
  mvir = vector(1, NHALO);
  for (i = 1; i <= NHALO; ++i)
  {
    indx[i] = i;
    mvir[i] = HALO[i].mass;
  }
  sort2(NHALO, mvir, indx);
  if (!SILENT) fprintf(stderr, "lsat_model_scatter> done sorting\n");

  for (i = 0; i < 150; ++i)
    nhr[i] = lsatr[i] = nhb[i] = lsatb[i] = 0;

  for (i = 0; i < NGAL; ++i)
  {
    if (GAL[i].psat > 0.5)
      continue;
    im = (int)(log10(GAL[i].lum) / 0.1 + 0.5);
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
  if (!SILENT) fprintf(stderr, "lsat_model_scatter> done with calculations\n");

  // output this to a pre-specified file
  // (plus we knoe the limits of the data)
  // TODO this OVERWRITES the lsat_groups.out file the other method makes!
  fp = fopen("lsat_groups2.out", "w");
  for (i = 88; i <= 106; ++i) // i=88 means 10^8.8 solar masses
    fprintf(fp, "%e %e %e\n", i / 10.0, log10(lsatr[i] / nhr[i]), log10(lsatb[i] / nhb[i]));
  //for (i = 88; i <= 107; ++i) // C250
  //  fprintf(fp, "%e %e %e\n", i / 10.0, log10(lsatr[i] / nhr[i]), log10(lsatb[i] / nhb[i]));
  fclose(fp);

  if (!SILENT) fprintf(stderr, "lsat_model_scatter> lsat_groups2.out written\n");

  return;
}

/*
 * Population a simulation's halo catalog using the tabulated HODs from tabulate_hods().
 */
void populate_simulation_omp(int imag, int blue_flag, int thisTask)
{
  // TODO BUG something from the last change broke this and caused it to make way more satellites
  // That is why it is now running way slower, and the clustering is way higher than it should be
  // static int flag=1;
  int i;
  FILE *fp;
  long IDUM3 = -555, iseed = 555;
  FILE *outf;
  char fname[1000];
  int j, nsat_rand, imag_offset, imag_mult, istart, iend, mag;
  float nsat, ncen, mass, xg[3], vg[3], xh[3], logm, bfit;
  // struct drand48_data drand_buf;
  double r;
  int warned = 0;

  // First time setup - read mock's halo file
  if (imag < 0)
  {
    srand48(555);
    if (!SILENT) fprintf(stderr, "popsim> reading mock halo data...\n");
    fp = fopen(MOCK_FILE, "r");
    //fp = fopen("/export/sirocco1/tinker/SIMULATIONS/BOLSHOI/hosthalo_z0.0_M1e10.dat", "r");
    //fp = fopen("/export/sirocco2/tinker/SIMULATIONS/C250_2560/hosthalo_z0.0_M1e10_Lsat.dat", "r");
    if (!fp)
    {
      fprintf(stderr, "popsim> could not open mock halo file\n");
      fflush(stderr);
      exit(ENOENT);
    }

    NHALO = filesize(fp);
    if (!SILENT) fprintf(stderr, "popsim> NHALO=%d\n", NHALO);
    HALO = calloc(NHALO, sizeof(struct halo));
    for (i = 0; i < NHALO; ++i)
    {
      fscanf(fp, "%f %f %f %f %f %f %f %f", &HALO[i].mass,
             &HALO[i].x, &HALO[i].y, &HALO[i].z,
             &HALO[i].vx, &HALO[i].vy, &HALO[i].vz, &HALO[i].lsat);
    }
    fclose(fp);
    if (!SILENT) fprintf(stderr, "popsim> done reading halo data [%d].\n", NHALO);

    // lets create a list of random numbers
    if (!SILENT) fprintf(stderr, "popsim> creating random numbers [%d].\n", NHALO);
    for (i = 0; i < NRANDOM; ++i)
    {
      UNIFORM_RANDOM[i] = drand48();
      GAUSSIAN_RANDOM[i] = gasdev(&IDUM3);
    }
    // each task gets its own counter
    for (i = 0; i < 100; ++i)
      IRAN_CURRENT[i] = (int)(drand48() * 100);
    if (!SILENT) fprintf(stderr, "popsim> done with randoms [%d].\n", NHALO);

    return;
  }

  /* Put this in a global so that we know which HOD
   * to use.
   */
  imag_offset = (int)fabs(maglim[0]);
  imag_mult = 1;
  if (STELLAR_MASS)
  {
    imag_offset = 90;
    imag_mult = 5;
  }
  mag = imag * imag_mult + imag_offset;
  if (!SILENT) fprintf(stderr, "popsim> starting population for imag=%d, blue=%d, mag=%d\n", imag, blue_flag, mag);

  /* We'll do simple linear interpolation for the HOD
   */
  if (blue_flag)
    sprintf(fname, "mock_blue_M%d.dat", mag);
  else
    sprintf(fname, "mock_red_M%d.dat", mag);
  outf = fopen(fname, "w");

  // srand48_r (iseed, &drand_buf);

  // fit the high-mass satellite occupation function, force slope=1
  istart = 130;
  if (imag >= 2)
    istart = 135;
  iend = 140;
  if (imag >= 2)
    iend = 145;

  bfit = 0;
  if (imag == 0)
  {
    istart = 120;
    iend = 130;
  }
  if (blue_flag)
  {
    for (i = istart; i <= iend; ++i)
      bfit += nsatb[imag][i] - i / 10.0; 
    bfit = bfit / (iend - istart + 1);
    //fprintf(stderr, "popsim> bfit=%f\n", bfit);
    for (i = iend; i <= 160; ++i)
      nsatb[imag][i] = 1 * i / 10.0 + bfit;
  }
  else
  {
    for (i = istart; i <= iend; ++i)
      bfit += nsatr[imag][i] - i / 10.0;
    bfit = bfit / (iend - istart + 1);
    //fprintf(stderr, "popsim> bfit=%f\n", bfit);
    for (i = iend; i <= 160; ++i)
      nsatr[imag][i] = 1 * i / 10.0 + bfit;
  }

  for (i = 0; i < NHALO; ++i)
  {
    mass = HALO[i].mass;
    logm = log10(mass);
    ncen = N_cen(mass, imag, blue_flag);
    // drand48_r(&drand_buf, &r);
    r = tabulated_uniform_random(thisTask);
    // r = UNIFORM_RANDOM[IRAN_CURRENT[thisTask]];
    // IRAN_CURRENT[thisTask]++;
    // if(IRAN_CURRENT[thisTask]==NRANDOM)IRAN_CURRENT[thisTask]=0;
    if (r < ncen)
    {
      fprintf(outf, "%.5f %.5f %.5f %f %f %f %d %f\n",
              HALO[i].x, HALO[i].y, HALO[i].z,
              HALO[i].vx, HALO[i].vy, HALO[i].vz, 0, logm);
    }
    nsat = N_sat(mass, imag, blue_flag);
    nsat_rand = poisson_deviate(nsat, thisTask);

    // For a really bad set of parameters, we can get a huge number of satellites for some halos.
    // Cap it so we don't print off a 10 Teraybyte file! Any MCMC or whatever will move on hopefully.
    if (nsat_rand > MAX_SATELLITES) {
      if (!warned) {
        fprintf(stderr, "popsim> WARNING: giving %d sats for halo %d\n", nsat_rand, i);
        warned = 1;
      }
      nsat_rand = MAX_SATELLITES;
    }

    for (j = 1; j <= nsat_rand; ++j)
    {
      NFW_position(mass, xg, thisTask);
      NFW_velocity(mass, vg, thisTask);
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
 * in the magnitude bin imag and color. Uses the tabulated HOD.
 */
float N_cen(float m, int imag, int blue_flag)
{
  int im;
  float x0, y0, x1, y1, yp, logm;
  logm = log10(m) / 0.1;
  im = (int)logm;
  x0 = im;
  x1 = im + 1;
  if (blue_flag)
  {
    y0 = ncenb[imag][im];
    y1 = ncenb[imag][im + 1];
  }
  else
  {
    y0 = ncenr[imag][im];
    y1 = ncenr[imag][im + 1];
  }
  yp = y0 + ((y1 - y0) / (x1 - x0)) * (logm - x0);
  // if(logm>124)
  //  printf("CEN %e %f %d %f %f %f %f %f\n",m,logm,im,x0,x1,y0,y1,yp);
  if (yp <= -10)
    return 0;
  return pow(10.0, yp);
}

/**
 * Return the number of satellites for a halo of mass m in the given 
 * magnitude bin and color. Uses the tabulated HOD.
 */
float N_sat(float m, int imag, int blue_flag)
{
  int im;
  float x0, y0, x1, y1, yp, logm;
  logm = log10(m) / 0.1;
  im = (int)logm;
  x0 = im;
  x1 = im + 1;
  if (blue_flag)
  {
    y0 = nsatb[imag][im];
    y1 = nsatb[imag][im + 1];
  }
  else
  {
    y0 = nsatr[imag][im];
    y1 = nsatr[imag][im + 1];
  }
  yp = y0 + ((y1 - y0) / (x1 - x0)) * (logm - x0);
  if (yp <= -10)
    return 0;
  return pow(10.0, yp);
}

//===========================================================================
//=  Function to generate Poisson distributed random variables              =
//=    - Input:  Mean value of distribution                                 =
//=    - Output: Returns with Poisson distributed random variable           =
//===========================================================================
int poisson_deviate(float x, int thisTask)
{
  float r;
  int poi_value; // Computed Poisson value to be returned
  double t_sum;  // Time sum value

  if (x > 50)
    return (int)x;

  x = 1 / x; //???
  // Loop to generate Poisson values using exponential distribution
  poi_value = 0;
  t_sum = 0.0;
  while (1)
  {
    r = tabulated_uniform_random(thisTask);
    t_sum = t_sum - x * log(r);
    // printf("POI %d %e %e %e %e\n",poi_value,x,r,log(r),t_sum);
    if (t_sum >= 1.0)
      break;
    poi_value++;
  }

  return (poi_value);
}

float tabulated_gaussian_random(int thisTask)
{
  float r;
  r = GAUSSIAN_RANDOM[IRAN_CURRENT[thisTask]];
  IRAN_CURRENT[thisTask]++;
  if (IRAN_CURRENT[thisTask] == NRANDOM)
    IRAN_CURRENT[thisTask] = 0;
  return r;
}
float tabulated_uniform_random(int thisTask)
{
  float r;
  r = UNIFORM_RANDOM[IRAN_CURRENT[thisTask]];
  IRAN_CURRENT[thisTask]++;
  if (IRAN_CURRENT[thisTask] == NRANDOM)
    IRAN_CURRENT[thisTask] = 0;
  return r;
}

/* Randomy generates a position away from the origin with
 * a probability given by the NFW profile for a halo of the input
 * mass (and including the CVIR_FAC)
 */
float NFW_position(float mass, float x[], int thisTask)
{
  float r, pr, max_p, costheta, sintheta, phi1, signs, rvir, rs, cvir, mfac = 1;
  double rr;
  cvir = halo_concentration(mass) * CVIR_FAC;
  rvir = pow(3 * mass / (4 * DELTA_HALO * PI * RHO_CRIT * OMEGA_M), 1.0 / 3.0);
  rs = rvir / cvir;
  max_p = NFW_density(rs, rs, 1.0) * rs * rs * 4.0 * PI;

  for (;;)
  {
    r = tabulated_uniform_random(thisTask) * rvir;
    pr = NFW_density(r, rs, 1.0) * r * r * 4.0 * PI / max_p;

    if (tabulated_uniform_random(thisTask) <= pr)
    {
      costheta = 2. * (tabulated_uniform_random(thisTask) - .5);
      sintheta = sqrt(1. - costheta * costheta);
      signs = 2. * (tabulated_uniform_random(thisTask) - .5);
      costheta = signs * costheta / fabs(signs);
      phi1 = 2.0 * PI * tabulated_uniform_random(thisTask);

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
float NFW_velocity(float mass, float v[], int thisTask)
{
  static long IDUM2 = -455;
  // static float fac = -1;
  float sigv, vbias = 1, mfac = 1;
  int i;
  float fac;

  fac = sqrt(4.499E-48) * pow(4 * DELTA_HALO * PI * OMEGA_M * RHO_CRIT / 3, 1.0 / 6.0) * 3.09E19 * sqrt(1 + REDSHIFT);
  sigv = fac * pow(mass, 1.0 / 3.0) / ROOT2;
  for (i = 0; i < 3; ++i)
    v[i] = tabulated_gaussian_random(thisTask) * sigv;
  return (0);
}

float halo_concentration(float mass)
{
  return 10 * pow(mass / 1.0E14, -0.11);
}

