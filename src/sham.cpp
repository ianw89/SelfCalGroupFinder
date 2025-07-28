#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
#include "nrutil.h"
#include "groups.hpp"
#include "sham.hpp"


double gsl_spline_eval_extrap(const gsl_spline *spline, const double *x, const double *y, int n, double xq, gsl_interp_accel *acc);

double gsl_spline_eval_extrap(const gsl_spline *spline, const double *x, const double *y, int n, double xq, gsl_interp_accel *acc)
{
    // Constant extrapolation below range - this only happens due to floating point inaccuracy issues
    if (xq < x[0]) {
        return gsl_spline_eval(spline, x[0], acc);
    } else if (xq > x[n-1]) {
        return gsl_spline_eval(spline, x[n-1], acc);
    } else {
        return gsl_spline_eval(spline, xq, acc);
    }
}


/* 
* For a galaxy at a certain redshift and vmax, use the provided halo mass function to
 * determine the host halo mass. 
 * For a given galaxy density, use the provided halo mass function to... */
float density2host_halo(float galaxy_density)
{
  //std::cerr << "Finding host halo mass for galaxy density = " << galaxy_density << std::endl;
  return exp(zbrent(func_match_nhost, log(HALO_MIN), log(HALO_MAX), 1.0E-5, galaxy_density));
}

/* For a galaxy at a certain redshift and vmax, use the provided halo mass function to
 * determine the host halo mass. For flux-limited mode. 
 * 
 * Using a vmax correction for galaxies that can't make it
 * to the end of the redshift bin.
 */
float density2host_halo_zbins3(float z, float vmax)
{
#define NZBIN 200
  int i, iz;
  float rlo, rhi, dz, dzmin, vv;
  static int flag = 1, negcnt[NZBIN];
  static double zcnt[NZBIN];
  static float volume[NZBIN], zlo[NZBIN], zhi[NZBIN],
      vhi[NZBIN], vlo[NZBIN];

  // if first call, get the volume in each dz bin
  if (flag)
  {
    for (i = 0; i < NZBIN; ++i)
    {
      zlo[i] = i * 1. / NZBIN;
      zhi[i] = zlo[i] + 0.05;
      if (i == 0)
        rlo = 0;
      else
        rlo = distance_redshift(zlo[i]);
      rhi = distance_redshift(zhi[i]);
      volume[i] = 4. / 3. * PI * (rhi * rhi * rhi - rlo * rlo * rlo) * FRAC_AREA;
      vhi[i] = 4. / 3. * PI * rhi * rhi * rhi * FRAC_AREA;
      vlo[i] = 4. / 3. * PI * rlo * rlo * rlo * FRAC_AREA;
      //fprintf(stderr,"%d: z_lo-z_hi: %f - %f",i,zlo[i],zhi[i]);
      //fprintf(stderr,"  r_lo-r_hi: %f - %f",rlo,rhi);
      //fprintf(stderr,"  volume= %e\n",volume[i]);
    }
    flag = 0;
  }
  // if negative redshift, reset the counters;
  if (z < 0)
  {
    // fprintf(stderr,"Resetting sham counts\n");
    for (i = 0; i < NZBIN; ++i)
      zcnt[i] = negcnt[i] = 0;
    return 0;
  }

  if (z > 100)
  {
    for (i = 0; i < NZBIN; ++i)
      if (negcnt[i])
        fprintf(stderr, "%d %f %d\n", i, zhi[i] - 0.025, negcnt[i]);
    return 0;
  }

  // what bins does this galaxy belong to?
  // TODO this can definitely be optimized to not have a loop NZBIN times for each galaxy.
  dzmin = 1;
  for (i = 0; i < NZBIN; ++i)
  {
    if (z >= zlo[i] && z < zhi[i])
    {
      //fprintf(stderr, "Matched z = %f to bin %d\n", z, i);
      if (vmax > vhi[i])
        vv = volume[i];
      else
        vv = vmax - vlo[i];
      if (vv < 0)
        vv = volume[i];
      negcnt[i]++;
      zcnt[i] += 1 / vv;

      if (vv < 0.0)
      {
        fprintf(stderr, "vmax = %e %e %e %e %e %e %e\n", vmax, vlo[i], vhi[i], zlo[i], zhi[i], z, zcnt[i]);
      }
      // fprintf(stdout,"> %d %e %e %e %e\n",i,vv,vmax,vlo[i],vhi[i]);
      // fflush(stdout);
    }
    dz = fabs(z - (zhi[i] + zlo[i]) / 2);
    if (dz < dzmin)
    {
      dzmin = dz;
      iz = i;
    }
  }
  // fprintf(stdout,"%f %d %e %e %f %f %f\n",z,iz,zcnt[iz],vmax,zlo[iz],zhi[iz],dzmin);
  // fflush(stdout);
  //fprintf(stderr, "Getting mass for z = %f, iz = %d, zcnt = %f", z, iz, zcnt[iz]);
  //float results = density2host_halo(zcnt[iz]);
  //fprintf(stderr, ". Result = %e\n", results);
  return density2host_halo(zcnt[iz]);

#undef NZBIN
}

float func_match_nhost(float mass, float galdensity)
{
    static int flag = 1, n = 100;
    static double *mh, *nh;
    static gsl_interp_accel *acc = nullptr;
    static gsl_spline *spline = nullptr;
    int i;
    double a, mlo, mhi, dlogm, n1;

    // First time setup, will read in a halo mass function file
    if (flag)
    {
        flag = 0;
        mh = (double*)malloc(n * sizeof(double));
        nh = (double*)malloc(n * sizeof(double));

        mlo = HALO_MIN;
        mhi = HALO_MAX;
        dlogm = log(mhi / mlo) / n;

        for (i = 0; i < n; ++i)
        {
            mh[i] = exp((i + 0.5) * dlogm) * mlo;
            n1 = qromo(halo_abundance2, log(mh[i]), log(HALO_MAX), midpnt);
            nh[i] = log(n1);
            mh[i] = log(mh[i]);
            //std::cerr << "mh[" << i << "] = " << mh[i] << ", nh[" << i << "] = " << nh[i] << std::endl;
        }

        acc = gsl_interp_accel_alloc();
        spline = gsl_spline_alloc(gsl_interp_cspline, n);
        gsl_spline_init(spline, mh, nh, n);
    }

    // Interpolate the halo mass function
    a = gsl_spline_eval_extrap(spline, mh, nh, n, mass, acc);
    return exp(a) - galdensity;
}

float halo_abundance2(float m)
{
  m = exp(m);
  return halo_abundance(m) * m;
}

float halo_abundance(float m)
{
    int i;
    FILE *fp;
    float a;
    static int n = 0;
    static double *x = nullptr, *y = nullptr;
    static gsl_interp_accel *acc = nullptr;
    static gsl_spline *spline = nullptr;
    char aa[1000];

    if (!n)
    {
        fp = openfile(HALO_MASS_FUNC_FILE);
        n = filesize(fp);
        x = (double*)malloc(n * sizeof(double));
        y = (double*)malloc(n * sizeof(double));
        for (i = 0; i < n; ++i)
        {
            float xf, yf;
            fscanf(fp, "%f %f", &xf, &yf);
            x[i] = log(xf);
            y[i] = log(yf);
            fgets(aa, 1000, fp);
        }
        fclose(fp);

        acc = gsl_interp_accel_alloc();
        spline = gsl_spline_alloc(gsl_interp_cspline, n);
        gsl_spline_init(spline, x, y, n);
    }
    a = gsl_spline_eval_extrap(spline, x, y, n, log(m), acc);
    return exp(a);
}
