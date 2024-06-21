#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "nrutil.h"
#include "groups.h"

#define HALO_MAX 1.0E+16

/*external functions
 */
float qromo(float (*func)(float), float a, float b,
            float (*choose)(float (*)(float), float, float, int));
float midpnt(float (*func)(float), float a, float b, int n);
void spline(float x[], float y[], int n, float yp1, float ypn, float y2[]);
void splint(float xa[], float ya[], float y2a[], int n, float x, float *y);
void sort2(int n, float arr[], int id[]);
float zbrent(float (*func)(float, float), float x1, float x2, float tol, float galaxy_density);

/* Local functions
 */
float halo_abundance(float m);
float halo_abundance2(float m);
float func_match_nhost(float mass, float galaxy_density);

float density2host_halo(float galaxy_density)
{
  return exp(zbrent(func_match_nhost, log(1.0E+7), log(HALO_MAX), 1.0E-5, galaxy_density));
}

/* Using a vmax correction for galaxies that can't make it
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
      // fprintf(stderr,"volume[%d]= %e %f\n",i,volume[i],r);
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
  dzmin = 1;
  for (i = 0; i < NZBIN; ++i)
  {
    if (z >= zlo[i] && z < zhi[i])
    {
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
  return density2host_halo(zcnt[iz]);

#undef NZBIN
}

float func_match_nhost(float mass, float g7_ngal)
{
  static int flag = 1, n = 100;
  static float *mh, *ms, *mx, *nh, mlo, mhi, dlogm, mmax;
  int i;
  float a, maglo, maghi, dmag, m, n1, n2;

  if (flag)
  {
    flag = 0;
    mh = vector(1, n);
    nh = vector(1, n);
    mx = vector(1, n);

    mlo = 1.0E+8;
    mhi = HALO_MAX;
    dlogm = log(mhi / mlo) / n;

    for (i = 1; i <= n; ++i)
    {
      mh[i] = exp((i - 0.5) * dlogm) * mlo;
      n1 = qromo(halo_abundance2, log(mh[i]), log(HALO_MAX), midpnt);
      nh[i] = log(n1);
      mh[i] = log(mh[i]);
      fflush(stdout);
    }
    spline(mh, nh, n, 1.0E+30, 1.0E+30, mx);
  }
  splint(mh, nh, mx, n, mass, &a);
  return exp(a) - g7_ngal;
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
  static float *x, *y, *z;
  char aa[1000];

  if (!n)
  {
    fp = openfile("halo_mass_function.dat");
    // fp = openfile("wmap1.massfunc");
    // fp = openfile("s8_0.7.massfunc");
    n = filesize(fp);
    x = vector(1, n);
    y = vector(1, n);
    z = vector(1, n);
    for (i = 1; i <= n; ++i)
    {
      fscanf(fp, "%f %f", &x[i], &y[i]);
      x[i] = log(x[i]);
      y[i] = log(y[i]);
      fgets(aa, 1000, fp);
    }
    spline(x, y, n, 1.0E+30, 1.0E+30, z);
    fclose(fp);
  }
  splint(x, y, z, n, log(m), &a);
  return exp(a);
}

