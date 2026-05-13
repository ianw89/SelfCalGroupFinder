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

struct HMFCumulative;

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


struct HMFCumulative {
    static constexpr int N = 100;
    double mh[N];
    double nh[N];
    gsl_interp_accel *acc = nullptr;
    gsl_spline *spline = nullptr;

    static HMFCumulative& get() {
        static HMFCumulative inst;
        return inst;
    }

    // Returns log( n(>exp(logmass)) ) via spline interpolation.
    double eval(double logmass) {
        if (!spline) build();
        return gsl_spline_eval_extrap(spline, mh, nh, N, logmass, acc);
    }

    // Free the spline so it is rebuilt on the next eval() call.
    void reset() {
        if (spline) { gsl_spline_free(spline); spline = nullptr; }
        if (acc)    { gsl_interp_accel_free(acc); acc = nullptr; }
    }

private:
    HMFCumulative() = default;

    void build() {
        double mlo = HALO_MIN, mhi = HALO_MAX;
        double dlogm = log(mhi / mlo) / N;
        for (int i = 0; i < N; ++i) {
            mh[i] = exp((i + 0.5) * dlogm) * mlo;
            double n1 = qromo(halo_abundance_log, log(mh[i]), log(HALO_MAX), midpnt);
            nh[i] = log(n1);
            mh[i] = log(mh[i]);
        }
        acc = gsl_interp_accel_alloc();
        spline = gsl_spline_alloc(gsl_interp_cspline, N); 
        gsl_spline_init(spline, mh, nh, N);
    }
};

float func_match_nhost(float mass, float galdensity)
{
    double a = HMFCumulative::get().eval(mass);
    return exp(a) - galdensity;
}

struct HaloMassFunction {

    static HaloMassFunction& get() {
        static HaloMassFunction inst;
        return inst;
    }

    float eval(float m) {
      if (!spline) build();
      float a = gsl_spline_eval_extrap(spline, x, y, n, log(m), acc);
      return exp(a);
    }

    float eval_log(float logM) {
      float m = exp(logM);
      return eval(m) * m; // eval(m) returns dn/dM, multiply by m to get dn/d(logM)
    }  

    private:
      int n = 0;
      double *x = nullptr, *y = nullptr;
      gsl_spline *spline = nullptr;
      gsl_interp_accel *acc = nullptr;

      HaloMassFunction() = default;

      void build() {
        char aa[1000];
        FILE *fp = openfile(HALO_MASS_FUNC_FILE);
        n = filesize(fp);
        x = (double*)malloc(n * sizeof(double));
        y = (double*)malloc(n * sizeof(double));
        for (int i = 0; i < n; ++i)
        {
            float xf, yf;
            fscanf(fp, "%f %f", &xf, &yf);
            x[i] = log(xf);
            y[i] = log(yf);
            fgets(aa, 1000, fp);
        }
        fclose(fp);

        acc = gsl_interp_accel_alloc();
        spline = gsl_spline_alloc(gsl_interp_cspline, n); // Consider gsl_interp_steffen to prevent oscillations between points.
        gsl_spline_init(spline, x, y, n);
      }
};

/**
 * Returns the density of halos at mass log(m) (number/(Mpc/h)^3) according to the provided halo mass function.
 */
float halo_abundance_log(float logm) {
    return HaloMassFunction::get().eval_log(logm);
}

/**
 * Returns the density of halos at mass m (number/(Mpc/h)^3) according to the provided halo mass function.
 */
float halo_abundance(float m) {
    return HaloMassFunction::get().eval(m);
}


float AbundanceMatchingManager::density2host_halo_zbins3(float z, double vmax) {
  int i, iz;
  double rlo, rhi, dz, dzmin, vv;

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
        LOG_ERROR("vmax = %e.  %e %e %e %e %e %e\n", vmax, vlo[i], vhi[i], zlo[i], zhi[i], z, zcnt[i]);
      }
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
  //return density2host_halo(zcnt[iz]);
  return match(zcnt[iz]);
}
  

HaloMassAMManager::HaloMassAMManager() {
  HMFCumulative::get(); // trigger loading of the halo mass function spline
}

float HaloMassAMManager::match(float galaxy_density) {
  return exp(zbrent(func_match_nhost, log(HALO_MIN), log(HALO_MAX), 1.0E-5, galaxy_density));
}







// Singleton holding splines for the 4 halo PCA component density functions.
// Each spline is a fit to the a PCA coordinate density (Mpc^-3 h^3) as read from input files.
// This is the equivalent of HaloMassFunction. 
// Call reset() if the files change between calls (e.g. in tests).
struct HaloPCADensityFuncs {
    static constexpr int NCOMP = 4;

    int n[NCOMP] = {0, 0, 0, 0};
    double *px[NCOMP] = {nullptr, nullptr, nullptr, nullptr};
    double *py[NCOMP] = {nullptr, nullptr, nullptr, nullptr};
    gsl_interp_accel *acc[NCOMP] = {nullptr, nullptr, nullptr, nullptr};
    gsl_spline *spline[NCOMP] = {nullptr, nullptr, nullptr, nullptr};

    static HaloPCADensityFuncs& get() {
        static HaloPCADensityFuncs inst;
        return inst;
    }

    // Returns the probability density at PCA coordinate x for PCA component comp (1-4).
    // Clamped to >= 0 since the cubic spline can dip below zero in sparse regions.
    double eval(int comp, double x) {
        int idx = comp - 1;
        if (!spline[idx]) build(idx);
        double v = gsl_spline_eval_extrap(spline[idx], px[idx], py[idx], n[idx], x, acc[idx]);
        return (v < 0.0) ? 0.0 : v;
    }

    void reset() {
        for (int c = 0; c < NCOMP; c++) {
            if (spline[c]) { gsl_spline_free(spline[c]); spline[c] = nullptr; }
            if (acc[c])    { gsl_interp_accel_free(acc[c]); acc[c] = nullptr; }
            if (px[c])     { free(px[c]); px[c] = nullptr; }
            if (py[c])     { free(py[c]); py[c] = nullptr; }
            n[c] = 0;
        }
    }

private:
    HaloPCADensityFuncs() = default;

    void build(int comp) {
        const char *files[NCOMP] = {
            HALO_PCA1_DENSITY_FUNC_FILE,
            HALO_PCA2_DENSITY_FUNC_FILE,
            HALO_PCA3_DENSITY_FUNC_FILE,
            HALO_PCA4_DENSITY_FUNC_FILE
        };
        FILE *fp = openfile(files[comp]);
        int cnt = filesize(fp);
        px[comp] = (double*)malloc(cnt * sizeof(double));
        py[comp] = (double*)malloc(cnt * sizeof(double));
        n[comp] = cnt;
        for (int i = 0; i < cnt; i++)
            fscanf(fp, "%lf %lf", &px[comp][i], &py[comp][i]);
        fclose(fp);
        acc[comp] = gsl_interp_accel_alloc();
        spline[comp] = gsl_spline_alloc(gsl_interp_cspline, cnt);
        gsl_spline_init(spline[comp], px[comp], py[comp], cnt);
    }
};




void set_halo_props_from_pca_props(struct galaxy *galaxy)
{
  const float MIN = -15.0;
  const float MAX = 15.0;
  if(galaxy->halo_pca1 < MIN) {
    galaxy->halo_pca1 = MIN;
    LOG_WARN("WARNING: galaxy halo_pca1 value %f is below minimum %f, clamping to minimum.\n", galaxy->halo_pca1, MIN);
  } else if (galaxy->halo_pca1 > MAX) {
    galaxy->halo_pca1 = MAX;
    LOG_WARN("WARNING: galaxy halo_pca1 value %f is above maximum %f, clamping to maximum.\n", galaxy->halo_pca1, MAX);
  }
  if (galaxy->halo_pca2 < MIN) {
    galaxy->halo_pca2 = MIN;
    LOG_WARN("WARNING: galaxy halo_pca2 value %f is below minimum %f, clamping to minimum.\n", galaxy->halo_pca2, MIN);
  } else if (galaxy->halo_pca2 > MAX) {
    galaxy->halo_pca2 = MAX;
    LOG_WARN("WARNING: galaxy halo_pca2 value %f is above maximum %f, clamping to maximum.\n", galaxy->halo_pca2, MAX);
  }
  if (galaxy->halo_pca3 < MIN) {
    galaxy->halo_pca3 = MIN;
    LOG_WARN("WARNING: galaxy halo_pca3 value %f is below minimum %f, clamping to minimum.\n", galaxy->halo_pca3, MIN);
  } else if (galaxy->halo_pca3 > MAX) {
    galaxy->halo_pca3 = MAX;
    LOG_WARN("WARNING: galaxy halo_pca3 value %f is above maximum %f, clamping to maximum.\n", galaxy->halo_pca3, MAX);
  }
  if (galaxy->halo_pca4 < MIN) {
    galaxy->halo_pca4 = MIN;
    LOG_WARN("WARNING: galaxy halo_pca4 value %f is below minimum %f, clamping to minimum.\n", galaxy->halo_pca4, MIN);
  } else if (galaxy->halo_pca4 > MAX) {
    galaxy->halo_pca4 = MAX;
    LOG_WARN("WARNING: galaxy halo_pca4 value %f is above maximum %f, clamping to maximum.\n", galaxy->halo_pca4, MAX);
  }

  // TODO finish here

  update_galaxy_halo_props(galaxy);
  return;
}