#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
#include <assert.h>
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

HMFCumulative::HMFCumulative() {
    build();
}

// Returns log( n(>exp(logmass)) ) via spline interpolation.
double HMFCumulative::eval(double logmass) {
    return gsl_spline_eval_extrap(spline, mh, nh, N, logmass, acc);
}

// Free the spline so it is rebuilt on the next eval() call.
void HMFCumulative::reset() {
    if (spline) { gsl_spline_free(spline); spline = nullptr; }
    if (acc)    { gsl_interp_accel_free(acc); acc = nullptr; }
}

void HMFCumulative::build() {
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
    if (spline == nullptr || acc == nullptr) {
        fprintf(stderr, "Failed to allocate GSL spline or accelerator\n");
        exit(1);
    }
    int status = gsl_spline_init(spline, mh, nh, N);
    if (status != GSL_SUCCESS) {
        fprintf(stderr, "Failed to initialize GSL spline: %s\n", gsl_strerror(status));
        exit(1);
    }

}


float func_match_nhost(float logmass, float galdensity)
{
    double a = HMFCumulative::get().eval(logmass);
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


float AbundanceMatchingManager::match_in_zbins(float z, double vmax) {
  int i, iz;
  double rlo, rhi, dz, dzmin, vv;

  // if first call, get the volume in each dz bin
  if (needs_setup) {
      for (i = 0; i < NZBIN; ++i) {
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
      needs_setup = false;
  }

  if (z < 0 || isnan(z) || !isfinite(z)) {
    LOG_ERROR("ERROR: invalid redshift %f\n", z);
    assert(false);
  }

  if (z > 100)
  {
    for (i = 0; i < NZBIN; ++i)
      if (negcnt[i])
        fprintf(stderr, "%d %f %d\n", i, zhi[i] - 0.025, negcnt[i]);
    return 0;
  }

  // Determine what bins this galaxy belong to
  // TODO this can definitely be optimized to not have a loop NZBIN times for each galaxy.
  dzmin = 1;
  for (i = 0; i < NZBIN; ++i) {
    if (z >= zlo[i] && z < zhi[i]) {
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
        LOG_ERROR("vmax = %e.  %e %e %e %e %e %e\n", vmax, vlo[i], vhi[i], zlo[i], zhi[i], z, zcnt[i]);
    }
    dz = fabs(z - (zhi[i] + zlo[i]) / 2);
    if (dz < dzmin) {
      dzmin = dz;
      iz = i;
    }
  }
  // fprintf(stdout,"%f %d %e %e %f %f %f\n",z,iz,zcnt[iz],vmax,zlo[iz],zhi[iz],dzmin);
  // fflush(stdout);
  //fprintf(stderr, "Getting mass for z = %f, iz = %d, zcnt = %f", z, iz, zcnt[iz]);
  //float results = match(zcnt[iz]);
  //fprintf(stderr, ". Result = %e\n", results);
  return match(zcnt[iz]);
}

void AbundanceMatchingManager::reset() {
    for (int i = 0; i < NZBIN; ++i)
      zcnt[i] = negcnt[i] = 0;
}
  
float HaloMassAMManager::match(float galaxy_density) {
    return exp(zbrent(func_match_nhost, log(HALO_MIN), log(HALO_MAX), 1.0E-5, galaxy_density));
}





HaloPCADensityFuncs::HaloPCADensityFuncs() {
    for (int i = 0; i < NCOMP; ++i)
        build(i);
}

double HaloPCADensityFuncs::eval(int idx, double val) {
    if (idx < 0 || idx >= NCOMP) {
        LOG_ERROR("Invalid PCA component index %d\n", idx);
        assert(false);
    }
    double v = gsl_spline_eval_extrap(spline[idx], px[idx], py[idx], n[idx], val, acc[idx]);
    return (v < 0.0) ? 0.0 : v;
}

void HaloPCADensityFuncs::build(int idx) {
    const char *files[NCOMP] = {
        HALO_PCA1_DENSITY_FUNC_FILE,
        HALO_PCA2_DENSITY_FUNC_FILE,
        HALO_PCA3_DENSITY_FUNC_FILE,
        HALO_PCA4_DENSITY_FUNC_FILE
    };
    FILE *fp = openfile(files[idx]);
    int cnt = filesize(fp);
    px[idx] = (double*)malloc(cnt * sizeof(double));
    py[idx] = (double*)malloc(cnt * sizeof(double));
    n[idx] = cnt;
    for (int i = 0; i < cnt; i++)
        fscanf(fp, "%lf %lf", &px[idx][i], &py[idx][i]);
    fclose(fp);
    acc[idx] = gsl_interp_accel_alloc();
    spline[idx] = gsl_spline_alloc(gsl_interp_cspline, cnt);
    if (!spline[idx] || !acc[idx]) {
        LOG_ERROR("Failed to allocate GSL spline or accelerator for PCA component %d\n", idx + 1);
        exit(1);
    }
    int status = gsl_spline_init(spline[idx], px[idx], py[idx], cnt);
    if (status) {
        LOG_ERROR("Failed to initialize GSL spline for PCA component %d: %s\n", idx + 1, gsl_strerror(status));
        exit(1);
    }
    //std::cout << "Loaded PCA density function for component " << (idx + 1) << " with " << cnt << " points." << std::endl;
}

HaloPCAFuncCumulative::HaloPCAFuncCumulative() {
    for (int i = 0; i < N_HPCA_COMP; ++i)
        build(i);
}

// Returns log( n(>pca_val) ) via spline interpolation.
double HaloPCAFuncCumulative::eval(double pca_val, int comp) {
    int idx = comp - 1;
    return gsl_spline_eval_extrap(spline[idx], mh[idx], nh[idx], N, pca_val, acc[idx]);
}

// Free the spline so it is rebuilt on the next eval() call.
void HaloPCAFuncCumulative::reset(int comp) {
    int idx = comp - 1;
    if (spline[idx]) { gsl_spline_free(spline[idx]); spline[idx] = nullptr; }
    if (acc[idx])    { gsl_interp_accel_free(acc[idx]); acc[idx] = nullptr; }
}

void HaloPCAFuncCumulative::build(int idx) {
    //double dlogm = log(HALO_PCA_MAX / HALO_PCA_MIN) / N;
    double dm = (HALO_PCA_MAX - HALO_PCA_MIN) / N;

    static int comp_hack = 1; // so I can use qromo. Once I switch to GSL integrator can likely do cleaner thing here
    comp_hack = idx;
    auto integrand = [](float x) -> float {
      return HaloPCADensityFuncs::get().eval(comp_hack, x);
    };

    for (int i = 0; i < N; ++i) {
        //mh[idx][i] = exp((i + 0.5) * dlogm) * mlo;
        mh[idx][i] = HALO_PCA_MIN + (i + 0.5) * dm;
        //double n1 = qromo(integrand, log(mh[idx][i]), log(HALO_PCA_MAX), midpnt);
        double n1 = qromo(integrand, mh[idx][i], HALO_PCA_MAX, midpnt);
        nh[idx][i] = (n1 > 0) ? log(n1) : -80.0; // clamp to avoid 0 in logspace
        //mh[idx][i] = log(mh[idx][i]);
    }
    
    acc[idx] = gsl_interp_accel_alloc();
    spline[idx] = gsl_spline_alloc(gsl_interp_cspline, N); 
    if (!spline[idx] || !acc[idx]) {
        LOG_ERROR("Failed to allocate GSL spline or accelerator for PCA component %d\n", idx + 1);
        exit(1);
    }
    int status = gsl_spline_init(spline[idx], mh[idx], nh[idx], N);
    if (status) {
        LOG_ERROR("Failed to initialize GSL spline for PCA component %d: %s\n", idx + 1, gsl_strerror(status));
        exit(1);
    }
}

float func_match_nhost_pca1(float pca_val, float galdensity) {
    std::cout << "Matching galaxy density " << galdensity << " to PCA component 1 value..." << std::endl;
    double a = HaloPCAFuncCumulative::get().eval(pca_val, 1);
    return exp(a) - galdensity;
}

float func_match_nhost_pca2(float pca_val, float galdensity) {
    double a = HaloPCAFuncCumulative::get().eval(pca_val, 2);
    return exp(a) - galdensity;
}

float func_match_nhost_pca3(float pca_val, float galdensity) {
    double a = HaloPCAFuncCumulative::get().eval(pca_val, 3);
    return exp(a) - galdensity;
}

float func_match_nhost_pca4(float pca_val, float galdensity) {
    double a = HaloPCAFuncCumulative::get().eval(pca_val, 4);
    return exp(a) - galdensity;
}


float HaloPCA1AMManager::match(float galaxy_density) {
    std::cout << "Matching galaxy density " << galaxy_density << " to PCA component 1 value..." << std::endl;
    return zbrent(func_match_nhost_pca1, HALO_PCA_MIN, HALO_PCA_MAX, 1.0E-5, galaxy_density);
}
float HaloPCA2AMManager::match(float galaxy_density) {
    return zbrent(func_match_nhost_pca2, HALO_PCA_MIN, HALO_PCA_MAX, 1.0E-5, galaxy_density);
}
float HaloPCA3AMManager::match(float galaxy_density) {
    return zbrent(func_match_nhost_pca3, HALO_PCA_MIN, HALO_PCA_MAX, 1.0E-5, galaxy_density);
}
float HaloPCA4AMManager::match(float galaxy_density) {
    return zbrent(func_match_nhost_pca4, HALO_PCA_MIN, HALO_PCA_MAX, 1.0E-5, galaxy_density);
}

void HaloPCAModel::load() {
    std::ifstream fp(HALO_PCA_MODEL_TEXT_FILE);
    if (!fp) { fprintf(stderr, "Could not open %s\n", HALO_PCA_MODEL_TEXT_FILE); exit(1); }
    std::string line;
    int block = 0;
    while (std::getline(fp, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        switch (block) {
            case 0: ss >> scaler_mean[0] >> scaler_mean[1] >> scaler_mean[2] >> scaler_mean[3]; break;
            case 1: ss >> scaler_scale[0] >> scaler_scale[1] >> scaler_scale[2] >> scaler_scale[3]; break;
            case 2: ss >> pca_mean[0] >> pca_mean[1] >> pca_mean[2] >> pca_mean[3]; break;
            case 3: ss >> W[0][0] >> W[0][1] >> W[0][2] >> W[0][3]; break;
            case 4: ss >> W[1][0] >> W[1][1] >> W[1][2] >> W[1][3]; break;
            case 5: ss >> W[2][0] >> W[2][1] >> W[2][2] >> W[2][3]; break;
            case 6: ss >> W[3][0] >> W[3][1] >> W[3][2] >> W[3][3]; break;
        }
        block++;
    }
    loaded = true;
}

void HaloPCAModel::inverse_transform(galaxy *galaxy) {
    if (!loaded) load();
    double xs[NFEAT] = {0};
    for (int j = 0; j < NFEAT; j++)
        for (int i = 0; i < NFEAT; i++)
            xs[j] += W[i][j] * galaxy->halo_pca[i];
    
    galaxy->mass = pow(10, ((xs[0] + pca_mean[0]) * scaler_scale[0] + scaler_mean[0]));
    galaxy->c = (xs[1] + pca_mean[1]) * scaler_scale[1] + scaler_mean[1];
    galaxy->spin = (xs[2] + pca_mean[2]) * scaler_scale[2] + scaler_mean[2];
    galaxy->age = (xs[3] + pca_mean[3]) * scaler_scale[3] + scaler_mean[3];
}

void HaloPCAModel::forward_transform(galaxy *galaxy) {
    // pca[NFEAT] in, x[NFEAT] out.
    // x[0]=LOGMHALO, x[1]=c, x[2]=Spin, x[3]=Halfmass_Scale
    if (!loaded) load();
    double xs[NFEAT];
    double x[NFEAT] = {log10(galaxy->mass), galaxy->c, galaxy->spin, galaxy->age};
    for (int j = 0; j < NFEAT; j++)
        xs[j] = (x[j] - scaler_mean[j]) / scaler_scale[j] - pca_mean[j];
    for (int i = 0; i < NFEAT; i++) {
        galaxy->halo_pca[i] = 0;
        for (int j = 0; j < NFEAT; j++)
            galaxy->halo_pca[i] += W[i][j] * xs[j];
    }
}

void set_halo_props_from_pca_props(galaxy *galaxy)
{
    const float MIN = -15.0;
    const float MAX = 15.0;
    for (int i = 0; i < 4; i++) {
      if (isnan(galaxy->halo_pca[i]) || !isfinite(galaxy->halo_pca[i])) {
        LOG_ERROR("ERROR: galaxy has invalid halo PCA value %f for component %d\n", galaxy->halo_pca[i], i+1);
        assert(false);
      }
    }
    for (int i = 0; i < 4; i++) {
      if (galaxy->halo_pca[i] < MIN) {
        galaxy->halo_pca[i] = MIN;
        LOG_WARN("WARNING: galaxy halo_pca%d value %f is below minimum %f, clamping to minimum.\n", i+1, galaxy->halo_pca[i], MIN);
      } else if (galaxy->halo_pca[i] > MAX) {
        galaxy->halo_pca[i] = MAX;
        LOG_WARN("WARNING: galaxy halo_pca%d value %f is above maximum %f, clamping to maximum.\n", i+1, galaxy->halo_pca[i], MAX);
      }
    }

    HaloPCAModel::get().inverse_transform(galaxy);
    update_galaxy_halo_props(galaxy);
}
