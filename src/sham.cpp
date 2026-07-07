#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "nrutil.h"
#include "groups.hpp"
#include "sham.hpp"

double gsl_spline_eval_extrap(const gsl_spline *spline, const double *x, const double *y, int n, double xq, gsl_interp_accel *acc) {
    // Constant extrapolation below range - this only happens due to floating point inaccuracy issues
    if (xq < x[0]) {
        LOG_WARN("xq (%.3f) is below the range. Extrapolating at left edge.\n", xq);
        return gsl_spline_eval(spline, x[0], acc);
    } else if (xq > x[n-1]) {
        LOG_WARN("xq (%.3f) is above the range. Extrapolating at right edge.\n", xq);
        return gsl_spline_eval(spline, x[n-1], acc);
    } else {
        return gsl_spline_eval(spline, xq, acc);
    }
}

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
  
/*
void set_halo_props_from_pca_props(galaxy *galaxy)
{
    // TODO these are pretty dumb
    const float MIN = -1000.0;
    const float MAX = 1000.0;
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

    HaloLatentModel::get().inverse_transform(galaxy);
    update_galaxy_halo_props(galaxy);
}
*/