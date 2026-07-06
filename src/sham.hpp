#pragma once

#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include "groups.hpp"

// Min/Max for abundance matching.
// The AM goes from the most massive halos to the least using the number density in the halo mass function file.
// The exact choice of upper bound doesn't matter much as long as its above the biggest halos in the HMF.
// The HMF will have the density extremely low there so it won't get used in practice.
// The lower bound just needs to be low enough that the that the least massive halo you will assign to the galaxy samlpe
// is above it; otherwise it doesn't matter.
//
// Note that when populating mock we have a tighter range; this isn't used for that.
#define HALO_MAX 1.0E+16
#define HALO_MIN 1.0E+8

// Interface for methods like density2host_halo

//float func_match_nhost(float logmass, float galaxy_density);
float halo_abundance_log(float logM);
float halo_abundance(float m);
void set_halo_props_from_pca_props(galaxy *galaxy);
double gsl_spline_eval_extrap(const gsl_spline *spline, const double *x, const double *y, int n, double xq, gsl_interp_accel *acc);

// *******************************************************************
// STANDARD HALO MASS ABUNDANCE MATCHING
// *******************************************************************

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
        std::ifstream fp = openfile(HALO_MASS_FUNC_FILE);
        n = filesize(HALO_MASS_FUNC_FILE);
        x = (double*)malloc(n * sizeof(double));
        y = (double*)malloc(n * sizeof(double));
        for (int i = 0; i < n; ++i)
        {
            float xf, yf;
            fp >> xf >> yf;
            x[i] = log(xf);
            y[i] = log(yf);
            std::string line;
            std::getline(fp, line);
        }
        fp.close();

        acc = gsl_interp_accel_alloc();
        spline = gsl_spline_alloc(gsl_interp_steffen, n); // Consider gsl_interp_steffen to prevent oscillations between points.
        gsl_spline_init(spline, x, y, n);
      }
};

class HMFCumulative {
public:
    static HMFCumulative& get() {
        static HMFCumulative inst;
        return inst;
    }    

    // Returns log( n(>exp(logmass)) ) via spline interpolation.
    //double eval(double logmass) {
    //    return gsl_spline_eval_extrap(spline, mh, nh, N, logmass, acc);
    //}

    // Returns the mass corresponding to a given cumulative density n(>M) via spline interpolation.
    double eval(double logCumulativeDensity) {
        if (logCumulativeDensity > 100) {
            LOG_WARN("Warning: logCumulativeDensity %f is very high; are you sure it is log?\n", logCumulativeDensity);
        }
        //LOG_INFO("Evaluating HMFCumulative for density = %e. exp(nh[0])=%e, exp(mh[0])=%e\n", exp(logCumulativeDensity), exp(nh[0]), exp(mh[0]));

        return exp(gsl_spline_eval_extrap(spline, nh.data(), mh.data(), N, logCumulativeDensity, acc));
    }

private:
    static constexpr int N = 200;
    std::vector<double> mh;
    std::vector<double> nh;
    gsl_interp_accel *acc = nullptr;
    gsl_spline *spline = nullptr;

    HMFCumulative() {
        build();
    }

    // TODO to switch over to the generic latsham AMCDF, we need to remake the density function file in the right density units
    void build() {

        double mlo = HALO_MIN, mhi = HALO_MAX;
        double dlogm = log(mhi / mlo) / N;
        mh.resize(N);
        nh.resize(N);
        for (int i = 0; i < N; ++i) {
            mh[i] = exp((i + 0.5) * dlogm) * mlo;
            double n1 = qromo(halo_abundance_log, log(mh[i]), log(HALO_MAX), midpnt);
            nh[i] = log(n1);
            mh[i] = log(mh[i]);
        }
        acc = gsl_interp_accel_alloc();
        spline = gsl_spline_alloc(gsl_interp_steffen, N); 
        if (spline == nullptr || acc == nullptr) {
            fprintf(stderr, "Failed to allocate GSL spline or accelerator\n");
            exit(1);
        }
        /* Instead of fitting spline for nh as a function of mh,
           do the inverse, which is what we actually want for abundance matching. */
        //int status = gsl_spline_init(spline, mh, nh, N);

        // Reverse the arrays to fit spline for mh as a function of nh
        std::reverse(mh.begin(), mh.end());
        std::reverse(nh.begin(), nh.end());

        // Print 
        for (int i = 0; i < N; ++i) {
            LOG_VERBOSE("Cumulative density function: log10(n(>M)) = %.6f, log10(M) = %.4f\n", log10(exp(nh[i])), log10(exp(mh[i])));
        }
        int status = gsl_spline_init(spline, nh.data(), mh.data(), N);
        if (status != GSL_SUCCESS) {
            fprintf(stderr, "Failed to initialize GSL spline: %s\n", gsl_strerror(status));
            exit(1);
        }

    }
};

#define NZBIN 110
class AbundanceMatchingManager {
    public:

    /**
     * Match a galaxy density to a halo density to acquire a halo property.
     * 
     * Units must be consistent and are usually [1 / (Mpc/h)^3].
     */
    virtual float match(float galaxy_density) = 0;

    /** 
     * Matching for flux-limited mode - tracks the density seperately for overlapping redshift bins.
     * 
     * Using a vmax correction for galaxies that can't make it to the end of the redshift bin.
     */
    float match_in_zbins(float z, double vmax) {
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
                fprintf(stderr, "WHAT IS THIS? %d %f %d\n", i, zhi[i] - 0.025, negcnt[i]);
            return 0;
        }

        // Determine what bins this galaxy belong to
        // TODO this can definitely be optimized to not have a loop NZBIN times for each galaxy.
        dzmin = 1;
        for (i = 0; i < NZBIN; ++i) {
            if (z >= zlo[i] && z <= zhi[i]) {
                if (vmax > vhi[i])
                    vv = volume[i];
                else
                    vv = vmax - vlo[i]; // could result in some tiny numbers near min. TODO ensure we compute vmax the same way...
                if (vv < 0) {
                    // TODO This happens a fairly often, I think it's because VMAX is calculated on un-k-corrected mag?
                    // Why do we set it equal to the z bin's volume in this case? To bail to something reasonable?
                    //LOG_WARN("Warning: vmax was less than vlo for z-bin %d, z=%f, vmax=%f, vlo=%f. Setting vv=volume.\n", i, z, vmax, vlo[i]);
                    vv = volume[i];
                    
                }

                negcnt[i]++;
                zcnt[i] += 1 / vv;
                //LOG_INFO("Matched z = %f to bin %d with vv= %.1e (vhi=%.1e, vlo=%.1e), new total density %.1e\n", z, i, vv, vhi[i], vlo[i], zcnt[i]);
            }

            // Remember the closest bin to this galaxy's redshift for returning a value
            dz = fabs(z - (zhi[i] + zlo[i]) / 2);
            if (dz < dzmin) {
                dzmin = dz;
                iz = i;
            }

        }
        //LOG_INFO("Returned best match for z = %f as bin %d with vv= %.1e (vhi=%.1e, vlo=%.1e), new total density %.1e\n", z, iz, vv, vhi[iz], vlo[iz], zcnt[iz]);
        return match(zcnt[iz]);
    }

    /**
     * Reset SHAM counters for the z-bins. Not needed in volume-limited mode.
     */
    void reset() {
        for (int i = 0; i < NZBIN; ++i)
        zcnt[i] = negcnt[i] = 0;
    }

    private:
      bool needs_setup = true;
      int negcnt[NZBIN];
      double zcnt[NZBIN];
      float volume[NZBIN], zlo[NZBIN], zhi[NZBIN], vhi[NZBIN], vlo[NZBIN];
};

class HaloMassAMManager : public AbundanceMatchingManager {
public:
    double max_density_seen = 0.0;

    static HaloMassAMManager& get() {
        static HaloMassAMManager inst;
        return inst;
    }

    /* 
    * For a galaxy at a certain redshift and vmax, use the provided halo mass function to
    * determine the host halo mass. 
    * 
    * galaxy_density [number/(Mpc/h)^3] is the running total of 1/VMAX for all galaxies up to this point in the AM ordering.
    */
    float match(float galaxy_density) {
        // Let's track the largest galaxy density we've seen so far and print it later
        if (galaxy_density > max_density_seen) {
            max_density_seen = galaxy_density;
        }
        return HMFCumulative::get().eval(log(galaxy_density));

        // Old implementation, when we stored a spline that was mass->density, and did root finding instead
        //return exp(zbrent(func_match_nhost_bymass, log(HALO_MIN), log(HALO_MAX), 1.0E-5, galaxy_density));
    }

private:
    
    bool loaded = false;
    HaloMassAMManager() {
        HMFCumulative::get(); // trigger loading of the halo mass function spline
    }

};


