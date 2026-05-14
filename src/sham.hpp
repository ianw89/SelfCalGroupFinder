#pragma once

#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
#include <fstream>
#include <sstream>
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

#define HALO_PCA_MIN -20.0
#define HALO_PCA_MAX 20.0

// Interface for methods like density2host_halo

float func_match_nhost(float logmass, float galaxy_density);
float halo_abundance_log(float logM);
float halo_abundance(float m);
void set_halo_props_from_pca_props(galaxy *galaxy);


// *******************************************************************
// STANDARD HALO MASS ABUNDANCE MATCHING
// *******************************************************************


class HMFCumulative {
    public:
    static HMFCumulative& get() {
        static HMFCumulative inst;
        return inst;
    }    
    double eval(double logmass);
    void reset();

    private:
    static constexpr int N = 100;
    double mh[N];
    double nh[N];
    gsl_interp_accel *acc = nullptr;
    gsl_spline *spline = nullptr;

    HMFCumulative();
    void build();
};

#define NZBIN 200
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
    float match_in_zbins(float z, double vmax);

    /**
     * Reset SHAM counters for the z-bins. Not needed in volume-limited mode.
     */
    void reset();

    private:
      bool needs_setup = true;
      int negcnt[NZBIN];
      double zcnt[NZBIN];
      float volume[NZBIN], zlo[NZBIN], zhi[NZBIN], vhi[NZBIN], vlo[NZBIN];
};

class HaloMassAMManager : public AbundanceMatchingManager {
    public:

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
    float match(float galaxy_density) override;

    private:
    bool loaded = false;
    HaloMassAMManager() {
        HMFCumulative::get(); // trigger loading of the halo mass function spline
    }

};





// *******************************************************************
// HALO PCA ABUNDANCE MATCHING
// *******************************************************************


// Singleton holding the PCA model matrices for halo property transforms.
// Reads HALO_PCA_MODEL_TEXT_FILE (written by pca_halo.ipynb).
// Features order: LOGMHALO, c, Spin, Halfmass_Scale


// Singleton holding splines for the 4 halo PCA component density functions.
// Each spline is a fit to the a PCA coordinate density (Mpc^-3 h^3) as read from input files.
// This is the equivalent of HaloMassFunction. 
class HaloPCADensityFuncs {
public:
    static constexpr int NCOMP = 4;

    static HaloPCADensityFuncs& get() {
        static HaloPCADensityFuncs inst;
        return inst;
    }

    /**
     * Returns the probability density at PCA coordinate x for PCA component index (0-3).
     * Both val and the returned density are in linear space (not log).
     * Clamped to >= 0 since the cubic spline can dip below zero in sparse regions.
     */
    double eval(int idx, double val);

private:
    int n[NCOMP] = {0, 0, 0, 0};
    double *px[NCOMP] = {nullptr, nullptr, nullptr, nullptr};
    double *py[NCOMP] = {nullptr, nullptr, nullptr, nullptr};
    gsl_interp_accel *acc[NCOMP] = {nullptr, nullptr, nullptr, nullptr};
    gsl_spline *spline[NCOMP] = {nullptr, nullptr, nullptr, nullptr};
    
    HaloPCADensityFuncs();
    void build(int idx);
};


class HaloPCAModel {
public:
    static constexpr int NFEAT = 4;

    static HaloPCAModel& get() {
        static HaloPCAModel inst;
        return inst;
    }

    // Inverse transform: PCA coords -> original feature space
    void inverse_transform(galaxy *galaxy);

    // Forward transform: original feature space -> PCA coords
    void forward_transform(galaxy *galaxy);

    void reset() { loaded = false; }

private:
    double scaler_mean[NFEAT];
    double scaler_scale[NFEAT];
    double pca_mean[NFEAT];
    double W[NFEAT][NFEAT]; // W[component][feature]
    bool loaded = false;

    HaloPCAModel() = default;

    void load();
};

class HaloPCAFuncCumulative {
    public:
    static HaloPCAFuncCumulative& get() {
        static HaloPCAFuncCumulative inst;
        return inst;
    }    
    double eval(double logmass, int comp);
    void reset(int comp);

    private:
    static constexpr int N = 200;
    double mh[N_HPCA_COMP][N]; // PCA coordinate values for each component (log mass, concentration, spin, age)
    double nh[N_HPCA_COMP][N]; // log cumulative number density for each PCA component
    gsl_interp_accel* acc[N_HPCA_COMP] = {nullptr, nullptr, nullptr, nullptr};
    gsl_spline* spline[N_HPCA_COMP] = {nullptr, nullptr, nullptr, nullptr};

    HaloPCAFuncCumulative();
    void build(int comp);
};


class HaloPCA1AMManager : public AbundanceMatchingManager {
    public:
    static HaloPCA1AMManager& get() {
        static HaloPCA1AMManager inst;
        return inst;
    }
    float match(float galaxy_density) override;

    private:
    HaloPCA1AMManager() {
        HaloPCAModel::get(); // load transformation matrices
        HaloPCAFuncCumulative::get(); // trigger loading of the PCA splines
    }
};

class HaloPCA2AMManager : public AbundanceMatchingManager {
    public:
    static HaloPCA2AMManager& get() {
        static HaloPCA2AMManager inst;
        return inst;
    }

    float match(float galaxy_density) override;

    private:
    HaloPCA2AMManager() {
        HaloPCAModel::get(); // trigger loading of the PCA splines
    }
};

class HaloPCA3AMManager : public AbundanceMatchingManager {
    public:
    static HaloPCA3AMManager& get() {
        static HaloPCA3AMManager inst;
        return inst;
    }

    float match(float galaxy_density) override;

    private:
    HaloPCA3AMManager() {
        HaloPCAModel::get(); // trigger loading of the PCA splines
    }
};

class HaloPCA4AMManager : public AbundanceMatchingManager {
    public:
    static HaloPCA4AMManager& get() {
        static HaloPCA4AMManager inst;
        return inst;
    }

    float match(float galaxy_density) override;

    private:
    HaloPCA4AMManager() {
        HaloPCAModel::get(); // trigger loading of the PCA splines
    }
};

