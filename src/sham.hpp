#pragma once

#include <fstream>
#include <sstream>
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

float density2host_halo_zbins3(float z, double vmax);
float density2host_halo(float galaxy_density);
float func_match_nhost(float mass, float galaxy_density);
float halo_abundance_log(float logM);
float halo_abundance(float m);


// Singleton holding the PCA model matrices for halo property transforms.
// Reads HALO_PCA_MODEL_TEXT_FILE (written by pca_halo.ipynb).
// Features order: LOGMHALO, c, Spin, Halfmass_Scale
struct HaloPCAModel {
    static constexpr int NFEAT = 4;
    double scaler_mean[NFEAT];
    double scaler_scale[NFEAT];
    double pca_mean[NFEAT];
    double W[NFEAT][NFEAT]; // W[component][feature]

    static HaloPCAModel& get() {
        static HaloPCAModel inst;
        return inst;
    }

    // Inverse transform: PCA coords -> original feature space.
    // pca[NFEAT] in, x[NFEAT] out.
    // x[0]=LOGMHALO, x[1]=c, x[2]=Spin, x[3]=Halfmass_Scale
    void inverse_transform(const double pca[NFEAT], double x[NFEAT]) {
        if (!loaded) load();
        double xs[NFEAT] = {0};
        for (int j = 0; j < NFEAT; j++)
            for (int i = 0; i < NFEAT; i++)
                xs[j] += W[i][j] * pca[i];
        for (int j = 0; j < NFEAT; j++)
            x[j] = (xs[j] + pca_mean[j]) * scaler_scale[j] + scaler_mean[j];
    }

    // Forward transform: original feature space -> PCA coords.
    void forward_transform(const double x[NFEAT], double pca[NFEAT]) {
        if (!loaded) load();
        double xs[NFEAT];
        for (int j = 0; j < NFEAT; j++)
            xs[j] = (x[j] - scaler_mean[j]) / scaler_scale[j] - pca_mean[j];
        for (int i = 0; i < NFEAT; i++) {
            pca[i] = 0;
            for (int j = 0; j < NFEAT; j++)
                pca[i] += W[i][j] * xs[j];
        }
    }

    void reset() { loaded = false; }

private:
    bool loaded = false;
    HaloPCAModel() = default;

    void load() {
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
};