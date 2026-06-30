#pragma once

#include "groups.hpp"
#include <fstream>
#include <sstream>
#include <cmath>
#include <iostream>

// Right now I don't think we need to keep the non-pca versions of the galaxy properties
// for any reason, so no need to use this yet.

/**

// Singleton holding the PCA model matrices for galaxy property transforms.
// Reads GAL_PCA_MODEL_TEXT_FILE (written by pca_gal.ipynb).
// Features order: x[0]=ABS MAG R, x[1]=G-R, x[2]=c9050, x[3]=Dn4000_MODEL
class GalPCAModel {
public:
    static constexpr int NFEAT = 4;
    
    static GalPCAModel& get() {
        static GalPCAModel inst;
        return inst;
    }

    // Inverse transform: PCA coords -> original feature space
    void inverse_transform(galaxy *galaxy) {
        if (!loaded) load();
        double xs[NFEAT] = {0};
        for (int j = 0; j < NFEAT; j++)
            for (int i = 0; i < NFEAT; i++)
                xs[j] += W[i][j] * galaxy->gal_pca[i];
        
        galaxy->??? = pow(10, ((xs[0] + pca_mean[0]) * scaler_scale[0] + scaler_mean[0]));
        galaxy->??? = (xs[1] + pca_mean[1]) * scaler_scale[1] + scaler_mean[1];
        galaxy->??? = (xs[2] + pca_mean[2]) * scaler_scale[2] + scaler_mean[2];
        galaxy->??? = (xs[3] + pca_mean[3]) * scaler_scale[3] + scaler_mean[3];
    }

    // Forward transform: original feature space -> PCA coords
    void forward_transform(galaxy *galaxy) {
        if (!loaded) load();
        double xs[NFEAT];
        double x[NFEAT] = {log10(galaxy->???), galaxy->???, galaxy->???, galaxy->???};
        for (int j = 0; j < NFEAT; j++)
            xs[j] = (x[j] - scaler_mean[j]) / scaler_scale[j] - pca_mean[j];
        for (int i = 0; i < NFEAT; i++) {
            galaxy->gal_pca[i] = 0;
            for (int j = 0; j < NFEAT; j++)
                galaxy->gal_pca[i] += W[i][j] * xs[j];
        }
    }

    void reset() { loaded = false; }

private:
    double scaler_mean[NFEAT];
    double scaler_scale[NFEAT];
    double pca_mean[NFEAT];
    double W[NFEAT][NFEAT]; // W[component][feature]
    bool loaded = false;

    GalPCAModel() = default;

    void load() {
        std::ifstream fp(GAL_PCA_MODEL_TEXT_FILE);
        if (!fp) { fprintf(stderr, "Could not open %s\n", GAL_PCA_MODEL_TEXT_FILE); exit(1); }
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


 */