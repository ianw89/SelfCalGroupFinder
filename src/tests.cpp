#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include "groups.hpp"
#include "fit_clustering_omp.hpp"
#include "sham.hpp"


void test_poisson_deviate_speed() {
    printf("=== POISSON DEVIATE SPEED TEST ===\n");
    double mean = 10.0;
    int n_trials = 5000000;
    int dummy = 0;
    double t1, t2;

    // Time poisson_deviate
    t1 = omp_get_wtime();
    for (int i = 0; i < n_trials; ++i) {
        dummy += poisson_deviate(mean);
    }
    t2 = omp_get_wtime();
    printf("poisson_deviate: %d trials took %.6f seconds\n", n_trials, t2 - t1);

    // Time poisson_deviate_old
    t1 = omp_get_wtime();
    for (int i = 0; i < n_trials; ++i) {
        dummy += poisson_deviate_old(mean);
    }
    t2 = omp_get_wtime();
    printf("poisson_deviate_old: %d trials took %.6f seconds\n", n_trials, t2 - t1);

    // Prevent compiler optimizing away
    if (dummy == -1) printf("Impossible!\n");

    printf(" *** poisson_deviate speed test complete.\n\n");
}

void test_poisson_deviate_basic() {
    printf("=== POISSON DEVIATE BASIC TESTS ===\n");
    double mean = 5.0;
    int n_trials = 10000;
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < n_trials; ++i) {
        int val = poisson_deviate(mean);
        assert(val >= 0 && "Poisson deviate should be non-negative");
        sum += val;
        sum_sq += val * val;
    }
    double avg = sum / n_trials;
    double var = sum_sq / n_trials - avg * avg;
    printf("Mean: %f, Variance: %f (expected mean ~%f, variance ~%f)\n", avg, var, mean, mean);
    assert(fabs(avg - mean) < 0.1 * mean && "Sample mean should be close to input mean for poisson");
    assert(fabs(var - mean) < 0.1 * mean && "Sample variance should be close to input mean for poisson");
    printf(" *** Basic poisson_deviate tests passed.\n\n");
}

void test_poisson_deviate_edge_cases() {
    printf("=== POISSON DEVIATE EDGE CASES ===\n");
    // Mean = 0 should always return 0
    for (int i = 0; i < 100; ++i) {
        double val = poisson_deviate(0.0);
        assert(val == 0.0 && "Poisson deviate with mean 0 should be 0");
    }
    // Large mean
    double mean = 100;
    int n_trials = 10000;
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < n_trials; ++i) {
        int val = poisson_deviate(mean);
        assert(val >= 0 && "Poisson deviate should be non-negative");
        sum += val;
        sum_sq += val * val;
    }
    double avg = sum / n_trials;
    double var = sum_sq / n_trials - avg * avg;
    printf("Mean: %f, Variance: %f (expected mean ~%f, variance ~%f)\n", avg, var, mean, mean);
    assert(fabs(avg - mean) < 0.1 * mean && "Sample mean should be close to input mean for poisson");
    assert(fabs(var - mean) < 0.1 * mean && "Sample variance should be close to input mean for poisson");
    printf(" *** Edge case poisson_deviate tests passed.\n\n");
}

void test_angular_separation() {
    float theta, theta_old;

    float CLOSE = 15.0/3600.0 * PI/180.0; // 15 arcsec
    float FAR = 3.0 * PI/180.0; // 3 degrees

    printf("\n=== ANGULAR SEPARATION TESTS ===\n");

    // Test 1: Same coordinates
    theta = angular_separation(0.0, 0.0, 0.0, 0.0);
    theta_old = angular_separation_old(0.0, 0.0, 0.0, 0.0);
    printf("Test 1: angular_separation(0.0, 0.0, 0.0, 0.0) = %f, old = %f\n", theta, theta_old);
    assert(theta == 0.0 && "angular_separation should be 0 for same coordinates");

    // Test 2: Small separation
    theta = angular_separation(0.0, 0.0, CLOSE, CLOSE);
    theta_old = angular_separation_old(0.0, 0.0, CLOSE, CLOSE);
    printf("Test 2: angular_separation(0.0, 0.0, 15 arcsec, 15 arcsec) = %f, old = %f\n", theta, theta_old);
    assert(theta > 0.0 && "angular_separation should be greater than 0 for small separation");

    // Test 3: Larger separation
    theta = angular_separation(0.0, 0.0, FAR, FAR);
    theta_old = angular_separation_old(0.0, 0.0, FAR, FAR);
    printf("Test 3: angular_separation(0.0, 0.0, 3 deg, 3 deg) = %f, old = %f\n", theta, theta_old);
    assert(theta > 0.0 && "angular_separation should be greater than 0 for larger separation");

    // Test 4: Different quadrants, very far away
    theta = angular_separation(1.0, 1.0, -1.0, -1.0);
    theta_old = angular_separation_old(1.0, 1.0, -1.0, -1.0);
    printf("Test 4: angular_separation(1.0, 1.0, -1.0, -1.0) = %f, old = %f\n", theta, theta_old);
    assert(theta > 0.0 && "angular_separation should be greater than 0 for different quadrants");

    // Test 5: Edge case at poles
    theta = angular_separation(PI/2, 0.0, -PI/2, 0.0);
    theta_old = angular_separation_old(PI/2, 0.0, -PI/2, 0.0);
    printf("Test 5: angular_separation(PI/2, 0.0, -PI/2, 0.0) = %f, old = %f\n", theta, theta_old);
    assert(fabs(theta-PI)<0.000001 && "angular_separation should be PI at poles");

    printf(" *** All angular_separation tests passed.\n\n");
}

void test_psat() {
    struct galaxy gal;
    gal.mass = 1E12;
    gal.redshift = 0.1;
    gal.rco = distance_redshift(gal.redshift);
    update_galaxy_halo_props(&gal);
    float arcmin = 0.5; // ~54kpc out at z=0.1, ~134 at z=0.3
    float delta_z = 0.001;
    float bsat = 10;
    float p0,p1,p2,p3,p4,p5,p6,p7 = 0.0;
    float ang_sep = angular_separation(0.0, 0.0, 0.0, (arcmin/60.0)*(PI/180.0)); 
    float result, result2 = 0.0;

    printf("=== PSAT TESTS ===\n");

    printf("Test 1: mass=%e, proj_dist=%f', delta_z=%f\n", gal.mass, arcmin, delta_z);
    result = compute_p_proj_g(&gal, ang_sep);
    result2 = compute_p_z(delta_z * SPEED_OF_LIGHT, gal.sigmav);
    p0 = psat( &gal, ang_sep, delta_z * SPEED_OF_LIGHT, bsat);
    printf("p_proj=%f, p_z=%f, psat=%f\n", result, result2, p0);
    assert(p0 < 0.5 && "psat should be less than 0.5");

    gal.mass*=10;
    update_galaxy_halo_props(&gal);
    printf("Test 2: mass=%e, proj_dist=%f', delta_z=%f\n", gal.mass, arcmin, delta_z);
    result = compute_p_proj_g(&gal, ang_sep);
    result2 = compute_p_z(delta_z * SPEED_OF_LIGHT, gal.sigmav);
    p1 = psat( &gal, ang_sep, delta_z * SPEED_OF_LIGHT, bsat);
    printf("p_proj=%f, p_z=%f, psat=%f\n", result, result2, p1);
    assert(p1 > 0.5 && "psat should be greater than 0.5");

    gal.mass*=10;
    update_galaxy_halo_props(&gal);
    printf("Test 3: mass=%e, proj_dist=%f', delta_z=%f\n", gal.mass, arcmin, delta_z);
    result = compute_p_proj_g(&gal, ang_sep);
    result2 = compute_p_z(delta_z * SPEED_OF_LIGHT, gal.sigmav);
    p2 = psat( &gal, ang_sep, delta_z * SPEED_OF_LIGHT, bsat);
    printf("p_proj=%f, p_z=%f, psat=%f\n", result, result2, p2);
    assert(p2 > p1 && "psat should be greater than previous");

    gal.mass*=10;
    update_galaxy_halo_props(&gal);
    printf("Test 4: mass=%e, proj_dist=%f', delta_z=%f\n", gal.mass, arcmin, delta_z);
    result = compute_p_proj_g(&gal, ang_sep);
    result2 = compute_p_z(delta_z * SPEED_OF_LIGHT, gal.sigmav);
    p3 = psat( &gal, ang_sep, delta_z * SPEED_OF_LIGHT, bsat);
    printf("p_proj=%f, p_z=%f, psat=%f\n", result, result2, p3);
    assert(p3 > p2 && "psat should be greater than previous");

    gal.redshift = 0.3;
    gal.rco = distance_redshift(gal.redshift);
    update_galaxy_halo_props(&gal);
    printf("Now changing central reshift to farther away\n");
    printf("Test 5: mass=%e, proj_dist=%f', delta_z=%f\n", gal.mass, arcmin, delta_z);
    result = compute_p_proj_g(&gal, ang_sep);
    result2 = compute_p_z(delta_z * SPEED_OF_LIGHT, gal.sigmav);
    p4 = psat( &gal, ang_sep, delta_z * SPEED_OF_LIGHT, bsat);    
    printf("p_proj=%f, p_z=%f, psat=%f\n", result, result2, p4);
    assert(p4 < p3 && "psat should go down compared to equivalent test at closer redshift");
    
    arcmin = 0.25;
    ang_sep = angular_separation(0.0, 0.0, 0.0, (arcmin/60.0)*(PI/180.0)); 
    printf("Test 6: mass=%e, proj_dist=%f', delta_z=%f\n", gal.mass, arcmin, delta_z);
    result = compute_p_proj_g(&gal, ang_sep);
    result2 = compute_p_z(delta_z * SPEED_OF_LIGHT, gal.sigmav);
    p5 = psat( &gal, ang_sep, delta_z * SPEED_OF_LIGHT, bsat);    
    printf("p_proj=%f, p_z=%f, psat=%f\n", result, result2, p5);
    assert(p5 > p4 && "psat should up compared to previous since small angular separation");  

    delta_z = 0.005; // This is what I found to be a reasonable threshold 
    printf("Test 7: mass=%e, proj_dist=%f', delta_z=%f\n", gal.mass, arcmin, delta_z);
    result = compute_p_proj_g(&gal, ang_sep);
    result2 = compute_p_z(delta_z * SPEED_OF_LIGHT, gal.sigmav);    
    p6 = psat( &gal, ang_sep, delta_z * SPEED_OF_LIGHT, bsat);
    printf("p_proj=%f, p_z=%f, psat=%f\n", result, result2, p6);
    assert(p6 < p5 && "psat should go down compared to previous since larger delta_z");  

    delta_z = 0.00001; // Almost the maximal value
    arcmin = 5; // 1.3 Mpc at z~0.3, over 5x the size of a 1E12 halo
    gal.mass = 1E12;
    ang_sep = angular_separation(0.0, 0.0, 0.0, (arcmin/60.0)*(PI/180.0));
    bsat = MIN_BSAT; // Min value in formula for Bsat
    gal.rco = distance_redshift(gal.redshift);
    update_galaxy_halo_props(&gal);
    printf("Test 8: mass=%e, proj_dist=%f', delta_z=%f, bsat=%f\n", gal.mass, arcmin, delta_z, bsat);
    result = compute_p_proj_g(&gal, ang_sep);
    result2 = compute_p_z(delta_z * SPEED_OF_LIGHT, gal.sigmav);
    p7 = psat( &gal, ang_sep, delta_z * SPEED_OF_LIGHT, bsat);
    printf("p_proj=%f, p_z=%f, psat=%f\n", result, result2, p7);
    assert(p7 < 0.5 && "even for very small bsat, it shouldn't be a satellite, when so far away");

    gal.mass=1e13;
    update_galaxy_halo_props(&gal);
    delta_z = 0.001;
    arcmin = 0.2; // 54kpc at z=0.3
    bsat = MAX_BSAT; // High value in formula for Bsat for red galaxies
    ang_sep = angular_separation(0.0, 0.0, 0.0, (arcmin/60.0)*(PI/180.0));
    printf("Test 9: mass=%e, proj_dist=%f', delta_z=%f, bsat=%f\n", gal.mass, arcmin, delta_z, bsat);
    result = compute_p_proj_g(&gal, ang_sep);
    result2 = compute_p_z(delta_z * SPEED_OF_LIGHT, gal.sigmav);
    float p8 = psat( &gal, ang_sep, delta_z * SPEED_OF_LIGHT, bsat);
    printf("p_proj=%f, p_z=%f, psat=%f\n", result, result2, p8);
    assert(p8 > 0.5 && "for large bsat you can still be a satellite when close enough");


    printf(" *** All psat tests passed.\n\n");
}


void test_func_match_nhost_baseline() {
    printf("=== FUNC_MATCH_NHOST BASELINE TESTS ===\n");
    float loggalden = 1e-5; // not sure at all if this is a reasonable value but it doesn't matter
    float tol = 1e-5; // Acceptable relative difference

    float baseline[] = {-0.0000100000, -0.0000099998, -0.0000099740, -0.0000094658, -0.0000057188, 0.0000103041, 0.0000608401, 0.0001940445, 0.0005124758, 0.0012343820, 0.0028256753, 0.0062867291, 0.0137773650, 0.0299867131, 0.0651531070, 0.1417509466, 0.3093677461, 0.6779714823, 1.4924468994, 3.2999482155};

    // Test over a range of halo masses
    for (int i = 0; i < 20; ++i) {
        float logmass = log(HALO_MAX) - i * (log(HALO_MAX) - log(HALO_MIN)) / 21.0;
        float new_val = func_match_nhost(logmass, loggalden);
        printf("mass=%.3e, old=%.6e, new=%.6e, rel_diff=%.2g\n", exp(logmass), baseline[i], new_val, fabs(baseline[i]-new_val)/fabs(baseline[i]));
        assert(fabs(baseline[i] - new_val) < tol * fabs(baseline[i]) + 1e-8 && "func_match_nhost results changed!");
    }
    printf(" *** func_match_nhost baseline tests passed.\n\n");
}

int main(int argc, char **argv) {

    setup_rng();

    test_func_match_nhost_baseline();
    test_poisson_deviate_speed();
    test_poisson_deviate_basic();
    test_poisson_deviate_edge_cases();
    test_angular_separation();
    test_psat();

    printf(" *** ALL TESTS PASSED ***\n");
}