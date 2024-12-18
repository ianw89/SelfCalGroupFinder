#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "groups.h"

int main(int argc, char **argv) {

    // TODO: switch to assertions when ready

    // Sanity test on distance_redshift, update_galaxy_halo_props, 
    // angular_separation, and radial_probability() functions
    struct galaxy gal;
    gal.mass = 1E12;
    gal.redshift = 0.1;
    gal.rco = distance_redshift(gal.redshift);
    update_galaxy_halo_props(&gal);
    float arcmin = 0.5; // like 50kpc out
    float delta_z = 0.001;
    float bprob = 10;
    float p0,p1,p2,p3,p4,p5,p6 = 0.0;
    float ang_sep = angular_separation(0.0, 0.0, 0.0, (arcmin/60.0)*(PI/180.0)); 
    float result, result2 = 0.0;

    printf("=== PSAT TESTS ===\n");

    printf("Test 1: mass=%e, proj_dist=%f', delta_z=%f\n", gal.mass, arcmin, delta_z);
    result = radial_probability_g(&gal, ang_sep);
    printf("radial_probability_g for mass=%e, ang_sep=%f') = %f\n", gal.mass, arcmin, result);
    result2 = compute_prob_rad(delta_z * SPEED_OF_LIGHT, gal.sigmav);
    printf("compute_prob_rad for dz=%f, sigmav=%f = %f\n", delta_z * SPEED_OF_LIGHT, gal.sigmav, result2);
    p0 = psat( &gal, ang_sep, delta_z * SPEED_OF_LIGHT, bprob);
    printf("psat = %f\n", p0);
    assert(p0 < 0.5 && "psat should be less than 0.5");

    gal.mass*=10;
    printf("Test 2: mass=%e, proj_dist=%f', delta_z=%f\n", gal.mass, arcmin, delta_z);
    update_galaxy_halo_props(&gal);
    result = radial_probability_g(&gal, ang_sep);
    printf("radial_probability_g for mass=%e, ang_sep=%f') = %f\n", gal.mass, arcmin, result);
    result2 = compute_prob_rad(delta_z * SPEED_OF_LIGHT, gal.sigmav);
    printf("compute_prob_rad for dz=%f, sigmav=%f = %f\n", delta_z * SPEED_OF_LIGHT, gal.sigmav, result2);
    p1 = psat( &gal, ang_sep, delta_z * SPEED_OF_LIGHT, bprob);
    printf("psat = %f\n", p1);
    assert(p1 > 0.5 && "psat should be greater than 0.5");

    gal.mass*=10;
    printf("Test 3: mass=%e, proj_dist=%f', delta_z=%f\n", gal.mass, arcmin, delta_z);
    update_galaxy_halo_props(&gal);
    result = radial_probability_g(&gal, ang_sep);
    printf("radial_probability_g for mass=%e, ang_sep=%f') = %f\n", gal.mass, arcmin, result);
    result2 = compute_prob_rad(delta_z * SPEED_OF_LIGHT, gal.sigmav);
    printf("compute_prob_rad for dz=%f, sigmav=%f = %f\n", delta_z * SPEED_OF_LIGHT, gal.sigmav, result2);
    p2 = psat( &gal, ang_sep, delta_z * SPEED_OF_LIGHT, bprob);
    printf("psat = %f\n", p2);
    assert(p2 > p1 && "psat should be greater than previous");

    gal.mass*=10;
    printf("Test 4: mass=%e, proj_dist=%f', delta_z=%f\n", gal.mass, arcmin, delta_z);
    update_galaxy_halo_props(&gal);
    result = radial_probability_g(&gal, ang_sep);
    printf("radial_probability_g for mass=%e, ang_sep=%f') = %f\n", gal.mass, arcmin, result);
    result2 = compute_prob_rad(delta_z * SPEED_OF_LIGHT, gal.sigmav);
    printf("compute_prob_rad for dz=%f, sigmav=%f = %f\n", delta_z * SPEED_OF_LIGHT, gal.sigmav, result2);
    p3 = psat( &gal, ang_sep, delta_z * SPEED_OF_LIGHT, bprob);
    printf("psat = %f\n", p3);
    assert(p3 > p2 && "psat should be greater than previous");

    gal.redshift = 0.3;
    gal.rco = distance_redshift(gal.redshift);
    update_galaxy_halo_props(&gal);
    printf("Now changing central reshift to farther away\n");
    printf("Test 5: mass=%e, proj_dist=%f', delta_z=%f\n", gal.mass, arcmin, delta_z);
    p4 = psat( &gal, ang_sep, delta_z * SPEED_OF_LIGHT, bprob);
    printf("psat = %f\n", p4); // should go down compared to first test
    assert(p4 < p3 && "psat should go down compared to equivalent test at closer redshift");
    
    arcmin = 0.25;
    ang_sep = angular_separation(0.0, 0.0, 0.0, (arcmin/60.0)*(PI/180.0)); 
    printf("Test 6: mass=%e, proj_dist=%f', delta_z=%f\n", gal.mass, arcmin, delta_z);
    p5 = psat( &gal, ang_sep, delta_z * SPEED_OF_LIGHT, bprob);
    printf("psat = %f\n", p5); // should up compared to previous
    assert(p5 > p4 && "psat should up compared to previous since small angular separation");  

    delta_z = 0.005; // This is what I found to be a reasonable threshold 
    printf("Test 7: mass=%e, proj_dist=%f', delta_z=%f\n", gal.mass, arcmin, delta_z);
    result2 = compute_prob_rad(delta_z * SPEED_OF_LIGHT, gal.sigmav);
    printf("compute_prob_rad for dz=%f, sigmav=%f = %f\n", delta_z * SPEED_OF_LIGHT, gal.sigmav, result2);
    p6 = psat( &gal, ang_sep, delta_z * SPEED_OF_LIGHT, bprob);
    printf("psat = %f\n", p6); // should go down compared to previous
    assert(p6 < p5 && "psat should go down compared to previous since larger delta_z");  

    printf(" ***TESTS PASSED***\n");


    printf("\n=== ABUNDANCE MATCHING TESTS ===\n");
    

}