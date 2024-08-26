// Definitions
#define OMEGA_M 0.25
#define PI 3.141592741
#define RHO_CRIT 2.775E+11
#define DELTA_HALO 200
#define SPEED_OF_LIGHT 3.0E+5
#define c_on_H0 2997.92
#define BIG_G 4.304E-9 /* BIG G in units of (km/s)^2*Mpc/M_sol */
#define G0 (1.0 / sqrt(2.0 * 3.14159))
#define ROOT2 1.41421
#define Q0 2.0
#define Q1 -1.0
#define QZ0 0.1
#define THIRD (1.0 / 3.0)
#define ANG (PI / 180.0)
#define RT2PI 2.50663

/* Structure definition for galaxies. */
extern struct galaxy {
  float x,y,z;
  float ra, dec; // in radians
  float redshift; 
  float rco; // comoving distnace (in Mpc??) set from distance_redshift(z)
  float mstellar, // mstellar might mean luminosity as well
    psat,
    color, // greater than 0.8 means red, otherwise blue
    propx,
    propx2,
    weight,
    vmax;
  int igrp;
  //int id;
  int listid;
  int next;
  int grp_rank;

  // halo properties  
  float mass,
    theta,
    rad,
    sigmav,
    mtot, // tracks total mstellar or luminosity of group
    nsat;
} *GAL;

/* Structure for the halos in the simulations */
extern struct halo {
  float x,y,z,vx,vy,vz,mass,lsat;
  int nsat;
} *HALO;
extern int NHALO;

/* The master array of galaxies */
struct galaxy *GAL;
extern int NGAL;

/* Options and general purpose globals */
extern int FLUXLIM;
extern int COLOR;
extern int USE_WCEN;
extern int USE_BSAT;
extern int STELLAR_MASS;
extern int SECOND_PARAMETER;
extern float FRAC_AREA;
extern float MAXREDSHIFT;
extern float MINREDSHIFT;
extern float GALAXY_DENSITY;
extern int SILENT;
extern int VERBOSE;
extern int RECENTERING;
extern int POPULATE_MOCK;
extern char *INPUTFILE;
extern char *HALO_MASS_FUNC_FILE;

/* Variables for determining threshold if a galaxy is a satellite */
extern const float BPROB_DEFAULT;
extern float BPROB_RED, BPROB_XRED;
extern float BPROB_BLUE, BPROB_XBLUE;

/* Variables for weighting of assigned halo masses for blue vs red centrals */
extern float WCEN_MASS, WCEN_SIG, WCEN_MASSR, WCEN_SIGR, WCEN_NORM, WCEN_NORMR;

/* Variables for affecting individual galaxy weights when assigning halo mass */
extern float PROPX_WEIGHT_RED, PROPX_WEIGHT_BLUE, PROPX_SLOPE_RED, PROPX_SLOPE_BLUE;
extern float PROPX2_WEIGHT_RED, PROPX2_WEIGHT_BLUE;

/* Imported functions from numerical recipes 
 */
float qromo(float (*func)(float), float a, float b,
       float (*choose)(float(*)(float), float, float, int));
float midpnt(float (*func)(float), float a, float b, int n);
void spline(float x[], float y[], int n, float yp1, float ypn, float y2[]);
void splint(float xa[], float ya[], float y2a[], int n, float x, float *y);
void sort2(int n, float arr[], int id[]);
float qtrap(float (*func)(float), float a, float b);
float gasdev(long *idum);

/* other functions shared by multiple files
 */
void update_galaxy_halo_props(struct galaxy *galaxy);

void groupfind(void);
float distance_redshift(float z);
float density2host_halo_zbins3(float z, float vmax);
float density2host_halo(float galaxy_density);
int search(int n, float *x, float val);
void test_centering(void *kd);
int group_center(int icen0, void *kd);
float angular_separation(float a1, float d1, float a2, float d2);
void test_fof(void *kd);
float compute_prob_rad(float dz, float sigmav);
float radial_probability(float mass, float dr, float rad, float ang_rad);
float radial_probability_g(struct galaxy *gal, float dr);
float psat(struct galaxy *central, float dr, float dz, float bprob);

