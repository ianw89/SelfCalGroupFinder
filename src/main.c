#include <argp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <sys/time.h>
#include <omp.h>
#include "nrutil.h"
#include "kdtree.h"
#include "groups.h"
#include "fit_clustering_omp.h"

/* Main Method and argument parsing for command-line executable kdGroupFinder
*/

// Input file format
// 1 galaxy per line
// RA DEC REDSHIFT MAINPROP [VMAX] [COLOR] [PROP1] [PROP2]
// MAINPROP can be either LUMINOSITY OR STELLAR MASS (then provide -m), 
// OR LOG10 OF EITHER (log is automatically detected)
// If -f for fluxlim mode is given, VMAX is required
// If -c for colors mode is given, the next color will be read as COLOR values. 
//   If the color is less than 0.8, the galaxy is considered blue, otherwise red.
//   When not provided, all galaxies are treated as blue.
// PROP1, PROP2 are optional 

// stdout is used for printing the output of the program; 1 galaxy per line
// stderr is used for printing the progress of the program and errors


/* Standard GNU command-line program stuff */
const char *argp_program_version =  "kdGroupFinder-2.0";
const char *argp_program_bug_address = "<imw2293@nyu.edu>";
static char doc[] = "kdGroupFinder: A self-calibrated galaxy group finder\
\vInput file format is 1 galaxy per line\n.\
 RA DEC REDSHIFT MAINPROP [VMAX] [COLOR] [PROP1] [PROP2]\n\
 MAINPROP can be either luminosity OR stellar mass (then provide -m),\
 or log10 of either (automatically detected).\
 If -f for fluxlim mode is given, a VMAX column is required.\
 If -c for colors mode is given, the a color column is required. \
 If the color is less than 0.8, the galaxy is considered blue, otherwise red.\
 When not provided, all galaxies are treated as blue.\
 PROP1, PROP2 are optional and not fully supported yet."; 
 
static char args_doc[] = "inputfile zmin zmax frac_area";

static struct argp_option options[] = {
  {"halomassfunc", 'h', "FILE",                               0,  "File containing the halo mass function", 1},
  {"fluxlim",      'f', "MODEL",                              0,  "Indicate a flux limited sample, and what model to use for correcting group luminosity for inverse-SHAM.", 1},
  {"stellarmass",  'm', 0,                                    0,  "Abundance match on stellar mass, not luminosity", 1},
  {"popmock",      'p', "MOCKFILE",                           0,  "Populate a mock catalog after group finding; provide the filepath", 1},
  {"colors",       'c', 0,                                    0,  "Read in and use galaxy colors", 1},
  {"random",       'r', 0,                                    0,  "Randomly perturb luminosity/mstar for first group finding", 1 },
  {"iterations",   'i', "N",                                  0,  "Number of iterations for group finding", 1},
  {"wcen",         'w', "MASS,SIGMA,MASSR,SIGMAR,NORM,NORMR", 0,  "Six parameters for weighting the centrals", 2},
  {"bsat",         'b', "RED,XRED,BLUE,XBLUE",                0,  "Four parameters for the satellite probability", 2},
  {"chi1",         'x', "WEIGHT_B,WEIGHT_R,SLOPE_B,SLOPE_R",  0,  "Four parameters per-galaxy extra property weighting", 2},
  {"verbose",      'v', 0,                                    0,  "Produce verbose output", 3},
  {"quiet",        'q', 0,                                    0,  "Don't produce any output", 3 },
  {"silent",       's', 0,                                    OPTION_ALIAS },
  //{"output",   'o', "FILE", 0, "Output to FILE instead of standard output" },
  { 0 }
};

/* Used by main to communicate with parse_opt. Mostly it just sets globals. */
struct arguments
{
  int wcen_set, bsat_set, chi1_set;
};

/* Parse a single option. */
static error_t parse_opt (int key, char *arg, struct argp_state *state)
{
  /* Get the input argument from argp_parse, which we
     know is a pointer to our arguments structure. */
  struct arguments *arguments = state->input;

  switch (key)
  {
    case 'f':
      FLUXLIM = 1;
      FLUXLIM_CORRECTION_MODEL = atoi(arg);
      break;
    case 'm':
      STELLAR_MASS = 1;
      break;
    case 'c':
      COLOR = 1;
      break;
    case 'r':
      PERTURB = 1;
      break;
    case 'i':
      MAX_ITER = atoi(arg);
      break;
    case 'p':
      POPULATE_MOCK = 1;
      MOCK_FILE = arg;
      break;
    case 'v':
      VERBOSE = 1;
      break; 
    case 'q': case 's':
      SILENT = 1;
      break;
    case 'w':
      USE_WCEN = 1;
      arguments->wcen_set = 1;
      sscanf(arg, "%f,%f,%f,%f,%f,%f", 
             &(WCEN_MASS), &(WCEN_SIG), &(WCEN_MASSR), &(WCEN_SIGR), &(WCEN_NORM), &(WCEN_NORMR));
      break;
    case 'b':
      USE_BSAT = 1;
      arguments->bsat_set = 1;
      sscanf(arg, "%f,%f,%f,%f", 
             &(BPROB_RED), &(BPROB_XRED), &(BPROB_BLUE), &(BPROB_XBLUE));
      break;
    case 'x':
      arguments->chi1_set = 1;
      sscanf(arg, "%f,%f,%f,%f", 
             &(PROPX_WEIGHT_BLUE), &(PROPX_WEIGHT_RED), &(PROPX_SLOPE_BLUE), &(PROPX_SLOPE_RED));
      break;
    case 'h':
      HALO_MASS_FUNC_FILE = arg;
      break;

    case ARGP_KEY_ARG:
      if (state->arg_num >= 4)
        /* Too many arguments. */
        argp_usage (state);

      switch (state->arg_num)
      {
        case 0:
          INPUTFILE = arg;
          break;
        case 1:
          MINREDSHIFT = atof(arg); // only used in volume-limited mode TODO don't force you to give it
          break;
        case 2:
          MAXREDSHIFT = atof(arg); // only used in volume-limited mode TODO don't force you to give it
          break;
        case 3:
          FRAC_AREA = atof(arg);
          break;
      }

      break;

    case ARGP_KEY_END:
      if (state->arg_num < 4)
        /* Not enough arguments. */
        argp_usage (state);
      break;

    default:
      return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

/* Our argp parser. */
static struct argp argp = { options, parse_opt, args_doc, doc };

/* Entry point for the main group finding program. */
int main(int argc, char **argv)
{
  double t0, t1, t2, t3;
  int istart, istep;
  int i;
  struct arguments arguments;
  arguments.wcen_set = 0;
  arguments.bsat_set = 0;
  arguments.chi1_set = 0;
  HALO_MASS_FUNC_FILE = "halo_mass_function.dat"; // Default value

  /* Parse our arguments; will set all the global variables the code uses directly. */
  argp_parse (&argp, argc, argv, 0, 0, &arguments);

  if (arguments.chi1_set)
      SECOND_PARAMETER += 1;

   /*
  if(argc>19)
    {
      SECOND_PARAMETER=2;
      PROPX2_WEIGHT_BLUE = atof(argv[19]);
      PROPX2_WEIGHT_RED = atof(argv[20]);
    }
  */
  /*
  if (argc > 21)
    STELLAR_MASS = atoi(argv[21]);
  */

  // Summarize input
  if (!SILENT)
  {
    fprintf(stderr, "input> FLUXLIM: %d, COLOR: %d, STELLAR_MASS: %d \n", FLUXLIM, COLOR, STELLAR_MASS);
    fprintf(stderr, "input> z: %f-%f, frac_area: %f\n", MINREDSHIFT, MAXREDSHIFT, FRAC_AREA);
    if (USE_WCEN)
      fprintf(stderr, "input> wcen ON: %f %f %f %f %f %f\n", WCEN_MASS, WCEN_SIG, WCEN_MASSR, WCEN_SIGR, WCEN_NORM, WCEN_NORMR);
    else 
      fprintf(stderr, "input> wcen OFF\n");
    if (USE_BSAT)
        fprintf(stderr, "input> Bsat ON: %f %f %f %f\n", BPROB_RED, BPROB_XRED, BPROB_BLUE, BPROB_XBLUE);
    else 
      fprintf(stderr, "input> Bsat OFF\n");

    fprintf(stderr, "input> SECOND_PARAMETER= %d\n", SECOND_PARAMETER);
    if (SECOND_PARAMETER)
        fprintf(stderr, "input> %f %f %f %f\n", PROPX_WEIGHT_BLUE, PROPX_WEIGHT_RED, PROPX_SLOPE_BLUE, PROPX_SLOPE_RED);
  }

  // Print Warnings
  if (POPULATE_MOCK && !COLOR)
  {
    fprintf(stderr, "Populating mock should only be used when galaxy colors are provided (-c).\n");
    exit(0);
  }
  if (!SILENT && USE_WCEN && !COLOR) 
    fprintf(stderr, "Weighting centrals but not using galaxy colors. All galaxies will use blue wcen values.\n");
  if (!SILENT && USE_BSAT && !COLOR) 
    fprintf(stderr, "Using custom Bsat but not using galaxy colors. All galaxies will use blue Bsat values.\n");

  // The primary method for group finding
  groupfind();

  if (POPULATE_MOCK)
  {
    t0 = omp_get_wtime();
    lsat_model();
    tabulate_hods();
    populate_simulation_omp(-1, 0, 0);
    t1 = omp_get_wtime();
    if (!SILENT) fprintf(stderr, "lsat + hod + prep popsim: %.2f sec\n", t1 - t0);

    // lsat_model_scatter(); // This is crashing for some reason...

    if (!SILENT) fprintf(stderr, "Populating mock catalog\n");

    t2 = omp_get_wtime();
    //for (i = 0; i < 10; i += 1)
    //{
    //  populate_simulation_omp(i / 2, i % 2, 1);
    //}

#pragma omp parallel private(i,istart,istep)
    {
      istart = omp_get_thread_num();
      istep = omp_get_num_threads();
      for(i=istart;i<10;i+=istep)
      {
        populate_simulation_omp(i/2,i%2,istart);
      }
    }
    
    t3 = omp_get_wtime();
    if (!SILENT) fprintf(stderr, "popsim> %.2f sec\n", t3 - t2);
  }
}