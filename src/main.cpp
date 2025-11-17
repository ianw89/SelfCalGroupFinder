#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <argp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <sys/time.h>
#include "timing.hpp"
#include <stdint.h>
#include <unistd.h>
#include <vector>
#include "groups.hpp"
#include "fit_clustering_omp.hpp"
#include "nrutil.h"
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

void write_fsat();
bool await_request();

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
  {"fluxlim",      'f', "MAG,MODEL",                          0,  "Indicate a flux limited sample, and what model to use for correcting group luminosity for inverse-SHAM.", 1},
  {"stellarmass",  'm', 0,                                    0,  "Abundance match on stellar mass, not luminosity", 1},
  {"popmock",      'p', "MOCKFILE,BINSFILE",                  0,  "Populate a mock catalog after group finding, provide mock file path and bin definitions", 1},
  {"colors",       'c', 0,                                    0,  "Read in and use galaxy colors", 1},
  {"iterations",   'i', "N",                                  0,  "Number of iterations for group finding", 1},
  {"earlyexit",    'e', 0,                                    0,  "Allow early exit from group finding for various scenarios", 1},
  {"wcen",         'w', "MASS,SIGMA,MASSR,SIGMAR,NORM,NORMR", 0,  "Six parameters for weighting the centrals", 2},
  {"bsat",         'b', "RED,XRED,BLUE,XBLUE",                0,  "Four parameters for the satellite probability", 2},
  {"chi1",         'x', "WEIGHT_B,WEIGHT_R,SLOPE_B,SLOPE_R",  0,  "Four parameters per-galaxy extra property weighting", 2},
  {"verbose",      'v', 0,                                    0,  "Produce verbose stderr logging", 3},
  {"quiet",        'q', 0,                                    0,  "Don't produce any output to stderr", 3 },
  {"silent",       's', 0,                                    OPTION_ALIAS },
  {"pipe",         'P', "PIPEID",                             0,  "Specify a pipe ID for the group find to write message to", 3},
  {"interactive",  'k', 0,                                    0,  "Do not terminate after group finding. Use pipe messages to control.", 3},
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
  struct arguments *arguments = (struct arguments *) (state->input);
  int pipe_id;

  switch (key)
  {
    case 'f':
      FLUXLIM = 1;
      sscanf(arg, "%f,%d", &(FLUXLIM_MAG), &(FLUXLIM_CORRECTION_MODEL));
      break;
    case 'm':
      STELLAR_MASS = 1;
      break;
    case 'c':
      COLOR = 1;
      break;
    case 'i':
      MAX_ITER = atoi(arg);
      break;
    case 'p':
      POPULATE_MOCK = 1;
      if (arg != NULL) {
          char *token = strtok(arg, ",");
          if (token != NULL) {
            MOCK_FILE = strdup(token);
            token = strtok(NULL, ",");
            if (token != NULL) {
              VOLUME_BINS_FILE = strdup(token);
            }
          } else {
            LOG_ERROR("Mock file and volume bins file must be specified with -p option.\n");
            exit(EPERM);
          }

      } else {
        LOG_ERROR("Mock file and volume bins file must be specified with -p option.\n");
        exit(EPERM);
      }
      break;
    case 'v':
      VERBOSE = 1;
      break; 
    case 'q': case 's':
      SILENT = 1;
      break;
    case 'e':
      ALLOW_EARLY_EXIT = 1;
      break;
    case 'w':
      USE_WCEN = 1;
      arguments->wcen_set = 1;
      sscanf(arg, "%lf,%lf,%lf,%lf,%lf,%lf", &(WCEN_MASS), &(WCEN_SIG), &(WCEN_MASSR), &(WCEN_SIGR), &(WCEN_NORM), &(WCEN_NORMR));
      break;
    case 'b':
      USE_BSAT = 1;
      arguments->bsat_set = 1;
      sscanf(arg, "%lf,%lf,%lf,%lf", &(BPROB_RED), &(BPROB_XRED), &(BPROB_BLUE), &(BPROB_XBLUE));
      break;
    case 'x':
      arguments->chi1_set = 1;
      sscanf(arg, "%lf,%lf,%lf,%lf", &(PROPX_WEIGHT_BLUE), &(PROPX_WEIGHT_RED), &(PROPX_SLOPE_BLUE), &(PROPX_SLOPE_RED));
      break;
    case 'h':
      HALO_MASS_FUNC_FILE = arg;
      break;
    case 'k':
      INTERACTIVE = 1;
      break;
    case 'P':
      if (arg == NULL)
      {
        LOG_ERROR("Pipe ID must be specified with -P option.\n");
        exit(EPERM);
      }
      pipe_id = atoi(arg);
      if (pipe_id < 0)
      {
        LOG_ERROR("Invalid pipe ID: %d\n", pipe_id);
        exit(EPERM);
      }
      MSG_PIPE = fdopen(pipe_id, "w");
      if (!MSG_PIPE) 
          perror("fdopen");
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
  double t0, t1, t2, t3, t_grp_s, t_grp_e;
  int istart, istep;
  int i;
  struct arguments arguments;
  arguments.wcen_set = 0;
  arguments.bsat_set = 0;
  arguments.chi1_set = 0;

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
  LOG_INFO("input> FLUXLIM: %d, COLOR: %d, STELLAR_MASS: %d \n", FLUXLIM, COLOR, STELLAR_MASS);
  LOG_INFO("input> z: %f-%f, frac_area: %f\n", MINREDSHIFT, MAXREDSHIFT, FRAC_AREA);
  if (USE_WCEN)
    LOG_INFO("input> wcen ON: %f %f %f %f %f %f\n", WCEN_MASS, WCEN_SIG, WCEN_MASSR, WCEN_SIGR, WCEN_NORM, WCEN_NORMR);
  else 
    LOG_INFO("input> wcen OFF\n");
  if (USE_BSAT)
    LOG_INFO("input> Bsat ON: %f %f %f %f\n", BPROB_RED, BPROB_XRED, BPROB_BLUE, BPROB_XBLUE);
  else 
    LOG_INFO("input> Bsat OFF\n");

  LOG_INFO("input> SECOND_PARAMETER= %d\n", SECOND_PARAMETER);
  if (SECOND_PARAMETER)
      LOG_INFO("input> %f %f %f %f\n", PROPX_WEIGHT_BLUE, PROPX_WEIGHT_RED, PROPX_SLOPE_BLUE, PROPX_SLOPE_RED);

  // Print Warnings
  if (POPULATE_MOCK && !COLOR)
  {
    LOG_ERROR("Populating mock should only be used when galaxy colors are provided (-c).\n");
    exit(EPERM);
  }
  if (USE_WCEN && !COLOR) 
    LOG_WARN("Weighting centrals but not using galaxy colors. All galaxies will use blue wcen values.\n");
  if (USE_BSAT && !COLOR) 
    LOG_WARN("Using custom Bsat but not using galaxy colors. All galaxies will use blue Bsat values.\n");
  if (INTERACTIVE && MSG_PIPE == NULL) 
  {
    LOG_ERROR("Interactive mode (-k) requires a pipe ID (-P).\n");
    exit(EPERM);
  }

  bool run = true;
  while (run)
  {
    if (!INTERACTIVE) 
      run = false;
      
    // The primary method for group finding
    t_grp_s = get_wtime();
    groupfind();
    t_grp_e = get_wtime();
    LOG_PERF("groupfind() took %.2f sec\n", t_grp_e - t_grp_s);

    // Populate Mock 
    if (POPULATE_MOCK)
    {
      t0 = get_wtime();
      lsat_model();
      tabulate_hods();
      prepare_halos();
      t1 = get_wtime();
      LOG_PERF("lsat + hod + prep popsim: %.2f sec\n", t1 - t0);

      // lsat_model_scatter(); // This is crashing for some reason...

      LOG_INFO("Populating mock catalog\n");

      t2 = get_wtime();
      
      //for (i = 0; i < HOD.NVOLUME_BINS*3; i += 1)
      //{
      //  populate_simulation_omp(i/3, static_cast<SampleType>(i%3));
      //}
      #pragma omp parallel private(i,istart,istep)
      {
        istart = get_thread_num();
        istep = get_num_threads();
        for(i=istart; i< HOD.NVOLUME_BINS*3; i+=istep)
        {
          populate_simulation_omp(i/3, static_cast<SampleType>(i%3));
        }
      }

      t3 = get_wtime();
      LOG_INFO("popsim> %.2f sec\n", t3 - t2);

    }
    
    write_fsat();

    // Done. Send pipe message that's were done and await instructions
    if (MSG_PIPE != NULL) 
    {
      //LOG_INFO("Group finding completed, sending message and awaiting next request...\n");
      uint8_t resp_msg_type = MSG_COMPLETED;
      uint8_t resp_data_type = TYPE_FLOAT;
      uint32_t resp_count = 0;

      fwrite(&resp_msg_type, 1, 1, MSG_PIPE);
      fwrite(&resp_data_type, 1, 1, MSG_PIPE);
      fwrite(&resp_count, sizeof(uint32_t), 1, MSG_PIPE);
      fflush(MSG_PIPE);
    }

    if (INTERACTIVE) {
      run = await_request();
    }

  }

  // FINAL CLEANUP
  if (MSG_PIPE != NULL) 
  {
    fflush(MSG_PIPE);
    fclose(MSG_PIPE);
  }
}

bool await_request() {
  uint8_t msg_type, data_type;
  uint32_t count;
  std::vector<double> params;

  // Read header: 1 byte msg_type, 1 byte data_type, 4 bytes count (little-endian)
  // TODO end gracefully when stdin is closed
  //if (feof(stdin)) {
  //  LOG_INFO("End of input stream reached, exiting...\n");
  //  return false; // No more requests
  //}
  uint8_t header[6];
  size_t n = fread(header, 1, 6, stdin);
  if (n != 6) {
    LOG_ERROR("Failed to read MSG_REQUEST header (got %zu bytes)\n", n);
    return false;
  }
  msg_type = header[0];
  data_type = header[1];
  memcpy(&count, header + 2, sizeof(uint32_t));
  if (msg_type != MSG_REQUEST) {
    LOG_ERROR("Expected MSG_REQUEST, got %d\n", msg_type);
    return false;
  }
  if (data_type != TYPE_DOUBLE) {
    LOG_ERROR("Expected TYPE_DOUBLE, got %d\n", data_type);
    return false;
  }

  // Read payload: count doubles
  params.resize(count);
  size_t n_payload = fread(params.data(), sizeof(double), count, stdin);
  if (n_payload != count) {
    fprintf(stderr, "Failed to read MSG_REQUEST payload (got %zu doubles)\n", n_payload);
    return false;
  }

  if (params.size() == 10) {
    assert(USE_WCEN && USE_BSAT && SECOND_PARAMETER == 0);
    WCEN_MASS = params[0];
    WCEN_SIG = params[1];
    WCEN_MASSR = params[2];
    WCEN_SIGR = params[3];
    WCEN_NORM = params[4];
    WCEN_NORMR = params[5];
    BPROB_RED = params[6];
    BPROB_XRED = params[7];
    BPROB_BLUE = params[8];
    BPROB_XBLUE = params[9];
  } else {
    LOG_ERROR("Unexpected number of parameters in MSG_REQUEST: %zu\n", params.size());
    return false;
  }

  return true;
}

void write_fsat() {
  // want L bins to be np.logspace(6, 12.5, 40) like in python postprocessing
  const int FSAT_BINS = 40;
  float logbin_interval = (12.5 - 6.0) / FSAT_BINS;
  float numr[FSAT_BINS], numb[FSAT_BINS], satsr[FSAT_BINS], satsb[FSAT_BINS], fsat[FSAT_BINS], fsatr[FSAT_BINS], fsatb[FSAT_BINS];
  int ibin, i;
  float nsats = 0;
  for (i = 0; i < FSAT_BINS; ++i)
  {
    numr[i] = numb[i] = satsr[i] = satsb[i] = fsat[i] = fsatr[i] = fsatb[i] = 0;
  }

  for (i = 0; i < NGAL; ++i)
  {
    ibin = (int)((GAL[i].loglum - 6.0) / logbin_interval);
    //fprintf(stderr, "GAL %d L=%e bin=%d\n", i, GAL[i].lum, ibin);

    if (ibin < 0 || ibin >= FSAT_BINS)
      continue;
    if (GAL[i].color > 0.8)
    {
      numr[ibin] += 1/GAL[i].vmax;
      if (GAL[i].psat > 0.5) {
        satsr[ibin] += 1/GAL[i].vmax;
      }
    }
    else
    {
      numb[ibin] += 1/GAL[i].vmax;
      if (GAL[i].psat > 0.5) {
        satsb[ibin] += 1/GAL[i].vmax;
      }
    }
  }

  for (i = 0; i < FSAT_BINS; ++i)
  {
    fsat[i] = (satsr[i] + satsb[i]) / (numr[i] + numb[i] + 1E-20);
    fsatr[i] = satsr[i] / (numr[i] + 1E-20);
    fsatb[i] = satsb[i] / (numb[i] + 1E-20);
    nsats += satsr[i] + satsb[i];
  }

  if (MSG_PIPE != NULL) 
  {
    LOG_INFO("Writing fsat to pipe\n");
    uint8_t resp_msg_type = MSG_FSAT;
    uint8_t resp_data_type = TYPE_FLOAT;
    uint32_t resp_count = FSAT_BINS * 3;
    fwrite(&resp_msg_type, 1, 1, MSG_PIPE);
    fwrite(&resp_data_type, 1, 1, MSG_PIPE);
    fwrite(&resp_count, sizeof(uint32_t), 1, MSG_PIPE);
    fwrite(&fsat, sizeof(float), FSAT_BINS, MSG_PIPE);
    fwrite(&fsatr, sizeof(float), FSAT_BINS, MSG_PIPE);
    fwrite(&fsatb, sizeof(float), FSAT_BINS, MSG_PIPE);
    fflush(MSG_PIPE);
  }
  else if (!SILENT) 
  {
    fprintf(stderr, "fsat total: %f\n", (float)nsats / (float)NGAL);
    //fprintf(stderr, "fsat> bin fsat fsatr fsatb\n");
    for (i = 0; i < FSAT_BINS; ++i)
    {
      fprintf(stderr, "fsat> %d %f %f %f\n", i, fsat[i], fsatr[i], fsatb[i]);
    }
  }


      
}