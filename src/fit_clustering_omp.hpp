enum SampleType {
  QUIESCENT,
  STARFORMING,
  ALL,
  NONE
};

#define MAXBINS 10 // This is the max number of magnitude bins we use for the HOD. The actual amount we use is read in from VOLUME_BINS_FILE
#define HALO_BINS 200 // log10(M_halo) / 0.1

struct hod {
  int NVOLUME_BINS = 0;
  // TODO replace these with dynamic sized arrays
  // These are the HOD in bins
  double ncenr[MAXBINS][HALO_BINS], nsatr[MAXBINS][HALO_BINS], ncenb[MAXBINS][HALO_BINS], nsatb[MAXBINS][HALO_BINS], ncen[MAXBINS][HALO_BINS], nsat[MAXBINS][HALO_BINS], nhalo[MAXBINS][HALO_BINS];
  double nhalo_int[MAXBINS][HALO_BINS]; // integer version of nhalo (no vmax weight)
  float maglim[MAXBINS]; // the fainter limit of the mag bin
  float magmax[MAXBINS]; // the brighter limit of the mag bin
  int color_sep[MAXBINS]; // 1 means do red/blue seperately, 0 means all together
  float maxz[MAXBINS];  // the max redshift of the mag bin, calculated from the fainter mag limit
  float volume[MAXBINS]; // volume of the mag bin in [Mpc/h]^3
};

extern struct hod HOD;

void prepare_halos();
int poisson_deviate(float mean, struct drand48_data *rng);
void print_hod(const char* filename);
void tabulate_hods();
void populate_simulation_omp(int imag, SampleType type);
void lsat_model();
void lsat_model_scatter();
