enum SampleType {
  QUIESCENT,
  STARFORMING,
  ALL,
  NONE
};

void setup_rng();
int poisson_deviate(float mean);
int poisson_deviate_old(float mean);
void print_hod(const char* filename);
void tabulate_hods();
void populate_simulation_omp(int imag, SampleType type);
void lsat_model();
void lsat_model_scatter();
