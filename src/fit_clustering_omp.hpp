enum SampleType {
  QUIESCENT,
  STARFORMING,
  ALL,
  NONE
};

void prepare_halos();
int poisson_deviate(float mean, struct drand48_data *rng);
void print_hod(const char* filename);
void tabulate_hods();
void populate_simulation_omp(int imag, SampleType type);
void lsat_model();
void lsat_model_scatter();
