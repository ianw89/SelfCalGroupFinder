enum SampleType {
  QUIESCENT,
  STARFORMING,
  ALL,
  NONE
};

int poisson_deviate(float nave, int thisTask);
int poisson_deviate_old(float nave, int thisTask);
void print_hod(const char* filename);
void tabulate_hods();
void populate_simulation_omp(int imag, enum SampleType type, int thisTask);
void lsat_model();
void lsat_model_scatter();
