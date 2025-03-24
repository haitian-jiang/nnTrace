#include <torch/extension.h>

torch::Tensor seed_to_rand_omp(const torch::Tensor &seeds);

torch::Tensor seed_to_rand_parallel(const torch::Tensor &seeds);
