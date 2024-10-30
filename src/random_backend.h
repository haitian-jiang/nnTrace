#include "rng.h"
#include <torch/extension.h>

torch::Tensor seed_to_rand(const torch::Tensor &seeds);