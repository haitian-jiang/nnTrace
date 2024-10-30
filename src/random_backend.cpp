#include "random_backend.h"


torch::Tensor seed_to_rand(const torch::Tensor &seeds) {
    assert(seeds.dtype() == torch::kUInt64);
    assert(seeds.is_contiguous());

    torch::TensorOptions opt = torch::TensorOptions().dtype(torch::kFloat32);
    auto rand_tensor = torch::zeros_like(seeds, opt);
    int64_t numel = seeds.numel();
    const uint64_t* seed_data = seeds.const_data_ptr<uint64_t>();
    float* rand_data = rand_tensor.data_ptr<float>();

    Random random_generator;

    for (int64_t i = 0; i < numel; ++i) {
        random_generator.SetSeed(seed_data[i]);
        rand_data[i] = random_generator.GetRealNonzero();
    }
    return rand_tensor;
}