#include <algorithm>
#include <execution>
#include <omp.h>
#include "random_backend.h"

/*
Bit mixing random number generator.  Cannot use linear congruential generator.
Has to map adjacent seeds to distant random numbers.
https://xorshift.di.unimi.it/splitmix64.c
*/
inline float fill_random(int64_t seed) {
    union {
        uint64_t i;
        double d;
    } tmp;

    uint64_t z = static_cast<uint64_t>(seed); // For compatability with torch<=2.1.0
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    z = z ^ (z >> 31);
    tmp.i = (z >> 12) | 0x3ff0000000000000ul;
    return static_cast<float>(2.0 - tmp.d);
}

torch::Tensor seed_to_rand(const torch::Tensor &seeds) {
    // assert(seeds.dtype() == torch::kUInt64);
    assert(seeds.is_contiguous());

    torch::TensorOptions opt = torch::TensorOptions().dtype(torch::kFloat32);
    auto rand_tensor = torch::zeros_like(seeds, opt);
    int64_t numel = seeds.numel();
    const int64_t* seed_data = seeds.const_data_ptr<int64_t>();
    float* rand_data = rand_tensor.data_ptr<float>();

    #pragma omp parallel for
    for (int64_t i = 0; i < numel; ++i) {
        rand_data[i] = fill_random(seed_data[i]);
    }
    return rand_tensor;
}


torch::Tensor seed_to_rand_parallel(const torch::Tensor &seeds) {
    // assert(seeds.dtype() == torch::kUInt64);
    assert(seeds.is_contiguous());

    torch::TensorOptions opt = torch::TensorOptions().dtype(torch::kFloat32);
    auto rand_tensor = torch::zeros_like(seeds, opt);
    int64_t numel = seeds.numel();
    const int64_t* seed_data = seeds.const_data_ptr<int64_t>();
    float* rand_data = rand_tensor.data_ptr<float>();

    std::transform(std::execution::par_unseq, seed_data, seed_data + numel, rand_data, fill_random);

    return rand_tensor;
}