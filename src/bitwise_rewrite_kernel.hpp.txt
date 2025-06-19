#pragma once

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/BFloat16.h>
#include <torch/extension.h>
#include "random_backend.hpp"


/*
Bit mixing random number generator.  Cannot use linear congruential generator.
Has to map adjacent seeds to distant random numbers.
https://xorshift.di.unimi.it/splitmix64.c
*/
__host__ __device__ inline float fill_random(int64_t seed) {
    union {
        uint64_t i;
        double d;
    } tmp;

    uint64_t z = static_cast<uint64_t>(seed); // Compatabile with torch<=2.1.0
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    z = z ^ (z >> 31);

    /* (z / UINT64_MAX) using bit manipulation
    * See https://github.com/NetSys/bess/blob/master/core/utils/random.h#L75
    * From the MSB,
    * 0: sign
    * 1-11: exponent (0x3ff == 0, 0x400 == 1)
    * 12-63: mantissa
    * The resulting double number is 1.(b0)(b1)...(b47),
    * where seed_ is (b0)(b1)...(b63).
    */
    tmp.i = (z >> 12) | 0x3ff0000000000000ul;
    return static_cast<float>(2.0 - tmp.d);
}



template <int N>
__global__ void rewrite_low_bits_bf16_kernel(
    c10::BFloat16* activation,
    int64_t* seeds,
    int64_t size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }
    // Generate random number from seed
    float rand_val = fill_random(seeds[idx]);
    uint32_t rand_bits = *reinterpret_cast<uint32_t*>(&rand_val);

    uint16_t* bf16_bits = reinterpret_cast<uint16_t*>(&activation[idx]);
    // Mask for the lowest N bits, no overflow as we assume N < 7
    uint16_t mask = (1u << N) - 1u;

    // float32: s|eeeeeeee|rrrjjjjjjjjjjjjjjjjjjjj
    // take the r bits (N=3 here) and (23-N) junk bits to be discarded.
    uint16_t rand_bits_16 = static_cast<uint16_t>(rand_bits >> (23 - N));

    // Clear the lowest N bits of bf16 and set them to the new random bits
    *bf16_bits = (*bf16_bits & ~mask) | (rand_bits_16 & mask);
}

template <int N>
void _rewrite_low_bits_bf16(
    torch::Tensor activation,
    torch::Tensor seeds
) {
    const int64_t size = seeds.numel();
    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;

    rewrite_low_bits_bf16_kernel<N><<<blocks, threads_per_block>>>(
        activation.data_ptr<c10::BFloat16>(),
        seeds.data_ptr<int64_t>(),
        size
    );
}
