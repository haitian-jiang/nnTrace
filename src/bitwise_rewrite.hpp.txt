# pragma once
#include <torch/extension.h>

void rewrite_low_bits_bf16(  // a wrapper for the template function
    torch::Tensor activation,
    torch::Tensor seeds,
    int N 
);
