#pragma once

#include <torch/extension.h>

int add(int a, int b);

torch::Tensor add_tensors(const torch::Tensor& a, const torch::Tensor& b);