#include <torch/extension.h>

std::tuple<std::string, double> tensor_rel_err_with_name(
    const std::string &name,
    const torch::Tensor &specimen,
    const torch::Tensor &oracle,
    const std::string &norm = "fro"
);

