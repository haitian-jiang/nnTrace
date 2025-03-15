#include "rel_err.hpp"


std::tuple<std::string, double> tensor_rel_err_with_name(
    const std::string &name,
    const torch::Tensor &specimen,
    const torch::Tensor &oracle,
    const std::string &norm
) {
    torch::Tensor specimen_fp64 = specimen.to(torch::kFloat64);
    torch::Tensor oracle_fp64 = oracle.to(torch::kFloat64);
    torch::Tensor mask = specimen_fp64.isnan() | oracle_fp64.isnan() | 
                         specimen_fp64.isinf() | oracle_fp64.isinf();

    specimen_fp64.masked_fill_(mask, 0);
    oracle_fp64.masked_fill_(mask, 0);

    torch::Tensor rerr;
    if (norm == "fro") {
        rerr = (specimen_fp64 - oracle_fp64).norm() / oracle_fp64.norm();
    } else if (norm == "1") {
        rerr = (specimen_fp64 - oracle_fp64).abs().sum() / oracle_fp64.abs().sum();
    } else if (norm == "inf") {
        rerr = (specimen_fp64 - oracle_fp64).abs().max() / oracle_fp64.abs().max();
    } else {
        throw std::invalid_argument("Unsupported norm type: " + norm);
    }

    return std::make_tuple(name, rerr.item<double>());
}