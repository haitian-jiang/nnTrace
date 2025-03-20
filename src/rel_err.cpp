#include <execution>
#include "rel_err.hpp"

using std::get;
using std::tuple;
using std::string;
using torch::Tensor;
using std::vector;


tuple<string, double> tensor_rel_err_with_name(
    const string &name,
    const Tensor &specimen,
    const Tensor &oracle,
    const string &norm
) {
    torch::set_num_threads(1);
    Tensor specimen_fp64 = specimen.to(torch::kFloat64);
    Tensor oracle_fp64 = oracle.to(torch::kFloat64);
    Tensor mask = specimen_fp64.isnan() | oracle_fp64.isnan() | 
                  specimen_fp64.isinf() | oracle_fp64.isinf();

    specimen_fp64.masked_fill_(mask, 0);
    oracle_fp64.masked_fill_(mask, 0);

    Tensor abs_err, oracle_norm, rerr;
    if (norm == "fro") {
        abs_err = (specimen_fp64 - oracle_fp64).norm();
        oracle_norm = oracle_fp64.norm();
    } else if (norm == "1") {
        abs_err = (specimen_fp64 - oracle_fp64).abs().sum();
        oracle_norm = oracle_fp64.abs().sum();
    } else if (norm == "inf") {
        abs_err = (specimen_fp64 - oracle_fp64).abs().max();
        oracle_norm = oracle_fp64.abs().max();
    } else {
        throw std::invalid_argument("Unsupported norm type: " + norm);
    }
    if (oracle_norm.item<double>() < 1e-14 * oracle_fp64.numel()) {
        if (abs_err.item<double>() < 2e-14 * abs_err.numel()) {
            return std::make_tuple(name, 0);
        } else {
            return std::make_tuple(name, 615.0);
        }
    } else {
        rerr = abs_err / oracle_norm;
    }

    return std::make_tuple(name, rerr.item<double>());
}


vector<tuple<string, double>> parallel_rel_err(
    const vector<tuple<string, Tensor, Tensor>> &input_list
) {
    vector<tuple<string, double>> output_list(input_list.size());

    std::transform(
        std::execution::par,
        input_list.begin(),
        input_list.end(),
        output_list.begin(),
        [](const tuple<string, Tensor, Tensor> &input) {
            return tensor_rel_err_with_name(
                get<0>(input), get<1>(input), get<2>(input), "fro"
            );
        }
    );

    return output_list;
}
