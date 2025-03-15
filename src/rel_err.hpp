#include <torch/extension.h>
#include <vector>

using std::vector;
using std::tuple;
using std::string;
using torch::Tensor;

tuple<string, double> tensor_rel_err_with_name(
    const string &name,
    const Tensor &specimen,
    const Tensor &oracle,
    const string &norm = "fro"
);

vector<tuple<string, double>> parallel_rel_err(
    const vector<tuple<string, Tensor, Tensor>> &input_list
);
