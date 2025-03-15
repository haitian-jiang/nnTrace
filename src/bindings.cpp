#include <pybind11/pybind11.h>
#include "random_backend.h"
#include "rel_err.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cpp_ext, m) {
    m.doc() = "Python bindings for parallel consistent random number generation.";
    // m.def("seed_to_rand_omp", &seed_to_rand_omp, "Elementwise independent random tensor generation from seed tensor of the same shape.");
    m.def("seed_to_rand_parallel", &seed_to_rand_parallel, "Docs");
    m.def("tensor_rel_err_with_name", &tensor_rel_err_with_name, 
          py::arg("name"), py::arg("specimen"), py::arg("oracle"), 
          py::arg("norm") = "fro",
          "Docs");
}
