#include <pybind11/pybind11.h>
#include "random_backend.hpp"
#include "rel_err.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cpp_ext, m) {
    m.doc() = "Python bindings for parallel consistent random number generation.";
    m.def("seed_to_rand_parallel", &seed_to_rand_parallel, "Docs");
    m.def("tensor_rel_err_with_name", &tensor_rel_err_with_name, 
          py::arg("name"), py::arg("specimen"), py::arg("oracle"), 
          py::arg("norm") = "fro", "Docs");
    m.def("parallel_rel_err", &parallel_rel_err, "Docs");
}
