#include <pybind11/pybind11.h>
#include "random_backend.h"

namespace py = pybind11;

PYBIND11_MODULE(random_backend, m) {
    m.doc() = "Python bindings for high-performance C++ functions";  // Optional module docstring
    m.def("seed_to_rand_cxx", &seed_to_rand, "Elementwise independent random tensor generation from seed tensor of the same shape.");
    m.def("seed_to_rand_parallel", &seed_to_rand_parallel, "Docs");
}