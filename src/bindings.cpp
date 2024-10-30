#include <pybind11/pybind11.h>
#include "example.h"  

namespace py = pybind11;

PYBIND11_MODULE(_neck, m) {
    m.doc() = "Python bindings for high-performance C++ functions";  // Optional module docstring
    m.def("add", &add, "A function that adds two numbers");
    m.def("add_tensors", &add_tensors, "A function that adds two tensors");
}