cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
project(neck)

# Set the Python executable to the one in the conda environment
set(Python3_EXECUTABLE $ENV{CONDA_PREFIX}/bin/python)
# Find Python interpreter and development files in the current environment
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Use pybind11's cmake module to locate the installation path for pybind11
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})


# add torch into cmake path
execute_process(
        COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
        OUTPUT_VARIABLE TorchPath
)
list(APPEND CMAKE_PREFIX_PATH ${TorchPath})

# show torch version
# execute_process(
#         COMMAND python -c "import torch; print(torch.__version__, end='')"
#         OUTPUT_VARIABLE TorchVersion
# )
# message(STATUS "Torch Version: ${TorchVersion}")

find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
message(STATUS "torch library: ${TORCH_LIBRARY}")
message(STATUS "Torch Include: ${TORCH_INCLUDE_DIRS}")


# Create the C++ Python module, linking necessary files
pybind11_add_module(_neck src/bindings.cpp src/example.cpp)

# Link PyTorch to the module
target_link_libraries(_neck PRIVATE ${TORCH_LIBRARIES})
target_include_directories(_neck PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_features(_neck PRIVATE cxx_std_14)

# To suppress warnings in PyTorch, you might want to set
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")