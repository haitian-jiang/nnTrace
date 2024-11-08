cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
project(random_backend)

# Set the Python executable to the one in the conda environment
set(Python3_EXECUTABLE $ENV{CONDA_PREFIX}/bin/python)  # need if/else for non-conda env
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
    OUTPUT_VARIABLE TORCH_PATH
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})

# show torch version
execute_process(
    COMMAND python -c "import torch; print(torch.__version__, end='')"
    OUTPUT_VARIABLE TORCH_VERSION
)

find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
message(STATUS "Found torch: version ${TORCH_VERSION}")
message(STATUS "torch library: ${TORCH_LIBRARY}")
message(STATUS "torch include: ${TORCH_INCLUDE_DIRS}")


# Create the C++ Python module, linking necessary files
file(GLOB SRC_FILES src/*.cpp)
pybind11_add_module(${PROJECT_NAME} ${SRC_FILES})

# Find libtorch_python.so for linux, but adding this for mac will cause symbol not found
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    message(STATUS "Compiling on Linux")
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
    message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_PYTHON_LIBRARY})
endif()

# Link PyTorch to the module
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
target_include_directories(${PROJECT_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)

# To suppress warnings in PyTorch, you might want to set
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")