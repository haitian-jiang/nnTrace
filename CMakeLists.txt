cmake_minimum_required(VERSION 3.12)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
project(cpp_ext LANGUAGES CXX CUDA)

# Find Python interpreter and development files in the current environment
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
# If Python3_EXECUTABLE is not set, manually check Conda and venv
if(NOT Python3_EXECUTABLE)
    if(DEFINED ENV{CONDA_PREFIX})
        set(Python3_EXECUTABLE $ENV{CONDA_PREFIX}/bin/python)
    elseif(DEFINED ENV{VIRTUAL_ENV})
        set(Python3_EXECUTABLE $ENV{VIRTUAL_ENV}/bin/python)
    else()
        message(FATAL_ERROR "Python3 executable not found! Please activate a Python environment.")
    endif()
endif()
message(STATUS "Using Python interpreter: ${Python3_EXECUTABLE}")

# Use pybind11's cmake module to locate the installation path for pybind11
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})

# add torch into cmake path
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
    OUTPUT_VARIABLE TORCH_PATH
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})

# show torch version
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.__version__, end='')"
    OUTPUT_VARIABLE TORCH_VERSION
)

find_package(Torch REQUIRED)
message(STATUS "Found torch: version ${TORCH_VERSION}")
message(STATUS "torch library: ${TORCH_LIBRARY}")
message(STATUS "torch include: ${TORCH_INCLUDE_DIRS}")

# CUDA settings
enable_language(CUDA)
find_package(CUDA REQUIRED)
message(STATUS "CUDA Toolkit version: ${CUDA_VERSION}")
message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
message(STATUS "CUDA Include dirs: ${CUDA_INCLUDE_DIRS}")
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

# Create the C++ Python module, linking necessary files
file(GLOB SRC_FILES src/*.cpp src/*.cu)
pybind11_add_module(${PROJECT_NAME} ${SRC_FILES})

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
)
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    message(STATUS "Compiling on Linux")
    # Find libtorch_python.so for linux, but adding this for mac will cause symbol not found
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
    message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_PYTHON_LIBRARY})
    find_package(TBB REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE TBB::tbb)
endif()

# find_package(OpenMP REQUIRED)
# target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)

# Link PyTorch to the module
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} CUDA::cudart)
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# To suppress warnings in PyTorch, you might want to set
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")