cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0148 NEW)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
project(cpp_ext LANGUAGES CXX)  # CUDA

function(message_bold)
    string(ASCII 27 ESCAPE_CHAR)
    set(STYLE_BOLD "${ESCAPE_CHAR}[1m")
    set(STYLE_NORMAL "${ESCAPE_CHAR}[0m")
    string(CONCAT message_string ${ARGN})
    message(STATUS "${STYLE_BOLD}${message_string}${STYLE_NORMAL}")
endfunction()

# Prioritize conda and venv
if(DEFINED ENV{CONDA_PREFIX})
    set(Python_EXECUTABLE $ENV{CONDA_PREFIX}/bin/python)
elseif(DEFINED ENV{VIRTUAL_ENV})
    set(Python_EXECUTABLE $ENV{VIRTUAL_ENV}/bin/python)
endif()
message_bold("[python] >> Searching for python...")
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Use pybind11's cmake module to locate the installation path for pybind11
execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message_bold("[pybind11] >> Searching for pybind11...")
find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})

# add torch into cmake path
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
    OUTPUT_VARIABLE TORCH_PATH
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})

# show torch version
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.__version__, end='')"
    OUTPUT_VARIABLE TORCH_VERSION
)

message_bold("[pytorch] >> Searching for pytorch...")
find_package(Torch REQUIRED)

message_bold("[NECK] >> Starting Compiling...")
message(STATUS "[NECK] Found torch: version ${TORCH_VERSION}")
message(STATUS "[NECK] torch library: ${TORCH_LIBRARY}")
message(STATUS "[NECK] torch include: ${TORCH_INCLUDE_DIRS}")

# CUDA settings
# enable_language(CUDA)
# find_package(CUDA REQUIRED)
# message(STATUS "[NECK] CUDA Toolkit version: ${CUDA_VERSION}")
# message(STATUS "[NECK] CUDA Libraries: ${CUDA_LIBRARIES}")
# message(STATUS "[NECK] CUDA Include dirs: ${CUDA_INCLUDE_DIRS}")
# set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

# Create the C++ Python module, linking necessary files
file(GLOB SRC_FILES src/*.cpp src/*.cu)
pybind11_add_module(${PROJECT_NAME} ${SRC_FILES})

# set_target_properties(${PROJECT_NAME} PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_STANDARD 17
# )
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    message(STATUS "[NECK] Compiling on Linux")
    # Find libtorch_python.so for linux, but adding this for mac will cause symbol not found
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
    message(STATUS "[NECK] TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_PYTHON_LIBRARY})
    find_package(TBB REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE TBB::tbb)
endif()

# Link PyTorch to the module
# target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} CUDA::cudart)
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# To suppress warnings in PyTorch, you might want to set
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
