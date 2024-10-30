# from setuptools import setup, Extension
# from pybind11.setup_helpers import Pybind11Extension, build_ext

# # Define the extension module
# ext_modules = [
#     Pybind11Extension(
#         "neck._neck",  # Module name
#         ["src/bindings.cpp", "src/example.cpp"],  # Source files
#     ),
# ]

# # Standard setup() function
# setup(
#     name="neck",
#     version="0.1.0",
#     author="Your Name",
#     author_email="your.email@example.com",
#     description="A Python library with high-performance C++ functions",
#     ext_modules=ext_modules,
#     cmdclass={"build_ext": build_ext},
#     packages=["neck"],
#     zip_safe=False,
# )

# from setuptools import setup, Extension
# from pybind11.setup_helpers import Pybind11Extension, build_ext
# import torch
# import os

# pytorch_path = torch.__path__[0]

# include_dirs = [os.path.join(pytorch_path, "include")]
# library_dirs = [os.path.join(pytorch_path, "lib")]
# # include_dirs = [torch.utils.cpp_extension.include_paths()[0]]
# # library_dirs = [torch.utils.cpp_extension.library_paths()[0]]
# extra_compile_args = ['-std=c++14']
# extra_link_args = ['-Wl,-rpath,' + library_dirs[0]]  # 在运行时找到 PyTorch 的动态库

# # 定义扩展模块
# ext_modules = [
#     Pybind11Extension(
#         "neck._neck",  # Module name, assuming neck is your package name
#         ["src/bindings.cpp", "src/mylib.cpp"],  # Source files
#         include_dirs=include_dirs,
#         library_dirs=library_dirs,
#         libraries=['torch', 'torch_python'],  # Link with PyTorch libraries
#         extra_compile_args=extra_compile_args,
#         extra_link_args=extra_link_args
#     ),
# ]

# # setup() 函数
# setup(
#     name="neck",
#     version="0.1.0",
#     author="Your Name",
#     author_email="your.email@example.com",
#     description="A Python library with high-performance C++ functions using PyTorch",
#     ext_modules=ext_modules,
#     cmdclass={"build_ext": build_ext},
#     packages=["neck"],
#     zip_safe=False,
# )

#-----------------------------------------------------#
import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.join(self.get_ext_fullpath(ext.name), "..", "neck"))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            # f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]
        build_args = []

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Configure and build the extension with CMake
        subprocess.check_call(
            ["cmake", os.path.abspath(".")] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )

setup(
    name="neck",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python library with high-performance C++ functions using PyTorch",
    ext_modules=[CMakeExtension("_neck")],
    cmdclass={"build_ext": CMakeBuild},
    packages=["neck"],
    zip_safe=False,
)
