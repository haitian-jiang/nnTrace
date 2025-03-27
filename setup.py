import os
import sys
import shutil
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
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            # f"-DPYTHON_EXECUTABLE={sys.executable}",
            # "-DCMAKE_BUILD_TYPE=Release",
            # "-DCMAKE_CXX_FLAGS_RELEASE=-O3",
        ]
        build_args = [
            "--parallel", str(os.cpu_count() or 1), 
        ]

        build_temp = os.path.abspath(os.path.join(".", 'build'))

        # Clean build directory if it exists
        if os.path.exists(build_temp):
            print(f"Removing old build directory: {build_temp}")
            shutil.rmtree(build_temp)
        os.makedirs(build_temp)

        # Configure and build the extension with CMake
        subprocess.check_call(
            ["cmake", os.path.abspath(".")] + cmake_args, cwd=build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=build_temp
        )

setup(
    name="neck",
    version="0.2.0",
    author="Haitian Jiang",
    author_email="haitian.jiang@nyu.edu",
    description="Neural Networks Equivalence Checking",
    ext_modules=[CMakeExtension("cpp_ext")],
    cmdclass={"build_ext": CMakeBuild},
    packages=["neck"],
    zip_safe=False,
)
