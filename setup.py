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
        extdir = os.path.abspath(os.path.join(self.get_ext_fullpath(ext.name), "..", "neck", "random"))
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
    author="Haitian Jiang",
    author_email="haitian.jiang@nyu.edu",
    description="Neural Networks Equivalence Checking",
    ext_modules=[CMakeExtension("random_backend")],
    cmdclass={"build_ext": CMakeBuild},
    packages=["neck"],
    zip_safe=False,
)
