import os
import shutil
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions)
            )

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
        print("-" * 10, f"Configuring CMake for {ext.name}", "-" * 10)
        subprocess.check_call(
            ["cmake", os.path.abspath(".")] + cmake_args, cwd=build_temp
        )

        print("-" * 10, f"Building {ext.name} with CMake", "-" * 10)
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=build_temp
        )

setup(
    ext_modules=[CMakeExtension("cpp_ext")],
    cmdclass={"build_ext": CMakeBuild},
    packages=["neck"],
    zip_safe=False,
)
