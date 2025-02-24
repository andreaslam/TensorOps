from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension
import sys
import os

# Determine the platform-specific compile arguments
if sys.platform == "win32":
    extra_compile_args = ["/std:c++17", "/openmp:experimental", "/DDEBUG"]
    extra_link_args = []
    compiler = "msvc"
else:
    extra_compile_args = ["-std=c++17", "-fopenmp", "-DDEBUG", "-lomp"]
    extra_link_args = ["-fopenmp", "-ltbb"]
    compiler = "gcc"

ext_modules = [
    Pybind11Extension(
        "hip_cpu_bindings",
        [
            "tensorops/bindings.cpp",
            "tensorops/tensor_backend.cpp",
        ],
        include_dirs=[
            ".",
            "./repos/HIP-CPU/include",
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
]

setup(
    name="hip_cpu_bindings",
    version="0.1",
    ext_modules=ext_modules,
    packages=find_packages(include=["tensorops", "tensorops.*"]),
    zip_safe=False,
)
