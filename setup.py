from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension
import sys
import subprocess


# if on Mac, retrieve the SDK path from xcrun
def get_sdk_path():
    try:
        return subprocess.check_output(["xcrun", "--show-sdk-path"]).decode().strip()
    except Exception:
        return ""


if sys.platform == "win32":
    extra_compile_args = ["/std:c++17", "/openmp:experimental", "/DDEBUG"]
    extra_link_args = []
    compiler = "msvc"
elif sys.platform == "darwin":
    extra_compile_args = [
        "-std=c++17",
        "-Xpreprocessor",
        "-fopenmp",
        "-DDEBUG",
        "-stdlib=libc++",
    ]
    extra_link_args = ["-Xpreprocessor", "-lomp", "-L/opt/homebrew/opt/libomp/lib"]


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
