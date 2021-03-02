from sys import platform
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


ext_modules = [
    Extension(
        "src.libs.cutils",
        ["src/libs/cutils.pyx"],
        extra_compile_args=['/openmp' if platform == "win32" else '-fopenmp']
    )
]

setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
)