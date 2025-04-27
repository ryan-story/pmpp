from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="conv_wrapper",
        sources=["conv_wrapper.pyx", "conv_ops.c"],  # Include both source files
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
        language="c"
    )
]

setup(
    ext_modules=cythonize(extensions)
)