# compile with:
# python setup_cython.py build_ext --inplace


from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

extensions = [
    Extension('im2col_cython', ['conv_backward.pyx'],
              include_dirs=[numpy.get_include()]
              ),
]

setup(
    ext_modules=cythonize(extensions),
)
