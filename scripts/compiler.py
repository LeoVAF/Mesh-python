from setuptools import setup
from Cython.Build import cythonize

import os
import numpy

setup(
    ext_modules=cythonize(os.path.join("src", "mesh", "compiling", "functions.pyx"),
                          compiler_directives={'language_level': "3"}),
    include_dirs=[numpy.get_include()]
)