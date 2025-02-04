import numpy

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("C:/Users/leove/Área de Trabalho/MESH/src/compiled/to_compile.pyx", compiler_directives={'language_level': "3"}),
    include_dirs=[numpy.get_include()]
)