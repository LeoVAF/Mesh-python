import numpy

from setuptools import setup
from Cython.Build import cythonize

setup(
    # ext_modules=cythonize("C:/Users/leove/Área de Trabalho/MESH/src/compiled/functions.pyx", compiler_directives={'language_level': "3"}),
    ext_modules=cythonize("/home/leonardo_filho/Área de Trabalho/MESH/src/compiled/functions.pyx", compiler_directives={'language_level': "3"}),
    include_dirs=[numpy.get_include()]
)