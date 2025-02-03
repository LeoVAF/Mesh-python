from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("src/compiled/to_compile.pyx", compiler_directives={'language_level': "3"})
)