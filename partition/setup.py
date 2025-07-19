from setuptools import setup, Extension
import pybind11

boost_include_dir = '/opt/homebrew/opt/boost'  # Adjust if your Boost is elsewhere

ext_modules = [
    Extension(
        'partition',
        sources=['partition.cpp', 'bindings.cpp'],  # add your source files here
        include_dirs=[pybind11.get_include(), boost_include_dir],
        language='c++',
        extra_compile_args=['-std=c++17'],  # or your preferred standard
    ),
]

setup(
    name='partition',
    version='0.0.1',
    author='Your Name',
    ext_modules=ext_modules,
)