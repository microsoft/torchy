from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension
from glob import glob

extra_compile_args = ['-std=c++14']
extra_link_args = extra_compile_args
sources = glob('*.cpp')

setup(
  name='torchy',
  version='0.1',
  description='Tracing JIT for PyTorch',
  packages=find_packages(exclude=['build']),
  ext_modules=[
    CppExtension(
      '_TORCHY',
      sources,
      extra_compile_args=extra_compile_args,
      extra_link_args=extra_link_args
    ),
  ],
)
