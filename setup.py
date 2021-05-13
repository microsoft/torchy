from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension
from glob import glob

extra_compile_args = ['-I.', '-std=c++14', '-fvisibility=hidden']
extra_link_args = ['-std=c++14']
sources = glob('**/*.cpp', recursive=True)

setup(
  name='torchy',
  version='0.1',
  description='Tracing JIT for PyTorch',
  packages=['torchy'], #find_packages(exclude=['build']),
  ext_modules=[
    CppExtension(
      '_TORCHY',
      sources,
      extra_compile_args=extra_compile_args,
      extra_link_args=extra_link_args
    ),
  ],
)
