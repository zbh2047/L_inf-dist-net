from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='norm_dist_cpp',
      ext_modules=[CUDAExtension('norm_dist_cpp', ['model/norm_dist.cpp', 'model/norm_dist_cuda.cu', 'model/inf_dist_cuda.cu'])],
      cmdclass={'build_ext': BuildExtension})
