import os
import glob
import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CppExtension, CUDA_HOME
from setuptools import setup, find_packages


# sources = ['src/roi_pooling.c']
# headers = ['src/roi_pooling.h']
# extra_objects = []
# defines = []
# with_cuda = False

# this_file = os.path.dirname(os.path.realpath(__file__))
# print(this_file)

# if torch.cuda.is_available():
#     print('Including CUDA code.')
#     sources += ['src/rroi_align_cuda.c']
#     headers += ['src/rroi_align_cuda.h']
#     defines += [('WITH_CUDA', None)]
#     with_cuda = True
#     extra_objects = ['src/rroi_align.cu.o']
#     extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

# # 这里就是编译
# ffi = create_extension(
#     '_ext.rroi_align',
#     headers=headers,
#     sources=sources,
#     define_macros=defines,
#     relative_to=__file__,
#     with_cuda=with_cuda,
#     extra_objects=extra_objects
# )

# python setup.py build_ext --inplace
def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir,"src")

    source_cpu = glob.glob(os.path.join(extensions_dir, "*.c"))
    source_cpu += glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "*.cu"))

    sources = source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        print(sources)
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "rotated_roi",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules

if __name__ == '__main__':
    setup(
    name="rotate_roi",
    # version="0.1",
    # author="fmassa",
    # url="https://github.com/facebookresearch/maskrcnn-benchmark",
    packages=find_packages(exclude=("configs", "tests",)),
    # install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension})
