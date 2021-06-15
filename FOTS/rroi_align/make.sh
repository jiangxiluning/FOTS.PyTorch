#!/usr/bin/env bash

# CUDA_PATH=/usr/local/cuda/

export CUDA_PATH=/usr/local/cuda/
#You may also want to ad the following
#export C_INCLUDE_PATH=/opt/cuda/include

export CXXFLAGS="-std=c++11"
export CFLAGS="-std=c99"

# python setup.py build_ext --inplace
# rm -rf build

CUDA_ARCH="-gencode arch=compute_30,code=sm_30 \
           -gencode arch=compute_35,code=sm_35 \
           -gencode arch=compute_50,code=sm_50 \
           -gencode arch=compute_52,code=sm_52 \
           -gencode arch=compute_60,code=sm_60 \
           -gencode arch=compute_61,code=sm_61 "


# compile roi_pooling//编译cuda文件
cd src
echo "Compiling roi pooling kernels by nvcc..."
nvcc -c -o rroi_align.cu.o rroi_align_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
python build.py build_ext