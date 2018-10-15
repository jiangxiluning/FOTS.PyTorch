#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda

echo "Compiling crop_and_resize kernels by nvcc..."
cd model/modules/roi_align/src/cuda
$CUDA_PATH/bin/nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_60 \
 -gencode=arch=compute_30,code=sm_30 \
 -gencode=arch=compute_50,code=sm_50 \
 -gencode=arch=compute_52,code=sm_52 \
 -gencode=arch=compute_60,code=sm_60 \
 -gencode=arch=compute_61,code=sm_61 \
 -gencode=arch=compute_62,code=sm_62 \

cd ../../../roi_align
python build.py

#cd ..
#python setup.py install
#find  $CONDA_PREFIX -name roi_align | awk '{mkdir $0 "/_ext" }' |bash
#find  $CONDA_PREFIX -name roi_align | awk '{print "cp -r roi_align/_ext/* " $0 "/_ext/" }' |bash