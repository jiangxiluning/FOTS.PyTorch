#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <math.h>
#include "rroi_align_kernel.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

int rroi_align_forward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                        torch::Tensor features, torch::Tensor rois, torch::Tensor output, 
                        torch::Tensor idx_x, torch::Tensor idx_y)
{
    CHECK_CUDA(features);
    CHECK_CUDA(rois);
    CHECK_CUDA(output);
    CHECK_CUDA(idx_x);
    CHECK_CUDA(idx_y);
    // Grab the input tensor
    at::Tensor data_flat = features.data();
    at::Tensor rois_flat = rois.data();

    at::Tensor output_flat = output.data();
    at::Tensor idx_x_flat = idx_x.data();                   // 每个rroi bin的中心索引
    at::Tensor idx_y_flat = idx_y.data();
    // int * argmax_flat = THCudaIntTensor_data(state, argmax);

    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 6)
    {
        return 0;
    }

    // data height
    int data_height = features.size(2);
    // data width
    int data_width = features.size(3);
    // Number of channels
    int num_channels = features.size(1);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    RROIAlignForwardLaucher(
        data_flat, spatial_scale, num_rois, data_height,
        data_width, num_channels, pooled_height,
        pooled_width, rois_flat,
        output_flat, idx_x_flat, idx_y_flat, stream);

    return 1;
}



// 反向传播
int rroi_align_backward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                        torch::Tensor top_grad, torch::Tensor rois, torch::Tensor bottom_grad, 
                        torch::Tensor idx_x, torch::Tensor idx_y)
{
    CHECK_CUDA(top_grad);
    CHECK_CUDA(rois);
    CHECK_CUDA(bottom_grad);
    CHECK_CUDA(idx_x);
    CHECK_CUDA(idx_y);
    // Grab the input tensor
    at::Tensor top_grad_flat = top_grad.data();
    at::Tensor rois_flat = rois.data();

    at::Tensor bottom_grad_flat = bottom_grad.data();
    at::Tensor idx_x_flat = idx_x.data();
    at::Tensor idx_y_flat = idx_y.data();

    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 6)
    {
        return 0;
    }

    // batch size
    int batch_size = bottom_grad.size(0);

    // data height
    int data_height = bottom_grad.size(2);
    // data width
    int data_width = bottom_grad.size(3);
    // Number of channels
    int num_channels = bottom_grad.size(1);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    RROIAlignBackwardLaucher(
        top_grad_flat, spatial_scale, batch_size, num_rois, data_height,
        data_width, num_channels, pooled_height,
        pooled_width, rois_flat, bottom_grad_flat, 
        idx_x_flat, idx_y_flat, stream);

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roi_align_rotated_forward",&rroi_align_forward_cuda,"ROIAlignRotated_forward");
  m.def("roi_align_rotated_backward",&rroi_align_backward_cuda,"ROIAlignRotated_backward");
}