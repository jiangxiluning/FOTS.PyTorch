#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <math.h>
#include "rroi_align_kernel.h"


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int rroi_align_forward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                        torch::Tensor features, torch::Tensor rois, torch::Tensor output, 
                        torch::Tensor idx_x, torch::Tensor idx_y)
{
    CHECK_INPUT(features);
    CHECK_INPUT(rois);
    CHECK_INPUT(output);
    CHECK_INPUT(idx_x);
    CHECK_INPUT(idx_y);

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

    c10::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    RROIAlignForwardLaucher(
        features.data<float>(), spatial_scale, num_rois, data_height,
        data_width, num_channels, pooled_height,
        pooled_width, rois.data<float>(),
        output.data<float>(), idx_x.data<float>(), idx_y.data<float>(), stream);

    return 1;
}



// 反向传播
int rroi_align_backward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                        torch::Tensor top_grad, torch::Tensor rois, torch::Tensor bottom_grad, 
                        torch::Tensor idx_x, torch::Tensor idx_y)
{
    CHECK_INPUT(top_grad);
    CHECK_INPUT(rois);
    CHECK_INPUT(bottom_grad);
    CHECK_INPUT(idx_x);
    CHECK_INPUT(idx_y);

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

    c10::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    RROIAlignBackwardLaucher(
        top_grad.data<float>(), spatial_scale, batch_size, num_rois, data_height,
        data_width, num_channels, pooled_height,
        pooled_width, rois.data<float>(), bottom_grad.data<float>(),
        idx_x.data<float>(), idx_y.data<float>(), stream);

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roi_align_rotated_forward",&rroi_align_forward_cuda,"ROIAlignRotated_forward");
  m.def("roi_align_rotated_backward",&rroi_align_backward_cuda,"ROIAlignRotated_backward");
}