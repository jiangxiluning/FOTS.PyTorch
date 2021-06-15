#pragma once
#include <torch/extension.h>

int RROIAlignForwardLaucher(
    const at::Tensor& bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const at::Tensor& bottom_rois,
    at::Tensor& top_data, at::Tensor& con_idx_x, at::Tensor& con_idx_y, cudaStream_t stream);

int RROIAlignBackwardLaucher(
    const at::Tensor& top_diff, const float spatial_scale,  const int batch_size, const int num_rois,
    const int height, const int width, const int channels, const int pooled_height,
    const int pooled_width, const at::Tensor& bottom_rois, at::Tensor& bottom_diff,
    const at::Tensor& con_idx_x, const at::Tensor& con_idx_y, cudaStream_t stream);

