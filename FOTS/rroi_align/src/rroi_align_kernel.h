#pragma once
#include <torch/extension.h>

int RROIAlignForwardLaucher(
    torch::Tensor bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, torch::Tensor bottom_rois,
    torch::Tensor top_data, torch::Tensor con_idx_x, torch::Tensor con_idx_y, cudaStream_t stream);

int RROIAlignBackwardLaucher(
    torch::Tensor top_diff, const float spatial_scale,  const int batch_size, const int num_rois,
    const int height, const int width, const int channels, const int pooled_height,
    const int pooled_width, torch::Tensor bottom_rois, torch::Tensor bottom_diff,
    torch::Tensor con_idx_x, torch::Tensor con_idx_y, cudaStream_t stream);

