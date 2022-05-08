#ifndef _ROI_POOLING_KERNEL
#define _ROI_POOLING_KERNEL
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

int RROIAlignForwardLaucher(
    const float* bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const float* bottom_rois,
    float* top_data, float* con_idx_x, float* con_idx_y, cudaStream_t stream);

int RROIAlignBackwardLaucher(
    const float* top_diff, const float spatial_scale,  const int batch_size, const int num_rois,
    const int height, const int width, const int channels, const int pooled_height,
    const int pooled_width, const float* bottom_rois, float* bottom_diff,
    const float* con_idx_x, const float* con_idx_y, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

