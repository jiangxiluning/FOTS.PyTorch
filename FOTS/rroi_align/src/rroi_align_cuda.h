#include <torch/extension.h>

int rroi_align_forward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                        torch::Tensor features, torch::Tensor rois, torch::Tensor output,
                        torch::Tensor idx_x, torch::Tensor idx_y);

int rroi_align_backward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                        torch::Tensor top_grad, torch::Tensor rois, torch::Tensor bottom_grad,
                        torch::Tensor idx_x, torch::Tensor idx_y);