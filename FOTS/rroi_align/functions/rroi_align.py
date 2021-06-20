import torch
from torch.autograd import Function
import rotated_roi as rroi_align
import pdb


class RRoiAlignFunction(Function):
    # def __init__(ctx, pooled_height, pooled_width, spatial_scale):
    #     ctx.pooled_width = pooled_width
    #     ctx.pooled_height = pooled_height
    #     ctx.spatial_scale = spatial_scale
    #     ctx.feature_size = None

    @staticmethod
    def forward(ctx, features, rois, pooled_height, pooled_width, spatial_scale): 

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, pooled_height, pooled_width).zero_().float()
        # ctx.argmax = features.new(num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_().int()
        idx_x = features.new(num_rois, num_channels, pooled_height, pooled_width).zero_().float()       # 都是float类型的变量
        idx_y = features.new(num_rois, num_channels, pooled_height, pooled_width).zero_().float()
        rois = rois
        if not features.is_cuda:
            _features = features.permute(0, 2, 3, 1)
            roi_pooling.roi_pooling_forward(pooled_height, pooled_width, spatial_scale,
                                            _features, rois, output)
        else:
            rroi_align.roi_align_rotated_forward(pooled_height, pooled_width, spatial_scale,
                                                 features, rois, output, idx_x, idx_y)
        
        ctx.pooled_width = pooled_width
        ctx.pooled_height = pooled_height
        ctx.spatial_scale = spatial_scale
        ctx.save_for_backward(features, rois, idx_x, idx_y)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        features, rois, idx_x, idx_y = ctx.saved_tensors
        assert(features is not None and grad_output.is_cuda)
        batch_size, num_channels, data_height, data_width = features.size()
        grad_input = grad_output.new(batch_size, num_channels, data_height, data_width).zero_().float()

        rroi_align.roi_align_rotated_backward(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
                                              grad_output, rois, grad_input, idx_x, idx_y)
        return grad_input, None, None, None, None
