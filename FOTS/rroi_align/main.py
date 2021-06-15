import cupy as cp
import numpy as np
import math

# bottom_data = cp.random.randn(1,3,40,40, dtype=np.float32)			# 特征feature 
bottom_data = np.random.randn(1,1,40,40)
bottom_data = cp.array(bottom_data, dtype=np.float32)
batch, channels, height, width = bottom_data.shape
spatial_scale = 1.0													# 原始特征和feature的比例
rois = cp.array([[0, 2, 2, 10, 10],
                 [0, 2, 4, 20, 10]], dtype=np.float32)				# rois
pooled_weight = 7													# 池化之后的宽度
pooled_height = 7													# 池化之后的高度

## 定义核函数
roi_pooling_2d_fwd = cp.ElementwiseKernel(
            '''
            raw T bottom_data, T spatial_scale, int32 channels,
            int32 height, int32 width, int32 pooled_height, int32 pooled_width,
            raw T bottom_rois
            ''',
            'T top_data, int32 argmax_data',
            '''
            // pos in output filter
            int pw = i % pooled_width;
            int ph = (i / pooled_width) % pooled_height;
            int c = (i / pooled_width / pooled_height) % channels;
            int num = i / pooled_width / pooled_height / channels;
            int roi_batch_ind = bottom_rois[num * 5 + 0];
            int roi_start_w = round(bottom_rois[num * 5 + 1] * spatial_scale);          // 读取rois的信息
            int roi_start_h = round(bottom_rois[num * 5 + 2] * spatial_scale);
            int roi_end_w = round(bottom_rois[num * 5 + 3] * spatial_scale);
            int roi_end_h = round(bottom_rois[num * 5 + 4] * spatial_scale);

            // Force malformed ROIs to be 1x1
            // 计算每块开始和结束的索引
            int roi_width = max(roi_end_w - roi_start_w + 1, 1);
            int roi_height = max(roi_end_h - roi_start_h + 1, 1);

            // 计算pooled_weight
            int rois_pooled_width = (int)(ceil((float)(pooled_height * roi_width) / (float)(roi_height) ));          // 等比例池化，减小
            float bin_size_h = static_cast<float>(roi_height)  / static_cast<float>(pooled_height);                         // static_cast强制类型转换
            float bin_size_w = static_cast<float>(roi_width)   / static_cast<float>(rois_pooled_width);

            int hstart = static_cast<int>(floor(static_cast<float>(ph) * bin_size_h));
            int wstart = static_cast<int>(floor(static_cast<float>(pw) * bin_size_w));
            int hend = static_cast<int>(ceil(static_cast<float>(ph + 1) * bin_size_h));
            int wend = static_cast<int>(ceil(static_cast<float>(pw + 1) * bin_size_w));

            // Add roi offsets and clip to input boundaries
            // 求每块的最大值
            hstart = min(max(hstart + roi_start_h, 0), height);
            hend = min(max(hend + roi_start_h, 0), height);
            wstart = min(max(wstart + roi_start_w, 0), width);
            wend = min(max(wend + roi_start_w, 0), width);
            bool is_empty = (hend <= hstart) || (wend <= wstart);
            // Define an empty pooling region to be zero
            float maxval = is_empty ? 0 : -1E+37;
            // If nothing is pooled, argmax=-1 causes nothing to be backprop'd

            int maxidx = -1;
            int data_offset = (roi_batch_ind * channels + c) * height * width;
            for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                    int bottom_index = h * width + w;
                    if (bottom_data[data_offset + bottom_index] > maxval) {
                        maxval = bottom_data[data_offset + bottom_index];
                        maxidx = bottom_index;
                    }
                }
            }
            top_data = maxval;
            argmax_data = maxidx;
            ''', 'roi_pooling_2d_fwd'
        )
pooled_height = 2
maxratio = (rois[:, 3] - rois[:, 1]) / (rois[:, 4] - rois[:, 2])
maxratio = maxratio.max()
pooled_width = math.ceil(pooled_height * maxratio)

top_data = cp.zeros((2, 3, pooled_height, pooled_width), dtype=np.float32)		# 输出的feature map
argmax_data = cp.zeros(top_data.shape, np.int32)								# 最大值对应的索引

roi_pooling_2d_fwd(bottom_data, spatial_scale, channels, height, width,
          pooled_height, pooled_width, rois, top_data, argmax_data)

print(top_data.shape)