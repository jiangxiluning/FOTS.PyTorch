
int rroi_align_forward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                        THCudaTensor * features, THCudaTensor * rois, THCudaTensor * output, 
                        THCudaTensor * idx_x, THCudaTensor * idx_y);

int rroi_align_backward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                        THCudaTensor * top_grad, THCudaTensor * rois, THCudaTensor * bottom_grad, 
                        THCudaTensor * idx_x, THCudaTensor * idx_y);