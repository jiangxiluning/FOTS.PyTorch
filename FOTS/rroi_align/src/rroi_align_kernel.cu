// #ifdef __cplusplus
// extern "C" {
// #endif

#include <stdio.h>
#include <vector>
#include <math.h>
#include <float.h>
#include "rroi_align_kernel.h"


#define DIVUP(m, n) ((m) / (m) + ((m) % (n) > 0))

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)


// /*
// rroi代码
// template <typename float>
__global__ void RROIAlignForward(
    const int nthreads,
    const float* bottom_data,
    const float spatial_scale,
    int height,
    int width,
    int channels,
    const int pooled_height,
    const int pooled_width,
    const float* bottom_rois,
    float* top_data,
    float* con_idx_x,
    float* con_idx_y)
{

    CUDA_KERNEL_LOOP(index, nthreads)
    {
        // +0.5 shift removed
        int imageWidth = width;
        int imageHeight = height;

        // (n, c, ph, pw) is an element in the pooled output
        int n = index;
        int pw = n % pooled_width;
        n /= pooled_width;
        int ph = n % pooled_height;
        n /= pooled_height;
        int c = n % channels;
        n /= channels;

        const float* offset_bottom_rois = bottom_rois + n * 6; // 标注信息

        int roi_batch_ind = offset_bottom_rois[0];
        float cx = offset_bottom_rois[1];
        float cy = offset_bottom_rois[2];
        float h = offset_bottom_rois[3];
        float w = offset_bottom_rois[4];
        float angle = offset_bottom_rois[5]/180.0*3.1415926535;

        //TransformPrepare
        float roi_pooled_width = pooled_height * w / h;         // 不同的高宽比
        float dx = -roi_pooled_width/2.0;
        float dy = -pooled_height/2.0;
        float Sx = w*spatial_scale/roi_pooled_width;
        float Sy = h*spatial_scale/pooled_height;
        float Alpha = cos(angle);
        float Beta = sin(angle);
        float Dx = cx*spatial_scale;
        float Dy = cy*spatial_scale;

        float M[2][3];                              // 旋转矩阵
        M[0][0] = Alpha*Sx;
        M[0][1] = Beta*Sy;
        M[0][2] = Alpha*Sx*dx+Beta*Sy*dy+Dx;
        M[1][0] = -Beta*Sx;
        M[1][1] = Alpha*Sy;
        M[1][2] = -Beta*Sx*dx+Alpha*Sy*dy+Dy;

        float P[8];                                 // 求原roi中4个点的坐标8个值
        P[0] = M[0][0]*pw+M[0][1]*ph+M[0][2];
        P[1] = M[1][0]*pw+M[1][1]*ph+M[1][2];
        P[2] = M[0][0]*pw+M[0][1]*(ph+1)+M[0][2];
        P[3] = M[1][0]*pw+M[1][1]*(ph+1)+M[1][2];
        P[4] = M[0][0]*(pw+1)+M[0][1]*ph+M[0][2];
        P[5] = M[1][0]*(pw+1)+M[1][1]*ph+M[1][2];
        P[6] = M[0][0]*(pw+1)+M[0][1]*(ph+1)+M[0][2];
        P[7] = M[1][0]*(pw+1)+M[1][1]*(ph+1)+M[1][2];

        // 求原rroi的中心，并用双线性插值求出f(x,y)
        float leftMost = (max(round(min(min(P[0],P[2]),min(P[4],P[6]))),0.0));
        float rightMost= (min(round(max(max(P[0],P[2]),max(P[4],P[6]))),imageWidth-1.0));
        float topMost= (max(round(min(min(P[1],P[3]),min(P[5],P[7]))),0.0));
        float bottomMost= (min(round(max(max(P[1],P[3]),max(P[5],P[7]))),imageHeight-1.0));

        const float* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;

        float bin_cx = (leftMost + rightMost) / 2.0; // rroi的中心
        float bin_cy = (topMost + bottomMost) / 2.0;

        const bool in_rroi = pw <= roi_pooled_width;        // 是否在rroi之内
        if (in_rroi){

            int bin_l = (int)floor(bin_cx);
            int bin_r = (int)ceil(bin_cx);
            int bin_t = (int)floor(bin_cy);
            int bin_b = (int)ceil(bin_cy);

            float lt_value = 0.0;
            if (bin_t > 0 && bin_l > 0 && bin_t < height && bin_l < width)
                lt_value = offset_bottom_data[bin_t * width + bin_l];
            float rt_value = 0.0;
            if (bin_t > 0 && bin_r > 0 && bin_t < height && bin_r < width)
                rt_value = offset_bottom_data[bin_t * width + bin_r];
            float lb_value = 0.0;
            if (bin_b > 0 && bin_l > 0 && bin_b < height && bin_l < width)
                lb_value = offset_bottom_data[bin_b * width + bin_l];
            float rb_value = 0.0;
            if (bin_b > 0 && bin_r > 0 && bin_b < height && bin_r < width)
                rb_value = offset_bottom_data[bin_b * width + bin_r];

            float rx = bin_cx - floor(bin_cx);
            float ry = bin_cy - floor(bin_cy);

            float wlt = (1.0 - rx) * (1.0 - ry);
            float wrt = rx * (1.0 - ry);
            float wrb = rx * ry;
            float wlb = (1.0 - rx) * ry;

            float inter_val = 0.0;

            inter_val += lt_value * wlt;
            inter_val += rt_value * wrt;
            inter_val += rb_value * wrb;
            inter_val += lb_value * wlb;

            atomicAdd(top_data + index, static_cast<float>(inter_val));
            atomicAdd(con_idx_x + index, static_cast<float>(bin_cx));
            atomicAdd(con_idx_y + index, static_cast<float>(bin_cy));

            //top_data[index] = static_cast<float>(inter_val);
            //con_idx_x[index] = bin_cx;
            //con_idx_y[index] = bin_cy;
        }
        else{
            // float inter_val = 0.0;
            // float bin_cx = 0.0;                        // -2只是为了反向传播时做标记，其他值也是可以的
            // float bin_cy = 0.0;
            // atomicAdd(top_data + index, static_cast<float>(inter_val));     // 可能多个点加了-2
            // atomicAdd(con_idx_x + index, static_cast<float>(bin_cx));
            // atomicAdd(con_idx_y + index, static_cast<float>(bin_cy));
            continue;
        }

    }
}

int RROIAlignForwardLaucher(
    const float* bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const float* bottom_rois,
    float* top_data, float* con_idx_x, float* con_idx_y, cudaStream_t stream)
{
    const int kThreadsPerBlock = 1024;
    const int output_size = num_rois * pooled_height * pooled_width * channels;
    cudaError_t err;


    RROIAlignForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      output_size, bottom_data, spatial_scale, height, width, channels, pooled_height,
      pooled_width, bottom_rois, top_data, con_idx_x, con_idx_y);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "RRoI forward cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}
// */



// 反向传播
__global__ void RROIAlignBackward(
    const int nthreads,
    const float* top_diff,
    const float* con_idx_x,
    const float* con_idx_y,
    const int num_rois,
    const float spatial_scale,
    const int height,
    const int width,
    const int channels,
    const int pooled_height,
    const int pooled_width,
    float* bottom_diff,
    const float* bottom_rois) {
        CUDA_KERNEL_LOOP(index, nthreads){

        // (n, c, ph, pw) is an element in the pooled output
        // int n = index;
        // //int w = n % width;
        // n /= pooled_width;
        // //int h = n % height;
        // n /= pooled_height;
        // int c = n % channels;
        // n /= channels;

        int n = index;
        int pw = n % pooled_width;
        n /= pooled_width;
        // int ph = n % pooled_height;
        n /= pooled_height;
        int c = n % channels;
        n /= channels;

        const float* offset_bottom_rois = bottom_rois + n * 6;                    // 第i个rroi
        int roi_batch_ind = offset_bottom_rois[0];
        float h = offset_bottom_rois[3];
        float w = offset_bottom_rois[4];
        float* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;      // 反向梯度的索引

        float bin_cx = con_idx_x[index];                // 每个rroi中心点的坐标
        float bin_cy = con_idx_y[index];

        // check whether in rroi
        float roi_pooled_width = pooled_height * w / h;         // 不同的高宽比

        const bool not_in_rroi = (pw > roi_pooled_width);    // 可能多个点多次加了-2, 所以不能采用这种方式判断

        if (not_in_rroi){                               // 如果不再rroi内则跳过当前循环，否则就按原来的操作
            continue;
        }
        else{

            float rx = bin_cx - floor(bin_cx);
            float ry = bin_cy - floor(bin_cy);

            float wlt = (1.0 - rx) * (1.0 - ry);
            float wrt = rx * (1.0 - ry);
            float wrb = rx * ry;
            float wlb = (1.0 - rx) * ry;

            int min_x = (int)floor(bin_cx);
            int max_x = (int)ceil(bin_cx);
            int min_y = (int)floor(bin_cy);
            int max_y = (int)ceil(bin_cy);

            float top_diff_of_bin = top_diff[index];

            float v1 = wlt * top_diff_of_bin;
            float v2 = wrt * top_diff_of_bin;
            float v3 = wrb * top_diff_of_bin;
            float v4 = wlb * top_diff_of_bin;

            // Atomic add

            if (min_y > 0 && min_x  > 0 && min_y < height - 1 && min_x < width - 1)
                atomicAdd(offset_bottom_diff + min_y * width + min_x, static_cast<float>(v1));          // 多个roi会重复操作
            if (min_y > 0 && max_x < width - 1 && min_y < height - 1 && max_x > 0)
                atomicAdd(offset_bottom_diff + min_y * width + max_x, static_cast<float>(v2));
            if (max_y < height - 1 && max_x < width - 1 && max_y > 0 && max_x > 0)
                atomicAdd(offset_bottom_diff + max_y * width + max_x, static_cast<float>(v3));
            if (max_y < height - 1 && min_x > 0 && max_y > 0 && min_x < width - 1)
                atomicAdd(offset_bottom_diff + max_y * width + min_x, static_cast<float>(v4));

        }
    }
}

int RROIAlignBackwardLaucher(
    const float* top_diff,
    const float spatial_scale,
    const int batch_size,
    const int num_rois,
    const int height,
    const int width,
    const int channels,
    const int pooled_height,
    const int pooled_width,
    const float* bottom_rois,
    float* bottom_diff,
    const float* con_idx_x,
    const float* con_idx_y,
    cudaStream_t stream)
{
    const int kThreadsPerBlock = 1024;
    const int output_size = num_rois * pooled_height * pooled_width * channels;//batch_size * height * width * channels;
    cudaError_t err;

    RROIAlignBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      output_size, top_diff, con_idx_x, con_idx_y, num_rois, spatial_scale, height, width, channels, pooled_height,
      pooled_width, bottom_diff, bottom_rois);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "RRoI backward cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


// #ifdef __cplusplus
// }
// #endif
