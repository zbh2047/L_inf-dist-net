#include <cassert>
#include <cstdio>

const int BLOCK_SIZE = 16;


#define getX (conv ? blockI[i][threadIdx.x] : blockI[threadIdx.y][i])
#define getW (conv ? blockW[i][threadIdx.y] : blockW[i][threadIdx.x])

template <bool conv>
__global__ void inf_dist_forward_kernel(const float* __restrict__ input, const float* __restrict__ weight,
                                        int B, int CO_div_G, int CI_div_G, int HW, int G,
                                        float* __restrict__ output, int* __restrict__ pos) {
    float output_reg = 0;
    int pos_reg = 0;

    int b_hw = blockIdx.y * BLOCK_SIZE + (conv ? threadIdx.x : threadIdx.y);
    int b = b_hw / HW;
    int hw = b_hw % HW;
    int write_co = blockIdx.x * BLOCK_SIZE + (conv ? threadIdx.y : threadIdx.x);
    int read_w_co = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    __shared__ float blockI[BLOCK_SIZE][BLOCK_SIZE]; // CI * B if conv else B * CI
    __shared__ float blockW[BLOCK_SIZE][BLOCK_SIZE + 2]; // CI * CO

    int k;
    for (k = 0; k < (CI_div_G & ~(BLOCK_SIZE - 1)); k += BLOCK_SIZE) {
        if (b < B) {
            if (conv) blockI[threadIdx.y][threadIdx.x] = input[((b * G + blockIdx.z) * CI_div_G + k + threadIdx.y) * HW + hw];
            else blockI[threadIdx.y][threadIdx.x] = input[((b * G + blockIdx.z) * CI_div_G + k + threadIdx.x) * HW + hw];
        }
        if (read_w_co < CO_div_G)
            blockW[threadIdx.x][threadIdx.y] = weight[(blockIdx.z * CO_div_G + read_w_co) * CI_div_G + k + threadIdx.x];
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float t = getX - getW;
            float abs_t = abs(t);
            if (abs_t > output_reg) {
                output_reg = abs_t;
                pos_reg = k + i + (t >= 0 ? 0 : 1 << 31);
            }
        }
        __syncthreads();
    }
    if (CI_div_G & (BLOCK_SIZE - 1)) {
        if (b < B) {
            if (conv && k + threadIdx.y < CI_div_G) blockI[threadIdx.y][threadIdx.x] = input[((b * G + blockIdx.z) * CI_div_G + k + threadIdx.y) * HW + hw];
            if (!conv && k + threadIdx.x < CI_div_G) blockI[threadIdx.y][threadIdx.x] = input[((b * G + blockIdx.z) * CI_div_G + k + threadIdx.x) * HW + hw];
        }
        if (k + threadIdx.x < CI_div_G && read_w_co < CO_div_G)
            blockW[threadIdx.x][threadIdx.y] = weight[(blockIdx.z * CO_div_G + read_w_co) * CI_div_G + k + threadIdx.x];
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < (CI_div_G & (BLOCK_SIZE - 1)); i++) {
            float t = getX - getW;
            float abs_t = abs(t);
            if (abs_t > output_reg) {
                output_reg = abs_t;
                pos_reg = k + i + (t >= 0 ? 0 : 1 << 31);
            }
        }
    }
    if (b < B && write_co < CO_div_G) {
        output[((b * G + blockIdx.z) * CO_div_G + write_co) * HW + hw] = output_reg;
        pos[((b * G + blockIdx.z) * CO_div_G + write_co) * HW + hw] = pos_reg;
    }
}

#define getXL (conv ? blockIL[i][threadIdx.x] : blockIL[threadIdx.y][i])
#define getXU (conv ? blockIU[i][threadIdx.x] : blockIU[threadIdx.y][i])

template <bool conv>
__global__ void inf_dist_bound_forward_kernel(const float* __restrict__ input_lower, const float* __restrict__ input_upper,
                                              const float* __restrict__ weight,
                                              int B, int CO_div_G, int CI_div_G, int HW, int G,
                                              float* __restrict__ output_lower, float* __restrict__ output_upper) {
    float output_regL = 0, output_regU= 0;

    int b_hw = blockIdx.y * BLOCK_SIZE + (conv ? threadIdx.x : threadIdx.y);
    int b = b_hw / HW;
    int hw = b_hw % HW;
    int write_co = blockIdx.x * BLOCK_SIZE + (conv ? threadIdx.y : threadIdx.x);
    int read_w_co = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    __shared__ float blockIL[BLOCK_SIZE][BLOCK_SIZE], blockIU[BLOCK_SIZE][BLOCK_SIZE]; // CI * B if conv else B * CI
    __shared__ float blockW[BLOCK_SIZE][BLOCK_SIZE + 2]; // CI * CO

    int k;
    for (k = 0; k < (CI_div_G & ~(BLOCK_SIZE - 1)); k += BLOCK_SIZE) {
        if (b < B) {
            if (conv) {
                blockIL[threadIdx.y][threadIdx.x] = input_lower[((b * G + blockIdx.z) * CI_div_G + k + threadIdx.y) * HW + hw];
                blockIU[threadIdx.y][threadIdx.x] = input_upper[((b * G + blockIdx.z) * CI_div_G + k + threadIdx.y) * HW + hw];
            }
            else {
                blockIL[threadIdx.y][threadIdx.x] = input_lower[((b * G + blockIdx.z) * CI_div_G + k + threadIdx.x) * HW + hw];
                blockIU[threadIdx.y][threadIdx.x] = input_upper[((b * G + blockIdx.z) * CI_div_G + k + threadIdx.x) * HW + hw];
            }
        }
        if (read_w_co < CO_div_G)
            blockW[threadIdx.x][threadIdx.y] = weight[(blockIdx.z * CO_div_G + read_w_co) * CI_div_G + k + threadIdx.x];
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float tL = getXL - getW, tU = getXU - getW;
            float abs_tL = abs(tL), abs_tU = abs(tU);
            output_regU = max(output_regU, max(abs_tL, abs_tU));
            if (!(tL < 0 && tU > 0))
                output_regL = max(output_regL, min(abs_tL, abs_tU));
        }
        __syncthreads();
    }
    if (CI_div_G & (BLOCK_SIZE - 1)) {
        if (b < B) {
            if (conv && k + threadIdx.y < CI_div_G) {
                blockIL[threadIdx.y][threadIdx.x] = input_lower[((b * G + blockIdx.z) * CI_div_G + k + threadIdx.y) * HW + hw];
                blockIU[threadIdx.y][threadIdx.x] = input_upper[((b * G + blockIdx.z) * CI_div_G + k + threadIdx.y) * HW + hw];
            }
            if (!conv && k + threadIdx.x < CI_div_G) {
                blockIL[threadIdx.y][threadIdx.x] = input_lower[((b * G + blockIdx.z) * CI_div_G + k + threadIdx.x) * HW + hw];
                blockIU[threadIdx.y][threadIdx.x] = input_upper[((b * G + blockIdx.z) * CI_div_G + k + threadIdx.x) * HW + hw];
            }
        }
        if (k + threadIdx.x < CI_div_G && read_w_co < CO_div_G)
            blockW[threadIdx.x][threadIdx.y] = weight[(blockIdx.z * CO_div_G + read_w_co) * CI_div_G + k + threadIdx.x];
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < (CI_div_G & (BLOCK_SIZE - 1)); i++) {
            float tL = getXL - getW, tU = getXU - getW;
            float abs_tL = abs(tL), abs_tU = abs(tU);
            output_regU = max(output_regU, max(abs_tL, abs_tU));
            if (!(tL < 0 && tU > 0))
                output_regL = max(output_regL, min(abs_tL, abs_tU));
        }
    }
    if (b < B && write_co < CO_div_G) {
        output_lower[((b * G + blockIdx.z) * CO_div_G + write_co) * HW + hw] = output_regL;
        output_upper[((b * G + blockIdx.z) * CO_div_G + write_co) * HW + hw] = output_regU;
    }
}

template <bool conv>
__global__ void inf_dist_backward_input_kernel(const float* __restrict__ grad_output, const int* __restrict__ pos,
                                                int B, int CO_div_G, int CI_div_G, int HW, int G, float* __restrict__ grad_input) {
    int b_hw = blockIdx.y * BLOCK_SIZE + (conv ? threadIdx.x : threadIdx.y);
    int b = b_hw / HW;
    int hw = b_hw % HW;
    int co = blockIdx.x * BLOCK_SIZE + (conv ? threadIdx.y : threadIdx.x);
    if (b < B && co < CO_div_G) {
        int pos_reg = pos[((b * G + blockIdx.z) * CO_div_G + co) * HW + hw];
        float grad = grad_output[((b * G + blockIdx.z) * CO_div_G + co) * HW + hw];
        int index = pos_reg & (~(1 << 31));
        atomicAdd(&grad_input[((b * G + blockIdx.z) * CI_div_G + index) * HW + hw], pos_reg >= 0 ? grad : -grad);
    }
}

template <bool conv>
__global__ void inf_dist_backward_weight_kernel(const float* __restrict__ grad_output, const int* __restrict__ pos,
                                                int B, int CO_div_G, int CI_div_G, int HW, int G, float* __restrict__ grad_weight) {
    int b_hw = blockIdx.y * BLOCK_SIZE + (conv ? threadIdx.x : threadIdx.y);
    int b = b_hw / HW;
    int hw = b_hw % HW;
    int co = blockIdx.x * BLOCK_SIZE + (conv ? threadIdx.y : threadIdx.x);
    if (b < B && co < CO_div_G) {
        int pos_reg = pos[((b * G + blockIdx.z) * CO_div_G + co) * HW + hw];
        float grad = grad_output[((b * G + blockIdx.z) * CO_div_G + co) * HW + hw];
        int index = pos_reg & (~(1 << 31));
        atomicAdd(&grad_weight[(blockIdx.z * CO_div_G + co) * CI_div_G + index], pos_reg < 0 ? grad : -grad);
    }
}

void inf_dist_forward_cuda(const float* input, const float* weight,
                       int B, int CO, int CI, int G, int HW, float* output, int* pos) {

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid2((CO / G + BLOCK_SIZE - 1) / BLOCK_SIZE, (B * HW + BLOCK_SIZE - 1) / BLOCK_SIZE, G);
    if (HW == 1) inf_dist_forward_kernel<false><<<dimGrid2, dimBlock>>>(input, weight, B, CO / G, CI / G, HW, G, output, pos);
    else inf_dist_forward_kernel<true><<<dimGrid2, dimBlock>>>(input, weight, B, CO / G, CI / G, HW, G, output, pos);
}
void inf_dist_bound_forward_cuda(const float* input_lower, const float* input_upper, const float* weight,
                                 int B, int CO, int CI, int G, int HW, float* output_lower, float* output_upper) {

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid2((CO / G + BLOCK_SIZE - 1) / BLOCK_SIZE, (B * HW + BLOCK_SIZE - 1) / BLOCK_SIZE, G);
    if (HW == 1) inf_dist_bound_forward_kernel<false><<<dimGrid2, dimBlock>>>(input_lower, input_upper, weight, B, CO / G, CI / G, HW, G, output_lower, output_upper);
    else inf_dist_bound_forward_kernel<true><<<dimGrid2, dimBlock>>>(input_lower, input_upper, weight, B, CO / G, CI / G, HW, G, output_lower, output_upper);
}
void inf_dist_backward_input_cuda(const float* grad_output, const int* pos,
                              int B, int CO, int CI, int G, int HW, float* grad_input) {

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid2((CO / G + BLOCK_SIZE - 1) / BLOCK_SIZE, (B * HW + BLOCK_SIZE - 1) / BLOCK_SIZE, G);
    if (HW == 1) inf_dist_backward_input_kernel<false><<<dimGrid2, dimBlock>>>(grad_output, pos, B, CO / G, CI / G, HW, G, grad_input);
    else inf_dist_backward_input_kernel<true><<<dimGrid2, dimBlock>>>(grad_output, pos, B, CO / G, CI / G, HW, G, grad_input);
}
void inf_dist_backward_weight_cuda(const float* grad_output, const int* pos,
                               int B, int CO, int CI, int G, int HW, float* grad_weight) {

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid2((CO / G + BLOCK_SIZE - 1) / BLOCK_SIZE, (B * HW + BLOCK_SIZE - 1) / BLOCK_SIZE, G);
    if (HW == 1) inf_dist_backward_weight_kernel<false><<<dimGrid2, dimBlock>>>(grad_output, pos, B, CO / G, CI / G, HW, G, grad_weight);
    else inf_dist_backward_weight_kernel<true><<<dimGrid2, dimBlock>>>(grad_output, pos, B, CO / G, CI / G, HW, G, grad_weight);
}
