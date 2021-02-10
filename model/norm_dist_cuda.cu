#include <cassert>
#include <cstdio>

#define THREAD_PER_BLOCK 256
#define BLOCKS_PER_SM 4
//each thread can only have 64 registers

__device__ __forceinline__ float pow_fun(float x, float p) {
    return __powf(x, p);
}

template <int CI_div_G, int CO_div_G>
__global__ void __launch_bounds__(THREAD_PER_BLOCK, BLOCKS_PER_SM)
norm_dist_forward_kernel(const float* __restrict__ input, const float* __restrict__ weight,
                                    int B, int CO, int CI, int HW, float* __restrict__ output, float p) {
    int b_hw = blockIdx.y * THREAD_PER_BLOCK + threadIdx.x;
    int b = b_hw / HW;
    int hw = b_hw % HW;
    int g = blockIdx.x;

    __shared__ float blockW[CO_div_G * CI_div_G];
    for (int pos = threadIdx.x; pos < CI_div_G * CO_div_G; pos += THREAD_PER_BLOCK)
        blockW[pos] = weight[g * CO_div_G * CI_div_G + pos];
    __syncthreads();

    if (b >= B) return;

    const int LOOP_CO = CO_div_G > 8 ? 8 : CO_div_G;
    float r_max_x_sub_w[LOOP_CO], ans[LOOP_CO];
    #pragma unroll(1)
    for (int step = 0; step < CO_div_G; step += LOOP_CO) {
        #pragma unroll
        for (int i = 0; i < LOOP_CO; i++)
            r_max_x_sub_w[i] = 1e-10f;
        #pragma unroll(1)
        for (int j = 0; j < CI_div_G; j++) {
            float x = input[(b * CI + g * CI_div_G + j) * HW + hw];
            #pragma unroll
            for (int i = 0; i < LOOP_CO; i++) {
                float w = blockW[(step + i) * CI_div_G + j];
                r_max_x_sub_w[i] = max(r_max_x_sub_w[i], abs(x - w));
            }
        }
        #pragma unroll
        for (int i = 0; i < LOOP_CO; i++) {
            r_max_x_sub_w[i] = 1.0f / r_max_x_sub_w[i];
            ans[i] = 1e-10f;
        }
        #pragma unroll(1)
        for (int j = CI_div_G - 1; j >= 0; j--) {
            float x = input[(b * CI + g * CI_div_G + j) * HW + hw];
            #pragma unroll
            for (int i = 0; i < LOOP_CO; i++) {
                float w = blockW[(step + i) * CI_div_G + j];
                ans[i] += pow_fun(abs(x - w) * r_max_x_sub_w[i], p);
            }
        }
        #pragma unroll
        for (int i = 0; i < LOOP_CO; i++) {
            float res = __powf(ans[i], 1.0f / p) / r_max_x_sub_w[i];
            output[(b * CO + g * CO_div_G + step + i) * HW + hw] = res;
        }
    }
}

template <int CI_div_G, int CO_div_G>
__global__ void __launch_bounds__(THREAD_PER_BLOCK, BLOCKS_PER_SM)
norm_dist_backward_input_kernel(const float* __restrict__ grad_output, const float* __restrict__ input,
                                           const float* __restrict__ weight, const float* __restrict__ output,
                                           int B, int CO, int CI, int HW, float* __restrict__ grad_input, float p) {
    int b_hw = blockIdx.y * THREAD_PER_BLOCK + threadIdx.x;
    int b = b_hw / HW;
    int hw = b_hw % HW;
    int g = blockIdx.x;

    __shared__ float blockW[CO_div_G * CI_div_G];
    for (int pos = threadIdx.x; pos < CI_div_G * CO_div_G; pos += THREAD_PER_BLOCK)
        blockW[pos] = weight[g * CO_div_G * CI_div_G + pos];
    __syncthreads();

    if (b >= B) return;

    const int LOOP_CO = CO_div_G > 8 ? 8 : CO_div_G;
    float grad_out[LOOP_CO], r_out[LOOP_CO];
    #pragma unroll(1)
    for (int step = 0; step < CO_div_G; step += LOOP_CO) {
        #pragma unroll
        for (int i = 0; i < LOOP_CO; i++) {
            grad_out[i] = grad_output[(b * CO + g * CO_div_G + step + i) * HW + hw];
            r_out[i] = 1.0 / output[(b * CO + g * CO_div_G + step + i) * HW + hw];
        }
        #pragma unroll(1)
        for (int j = 0; j < CI_div_G; j++) {
            float x = input[(b * CI + g * CI_div_G + j) * HW + hw];
            float ans = 0.0f;
            #pragma unroll
            for (int i = 0; i < LOOP_CO; i++) {
                float w = blockW[(step + i) * CI_div_G + j];
                float t = x - w;
                ans += grad_out[i] * pow_fun(abs(t) * r_out[i], p - 1) * (t > 0 ? 1 : -1);
            }
            if (step == 0) grad_input[(b * CI + g * CI_div_G + j) * HW + hw] = ans;
            else grad_input[(b * CI + g * CI_div_G + j) * HW + hw] += ans;
        }
    }
}

template <int CI_div_G, int CO_div_G>
__global__ void __launch_bounds__(THREAD_PER_BLOCK, BLOCKS_PER_SM)
norm_dist_backward_weight_kernel(const float* __restrict__ grad_output, const float* __restrict__ input,
                                            const float* __restrict__ weight, const float* __restrict__ output,
                                            int B, int CO, int CI, int HW, float* __restrict__ grad_weight, float p) {
    const int LOOP_HW = 8;
    const int LOOP_CO = CO_div_G > THREAD_PER_BLOCK / 64 ? THREAD_PER_BLOCK / 64 : CO_div_G;
    int b_hw_start = blockIdx.y * 64 * LOOP_HW;
    int g = blockIdx.x;

    float grad_out[LOOP_HW], r_out[LOOP_HW];
    __shared__ float blockW[LOOP_CO * CI_div_G];
    __shared__ float ans[LOOP_CO * CI_div_G];
    __shared__ float blockI[64 * LOOP_HW];

    #pragma unroll(1)
    for (int step = 0; step < CO_div_G; step += LOOP_CO) {
        for (int pos = threadIdx.x; pos < CI_div_G * LOOP_CO; pos += THREAD_PER_BLOCK)
            blockW[pos] = weight[(g * CO_div_G + step) * CI_div_G + pos];
        __syncthreads();
        int co = (threadIdx.x >> 6) + step;
        #pragma unroll
        for (int k = 0; k < LOOP_HW; k++) {
            int b_hw = b_hw_start + (threadIdx.x & 63) + k * 64;
            int b = b_hw / HW;
            int hw = b_hw % HW;
            if (b < B) {
                grad_out[k] = grad_output[(b * CO + g * CO_div_G + co) * HW + hw];
                r_out[k] = 1.0f / output[(b * CO + g * CO_div_G + co) * HW + hw];
            }
            else {
                grad_out[k] = 0.0f;
                r_out[k] = 1e-10f;
            }
        }
        #pragma unroll(1)
        for (int j = 0; j < CI_div_G; j++) {
            float w = blockW[(threadIdx.x >> 6) * CI_div_G + j];
            #pragma unroll
            for (int kk = 0; kk < LOOP_HW * 64; kk += LOOP_CO * 64) {
                int b = (b_hw_start + kk + threadIdx.x) / HW;
                int hw = (b_hw_start + kk + threadIdx.x) % HW;
                blockI[kk + threadIdx.x] = b < B ? input[(b * CI + g * CI_div_G + j) * HW + hw] : 0.0f;
            }
            __syncthreads();
            float res = 0.0f;
            #pragma unroll
            for (int k = 0; k < LOOP_HW; k++) {
                float x = blockI[k * 64 + (threadIdx.x & 63)];
                float t = w - x;
                res += grad_out[k] * pow_fun(abs(t) * r_out[k], p - 1) * (t > 0 ? 1 : -1);
            }
            res += __shfl_xor_sync(0xffffffff, res, 1);
            res += __shfl_xor_sync(0xffffffff, res, 2);
            res += __shfl_xor_sync(0xffffffff, res, 4);
            res += __shfl_xor_sync(0xffffffff, res, 8);
            res += __shfl_xor_sync(0xffffffff, res, 16);
            if ((threadIdx.x & 63) == 0) ans[(threadIdx.x >> 6) * CI_div_G + j] = res;
            __syncthreads();
            if ((threadIdx.x & 63) == 32) ans[(threadIdx.x >> 6) * CI_div_G + j] += res;
        }
        __syncthreads();
        for (int pos = threadIdx.x; pos < CI_div_G * LOOP_CO; pos += THREAD_PER_BLOCK)
            atomicAdd(&grad_weight[(g * CO_div_G + step) * CI_div_G + pos], ans[pos]);
    }
}

#define CALL_FUNC(func, thread, dim, var1, var2, paras...) \
    bool success = true; \
    if (var1 == 1 && var2 == 1) func<1, 1><<<dim, thread>>>(paras); \
    else if (var1 == 1 && var2 == 2) func<1, 2><<<dim, thread>>>(paras); \
    else if (var1 == 2 && var2 == 1) func<2, 1><<<dim, thread>>>(paras); \
    else if (var1 == 2 && var2 == 2) func<2, 2><<<dim, thread>>>(paras); \
    else if (var1 == 2 && var2 == 4) func<2, 4><<<dim, thread>>>(paras); \
    else if (var1 == 4 && var2 == 2) func<4, 2><<<dim, thread>>>(paras); \
    else if (var1 == 4 && var2 == 4) func<4, 4><<<dim, thread>>>(paras); \
    else if (var1 == 4 && var2 == 8) func<4, 8><<<dim, thread>>>(paras); \
    else if (var1 == 8 && var2 == 4) func<8, 4><<<dim, thread>>>(paras); \
    else if (var1 == 8 && var2 == 8) func<8, 8><<<dim, thread>>>(paras); \
    else if (var1 == 8 && var2 == 16) func<8, 16><<<dim, thread>>>(paras); \
    else if (var1 == 16 && var2 == 8) func<16, 8><<<dim, thread>>>(paras); \
    else if (var1 == 16 && var2 == 16) func<16, 16><<<dim, thread>>>(paras); \
    else if (var1 == 16 && var2 == 32) func<16, 32><<<dim, thread>>>(paras); \
    else if (var1 == 32 && var2 == 16) func<32, 16><<<dim, thread>>>(paras); \
    else if (var1 == 32 && var2 == 32) func<32, 32><<<dim, thread>>>(paras); \
    else if (var1 == 32 && var2 == 64) func<32, 64><<<dim, thread>>>(paras); \
    else if (var1 == 64 && var2 == 32) func<64, 32><<<dim, thread>>>(paras); \
    else if (var1 == 64 && var2 == 64) func<64, 64><<<dim, thread>>>(paras); \
    else if (var1 == 9 && var2 == 1) func<9, 1><<<dim, thread>>>(paras); \
    else if (var1 == 9 && var2 == 2) func<9, 2><<<dim, thread>>>(paras); \
    else if (var1 == 18 && var2 == 1) func<18, 1><<<dim, thread>>>(paras); \
    else if (var1 == 18 && var2 == 2) func<18, 2><<<dim, thread>>>(paras); \
    else if (var1 == 18 && var2 == 4) func<18, 4><<<dim, thread>>>(paras); \
    else if (var1 == 36 && var2 == 2) func<36, 2><<<dim, thread>>>(paras); \
    else if (var1 == 36 && var2 == 4) func<36, 4><<<dim, thread>>>(paras); \
    else if (var1 == 36 && var2 == 8) func<36, 8><<<dim, thread>>>(paras); \
    else if (var1 == 72 && var2 == 4) func<72, 4><<<dim, thread>>>(paras); \
    else if (var1 == 72 && var2 == 8) func<72, 8><<<dim, thread>>>(paras); \
    else if (var1 == 72 && var2 == 16) func<72, 16><<<dim, thread>>>(paras); \
    else if (var1 == 144 && var2 == 8) func<144, 8><<<dim, thread>>>(paras); \
    else if (var1 == 144 && var2 == 16) func<144, 16><<<dim, thread>>>(paras); \
    else if (var1 == 144 && var2 == 32) func<144, 32><<<dim, thread>>>(paras); \
    else success = false; \
    if (success) return;

const int BLOCK_SIZE = 16;
const int BATCH_BLOCK_SIZE = 8;

__device__ __forceinline__ float update_forward(float x, float w, float p, float r_max_x_sub_w) {
    float t = abs(x - w);
    return pow_fun(t * r_max_x_sub_w, p);
}

__device__ __forceinline__ void normalize(float& output_reg, float old_max, float r_new_max, float p) {
    output_reg = output_reg * pow_fun(old_max * r_new_max, p);
}

#define getX (conv ? blockI[i][threadIdx.x] : blockI[threadIdx.y][i])
#define getW (conv ? blockW[i][threadIdx.y] : blockW[i][threadIdx.x])

template <bool conv>
__global__ void norm_dist_forward_kernel(const float* __restrict__ input, const float* __restrict__ weight,
                                    int B, int CO_div_G, int CI_div_G, int HW, int G,
                                    float* __restrict__ output, float p) {
    float output_reg = 1e-10;
    float max_x_sub_w = 1e-10, r_max_x_sub_w = 1.0 / max_x_sub_w;

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
        float max_x_sub_w_batch = 0;
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++)
            max_x_sub_w_batch = max(max_x_sub_w_batch, abs(getX - getW));
        if (max_x_sub_w_batch > max_x_sub_w) {
            r_max_x_sub_w = __frcp_rn(max_x_sub_w_batch);
            normalize(output_reg, max_x_sub_w, r_max_x_sub_w, p);
            max_x_sub_w = max_x_sub_w_batch;
        }
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++)
            output_reg += update_forward(getX, getW, p, r_max_x_sub_w);
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
        float max_x_sub_w_batch = 0;
        for (int i = 0; i < (CI_div_G & (BLOCK_SIZE - 1)); i++)
            max_x_sub_w_batch = max(max_x_sub_w_batch, abs(getX - getW));
        if (max_x_sub_w_batch > max_x_sub_w) {
            r_max_x_sub_w = __frcp_rn(max_x_sub_w_batch);
            normalize(output_reg, max_x_sub_w, r_max_x_sub_w, p);
            max_x_sub_w = max_x_sub_w_batch;
        }
        #pragma unroll
        for (int i = 0; i < (CI_div_G & (BLOCK_SIZE - 1)); i++)
            output_reg += update_forward(getX, getW, p, r_max_x_sub_w);
        __syncthreads();
    }
    if (b < B && write_co < CO_div_G) {
        output_reg = __powf(output_reg, 1.0 / p) * max_x_sub_w;
        output[((b * G + blockIdx.z) * CO_div_G + write_co) * HW + hw] = output_reg;
    }
}

__device__ __forceinline__ float update_backward_input(float x, float w, float r_o, float g, float p) {
    float t = x - w;
    return g * pow_fun(abs(t) * r_o, p - 1) * (t > 0 ? 1 : -1);
}

template <bool conv>
__global__ void norm_dist_backward_input_kernel(const float* __restrict__ grad_output, const float* __restrict__ input,
                                                const float* __restrict__ weight, const float* __restrict__ output,
                                                int B, int CO_div_G, int CI_div_G, int HW, int G, float* __restrict__ grad_input, float p) {
    float output_reg = 0;

    int b_hw = blockIdx.y * BLOCK_SIZE + (conv ? threadIdx.x : threadIdx.y);
    int b = b_hw / HW;
    int hw = b_hw % HW;
    int write_ci = blockIdx.x * BLOCK_SIZE + (conv ? threadIdx.y : threadIdx.x);
    int read_ci = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    __shared__ float blockO[BLOCK_SIZE][BLOCK_SIZE]; // CO * B if conv else B * CO
    __shared__ float blockG[BLOCK_SIZE][BLOCK_SIZE]; // CO * B if conv else B * CO
    __shared__ float blockW[BLOCK_SIZE][BLOCK_SIZE + 2]; // CI * CO

    float x = 0;
    if (b < B && write_ci < CI_div_G) x = input[((b * G + blockIdx.z) * CI_div_G + write_ci) * HW + hw];
    int k;
    for (k = 0; k < (CO_div_G & ~(BLOCK_SIZE - 1)); k += BLOCK_SIZE) {
        if (b < B) {
            if (conv) {
                blockO[threadIdx.y][threadIdx.x] = __frcp_rn(output[((b * G + blockIdx.z) * CO_div_G + k + threadIdx.y) * HW + hw]);
                blockG[threadIdx.y][threadIdx.x] = grad_output[((b * G + blockIdx.z) * CO_div_G + k + threadIdx.y) * HW + hw];
            }
            else {
                blockO[threadIdx.y][threadIdx.x] = __frcp_rn(output[((b * G + blockIdx.z) * CO_div_G + k + threadIdx.x) * HW + hw]);
                blockG[threadIdx.y][threadIdx.x] = grad_output[((b * G + blockIdx.z) * CO_div_G + k + threadIdx.x) * HW + hw];
            }
        }
        if (read_ci < CI_div_G)
            blockW[threadIdx.x][threadIdx.y] = weight[(blockIdx.z * CO_div_G + k + threadIdx.y) * CI_div_G + read_ci];
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (conv) output_reg += update_backward_input(x, blockW[threadIdx.y][i], blockO[i][threadIdx.x], blockG[i][threadIdx.x], p);
            else output_reg += update_backward_input(x, blockW[threadIdx.x][i], blockO[threadIdx.y][i], blockG[threadIdx.y][i], p);
        }
        __syncthreads();
    }
    if (CO_div_G & (BLOCK_SIZE - 1)) {
        if (b < B) {
            if (conv && k + threadIdx.y < CO_div_G){
                blockO[threadIdx.y][threadIdx.x] = __frcp_rn(output[((b * G + blockIdx.z) * CO_div_G + k + threadIdx.y) * HW + hw]);
                blockG[threadIdx.y][threadIdx.x] = grad_output[((b * G + blockIdx.z) * CO_div_G + k + threadIdx.y) * HW + hw];
            }
            if (!conv && k + threadIdx.x < CO_div_G){
                blockO[threadIdx.y][threadIdx.x] = __frcp_rn(output[((b * G + blockIdx.z) * CO_div_G + k + threadIdx.x) * HW + hw]);
                blockG[threadIdx.y][threadIdx.x] = grad_output[((b * G + blockIdx.z) * CO_div_G + k + threadIdx.x) * HW + hw];
            }
        }
        if (k + threadIdx.y < CO_div_G && read_ci < CI_div_G)
            blockW[threadIdx.x][threadIdx.y] = weight[(blockIdx.z * CO_div_G + k + threadIdx.y) * CI_div_G + read_ci];
        __syncthreads();
        for (int i = 0; i < (CO_div_G & (BLOCK_SIZE - 1)); i++) {
            if (conv) output_reg += update_backward_input(x, blockW[threadIdx.y][i], blockO[i][threadIdx.x], blockG[i][threadIdx.x], p);
            else output_reg += update_backward_input(x, blockW[threadIdx.x][i], blockO[threadIdx.y][i], blockG[threadIdx.y][i], p);
        }
        __syncthreads();
    }
    if (b < B && write_ci < CI_div_G)
        grad_input[((b * G + blockIdx.z) * CI_div_G + write_ci) * HW + hw] = output_reg;
}

__device__ __forceinline__ float update_backward_weight(float x, float w, float r_o, float g, float p) {
    float t = w - x;
    return g * pow_fun(abs(t) * r_o, p - 1) * (t > 0 ? 1 : -1);
}

template <bool conv>
__global__ void norm_dist_backward_weight_kernel(const float* __restrict__ grad_output, const float* __restrict__ input,
                                                const float* __restrict__ weight, const float* __restrict__ output,
                                                int B, int CO_div_G, int CI_div_G, int HW, int G, float* __restrict__ grad_weight, float p) {
    float output_reg = 0;

    int write_co = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int write_ci = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int read_co = blockIdx.y * BLOCK_SIZE + (conv ? threadIdx.y : threadIdx.x);
    int read_ci = blockIdx.x * BLOCK_SIZE + (conv ? threadIdx.y : threadIdx.x);
    int B_start = B * (blockIdx.z % BATCH_BLOCK_SIZE) / BATCH_BLOCK_SIZE;
    int B_end = B * (blockIdx.z % BATCH_BLOCK_SIZE + 1) / BATCH_BLOCK_SIZE;
    int g = blockIdx.z / BATCH_BLOCK_SIZE;
    int B_num = B_end - B_start;

    __shared__ float blockI[BLOCK_SIZE][BLOCK_SIZE + 2]; // B * CI if conv else CI * B
    __shared__ float blockO[BLOCK_SIZE][BLOCK_SIZE]; // CO * B if conv else B * CO
    __shared__ float blockG[BLOCK_SIZE][BLOCK_SIZE]; // CO * B if conv else B * CO

    float w = 0;
    if (write_co < CO_div_G && write_ci < CI_div_G) w = weight[(g * CO_div_G + write_co) * CI_div_G + write_ci];
    int k;
    for (k = 0; k < ((B_num * HW) & ~(BLOCK_SIZE - 1)); k += BLOCK_SIZE) {
        int b = B_start + (conv ? (k + threadIdx.x) / HW : k + threadIdx.y);
        int hw = conv ? (k + threadIdx.x) % HW : 0;
        if (read_ci < CI_div_G) blockI[threadIdx.x][threadIdx.y] = input[((b * G + g) * CI_div_G + read_ci) * HW + hw];
        if (read_co < CO_div_G) {
            blockO[threadIdx.y][threadIdx.x] = __frcp_rn(output[((b * G + g) * CO_div_G + read_co) * HW + hw]);
            blockG[threadIdx.y][threadIdx.x] = grad_output[((b * G + g) * CO_div_G + read_co) * HW + hw];
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (conv) output_reg += update_backward_weight(blockI[i][threadIdx.x], w, blockO[threadIdx.y][i], blockG[threadIdx.y][i], p);
            else output_reg += update_backward_weight(blockI[threadIdx.x][i], w, blockO[i][threadIdx.y], blockG[i][threadIdx.y], p);
        }
        __syncthreads();
    }
    if ((B_num * HW) & (BLOCK_SIZE - 1)) {
        int b = B_start + (conv ? (k + threadIdx.x) / HW : k + threadIdx.y);
        int hw = conv ? (k + threadIdx.x) % HW : 0;
        if (b < B_end) {
            if (read_ci < CI_div_G) blockI[threadIdx.x][threadIdx.y] = input[((b * G + g) * CI_div_G + read_ci) * HW + hw];
            if (read_co < CO_div_G) {
                blockO[threadIdx.y][threadIdx.x] = __frcp_rn(output[((b * G + g) * CO_div_G + read_co) * HW + hw]);
                blockG[threadIdx.y][threadIdx.x] = grad_output[((b * G + g) * CO_div_G + read_co) * HW + hw];
            }
        }
        __syncthreads();
        for (int i = 0; i < ((B_num * HW) & (BLOCK_SIZE - 1)); i++) {
            if (conv) output_reg += update_backward_weight(blockI[i][threadIdx.x], w, blockO[threadIdx.y][i], blockG[threadIdx.y][i], p);
            else output_reg += update_backward_weight(blockI[threadIdx.x][i], w, blockO[i][threadIdx.y], blockG[i][threadIdx.y], p);
        }
        __syncthreads();
    }
    if (write_co < CO_div_G && write_ci < CI_div_G)
        atomicAdd(&grad_weight[(g * CO_div_G + write_co) * CI_div_G + write_ci], output_reg);
}

void norm_dist_forward_cuda(const float* input, const float* weight,
                       int B, int CO, int CI, int G, int HW, float* output, float p) {

    dim3 dimGrid(G, (B * HW - 1) / THREAD_PER_BLOCK + 1);
    CALL_FUNC(norm_dist_forward_kernel, THREAD_PER_BLOCK, dimGrid, CI / G, CO / G, input, weight, B, CO, CI, HW, output, p);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid2((CO / G + BLOCK_SIZE - 1) / BLOCK_SIZE, (B * HW + BLOCK_SIZE - 1) / BLOCK_SIZE, G);
    if (HW == 1) norm_dist_forward_kernel<false><<<dimGrid2, dimBlock>>>(input, weight, B, CO / G, CI / G, HW, G, output, p);
    else norm_dist_forward_kernel<true><<<dimGrid2, dimBlock>>>(input, weight, B, CO / G, CI / G, HW, G, output, p);
}
void norm_dist_backward_input_cuda(const float* grad_output, const float* input, const float* weight, const float* output,
                              int B, int CO, int CI, int G, int HW, float* grad_input, float p) {

    dim3 dimGrid(G, (B * HW - 1) / THREAD_PER_BLOCK + 1);
    CALL_FUNC(norm_dist_backward_input_kernel, THREAD_PER_BLOCK, dimGrid, CI / G, CO / G, grad_output, input, weight, output, B, CO, CI, HW, grad_input, p);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid2((CI / G + BLOCK_SIZE - 1) / BLOCK_SIZE, (B * HW + BLOCK_SIZE - 1) / BLOCK_SIZE, G);
    if (HW == 1) norm_dist_backward_input_kernel<false><<<dimGrid2, dimBlock>>>(grad_output, input, weight, output, B, CO / G, CI / G, HW, G, grad_input, p);
    else norm_dist_backward_input_kernel<true><<<dimGrid2, dimBlock>>>(grad_output, input, weight, output, B, CO / G, CI / G, HW, G, grad_input, p);
}
void norm_dist_backward_weight_cuda(const float* grad_output, const float* input, const float* weight, const float* output,
                               int B, int CO, int CI, int G, int HW, float* grad_weight, float p) {

    dim3 dimGrid(G, (B * HW - 1) / (64 * 8) + 1);
    CALL_FUNC(norm_dist_backward_weight_kernel, min(64 * (CO / G), THREAD_PER_BLOCK), dimGrid, CI / G, CO / G, grad_output, input, weight, output, B, CO, CI, HW, grad_weight, p);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid2((CI / G + BLOCK_SIZE - 1) / BLOCK_SIZE, (CO / G + BLOCK_SIZE - 1) / BLOCK_SIZE, BATCH_BLOCK_SIZE * G);
    if (HW == 1) norm_dist_backward_weight_kernel<false><<<dimGrid2, dimBlock>>>(grad_output, input, weight, output, B, CO / G, CI / G, HW, G, grad_weight, p);
    else norm_dist_backward_weight_kernel<true><<<dimGrid2, dimBlock>>>(grad_output, input, weight, output, B, CO / G, CI / G, HW, G, grad_weight, p);
}
