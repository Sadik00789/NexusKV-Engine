// csrc/paged_attention.cu
#include <cuda_runtime.h>
#include <cstdint>
#include <float.h>
#include <math.h>

// Every launch checks the asynchronous launch status. The hot path never calls
// cudaDeviceSynchronize; callers synchronize when they read results back.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t status__ = (call);                                         \
        if (status__ != cudaSuccess) {                                         \
            return static_cast<int>(status__);                                 \
        }                                                                      \
    } while (0)

/*
 * Paged decode attention.
 *
 * One block per (sequence, head), 128 threads. The query vector is loaded into
 * shared memory, each token's dot product is reduced with warp shuffles and a
 * cross-warp reduction, and the output/value accumulation uses online softmax:
 *
 *   m_new = max(m, s); corr = exp(m - m_new); p = exp(s - m_new)
 *   l = l * corr + p; acc[d] = acc[d] * corr + p * v[d]; out = acc / l
 *
 * head_dim is limited to 128 so each thread owns one output dimension.
 */
__global__ void paged_decode_attention_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens,
    const int32_t max_blocks_per_seq,
    const int32_t num_heads,
    const int32_t num_kv_heads,
    const int32_t head_dim,
    const int32_t block_size,
    const float scale)
{
    const int seq = blockIdx.x;
    const int head = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_warps = blockDim.x >> 5;
    const int seq_len = seq_lens[seq];
    if (seq_len <= 0 || num_kv_heads <= 0) {
        return;
    }
    const int n_rep = num_heads / num_kv_heads;
    const int kv_head = head / n_rep;

    extern __shared__ float shared[];
    float* q_shared = shared;
    float* warp_reduce = shared + head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        q_shared[d] = q[(seq * num_heads + head) * head_dim + d] * scale;
    }
    __syncthreads();

    float running_max = -FLT_MAX;
    float denominator = 0.0f;
    float accumulator = 0.0f;
    const int out_dim = tid;

    for (int token = 0; token < seq_len; ++token) {
        const int logical_block = token / block_size;
        const int block_offset = token % block_size;
        const int32_t physical_block =
            block_tables[seq * max_blocks_per_seq + logical_block];
        if (physical_block < 0) {
            continue;
        }
        const int kv_base =
            ((physical_block * block_size + block_offset) * num_kv_heads + kv_head) * head_dim;

        float partial = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            partial += q_shared[d] * k_cache[kv_base + d];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(0xffffffffu, partial, offset);
        }
        if (lane == 0) {
            warp_reduce[warp] = partial;
        }
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (int w = 0; w < num_warps; ++w) {
                total += warp_reduce[w];
            }
            warp_reduce[0] = total;
        }
        __syncthreads();
        const float score = warp_reduce[0];

        const float new_max = fmaxf(running_max, score);
        const float correction = expf(running_max - new_max);
        const float probability = expf(score - new_max);
        denominator = denominator * correction + probability;
        if (out_dim < head_dim) {
            accumulator = accumulator * correction + probability * v_cache[kv_base + out_dim];
        }
        running_max = new_max;
        __syncthreads();  // warp_reduce[0] is reused next iteration
    }

    if (out_dim < head_dim) {
        out[(seq * num_heads + head) * head_dim + out_dim] =
            denominator > 0.0f ? accumulator / denominator : 0.0f;
    }
}

__global__ void reshape_and_cache_kernel(
    const float* __restrict__ key,
    const float* __restrict__ value,
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const int32_t* __restrict__ slot_mapping,
    const int32_t num_tokens,
    const int32_t num_heads,
    const int32_t head_dim)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t per_token = static_cast<int64_t>(num_heads) * head_dim;
    const int64_t total = static_cast<int64_t>(num_tokens) * per_token;
    if (index >= total) {
        return;
    }
    const int token = static_cast<int>(index / per_token);
    const int remainder = static_cast<int>(index % per_token);
    const int32_t slot = slot_mapping[token];
    if (slot < 0) {
        return;
    }
    const int64_t destination = static_cast<int64_t>(slot) * per_token + remainder;
    k_cache[destination] = key[index];
    v_cache[destination] = value[index];
}

__global__ void copy_block_kernel(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const int32_t dst_block,
    const int32_t src_block,
    const int32_t block_size,
    const int32_t num_heads,
    const int32_t head_dim)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t per_block = static_cast<int64_t>(block_size) * num_heads * head_dim;
    if (index >= per_block) {
        return;
    }
    const int64_t source = static_cast<int64_t>(src_block) * per_block + index;
    const int64_t destination = static_cast<int64_t>(dst_block) * per_block + index;
    k_cache[destination] = k_cache[source];
    v_cache[destination] = v_cache[source];
}

extern "C" int launch_paged_attention(
    float* out,
    const float* q,
    const float* k_cache,
    const float* v_cache,
    const int32_t* block_tables,
    const int32_t* seq_lens,
    int32_t max_blocks_per_seq,
    int32_t num_seqs,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t block_size,
    float scale)
{
    if (out == nullptr || q == nullptr || k_cache == nullptr || v_cache == nullptr ||
        block_tables == nullptr || seq_lens == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (num_seqs <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
        head_dim > 128 || num_heads % num_kv_heads != 0 || block_size <= 0 ||
        max_blocks_per_seq <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    dim3 grid(static_cast<unsigned>(num_seqs), static_cast<unsigned>(num_heads));
    dim3 block(128);
    const size_t shared_bytes = static_cast<size_t>(head_dim + 4) * sizeof(float);
    paged_decode_attention_kernel<<<grid, block, shared_bytes, 0>>>(
        out, q, k_cache, v_cache, block_tables, seq_lens,
        max_blocks_per_seq, num_heads, num_kv_heads, head_dim, block_size, scale);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int launch_reshape_and_cache(
    const float* key,
    const float* value,
    float* k_cache,
    float* v_cache,
    const int32_t* slot_mapping,
    int32_t num_tokens,
    int32_t num_heads,
    int32_t head_dim)
{
    if (key == nullptr || value == nullptr || k_cache == nullptr || v_cache == nullptr ||
        slot_mapping == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (num_tokens <= 0 || num_heads <= 0 || head_dim <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int64_t total = static_cast<int64_t>(num_tokens) * num_heads * head_dim;
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    reshape_and_cache_kernel<<<blocks, threads, 0, 0>>>(
        key, value, k_cache, v_cache, slot_mapping, num_tokens, num_heads, head_dim);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int launch_copy_block(
    float* k_cache,
    float* v_cache,
    int32_t dst_block,
    int32_t src_block,
    int32_t block_size,
    int32_t num_heads,
    int32_t head_dim)
{
    if (k_cache == nullptr || v_cache == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (dst_block < 0 || src_block < 0 || block_size <= 0 || num_heads <= 0 ||
        head_dim <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int64_t total = static_cast<int64_t>(block_size) * num_heads * head_dim;
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    copy_block_kernel<<<blocks, threads, 0, 0>>>(
        k_cache, v_cache, dst_block, src_block, block_size, num_heads, head_dim);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// Device-buffer helpers for the safe Rust reference wrappers and GPU tests.
// The performance hot path should keep persistent device buffers and call the
// launch_* functions directly instead of copying host slices every call.
extern "C" int nexuskv_device_alloc(size_t bytes, void** pointer)
{
    if (pointer == nullptr || bytes == 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    return static_cast<int>(cudaMalloc(pointer, bytes));
}

extern "C" int nexuskv_device_free(void* pointer)
{
    return static_cast<int>(cudaFree(pointer));
}

extern "C" int nexuskv_copy_to_device(void* destination, const void* source, size_t bytes)
{
    if (destination == nullptr || source == nullptr || bytes == 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    return static_cast<int>(
        cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice));
}

extern "C" int nexuskv_copy_to_host(void* destination, const void* source, size_t bytes)
{
    if (destination == nullptr || source == nullptr || bytes == 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    return static_cast<int>(
        cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost));
}
