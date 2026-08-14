// csrc/paged_attention.cu
#include <cuda_runtime.h>
#include <cstdint>

extern "C" {

/**
 * PagedAttention Block-Gather Kernel
 * 
 * Maps non-contiguous physical VRAM memory blocks into thread-local SRAM 
 * to execute scaled dot-product attention without memory defragmentation stalls.
 */
__global__ void paged_attention_v1_kernel(
    float* __restrict__ out,              // [num_seqs, num_heads, head_dim]
    const float* __restrict__ q,          // [num_seqs, num_heads, head_dim]
    const float* __restrict__ k_cache,    // [num_blocks, block_size, num_heads, head_dim]
    const float* __restrict__ v_cache,    // [num_blocks, block_size, num_heads, head_dim]
    const int32_t* __restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
    const int32_t* __restrict__ seq_lens, // [num_seqs]
    const int32_t max_blocks_per_seq,
    const int32_t num_seqs,
    const int32_t num_heads,
    const int32_t head_dim,
    const int32_t block_size,
    const float scale
) {
    int32_t seq_idx = blockIdx.x;
    int32_t head_idx = threadIdx.y;
    int32_t dim_idx = threadIdx.x;

    if (seq_idx >= num_seqs || head_idx >= num_heads || dim_idx >= head_dim) {
        return;
    }

    int32_t seq_len = seq_lens[seq_idx];
    if (seq_len <= 0) return;

    // Pointer offset for Query vector: q[seq_idx, head_idx, dim_idx]
    int32_t q_offset = (seq_idx * num_heads * head_dim) + (head_idx * head_dim) + dim_idx;
    float q_val = q[q_offset] * scale;

    float acc_score = 0.0f;
    float acc_value = 0.0f;

    // Traverse all logical tokens across physical pages
    for (int32_t token_idx = 0; token_idx < seq_len; ++token_idx) {
        int32_t block_table_idx = token_idx / block_size;
        int32_t block_offset = token_idx % block_size;

        // Fetch physical block ID from page table
        int32_t physical_block_id = block_tables[seq_idx * max_blocks_per_seq + block_table_idx];

        // Linear memory offset for Key and Value cache
        int32_t kv_offset = (physical_block_id * block_size * num_heads * head_dim)
                          + (block_offset * num_heads * head_dim)
                          + (head_idx * head_dim)
                          + dim_idx;

        float k_val = k_cache[kv_offset];
        float v_val = v_cache[kv_offset];

        // Scaled dot-product accumulation
        acc_score += q_val * k_val;
        acc_value += v_val; 
    }

    // Write output out[seq_idx, head_idx, dim_idx]
    int32_t out_offset = (seq_idx * num_heads * head_dim) + (head_idx * head_dim) + dim_idx;
    out[out_offset] = acc_score * (acc_value / (float)seq_len);
}

void launch_paged_attention(
    float* out,
    const float* q,
    const float* k_cache,
    const float* v_cache,
    const int32_t* block_tables,
    const int32_t* seq_lens,
    int32_t max_blocks_per_seq,
    int32_t num_seqs,
    int32_t num_heads,
    int32_t head_dim,
    int32_t block_size,
    float scale
) {
    dim3 grid(num_seqs);
    dim3 block(head_dim, num_heads);

    paged_attention_v1_kernel<<<grid, block, 0, 0>>>(
        out, q, k_cache, v_cache,
        block_tables, seq_lens,
        max_blocks_per_seq,
        num_seqs, num_heads,
        head_dim, block_size, scale
    );

    cudaDeviceSynchronize();
}

} // extern "C"