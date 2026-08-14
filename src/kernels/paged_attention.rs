// src/kernels/paged_attention.rs
use std::os::raw::c_int;

// Rust 2024 explicit unsafe extern FFI declaration
#[allow(dead_code)]
unsafe extern "C" {
    pub fn launch_paged_attention(
        out_ptr: *mut f32,
        q_ptr: *const f32,
        k_cache_ptr: *const f32,
        v_cache_ptr: *const f32,
        block_tables_ptr: *const i32,
        seq_lens_ptr: *const i32,
        max_blocks_per_seq: c_int,
        num_seqs: c_int,
        num_heads: c_int,
        head_dim: c_int,
        block_size: c_int,
        scale: f32,
    );
}

/// Safe high-level Rust entrypoint for the PagedAttention CUDA kernel.
#[allow(dead_code)]
pub fn run_paged_attention(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    block_tables: &[i32],
    seq_lens: &[i32],
    max_blocks_per_seq: usize,
    num_seqs: usize,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<(), &'static str> {
    let expected_out_len = num_seqs * num_heads * head_dim;
    if out.len() < expected_out_len || q.len() < expected_out_len {
        return Err("Buffer length mismatch for output or query vectors");
    }

    if seq_lens.len() < num_seqs {
        return Err("Sequence lengths slice size is smaller than num_seqs");
    }

    let scale = 1.0f32 / (head_dim as f32).sqrt();

    unsafe {
        launch_paged_attention(
            out.as_mut_ptr(),
            q.as_ptr(),
            k_cache.as_ptr(),
            v_cache.as_ptr(),
            block_tables.as_ptr(),
            seq_lens.as_ptr(),
            max_blocks_per_seq as c_int,
            num_seqs as c_int,
            num_heads as c_int,
            head_dim as c_int,
            block_size as c_int,
            scale,
        );
    }

    Ok(())
}

/// Pure Rust reference implementation of PagedAttention for CPU fallback and automated test validation.
#[allow(dead_code)]
pub fn run_paged_attention_cpu_ref(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    block_tables: &[i32],
    seq_lens: &[i32],
    max_blocks_per_seq: usize,
    num_seqs: usize,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
) {
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    for seq_idx in 0..num_seqs {
        let seq_len = seq_lens[seq_idx] as usize;
        if seq_len == 0 {
            continue;
        }

        for head_idx in 0..num_heads {
            for dim_idx in 0..head_dim {
                let q_offset = (seq_idx * num_heads * head_dim) + (head_idx * head_dim) + dim_idx;
                let q_val = q[q_offset] * scale;

                let mut acc_score = 0.0f32;
                let mut acc_value = 0.0f32;

                for token_idx in 0..seq_len {
                    let block_table_idx = token_idx / block_size;
                    let block_offset = token_idx % block_size;

                    let phys_block_id = block_tables[seq_idx * max_blocks_per_seq + block_table_idx] as usize;

                    let kv_offset = (phys_block_id * block_size * num_heads * head_dim)
                        + (block_offset * num_heads * head_dim)
                        + (head_idx * head_dim)
                        + dim_idx;

                    let k_val = k_cache[kv_offset];
                    let v_val = v_cache[kv_offset];

                    acc_score += q_val * k_val;
                    acc_value += v_val;
                }

                let out_offset = (seq_idx * num_heads * head_dim) + (head_idx * head_dim) + dim_idx;
                out[out_offset] = acc_score * (acc_value / seq_len as f32);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_ref_attention() {
        let num_seqs = 1;
        let num_heads = 2;
        let head_dim = 4;
        let block_size = 4;
        let max_blocks = 2;

        let mut out = vec![0.0f32; num_seqs * num_heads * head_dim];
        let q = vec![1.0f32; num_seqs * num_heads * head_dim];
        let k_cache = vec![0.5f32; 2 * block_size * num_heads * head_dim];
        let v_cache = vec![2.0f32; 2 * block_size * num_heads * head_dim];
        let block_tables = vec![0, 1];
        let seq_lens = vec![4];

        run_paged_attention_cpu_ref(
            &mut out,
            &q,
            &k_cache,
            &v_cache,
            &block_tables,
            &seq_lens,
            max_blocks,
            num_seqs,
            num_heads,
            head_dim,
            block_size,
        );

        assert!(out[0] > 0.0);
    }
}