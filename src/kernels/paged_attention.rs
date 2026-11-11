// src/kernels/paged_attention.rs
//!
//! Paged decode attention.
//!
//! The CPU implementation is a real scaled dot-product attention reference with
//! softmax and is used as the numerical oracle for the CUDA kernel. The CUDA
//! entry points are compiled only with the `cuda` feature; the default build has
//! no nvcc or GPU dependency.

#[cfg(feature = "cuda")]
use half::f16;
#[cfg(feature = "cuda")]
use std::ffi::c_void;
#[cfg(feature = "cuda")]
use std::mem::size_of_val;
#[cfg(feature = "cuda")]
use std::os::raw::c_int;

#[cfg(feature = "cuda")]
mod ffi {
    use std::ffi::c_void;
    use std::os::raw::{c_float, c_int};

    // SAFETY: these symbols are provided by `csrc/paged_attention.cu` when the
    // `cuda` feature is enabled, and the signatures below match its extern "C"
    // surface exactly.
    unsafe extern "C" {
        pub fn launch_paged_attention(
            out: *mut c_float,
            q: *const c_float,
            k_cache: *const c_float,
            v_cache: *const c_float,
            block_tables: *const c_int,
            seq_lens: *const c_int,
            max_blocks_per_seq: c_int,
            num_seqs: c_int,
            num_heads: c_int,
            num_kv_heads: c_int,
            head_dim: c_int,
            block_size: c_int,
            scale: c_float,
        ) -> c_int;

        /// Half-precision variant. `out`, `q`, `k_cache`, and `v_cache` are
        /// `__half*`; `stream` is a CUDA stream handle (null for default).
        pub fn launch_paged_attention_f16(
            out: *mut c_void,
            q: *const c_void,
            k_cache: *const c_void,
            v_cache: *const c_void,
            block_tables: *const c_int,
            seq_lens: *const c_int,
            max_blocks_per_seq: c_int,
            num_seqs: c_int,
            num_heads: c_int,
            num_kv_heads: c_int,
            head_dim: c_int,
            block_size: c_int,
            scale: c_float,
            stream: *mut c_void,
        ) -> c_int;

        pub fn launch_reshape_and_cache(
            key: *const c_float,
            value: *const c_float,
            k_cache: *mut c_float,
            v_cache: *mut c_float,
            slot_mapping: *const c_int,
            num_tokens: c_int,
            num_heads: c_int,
            head_dim: c_int,
        ) -> c_int;

        pub fn launch_copy_block(
            k_cache: *mut c_float,
            v_cache: *mut c_float,
            dst_block: c_int,
            src_block: c_int,
            block_size: c_int,
            num_heads: c_int,
            head_dim: c_int,
        ) -> c_int;

        pub fn nexuskv_device_alloc(bytes: usize, pointer: *mut *mut c_void) -> c_int;
        pub fn nexuskv_device_free(pointer: *mut c_void) -> c_int;
        pub fn nexuskv_copy_to_device(
            destination: *mut c_void,
            source: *const c_void,
            bytes: usize,
        ) -> c_int;
        pub fn nexuskv_copy_to_host(
            destination: *mut c_void,
            source: *const c_void,
            bytes: usize,
        ) -> c_int;
    }
}

/// Host-slice convenience buffer for the CUDA reference wrappers. It allocates
/// device memory, copies inputs in, and copies results back. The performance
/// hot path should keep persistent device buffers and call the launch functions
/// directly.
#[cfg(feature = "cuda")]
struct DeviceBuffer {
    pointer: *mut c_void,
    bytes: usize,
}

#[cfg(feature = "cuda")]
impl DeviceBuffer {
    fn new(bytes: usize) -> Result<Self, String> {
        if bytes == 0 {
            return Err("cannot allocate an empty device buffer".to_string());
        }
        let mut pointer = std::ptr::null_mut();
        // SAFETY: `pointer` is a valid out-parameter and `bytes` is non-zero.
        let code = unsafe { ffi::nexuskv_device_alloc(bytes, &mut pointer) };
        if code != 0 {
            return Err(format!("cudaMalloc failed with code {code}"));
        }
        Ok(Self { pointer, bytes })
    }

    fn from_slice<T>(data: &[T]) -> Result<Self, String> {
        let buffer = Self::new(size_of_val(data))?;
        buffer.copy_in(data)?;
        Ok(buffer)
    }

    fn copy_in<T>(&self, data: &[T]) -> Result<(), String> {
        if size_of_val(data) != self.bytes {
            return Err("device buffer size mismatch on copy-in".to_string());
        }
        // SAFETY: the source slice is valid for `self.bytes`, and the
        // destination is a live device allocation of exactly that size.
        let code = unsafe {
            ffi::nexuskv_copy_to_device(self.pointer, data.as_ptr() as *const c_void, self.bytes)
        };
        if code != 0 {
            return Err(format!("cudaMemcpy host->device failed with code {code}"));
        }
        Ok(())
    }

    fn copy_out<T>(&self, data: &mut [T]) -> Result<(), String> {
        if size_of_val(data) != self.bytes {
            return Err("device buffer size mismatch on copy-out".to_string());
        }
        // SAFETY: the destination slice is valid for `self.bytes`, and the
        // source is a live device allocation of exactly that size.
        let code = unsafe {
            ffi::nexuskv_copy_to_host(data.as_mut_ptr() as *mut c_void, self.pointer, self.bytes)
        };
        if code != 0 {
            return Err(format!("cudaMemcpy device->host failed with code {code}"));
        }
        Ok(())
    }

    fn as_ptr<T>(&self) -> *const T {
        self.pointer as *const T
    }

    fn as_mut_ptr<T>(&self) -> *mut T {
        self.pointer as *mut T
    }
}

#[cfg(feature = "cuda")]
impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // SAFETY: the pointer was allocated by cudaMalloc in `new` and is freed
        // exactly once here.
        let _ = unsafe { ffi::nexuskv_device_free(self.pointer) };
    }
}

/// Safe CUDA entry point. Returns an error without the `cuda` feature.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
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
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<(), String> {
    if head_dim == 0
        || num_kv_heads == 0
        || num_heads == 0
        || !num_heads.is_multiple_of(num_kv_heads)
    {
        return Err("num_heads must be a positive multiple of num_kv_heads".to_string());
    }
    let expected = num_seqs
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| "attention size overflow".to_string())?;
    if out.len() < expected || q.len() < expected {
        return Err("out/q buffers are smaller than num_seqs*num_heads*head_dim".to_string());
    }
    if seq_lens.len() < num_seqs {
        return Err("seq_lens is smaller than num_seqs".to_string());
    }
    if block_tables.len() < num_seqs * max_blocks_per_seq {
        return Err("block_tables is smaller than num_seqs*max_blocks_per_seq".to_string());
    }
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let out_buffer = DeviceBuffer::new(expected * std::mem::size_of::<f32>())?;
    let q_buffer = DeviceBuffer::from_slice(&q[..expected])?;
    let k_buffer = DeviceBuffer::from_slice(k_cache)?;
    let v_buffer = DeviceBuffer::from_slice(v_cache)?;
    let table_buffer = DeviceBuffer::from_slice(block_tables)?;
    let lens_buffer = DeviceBuffer::from_slice(seq_lens)?;

    // SAFETY: every device pointer is a live allocation of the size copied in
    // above; the kernel reads q/k/v/tables/lens and writes exactly `expected`
    // output elements.
    let code = unsafe {
        ffi::launch_paged_attention(
            out_buffer.as_mut_ptr::<f32>(),
            q_buffer.as_ptr::<f32>(),
            k_buffer.as_ptr::<f32>(),
            v_buffer.as_ptr::<f32>(),
            table_buffer.as_ptr::<i32>(),
            lens_buffer.as_ptr::<i32>(),
            max_blocks_per_seq as c_int,
            num_seqs as c_int,
            num_heads as c_int,
            num_kv_heads as c_int,
            head_dim as c_int,
            block_size as c_int,
            scale,
        )
    };
    if code != 0 {
        return Err(format!("CUDA paged attention failed with code {code}"));
    }
    out_buffer.copy_out(&mut out[..expected])?;
    Ok(())
}

#[allow(clippy::too_many_arguments, unused_variables)]
#[cfg(not(feature = "cuda"))]
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
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<(), String> {
    Err("NexusKV was built without the `cuda` feature".to_string())
}

/// Safe f16 CUDA entry point used by parity tests. Inputs and outputs are f32
/// host slices; the buffer is converted to half on the device.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
pub fn run_paged_attention_f16(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    block_tables: &[i32],
    seq_lens: &[i32],
    max_blocks_per_seq: usize,
    num_seqs: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<(), String> {
    if head_dim == 0
        || num_kv_heads == 0
        || num_heads == 0
        || !num_heads.is_multiple_of(num_kv_heads)
    {
        return Err("num_heads must be a positive multiple of num_kv_heads".to_string());
    }
    let expected = num_seqs
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| "attention size overflow".to_string())?;
    if out.len() < expected || q.len() < expected {
        return Err("out/q buffers are smaller than num_seqs*num_heads*head_dim".to_string());
    }
    if seq_lens.len() < num_seqs {
        return Err("seq_lens is smaller than num_seqs".to_string());
    }
    if block_tables.len() < num_seqs * max_blocks_per_seq {
        return Err("block_tables is smaller than num_seqs*max_blocks_per_seq".to_string());
    }
    let to_half =
        |values: &[f32]| -> Vec<f16> { values.iter().map(|&value| f16::from_f32(value)).collect() };
    let q_half = to_half(&q[..expected]);
    let k_half = to_half(k_cache);
    let v_half = to_half(v_cache);
    let out_buffer = DeviceBuffer::new(expected * std::mem::size_of::<f16>())?;
    let q_buffer = DeviceBuffer::from_slice(&q_half)?;
    let k_buffer = DeviceBuffer::from_slice(&k_half)?;
    let v_buffer = DeviceBuffer::from_slice(&v_half)?;
    let table_buffer = DeviceBuffer::from_slice(block_tables)?;
    let lens_buffer = DeviceBuffer::from_slice(seq_lens)?;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    // SAFETY: every device pointer is a live allocation of the size copied in
    // above; the kernel reads q/k/v/tables/lens and writes exactly `expected`
    // half elements. A null stream means the default stream, and the safe
    // wrapper copies results back synchronously.
    let code = unsafe {
        ffi::launch_paged_attention_f16(
            out_buffer.as_mut_ptr::<f16>() as *mut c_void,
            q_buffer.as_ptr::<f16>() as *const c_void,
            k_buffer.as_ptr::<f16>() as *const c_void,
            v_buffer.as_ptr::<f16>() as *const c_void,
            table_buffer.as_ptr::<i32>(),
            lens_buffer.as_ptr::<i32>(),
            max_blocks_per_seq as c_int,
            num_seqs as c_int,
            num_heads as c_int,
            num_kv_heads as c_int,
            head_dim as c_int,
            block_size as c_int,
            scale,
            std::ptr::null_mut(),
        )
    };
    if code != 0 {
        return Err(format!("CUDA f16 paged attention failed with code {code}"));
    }
    let mut half_out = vec![f16::ZERO; expected];
    out_buffer.copy_out(&mut half_out)?;
    for (destination, value) in out[..expected].iter_mut().zip(half_out) {
        *destination = value.to_f32();
    }
    Ok(())
}

/// Raw device-pointer entry point for the live candle decode path.
///
/// # Safety
/// `out`, `q`, `k_cache`, and `v_cache` must be live `__half` device buffers of
/// the sizes implied by the dimensions; `block_tables`/`seq_lens` must be live
/// device `i32` buffers; `stream` must be a valid CUDA stream from the same
/// context. The caller (candle executor) upholds these by holding the tensor
/// storage alive through the launch.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_paged_attention_f16_device(
    out: *mut c_void,
    q: *const c_void,
    k_cache: *const c_void,
    v_cache: *const c_void,
    block_tables: *const i32,
    seq_lens: *const i32,
    max_blocks_per_seq: usize,
    num_seqs: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    scale: f32,
    stream: *mut c_void,
) -> Result<(), String> {
    // SAFETY: the caller guarantees every pointer is valid for the dimensions
    // and remains alive until the stream-ordered launch completes.
    let code = unsafe {
        ffi::launch_paged_attention_f16(
            out,
            q,
            k_cache,
            v_cache,
            block_tables,
            seq_lens,
            max_blocks_per_seq as c_int,
            num_seqs as c_int,
            num_heads as c_int,
            num_kv_heads as c_int,
            head_dim as c_int,
            block_size as c_int,
            scale,
            stream,
        )
    };
    if code != 0 {
        return Err(format!("CUDA f16 paged attention failed with code {code}"));
    }
    Ok(())
}

/// Write key/value rows into the flat slot-indexed KV pool (CUDA feature only).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn run_reshape_and_cache(
    key: &[f32],
    value: &[f32],
    k_cache: &mut [f32],
    v_cache: &mut [f32],
    slot_mapping: &[i32],
    num_tokens: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<(), String> {
    let expected = num_tokens
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| "cache size overflow".to_string())?;
    if key.len() < expected || value.len() < expected {
        return Err("key/value buffers are too small".to_string());
    }
    if slot_mapping.len() < num_tokens {
        return Err("slot_mapping is too small".to_string());
    }
    if k_cache.is_empty() || v_cache.is_empty() {
        return Err("KV cache buffers must not be empty".to_string());
    }

    let key_buffer = DeviceBuffer::from_slice(&key[..expected])?;
    let value_buffer = DeviceBuffer::from_slice(&value[..expected])?;
    let k_buffer = DeviceBuffer::from_slice(k_cache)?;
    let v_buffer = DeviceBuffer::from_slice(v_cache)?;
    let slot_buffer = DeviceBuffer::from_slice(&slot_mapping[..num_tokens])?;

    // SAFETY: device allocations are live and sized; the kernel writes only the
    // mapped slots and never reads outside the copied inputs.
    let code = unsafe {
        ffi::launch_reshape_and_cache(
            key_buffer.as_ptr::<f32>(),
            value_buffer.as_ptr::<f32>(),
            k_buffer.as_mut_ptr::<f32>(),
            v_buffer.as_mut_ptr::<f32>(),
            slot_buffer.as_ptr::<i32>(),
            num_tokens as c_int,
            num_heads as c_int,
            head_dim as c_int,
        )
    };
    if code != 0 {
        return Err(format!("CUDA reshape_and_cache failed with code {code}"));
    }
    k_buffer.copy_out(k_cache)?;
    v_buffer.copy_out(v_cache)?;
    Ok(())
}

/// Copy one physical block between slots (CUDA feature only).
#[cfg(feature = "cuda")]
pub fn run_copy_block(
    k_cache: &mut [f32],
    v_cache: &mut [f32],
    dst_block: usize,
    src_block: usize,
    block_size: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<(), String> {
    if k_cache.is_empty() || v_cache.is_empty() {
        return Err("KV cache buffers must not be empty".to_string());
    }
    let k_buffer = DeviceBuffer::from_slice(k_cache)?;
    let v_buffer = DeviceBuffer::from_slice(v_cache)?;

    // SAFETY: the device caches are live and sized; the kernel copies one block
    // range and never accesses outside that range.
    let code = unsafe {
        ffi::launch_copy_block(
            k_buffer.as_mut_ptr::<f32>(),
            v_buffer.as_mut_ptr::<f32>(),
            dst_block as c_int,
            src_block as c_int,
            block_size as c_int,
            num_heads as c_int,
            head_dim as c_int,
        )
    };
    if code != 0 {
        return Err(format!("CUDA copy_block failed with code {code}"));
    }
    k_buffer.copy_out(k_cache)?;
    v_buffer.copy_out(v_cache)?;
    Ok(())
}

/// Correct CPU reference: scaled dot-product attention with softmax over the
/// sequence's paged keys, followed by a probability-weighted value sum.
#[allow(clippy::too_many_arguments)]
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
    run_paged_attention_cpu_ref_gqa(
        out,
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        max_blocks_per_seq,
        num_seqs,
        num_heads,
        num_heads,
        head_dim,
        block_size,
    );
}

/// GQA-aware CPU reference. Query head `h` reads KV head
/// `h / (num_heads / num_kv_heads)`, matching the CUDA kernel and candle's
/// consecutive `repeat_kv` layout.
#[allow(clippy::too_many_arguments)]
pub fn run_paged_attention_cpu_ref_gqa(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    block_tables: &[i32],
    seq_lens: &[i32],
    max_blocks_per_seq: usize,
    num_seqs: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
) {
    assert!(
        num_kv_heads > 0 && num_heads.is_multiple_of(num_kv_heads),
        "num_heads must be a positive multiple of num_kv_heads"
    );
    let n_rep = num_heads / num_kv_heads;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    for seq in 0..num_seqs {
        let seq_len = seq_lens[seq].max(0) as usize;
        if seq_len == 0 {
            continue;
        }
        for head in 0..num_heads {
            let kv_head = head / n_rep;
            let mut scores = vec![0f32; seq_len];
            let mut max_score = f32::NEG_INFINITY;
            for (token, score) in scores.iter_mut().enumerate() {
                let block = block_tables[seq * max_blocks_per_seq + token / block_size] as usize;
                let offset = token % block_size;
                let kv_offset = ((block * block_size + offset) * num_kv_heads + kv_head) * head_dim;
                let mut dot = 0.0f32;
                for dim in 0..head_dim {
                    let q_offset = (seq * num_heads + head) * head_dim + dim;
                    dot += q[q_offset] * k_cache[kv_offset + dim];
                }
                *score = dot * scale;
                max_score = max_score.max(*score);
            }

            let mut denominator = 0.0f32;
            for score in &mut scores {
                let value = (*score - max_score).exp();
                *score = value;
                denominator += value;
            }
            if denominator == 0.0 {
                denominator = 1.0;
            }
            for dim in 0..head_dim {
                let mut accumulator = 0.0f32;
                for (token, probability) in scores.iter().enumerate() {
                    let block =
                        block_tables[seq * max_blocks_per_seq + token / block_size] as usize;
                    let offset = token % block_size;
                    let kv_offset =
                        ((block * block_size + offset) * num_kv_heads + kv_head) * head_dim;
                    accumulator += (probability / denominator) * v_cache[kv_offset + dim];
                }
                out[(seq * num_heads + head) * head_dim + dim] = accumulator;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_f32(&mut self, low: f32, high: f32) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = ((self.0 >> 33) as u32 as f32) / (1u32 << 31) as f32;
            low + unit * (high - low)
        }
    }

    /// Independent naive attention over a dense token-major cache.
    fn naive_attention(
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        q_data: &[f32],
        k_data: &[f32],
        v_data: &[f32],
    ) -> Vec<f32> {
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut out = vec![0f32; num_heads * head_dim];
        for head in 0..num_heads {
            let mut scores = vec![0f64; seq_len];
            let mut max_score = f64::NEG_INFINITY;
            for token in 0..seq_len {
                let mut dot = 0f64;
                for dim in 0..head_dim {
                    dot += (q_data[head * head_dim + dim]
                        * k_data[(token * num_heads + head) * head_dim + dim])
                        as f64;
                }
                scores[token] = dot * scale;
                max_score = max_score.max(scores[token]);
            }
            let mut denominator = 0f64;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                denominator += *score;
            }
            for dim in 0..head_dim {
                let mut accumulator = 0f64;
                for token in 0..seq_len {
                    accumulator += scores[token] / denominator
                        * v_data[(token * num_heads + head) * head_dim + dim] as f64;
                }
                out[head * head_dim + dim] = accumulator as f32;
            }
        }
        out
    }

    #[test]
    fn reference_matches_naive_attention() {
        let mut rng = Lcg::new(1234);
        let head_dim = 64usize;
        let num_heads = 3usize;
        let seq_len = 137usize;
        let block_size = 16usize;
        let max_blocks = seq_len.div_ceil(block_size);
        let num_blocks = max_blocks + 3;

        for trial in 0..20 {
            let q: Vec<f32> = (0..num_heads * head_dim)
                .map(|_| rng.next_f32(-1.0, 1.0))
                .collect();
            let k: Vec<f32> = (0..num_blocks * block_size * num_heads * head_dim)
                .map(|_| rng.next_f32(-1.0, 1.0))
                .collect();
            let v: Vec<f32> = (0..num_blocks * block_size * num_heads * head_dim)
                .map(|_| rng.next_f32(-1.0, 1.0))
                .collect();
            let mut block_tables = vec![0i32; max_blocks];
            for (index, block) in block_tables.iter_mut().enumerate() {
                *block = ((index * 3 + trial) % num_blocks) as i32;
            }

            let mut actual = vec![0f32; num_heads * head_dim];
            run_paged_attention_cpu_ref(
                &mut actual,
                &q,
                &k,
                &v,
                &block_tables,
                &[seq_len as i32],
                max_blocks,
                1,
                num_heads,
                head_dim,
                block_size,
            );

            // Build a dense token-major cache in the same physical order.
            let mut dense_k = vec![0f32; seq_len * num_heads * head_dim];
            let mut dense_v = vec![0f32; seq_len * num_heads * head_dim];
            for token in 0..seq_len {
                let block = block_tables[token / block_size] as usize;
                let offset = token % block_size;
                let source = (block * block_size + offset) * num_heads * head_dim;
                let destination = token * num_heads * head_dim;
                dense_k[destination..destination + num_heads * head_dim]
                    .copy_from_slice(&k[source..source + num_heads * head_dim]);
                dense_v[destination..destination + num_heads * head_dim]
                    .copy_from_slice(&v[source..source + num_heads * head_dim]);
            }
            let expected = naive_attention(seq_len, num_heads, head_dim, &q, &dense_k, &dense_v);
            for (index, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-5,
                    "trial {trial} output {index}: ref={a} naive={b}"
                );
            }
        }
    }

    #[test]
    fn cpu_reference_single_token_returns_value() {
        let head_dim = 4;
        let num_heads = 2;
        let block_size = 2;
        let q = vec![1.0f32; num_heads * head_dim];
        let k = vec![0.3f32; block_size * num_heads * head_dim];
        let mut v = vec![0.0f32; block_size * num_heads * head_dim];
        for head in 0..num_heads {
            for dim in 0..head_dim {
                v[head * head_dim + dim] = (head * 10 + dim) as f32;
            }
        }
        let mut out = vec![0f32; num_heads * head_dim];
        run_paged_attention_cpu_ref(
            &mut out,
            &q,
            &k,
            &v,
            &[0],
            &[1],
            1,
            1,
            num_heads,
            head_dim,
            block_size,
        );
        for head in 0..num_heads {
            for dim in 0..head_dim {
                assert!((out[head * head_dim + dim] - (head * 10 + dim) as f32).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn cpu_reference_gqa_maps_sixteen_query_heads_to_eight_kv_heads() {
        let num_heads = 16usize;
        let num_kv_heads = 8usize;
        let head_dim = 2usize;
        let block_size = 2usize;
        let seq_len = 2usize;
        let max_blocks = 1usize;
        // Equal scores (q and k are zero) => uniform attention over the two
        // tokens. Each KV head has a distinct value so a wrong mapping is
        // visible in the output.
        let q = vec![0f32; num_heads * head_dim];
        let k = vec![0f32; block_size * num_kv_heads * head_dim];
        let mut v = vec![0f32; block_size * num_kv_heads * head_dim];
        for kv_head in 0..num_kv_heads {
            for token in 0..block_size {
                for dim in 0..head_dim {
                    v[(token * num_kv_heads + kv_head) * head_dim + dim] =
                        kv_head as f32 * 10.0 + dim as f32 + token as f32 * 0.5;
                }
            }
        }
        let mut out = vec![0f32; num_heads * head_dim];
        run_paged_attention_cpu_ref_gqa(
            &mut out,
            &q,
            &k,
            &v,
            &[0],
            &[seq_len as i32],
            max_blocks,
            1,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
        );
        let n_rep = num_heads / num_kv_heads;
        for head in 0..num_heads {
            let kv_head = head / n_rep;
            for dim in 0..head_dim {
                let expected = kv_head as f32 * 10.0 + dim as f32 + 0.25;
                let actual = out[head * head_dim + dim];
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "head {head} dim {dim}: {actual} != {expected}"
                );
            }
        }
    }

    // GPU-only tests. They require an NVIDIA GPU, a driver, nvcc, and
    // `cargo test --features cuda -- --ignored`.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires an NVIDIA GPU and a build with --features cuda"]
    fn cuda_attention_parity_with_cpu_reference() {
        let mut rng = Lcg::new(99);
        let head_dim = 64usize;
        let num_heads = 2usize;
        let seq_len = 137usize;
        let block_size = 16usize;
        let max_blocks = seq_len.div_ceil(block_size);
        let num_blocks = max_blocks + 2;
        let q: Vec<f32> = (0..num_heads * head_dim)
            .map(|_| rng.next_f32(-1.0, 1.0))
            .collect();
        let k: Vec<f32> = (0..num_blocks * block_size * num_heads * head_dim)
            .map(|_| rng.next_f32(-1.0, 1.0))
            .collect();
        let v: Vec<f32> = (0..num_blocks * block_size * num_heads * head_dim)
            .map(|_| rng.next_f32(-1.0, 1.0))
            .collect();
        let block_tables: Vec<i32> = (0..max_blocks).map(|index| index as i32).collect();
        let mut expected = vec![0f32; num_heads * head_dim];
        run_paged_attention_cpu_ref(
            &mut expected,
            &q,
            &k,
            &v,
            &block_tables,
            &[seq_len as i32],
            max_blocks,
            1,
            num_heads,
            head_dim,
            block_size,
        );
        let mut actual = vec![0f32; num_heads * head_dim];
        run_paged_attention(
            &mut actual,
            &q,
            &k,
            &v,
            &block_tables,
            &[seq_len as i32],
            max_blocks,
            1,
            num_heads,
            num_heads,
            head_dim,
            block_size,
        )
        .expect("CUDA attention");
        let max_error = actual
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_error < 1e-3, "CUDA attention max error {max_error}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires an NVIDIA GPU and a build with --features cuda"]
    fn cuda_f16_gqa_attention_parity_with_cpu_reference() {
        let mut rng = Lcg::new(2026);
        let head_dim = 64usize;
        let num_heads = 16usize;
        let num_kv_heads = 8usize;
        let seq_len = 137usize;
        let block_size = 16usize;
        let max_blocks = seq_len.div_ceil(block_size);
        let num_blocks = max_blocks + 2;
        let q: Vec<f32> = (0..num_heads * head_dim)
            .map(|_| rng.next_f32(-1.0, 1.0))
            .collect();
        let k: Vec<f32> = (0..num_blocks * block_size * num_kv_heads * head_dim)
            .map(|_| rng.next_f32(-1.0, 1.0))
            .collect();
        let v: Vec<f32> = (0..num_blocks * block_size * num_kv_heads * head_dim)
            .map(|_| rng.next_f32(-1.0, 1.0))
            .collect();
        let block_tables: Vec<i32> = (0..max_blocks).map(|index| index as i32).collect();
        let mut expected = vec![0f32; num_heads * head_dim];
        run_paged_attention_cpu_ref_gqa(
            &mut expected,
            &q,
            &k,
            &v,
            &block_tables,
            &[seq_len as i32],
            max_blocks,
            1,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
        );
        let mut actual = vec![0f32; num_heads * head_dim];
        run_paged_attention_f16(
            &mut actual,
            &q,
            &k,
            &v,
            &block_tables,
            &[seq_len as i32],
            max_blocks,
            1,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
        )
        .expect("CUDA f16 attention");
        let mut worst = 0f32;
        for (index, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            let relative = (a - b).abs() / b.abs().max(1e-3);
            if relative > worst {
                worst = relative;
            }
            assert!(
                relative < 2e-2,
                "f16 GQA output {index}: actual={a} expected={b} relative={relative}"
            );
        }
        assert!(worst < 2e-2, "worst relative error {worst}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires an NVIDIA GPU and a build with --features cuda"]
    fn cuda_slot_write_read_roundtrip() {
        let head_dim = 4usize;
        let num_heads = 2usize;
        let block_size = 2usize;
        let num_tokens = 2usize;
        let key = vec![0.25f32; num_tokens * num_heads * head_dim];
        let mut value = vec![0.0f32; num_tokens * num_heads * head_dim];
        for (index, element) in value.iter_mut().enumerate() {
            *element = index as f32;
        }
        let mut k_cache = vec![0f32; block_size * num_heads * head_dim];
        let mut v_cache = vec![0f32; block_size * num_heads * head_dim];
        let slots = [0i32, 1i32];
        run_reshape_and_cache(
            &key,
            &value,
            &mut k_cache,
            &mut v_cache,
            &slots,
            num_tokens,
            num_heads,
            head_dim,
        )
        .expect("reshape_and_cache");

        // A query on the first slot (one visible key) must return that value.
        let q = vec![1.0f32; num_heads * head_dim];
        let mut out = vec![0f32; num_heads * head_dim];
        run_paged_attention(
            &mut out,
            &q,
            &k_cache,
            &v_cache,
            &[0],
            &[1],
            1,
            1,
            num_heads,
            num_heads,
            head_dim,
            block_size,
        )
        .expect("CUDA attention");
        for (index, (actual, expected)) in out.iter().zip(value.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "slot roundtrip mismatch at {index}: {actual} vs {expected}"
            );
        }
    }
}
