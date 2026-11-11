// src/model/qwen2.rs
use super::{Executor, ModelError, Tokenizer};
use crate::config::EngineConfig;
use crate::memory::{BlockSpaceManager, SeqId};
use candle_core::{D, DType, Device, Module, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder as NnVarBuilder, embedding};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone, Deserialize)]
struct HfConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default = "default_max_positions")]
    max_position_embeddings: usize,
    #[serde(default = "default_rms_eps")]
    rms_norm_eps: f64,
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    hidden_act: Option<String>,
    #[serde(default)]
    head_dim: Option<usize>,
}

fn default_max_positions() -> usize {
    32768
}

fn default_rms_eps() -> f64 {
    1e-6
}

#[derive(Debug, Clone)]
struct Qwen2Config {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    tie_word_embeddings: bool,
}

impl Qwen2Config {
    fn from_hf(config: HfConfig) -> anyhow::Result<Self> {
        let num_kv_heads = config
            .num_key_value_heads
            .unwrap_or(config.num_attention_heads);
        if num_kv_heads == 0
            || config.num_attention_heads == 0
            || !config.num_attention_heads.is_multiple_of(num_kv_heads)
        {
            anyhow::bail!(
                "num_attention_heads {} must be a positive multiple of num_key_value_heads {}",
                config.num_attention_heads,
                num_kv_heads
            );
        }
        let head_dim = config
            .head_dim
            .unwrap_or(config.hidden_size / config.num_attention_heads);
        if head_dim == 0
            || !config
                .hidden_size
                .is_multiple_of(config.num_attention_heads)
        {
            anyhow::bail!("invalid hidden_size/num_attention_heads configuration");
        }
        if let Some(activation) = config.hidden_act.as_deref()
            && !activation.eq_ignore_ascii_case("silu")
            && !activation.eq_ignore_ascii_case("swish")
        {
            tracing::warn!("Qwen2 hidden_act {activation} treated as SwiGLU/SiLU");
        }
        Ok(Self {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_layers: config.num_hidden_layers,
            num_heads: config.num_attention_heads,
            num_kv_heads,
            head_dim,
            max_position_embeddings: config.max_position_embeddings,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta.unwrap_or(1_000_000.0),
            tie_word_embeddings: config.tie_word_embeddings.unwrap_or(false),
        })
    }

    fn attention_width(&self) -> usize {
        self.num_heads * self.head_dim
    }
}

fn read_hf_config(directory: &Path) -> anyhow::Result<HfConfig> {
    let path = directory.join("config.json");
    let raw: serde_json::Value = serde_json::from_slice(
        &fs::read(&path).map_err(|err| anyhow::anyhow!("cannot read {}: {err}", path.display()))?,
    )?;
    let value = if raw.get("hidden_size").is_some() {
        raw
    } else if let Some(text_config) = raw.get("text_config") {
        text_config.clone()
    } else {
        raw
    };
    Ok(serde_json::from_value(value)?)
}

fn select_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        match Device::cuda_if_available(0) {
            Ok(device) if device.is_cuda() => {
                tracing::info!("Candle CUDA device selected");
                device
            }
            _ => {
                tracing::warn!("CUDA feature enabled but no CUDA device is available; using CPU");
                Device::Cpu
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        Device::Cpu
    }
}

fn load_linear(vb: NnVarBuilder<'_>, in_dim: usize, out_dim: usize) -> anyhow::Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    let bias = vb.get(out_dim, "bias").ok();
    Ok(Linear::new(weight, bias))
}

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn load(vb: NnVarBuilder<'_>, hidden_size: usize, eps: f64) -> anyhow::Result<Self> {
        Ok(Self {
            weight: vb.get(hidden_size, "weight")?,
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let denominator = variance.affine(1.0, self.eps)?.sqrt()?.recip()?;
        let normalized = x.broadcast_mul(&denominator)?;
        normalized
            .to_dtype(self.weight.dtype())?
            .broadcast_mul(&self.weight)
    }
}

struct Qwen2Layer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen2Layer {
    fn load(vb: NnVarBuilder<'_>, config: &Qwen2Config) -> anyhow::Result<Self> {
        let attention = vb.pp("self_attn");
        let hidden = config.hidden_size;
        let attention_width = config.attention_width();
        let kv_width = config.num_kv_heads * config.head_dim;
        let mlp = vb.pp("mlp");
        Ok(Self {
            q_proj: load_linear(attention.pp("q_proj"), hidden, attention_width)?,
            k_proj: load_linear(attention.pp("k_proj"), hidden, kv_width)?,
            v_proj: load_linear(attention.pp("v_proj"), hidden, kv_width)?,
            o_proj: load_linear(attention.pp("o_proj"), attention_width, hidden)?,
            input_layernorm: RmsNorm::load(vb.pp("input_layernorm"), hidden, config.rms_norm_eps)?,
            post_attention_layernorm: RmsNorm::load(
                vb.pp("post_attention_layernorm"),
                hidden,
                config.rms_norm_eps,
            )?,
            gate_proj: load_linear(mlp.pp("gate_proj"), hidden, config.intermediate_size)?,
            up_proj: load_linear(mlp.pp("up_proj"), hidden, config.intermediate_size)?,
            down_proj: load_linear(mlp.pp("down_proj"), config.intermediate_size, hidden)?,
        })
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> CandleResult<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (batch, kv_heads, seq_len, head_dim) = x.dims4()?;
    let expanded = x
        .unsqueeze(2)?
        .expand((batch, kv_heads, n_rep, seq_len, head_dim))?;
    expanded.reshape((batch, kv_heads * n_rep, seq_len, head_dim))
}

fn apply_rope(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    start: usize,
    seq_len: usize,
) -> CandleResult<Tensor> {
    let head_dim = x.dim(D::Minus1)?;
    let half = head_dim / 2;
    let cos = cos.narrow(0, start, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.narrow(0, start, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
    let first = x.narrow(D::Minus1, 0, half)?;
    let second = x.narrow(D::Minus1, half, half)?;
    let rotated = Tensor::cat(&[&second.neg()?, &first], D::Minus1)?;
    x.broadcast_mul(&cos)?.add(&rotated.broadcast_mul(&sin)?)
}

fn causal_mask(start: usize, q_len: usize, k_len: usize, device: &Device) -> CandleResult<Tensor> {
    let mut mask = vec![0f32; q_len * k_len];
    for query in 0..q_len {
        let limit = start + query;
        for key in 0..k_len {
            if key > limit {
                mask[query * k_len + key] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(mask, (1, 1, q_len, k_len), device)
}

fn gather_attention(
    q: &Tensor,
    k_pool: &Tensor,
    v_pool: &Tensor,
    slots: &Tensor,
    mask: &Tensor,
    n_rep: usize,
) -> CandleResult<Tensor> {
    let k = k_pool
        .index_select(slots, 0)?
        .transpose(0, 1)?
        .unsqueeze(0)?;
    let v = v_pool
        .index_select(slots, 0)?
        .transpose(0, 1)?
        .unsqueeze(0)?;
    let k = repeat_kv(k, n_rep)?;
    let v = repeat_kv(v, n_rep)?;
    let head_dim = q.dim(D::Minus1)?;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let scores = q.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?;
    let scores = scores.broadcast_add(mask)?;
    let probabilities = candle_nn::ops::softmax_last_dim(&scores)?;
    probabilities.matmul(&v)
}

fn write_kv_block(
    k_pool: &Tensor,
    v_pool: &Tensor,
    k: &Tensor,
    v: &Tensor,
    slots: &[i64],
) -> CandleResult<()> {
    // [1, kv_heads, len, head_dim] -> [len, kv_heads, head_dim]
    let k = k.squeeze(0)?.transpose(0, 1)?.contiguous()?;
    let v = v.squeeze(0)?.transpose(0, 1)?.contiguous()?;

    let mut run_start = 0usize;
    while run_start < slots.len() {
        let mut run_end = run_start + 1;
        while run_end < slots.len() && slots[run_end] == slots[run_end - 1] + 1 {
            run_end += 1;
        }
        let len = run_end - run_start;
        let k_run = k.narrow(0, run_start, len)?.contiguous()?;
        let v_run = v.narrow(0, run_start, len)?.contiguous()?;
        k_pool.slice_set(&k_run, 0, slots[run_start] as usize)?;
        v_pool.slice_set(&v_run, 0, slots[run_start] as usize)?;
        run_start = run_end;
    }
    Ok(())
}

pub struct Qwen2Model {
    config: Qwen2Config,
    device: Device,
    embed_tokens: Embedding,
    layers: Vec<Qwen2Layer>,
    norm: RmsNorm,
    lm_head: Option<Linear>,
    tied_weight: Tensor,
    cos: Tensor,
    sin: Tensor,
    k_pools: Vec<Tensor>,
    v_pools: Vec<Tensor>,
    block_size: usize,
}

impl Qwen2Model {
    fn load(
        config: &Qwen2Config,
        vb: &NnVarBuilder<'_>,
        device: Device,
        total_slots: usize,
        block_size: usize,
        max_rope_positions: usize,
    ) -> anyhow::Result<Self> {
        let vocab = config.vocab_size;
        let hidden = config.hidden_size;

        // HF checkpoints usually prefix everything with `model.`. Fall back to
        // a root layout if the prefixed embedding tensor cannot be loaded.
        let prefixed = vb.clone().pp("model");
        let (model_vb, embed_tokens) =
            match embedding(vocab, hidden, prefixed.clone().pp("embed_tokens")) {
                Ok(embedding) => (prefixed, embedding),
                Err(_) => (
                    vb.clone(),
                    embedding(vocab, hidden, vb.clone().pp("embed_tokens"))?,
                ),
            };

        let mut layers = Vec::with_capacity(config.num_layers);
        let layer_vb = model_vb.pp("layers");
        for index in 0..config.num_layers {
            layers.push(Qwen2Layer::load(layer_vb.pp(index.to_string()), config)?);
        }
        let norm = RmsNorm::load(model_vb.pp("norm"), hidden, config.rms_norm_eps)?;

        let tied_weight = embed_tokens.embeddings().clone();
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            match vb.get((vocab, hidden), "lm_head.weight") {
                Ok(weight) => Some(Linear::new(weight, None)),
                Err(_) => {
                    tracing::warn!("lm_head.weight missing; using tied embeddings");
                    None
                }
            }
        };

        let (cos, sin) = rope_tables(
            config.head_dim,
            config.rope_theta,
            max_rope_positions.max(1),
            &device,
        )?;

        let kv_shape = (total_slots, config.num_kv_heads, config.head_dim);
        let mut k_pools = Vec::with_capacity(config.num_layers);
        let mut v_pools = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            k_pools.push(Tensor::zeros(kv_shape, DType::F32, &device)?);
            v_pools.push(Tensor::zeros(kv_shape, DType::F32, &device)?);
        }

        Ok(Self {
            config: config.clone(),
            device,
            embed_tokens,
            layers,
            norm,
            lm_head,
            tied_weight,
            cos,
            sin,
            k_pools,
            v_pools,
            block_size,
        })
    }

    fn forward(
        &mut self,
        block_manager: &mut BlockSpaceManager,
        seq: SeqId,
        tokens: &[u32],
        start: usize,
        all_logits: bool,
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::Invalid(
                "forward requires at least one token".to_string(),
            ));
        }
        for &token in tokens {
            if token as usize >= self.config.vocab_size {
                return Err(ModelError::Invalid(format!(
                    "token id {token} exceeds model vocab {}",
                    self.config.vocab_size
                )));
            }
        }
        let seq_len = tokens.len();
        let total_len = start + seq_len;
        let slots = block_manager
            .slot_mapping(seq, start, seq_len)
            .map_err(|err| ModelError::Backend(err.to_string()))?;
        let all_slots = block_manager
            .slot_mapping(seq, 0, total_len)
            .map_err(|err| ModelError::Backend(err.to_string()))?;
        let slot_indices: Vec<u32> = all_slots.iter().map(|&slot| slot as u32).collect();

        let device = &self.device;
        let input_ids =
            Tensor::from_vec(tokens.to_vec(), (1, seq_len), device).map_err(candle_error)?;
        let mut hidden = self
            .embed_tokens
            .forward(&input_ids)
            .map_err(candle_error)?;
        let mask = causal_mask(start, seq_len, total_len, device).map_err(candle_error)?;
        let slot_tensor =
            Tensor::from_vec(slot_indices, (total_len,), device).map_err(candle_error)?;
        let n_rep = self.config.num_heads / self.config.num_kv_heads;

        {
            let Qwen2Model {
                layers,
                k_pools,
                v_pools,
                cos,
                sin,
                config,
                ..
            } = self;
            for (layer_index, layer) in layers.iter().enumerate() {
                let residual = &hidden;
                let normalized = layer
                    .input_layernorm
                    .forward(&hidden)
                    .map_err(candle_error)?;
                let q = layer.q_proj.forward(&normalized).map_err(candle_error)?;
                let k = layer.k_proj.forward(&normalized).map_err(candle_error)?;
                let v = layer.v_proj.forward(&normalized).map_err(candle_error)?;

                let q = q
                    .reshape((1, seq_len, config.num_heads, config.head_dim))
                    .and_then(|tensor| tensor.transpose(1, 2))
                    .map_err(candle_error)?;
                let k = k
                    .reshape((1, seq_len, config.num_kv_heads, config.head_dim))
                    .and_then(|tensor| tensor.transpose(1, 2))
                    .map_err(candle_error)?;
                let v = v
                    .reshape((1, seq_len, config.num_kv_heads, config.head_dim))
                    .and_then(|tensor| tensor.transpose(1, 2))
                    .map_err(candle_error)?;

                let q = apply_rope(&q, cos, sin, start, seq_len).map_err(candle_error)?;
                let k = apply_rope(&k, cos, sin, start, seq_len).map_err(candle_error)?;

                write_kv_block(&k_pools[layer_index], &v_pools[layer_index], &k, &v, &slots)
                    .map_err(candle_error)?;

                let attention = gather_attention(
                    &q,
                    &k_pools[layer_index],
                    &v_pools[layer_index],
                    &slot_tensor,
                    &mask,
                    n_rep,
                )
                .map_err(candle_error)?;
                let attention = attention
                    .transpose(1, 2)
                    .and_then(|tensor| tensor.reshape((1, seq_len, config.attention_width())))
                    .map_err(candle_error)?;
                let attention = layer.o_proj.forward(&attention).map_err(candle_error)?;
                hidden = residual.add(&attention).map_err(candle_error)?;

                let residual = &hidden;
                let normalized = layer
                    .post_attention_layernorm
                    .forward(&hidden)
                    .map_err(candle_error)?;
                let gate = layer.gate_proj.forward(&normalized).map_err(candle_error)?;
                let up = layer.up_proj.forward(&normalized).map_err(candle_error)?;
                let activated = candle_nn::ops::silu(&gate).map_err(candle_error)?;
                let mlp = layer
                    .down_proj
                    .forward(&activated.mul(&up).map_err(candle_error)?)
                    .map_err(candle_error)?;
                hidden = residual.add(&mlp).map_err(candle_error)?;
            }
        }

        let hidden = if all_logits {
            hidden
        } else {
            hidden.narrow(1, seq_len - 1, 1).map_err(candle_error)?
        };
        let hidden = self.norm.forward(&hidden).map_err(candle_error)?;
        let logits = match &self.lm_head {
            Some(head) => head.forward(&hidden).map_err(candle_error)?,
            None => hidden
                .matmul(
                    &self
                        .tied_weight
                        .transpose(0, 1)
                        .and_then(|weight| weight.unsqueeze(0))
                        .map_err(candle_error)?,
                )
                .map_err(candle_error)?,
        };
        let logits = if all_logits {
            logits.squeeze(0).map_err(candle_error)?
        } else {
            logits
                .squeeze(1)
                .and_then(|tensor| tensor.squeeze(0))
                .map_err(candle_error)?
        };
        let flat = logits
            .flatten_all()
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .map_err(candle_error)?;
        let vocabulary = self.config.vocab_size.max(1);
        Ok(flat
            .chunks(vocabulary)
            .map(|chunk| chunk.to_vec())
            .collect())
    }

    fn copy_block(&mut self, src: usize, dst: usize) -> Result<(), ModelError> {
        if src == dst {
            return Ok(());
        }
        let block_size = self.block_size;
        let start = src * block_size;
        let destination = dst * block_size;
        let total_slots = self
            .k_pools
            .first()
            .map(|pool| pool.dim(0))
            .transpose()
            .map_err(candle_error)?
            .unwrap_or(0);
        if start + block_size > total_slots || destination + block_size > total_slots {
            return Err(ModelError::Invalid(format!(
                "block copy out of range: src={src} dst={dst} slots={total_slots}"
            )));
        }
        for (k_pool, v_pool) in self.k_pools.iter().zip(self.v_pools.iter()) {
            let k = k_pool
                .narrow(0, start, block_size)
                .and_then(|tensor| tensor.contiguous())
                .map_err(candle_error)?;
            let v = v_pool
                .narrow(0, start, block_size)
                .and_then(|tensor| tensor.contiguous())
                .map_err(candle_error)?;
            k_pool.slice_set(&k, 0, destination).map_err(candle_error)?;
            v_pool.slice_set(&v, 0, destination).map_err(candle_error)?;
        }
        Ok(())
    }
}

fn rope_tables(
    head_dim: usize,
    theta: f64,
    positions: usize,
    device: &Device,
) -> anyhow::Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let mut cos = vec![0f32; positions * head_dim];
    let mut sin = vec![0f32; positions * head_dim];
    for position in 0..positions {
        for index in 0..half {
            let frequency = 1.0 / theta.powf((2 * index) as f64 / head_dim as f64);
            let angle = position as f64 * frequency;
            let (sine, cosine) = angle.sin_cos();
            cos[position * head_dim + index] = cosine as f32;
            cos[position * head_dim + half + index] = cosine as f32;
            sin[position * head_dim + index] = sine as f32;
            sin[position * head_dim + half + index] = sine as f32;
        }
    }
    Ok((
        Tensor::from_vec(cos, (positions, head_dim), device)?,
        Tensor::from_vec(sin, (positions, head_dim), device)?,
    ))
}

fn candle_error(error: candle_core::Error) -> ModelError {
    ModelError::Backend(error.to_string())
}

pub struct QwenTokenizer {
    inner: tokenizers::Tokenizer,
    eos: Vec<u32>,
}

impl QwenTokenizer {
    fn load(directory: &Path) -> anyhow::Result<Self> {
        let path = directory.join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&path)
            .map_err(|err| anyhow::anyhow!("cannot load {}: {err}", path.display()))?;
        let mut eos = Vec::new();
        for token in ["<|im_end|>", "<|endoftext|>"] {
            if let Some(id) = inner.token_to_id(token)
                && !eos.contains(&id)
            {
                eos.push(id);
            }
        }
        Ok(Self { inner, eos })
    }
}

impl Tokenizer for QwenTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner
            .encode(text, true)
            .map(|encoding| encoding.get_ids().to_vec())
            .unwrap_or_default()
    }

    fn decode(&self, ids: &[u32]) -> String {
        self.inner
            .decode(ids, true)
            .unwrap_or_else(|_| String::new())
    }

    fn encode_chat(&self, messages: &[(String, String)]) -> Vec<u32> {
        let mut text = String::new();
        for (role, content) in messages {
            text.push_str("<|im_start|>");
            text.push_str(role);
            text.push('\n');
            text.push_str(content);
            text.push_str("<|im_end|>\n");
        }
        text.push_str("<|im_start|>assistant\n");
        self.encode(&text)
    }

    fn eos_token_ids(&self) -> &[u32] {
        &self.eos
    }
}

pub struct Qwen2Executor {
    model: Qwen2Model,
}

impl Executor for Qwen2Executor {
    fn forward(
        &mut self,
        block_manager: &mut BlockSpaceManager,
        seq: SeqId,
        tokens: &[u32],
        start: usize,
        all_logits: bool,
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        self.model
            .forward(block_manager, seq, tokens, start, all_logits)
    }

    fn copy_block(&mut self, src: usize, dst: usize) -> Result<(), ModelError> {
        self.model.copy_block(src, dst)
    }

    fn vocab_size(&self) -> usize {
        self.model.config.vocab_size
    }

    fn is_mock(&self) -> bool {
        false
    }

    fn name(&self) -> &'static str {
        "qwen2"
    }
}

/// Load a HuggingFace Qwen2 directory (`config.json`, `tokenizer.json`, and at
/// least one `*.safetensors` shard) into an executor plus its shared tokenizer.
pub fn load_model_and_tokenizer(
    directory: &Path,
    engine_config: &EngineConfig,
) -> anyhow::Result<(Qwen2Executor, Arc<dyn Tokenizer>)> {
    let hf = read_hf_config(directory)?;
    let config = Qwen2Config::from_hf(hf)?;
    if let Ok(raw) = fs::read(directory.join("config.json"))
        && let Ok(value) = serde_json::from_slice::<serde_json::Value>(&raw)
        && value.get("rope_scaling").is_some()
    {
        tracing::warn!("rope_scaling in config.json is not implemented; using default RoPE");
    }

    let device = select_device();
    let mut shards: Vec<PathBuf> = fs::read_dir(directory)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("safetensors"))
        .collect();
    shards.sort();
    if shards.is_empty() {
        anyhow::bail!("no *.safetensors shards found in {}", directory.display());
    }

    // SAFETY: the model directory is supplied by the operator and is expected to
    // contain immutable checkpoint files for the lifetime of the process.
    let vb = unsafe { NnVarBuilder::from_mmaped_safetensors(&shards, DType::F32, &device)? };

    let total_slots = engine_config.total_physical_blocks * engine_config.block_size;
    if total_slots == 0 || engine_config.block_size == 0 {
        anyhow::bail!("paged KV pool must have at least one slot");
    }
    let rope_positions = config
        .max_position_embeddings
        .min(engine_config.max_seq_token_length().max(1));
    let model = Qwen2Model::load(
        &config,
        &vb,
        device,
        total_slots,
        engine_config.block_size,
        rope_positions,
    )?;
    let concrete = Arc::new(QwenTokenizer::load(directory)?);
    let shared: Arc<dyn Tokenizer> = Arc::clone(&concrete) as Arc<dyn Tokenizer>;
    Ok((Qwen2Executor { model }, shared))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    fn tiny_config() -> Qwen2Config {
        Qwen2Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            tie_word_embeddings: true,
        }
    }

    #[test]
    fn gqa_expansion_matches_explicit_repeat() {
        let device = Device::Cpu;
        let values: Vec<f32> = (0..24).map(|value| value as f32).collect();
        let x = Tensor::from_vec(values.clone(), (1, 2, 3, 4), &device).unwrap();
        let repeated = repeat_kv(x, 3).unwrap();
        assert_eq!(repeated.dims(), &[1usize, 6, 3, 4]);
        let actual = repeated.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let mut expected = vec![0f32; 6 * 3 * 4];
        let kv_heads = 2;
        let seq = 3;
        let head_dim = 4;
        for kv_head in 0..kv_heads {
            for repetition in 0..3 {
                for token in 0..seq {
                    for dim in 0..head_dim {
                        let source = kv_head * seq * head_dim + token * head_dim + dim;
                        let destination =
                            ((kv_head * 3 + repetition) * seq + token) * head_dim + dim;
                        expected[destination] = values[source];
                    }
                }
            }
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn rope_chunking_is_position_equivalent() {
        let device = Device::Cpu;
        let (cos, sin) = rope_tables(8, 10_000.0, 16, &device).unwrap();
        let values: Vec<f32> = (0..(2 * 8 * 8))
            .map(|value| (value as f32 * 0.013).sin())
            .collect();
        let x = Tensor::from_vec(values, (1, 2, 8, 8), &device).unwrap();

        let full = apply_rope(&x, &cos, &sin, 0, 8).unwrap();
        let first = apply_rope(&x.narrow(2, 0, 4).unwrap(), &cos, &sin, 0, 4).unwrap();
        let second = apply_rope(&x.narrow(2, 4, 4).unwrap(), &cos, &sin, 4, 4).unwrap();
        let chunked = Tensor::cat(&[&first, &second], 2).unwrap();
        let difference = full
            .sub(&chunked)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(difference < 1e-6, "rope difference {difference}");
    }

    #[test]
    fn chunked_prefill_matches_full_prefill_logits() {
        let device = Device::Cpu;
        let config = tiny_config();
        let var_map = VarMap::new();
        let vb = NnVarBuilder::from_varmap(&var_map, DType::F32, &device);
        let mut model = Qwen2Model::load(&config, &vb, device, 64, 4, 64).unwrap();

        let prompt: Vec<u32> = vec![1, 5, 2, 9, 4, 7, 3, 8];
        let mut full_mgr = BlockSpaceManager::new(64, 4, 16);
        full_mgr
            .match_or_allocate_prefix(1, &prompt)
            .expect("allocate full prompt");
        let full = model
            .forward(&mut full_mgr, 1, &prompt, 0, true)
            .expect("full prefill");
        full_mgr.assert_invariants();

        let mut chunk_mgr = BlockSpaceManager::new(64, 4, 16);
        chunk_mgr
            .match_or_allocate_prefix(2, &prompt)
            .expect("allocate chunked prompt");
        model
            .forward(&mut chunk_mgr, 2, &prompt[..4], 0, false)
            .expect("first chunk");
        let last = model
            .forward(&mut chunk_mgr, 2, &prompt[4..], 4, true)
            .expect("second chunk");
        chunk_mgr.assert_invariants();

        let full_last = full.last().expect("full logits");
        let chunk_last = last.last().expect("chunk logits");
        assert_eq!(full_last.len(), chunk_last.len());
        for (index, (a, b)) in full_last.iter().zip(chunk_last.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "logit {index} differs: full={a} chunked={b}"
            );
        }
    }

    #[test]
    fn candle_gather_attention_matches_cpu_reference() {
        let device = Device::Cpu;
        let num_heads = 2;
        let head_dim = 8;
        let block_size = 4;
        let seq_len = 16;
        let num_blocks = 4;

        let mut seed = 7u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as u32 as f32 / (1u32 << 31) as f32) - 0.5
        };

        let q_data: Vec<f32> = (0..num_heads * head_dim).map(|_| next()).collect();
        let k_data: Vec<f32> = (0..num_blocks * block_size * num_heads * head_dim)
            .map(|_| next())
            .collect();
        let v_data: Vec<f32> = (0..num_blocks * block_size * num_heads * head_dim)
            .map(|_| next())
            .collect();

        let mut cpu_out = vec![0f32; num_heads * head_dim];
        crate::kernels::run_paged_attention_cpu_ref(
            &mut cpu_out,
            &q_data,
            &k_data,
            &v_data,
            &[0, 1, 2, 3],
            &[seq_len as i32],
            num_blocks,
            1,
            num_heads,
            head_dim,
            block_size,
        );

        // Convert [block, token, head, dim] to the flat pool [slot, head, dim].
        let pool_from_cache = |cache: &[f32]| -> Vec<f32> {
            let mut pool = vec![0f32; num_blocks * block_size * num_heads * head_dim];
            for block in 0..num_blocks {
                for token in 0..block_size {
                    let slot = block * block_size + token;
                    for head in 0..num_heads {
                        for dim in 0..head_dim {
                            let source =
                                ((block * block_size + token) * num_heads + head) * head_dim + dim;
                            let destination = (slot * num_heads + head) * head_dim + dim;
                            pool[destination] = cache[source];
                        }
                    }
                }
            }
            pool
        };

        let k_pool = Tensor::from_vec(
            pool_from_cache(&k_data),
            (num_blocks * block_size, num_heads, head_dim),
            &device,
        )
        .unwrap();
        let v_pool = Tensor::from_vec(
            pool_from_cache(&v_data),
            (num_blocks * block_size, num_heads, head_dim),
            &device,
        )
        .unwrap();
        let q = Tensor::from_vec(q_data, (1, num_heads, 1, head_dim), &device).unwrap();
        let slots: Vec<u32> = (0..seq_len as u32).collect();
        let slots = Tensor::from_vec(slots, (seq_len,), &device).unwrap();
        let mask = Tensor::zeros((1, 1, 1, seq_len), DType::F32, &device).unwrap();
        let actual = gather_attention(&q, &k_pool, &v_pool, &slots, &mask, 1)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .squeeze(1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let max_error = actual
            .iter()
            .zip(cpu_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_error < 1e-3, "attention max error {max_error}");
    }

    #[test]
    fn tokenizer_roundtrip_and_chat_template() {
        let directory =
            std::env::temp_dir().join(format!("nexuskv-tokenizer-test-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let fixture = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 5, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 6, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 7, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "user": 3, "assistant": 4},
                "unk_token": "[UNK]"
            }
        });
        fs::write(directory.join("tokenizer.json"), fixture.to_string()).unwrap();

        let tokenizer = QwenTokenizer::load(&directory).expect("load tokenizer fixture");
        let encoded = tokenizer.encode("hello world");
        assert!(!encoded.is_empty());
        assert_eq!(tokenizer.decode(&encoded), "hello world");

        let chat = tokenizer.encode_chat(&[("user".to_string(), "hello world".to_string())]);
        let rendered = tokenizer.inner.decode(&chat, false).unwrap();
        let compact = rendered.replace(' ', "");
        assert!(compact.contains("<|im_start|>user"), "rendered: {rendered}");
        assert!(rendered.contains("hello world"), "rendered: {rendered}");
        assert!(rendered.contains("<|im_end|>"), "rendered: {rendered}");
        assert!(
            compact.contains("<|im_start|>assistant"),
            "rendered: {rendered}"
        );
        assert_eq!(tokenizer.eos_token_ids(), &[6, 7]);

        let _ = fs::remove_dir_all(&directory);
    }
}
