// src/lib.rs
//! NexusKV engine library.
//!
//! The default feature set is CPU-only and requires no model weights. Real
//! inference lives behind the `candle-model` feature and custom CUDA kernels
//! behind `cuda` (which implies `candle-model`).

pub mod build_support;
pub mod config;
pub mod engine;
pub mod kernels;
pub mod memory;
pub mod model;
pub mod server;
