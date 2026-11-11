// src/memory/mod.rs
pub mod block_manager;

#[cfg(test)]
mod fuzz;
#[cfg(test)]
mod tests;

pub use block_manager::*;
