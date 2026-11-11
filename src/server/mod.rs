// src/server/mod.rs
pub mod routes;
pub mod sse_handler;

pub use routes::router;
pub use sse_handler::*;
