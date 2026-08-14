// build.rs
use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=csrc/paged_attention.cu");

    // Resolve CUDA directory path
    let cuda_home = env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .unwrap_or_else(|_| {
            if Path::new("/usr/local/cuda-13.1").exists() {
                "/usr/local/cuda-13.1".to_string()
            } else {
                "/usr/local/cuda".to_string()
            }
        });

    // Link search paths for CUDA Runtime and WSL2 Driver
    println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
    println!("cargo:rustc-link-search=native={}/lib64/stubs", cuda_home);
    println!("cargo:rustc-link-search=native=/usr/lib/wsl/lib");
    println!("cargo:rustc-link-lib=dylib=cudart");

    // Compile custom CUDA kernel
    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-O3")
        .flag("-std=c++17")
        .flag("-gencode=arch=compute_86,code=sm_86") // RTX 3050 Ampere Arch
        .file("csrc/paged_attention.cu")
        .compile("paged_attention_kernel");
}