// build.rs
//! Compiles the custom CUDA kernels only when the `cuda` feature is enabled.
//! The default build must work on a machine with no toolkit, no GPU, and no
//! nvcc; path resolution lives in `src/build_support.rs` so it is unit-tested
//! without a build script invocation.

use std::env;
use std::path::PathBuf;

#[path = "src/build_support.rs"]
mod build_support;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=csrc/paged_attention.cu");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=NVCC");

    if !build_support::cuda_feature_enabled(env::var("CARGO_FEATURE_CUDA").ok().as_deref()) {
        return;
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target = env::var("CARGO_CFG_TARGET_TRIPLE").unwrap_or_default();

    let cuda_home = resolve_cuda_home(&target_os).unwrap_or_else(|| {
        panic!(
            "the `cuda` feature requires the CUDA toolkit, but neither CUDA_HOME/CUDA_PATH \
             nor a conventional toolkit directory was found.\n\
             Fix: run `powershell -ExecutionPolicy Bypass -File scripts/check_env.ps1` \
             and follow its instructions (native Windows: install CUDA Toolkit 12.6, then \
             set CUDA_PATH to its root)."
        )
    });
    println!(
        "cargo:warning=using CUDA toolkit at {}",
        cuda_home.display()
    );

    let lib_dir = build_support::cuda_lib_dir(&cuda_home, &target_os);
    if !lib_dir.is_dir() {
        panic!(
            "CUDA toolkit at {} has no library directory {}.\n\
             Windows expects lib\\x64 (not lib64). Re-run scripts/check_env.ps1.",
            cuda_home.display(),
            lib_dir.display()
        );
    }

    for dir in build_support::link_search_dirs(&cuda_home, &target_os) {
        if dir.is_dir() {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    }
    println!("cargo:rustc-link-lib=dylib=cudart");

    // cc looks up nvcc through $NVCC or PATH. Point it at the resolved toolkit
    // so `cargo build --features cuda` works from a shell that only knows
    // CUDA_PATH, without requiring the caller to edit PATH.
    if env::var_os("NVCC").is_none() {
        let nvcc = cuda_home.join("bin").join(if target_os == "windows" {
            "nvcc.exe"
        } else {
            "nvcc"
        });
        if nvcc.is_file() {
            // SAFETY: build scripts run single-threaded before cc spawns its
            // compiler processes, and no other thread reads the environment.
            unsafe {
                env::set_var("NVCC", &nvcc);
            }
        }
    }

    let mut build = cc::Build::new();
    build.cuda(true).file("csrc/paged_attention.cu");
    for flag in build_support::nvcc_flags(&target_os) {
        build.flag(flag);
    }
    if build_support::is_msvc_target(&target) {
        println!(
            "cargo:warning=CUDA 12.6 predates the installed MSVC minor; \
             -allow-unsupported-compiler is enabled (host compiler is still cl.exe)"
        );
    }
    build.compile("paged_attention_kernel");
}

fn resolve_cuda_home(target_os: &str) -> Option<PathBuf> {
    let from_env = build_support::resolve_cuda_home(
        env::var("CUDA_HOME").ok().as_deref(),
        env::var("CUDA_PATH").ok().as_deref(),
    );
    if from_env.is_some() {
        return from_env;
    }
    let user_profile = env::var_os("USERPROFILE").map(PathBuf::from);
    build_support::default_cuda_candidates(target_os, user_profile.as_deref())
        .into_iter()
        .find(|candidate| candidate.is_dir())
}
