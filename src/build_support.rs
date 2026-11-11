// src/build_support.rs
//! Cross-platform CUDA path and flag resolution shared by `build.rs` and unit
//! tests. The default build never calls into this module, so a machine without
//! CUDA can still compile and test the engine.

use std::path::{Path, PathBuf};

/// The `cuda` cargo feature is the only thing that may require nvcc.
pub fn cuda_feature_enabled(feature: Option<&str>) -> bool {
    feature.is_some()
}

/// MSVC targets link with `lib.exe` and use `cl.exe` as nvcc's host compiler.
pub fn is_msvc_target(target: &str) -> bool {
    target.contains("msvc")
}

/// Resolve the toolkit root. `CUDA_HOME` wins over `CUDA_PATH` so callers can
/// deliberately override an installer-provided path.
pub fn resolve_cuda_home(cuda_home: Option<&str>, cuda_path: Option<&str>) -> Option<PathBuf> {
    for candidate in [cuda_home, cuda_path].into_iter().flatten() {
        if !candidate.trim().is_empty() {
            return Some(PathBuf::from(candidate));
        }
    }
    None
}

/// Windows libraries live in `lib\x64`; Linux uses `lib64` (plus stubs).
pub fn cuda_lib_dir(cuda_home: &Path, target_os: &str) -> PathBuf {
    if target_os == "windows" {
        cuda_home.join("lib").join("x64")
    } else {
        cuda_home.join("lib64")
    }
}

/// Directories emitted as `cargo:rustc-link-search` for a CUDA build.
pub fn link_search_dirs(cuda_home: &Path, target_os: &str) -> Vec<PathBuf> {
    if target_os == "windows" {
        vec![cuda_lib_dir(cuda_home, target_os)]
    } else {
        let mut dirs = vec![
            cuda_lib_dir(cuda_home, target_os),
            cuda_home.join("lib64").join("stubs"),
        ];
        // WSL exposes the driver library here; native Linux does not have it.
        let wsl = PathBuf::from("/usr/lib/wsl/lib");
        if wsl.is_dir() {
            dirs.push(wsl);
        }
        dirs
    }
}

/// nvcc flags shared by the Windows and Linux toolkit layouts.
pub fn nvcc_flags(target_os: &str) -> Vec<&'static str> {
    let mut flags = vec![
        "-O3",
        "-std=c++17",
        "-cudart=shared",
        // Ampere SASS for this GPU plus PTX for forward JIT on newer GPUs.
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_86,code=compute_86",
    ];
    if target_os == "windows" {
        // CUDA toolkits frequently ship before the newest MSVC minor releases.
        // The flag only relaxes nvcc's version gate; the produced binaries still
        // target the installed cl.exe. The build script logs when it is used.
        flags.push("-allow-unsupported-compiler");
    }
    flags
}

/// Conventional toolkit locations when neither `CUDA_HOME` nor `CUDA_PATH` is
/// set. Newest first so a 13.4 install wins over an older 12.6 one. The per-user
/// tree is the documented non-admin layout; the Program Files entries are the
/// normal installer layout.
///
/// The ceiling is cudarc 0.19.10 (candle 0.9.x): its build script panics on any
/// toolkit newer than 13.4, so 13.4 is the newest usable major.minor.
pub fn default_cuda_candidates(target_os: &str, user_profile: Option<&Path>) -> Vec<PathBuf> {
    if target_os == "windows" {
        let mut candidates = Vec::new();
        if let Some(profile) = user_profile {
            for version in ["13.4", "13.2", "12.6", "12.4"] {
                candidates.push(profile.join(format!("cuda-{version}")));
            }
        }
        for version in ["v13.4", "v13.2", "v12.6", "v12.4"] {
            candidates.push(PathBuf::from(format!(
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{version}"
            )));
        }
        candidates
    } else {
        vec![
            PathBuf::from("/usr/local/cuda-13.4"),
            PathBuf::from("/usr/local/cuda"),
            PathBuf::from("/usr/local/cuda-12.6"),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_gating_only_enables_cuda_work_when_requested() {
        assert!(!cuda_feature_enabled(None));
        assert!(cuda_feature_enabled(Some("1")));
    }

    #[test]
    fn msvc_detection_is_target_driven() {
        assert!(is_msvc_target("x86_64-pc-windows-msvc"));
        assert!(!is_msvc_target("x86_64-unknown-linux-gnu"));
    }

    #[test]
    fn cuda_home_takes_precedence_and_empty_values_are_ignored() {
        assert_eq!(
            resolve_cuda_home(Some(r"C:\cuda-home"), Some(r"C:\cuda-path")),
            Some(PathBuf::from(r"C:\cuda-home"))
        );
        assert_eq!(
            resolve_cuda_home(Some("  "), Some(r"C:\cuda-path")),
            Some(PathBuf::from(r"C:\cuda-path"))
        );
        assert_eq!(resolve_cuda_home(None, None), None);
    }

    #[test]
    fn windows_uses_lib_x64_and_never_wsl_paths() {
        let root = Path::new(r"C:\CUDA\v12.6");
        assert_eq!(cuda_lib_dir(root, "windows"), root.join("lib").join("x64"));
        let dirs = link_search_dirs(root, "windows");
        assert_eq!(dirs, vec![root.join("lib").join("x64")]);
        assert!(
            dirs.iter()
                .all(|dir| !dir.to_string_lossy().contains("wsl"))
        );
    }

    #[test]
    fn linux_keeps_lib64_and_stubs() {
        let root = Path::new("/usr/local/cuda-12.6");
        assert_eq!(cuda_lib_dir(root, "linux"), root.join("lib64"));
        let dirs = link_search_dirs(root, "linux");
        assert!(dirs.contains(&root.join("lib64")));
        assert!(dirs.contains(&root.join("lib64").join("stubs")));
    }

    #[test]
    fn nvcc_emits_sm86_sass_and_ptx_on_both_platforms() {
        for target in ["windows", "linux"] {
            let flags = nvcc_flags(target);
            assert!(flags.contains(&"-gencode=arch=compute_86,code=sm_86"));
            assert!(flags.contains(&"-gencode=arch=compute_86,code=compute_86"));
            assert!(flags.contains(&"-cudart=shared"));
        }
        assert!(nvcc_flags("windows").contains(&"-allow-unsupported-compiler"));
        assert!(!nvcc_flags("linux").contains(&"-allow-unsupported-compiler"));
    }

    #[test]
    fn default_candidates_cover_user_and_installer_layouts() {
        let profile = Path::new(r"C:\Users\test");
        let windows = default_cuda_candidates("windows", Some(profile));
        assert!(windows.contains(&profile.join("cuda-13.4")));
        assert!(windows.contains(&profile.join("cuda-12.6")));
        assert!(
            windows
                .iter()
                .any(|path| path.to_string_lossy().contains("v13.4"))
        );
        // Newest candidate first: a 13.4 install must win over 12.6.
        assert!(
            windows
                .iter()
                .position(|path| path.to_string_lossy().contains("13.4"))
                .unwrap()
                < windows
                    .iter()
                    .position(|path| path.to_string_lossy().contains("12.6"))
                    .unwrap()
        );
        let linux = default_cuda_candidates("linux", None);
        assert!(linux.contains(&PathBuf::from("/usr/local/cuda")));
    }
}
