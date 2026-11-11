// src/memory/tier.rs
use super::{BlockId, BlockIo, BlockIoStats};
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Byte-level bridge between the paged KV pool and the tier store.
///
/// The v1 contract is an f32 byte container: implementations serialize one
/// block's K/V bytes. A production GPU path can replace this with device-to-host
/// copies without changing the block manager.
pub trait GpuBridge: Send {
    fn block_bytes(&self) -> usize;
    fn read_block(&mut self, block: BlockId) -> Result<Vec<u8>, String>;
    fn write_block(&mut self, block: BlockId, bytes: &[u8]) -> Result<(), String>;
}

/// In-memory bridge used by mock mode and tests. It stores raw block bytes and
/// has no model semantics; the paging machinery around it is real.
#[derive(Clone)]
pub struct MemoryBridge {
    inner: Arc<Mutex<HashMap<BlockId, Vec<u8>>>>,
    block_bytes: usize,
}

impl MemoryBridge {
    pub fn new(block_bytes: usize) -> Self {
        assert!(block_bytes > 0, "block_bytes must be positive");
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            block_bytes,
        }
    }

    pub fn set_block(&self, block: BlockId, bytes: impl Into<Vec<u8>>) {
        let bytes = bytes.into();
        assert_eq!(
            bytes.len(),
            self.block_bytes,
            "memory bridge block content has the wrong size"
        );
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(block, bytes);
    }

    pub fn block(&self, block: BlockId) -> Option<Vec<u8>> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&block)
            .cloned()
    }

    pub fn copy_block(&self, src: BlockId, dst: BlockId) {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(bytes) = inner.get(&src).cloned() {
            inner.insert(dst, bytes);
        }
    }
}

impl GpuBridge for MemoryBridge {
    fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    fn read_block(&mut self, block: BlockId) -> Result<Vec<u8>, String> {
        Ok(self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&block)
            .cloned()
            .unwrap_or_else(|| vec![0; self.block_bytes]))
    }

    fn write_block(&mut self, block: BlockId, bytes: &[u8]) -> Result<(), String> {
        if bytes.len() != self.block_bytes {
            return Err(format!(
                "memory bridge expected {} bytes, got {}",
                self.block_bytes,
                bytes.len()
            ));
        }
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(block, bytes.to_vec());
        Ok(())
    }
}

/// Shared fault-injection handle for deterministic IO failure tests.
#[derive(Clone, Default)]
pub struct TierFaults {
    flags: Arc<Mutex<(bool, bool)>>,
}

impl TierFaults {
    pub fn set_fail_save(&self, fail: bool) {
        self.flags
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .0 = fail;
    }

    pub fn set_fail_load(&self, fail: bool) {
        self.flags
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .1 = fail;
    }

    fn fail_save(&self) -> bool {
        self.flags
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .0
    }

    fn fail_load(&self) -> bool {
        self.flags
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .1
    }
}

const MAGIC: &[u8; 4] = b"NKV1";
const HEADER: usize = 4 + 8 + 8 + 8;

fn checksum(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

fn encode(hash: u64, data: &[u8]) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(HEADER + data.len());
    encoded.extend_from_slice(MAGIC);
    encoded.extend_from_slice(&hash.to_le_bytes());
    encoded.extend_from_slice(&(data.len() as u64).to_le_bytes());
    encoded.extend_from_slice(&checksum(data).to_le_bytes());
    encoded.extend_from_slice(data);
    encoded
}

fn decode(expected_hash: u64, encoded: &[u8]) -> Result<Vec<u8>, String> {
    if encoded.len() < HEADER {
        return Err(format!(
            "corrupt tier file for hash {expected_hash:016x}: truncated header"
        ));
    }
    if &encoded[..4] != MAGIC {
        return Err(format!(
            "corrupt tier file for hash {expected_hash:016x}: bad magic"
        ));
    }
    let hash = u64::from_le_bytes(encoded[4..12].try_into().expect("8 bytes"));
    if hash != expected_hash {
        return Err(format!(
            "corrupt tier file: expected hash {expected_hash:016x}, found {hash:016x}"
        ));
    }
    let length = u64::from_le_bytes(encoded[12..20].try_into().expect("8 bytes")) as usize;
    let expected_checksum = u64::from_le_bytes(encoded[20..28].try_into().expect("8 bytes"));
    if encoded.len() != HEADER + length {
        return Err(format!(
            "corrupt tier file for hash {expected_hash:016x}: length mismatch"
        ));
    }
    let data = &encoded[HEADER..];
    if checksum(data) != expected_checksum {
        return Err(format!(
            "corrupt tier file for hash {expected_hash:016x}: checksum mismatch"
        ));
    }
    Ok(data.to_vec())
}

/// Content-addressed host/NVMe tier for KV blocks.
///
/// Evicted blocks are read through `GpuBridge`, kept in a byte-budgeted host
/// LRU, and spilled to `{dir}/{hash:016x}.kv` when the host budget is exceeded.
/// Loads reverse the path and fail loudly on missing or corrupt files.
pub struct TierStore<B: GpuBridge> {
    bridge: B,
    directory: PathBuf,
    host_budget: usize,
    host: HashMap<u64, Vec<u8>>,
    host_lru: VecDeque<u64>,
    host_bytes: usize,
    stats: BlockIoStats,
    faults: TierFaults,
}

impl<B: GpuBridge> TierStore<B> {
    pub fn new(
        bridge: B,
        directory: impl Into<PathBuf>,
        host_budget: usize,
    ) -> Result<Self, String> {
        let directory = directory.into();
        fs::create_dir_all(&directory).map_err(|error| {
            format!(
                "cannot create tier directory {}: {error}",
                directory.display()
            )
        })?;
        Ok(Self {
            bridge,
            directory,
            host_budget,
            host: HashMap::new(),
            host_lru: VecDeque::new(),
            host_bytes: 0,
            stats: BlockIoStats::default(),
            faults: TierFaults::default(),
        })
    }

    pub fn fault_handle(&self) -> TierFaults {
        self.faults.clone()
    }

    pub fn stats(&self) -> BlockIoStats {
        self.stats
    }

    pub fn host_bytes(&self) -> usize {
        self.host_bytes
    }

    pub fn host_entries(&self) -> usize {
        self.host.len()
    }

    pub fn bridge(&self) -> &B {
        &self.bridge
    }

    fn path_for(&self, hash: u64) -> PathBuf {
        self.directory.join(format!("{hash:016x}.kv"))
    }

    fn remove_host(&mut self, hash: u64) -> Option<Vec<u8>> {
        let data = self.host.remove(&hash)?;
        self.host_bytes = self.host_bytes.saturating_sub(data.len());
        Some(data)
    }

    fn spill_oldest(&mut self) -> Result<(), String> {
        while let Some(hash) = self.host_lru.pop_front() {
            let Some(data) = self.host.get(&hash).cloned() else {
                continue;
            };
            let path = self.path_for(hash);
            let temporary = path.with_extension("kv.tmp");
            let encoded = encode(hash, &data);
            if let Err(error) = fs::write(&temporary, &encoded) {
                let _ = fs::remove_file(&temporary);
                return Err(format!(
                    "cannot spill block hash {hash:016x} to {}: {error}",
                    temporary.display()
                ));
            }
            if let Err(error) = fs::rename(&temporary, &path) {
                let _ = fs::remove_file(&temporary);
                return Err(format!(
                    "cannot commit tier file {}: {error}",
                    path.display()
                ));
            }
            self.remove_host(hash);
            self.stats.disk_spills += 1;
            return Ok(());
        }
        Ok(())
    }

    fn read_disk(&self, hash: u64) -> Result<Vec<u8>, String> {
        let path = self.path_for(hash);
        let encoded = fs::read(&path)
            .map_err(|error| format!("tier file {} is missing: {error}", path.display()))?;
        decode(hash, &encoded)
    }
}

impl<B: GpuBridge> BlockIo for TierStore<B> {
    fn save(&mut self, hash: u64, block: BlockId) -> Result<(), String> {
        if self.faults.fail_save() {
            return Err("injected tier save failure".to_string());
        }
        let data = self.bridge.read_block(block)?;
        if data.len() != self.bridge.block_bytes() {
            return Err(format!(
                "bridge returned {} bytes for block {block}, expected {}",
                data.len(),
                self.bridge.block_bytes()
            ));
        }
        if let Some(previous) = self.remove_host(hash) {
            let _ = previous;
        }
        self.host_bytes += data.len();
        self.host.insert(hash, data);
        self.host_lru.push_back(hash);

        while self.host_bytes > self.host_budget {
            if let Err(error) = self.spill_oldest() {
                self.remove_host(hash);
                return Err(error);
            }
        }
        self.stats.swapped_out += 1;
        Ok(())
    }

    fn load(&mut self, hash: u64, block: BlockId) -> Result<(), String> {
        if self.faults.fail_load() {
            return Err("injected tier load failure".to_string());
        }
        let (data, from_disk) = match self.host.get(&hash).cloned() {
            Some(data) => (data, false),
            None => (self.read_disk(hash)?, true),
        };
        self.bridge.write_block(block, &data)?;
        if !from_disk {
            self.remove_host(hash);
        } else {
            self.stats.disk_loads += 1;
        }
        self.stats.swapped_in += 1;
        Ok(())
    }

    fn stats(&self) -> BlockIoStats {
        self.stats
    }
}
