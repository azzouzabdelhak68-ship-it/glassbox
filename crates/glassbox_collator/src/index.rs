use crate::wal::{
    dtype_bytes, half_to_float, now_ns, ActivationMeta, ParsedActivationRecord, ParsedRecord,
    SegmentHeader, WalSegmentReader, DTYPE_BF16, DTYPE_FP16, DTYPE_FP32,
};
use byteorder::{ByteOrder, LittleEndian};
use memmap2::Mmap;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone)]
pub struct RankStepEntry {
    pub rank_id: u32,
    pub record: ParsedActivationRecord,
}

#[derive(Debug, Clone)]
pub enum StepStatus {
    Active {
        rank_records: Vec<RankStepEntry>,
        num_tokens: usize,
    },
    Dropped {
        reason: String,
        timestamp_ns: u64,
    },
}

pub struct VirtualIndex {
    pub run_dir: PathBuf,
    pub steps: BTreeMap<u64, StepStatus>,
    pub watched_addresses: Vec<String>,
    mmap_cache: Arc<RwLock<HashMap<PathBuf, Arc<Mmap>>>>,
}

impl VirtualIndex {
    pub fn scan<P: AsRef<Path>>(run_dir: P) -> io::Result<Self> {
        let root = run_dir.as_ref().to_path_buf();
        let mut steps: BTreeMap<u64, StepStatus> = BTreeMap::new();
        let mut all_addresses: HashSet<String> = HashSet::new();

        // Locate rank directories: check run_dir/ranks/rank_* first, then run_dir/rank_*
        let mut rank_dirs = Vec::new();
        let ranks_sub = root.join("ranks");
        if ranks_sub.is_dir() {
            if let Ok(entries) = fs::read_dir(&ranks_sub) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_dir() && p.file_name().map_or(false, |n| n.to_string_lossy().starts_with("rank_")) {
                        rank_dirs.push(p);
                    }
                }
            }
        }

        if rank_dirs.is_empty() {
            if let Ok(entries) = fs::read_dir(&root) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_dir() && p.file_name().map_or(false, |n| n.to_string_lossy().starts_with("rank_")) {
                        rank_dirs.push(p);
                    }
                }
            }
        }

        // If no rank dirs found, check if segments are directly in root
        if rank_dirs.is_empty() {
            rank_dirs.push(root.clone());
        }

        // Sort rank directories for deterministic ordering
        rank_dirs.sort();

        // First pass: scan all segments across all ranks
        for r_dir in &rank_dirs {
            let mut seg_files = Vec::new();
            if let Ok(entries) = fs::read_dir(r_dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_file() && p.file_name().map_or(false, |n| n.to_string_lossy().ends_with(".bin")) {
                        seg_files.push(p);
                    }
                }
            }
            seg_files.sort();

            for seg_path in seg_files {
                let (seg_hdr, records) = match WalSegmentReader::read_segment(&seg_path) {
                    Ok(res) => res,
                    Err(_) => continue, // Skip corrupted or incomplete segments gracefully
                };

                for rec in records {
                    match rec {
                        ParsedRecord::Tombstone(tomb) => {
                            let step_id = tomb.frame.step_id;
                            // Tombstone marks step as dropped
                            steps.insert(
                                step_id,
                                StepStatus::Dropped {
                                    reason: tomb.meta.reason,
                                    timestamp_ns: tomb.header.timestamp_ns,
                                },
                            );
                        }
                        ParsedRecord::Activation(act) => {
                            let step_id = act.frame.step_id;
                            for addr in &act.meta.addresses {
                                all_addresses.insert(addr.clone());
                            }

                            // If step already marked dropped, do not reactivate
                            let is_dropped = match steps.get(&step_id) {
                                Some(StepStatus::Dropped { .. }) => true,
                                _ => false,
                            };
                            if is_dropped {
                                continue;
                            }

                            let tokens = act.header.num_tokens as usize;
                            let entry = RankStepEntry {
                                rank_id: seg_hdr.rank_id,
                                record: act,
                            };

                            match steps.get_mut(&step_id) {
                                Some(StepStatus::Active { rank_records, num_tokens: existing_tokens }) => {
                                    *existing_tokens = (*existing_tokens).max(tokens);
                                    rank_records.push(entry);
                                }
                                _ => {
                                    steps.insert(
                                        step_id,
                                        StepStatus::Active {
                                            rank_records: vec![entry],
                                            num_tokens: tokens,
                                        },
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut sorted_addresses: Vec<String> = all_addresses.into_iter().collect();
        sorted_addresses.sort();

        Ok(Self {
            run_dir: root,
            steps,
            watched_addresses: sorted_addresses,
            mmap_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    fn get_mmap(&self, path: &Path) -> io::Result<Arc<Mmap>> {
        {
            let cache = self.mmap_cache.read().unwrap();
            if let Some(mmap) = cache.get(path) {
                return Ok(mmap.clone());
            }
        }
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let arc_mmap = Arc::new(mmap);
        let mut cache = self.mmap_cache.write().unwrap();
        cache.insert(path.to_path_buf(), arc_mmap.clone());
        Ok(arc_mmap)
    }

    pub fn get_steps(&self) -> Vec<u64> {
        self.steps.keys().copied().collect()
    }

    pub fn get_active_steps(&self) -> Vec<u64> {
        self.steps
            .iter()
            .filter_map(|(&k, v)| match v {
                StepStatus::Active { .. } => Some(k),
                StepStatus::Dropped { .. } => None,
            })
            .collect()
    }

    pub fn get_dropped_steps(&self) -> Vec<u64> {
        self.steps
            .iter()
            .filter_map(|(&k, v)| match v {
                StepStatus::Dropped { .. } => Some(k),
                StepStatus::Active { .. } => None,
            })
            .collect()
    }

    pub fn is_step_dropped(&self, step_id: u64) -> bool {
        match self.steps.get(&step_id) {
            Some(StepStatus::Dropped { .. }) => true,
            _ => false,
        }
    }

    pub fn get_step_num_tokens(&self, step_id: u64) -> Option<usize> {
        match self.steps.get(&step_id)? {
            StepStatus::Active { num_tokens, .. } => Some(*num_tokens),
            StepStatus::Dropped { .. } => None,
        }
    }

    pub fn get_step_req_ids(&self, step_id: u64) -> Vec<String> {
        match self.steps.get(&step_id) {
            Some(StepStatus::Active { rank_records, .. }) => {
                if let Some(first) = rank_records.first() {
                    first.record.meta.req_ids.clone()
                } else {
                    Vec::new()
                }
            }
            _ => Vec::new(),
        }
    }

    /// Query slice for (step_id, req_id, address).
    /// If req_id is None, empty, or "*", extracts full step tokens.
    /// Returns Ok(None) if address or req_id not found, Err if step dropped or file error.
    pub fn query_slice(
        &self,
        step_id: u64,
        req_id: Option<&str>,
        address: &str,
    ) -> Result<Option<Vec<f32>>, String> {
        let step = match self.steps.get(&step_id) {
            Some(s) => s,
            None => return Ok(None),
        };

        let (rank_records, total_tokens) = match step {
            StepStatus::Dropped { reason, .. } => {
                return Err(format!("Step {} was DROPPED: {}", step_id, reason));
            }
            StepStatus::Active { rank_records, num_tokens } => (rank_records, *num_tokens),
        };

        // Check if address is LOGITS-ENTROPY and handled via multi-rank reduction
        if address == "LOGITS-ENTROPY" {
            // Check if rank_records have 3-scalar reduction per rank
            let mut has_reduction = true;
            for r_rec in rank_records {
                let addrs = &r_rec.record.meta.addresses;
                if !(addrs.contains(&"LOGITS-ENTROPY".to_string())
                    || (addrs.contains(&"LOGITS-ENTROPY_m".to_string())
                        && addrs.contains(&"LOGITS-ENTROPY_s".to_string())
                        && addrs.contains(&"LOGITS-ENTROPY_w".to_string())))
                {
                    has_reduction = false;
                    break;
                }
            }
            if has_reduction && rank_records.len() > 1 {
                return self.query_logits_entropy_reduced(rank_records, req_id, total_tokens);
            }
        }

        // Standard address lookup: find rank record containing address
        for r_rec in rank_records {
            if let Some(col_idx) = r_rec.record.meta.addresses.iter().position(|a| a == address) {
                // Determine token bounds
                let (start_tok, end_tok) = match req_id {
                    Some(req) if !req.is_empty() && req != "*" => {
                        let req_pos = match r_rec.record.meta.req_ids.iter().position(|r| r == req) {
                            Some(pos) => pos,
                            None => return Ok(None),
                        };
                        let qsl = &r_rec.record.meta.query_start_loc;
                        if req_pos + 1 >= qsl.len() {
                            return Ok(None);
                        }
                        (qsl[req_pos], qsl[req_pos + 1])
                    }
                    _ => (0, r_rec.record.header.num_tokens as usize),
                };

                if start_tok >= end_tok || end_tok > r_rec.record.header.num_tokens as usize {
                    return Ok(Some(Vec::new()));
                }

                let mmap = self.get_mmap(&r_rec.record.segment_path)
                    .map_err(|e| format!("Failed to mmap segment: {}", e))?;

                let num_probes = r_rec.record.header.num_probes as usize;
                let dtype = r_rec.record.header.dtype_code;
                let b = dtype_bytes(dtype);
                let matrix_offset = r_rec.record.matrix_file_offset as usize;

                let slice_len = end_tok - start_tok;
                let mut result = Vec::with_capacity(slice_len);

                for t in start_tok..end_tok {
                    let byte_idx = matrix_offset + (t * num_probes + col_idx) * b;
                    if byte_idx + b > mmap.len() {
                        return Err(format!("Slice index {} out of bounds (file len {})", byte_idx + b, mmap.len()));
                    }

                    let val = match dtype {
                        DTYPE_FP16 => {
                            let bits = LittleEndian::read_u16(&mmap[byte_idx..byte_idx + 2]);
                            half_to_float(bits)
                        }
                        DTYPE_FP32 => {
                            LittleEndian::read_f32(&mmap[byte_idx..byte_idx + 4])
                        }
                        DTYPE_BF16 => {
                            let bits = LittleEndian::read_u16(&mmap[byte_idx..byte_idx + 2]);
                            f32::from_bits((bits as u32) << 16)
                        }
                        _ => 0.0f32,
                    };
                    result.push(val);
                }

                return Ok(Some(result));
            }
        }

        Ok(None)
    }

    /// Merges 3-scalar reduction (m_r, s_r, w_r) across ranks to compute exact Shannon entropy
    fn query_logits_entropy_reduced(
        &self,
        rank_records: &[RankStepEntry],
        req_id: Option<&str>,
        total_tokens: usize,
    ) -> Result<Option<Vec<f32>>, String> {
        let mut rank_m: Vec<Vec<f32>> = Vec::new();
        let mut rank_s: Vec<Vec<f32>> = Vec::new();
        let mut rank_w: Vec<Vec<f32>> = Vec::new();

        for r_rec in rank_records {
            let m_vec = self.read_rank_probe(r_rec, "LOGITS-ENTROPY_m", req_id)?
                .or_else(|| self.read_rank_probe_offset(r_rec, "LOGITS-ENTROPY", 0, req_id).ok().flatten());
            let s_vec = self.read_rank_probe(r_rec, "LOGITS-ENTROPY_s", req_id)?
                .or_else(|| self.read_rank_probe_offset(r_rec, "LOGITS-ENTROPY", 1, req_id).ok().flatten());
            let w_vec = self.read_rank_probe(r_rec, "LOGITS-ENTROPY_w", req_id)?
                .or_else(|| self.read_rank_probe_offset(r_rec, "LOGITS-ENTROPY", 2, req_id).ok().flatten());

            if let (Some(m), Some(s), Some(w)) = (m_vec, s_vec, w_vec) {
                rank_m.push(m);
                rank_s.push(s);
                rank_w.push(w);
            }
        }

        if rank_m.is_empty() {
            return Ok(None);
        }

        let slice_len = rank_m[0].len();
        let num_ranks = rank_m.len();
        let mut entropy = Vec::with_capacity(slice_len);

        for t in 0..slice_len {
            // Find global max m = max_r m_r
            let mut max_m = f32::NEG_INFINITY;
            for r in 0..num_ranks {
                if rank_m[r][t] > max_m {
                    max_m = rank_m[r][t];
                }
            }

            // Global s and w
            let mut global_s = 0.0f32;
            let mut global_w = 0.0f32;

            for r in 0..num_ranks {
                let mr = rank_m[r][t];
                let sr = rank_s[r][t];
                let wr = rank_w[r][t];
                let alpha = (mr - max_m).exp();
                global_s += sr * alpha;
                global_w += wr * alpha + (mr - max_m) * sr * alpha;
            }

            let h = if global_s > 0.0 {
                global_s.ln() - (global_w / global_s)
            } else {
                0.0f32
            };
            entropy.push(h);
        }

        Ok(Some(entropy))
    }

    fn read_rank_probe(
        &self,
        r_rec: &RankStepEntry,
        probe_name: &str,
        req_id: Option<&str>,
    ) -> Result<Option<Vec<f32>>, String> {
        let pos = match r_rec.record.meta.addresses.iter().position(|a| a == probe_name) {
            Some(p) => p,
            None => return Ok(None),
        };
        self.read_rank_probe_col(r_rec, pos, req_id)
    }

    fn read_rank_probe_offset(
        &self,
        r_rec: &RankStepEntry,
        base_name: &str,
        offset_idx: usize,
        req_id: Option<&str>,
    ) -> Result<Option<Vec<f32>>, String> {
        let base_pos = match r_rec.record.meta.addresses.iter().position(|a| a == base_name) {
            Some(p) => p,
            None => return Ok(None),
        };
        self.read_rank_probe_col(r_rec, base_pos + offset_idx, req_id)
    }

    fn read_rank_probe_col(
        &self,
        r_rec: &RankStepEntry,
        col_idx: usize,
        req_id: Option<&str>,
    ) -> Result<Option<Vec<f32>>, String> {
        let (start_tok, end_tok) = match req_id {
            Some(req) if !req.is_empty() && req != "*" => {
                let req_pos = match r_rec.record.meta.req_ids.iter().position(|r| r == req) {
                    Some(pos) => pos,
                    None => return Ok(None),
                };
                let qsl = &r_rec.record.meta.query_start_loc;
                if req_pos + 1 >= qsl.len() {
                    return Ok(None);
                }
                (qsl[req_pos], qsl[req_pos + 1])
            }
            _ => (0, r_rec.record.header.num_tokens as usize),
        };

        if start_tok >= end_tok || end_tok > r_rec.record.header.num_tokens as usize {
            return Ok(Some(Vec::new()));
        }

        let mmap = self.get_mmap(&r_rec.record.segment_path)
            .map_err(|e| format!("Failed to mmap segment: {}", e))?;

        let num_probes = r_rec.record.header.num_probes as usize;
        let dtype = r_rec.record.header.dtype_code;
        let b = dtype_bytes(dtype);
        let matrix_offset = r_rec.record.matrix_file_offset as usize;

        let slice_len = end_tok - start_tok;
        let mut result = Vec::with_capacity(slice_len);

        for t in start_tok..end_tok {
            let byte_idx = matrix_offset + (t * num_probes + col_idx) * b;
            if byte_idx + b > mmap.len() {
                return Err(format!("Index out of bounds in rank slice"));
            }

            let val = match dtype {
                DTYPE_FP16 => {
                    let bits = LittleEndian::read_u16(&mmap[byte_idx..byte_idx + 2]);
                    half_to_float(bits)
                }
                DTYPE_FP32 => {
                    LittleEndian::read_f32(&mmap[byte_idx..byte_idx + 4])
                }
                DTYPE_BF16 => {
                    let bits = LittleEndian::read_u16(&mmap[byte_idx..byte_idx + 2]);
                    f32::from_bits((bits as u32) << 16)
                }
                _ => 0.0f32,
            };
            result.push(val);
        }

        Ok(Some(result))
    }
}
