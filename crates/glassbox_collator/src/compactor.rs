use crate::index::{StepStatus, VirtualIndex};
use crate::wal::{
    dtype_bytes, float_to_bfloat16, float_to_half, parse_dtype, DTYPE_BF16, DTYPE_FP16, DTYPE_FP32,
};
use byteorder::{LittleEndian, WriteBytesExt};
use flate2::write::ZlibEncoder;
use flate2::Compression;
use serde_json::{json, Value};
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

pub const SPARSE_MAGIC: &[u8; 8] = b"GBSPR001";

pub struct CompactorOptions {
    pub precision: String,
    pub chunk_tokens: usize,
}

impl Default for CompactorOptions {
    fn default() -> Self {
        Self {
            precision: "fp16".to_string(),
            chunk_tokens: 64,
        }
    }
}

pub fn compact_index(
    index: &VirtualIndex,
    output_dir: &Path,
    options: &CompactorOptions,
) -> Result<String, String> {
    fs::create_dir_all(output_dir).map_err(|e| format!("Failed to create output dir: {}", e))?;

    let run_id = output_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "RUN-000001".to_string());

    // Filter active steps and sort
    let active_steps = index.get_active_steps();
    if active_steps.is_empty() {
        return Err("No active (non-dropped) steps to compact".to_string());
    }

    // Determine watched probe addresses
    // Check if LOGITS-ENTROPY is present in any variant
    let mut watched_columns: Vec<String> = Vec::new();
    let mut has_logits_entropy = false;

    for addr in &index.watched_addresses {
        if addr == "LOGITS-ENTROPY"
            || addr == "LOGITS-ENTROPY_m"
            || addr == "LOGITS-ENTROPY_s"
            || addr == "LOGITS-ENTROPY_w"
        {
            has_logits_entropy = true;
        } else if !watched_columns.contains(addr) {
            watched_columns.push(addr.clone());
        }
    }

    if has_logits_entropy && !watched_columns.contains(&"LOGITS-ENTROPY".to_string()) {
        watched_columns.push("LOGITS-ENTROPY".to_string());
    }
    watched_columns.sort();

    let num_columns = watched_columns.len();
    if num_columns == 0 {
        return Err("No watched addresses found to compact".to_string());
    }

    // Calculate total tokens T across active steps
    let mut total_tokens = 0usize;
    for &step_id in &active_steps {
        if let Some(toks) = index.get_step_num_tokens(step_id) {
            total_tokens += toks;
        }
    }

    if total_tokens == 0 {
        return Err("Total tokens across steps is 0".to_string());
    }

    // Build dense row-major matrix: [T, W] in float32
    let mut dense_matrix: Vec<f32> = vec![0.0f32; total_tokens * num_columns];
    let mut token_cursor = 0usize;

    for &step_id in &active_steps {
        let step_tokens = match index.get_step_num_tokens(step_id) {
            Some(t) => t,
            None => continue,
        };

        for (col_idx, addr) in watched_columns.iter().enumerate() {
            if let Ok(Some(slice)) = index.query_slice(step_id, None, addr) {
                let copy_len = slice.len().min(step_tokens);
                for t in 0..copy_len {
                    let global_row = token_cursor + t;
                    dense_matrix[global_row * num_columns + col_idx] = slice[t];
                }
            }
        }

        token_cursor += step_tokens;
    }

    // Encode matrix into target precision
    let prec_code = parse_dtype(&options.precision);
    let bytes_per_elem = dtype_bytes(prec_code);
    let mut raw_bytes: Vec<u8> = Vec::with_capacity(dense_matrix.len() * bytes_per_elem);

    match prec_code {
        DTYPE_FP16 => {
            for &val in &dense_matrix {
                let h = float_to_half(val);
                raw_bytes.write_u16::<LittleEndian>(h).unwrap();
            }
        }
        DTYPE_FP32 => {
            for &val in &dense_matrix {
                raw_bytes.write_f32::<LittleEndian>(val).unwrap();
            }
        }
        DTYPE_BF16 => {
            for &val in &dense_matrix {
                let b = float_to_bfloat16(val);
                raw_bytes.write_u16::<LittleEndian>(b).unwrap();
            }
        }
        _ => {
            for &val in &dense_matrix {
                let h = float_to_half(val);
                raw_bytes.write_u16::<LittleEndian>(h).unwrap();
            }
        }
    }

    // Chunk into chunk_tokens rows
    let chunk_tokens = if options.chunk_tokens > 0 { options.chunk_tokens } else { 64 };
    let num_chunks = (total_tokens + chunk_tokens - 1) / chunk_tokens;

    let row_stride_bytes = num_columns * bytes_per_elem;
    let mut compressed_chunks: Vec<Vec<u8>> = Vec::with_capacity(num_chunks);
    let mut chunk_meta: Vec<(u32, u32)> = Vec::with_capacity(num_chunks); // (comp_len, uncomp_len)

    for c in 0..num_chunks {
        let start_row = c * chunk_tokens;
        let end_row = (start_row + chunk_tokens).min(total_tokens);
        let num_chunk_rows = end_row - start_row;

        let start_byte = start_row * row_stride_bytes;
        let end_byte = end_row * row_stride_bytes;
        let chunk_slice = &raw_bytes[start_byte..end_byte];
        let uncomp_len = chunk_slice.len() as u32;

        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(6));
        encoder.write_all(chunk_slice).map_err(|e| format!("Zlib compress error: {}", e))?;
        let comp_data = encoder.finish().map_err(|e| format!("Zlib finish error: {}", e))?;
        let comp_len = comp_data.len() as u32;

        compressed_chunks.push(comp_data);
        chunk_meta.push((comp_len, uncomp_len));
    }

    // Write deep.bin
    let hdr_size = 24usize; // <8sIIII
    let idx_entry_size = 16usize; // <QII
    let payload_start = hdr_size + num_chunks * idx_entry_size;

    let deep_bin_path = output_dir.join("deep.bin");
    let mut file = File::create(&deep_bin_path)
        .map_err(|e| format!("Failed to create deep.bin: {}", e))?;

    // Header: <8sIIII: SPARSE_MAGIC (8), version (4), chunk_tokens (4), num_chunks (4), prec_code (4)
    file.write_all(SPARSE_MAGIC)
        .map_err(|e| format!("Write magic error: {}", e))?;
    file.write_u32::<LittleEndian>(1)
        .map_err(|e| format!("Write version error: {}", e))?;
    file.write_u32::<LittleEndian>(chunk_tokens as u32)
        .map_err(|e| format!("Write chunk_tokens error: {}", e))?;
    file.write_u32::<LittleEndian>(num_chunks as u32)
        .map_err(|e| format!("Write num_chunks error: {}", e))?;
    file.write_u32::<LittleEndian>(prec_code as u32)
        .map_err(|e| format!("Write prec_code error: {}", e))?;

    // Index table: <QII (offset, comp_len, uncomp_len)
    let mut cur_offset = payload_start as u64;
    for (comp_len, uncomp_len) in &chunk_meta {
        file.write_u64::<LittleEndian>(cur_offset)
            .map_err(|e| format!("Write index offset error: {}", e))?;
        file.write_u32::<LittleEndian>(*comp_len)
            .map_err(|e| format!("Write index comp_len error: {}", e))?;
        file.write_u32::<LittleEndian>(*uncomp_len)
            .map_err(|e| format!("Write index uncomp_len error: {}", e))?;
        cur_offset += *comp_len as u64;
    }

    // Write compressed chunks
    for chunk in &compressed_chunks {
        file.write_all(chunk)
            .map_err(|e| format!("Write chunk data error: {}", e))?;
    }
    file.flush()
        .map_err(|e| format!("Flush deep.bin error: {}", e))?;

    // Write deep_index.json
    let deep_index_path = output_dir.join("deep_index.json");
    let mut columns_obj = serde_json::Map::new();
    for (offset, addr) in watched_columns.iter().enumerate() {
        columns_obj.insert(
            addr.clone(),
            json!({
                "offset": offset,
                "dtype": options.precision,
                "shape": [total_tokens]
            }),
        );
    }

    let token_positions: Vec<usize> = (0..total_tokens).collect();

    let index_json = json!({
        "run_id": run_id,
        "model_id": "M-001",
        "checkpoint_sha256": "0".repeat(64),
        "triggered": true,
        "dtype": options.precision,
        "tokens": total_tokens,
        "every_n": 1,
        "token_positions": token_positions,
        "capture_mode": "all",
        "watched": watched_columns,
        "columns": columns_obj,
        "dtype_meta": {
            "endian": "little",
            "order": "row_major_token_major",
            "source": if options.precision == "bf16" { "ieee_bfloat16_le" } else { "logical_values" },
            "stored": options.precision
        },
        "flags": {
            "bad_output": false,
            "ring_overflow": false,
            "hook_detached": false,
            "truncated": false
        },
        "timing": {
            "ms_per_token": 0.0,
            "cuda_event_ms": 0.0
        },
        "tags": [],
        "error": Value::Null
    });

    let index_file = File::create(&deep_index_path)
        .map_err(|e| format!("Failed to create deep_index.json: {}", e))?;
    serde_json::to_writer_pretty(index_file, &index_json)
        .map_err(|e| format!("Failed to write deep_index.json: {}", e))?;

    Ok(deep_bin_path.to_string_lossy().to_string())
}
