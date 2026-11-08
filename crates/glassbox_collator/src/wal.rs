use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use crc32fast::Hasher;
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const WAL_MAGIC: &[u8; 8] = b"GBWAL001";
pub const RECORD_MAGIC: &[u8; 4] = b"WALR";
pub const SEGMENT_HEADER_SIZE: usize = 64;
pub const RECORD_FRAME_SIZE: usize = 24;
pub const ACTIVATION_PAYLOAD_HEADER_SIZE: usize = 28;
pub const TOMBSTONE_PAYLOAD_HEADER_SIZE: usize = 16;
pub const DEFAULT_MAX_SEGMENT_SIZE: u64 = 256 * 1024 * 1024; // 256MB

pub const RECORD_TYPE_ACTIVATION: u16 = 1;
pub const RECORD_TYPE_TOMBSTONE: u16 = 2;

pub const DTYPE_FP16: u16 = 0;
pub const DTYPE_FP32: u16 = 1;
pub const DTYPE_BF16: u16 = 2;

pub fn dtype_bytes(code: u16) -> usize {
    match code {
        DTYPE_FP16 => 2,
        DTYPE_FP32 => 4,
        DTYPE_BF16 => 2,
        _ => 2,
    }
}

pub fn dtype_name(code: u16) -> &'static str {
    match code {
        DTYPE_FP16 => "fp16",
        DTYPE_FP32 => "fp32",
        DTYPE_BF16 => "bf16",
        _ => "unknown",
    }
}

pub fn parse_dtype(name: &str) -> u16 {
    match name {
        "fp16" => DTYPE_FP16,
        "fp32" => DTYPE_FP32,
        "bf16" => DTYPE_BF16,
        _ => DTYPE_FP16,
    }
}

pub fn half_to_float(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x03FF) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut e = 113u32;
        let mut m = frac;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF;
        return f32::from_bits((sign << 31) | (e << 23) | (m << 13));
    } else if exp == 0x1F {
        let f_exp = 0xFF;
        let f_frac = if frac == 0 { 0 } else { (frac << 13) | 0x00400000 };
        return f32::from_bits((sign << 31) | (f_exp << 23) | f_frac);
    }

    let f_exp = exp + 127 - 15;
    let f_frac = frac << 13;
    f32::from_bits((sign << 31) | (f_exp << 23) | f_frac)
}

pub fn float_to_half(val: f32) -> u16 {
    let fbits = val.to_bits();
    let sign = ((fbits >> 31) & 1) as u16;
    let exp = ((fbits >> 23) & 0xFF) as i32;
    let frac = (fbits & 0x007FFFFF) as u32;

    if exp == 0xFF {
        if frac != 0 {
            return (sign << 15) | 0x7E00;
        } else {
            return (sign << 15) | 0x7C00;
        }
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return (sign << 15) | 0x7C00;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign << 15;
        }
        let full_frac = frac | 0x00800000;
        let shift = (14 - new_exp) as u32;
        let h_frac = (full_frac >> shift) as u16;
        return (sign << 15) | h_frac;
    }

    let h_exp = (new_exp as u16) << 10;
    let h_frac = ((frac + 0x00001000) >> 13) as u16;
    (sign << 15) | h_exp | (h_frac & 0x03FF)
}

pub fn float_to_bfloat16(val: f32) -> u16 {
    let u32_bits = val.to_bits();
    let rounding_bias = 0x7FFFu32 + ((u32_bits >> 16) & 1);
    if val.is_nan() {
        ((u32_bits | 0x00400000) >> 16) as u16
    } else {
        ((u32_bits.wrapping_add(rounding_bias)) >> 16) as u16
    }
}

pub fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

#[derive(Debug, Clone)]
pub struct SegmentHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub rank_id: u32,
    pub seq: u64,
    pub max_segment_size: u64,
    pub timestamp_ns: u64,
    pub reserved: [u8; 24],
}

impl SegmentHeader {
    pub fn new(rank_id: u32, seq: u64, max_segment_size: u64) -> Self {
        Self {
            magic: *WAL_MAGIC,
            version: 1,
            rank_id,
            seq,
            max_segment_size,
            timestamp_ns: now_ns(),
            reserved: [0u8; 24],
        }
    }

    pub fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(&self.magic)?;
        w.write_u32::<LittleEndian>(self.version)?;
        w.write_u32::<LittleEndian>(self.rank_id)?;
        w.write_u64::<LittleEndian>(self.seq)?;
        w.write_u64::<LittleEndian>(self.max_segment_size)?;
        w.write_u64::<LittleEndian>(self.timestamp_ns)?;
        w.write_all(&self.reserved)?;
        Ok(())
    }

    pub fn read_from<R: Read>(r: &mut R) -> io::Result<Self> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != WAL_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid WAL magic: {:?}", magic),
            ));
        }
        let version = r.read_u32::<LittleEndian>()?;
        let rank_id = r.read_u32::<LittleEndian>()?;
        let seq = r.read_u64::<LittleEndian>()?;
        let max_segment_size = r.read_u64::<LittleEndian>()?;
        let timestamp_ns = r.read_u64::<LittleEndian>()?;
        let mut reserved = [0u8; 24];
        r.read_exact(&mut reserved)?;

        Ok(Self {
            magic,
            version,
            rank_id,
            seq,
            max_segment_size,
            timestamp_ns,
            reserved,
        })
    }
}

#[derive(Debug, Clone)]
pub struct RecordFrame {
    pub magic: [u8; 4],
    pub record_type: u16,
    pub flags: u16,
    pub payload_len: u32,
    pub step_id: u64,
    pub crc32: u32,
}

impl RecordFrame {
    pub fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(&self.magic)?;
        w.write_u16::<LittleEndian>(self.record_type)?;
        w.write_u16::<LittleEndian>(self.flags)?;
        w.write_u32::<LittleEndian>(self.payload_len)?;
        w.write_u64::<LittleEndian>(self.step_id)?;
        w.write_u32::<LittleEndian>(self.crc32)?;
        Ok(())
    }

    pub fn read_from<R: Read>(r: &mut R) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != RECORD_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid record magic: {:?}", magic),
            ));
        }
        let record_type = r.read_u16::<LittleEndian>()?;
        let flags = r.read_u16::<LittleEndian>()?;
        let payload_len = r.read_u32::<LittleEndian>()?;
        let step_id = r.read_u64::<LittleEndian>()?;
        let crc32 = r.read_u32::<LittleEndian>()?;

        Ok(Self {
            magic,
            record_type,
            flags,
            payload_len,
            step_id,
            crc32,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationMeta {
    pub req_ids: Vec<String>,
    pub query_start_loc: Vec<usize>,
    pub addresses: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logical_positions: Option<Vec<usize>>,
}

#[derive(Debug, Clone)]
pub struct ActivationPayloadHeader {
    pub timestamp_ns: u64,
    pub num_tokens: u32,
    pub num_probes: u32,
    pub dtype_code: u16,
    pub reserved: u16,
    pub num_reqs: u32,
    pub meta_json_len: u32,
}

impl ActivationPayloadHeader {
    pub fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_u64::<LittleEndian>(self.timestamp_ns)?;
        w.write_u32::<LittleEndian>(self.num_tokens)?;
        w.write_u32::<LittleEndian>(self.num_probes)?;
        w.write_u16::<LittleEndian>(self.dtype_code)?;
        w.write_u16::<LittleEndian>(self.reserved)?;
        w.write_u32::<LittleEndian>(self.num_reqs)?;
        w.write_u32::<LittleEndian>(self.meta_json_len)?;
        Ok(())
    }

    pub fn read_from<R: Read>(r: &mut R) -> io::Result<Self> {
        let timestamp_ns = r.read_u64::<LittleEndian>()?;
        let num_tokens = r.read_u32::<LittleEndian>()?;
        let num_probes = r.read_u32::<LittleEndian>()?;
        let dtype_code = r.read_u16::<LittleEndian>()?;
        let reserved = r.read_u16::<LittleEndian>()?;
        let num_reqs = r.read_u32::<LittleEndian>()?;
        let meta_json_len = r.read_u32::<LittleEndian>()?;

        Ok(Self {
            timestamp_ns,
            num_tokens,
            num_probes,
            dtype_code,
            reserved,
            num_reqs,
            meta_json_len,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TombstoneMeta {
    pub status: String,
    pub reason: String,
    pub dropped_step_id: u64,
}

#[derive(Debug, Clone)]
pub struct TombstonePayloadHeader {
    pub timestamp_ns: u64,
    pub reason_code: u32,
    pub meta_json_len: u32,
}

impl TombstonePayloadHeader {
    pub fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_u64::<LittleEndian>(self.timestamp_ns)?;
        w.write_u32::<LittleEndian>(self.reason_code)?;
        w.write_u32::<LittleEndian>(self.meta_json_len)?;
        Ok(())
    }

    pub fn read_from<R: Read>(r: &mut R) -> io::Result<Self> {
        let timestamp_ns = r.read_u64::<LittleEndian>()?;
        let reason_code = r.read_u32::<LittleEndian>()?;
        let meta_json_len = r.read_u32::<LittleEndian>()?;

        Ok(Self {
            timestamp_ns,
            reason_code,
            meta_json_len,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ParsedActivationRecord {
    pub frame: RecordFrame,
    pub header: ActivationPayloadHeader,
    pub meta: ActivationMeta,
    pub matrix_file_offset: u64,
    pub matrix_len: usize,
    pub segment_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct ParsedTombstoneRecord {
    pub frame: RecordFrame,
    pub header: TombstonePayloadHeader,
    pub meta: TombstoneMeta,
    pub segment_path: PathBuf,
}

#[derive(Debug, Clone)]
pub enum ParsedRecord {
    Activation(ParsedActivationRecord),
    Tombstone(ParsedTombstoneRecord),
}

pub struct WalWriter {
    rank_dir: PathBuf,
    rank_id: u32,
    max_segment_size: u64,
    current_seq: u64,
    current_file: Option<File>,
    current_file_path: Option<PathBuf>,
    current_segment_bytes: u64,
}

impl WalWriter {
    pub fn new<P: AsRef<Path>>(
        rank_dir: P,
        rank_id: u32,
        max_segment_size: u64,
    ) -> io::Result<Self> {
        let path = rank_dir.as_ref().to_path_buf();
        fs::create_dir_all(&path)?;

        let mut writer = Self {
            rank_dir: path,
            rank_id,
            max_segment_size: if max_segment_size > 0 {
                max_segment_size
            } else {
                DEFAULT_MAX_SEGMENT_SIZE
            },
            current_seq: 0,
            current_file: None,
            current_file_path: None,
            current_segment_bytes: 0,
        };

        writer.rotate_segment()?;
        Ok(writer)
    }

    fn advise_dontneed(file: &File) {
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::io::AsRawFd;
            unsafe {
                libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = file;
        }
    }

    pub fn rotate_segment(&mut self) -> io::Result<()> {
        if let Some(ref mut file) = self.current_file {
            file.flush()?;
            Self::advise_dontneed(file);
        }

        let seg_name = format!("segment_{:06}.bin", self.current_seq);
        let seg_path = self.rank_dir.join(&seg_name);

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&seg_path)?;

        let header = SegmentHeader::new(self.rank_id, self.current_seq, self.max_segment_size);
        header.write_to(&mut file)?;
        file.flush()?;

        self.current_file = Some(file);
        self.current_file_path = Some(seg_path);
        self.current_segment_bytes = SEGMENT_HEADER_SIZE as u64;
        self.current_seq += 1;

        Ok(())
    }

    pub fn write_activation(
        &mut self,
        step_id: u64,
        num_tokens: u32,
        num_probes: u32,
        dtype_code: u16,
        meta: &ActivationMeta,
        raw_matrix: &[u8],
    ) -> io::Result<()> {
        let meta_json = serde_json::to_vec(meta)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let meta_json_len = meta_json.len() as u32;

        let payload_hdr = ActivationPayloadHeader {
            timestamp_ns: now_ns(),
            num_tokens,
            num_probes,
            dtype_code,
            reserved: 0,
            num_reqs: meta.req_ids.len() as u32,
            meta_json_len,
        };

        // Header (28 bytes) + JSON
        let unpadded_len = ACTIVATION_PAYLOAD_HEADER_SIZE + meta_json.len();
        // Matrix 8-byte alignment relative to payload start:
        // (RECORD_FRAME_SIZE(24) + ACTIVATION_PAYLOAD_HEADER_SIZE(28) + meta_json.len() + pad) % 8 == 0
        // 24 % 8 == 0, so (28 + meta_json.len() + pad) % 8 == 0 => (4 + meta_json.len() + pad) % 8 == 0
        let pad_len = (8 - ((ACTIVATION_PAYLOAD_HEADER_SIZE + meta_json.len()) % 8)) % 8;
        let matrix_len = raw_matrix.len();
        let payload_body_len = unpadded_len + pad_len + matrix_len;
        // Total payload length aligned to 8 bytes:
        let trailing_pad_len = (8 - (payload_body_len % 8)) % 8;
        let total_payload_len = payload_body_len + trailing_pad_len;

        let mut payload_buf = Vec::with_capacity(total_payload_len);
        payload_hdr.write_to(&mut payload_buf)?;
        payload_buf.extend_from_slice(&meta_json);
        payload_buf.resize(payload_buf.len() + pad_len, 0u8);
        payload_buf.extend_from_slice(raw_matrix);
        payload_buf.resize(payload_buf.len() + trailing_pad_len, 0u8);

        let mut hasher = Hasher::new();
        hasher.update(&payload_buf);
        let crc32 = hasher.finalize();

        let frame = RecordFrame {
            magic: *RECORD_MAGIC,
            record_type: RECORD_TYPE_ACTIVATION,
            flags: 0,
            payload_len: total_payload_len as u32,
            step_id,
            crc32,
        };

        let total_record_size = RECORD_FRAME_SIZE as u64 + total_payload_len as u64;
        if self.current_segment_bytes + total_record_size > self.max_segment_size
            && self.current_segment_bytes > SEGMENT_HEADER_SIZE as u64
        {
            self.rotate_segment()?;
        }

        let file = self.current_file.as_mut().ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotConnected, "No segment file open")
        })?;

        frame.write_to(file)?;
        file.write_all(&payload_buf)?;
        file.flush()?;

        self.current_segment_bytes += total_record_size;
        Ok(())
    }

    pub fn write_tombstone(
        &mut self,
        step_id: u64,
        reason_code: u32,
        reason_str: &str,
    ) -> io::Result<()> {
        let meta = TombstoneMeta {
            status: "DROPPED".to_string(),
            reason: reason_str.to_string(),
            dropped_step_id: step_id,
        };

        let meta_json = serde_json::to_vec(&meta)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let meta_json_len = meta_json.len() as u32;

        let payload_hdr = TombstonePayloadHeader {
            timestamp_ns: now_ns(),
            reason_code,
            meta_json_len,
        };

        let unpadded_len = TOMBSTONE_PAYLOAD_HEADER_SIZE + meta_json.len();
        let pad_len = (8 - (unpadded_len % 8)) % 8;
        let total_payload_len = unpadded_len + pad_len;

        let mut payload_buf = Vec::with_capacity(total_payload_len);
        payload_hdr.write_to(&mut payload_buf)?;
        payload_buf.extend_from_slice(&meta_json);
        payload_buf.resize(payload_buf.len() + pad_len, 0u8);

        let mut hasher = Hasher::new();
        hasher.update(&payload_buf);
        let crc32 = hasher.finalize();

        let frame = RecordFrame {
            magic: *RECORD_MAGIC,
            record_type: RECORD_TYPE_TOMBSTONE,
            flags: 0,
            payload_len: total_payload_len as u32,
            step_id,
            crc32,
        };

        let total_record_size = RECORD_FRAME_SIZE as u64 + total_payload_len as u64;
        if self.current_segment_bytes + total_record_size > self.max_segment_size
            && self.current_segment_bytes > SEGMENT_HEADER_SIZE as u64
        {
            self.rotate_segment()?;
        }

        let file = self.current_file.as_mut().ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotConnected, "No segment file open")
        })?;

        frame.write_to(file)?;
        file.write_all(&payload_buf)?;
        file.flush()?;

        self.current_segment_bytes += total_record_size;
        Ok(())
    }

    pub fn close(&mut self) -> io::Result<()> {
        if let Some(ref mut file) = self.current_file {
            file.flush()?;
            Self::advise_dontneed(file);
        }
        self.current_file = None;
        Ok(())
    }
}

impl Drop for WalWriter {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

pub struct WalSegmentReader;

impl WalSegmentReader {
    pub fn read_segment<P: AsRef<Path>>(
        path: P,
    ) -> io::Result<(SegmentHeader, Vec<ParsedRecord>)> {
        let seg_path = path.as_ref().to_path_buf();
        let mut file = File::open(&seg_path)?;
        let file_len = file.metadata()?.len();

        if file_len < SEGMENT_HEADER_SIZE as u64 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "File too small for segment header",
            ));
        }

        let seg_header = SegmentHeader::read_from(&mut file)?;
        let mut records = Vec::new();
        let mut cur_pos = SEGMENT_HEADER_SIZE as u64;

        while cur_pos + (RECORD_FRAME_SIZE as u64) <= file_len {
            file.seek(SeekFrom::Start(cur_pos))?;
            let frame = match RecordFrame::read_from(&mut file) {
                Ok(f) => f,
                Err(_e) => {
                    // Possible clean EOF or partial record at end of file
                    break;
                }
            };

            let payload_len = frame.payload_len as u64;
            if cur_pos + (RECORD_FRAME_SIZE as u64) + payload_len > file_len {
                // Incomplete record
                break;
            }

            let mut payload = vec![0u8; frame.payload_len as usize];
            file.read_exact(&mut payload)?;

            // Verify CRC32
            let mut hasher = Hasher::new();
            hasher.update(&payload);
            let computed_crc = hasher.finalize();
            if computed_crc != frame.crc32 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "CRC32 mismatch at offset {}: expected 0x{:08X}, got 0x{:08X}",
                        cur_pos, frame.crc32, computed_crc
                    ),
                ));
            }

            match frame.record_type {
                RECORD_TYPE_ACTIVATION => {
                    if payload.len() < ACTIVATION_PAYLOAD_HEADER_SIZE {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Activation payload too small",
                        ));
                    }
                    let mut cursor = Cursor::new(&payload[0..ACTIVATION_PAYLOAD_HEADER_SIZE]);
                    let act_hdr = ActivationPayloadHeader::read_from(&mut cursor)?;

                    let json_start = ACTIVATION_PAYLOAD_HEADER_SIZE;
                    let json_end = json_start + (act_hdr.meta_json_len as usize);
                    if json_end > payload.len() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Activation JSON bounds exceed payload",
                        ));
                    }

                    let meta: ActivationMeta = serde_json::from_slice(&payload[json_start..json_end])
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

                    let pad_len = (8 - ((ACTIVATION_PAYLOAD_HEADER_SIZE + meta_json_len(act_hdr.meta_json_len)) % 8)) % 8;
                    let matrix_start_in_payload = ACTIVATION_PAYLOAD_HEADER_SIZE + (act_hdr.meta_json_len as usize) + pad_len;
                    let matrix_len = (act_hdr.num_tokens as usize)
                        * (act_hdr.num_probes as usize)
                        * dtype_bytes(act_hdr.dtype_code);

                    let matrix_file_offset = cur_pos + (RECORD_FRAME_SIZE as u64) + (matrix_start_in_payload as u64);

                    records.push(ParsedRecord::Activation(ParsedActivationRecord {
                        frame: frame.clone(),
                        header: act_hdr,
                        meta,
                        matrix_file_offset,
                        matrix_len,
                        segment_path: seg_path.clone(),
                    }));
                }
                RECORD_TYPE_TOMBSTONE => {
                    if payload.len() < TOMBSTONE_PAYLOAD_HEADER_SIZE {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Tombstone payload too small",
                        ));
                    }
                    let mut cursor = Cursor::new(&payload[0..TOMBSTONE_PAYLOAD_HEADER_SIZE]);
                    let tomb_hdr = TombstonePayloadHeader::read_from(&mut cursor)?;

                    let json_start = TOMBSTONE_PAYLOAD_HEADER_SIZE;
                    let json_end = json_start + (tomb_hdr.meta_json_len as usize);
                    if json_end > payload.len() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Tombstone JSON bounds exceed payload",
                        ));
                    }

                    let meta: TombstoneMeta = serde_json::from_slice(&payload[json_start..json_end])
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

                    records.push(ParsedRecord::Tombstone(ParsedTombstoneRecord {
                        frame: frame.clone(),
                        header: tomb_hdr,
                        meta,
                        segment_path: seg_path.clone(),
                    }));
                }
                _ => {
                    // Unknown record type - skipped safely
                }
            }

            cur_pos += (RECORD_FRAME_SIZE as u64) + payload_len;
        }

        Ok((seg_header, records))
    }
}

fn meta_json_len(len: u32) -> usize {
    len as usize
}
