//! Minimal GGUF header reader: metadata and tensor shapes, never tensor data.
//! Large arrays (the tokenizer vocabulary) are skipped and only their length is kept.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::collections::BTreeMap;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(untagged)]
pub enum Value {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    /// Small numeric/bool arrays are kept.
    Arr(Vec<Value>),
    /// Large or string arrays: only the length (strings longer than 256 KB are dropped too).
    Skipped { len: u64 },
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U64(v) => Some(*v),
            Value::I64(v) if *v >= 0 => Some(*v as u64),
            Value::F64(v) if *v >= 0.0 => Some(*v as u64),
            Value::Bool(b) => Some(*b as u64),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::U64(v) => Some(*v as f64),
            Value::I64(v) => Some(*v as f64),
            Value::F64(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> Option<u64> {
        match self {
            Value::Arr(a) => Some(a.len() as u64),
            Value::Skipped { len } => Some(*len),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct GgufInfo {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: BTreeMap<String, Value>,
    /// Sum of the element counts of all tensors.
    pub parameter_count: u64,
}

impl GgufInfo {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }
    pub fn arch(&self) -> &str {
        self.get("general.architecture").and_then(Value::as_str).unwrap_or("")
    }
    pub fn arch_u64(&self, suffix: &str) -> Option<u64> {
        self.get(&format!("{}.{suffix}", self.arch())).and_then(Value::as_u64)
    }
    pub fn arch_val(&self, suffix: &str) -> Option<&Value> {
        self.get(&format!("{}.{suffix}", self.arch()))
    }
    pub fn str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Value::as_str)
    }
    /// True for vision/audio projector files (mmproj).
    pub fn is_projector(&self) -> bool {
        matches!(self.str("general.type"), Some("mmproj") | Some("projector"))
            || matches!(self.arch(), "clip" | "mmproj")
    }
}

const MAX_KEEP_ARRAY: u64 = 4096;
const MAX_KEEP_STRING: u64 = 256 * 1024;

struct R<T: Read + Seek> {
    r: BufReader<T>,
}

impl<T: Read + Seek> R<T> {
    fn u8(&mut self) -> Result<u8> {
        let mut b = [0u8; 1];
        self.r.read_exact(&mut b)?;
        Ok(b[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let mut b = [0u8; 2];
        self.r.read_exact(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    fn u32(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.r.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn u64(&mut self) -> Result<u64> {
        let mut b = [0u8; 8];
        self.r.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    fn skip(&mut self, n: u64) -> Result<()> {
        self.r.seek_relative(n as i64)?;
        Ok(())
    }
    fn string(&mut self) -> Result<Option<String>> {
        let len = self.u64()?;
        if len > 1 << 30 {
            bail!("string length {len} is not plausible");
        }
        if len > MAX_KEEP_STRING {
            self.skip(len)?;
            return Ok(None);
        }
        let mut buf = vec![0u8; len as usize];
        self.r.read_exact(&mut buf)?;
        Ok(Some(String::from_utf8_lossy(&buf).into_owned()))
    }

    fn scalar_size(t: u32) -> Option<u64> {
        Some(match t {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            10..=12 => 8,
            _ => return None,
        })
    }

    fn value(&mut self, t: u32) -> Result<Value> {
        Ok(match t {
            0 => Value::U64(self.u8()? as u64),
            1 => Value::I64(self.u8()? as i8 as i64),
            2 => Value::U64(self.u16()? as u64),
            3 => Value::I64(self.u16()? as i16 as i64),
            4 => Value::U64(self.u32()? as u64),
            5 => Value::I64(self.u32()? as i32 as i64),
            6 => Value::F64(f32::from_bits(self.u32()?) as f64),
            7 => Value::Bool(self.u8()? != 0),
            8 => match self.string()? {
                Some(s) => Value::Str(s),
                None => Value::Skipped { len: 0 },
            },
            9 => {
                let et = self.u32()?;
                let n = self.u64()?;
                if let (Some(sz), true) = (Self::scalar_size(et), n > MAX_KEEP_ARRAY) {
                    self.skip(sz * n)?;
                    return Ok(Value::Skipped { len: n });
                }
                if et == 8 {
                    for _ in 0..n {
                        let l = self.u64()?;
                        self.skip(l)?;
                    }
                    return Ok(Value::Skipped { len: n });
                }
                let mut v = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    v.push(self.value(et)?);
                }
                Value::Arr(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            other => bail!("unknown GGUF value type {other}"),
        })
    }
}

pub fn read(path: &Path) -> Result<GgufInfo> {
    let f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    read_from(f).with_context(|| format!("reading GGUF header of {}", path.display()))
}

pub fn read_from<T: Read + Seek>(f: T) -> Result<GgufInfo> {
    let mut r = R { r: BufReader::with_capacity(1 << 20, f) };
    let mut magic = [0u8; 4];
    r.r.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        bail!("not a GGUF file");
    }
    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        bail!("unsupported GGUF version {version}");
    }
    let tensor_count = r.u64()?;
    let kv_count = r.u64()?;
    let mut metadata = BTreeMap::new();
    for _ in 0..kv_count {
        let key = r.string()?.unwrap_or_default();
        let t = r.u32()?;
        let v = r.value(t)?;
        metadata.insert(key, v);
    }
    let mut parameter_count = 0u64;
    for _ in 0..tensor_count {
        let _name = r.string()?;
        let n_dims = r.u32()?;
        let mut elems = 1u64;
        for _ in 0..n_dims {
            elems = elems.saturating_mul(r.u64()?);
        }
        let _ty = r.u32()?;
        let _off = r.u64()?;
        parameter_count = parameter_count.saturating_add(elems);
    }
    Ok(GgufInfo { version, tensor_count, metadata, parameter_count })
}

/// Writer used by tests and fixtures to build small GGUF headers.
pub mod write {
    use std::io::Write;

    pub enum W<'a> {
        U32(u32),
        U64(u64),
        F32(f32),
        Bool(bool),
        Str(&'a str),
        ArrU32(Vec<u32>),
        ArrI32(Vec<i32>),
        ArrBool(Vec<bool>),
        ArrStr(Vec<&'a str>),
    }

    fn s(out: &mut Vec<u8>, v: &str) {
        out.extend((v.len() as u64).to_le_bytes());
        out.extend(v.as_bytes());
    }

    /// Build a GGUF v3 header with the given metadata and tensors (name, dims).
    pub fn build(meta: &[(&str, W)], tensors: &[(&str, Vec<u64>)]) -> Vec<u8> {
        let mut o = Vec::new();
        o.extend(b"GGUF");
        o.extend(3u32.to_le_bytes());
        o.extend((tensors.len() as u64).to_le_bytes());
        o.extend((meta.len() as u64).to_le_bytes());
        for (k, v) in meta {
            s(&mut o, k);
            match v {
                W::U32(x) => {
                    o.extend(4u32.to_le_bytes());
                    o.extend(x.to_le_bytes());
                }
                W::U64(x) => {
                    o.extend(10u32.to_le_bytes());
                    o.extend(x.to_le_bytes());
                }
                W::F32(x) => {
                    o.extend(6u32.to_le_bytes());
                    o.extend(x.to_le_bytes());
                }
                W::Bool(x) => {
                    o.extend(7u32.to_le_bytes());
                    o.push(*x as u8);
                }
                W::Str(x) => {
                    o.extend(8u32.to_le_bytes());
                    s(&mut o, x);
                }
                W::ArrU32(a) => {
                    o.extend(9u32.to_le_bytes());
                    o.extend(4u32.to_le_bytes());
                    o.extend((a.len() as u64).to_le_bytes());
                    for x in a {
                        o.extend(x.to_le_bytes());
                    }
                }
                W::ArrI32(a) => {
                    o.extend(9u32.to_le_bytes());
                    o.extend(5u32.to_le_bytes());
                    o.extend((a.len() as u64).to_le_bytes());
                    for x in a {
                        o.extend(x.to_le_bytes());
                    }
                }
                W::ArrBool(a) => {
                    o.extend(9u32.to_le_bytes());
                    o.extend(7u32.to_le_bytes());
                    o.extend((a.len() as u64).to_le_bytes());
                    for x in a {
                        o.push(*x as u8);
                    }
                }
                W::ArrStr(a) => {
                    o.extend(9u32.to_le_bytes());
                    o.extend(8u32.to_le_bytes());
                    o.extend((a.len() as u64).to_le_bytes());
                    for x in a {
                        s(&mut o, x);
                    }
                }
            }
        }
        for (name, dims) in tensors {
            s(&mut o, name);
            o.extend((dims.len() as u32).to_le_bytes());
            for d in dims {
                o.extend(d.to_le_bytes());
            }
            o.extend(0u32.to_le_bytes());
            o.extend(0u64.to_le_bytes());
        }
        o
    }

    pub fn to_file(path: &std::path::Path, bytes: &[u8], pad_to: u64) {
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(bytes).unwrap();
        if pad_to > bytes.len() as u64 {
            f.set_len(pad_to).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::write::*;
    use super::*;

    #[test]
    fn parses_metadata_and_skips_vocab() {
        let vocab: Vec<&str> = (0..5000).map(|_| "tok").collect();
        let bytes = build(
            &[
                ("general.architecture", W::Str("gemma4")),
                ("gemma4.block_count", W::U32(48)),
                ("gemma4.attention.sliding_window", W::U32(1024)),
                ("gemma4.attention.sliding_window_pattern", W::ArrBool(vec![true, true, true, true, true, false])),
                ("tokenizer.ggml.tokens", W::ArrStr(vocab)),
                ("general.parameter_count", W::U64(12_000_000_000)),
            ],
            &[("blk.0.attn_q.weight", vec![3840, 4096]), ("token_embd.weight", vec![3840, 262144])],
        );
        let info = read_from(std::io::Cursor::new(bytes)).unwrap();
        assert_eq!(info.arch(), "gemma4");
        assert_eq!(info.arch_u64("block_count"), Some(48));
        assert_eq!(info.get("tokenizer.ggml.tokens").unwrap().len(), Some(5000));
        assert_eq!(info.arch_val("attention.sliding_window_pattern").unwrap().len(), Some(6));
        assert_eq!(info.parameter_count, 3840 * 4096 + 3840 * 262144);
    }

    #[test]
    fn rejects_non_gguf() {
        assert!(read_from(std::io::Cursor::new(b"NOPE0000".to_vec())).is_err());
    }
}
