use std::{fs, path::Path};
use protobuf::Message;
use serde_json::{json, Map, Value};
mod generated { include!(concat!(env!("OUT_DIR"), "/protos/mod.rs")); }
use generated::trajectory_bench::{Mutation, Step, Trajectory};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn extras(bytes: &[u8]) -> Result<Map<String, Value>> {
    if bytes.is_empty() { Ok(Map::new()) } else { Ok(serde_json::from_slice(bytes)?) }
}

fn mutation(item: &Mutation) -> Result<Value> {
    let mut out = extras(&item.extras_json)?;
    for (i, (key, value)) in [
        ("occupied", json!(item.occupied)), ("kind", json!(item.kind)),
        ("active", json!(item.active)), ("reward_weight", json!(item.reward_weight)),
        ("coordinate", json!(item.coordinate)),
    ].into_iter().enumerate() {
        if item.presence & (1 << i) != 0 { out.insert(key.into(), value); }
    }
    Ok(Value::Object(out))
}

fn step(item: &Step) -> Result<Value> {
    let mut out = extras(&item.extras_json)?;
    for (i, (key, value)) in [
        ("step", json!(item.step)), ("action", json!(item.action)),
        ("reward", json!(item.reward)), ("done", json!(item.done)),
        ("cursor_x", json!(item.cursor_x)), ("cursor_y", json!(item.cursor_y)),
        ("cursor_z", json!(item.cursor_z)), ("voxel_count", json!(item.voxel_count)),
        ("placed", json!(item.placed)),
    ].into_iter().enumerate() {
        if item.presence & (1 << i) != 0 { out.insert(key.into(), value); }
    }
    let variables = [
        ("mutations", Value::Array(item.mutations.iter().map(mutation).collect::<Result<_>>()?)),
        ("actions", json!(item.actions)), ("rewards", json!(item.rewards)),
        ("cursors", json!(item.cursors.iter().map(|c| &c.values).collect::<Vec<_>>())),
        ("placed_per_agent", json!(item.placed_per_agent)), ("dones", json!(item.dones)),
    ];
    for (i, (key, value)) in variables.into_iter().enumerate() {
        if item.presence & (1 << (i + 9)) != 0 { out.insert(key.into(), value); }
    }
    Ok(Value::Object(out))
}

fn bytes(path: &Path, compressed: bool) -> Result<Vec<u8>> {
    let raw = fs::read(path)?;
    Ok(if compressed { zstd::stream::decode_all(&raw[..])? } else { raw })
}

fn fixed_step(raw: &[u8]) -> Result<Value> {
    if raw.len() != 44 { return Err("invalid record length".into()); }
    let mask = u16::from_le_bytes(raw[0..2].try_into()?);
    let fields = [
        ("step", json!(i64::from_le_bytes(raw[2..10].try_into()?))),
        ("action", json!(i32::from_le_bytes(raw[10..14].try_into()?))),
        ("reward", json!(f64::from_le_bytes(raw[14..22].try_into()?))),
        ("done", json!(raw[22] != 0)),
        ("cursor_x", json!(u32::from_le_bytes(raw[23..27].try_into()?))),
        ("cursor_y", json!(u32::from_le_bytes(raw[27..31].try_into()?))),
        ("cursor_z", json!(u32::from_le_bytes(raw[31..35].try_into()?))),
        ("voxel_count", json!(u64::from_le_bytes(raw[35..43].try_into()?))),
        ("placed", json!(raw[43] != 0)),
    ];
    let mut out = Map::new();
    for (i, (key, value)) in fields.into_iter().enumerate() {
        if mask & (1 << i) != 0 { out.insert(key.into(), value); }
    }
    Ok(Value::Object(out))
}

pub fn read(dir: &Path, format: &str) -> Result<Value> {
    let compressed = format.ends_with("zstd");
    if format.starts_with("json") {
        return Ok(serde_json::from_slice(&bytes(&dir.join("trajectory.data"), compressed)?)?);
    }
    if format.starts_with("protobuf") {
        let msg = Trajectory::parse_from_bytes(&bytes(&dir.join("trajectory.data"), compressed)?)?;
        if msg.version != 1 { return Err("unsupported protobuf version".into()); }
        let mut payload: Value = serde_json::from_slice(&msg.metadata_json)?;
        payload["episode"]["steps"] = Value::Array(msg.steps.iter().map(step).collect::<Result<_>>()?);
        return Ok(payload);
    }
    if format != "binary" && format != "binary-zstd" { return Err("unknown format".into()); }
    let manifest: Value = serde_json::from_slice(&fs::read(dir.join("manifest.json"))?)?;
    let records = bytes(&dir.join("steps.bin"), compressed)?;
    let offsets = bytes(&dir.join("offsets.bin"), compressed)?;
    let events = bytes(&dir.join("events.pb"), compressed)?;
    if records.len() < 16 || &records[..4] != b"TSB1" || records[4..8] != 1u32.to_le_bytes()
        || manifest["benchmark_version"] != 1 || manifest["record_bytes"] != 44 {
        return Err("invalid binary header".into());
    }
    let count = usize::try_from(u64::from_le_bytes(records[8..16].try_into()?))?;
    if records.len().checked_sub(16) != count.checked_mul(44)
        || offsets.len() != count.checked_add(1).and_then(|n| n.checked_mul(8)).ok_or("index overflow")?
        || manifest["step_count"].as_u64() != Some(count as u64) {
        return Err("invalid binary lengths".into());
    }
    let index = offsets.chunks_exact(8).map(|b| usize::try_from(u64::from_le_bytes(b.try_into().unwrap())))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if index.first() != Some(&0) || index.last() != Some(&events.len()) || index.windows(2).any(|p| p[0] > p[1]) {
        return Err("invalid event index".into());
    }
    let mut steps = Vec::with_capacity(count);
    for (i, record) in records[16..].chunks_exact(44).enumerate() {
        let mut value = step(&Step::parse_from_bytes(&events[index[i]..index[i+1]])?)?;
        value.as_object_mut().ok_or("invalid event")?.extend(fixed_step(record)?.as_object().unwrap().clone());
        steps.push(value);
    }
    let mut payload = manifest["metadata"].clone();
    payload["episode"]["steps"] = Value::Array(steps);
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn record_width_and_large_coordinates() {
        let mut raw = [0u8; 44];
        raw[..2].copy_from_slice(&(1u16 << 4).to_le_bytes());
        raw[23..27].copy_from_slice(&70_001u32.to_le_bytes());
        assert_eq!(fixed_step(&raw).unwrap(), json!({"cursor_x": 70001}));
        assert!(fixed_step(&raw[..43]).is_err());
    }
    #[test]
    fn protobuf_presence_and_events() {
        let mut item = Step::new();
        item.presence = (1 << 1) | (1 << 9);
        assert_eq!(step(&item).unwrap(), json!({"action":0,"mutations":[]}));
    }
}
