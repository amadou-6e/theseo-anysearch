//! Load once into memory: new compressed JSON and legacy plain JSON trajectories.
use std::{
    collections::BTreeMap,
    fs, io,
    path::{Path, PathBuf},
};

pub fn read_json(path: &Path) -> io::Result<Vec<u8>> {
    let raw = fs::read(path)?;
    if path.to_string_lossy().ends_with(".json.zst") {
        zstd::stream::decode_all(&raw[..])
    } else {
        Ok(raw)
    }
}

pub fn find_trajectory(directory: &Path, stem: &str) -> Option<PathBuf> {
    [".json.zst", ".json"]
        .into_iter()
        .map(|suffix| directory.join(format!("{stem}{suffix}")))
        .find(|path| path.is_file())
}

pub fn iteration_files(directory: &Path) -> Vec<PathBuf> {
    let mut paths = BTreeMap::new();
    let Ok(entries) = fs::read_dir(directory) else {
        return Vec::new();
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let stem = name
            .strip_suffix(".json.zst")
            .or_else(|| name.strip_suffix(".json"));
        if let Some(stem) = stem.filter(|stem| stem.starts_with("iter_")) {
            if !paths.contains_key(stem) || name.ends_with(".zst") {
                paths.insert(stem.to_owned(), path);
            }
        }
    }
    paths.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn directory() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "theseo-storage-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&path).unwrap();
        path
    }

    #[test]
    fn reads_both_formats_and_rejects_truncation() {
        let dir = directory();
        let raw =
            br#"{"schema_version":2,"episode":{"steps":[{"cursor_x":70001,"reward":0.125}]}}"#;
        fs::write(dir.join("iter_000001.json"), raw).unwrap();
        let compressed = zstd::stream::encode_all(&raw[..], 3).unwrap();
        fs::write(dir.join("iter_000002.json.zst"), &compressed).unwrap();
        assert_eq!(read_json(&dir.join("iter_000001.json")).unwrap(), raw);
        assert_eq!(read_json(&dir.join("iter_000002.json.zst")).unwrap(), raw);
        fs::write(
            dir.join("broken.json.zst"),
            &compressed[..compressed.len() - 1],
        )
        .unwrap();
        assert!(read_json(&dir.join("broken.json.zst")).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn deduplicates_iterations_and_prefers_compressed_best() {
        let dir = directory();
        for name in [
            "iter_000002.json",
            "iter_000002.json.zst",
            "iter_000001.json",
            "best.json",
            "best.json.zst",
            "best_meta.json",
            "iter_000003.json.zst.tmp",
        ] {
            fs::write(dir.join(name), b"{}").unwrap();
        }
        assert_eq!(
            iteration_files(&dir),
            vec![
                dir.join("iter_000001.json"),
                dir.join("iter_000002.json.zst")
            ]
        );
        assert_eq!(
            find_trajectory(&dir, "best"),
            Some(dir.join("best.json.zst"))
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn reads_python_zstandard_frame() {
        // Produced by Python zstandard 0.25.0, level 3, compact UTF-8 JSON.
        let frame = [
            40, 181, 47, 253, 32, 148, 253, 3, 0, 114, 200, 28, 30, 96, 181, 206, 1, 176, 207, 2,
            109, 50, 150, 161, 28, 87, 229, 218, 255, 65, 16, 216, 37, 242, 125, 251, 75, 119, 67,
            65, 16, 197, 5, 49, 46, 113, 137, 218, 6, 70, 200, 59, 125, 15, 212, 218, 62, 45, 70,
            42, 148, 188, 66, 230, 60, 199, 187, 206, 62, 200, 32, 212, 60, 33, 200, 97, 20, 196,
            82, 3, 65, 130, 28, 70, 65, 172, 128, 202, 59, 102, 0, 202, 79, 75, 41, 149, 228, 3,
            24, 222, 102, 168, 61, 82, 242, 253, 40, 236, 71, 121, 117, 219, 239, 209, 39, 34, 239,
            236, 107, 214, 3, 195, 26, 45, 180, 126, 20, 2, 0, 66, 24, 9, 254, 76, 153, 1,
        ];
        let dir = directory();
        let path = dir.join("python.json.zst");
        fs::write(&path, frame).unwrap();
        let raw = read_json(&path).unwrap();
        assert_eq!(raw, br#"{"schema_version":2,"episode":{"steps":[{"cursor_x":70001,"reward":0.12345678901234566,"mutations":[{"coordinate":[70001,3,4],"occupied":false}]}]}}"#);
        fs::remove_dir_all(dir).unwrap();
    }
}
