use std::{fs, path::PathBuf};

use theseo_core::verification::benchmark::{self, BenchmarkConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut smoke = false;
    let mut baseline = false;
    let mut output = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--smoke" => smoke = true,
            "--baseline" => baseline = true,
            "--output" => output = arguments.next().map(PathBuf::from),
            _ => return Err(format!("unknown argument: {argument}").into()),
        }
    }
    let config = if smoke {
        BenchmarkConfig::smoke()
    } else if baseline {
        BenchmarkConfig::baseline()
    } else {
        BenchmarkConfig::full()
    };
    let json = serde_json::to_string_pretty(&benchmark::run(&config))?;
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, &json)?;
        println!("benchmark report written to {}", path.display());
    } else {
        println!("{json}");
    }
    Ok(())
}
