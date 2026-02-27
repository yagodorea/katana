use std::fs;

use clap::Parser;
use katana_core::stl;

#[derive(Parser)]
#[command(name = "katana", about = "3D printing slicer")]
struct Args {
    /// Path to an STL file
    file: String,
}

fn main() {
    let args = Args::parse();

    let data = fs::read(&args.file).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {e}", args.file);
        std::process::exit(1);
    });

    let mesh = stl::load_stl(&data).unwrap_or_else(|e| {
        eprintln!("Failed to parse STL: {e}");
        std::process::exit(1);
    });

    let (min, max) = mesh.bounding_box();

    println!("Loaded: {}", args.file);
    println!("  Source Type: {}", mesh.source);
    println!("  Triangles: {}", mesh.triangles.len());
    println!("  Bounding box:");
    println!("    min: ({:.3}, {:.3}, {:.3})", min.x, min.y, min.z);
    println!("    max: ({:.3}, {:.3}, {:.3})", max.x, max.y, max.z);
    println!(
        "    size: {:.3} x {:.3} x {:.3}",
        max.x - min.x,
        max.y - min.y,
        max.z - min.z
    );
    println!("  Volume: {:.3}", mesh.volume());
}
