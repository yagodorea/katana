use std::fs;

use clap::{Parser, Subcommand};
use katana_core::{slicer, stl, svg};

#[derive(Parser)]
#[command(name = "katana", about = "3D printing slicer")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Inspect an STL file (triangle count, bounding box, volume)
    Info {
        /// Path to an STL file
        file: String,
    },
    /// Slice an STL file and output SVG layers
    Slice {
        /// Path to an STL file
        file: String,
        /// Layer height in mm
        #[arg(short, long, default_value_t = 0.2)]
        layer_height: f32,
        /// Output directory for SVG files
        #[arg(short, long, default_value = "output")]
        output: String,
    },
}

fn main() {
    let args = Args::parse();

    match args.command {
        Command::Info { file } => cmd_info(&file),
        Command::Slice {
            file,
            layer_height,
            output,
        } => cmd_slice(&file, layer_height, &output),
    }
}

fn load_mesh(path: &str) -> katana_core::mesh::Mesh {
    let data = fs::read(path).unwrap_or_else(|e| {
        eprintln!("Failed to read {path}: {e}");
        std::process::exit(1);
    });
    stl::load_stl(&data).unwrap_or_else(|e| {
        eprintln!("Failed to parse STL: {e}");
        std::process::exit(1);
    })
}

fn cmd_info(file: &str) {
    let mesh = load_mesh(file);
    let (min, max) = mesh.bounding_box();

    println!("Loaded: {file}");
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

fn cmd_slice(file: &str, layer_height: f32, output_dir: &str) {
    let mesh = load_mesh(file);
    let (min, max) = mesh.bounding_box();

    println!("Slicing: {file}");
    println!("  Layer height: {layer_height} mm");
    println!(
        "  Z range: {:.3} to {:.3}",
        min.z, max.z
    );

    let result = slicer::slice_mesh(&mesh, layer_height);

    println!("  Layers: {}", result.layers.len());

    fs::create_dir_all(output_dir).unwrap_or_else(|e| {
        eprintln!("Failed to create output directory: {e}");
        std::process::exit(1);
    });

    for (i, layer) in result.layers.iter().enumerate() {
        let svg_content = svg::layer_to_svg(layer, 2.0);
        let path = format!("{output_dir}/layer_{i:04}.svg");
        fs::write(&path, &svg_content).unwrap_or_else(|e| {
            eprintln!("Failed to write {path}: {e}");
            std::process::exit(1);
        });
    }

    println!("  SVGs written to: {output_dir}/");
}
