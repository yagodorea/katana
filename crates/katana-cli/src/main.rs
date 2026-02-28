use std::fs;
use std::time::Instant;

use clap::{Parser, Subcommand};
use katana_core::{offset, slicer, stl, svg};
use rayon::prelude::*;

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
        /// Nozzle diameter in mm
        #[arg(short, long, default_value_t = 0.4)]
        nozzle_width: f32,
        /// Number of perimeter walls
        #[arg(short, long, default_value_t = 3)]
        perimeters: usize,
        /// Infill density (0.0 = hollow, 1.0 = solid)
        #[arg(short = 'd', long, default_value_t = 0.2)]
        infill_density: f32,
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
            nozzle_width,
            perimeters,
            infill_density,
        } => cmd_slice(&file, layer_height, &output, nozzle_width, perimeters, infill_density),
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

fn cmd_slice(file: &str, layer_height: f32, output_dir: &str, nozzle_width: f32, perimeters: usize, infill_density: f32) {
    let t_load = Instant::now();
    let mesh = load_mesh(file);
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
    let (min, max) = mesh.bounding_box();

    println!("Slicing: {file}");
    println!("  Triangles: {} (loaded in {:.1}ms)", mesh.triangles.len(), load_ms);
    println!("  Layer height: {layer_height} mm");
    println!("  Nozzle: {nozzle_width} mm, {perimeters} perimeters, {:.0}% infill", infill_density * 100.0);
    println!(
        "  Z range: {:.3} to {:.3}",
        min.z, max.z
    );

    let t_slice = Instant::now();
    let result = slicer::slice_mesh(&mesh, layer_height);
    let slice_ms = t_slice.elapsed().as_secs_f64() * 1000.0;

    println!("  Layers: {} (sliced in {:.1}ms)", result.layers.len(), slice_ms);

    let perim_config = offset::PerimeterConfig {
        nozzle_width,
        perimeter_count: perimeters,
    };
    let infill_config = offset::InfillConfig {
        density: infill_density,
        nozzle_width,
    };

    let t_offset = Instant::now();
    let toolpath_result = offset::generate_toolpaths(&result, &perim_config, &infill_config);
    let offset_ms = t_offset.elapsed().as_secs_f64() * 1000.0;

    println!("  Perimeters generated in {:.1}ms", offset_ms);

    let t_svg = Instant::now();
    fs::create_dir_all(output_dir).unwrap_or_else(|e| {
        eprintln!("Failed to create output directory: {e}");
        std::process::exit(1);
    });

    let output_dir_owned = output_dir.to_string();
    toolpath_result
        .layers
        .par_iter()
        .zip(result.layers.par_iter())
        .enumerate()
        .for_each(|(i, (tp_layer, orig_layer))| {
            let svg_content = svg::toolpath_layer_to_svg(tp_layer, orig_layer, 2.0);
            let path = format!("{output_dir_owned}/layer_{i:04}.svg");
            fs::write(&path, &svg_content).unwrap_or_else(|e| {
                eprintln!("Failed to write {path}: {e}");
                std::process::exit(1);
            });
        });
    let svg_ms = t_svg.elapsed().as_secs_f64() * 1000.0;

    println!("  SVGs written to: {output_dir}/ ({:.1}ms)", svg_ms);
}
