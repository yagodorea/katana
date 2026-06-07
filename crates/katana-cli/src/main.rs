use std::fs;
use std::time::Instant;

use clap::{ Parser, Subcommand };
use katana_core::{ gcode::{ self, GcodeConfig }, offset, slicer, stl, planner };

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
        /// Number of bottom solid layers
        #[arg(long, default_value_t = 3)]
        bottom_layers: usize,
        /// Number of top solid layers
        #[arg(long, default_value_t = 3)]
        top_layers: usize,
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
            bottom_layers,
            top_layers,
        } =>
            cmd_slice(
                &file,
                layer_height,
                &output,
                nozzle_width,
                perimeters,
                infill_density,
                bottom_layers,
                top_layers
            ),
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
    println!("    size: {:.3} x {:.3} x {:.3}", max.x - min.x, max.y - min.y, max.z - min.z);
    println!("  Volume: {:.3}", mesh.volume());
}

fn cmd_slice(
    file: &str,
    layer_height: f32,
    output_dir: &str,
    nozzle_width: f32,
    perimeters: usize,
    infill_density: f32,
    bottom_layers: usize,
    top_layers: usize
) {
    let t_load = Instant::now();
    let mesh = load_mesh(file);
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
    let (min, max) = mesh.bounding_box();

    println!("Slicing: {file}");
    println!("  Triangles: {} (loaded in {:.1}ms)", mesh.triangles.len(), load_ms);
    println!("  Layer height: {layer_height} mm");
    println!(
        "  Nozzle: {nozzle_width} mm, {perimeters} perimeters, {:.0}% infill",
        infill_density * 100.0
    );
    println!("  Bottom layers: {bottom_layers}, Top layers: {top_layers}");
    println!("  Z range: {:.3} to {:.3}", min.z, max.z);

    let t_slice = Instant::now();
    let result = slicer::slice_mesh(&mesh, layer_height);
    let slice_ms = t_slice.elapsed().as_secs_f64() * 1000.0;

    println!("  Layers: {} (sliced in {:.1}ms)", result.layers.len(), slice_ms);

    let perim_config = offset::PerimeterConfig {
        nozzle_width,
        perimeter_count: perimeters,
        layer_height,
    };
    let infill_config = offset::InfillConfig {
        density: infill_density,
        nozzle_width,
    };
    let surface_config = offset::SurfaceConfig {
        bottom_layers,
        top_layers,
    };

    let t_offset = Instant::now();
    let toolpath_result = offset::generate_toolpaths(
        &result,
        &perim_config,
        &infill_config,
        &surface_config
    );
    let offset_ms = t_offset.elapsed().as_secs_f64() * 1000.0;

    println!("  Perimeters generated in {:.1}ms", offset_ms);

    let t_plan = Instant::now();
    let speed_config = planner::SpeedConfig {
        travel: 150.0,
        perimeter: 30.0,
        infill: 60.0,
        surface: 40.0,
    };
    let planned_result = planner::plan_toolpaths(&toolpath_result, &speed_config);
    let plan_ms = t_plan.elapsed().as_secs_f64() * 1000.0;

    println!("  Planned result generated in {:.1}ms", plan_ms);

    let t_gcode = Instant::now();
    let mut exporter = gcode::Gcode {
        e: 0.0,
        config: GcodeConfig {
            filament_diameter: 1.75,
            nozzle_width,
            layer_height,
        },
        out: String::new(),
    };
    let out = exporter.export(&planned_result);
    let gcode_file = file.replace(".stl", ".gcode");
    fs::write(gcode_file, out).unwrap_err();
    let gcode_ms = t_gcode.elapsed().as_secs_f64() * 1000.0;

    println!("  G-code written to: {output_dir}/ ({:.1}ms)", gcode_ms);
}
