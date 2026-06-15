// Native entry point. On WASM the entry is in lib.rs (start_web).
// This file is compiled for both targets but the native code is cfg-gated.

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::sync::{ Arc, Mutex };
    use std::time::Instant;

    use clap::Parser;
    use eframe::egui;
    use katana_core::gcode;
    use katana_viewer::{ renderer_wgpu_port, BgMode, Phase, SliceView, Stats, ViewerApp };
    use nalgebra::{ Point2, Vector2 };

    #[derive(Parser)]
    #[command(name = "katana-viewer", about = "2D layer viewer for sliced meshes")]
    pub struct Args {
        /// Path to an STL file (optional — omit to start with Import dialog)
        file: Option<String>,
        /// Layer height in mm
        #[arg(short, long, default_value_t = 0.2)]
        layer_height: f32,
        /// Nozzle diameter in mm
        #[arg(short, long, default_value_t = 0.4)]
        nozzle_width: f32,
        /// Number of perimeter walls
        #[arg(short, long, default_value_t = 3)]
        perimeters: usize,
        /// Infill density %
        #[arg(short, long, default_value_t = 20)]
        infill_density: usize,
        /// Number of bottom solid layers
        #[arg(long, default_value_t = 3)]
        bottom_layers: usize,
        /// Number of top solid layers
        #[arg(long, default_value_t = 3)]
        top_layers: usize,
        /// Speed configs in mm/s
        #[arg(long, default_value_t = 150.0)]
        travel_speed: f32,
        #[arg(long, default_value_t = 30.0)]
        perimeter_speed: f32,
        #[arg(long, default_value_t = 60.0)]
        infill_speed: f32,
        #[arg(long, default_value_t = 40.0)]
        surface_speed: f32,
    }

    pub fn run() -> eframe::Result {
        let args = Args::parse();

        let initial_phase;
        let mut initial_triangles: Option<Vec<katana_core::mesh::Triangle>> = None;
        let mut initial_source = String::new();
        let mut initial_gcode_offset = Vector2::zeros();
        let mut initial_mesh_min = nalgebra::Point3::origin();
        let mut initial_mesh_max = nalgebra::Point3::origin();
        let mut initial_center = [0.0f32, 0.0, 128.0];
        let mut initial_extent = 256.0f32;
        let mut initial_stats = Stats {
            triangles: 0,
            load_ms: 0.0,
            slice_ms: 0.0,
            offset_ms: 0.0,
            plan_ms: 0.0,
        };

        if let Some(ref file) = args.file {
            let t_load = Instant::now();
            let data = std::fs::read(file).unwrap_or_else(|e| {
                eprintln!("Failed to read {file}: {e}");
                std::process::exit(1);
            });
            let mesh = katana_core::stl::load_stl(&data).unwrap_or_else(|e| {
                eprintln!("Failed to parse STL: {e}");
                std::process::exit(1);
            });
            let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

            let (mesh_min, mesh_max) = mesh.bounding_box();
            let num_triangles = mesh.triangles.len();

            let bed = gcode::BedConfig {
                width: 256.0,
                depth: 256.0,
            };
            let gcode_offset = gcode::bed_offset(
                bed,
                Point2::new(mesh_min.x, mesh_min.y),
                Point2::new(mesh_max.x, mesh_max.y)
            );

            let center_x = (mesh_min.x + mesh_max.x) / 2.0;
            let center_y = (mesh_min.y + mesh_max.y) / 2.0;
            let center_z = (mesh_min.z + mesh_max.z) / 2.0;
            let extent = (mesh_max.x - mesh_min.x)
                .max(mesh_max.y - mesh_min.y)
                .max(mesh_max.z - mesh_min.z);

            println!("Loaded {file} ({num_triangles} triangles) in {load_ms:.1}ms");

            initial_phase = Phase::Model;
            initial_triangles = Some(mesh.triangles);
            initial_source = file.clone();
            initial_gcode_offset = gcode_offset;
            initial_mesh_min = mesh_min;
            initial_mesh_max = mesh_max;
            initial_center = [center_x, center_y, center_z];
            initial_extent = extent;
            initial_stats = Stats {
                triangles: num_triangles,
                load_ms,
                slice_ms: 0.0,
                offset_ms: 0.0,
                plan_ms: 0.0,
            };
        } else {
            initial_phase = Phase::Import;
        }

        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 800.0]),
            renderer: eframe::Renderer::Wgpu,
            wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
                wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(
                    eframe::egui_wgpu::WgpuSetupCreateNew {
                        device_descriptor: Arc::new(|adapter| {
                            let mut limits = adapter.limits();
                            limits.max_buffer_size = 1 << 30; // 1 GiB
                            eframe::wgpu::DeviceDescriptor {
                                label: Some("katana-viewer device"),
                                required_features: eframe::wgpu::Features::empty(),
                                required_limits: limits,
                                memory_hints: eframe::wgpu::MemoryHints::default(),
                            }
                        }),
                        ..Default::default()
                    }
                ),
                ..Default::default()
            },
            ..Default::default()
        };

        let start_triangles = initial_triangles;

        eframe::run_native(
            "katana viewer",
            options,
            Box::new(move |cc| {
                let render_state = cc.wgpu_render_state
                    .as_ref()
                    .expect("eframe wgpu backend required");
                let device = render_state.device.clone();
                let queue = render_state.queue.clone();
                let target_format = render_state.target_format;

                let mut gpu = renderer_wgpu_port::Renderer::new(device, queue, target_format, 1, 1);

                if let Some(ref tris) = start_triangles {
                    gpu.upload_mesh(tris);
                }

                let renderer = Arc::new(Mutex::new(gpu));

                Ok(
                    Box::new(ViewerApp {
                        phase: initial_phase,
                        renderer,
                        mesh_triangles: start_triangles,
                        mesh_min: initial_mesh_min,
                        mesh_max: initial_mesh_max,
                        source_file: initial_source,
                        gcode_offset: initial_gcode_offset,
                        planned_result: None,
                        layers: Vec::new(),
                        num_layers: 0,
                        max_layer: 0,
                        prev_max_layer: 0,
                        min_layer: 0,
                        slice_view: SliceView::Toolpaths,
                        center: initial_center,
                        extent: initial_extent,
                        azimuth: std::f32::consts::FRAC_PI_4 + std::f32::consts::PI,
                        elevation: std::f32::consts::FRAC_PI_6,
                        zoom: 1.0,
                        pan: egui::Vec2::ZERO,
                        bg_mode: BgMode::Mesh,
                        stats: initial_stats,
                        show_travel_moves: false,
                        show_filaments: true,
                        scrub: 1.0,
                        fps: 0.0,
                        frame_time: 0.0,
                        last_update: Instant::now(),
                        frame_count: 0,
                        nozzle_width: args.nozzle_width,
                        layer_height: args.layer_height,
                        perimeters: args.perimeters,
                        infill_density: args.infill_density,
                        bottom_layers: args.bottom_layers,
                        top_layers: args.top_layers,
                        travel_speed: args.travel_speed,
                        perimeter_speed: args.perimeter_speed,
                        infill_speed: args.infill_speed,
                        surface_speed: args.surface_speed,
                    })
                )
            })
        )
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    native::run()
}

// WASM: no main needed — entry point is start_web() in lib.rs via #[wasm_bindgen(start)].
// This dummy main satisfies cargo when building the bin target for WASM.
#[cfg(target_arch = "wasm32")]
fn main() {}
