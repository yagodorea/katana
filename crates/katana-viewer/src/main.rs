// Native entry point. On WASM the entry is in lib.rs (start_web).
// This file is compiled for both targets but the native code is cfg-gated.

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::sync::{ Arc, Mutex };

    use clap::Parser;
    use eframe::egui;
    use katana_viewer::{ renderer_wgpu_port, Phase, ViewerApp };

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

                let gpu = renderer_wgpu_port::Renderer::new(device, queue, target_format, 1, 1);
                let renderer = Arc::new(Mutex::new(gpu));

                let mut app = ViewerApp::new(renderer);
                app.zoom = 1.3;
                app.nozzle_width = args.nozzle_width;
                app.layer_height = args.layer_height;
                app.perimeters = args.perimeters;
                app.infill_density = args.infill_density;
                app.bottom_layers = args.bottom_layers;
                app.top_layers = args.top_layers;
                app.travel_speed = args.travel_speed;
                app.perimeter_speed = args.perimeter_speed;
                app.infill_speed = args.infill_speed;
                app.surface_speed = args.surface_speed;

                if let Some(ref file) = args.file {
                    app.import_stl(file);
                    if app.phase == Phase::Import {
                        // import_stl already printed the error
                        std::process::exit(1);
                    }
                }

                Ok(Box::new(app))
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
