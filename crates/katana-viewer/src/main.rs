use std::fs;
use std::sync::{ Arc, Mutex };
use std::time::{ Duration, Instant };

use clap::Parser;
use eframe::egui;
use eframe::egui_wgpu;
use katana_core::{ gcode::{ self, Gcode, GcodeConfig }, offset, planner, slicer, stl };
use nalgebra::{ Point2, Vector2 };

mod renderer_wgpu_port;

#[derive(Parser)]
#[command(name = "katana-viewer", about = "2D layer viewer for sliced meshes")]
struct Args {
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

fn main() -> eframe::Result {
    let args = Args::parse();

    // Determine starting phase: if a file was provided, load it now (Model phase);
    // otherwise start at Import phase with an empty viewer.
    let initial_phase;
    let mut initial_triangles: Option<Vec<katana_core::mesh::Triangle>> = None;
    let mut initial_source = String::new();
    let mut initial_gcode_offset = Vector2::zeros();
    let mut initial_mesh_min = nalgebra::Point3::origin();
    let mut initial_mesh_max = nalgebra::Point3::origin();
    let mut initial_center = [0.0f32, 0.0, 128.0]; // center of a 256mm bed
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
        let mesh = stl::load_stl(&data).unwrap_or_else(|e| {
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
                        // Bump buffer max size for bigger STLs
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

    // Move data into the closure for eframe::run_native
    let start_triangles = initial_triangles;

    eframe::run_native(
        "katana viewer",
        options,
        Box::new(move |cc| {
            let render_state = cc.wgpu_render_state.as_ref().expect("eframe wgpu backend required");
            let device = render_state.device.clone();
            let queue = render_state.queue.clone();
            let target_format = render_state.target_format;

            let mut gpu = renderer_wgpu_port::Renderer::new(device, queue, target_format, 1, 1);

            // If a file was provided at startup, upload the mesh now
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

#[derive(PartialEq, Clone, Copy)]
pub enum BgMode {
    None,
    Mesh,
    Layers,
}

#[derive(PartialEq, Clone, Copy)]
enum SliceView {
    Contours,
    Toolpaths,
}

#[derive(PartialEq, Clone, Copy)]
enum Phase {
    Import, // no model loaded — show Import button
    Model, // mesh loaded — show mesh + Slice button
    Sliced, // slicing done — full viewer + Export button
}

struct Stats {
    triangles: usize,
    load_ms: f64,
    slice_ms: f64,
    offset_ms: f64,
    plan_ms: f64,
}

struct ViewerApp {
    phase: Phase,
    renderer: Arc<Mutex<renderer_wgpu_port::Renderer>>,

    // Mesh data (populated on import)
    mesh_triangles: Option<Vec<katana_core::mesh::Triangle>>,
    mesh_min: nalgebra::Point3<f32>,
    mesh_max: nalgebra::Point3<f32>,
    /// Source STL path, used to suggest a default G-code filename.
    source_file: String,
    gcode_offset: Vector2<f32>,

    // Slicing results (populated on slice)
    planned_result: Option<planner::PlannedResult>,
    layers: Vec<slicer::Layer>,
    num_layers: usize,
    max_layer: usize,
    /// `max_layer` from the previous frame, to detect layer change and reset the scrubber
    prev_max_layer: usize,
    min_layer: usize,
    slice_view: SliceView,
    center: [f32; 3],
    extent: f32,
    azimuth: f32,
    elevation: f32,
    zoom: f32,
    pan: egui::Vec2,
    bg_mode: BgMode,
    show_travel_moves: bool,
    show_filaments: bool,
    scrub: f32,
    stats: Stats,
    fps: f32,
    frame_time: f32,
    last_update: Instant,
    frame_count: u32,

    // Slicer config (from CLI args, used when Slice is clicked)
    nozzle_width: f32,
    layer_height: f32,
    perimeters: usize,
    infill_density: usize,
    bottom_layers: usize,
    top_layers: usize,
    travel_speed: f32,
    perimeter_speed: f32,
    infill_speed: f32,
    surface_speed: f32,
}

impl ViewerApp {
    /// Open a native save dialog and write the planned toolpath as G-code.
    fn export_gcode(&self) {
        let Some(ref planned) = self.planned_result else {
            return;
        };

        // Suggest "<model>.gcode" derived from the source STL's stem.
        let default_name = std::path::Path
            ::new(&self.source_file)
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| format!("{s}.gcode"))
            .unwrap_or_else(|| "output.gcode".to_string());

        // Blocking rfd file dialog, talks to OS filesystem
        let Some(path) = rfd::FileDialog
            ::new()
            .set_file_name(default_name)
            .add_filter("G-code", &["gcode"])
            .save_file() else {
            return; // user cancelled the dialog
        };

        let mut exporter = Gcode {
            config: GcodeConfig {
                filament_diameter: 1.75,
                nozzle_width: self.nozzle_width,
                layer_height: self.layer_height,
            },
            offset: self.gcode_offset,
            // TODO: remove these guys from the constructor
            e: 0.0,
            out: String::new(),
        };
        let out = exporter.export(planned);
        if let Err(e) = fs::write(path, out) {
            eprintln!("Error writing to file! {}", e.to_string());
        }
    }

    /// Load an STL file from a path, upload mesh to GPU, transition to Model phase.
    fn import_stl(&mut self, path: &str) {
        let t_load = Instant::now();
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Failed to read {path}: {e}");
                return;
            }
        };
        let mesh = match stl::load_stl(&data) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Failed to parse STL: {e}");
                return;
            }
        };
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

        // Upload mesh to GPU
        {
            let mut r = self.renderer.lock().unwrap();
            r.upload_mesh(&mesh.triangles);
            // Clear any previous slicing data
            r.slices_buffer = None;
            r.current_slice_buffer = None;
            r.toolpath_lines_buffer = None;
            r.toolpath_path_lines_buffer = None;
            r.toolpath_rhombuses = None;
        }

        println!("Loaded {path} ({num_triangles} triangles) in {load_ms:.1}ms");

        self.mesh_triangles = Some(mesh.triangles);
        self.mesh_min = mesh_min;
        self.mesh_max = mesh_max;
        self.source_file = path.to_string();
        self.gcode_offset = gcode_offset;
        self.center = [center_x, center_y, center_z];
        self.extent = extent;
        self.stats = Stats {
            triangles: num_triangles,
            load_ms,
            slice_ms: 0.0,
            offset_ms: 0.0,
            plan_ms: 0.0,
        };
        self.phase = Phase::Model;
    }

    /// Run the full slicing pipeline and upload results to GPU.
    fn run_slice(&mut self) {
        let Some(ref triangles) = self.mesh_triangles else {
            return;
        };
        let mesh = katana_core::mesh::Mesh {
            triangles: triangles.clone(),
            source: katana_core::mesh::MeshSource::StlBinary, // doesn't matter for slicing
        };

        let t_slice = Instant::now();
        let result = slicer::slice_mesh(&mesh, self.layer_height);
        let slice_ms = t_slice.elapsed().as_secs_f64() * 1000.0;

        let perim_config = offset::PerimeterConfig {
            nozzle_width: self.nozzle_width,
            perimeter_count: self.perimeters,
            layer_height: self.layer_height,
        };
        let infill_config = offset::InfillConfig {
            density: (self.infill_density as f32) / 100.0,
            nozzle_width: self.nozzle_width,
        };
        let surface_config = offset::SurfaceConfig {
            bottom_layers: self.bottom_layers,
            top_layers: self.top_layers,
        };

        let t_offset = Instant::now();
        let toolpath_result = offset::generate_toolpaths(
            &result,
            &perim_config,
            &infill_config,
            &surface_config
        );
        let offset_ms = t_offset.elapsed().as_secs_f64() * 1000.0;

        let speed_config = planner::SpeedConfig {
            travel: self.travel_speed,
            perimeter: self.perimeter_speed,
            infill: self.infill_speed,
            surface: self.surface_speed,
        };

        let t_plan = Instant::now();
        let planned_result = planner::plan_toolpaths(&toolpath_result, &speed_config);
        let plan_ms = t_plan.elapsed().as_secs_f64() * 1000.0;

        let layers = result.layers;
        let num_layers = layers.len();
        let last_layer = num_layers.saturating_sub(1);

        println!(
            "Sliced {num_layers} layers in {slice_ms:.1}ms, perimeters in {offset_ms:.1}ms, planning in {plan_ms:.1}ms"
        );

        // Upload slicing results to GPU
        {
            let mut r = self.renderer.lock().unwrap();
            r.upload_all_slices(&layers, 1);
            r.upload_current_slice(&layers);
            r.upload_planned_toolpath(&planned_result.layers, self.nozzle_width, self.layer_height);
            if !layers.is_empty() {
                r.clip_z_max = layers[last_layer].z + 0.001;
                r.clip_z_min = layers[0].z - 0.001;
            }
        }

        self.layers = layers;
        self.num_layers = num_layers;
        self.max_layer = last_layer;
        self.prev_max_layer = last_layer;
        self.min_layer = 0;
        self.planned_result = Some(planned_result);
        self.slice_view = SliceView::Toolpaths;
        self.bg_mode = BgMode::Mesh;
        self.stats.slice_ms = slice_ms;
        self.stats.offset_ms = offset_ms;
        self.stats.plan_ms = plan_ms;
        self.phase = Phase::Sliced;
    }
}

impl eframe::App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Update FPS counter (manually tracked)
        self.frame_count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update);
        if elapsed >= Duration::from_secs(1) {
            self.fps = (self.frame_count as f32) / elapsed.as_secs_f32();
            self.frame_time = if self.fps > 0.0 { 1000.0 / self.fps } else { 0.0 };
            self.last_update = now;
            self.frame_count = 0;
        }

        match self.phase {
            Phase::Import => self.update_import(ctx),
            Phase::Model => self.update_model(ctx),
            Phase::Sliced => self.update_sliced(ctx),
        }

        // FPS counter (bottom-right corner)
        egui::Area
            ::new(egui::Id::new("fps_counter"))
            .anchor(egui::Align2::RIGHT_BOTTOM, [10.0, 10.0])
            .show(ctx, |ui| {
                ui.colored_label(
                    egui::Color32::YELLOW,
                    format!("{:.1} FPS ({:.1} ms)", self.fps, self.frame_time)
                );
            });
    }

    // No on_exit needed: wgpu resources clean up via Drop.
}

impl ViewerApp {
    /// Phase 1: Empty screen with a centered Import button.
    fn update_import(&mut self, ctx: &egui::Context) {
        egui::CentralPanel
            ::default()
            .frame(egui::Frame::NONE.fill(egui::Color32::from_rgb(26, 26, 46)))
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(ui.available_height() / 3.0);
                    ui.heading(
                        egui::RichText
                            ::new("Katana")
                            .size(36.0)
                            .color(egui::Color32::from_rgb(200, 200, 220))
                    );
                    ui.add_space(24.0);
                    let import_btn = ui.add_sized(
                        [200.0, 48.0],
                        egui::Button
                            ::new(egui::RichText::new("📂 Import STL").size(20.0))
                            .fill(egui::Color32::from_rgb(60, 60, 90))
                    );
                    if import_btn.clicked() {
                        if
                            let Some(path) = rfd::FileDialog
                                ::new()
                                .add_filter("STL", &["stl"])
                                .pick_file()
                        {
                            if let Some(path_str) = path.to_str() {
                                self.import_stl(path_str);
                            }
                        }
                    }
                });
            });
    }

    /// Phase 2: Mesh viewport with Slice button in top panel.
    fn update_model(&mut self, ctx: &egui::Context) {
        // Top panel: mesh info + Slice button
        egui::TopBottomPanel::top("model_info").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("{} triangles", self.stats.triangles));
                ui.label(format!("Load: {:.0}ms", self.stats.load_ms));
                ui.separator();
                ui.label(&self.source_file);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if
                        ui
                            .add_sized(
                                [120.0, 28.0],
                                egui::Button
                                    ::new(egui::RichText::new("🔪 Slice").size(16.0))
                                    .fill(egui::Color32::from_rgb(80, 50, 50))
                            )
                            .clicked()
                    {
                        self.run_slice();
                    }
                });
            });
        });

        self.update_viewport(ctx);
    }

    /// Phase 3: Full viewer with layer controls, scrubber, and export.
    fn update_sliced(&mut self, ctx: &egui::Context) {
        // Top panel: navigation + export
        egui::TopBottomPanel::top("info").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if self.num_layers == 0 {
                    ui.label("No layers");
                    return;
                }
                if ui.button("◀ Prev").clicked() && self.max_layer > 0 {
                    self.max_layer -= 1;
                }
                if
                    ui.button("Next ▶").clicked() &&
                    self.max_layer < self.num_layers.saturating_sub(1)
                {
                    self.max_layer += 1;
                }
                ui.separator();
                if ui.button("💾 Export G-code").clicked() {
                    self.export_gcode();
                }
            });
        });

        // Left sidebar: rendering options (collapsible, collapsed by default)
        egui::SidePanel
            ::left("rendering_options")
            .resizable(false)
            .exact_width(160.0)
            .show(ctx, |ui| {
                if self.num_layers == 0 {
                    return;
                }
                ui.collapsing("🎨 Rendering options", |ui| {
                    ui.label("BG:");
                    ui.selectable_value(&mut self.bg_mode, BgMode::Mesh, "Mesh");
                    ui.selectable_value(&mut self.bg_mode, BgMode::Layers, "Layers");
                    ui.selectable_value(&mut self.bg_mode, BgMode::None, "None");
                    ui.separator();
                    ui.label("View:");
                    ui.selectable_value(&mut self.slice_view, SliceView::Contours, "Contours");
                    ui.selectable_value(&mut self.slice_view, SliceView::Toolpaths, "Toolpaths");
                    ui.separator();
                    ui.checkbox(&mut self.show_filaments, "3D filaments");
                    ui.checkbox(&mut self.show_travel_moves, "Travel moves");
                });
            });

        // Collapsible translucent stats window (top-right)
        if self.num_layers > 0 {
            let height = self.layers[self.num_layers.saturating_sub(1)].z;
            let top_z = self.layers[self.max_layer].z;
            let bottom_z = self.layers[self.min_layer].z;
            egui::Window::new("📊 Stats")
                .id(egui::Id::new("stats_window"))
                .collapsible(true)
                .default_open(false)
                .anchor(egui::Align2::RIGHT_TOP, [-8.0, 8.0])
                .resizable(false)
                .frame(
                    egui::Frame::NONE
                        .fill(egui::Color32::from_rgba_premultiplied(20, 20, 30, 200))
                        .corner_radius(6.0)
                        .inner_margin(8.0)
                )
                .show(ctx, |ui| {
                    ui.label(format!("{} layers · {:.3} mm", self.num_layers, height));
                    ui.label(format!("Top    layer {} · z {:.3} mm", self.max_layer, top_z));
                    ui.label(format!("Bottom layer {} · z {:.3} mm", self.min_layer, bottom_z));
                    ui.separator();
                    ui.label(
                        format!(
                            "{} tris · load {:.0}ms · slice {:.0}ms · offset {:.0}ms · plan {:.0}ms",
                            self.stats.triangles,
                            self.stats.load_ms,
                            self.stats.slice_ms,
                            self.stats.offset_ms,
                            self.stats.plan_ms
                        )
                    );
                });
        }

        // Left panel: top layer slider
        egui::SidePanel
            ::left("slider_top")
            .resizable(false)
            .exact_width(32.0)
            .show(ctx, |ui| {
                if self.num_layers == 0 {
                    return;
                }
                let max = self.num_layers.saturating_sub(1);
                ui.spacing_mut().slider_width = ui.available_height() - 16.0;
                ui.add(
                    egui::Slider
                        ::new(&mut self.max_layer, 0..=max)
                        .vertical()
                        .show_value(false)
                );
            });
        // Bottom layer
        egui::SidePanel
            ::left("slider_bottom")
            .resizable(false)
            .exact_width(32.0)
            .show(ctx, |ui| {
                if self.num_layers == 0 {
                    return;
                }
                let max = self.num_layers.saturating_sub(1);
                ui.spacing_mut().slider_width = ui.available_height() - 16.0;
                ui.add(
                    egui::Slider
                        ::new(&mut self.min_layer, 0..=max)
                        .vertical()
                        .show_value(false)
                );
            });

        // Bottom panel: horizontal scrubber for the top layer
        egui::TopBottomPanel
            ::bottom("scrubber")
            .resizable(false)
            .show(ctx, |ui| {
                if self.num_layers == 0 {
                    return;
                }
                ui.horizontal(|ui| {
                    ui.label("Layer progress");
                    ui.spacing_mut().slider_width = (ui.available_width() - 80.0).max(64.0);
                    ui.add(
                        egui::Slider
                            ::new(&mut self.scrub, 0.0..=1.0)
                            .custom_formatter(|v, _| format!("{:.0}%", v * 100.0))
                    );
                });
            });

        self.update_viewport(ctx);
    }

    /// Shared viewport rendering + camera controls for Model and Sliced phases.
    fn update_viewport(&mut self, ctx: &egui::Context) {
        egui::CentralPanel
            ::default()
            .frame(egui::Frame::NONE.fill(egui::Color32::from_rgb(26, 26, 46)))
            .show(ctx, |ui| {
                let (response, painter) = ui.allocate_painter(
                    ui.available_size(),
                    egui::Sense::click_and_drag()
                );

                if response.dragged_by(egui::PointerButton::Primary) {
                    let delta = response.drag_delta();
                    let command_pressed = ui.input(|i| (i.modifiers.command || i.modifiers.ctrl));
                    if command_pressed {
                        let ca = self.azimuth.cos();
                        let sa = self.azimuth.sin();
                        let ce = self.elevation.cos();
                        let se = self.elevation.sin();
                        let pan_world_scale = self.extent / (2.0 * self.zoom);
                        let right_x = ca;
                        let right_y = -sa;
                        let right_z = 0.0;
                        let up_x = -sa * se;
                        let up_y = ca * se;
                        let up_z = ce;
                        self.center[0] +=
                            (delta.x * right_x - delta.y * up_x) * pan_world_scale * 0.001;
                        self.center[1] +=
                            (delta.x * right_y - delta.y * up_y) * pan_world_scale * 0.001;
                        self.center[2] +=
                            (delta.x * right_z - delta.y * up_z) * pan_world_scale * 0.001;
                    } else {
                        self.azimuth -= delta.x * 0.005;
                        self.elevation = (self.elevation + delta.y * 0.005).clamp(
                            -std::f32::consts::FRAC_PI_2 + 0.01,
                            std::f32::consts::FRAC_PI_2 - 0.01
                        );
                    }
                }
                if
                    response.dragged_by(egui::PointerButton::Middle) ||
                    response.dragged_by(egui::PointerButton::Secondary)
                {
                    let delta = response.drag_delta();
                    let ca = self.azimuth.cos();
                    let sa = self.azimuth.sin();
                    let ce = self.elevation.cos();
                    let se = self.elevation.sin();
                    let pan_world_scale = self.extent / (2.0 * self.zoom);
                    let right_x = ca;
                    let right_y = -sa;
                    let right_z = 0.0;
                    let up_x = -sa * se;
                    let up_y = ca * se;
                    let up_z = ce;
                    self.center[0] +=
                        (delta.x * right_x - delta.y * up_x) * pan_world_scale * 0.001;
                    self.center[1] +=
                        (delta.x * right_y - delta.y * up_y) * pan_world_scale * 0.001;
                    self.center[2] +=
                        (delta.x * right_z - delta.y * up_z) * pan_world_scale * 0.001;
                }

                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll != 0.0 {
                    let factor = 1.0 + scroll * 0.002;
                    self.zoom = (self.zoom * factor).clamp(0.1, 50.0);
                }

                // Layer navigation (only meaningful in Sliced phase)
                if self.phase == Phase::Sliced {
                    ui.input(|i| {
                        if
                            i.key_pressed(egui::Key::ArrowUp) ||
                            i.key_pressed(egui::Key::ArrowRight)
                        {
                            if self.max_layer < self.num_layers.saturating_sub(1) {
                                self.max_layer += 1;
                            }
                        }
                        if
                            i.key_pressed(egui::Key::ArrowDown) ||
                            i.key_pressed(egui::Key::ArrowLeft)
                        {
                            if self.max_layer > 0 {
                                self.max_layer -= 1;
                            }
                        }
                        if i.key_pressed(egui::Key::Home) {
                            self.max_layer = 0;
                        }
                        if i.key_pressed(egui::Key::End) {
                            self.max_layer = self.num_layers.saturating_sub(1);
                        }
                    });

                    // Switching the top layer snaps the scrubber back to 1
                    if self.max_layer != self.prev_max_layer {
                        self.scrub = 1.0;
                        self.prev_max_layer = self.max_layer;
                    }
                }

                // Update renderer state (clip_z, draw mode) — no re-upload needed
                if !self.layers.is_empty() {
                    let mut r = self.renderer.lock().unwrap();
                    r.clip_z_max = self.layers[self.max_layer].z + 0.001;
                    r.clip_z_min = self.layers[self.min_layer].z - 0.001;
                    r.draw_contours = self.slice_view == SliceView::Contours;
                    r.draw_toolpaths = self.slice_view == SliceView::Toolpaths;
                    r.show_travel_moves = self.show_travel_moves;
                    r.show_filaments = self.show_filaments;
                    // Anything below full means we're actively scrubbing,
                    // which dims the layers beneath the top one to highlight it.
                    r.scrub_fraction = self.scrub;
                    r.is_scrubbing = self.scrub < 0.999;
                    r.scrub_top_z = self.layers[self.max_layer].z - 0.0001;
                }

                let rect = response.rect;
                let aspect = rect.width() / rect.height();
                let mvp = renderer_wgpu_port::build_mvp(
                    self.center,
                    self.azimuth,
                    self.elevation,
                    self.zoom,
                    self.extent,
                    aspect,
                    (self.pan.x, self.pan.y)
                );

                let bg_mode = if self.phase == Phase::Model { BgMode::Mesh } else { self.bg_mode };
                let light_dir = renderer_wgpu_port::headlight_dir(self.azimuth, self.elevation);
                let renderer = self.renderer.clone();
                let ppp = ctx.pixels_per_point();
                let vw = (rect.width() * ppp).max(1.0) as u32;
                let vh = (rect.height() * ppp).max(1.0) as u32;

                let callback = egui_wgpu::Callback::new_paint_callback(rect, ViewerCallback {
                    renderer,
                    mvp,
                    light_dir,
                    bg_mode,
                    width: vw,
                    height: vh,
                });
                painter.add(callback);
            });
    }
}

// ---------------------------------------------------------------------------
// PaintCallback bridge: ferries per-frame state from the egui side into our
// Renderer's prepare() and paint() methods via the egui_wgpu callback trait.
// ---------------------------------------------------------------------------

struct ViewerCallback {
    renderer: Arc<Mutex<renderer_wgpu_port::Renderer>>,
    mvp: [f32; 16],
    light_dir: [f32; 3],
    bg_mode: BgMode,
    width: u32,
    height: u32,
}

impl egui_wgpu::CallbackTrait for ViewerCallback {
    fn prepare(
        &self,
        device: &eframe::wgpu::Device,
        queue: &eframe::wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        encoder: &mut eframe::wgpu::CommandEncoder,
        _callback_resources: &mut egui_wgpu::CallbackResources
    ) -> Vec<eframe::wgpu::CommandBuffer> {
        self.renderer
            .lock()
            .unwrap()
            .prepare(
                device,
                queue,
                encoder,
                &self.mvp,
                &self.light_dir,
                self.bg_mode,
                self.width,
                self.height
            );
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut eframe::wgpu::RenderPass<'static>,
        _callback_resources: &egui_wgpu::CallbackResources
    ) {
        self.renderer.lock().unwrap().paint(render_pass);
    }
}
