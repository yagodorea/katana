use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use clap::Parser;
use eframe::egui;
use eframe::egui_wgpu;
use katana_core::{offset, planner, slicer, stl};

mod renderer_wgpu_port;

#[derive(Parser)]
#[command(name = "katana-viewer", about = "2D layer viewer for sliced meshes")]
struct Args {
    /// Path to an STL file
    file: String,
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
}

fn main() -> eframe::Result {
    let args = Args::parse();

    let t_load = Instant::now();
    let data = std::fs::read(&args.file).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {e}", args.file);
        std::process::exit(1);
    });
    let mesh = stl::load_stl(&data).unwrap_or_else(|e| {
        eprintln!("Failed to parse STL: {e}");
        std::process::exit(1);
    });
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

    let (mesh_min, mesh_max) = mesh.bounding_box();
    let num_triangles = mesh.triangles.len();

    let t_slice = Instant::now();
    let result = slicer::slice_mesh(&mesh, args.layer_height);
    let slice_ms = t_slice.elapsed().as_secs_f64() * 1000.0;

    let perim_config = offset::PerimeterConfig {
        nozzle_width: args.nozzle_width,
        perimeter_count: args.perimeters,
        layer_height: args.layer_height,
    };
    let infill_config = offset::InfillConfig {
        density: args.infill_density as f32 / 100.0,
        nozzle_width: args.nozzle_width,
    };
    let surface_config = offset::SurfaceConfig {
        bottom_layers: args.bottom_layers,
        top_layers: args.top_layers,
    };

    let t_offset = Instant::now();
    let toolpath_result = offset::generate_toolpaths(&result, &perim_config, &infill_config, &surface_config);
    let offset_ms = t_offset.elapsed().as_secs_f64() * 1000.0;

    let t_plan = Instant::now();
    let planned_result = planner::plan_toolpaths(&toolpath_result);
    let plan_ms = t_plan.elapsed().as_secs_f64() * 1000.0;

    println!(
        "Loaded {} ({} triangles) in {:.1}ms",
        args.file, num_triangles, load_ms
    );
    println!(
        "Sliced {} layers in {:.1}ms, perimeters in {:.1}ms, planning in {:.1}ms",
        result.layers.len(),
        slice_ms,
        offset_ms,
        plan_ms,
    );

    let center_x = (mesh_min.x + mesh_max.x) / 2.0;
    let center_y = (mesh_min.y + mesh_max.y) / 2.0;
    let center_z = (mesh_min.z + mesh_max.z) / 2.0;
    let extent = (mesh_max.x - mesh_min.x)
        .max(mesh_max.y - mesh_min.y)
        .max(mesh_max.z - mesh_min.z);

    let triangles = mesh.triangles;
    let layers = result.layers;
    let num_layers = layers.len();

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
                },
            ),
            ..Default::default()
        },
        ..Default::default()
    };

    eframe::run_native(
        "katana viewer",
        options,
        Box::new(move |cc| {
            let render_state = cc.wgpu_render_state.as_ref()
                .expect("eframe wgpu backend required");
            let device = render_state.device.clone();   // Arc<wgpu::Device>
            let queue  = render_state.queue.clone();    // Arc<wgpu::Queue>
            let target_format = render_state.target_format;

            // Initial size is a placeholder; first frame's resize() fixes it.
            let mut gpu = renderer_wgpu_port::Renderer::new(
                device, queue, target_format, 1, 1,
            );

            gpu.upload_mesh(&triangles);
            gpu.upload_all_slices(&layers, 1);
            gpu.upload_current_slice(&layers);
            gpu.upload_planned_toolpath(&planned_result.layers, args.nozzle_width, args.layer_height);

            // Start showing all layers
            let last_layer = num_layers.saturating_sub(1);
            if !layers.is_empty() {
                gpu.clip_z_max = layers[last_layer].z + 0.001;
                gpu.clip_z_min = layers[0].z - 0.001;
            }

            let renderer = Arc::new(Mutex::new(gpu));

            Ok(Box::new(ViewerApp {
                renderer,
                layers,
                num_layers,
                max_layer: last_layer,
                min_layer: 0,
                slice_view: SliceView::Toolpaths,
                center: [center_x, center_y, center_z],
                extent,
                azimuth: std::f32::consts::FRAC_PI_4 + std::f32::consts::PI,
                elevation: std::f32::consts::FRAC_PI_6,
                zoom: 1.0,
                pan: egui::Vec2::ZERO,
                bg_mode: BgMode::Mesh,
                stats: Stats {
                    triangles: num_triangles,
                    load_ms,
                    slice_ms,
                    offset_ms,
                    plan_ms,
                },
                show_travel_moves: true,
                show_filaments: true,
                fps: 0.0,
                frame_time: 0.0,
                last_update: Instant::now(),
                frame_count: 0,
            }))
        }),
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

struct Stats {
    triangles: usize,
    load_ms: f64,
    slice_ms: f64,
    offset_ms: f64,
    plan_ms: f64,
}

struct ViewerApp {
    renderer: Arc<Mutex<renderer_wgpu_port::Renderer>>,
    layers: Vec<slicer::Layer>,
    num_layers: usize,
    max_layer: usize,
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
    stats: Stats,
    fps: f32,
    frame_time: f32,
    last_update: Instant,
    frame_count: u32,
}

impl eframe::App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Update FPS counter (manually tracked)
        self.frame_count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update);
        if elapsed >= Duration::from_secs(1) {
            self.fps = self.frame_count as f32 / elapsed.as_secs_f32();
            self.frame_time = if self.fps > 0.0 { 1000.0 / self.fps } else { 0.0 };
            self.last_update = now;
            self.frame_count = 0;
        }

        // Top panel
        egui::TopBottomPanel::top("info").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if self.num_layers == 0 {
                    ui.label("No layers");
                    return;
                }
                let height = self.layers[self.num_layers.saturating_sub(1)].z;
                ui.label(format!(
                    "{} layers. Height {:.3} mm",
                    self.num_layers,
                    height,
                ));
                let top_z = &self.layers[self.max_layer].z;
                ui.label(format!(
                    "Top layer {} | z = {:.3} mm",
                    self.max_layer,
                    top_z,
                ));
                let bottom_z = &self.layers[self.min_layer].z;
                ui.label(format!(
                    "Bottom layer {} | z = {:.3} mm",
                    self.min_layer,
                    bottom_z,
                ));
                if ui.button("◀ Prev").clicked() && self.max_layer > 0 {
                    self.max_layer -= 1;
                }
                if ui.button("Next ▶").clicked() && self.max_layer < self.num_layers.saturating_sub(1) {
                    self.max_layer += 1;
                }
                ui.separator();
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
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!(
                        "{} tris | load: {:.0}ms, slice: {:.0}ms, offset: {:.0}ms, plan: {:.0}ms",
                        self.stats.triangles,
                        self.stats.load_ms,
                        self.stats.slice_ms,
                        self.stats.offset_ms,
                        self.stats.plan_ms,
                    ));
                });
            });
        });

        // Left panel: top layer slider (TODO: merge into a single slider with two knobs)
        // Top layer
        egui::SidePanel::left("slider_top")
            .resizable(false)
            .exact_width(32.0)
            .show(ctx, |ui| {
                if self.num_layers == 0 {
                    return;
                }
                let max = self.num_layers.saturating_sub(1);
                ui.spacing_mut().slider_width = ui.available_height() - 16.0;
                ui.add(
                    egui::Slider::new(&mut self.max_layer, 0..=max)
                        .vertical()
                        .show_value(false),
                );
            });
        // Bottom layer
        egui::SidePanel::left("slider_bottom")
            .resizable(false)
            .exact_width(32.0)
            .show(ctx, |ui| {
                if self.num_layers == 0 {
                    return;
                }
                let max = self.num_layers.saturating_sub(1);
                ui.spacing_mut().slider_width = ui.available_height() - 16.0;
                ui.add(
                    egui::Slider::new(&mut self.min_layer, 0..=max)
                        .vertical()
                        .show_value(false),
                );
            });

        // Central panel
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(egui::Color32::from_rgb(26, 26, 46)))
            .show(ctx, |ui| {
                let (response, painter) =
                    ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());

                if response.dragged_by(egui::PointerButton::Primary) {
                    let delta = response.drag_delta();
                    // Check if Command (Mac) or Ctrl (Windows/Linux) is pressed for panning
                    let command_pressed = ui.input(|i| i.modifiers.command || i.modifiers.ctrl);
                    if command_pressed {
                        // Command+drag: pan the camera in world space
                        // Transform screen-space delta to world space based on camera orientation
                        let ca = self.azimuth.cos();
                        let sa = self.azimuth.sin();
                        let ce = self.elevation.cos();
                        let se = self.elevation.sin();
                        let pan_world_scale = self.extent / (2.0 * self.zoom);
                        // Screen X -> world: rotated by azimuth (horizontal plane only)
                        let right_x = ca;
                        let right_y = -sa;
                        let right_z = 0.0;
                        // Screen Y -> world: camera's up vector, affected by both azimuth and elevation
                        // After rotation: up in world space is (-sa*se, ca*se, ce)
                        let up_x = -sa * se;
                        let up_y = ca * se;
                        let up_z = ce;
                        // Combine deltas (note: dragging UP on screen means looking DOWN in world)
                        self.center[0] += (delta.x * right_x - delta.y * up_x) * pan_world_scale * 0.001;
                        self.center[1] += (delta.x * right_y - delta.y * up_y) * pan_world_scale * 0.001;
                        self.center[2] += (delta.x * right_z - delta.y * up_z) * pan_world_scale * 0.001;
                    } else {
                        // Regular drag: rotate the camera
                        self.azimuth -= delta.x * 0.005;
                        self.elevation = (self.elevation + delta.y * 0.005).clamp(
                            -std::f32::consts::FRAC_PI_2 + 0.01,
                            std::f32::consts::FRAC_PI_2 - 0.01,
                        );
                    }
                }
                if response.dragged_by(egui::PointerButton::Middle)
                    || response.dragged_by(egui::PointerButton::Secondary)
                {
                    let delta = response.drag_delta();
                    // Middle/right drag: also pan in world space
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
                    self.center[0] += (delta.x * right_x - delta.y * up_x) * pan_world_scale * 0.001;
                    self.center[1] += (delta.x * right_y - delta.y * up_y) * pan_world_scale * 0.001;
                    self.center[2] += (delta.x * right_z - delta.y * up_z) * pan_world_scale * 0.001;
                }

                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll != 0.0 {
                    let factor = 1.0 + scroll * 0.002;
                    self.zoom = (self.zoom * factor).clamp(0.1, 50.0);
                }

                ui.input(|i| {
                    if i.key_pressed(egui::Key::ArrowUp) || i.key_pressed(egui::Key::ArrowRight) {
                        if self.max_layer < self.num_layers.saturating_sub(1) {
                            self.max_layer += 1;
                        }
                    }
                    if i.key_pressed(egui::Key::ArrowDown) || i.key_pressed(egui::Key::ArrowLeft) {
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

                // Update renderer state (clip_z, draw mode) — no re-upload needed
                if !self.layers.is_empty() {
                    let mut r = self.renderer.lock().unwrap();
                    r.clip_z_max = self.layers[self.max_layer].z + 0.001;
                    r.clip_z_min = self.layers[self.min_layer].z - 0.001;
                    r.draw_contours = self.slice_view == SliceView::Contours;
                    r.draw_toolpaths = self.slice_view == SliceView::Toolpaths;
                    r.show_travel_moves = self.show_travel_moves;
                    r.show_filaments = self.show_filaments;
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
                    (self.pan.x, self.pan.y),
                );

                let bg_mode = self.bg_mode;
                let light_dir = renderer_wgpu_port::headlight_dir(self.azimuth, self.elevation);
                let cam_forward = renderer_wgpu_port::camera_forward(self.azimuth, self.elevation);
                let renderer = self.renderer.clone();
                let ppp = ctx.pixels_per_point();
                let vw = (rect.width() * ppp).max(1.0) as u32;
                let vh = (rect.height() * ppp).max(1.0) as u32;

                let callback = egui_wgpu::Callback::new_paint_callback(
                    rect,
                    ViewerCallback {
                        renderer,
                        mvp,
                        light_dir,
                        cam_forward,
                        bg_mode,
                        width: vw,
                        height: vh,
                    },
                );
                painter.add(callback);
        });

        // FPS counter (bottom-right corner)
        egui::Area::new(egui::Id::new("fps_counter"))
            .anchor(egui::Align2::RIGHT_BOTTOM, [10.0, 10.0])
            .show(ctx, |ui| {
                ui.colored_label(egui::Color32::YELLOW, format!("{:.1} FPS ({:.1} ms)", self.fps, self.frame_time));
            });

    }

    // No on_exit needed: wgpu resources clean up via Drop.
}

// ---------------------------------------------------------------------------
// PaintCallback bridge: ferries per-frame state from the egui side into our
// Renderer's prepare() and paint() methods via the egui_wgpu callback trait.
// ---------------------------------------------------------------------------

struct ViewerCallback {
    renderer: Arc<Mutex<renderer_wgpu_port::Renderer>>,
    mvp: [f32; 16],
    light_dir: [f32; 3],
    cam_forward: [f32; 3],
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
        _callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<eframe::wgpu::CommandBuffer> {
        self.renderer.lock().unwrap().prepare(
            device, queue, encoder,
            &self.mvp, &self.light_dir, &self.cam_forward, self.bg_mode,
            self.width, self.height,
        );
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut eframe::wgpu::RenderPass<'static>,
        _callback_resources: &egui_wgpu::CallbackResources,
    ) {
        self.renderer.lock().unwrap().paint(render_pass);
    }
}
