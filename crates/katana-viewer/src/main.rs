use std::sync::{Arc, Mutex};
use std::time::Instant;

use clap::Parser;
use eframe::egui;
use eframe::glow;
use katana_core::{offset, planner, slicer, stl};

mod renderer;

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
        ..Default::default()
    };

    eframe::run_native(
        "katana viewer",
        options,
        Box::new(move |cc| {
            let gl = cc.gl.as_ref().expect("eframe glow backend required");
            let mut gpu = renderer::Renderer::new(gl.clone());

            gpu.upload_mesh(&triangles);
            gpu.upload_all_slices(&layers, 1);

            // Upload ALL contour + toolpath data once; layer visibility
            // is controlled by the u_clip_z shader uniform.
            gpu.upload_current_slice(&layers);
            gpu.upload_planned_toolpath(&planned_result.layers, args.nozzle_width);

            // Start showing all layers from the bottom
            if !layers.is_empty() {
                gpu.clip_z = layers[0].z - 0.001;
            }

            let renderer = Arc::new(Mutex::new(gpu));

            Ok(Box::new(ViewerApp {
                renderer,
                layers,
                num_layers,
                current_layer: 0,
                slice_view: SliceView::Toolpaths,
                center: [center_x, center_y, center_z],
                extent,
                azimuth: std::f32::consts::FRAC_PI_4,
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
    renderer: Arc<Mutex<renderer::Renderer>>,
    layers: Vec<slicer::Layer>,
    num_layers: usize,
    current_layer: usize,
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
}

impl eframe::App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Top panel
        egui::TopBottomPanel::top("info").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if self.num_layers == 0 {
                    ui.label("No layers");
                    return;
                }
                let layer = &self.layers[self.current_layer];
                ui.label(format!(
                    "Layer {}/{} | z = {:.3} mm",
                    self.current_layer + 1,
                    self.num_layers,
                    layer.z,
                ));
                if ui.button("◀ Prev").clicked() && self.current_layer > 0 {
                    self.current_layer -= 1;
                }
                if ui.button("Next ▶").clicked() && self.current_layer < self.num_layers.saturating_sub(1) {
                    self.current_layer += 1;
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

        // Left panel: vertical layer slider
        egui::SidePanel::left("slider")
            .resizable(false)
            .exact_width(32.0)
            .show(ctx, |ui| {
                if self.num_layers == 0 {
                    return;
                }
                let max = self.num_layers.saturating_sub(1);
                // Invert so top of slider = layer 0 (all layers visible)
                let mut inverted = max - self.current_layer;
                ui.spacing_mut().slider_width = ui.available_height() - 16.0;
                ui.add(
                    egui::Slider::new(&mut inverted, 0..=max)
                        .vertical()
                        .show_value(false),
                );
                self.current_layer = max - inverted;
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
                        if self.current_layer < self.num_layers.saturating_sub(1) {
                            self.current_layer += 1;
                        }
                    }
                    if i.key_pressed(egui::Key::ArrowDown) || i.key_pressed(egui::Key::ArrowLeft) {
                        if self.current_layer > 0 {
                            self.current_layer -= 1;
                        }
                    }
                    if i.key_pressed(egui::Key::Home) {
                        self.current_layer = 0;
                    }
                    if i.key_pressed(egui::Key::End) {
                        self.current_layer = self.num_layers.saturating_sub(1);
                    }
                });

                // Update renderer state (clip_z, draw mode) — no re-upload needed
                if !self.layers.is_empty() {
                    let mut r = self.renderer.lock().unwrap();
                    r.clip_z = self.layers[self.current_layer].z - 0.001;
                    r.draw_contours = self.slice_view == SliceView::Contours;
                    r.draw_toolpaths = self.slice_view == SliceView::Toolpaths;
                    r.show_travel_moves = self.show_travel_moves;
                    r.show_filaments = self.show_filaments;
                }

                let rect = response.rect;
                let aspect = rect.width() / rect.height();
                let mvp = renderer::build_mvp(
                    self.center,
                    self.azimuth,
                    self.elevation,
                    self.zoom,
                    self.extent,
                    aspect,
                    (self.pan.x, self.pan.y),
                );

                let bg_mode = self.bg_mode;
                let light_dir = renderer::headlight_dir(self.azimuth, self.elevation);
                let renderer = self.renderer.clone();
                let ppp = ctx.pixels_per_point();
                let vw = (rect.width() * ppp) as i32;
                let vh = (rect.height() * ppp) as i32;

                let callback = egui::PaintCallback {
                    rect,
                    callback: Arc::new(eframe::egui_glow::CallbackFn::new(
                        move |info, _painter| {
                            let vp = info.viewport_in_pixels();
                            let sx = vp.left_px;
                            let sy = vp.from_bottom_px;
                            renderer.lock().unwrap().draw(
                                &mvp, &light_dir, &bg_mode, vw, vh, sx, sy,
                            );
                        },
                    )),
                };
                painter.add(callback);
            });
    }

    fn on_exit(&mut self, gl: Option<&glow::Context>) {
        if gl.is_some() {
            self.renderer.lock().unwrap().destroy();
        }
    }
}
