use std::sync::{Arc, Mutex};
use std::time::Instant;

use clap::Parser;
use eframe::egui;
use eframe::glow;
use katana_core::{slicer, stl};

mod renderer;

#[derive(Parser)]
#[command(name = "katana-viewer", about = "2D layer viewer for sliced meshes")]
struct Args {
    /// Path to an STL file
    file: String,
    /// Layer height in mm
    #[arg(short, long, default_value_t = 0.2)]
    layer_height: f32,
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

    println!(
        "Loaded {} ({} triangles) in {:.1}ms",
        args.file, num_triangles, load_ms
    );
    println!(
        "Sliced {} layers (z {:.2} to {:.2}) in {:.1}ms",
        result.layers.len(),
        mesh_min.z,
        mesh_max.z,
        slice_ms,
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
        viewport: egui::ViewportBuilder::default().with_inner_size([900.0, 700.0]),
        ..Default::default()
    };

    eframe::run_native(
        "katana viewer",
        options,
        Box::new(move |cc| {
            let gl = cc.gl.as_ref().expect("eframe glow backend required");
            let mut gpu = renderer::Renderer::new(gl.clone());

            // Upload mesh wireframe to GPU
            gpu.upload_mesh(&triangles);

            // Upload all slices — single GPU draw call, no need to skip layers
            gpu.upload_all_slices(&layers, 1);

            // Upload first layer
            if !layers.is_empty() {
                gpu.upload_current_slice(&layers[0]);
            }

            let renderer = Arc::new(Mutex::new(gpu));

            Ok(Box::new(ViewerApp {
                renderer,
                layers,
                num_layers,
                current_layer: 0,
                prev_layer: usize::MAX,
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
                },
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

struct Stats {
    triangles: usize,
    load_ms: f64,
    slice_ms: f64,
}

struct ViewerApp {
    renderer: Arc<Mutex<renderer::Renderer>>,
    layers: Vec<slicer::Layer>,
    num_layers: usize,
    current_layer: usize,
    prev_layer: usize,
    center: [f32; 3],
    extent: f32,
    azimuth: f32,
    elevation: f32,
    zoom: f32,
    pan: egui::Vec2,
    bg_mode: BgMode,
    stats: Stats,
}

impl eframe::App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Re-upload current slice if it changed
        if self.current_layer != self.prev_layer && !self.layers.is_empty() {
            self.renderer
                .lock()
                .unwrap()
                .upload_current_slice(&self.layers[self.current_layer]);
            self.prev_layer = self.current_layer;
        }

        // Top panel
        egui::TopBottomPanel::top("info").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if self.num_layers == 0 {
                    ui.label("No layers");
                    return;
                }
                let layer = &self.layers[self.current_layer];
                ui.label(format!(
                    "Layer {}/{} | z = {:.3} mm | {} contour{}",
                    self.current_layer + 1,
                    self.num_layers,
                    layer.z,
                    layer.contours.len(),
                    if layer.contours.len() == 1 { "" } else { "s" },
                ));
                ui.separator();
                ui.selectable_value(&mut self.bg_mode, BgMode::Mesh, "Mesh");
                ui.selectable_value(&mut self.bg_mode, BgMode::Layers, "Layers");
                ui.selectable_value(&mut self.bg_mode, BgMode::None, "None");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!(
                        "zoom: {:.0}% | {} tris, load: {:.1}ms, slice: {:.1}ms",
                        self.zoom * 100.0,
                        self.stats.triangles,
                        self.stats.load_ms,
                        self.stats.slice_ms,
                    ));
                });
            });
        });

        // Bottom panel: slider
        egui::TopBottomPanel::bottom("slider").show(ctx, |ui| {
            if self.num_layers == 0 {
                return;
            }
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("Layer:");
                let max = self.num_layers.saturating_sub(1);
                ui.add(
                    egui::Slider::new(&mut self.current_layer, 0..=max).show_value(false),
                );
            });
            ui.add_space(4.0);
        });

        // Central panel
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(egui::Color32::from_rgb(26, 26, 46)))
            .show(ctx, |ui| {
                let (response, painter) =
                    ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());

                // Input handling
                if response.dragged_by(egui::PointerButton::Primary) {
                    let delta = response.drag_delta();
                    self.azimuth -= delta.x * 0.005;
                    self.elevation = (self.elevation + delta.y * 0.005).clamp(
                        -std::f32::consts::FRAC_PI_2 + 0.01,
                        std::f32::consts::FRAC_PI_2 - 0.01,
                    );
                }
                if response.dragged_by(egui::PointerButton::Middle)
                    || response.dragged_by(egui::PointerButton::Secondary)
                {
                    self.pan += response.drag_delta();
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

                // Build MVP matrix
                let rect = response.rect;
                let aspect = rect.width() / rect.height();
                let mvp = renderer::build_mvp(
                    self.center,
                    self.azimuth,
                    self.elevation,
                    self.zoom,
                    self.extent,
                    aspect,
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
                            // Screen position of the viewport (bottom-left in GL coords)
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
