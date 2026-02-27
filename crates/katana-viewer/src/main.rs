use std::time::Instant;

use clap::Parser;
use eframe::egui;
use katana_core::{mesh, slicer, stl};

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
    let triangles = mesh.triangles.len();

    let t_slice = Instant::now();
    let result = slicer::slice_mesh(&mesh, args.layer_height);
    let slice_ms = t_slice.elapsed().as_secs_f64() * 1000.0;

    println!(
        "Loaded {} ({} triangles) in {:.1}ms",
        args.file, triangles, load_ms
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

    let app = ViewerApp {
        mesh_triangles: mesh.triangles,
        layers: result.layers,
        current_layer: 0,
        center: [center_x, center_y, center_z],
        extent,
        azimuth: std::f32::consts::FRAC_PI_4,
        elevation: std::f32::consts::FRAC_PI_6,
        zoom: 1.0,
        pan: egui::Vec2::ZERO,
        bg_mode: BgMode::Mesh,
        stats: Stats {
            triangles,
            load_ms,
            slice_ms,
        },
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([900.0, 700.0]),
        ..Default::default()
    };

    eframe::run_native(
        "katana viewer",
        options,
        Box::new(|_cc| Ok(Box::new(app))),
    )
}

#[derive(PartialEq)]
enum BgMode {
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
    mesh_triangles: Vec<mesh::Triangle>,
    layers: Vec<slicer::Layer>,
    current_layer: usize,
    center: [f32; 3],
    extent: f32,
    azimuth: f32,
    elevation: f32,
    zoom: f32,
    pan: egui::Vec2,
    bg_mode: BgMode,
    stats: Stats,
}

impl ViewerApp {
    /// Project a 3D mesh-space point to 2D screen-space using an orthographic
    /// projection with camera rotation (azimuth + elevation).
    fn project(&self, x: f32, y: f32, z: f32, canvas: &egui::Rect) -> egui::Pos2 {
        let dx = x - self.center[0];
        let dy = y - self.center[1];
        let dz = z - self.center[2];

        let cos_a = self.azimuth.cos();
        let sin_a = self.azimuth.sin();
        let rx = dx * cos_a - dy * sin_a;
        let ry = dx * sin_a + dy * cos_a;
        let rz = dz;

        let cos_e = self.elevation.cos();
        let sin_e = self.elevation.sin();
        let screen_x = rx;
        let screen_y = -(rz * cos_e - ry * sin_e);

        let margin = 60.0;
        let available = (canvas.width() - 2.0 * margin).min(canvas.height() - 2.0 * margin);
        let scale = available / self.extent * self.zoom;

        let cx = canvas.center().x + self.pan.x;
        let cy = canvas.center().y + self.pan.y;

        egui::pos2(screen_x * scale + cx, screen_y * scale + cy)
    }

    /// Compute the camera's view direction (used for backface culling).
    fn view_direction(&self) -> [f32; 3] {
        let cos_a = self.azimuth.cos();
        let sin_a = self.azimuth.sin();
        let cos_e = self.elevation.cos();
        let sin_e = self.elevation.sin();
        // Camera looks along -Z in view space, which in world space is:
        [sin_a * cos_e, -cos_a * cos_e, -sin_e]
    }

    fn draw_mesh_wireframe(&self, painter: &egui::Painter, canvas: &egui::Rect) {
        let view_dir = self.view_direction();
        let stroke = egui::Stroke::new(
            0.5,
            egui::Color32::from_rgba_premultiplied(70, 80, 110, 80),
        );

        for tri in &self.mesh_triangles {
            // Backface culling: skip triangles facing away from camera
            let n = &tri.normal;
            let dot = n.x * view_dir[0] + n.y * view_dir[1] + n.z * view_dir[2];
            if dot > 0.0 {
                continue;
            }

            let p0 = self.project(
                tri.vertices[0].x,
                tri.vertices[0].y,
                tri.vertices[0].z,
                canvas,
            );
            let p1 = self.project(
                tri.vertices[1].x,
                tri.vertices[1].y,
                tri.vertices[1].z,
                canvas,
            );
            let p2 = self.project(
                tri.vertices[2].x,
                tri.vertices[2].y,
                tri.vertices[2].z,
                canvas,
            );

            painter.line_segment([p0, p1], stroke);
            painter.line_segment([p1, p2], stroke);
            painter.line_segment([p2, p0], stroke);
        }
    }

    fn draw_bg_layers(&self, painter: &egui::Painter, canvas: &egui::Rect) {
        let stride = (self.layers.len() / 100).max(1);
        let bg_stroke = egui::Stroke::new(
            0.5,
            egui::Color32::from_rgba_premultiplied(80, 80, 120, 60),
        );

        for (i, layer) in self.layers.iter().enumerate() {
            if i == self.current_layer || i % stride != 0 {
                continue;
            }
            for contour in &layer.contours {
                if contour.points.len() < 2 {
                    continue;
                }
                let pts: Vec<egui::Pos2> = contour
                    .points
                    .iter()
                    .map(|p| self.project(p.x, p.y, layer.z, canvas))
                    .collect();
                painter.add(egui::Shape::closed_line(pts, bg_stroke));
            }
        }
    }

    fn draw_current_layer(&self, painter: &egui::Painter, canvas: &egui::Rect) {
        if self.layers.is_empty() {
            return;
        }
        let layer = &self.layers[self.current_layer];
        let stroke = egui::Stroke::new(1.5, egui::Color32::from_rgb(233, 69, 96));

        for contour in &layer.contours {
            if contour.points.len() < 2 {
                continue;
            }
            let pts: Vec<egui::Pos2> = contour
                .points
                .iter()
                .map(|p| self.project(p.x, p.y, layer.z, canvas))
                .collect();
            painter.add(egui::Shape::closed_line(pts, stroke));
        }
    }
}

impl eframe::App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Top panel: info
        egui::TopBottomPanel::top("info").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if self.layers.is_empty() {
                    ui.label("No layers");
                    return;
                }
                let layer = &self.layers[self.current_layer];
                ui.label(format!(
                    "Layer {}/{} | z = {:.3} mm | {} contour{}",
                    self.current_layer + 1,
                    self.layers.len(),
                    layer.z,
                    layer.contours.len(),
                    if layer.contours.len() == 1 { "" } else { "s" }
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

        // Bottom panel: layer slider
        egui::TopBottomPanel::bottom("slider").show(ctx, |ui| {
            if self.layers.is_empty() {
                return;
            }
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("Layer:");
                let max = self.layers.len().saturating_sub(1);
                ui.add(
                    egui::Slider::new(&mut self.current_layer, 0..=max).show_value(false),
                );
            });
            ui.add_space(4.0);
        });

        // Central canvas
        egui::CentralPanel::default().show(ctx, |ui| {
            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());
            let canvas = response.rect;

            painter.rect_filled(canvas, 0.0, egui::Color32::from_rgb(26, 26, 46));

            // Left-drag = rotate, right/middle-drag = pan
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
                    if self.current_layer < self.layers.len().saturating_sub(1) {
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
                    self.current_layer = self.layers.len().saturating_sub(1);
                }
            });

            // Draw background then current layer on top
            match self.bg_mode {
                BgMode::Mesh => self.draw_mesh_wireframe(&painter, &canvas),
                BgMode::Layers => self.draw_bg_layers(&painter, &canvas),
                BgMode::None => {}
            }
            self.draw_current_layer(&painter, &canvas);
        });
    }
}
