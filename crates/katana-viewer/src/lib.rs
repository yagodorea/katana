use std::sync::{ Arc, Mutex };

use eframe::egui;
use eframe::egui_wgpu;
use katana_core::{ gcode::{ self, Gcode, GcodeConfig }, offset, planner, slicer, stl };
use nalgebra::{ Point2, Vector2 };

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// Re-exporting this generates the `wbg_rayon_start_worker` export that each
// spawned Web Worker boots into. Required by wasm-bindgen-rayon — do not remove.
#[cfg(target_arch = "wasm32")]
pub use wasm_bindgen_rayon::init_thread_pool;

pub mod renderer_wgpu_port;

// ---------------------------------------------------------------------------
// Platform-specific imports
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
use std::time::{ Duration, Instant };

#[cfg(target_arch = "wasm32")]
use web_time::{ Duration, Instant };

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(PartialEq, Clone, Copy)]
pub enum BgMode {
    None,
    Mesh,
    Layers,
}

#[derive(PartialEq, Clone, Copy)]
pub enum SliceView {
    Contours,
    Toolpaths,
}

#[derive(PartialEq, Clone, Copy)]
pub enum Phase {
    Import,
    Model,
    Processing,
    Sliced,
}

pub struct Stats {
    pub triangles: usize,
    pub load_ms: f64,
    pub slice_ms: f64,
    pub offset_ms: f64,
    pub plan_ms: f64,
}

// ---------------------------------------------------------------------------
// ViewerApp
// ---------------------------------------------------------------------------

pub struct ViewerApp {
    pub phase: Phase,
    pub renderer: Arc<Mutex<renderer_wgpu_port::Renderer>>,

    // Mesh data (populated on import)
    pub mesh_triangles: Option<Vec<katana_core::mesh::Triangle>>,
    pub mesh_min: nalgebra::Point3<f32>,
    pub mesh_max: nalgebra::Point3<f32>,
    pub source_file: String,
    pub gcode_offset: Vector2<f32>,

    // Slicing results (populated on slice)
    pub planned_result: Option<planner::PlannedResult>,
    pub layers: Vec<slicer::Layer>,
    pub num_layers: usize,
    pub max_layer: usize,
    pub prev_max_layer: usize,
    pub min_layer: usize,
    pub slice_view: SliceView,
    pub center: [f32; 3],
    pub extent: f32,
    pub azimuth: f32,
    pub elevation: f32,
    pub zoom: f32,
    pub pan: egui::Vec2,
    pub bg_mode: BgMode,
    pub show_travel_moves: bool,
    pub show_filaments: bool,
    pub scrub: f32,
    pub stats: Stats,
    pub fps: f32,
    pub frame_time: f32,
    pub last_update: Instant,
    pub frame_count: u32,

    // Slicer config (from CLI args or defaults, used when Slice is clicked)
    pub nozzle_width: f32,
    pub layer_height: f32,
    pub perimeters: usize,
    pub infill_density: usize,
    pub bottom_layers: usize,
    pub top_layers: usize,
    pub travel_speed: f32,
    pub perimeter_speed: f32,
    pub infill_speed: f32,
    pub surface_speed: f32,

    // Slicing state (for async processing)
    pub slicing_progress: f32,
    pub slicing_status: String,
    pub slicing_cancelled: bool,

    // Web file picking: the async file dialog stashes (name, bytes) here for
    // the update loop to pick up (the dialog future can't touch `self` directly).
    #[cfg(target_arch = "wasm32")]
    pub pending_file: std::rc::Rc<std::cell::RefCell<Option<(String, Vec<u8>)>>>,
}

impl ViewerApp {
    /// Create a ViewerApp for the web target with sensible defaults.
    pub fn new_web(renderer: Arc<Mutex<renderer_wgpu_port::Renderer>>) -> Self {
        Self {
            phase: Phase::Import,
            renderer,
            mesh_triangles: None,
            mesh_min: nalgebra::Point3::origin(),
            mesh_max: nalgebra::Point3::origin(),
            source_file: String::new(),
            gcode_offset: Vector2::zeros(),
            planned_result: None,
            layers: Vec::new(),
            num_layers: 0,
            max_layer: 0,
            prev_max_layer: 0,
            min_layer: 0,
            slice_view: SliceView::Toolpaths,
            center: [0.0, 0.0, 128.0],
            extent: 256.0,
            azimuth: std::f32::consts::FRAC_PI_4 + std::f32::consts::PI,
            elevation: std::f32::consts::FRAC_PI_6,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            bg_mode: BgMode::Mesh,
            stats: Stats {
                triangles: 0,
                load_ms: 0.0,
                slice_ms: 0.0,
                offset_ms: 0.0,
                plan_ms: 0.0,
            },
            show_travel_moves: false,
            show_filaments: true,
            scrub: 1.0,
            fps: 0.0,
            frame_time: 0.0,
            last_update: Instant::now(),
            frame_count: 0,
            nozzle_width: 0.4,
            layer_height: 0.2,
            perimeters: 3,
            infill_density: 20,
            bottom_layers: 3,
            top_layers: 3,
            travel_speed: 150.0,
            perimeter_speed: 30.0,
            infill_speed: 60.0,
            surface_speed: 40.0,
            slicing_progress: 0.0,
            slicing_status: String::new(),
            slicing_cancelled: false,
            #[cfg(target_arch = "wasm32")]
            pending_file: std::rc::Rc::new(std::cell::RefCell::new(None)),
        }
    }

    // -----------------------------------------------------------------------
    // Platform-agnostic core logic
    // -----------------------------------------------------------------------

    /// Load an STL from raw bytes, upload mesh to GPU, transition to Model phase.
    pub fn load_stl_from_bytes(&mut self, data: &[u8], source_name: &str) {
        let t_load = Instant::now();
        let mesh = match stl::load_stl(data) {
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
            r.slices_buffer = None;
            r.current_slice_buffer = None;
            r.toolpath_lines_buffer = None;
            r.toolpath_path_lines_buffer = None;
            r.toolpath_rhombuses = None;
        }

        println!("Loaded {source_name} ({num_triangles} triangles) in {load_ms:.1}ms");

        self.mesh_triangles = Some(mesh.triangles);
        self.mesh_min = mesh_min;
        self.mesh_max = mesh_max;
        self.source_file = source_name.to_string();
        self.gcode_offset = gcode_offset;
        self.center = [center_x, center_y, center_z];
        self.pan = egui::Vec2::ZERO;
        self.extent = extent;
        self.stats = Stats {
            triangles: num_triangles,
            load_ms,
            slice_ms: 0.0,
            offset_ms: 0.0,
            plan_ms: 0.0,
        };
        self.slicing_progress = 0.0;
        self.slicing_status.clear();
        self.slicing_cancelled = false;
        self.phase = Phase::Model;
    }

    /// Generate the G-code string from the planned result.
    pub fn generate_gcode_string(&self) -> Option<String> {
        let planned = self.planned_result.as_ref()?;
        let mut exporter = Gcode {
            config: GcodeConfig {
                filament_diameter: 1.75,
                nozzle_width: self.nozzle_width,
                layer_height: self.layer_height,
            },
            offset: self.gcode_offset,
            e: 0.0,
            out: String::new(),
        };
        Some(exporter.export(planned))
    }

    /// Reset to import screen (clear everything)
    pub fn reset_to_import(&mut self) {
        self.phase = Phase::Import;
        self.mesh_triangles = None;
        self.layers.clear();
        self.planned_result = None;
        self.slicing_progress = 0.0;
        self.slicing_status.clear();
        self.slicing_cancelled = false;
        
        // Clear renderer buffers (set to None instead of empty - avoids wgpu panic on slice)
        let mut r = self.renderer.lock().unwrap();
        r.mesh_buffer = None;
        r.slices_buffer = None;
        r.current_slice_buffer = None;
        r.toolpath_lines_buffer = None;
        r.toolpath_path_lines_buffer = None;
        r.toolpath_rhombuses = None;
    }

    /// Clear slice results and return to model view
    pub fn clear_slice(&mut self) {
        self.phase = Phase::Model;
        self.layers.clear();
        self.planned_result = None;
        self.num_layers = 0;
        self.max_layer = 0;
        self.prev_max_layer = 0;
        self.min_layer = 0;
        self.scrub = 1.0;
        self.slicing_progress = 0.0;
        self.slicing_status.clear();
        self.slicing_cancelled = false;
        
        // Clear renderer buffers
        let mut r = self.renderer.lock().unwrap();
        r.slices_buffer = None;
        r.current_slice_buffer = None;
        r.toolpath_lines_buffer = None;
        r.toolpath_path_lines_buffer = None;
        r.toolpath_rhombuses = None;
    }

    /// Cancel ongoing slicing operation
    pub fn cancel_slicing(&mut self) {
        self.slicing_cancelled = true;
    }

    /// Check if slicing is currently in progress
    pub fn is_slicing(&self) -> bool {
        self.phase == Phase::Processing
    }

    /// Run the full slicing pipeline and upload results to GPU.
    pub fn run_slice(&mut self) {
        let Some(ref triangles) = self.mesh_triangles else {
            return;
        };
        
        // Enter processing phase
        self.phase = Phase::Processing;
        self.slicing_progress = 0.0;
        self.slicing_status = "Slicing mesh...".to_string();
        self.slicing_cancelled = false;
        
        let mesh = katana_core::mesh::Mesh {
            triangles: triangles.clone(),
            source: katana_core::mesh::MeshSource::StlBinary,
        };

        // Step 1: Slice
        self.slicing_progress = 0.15;
        let t_slice = Instant::now();
        let result = slicer::slice_mesh(&mesh, self.layer_height);
        let slice_ms = t_slice.elapsed().as_secs_f64() * 1000.0;

        if self.slicing_cancelled {
            self.slicing_status = "Cancelled".to_string();
            self.phase = Phase::Model;
            return;
        }

        // Step 2: Generate toolpaths
        self.slicing_progress = 0.40;
        self.slicing_status = "Generating toolpaths...".to_string();
        
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

        if self.slicing_cancelled {
            self.slicing_status = "Cancelled".to_string();
            self.phase = Phase::Model;
            return;
        }

        // Step 3: Plan
        self.slicing_progress = 0.65;
        self.slicing_status = "Planning moves...".to_string();
        
        let speed_config = planner::SpeedConfig {
            travel: self.travel_speed,
            perimeter: self.perimeter_speed,
            infill: self.infill_speed,
            surface: self.surface_speed,
        };

        let t_plan = Instant::now();
        let planned_result = planner::plan_toolpaths(&toolpath_result, &speed_config);
        let plan_ms = t_plan.elapsed().as_secs_f64() * 1000.0;

        if self.slicing_cancelled {
            self.slicing_status = "Cancelled".to_string();
            self.phase = Phase::Model;
            return;
        }

        // Step 4: Upload to GPU
        self.slicing_progress = 0.85;
        self.slicing_status = "Uploading to GPU...".to_string();
        
        let layers = result.layers;
        let num_layers = layers.len();
        let last_layer = num_layers.saturating_sub(1);

        println!(
            "Sliced {num_layers} layers in {slice_ms:.1}ms, perimeters in {offset_ms:.1}ms, planning in {plan_ms:.1}ms"
        );

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

        if self.slicing_cancelled {
            self.slicing_status = "Cancelled".to_string();
            self.phase = Phase::Model;
            self.layers.clear();
            self.planned_result = None;
            return;
        }

        // Done
        self.slicing_progress = 1.0;
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
        self.slicing_status.clear();
    }

    // -----------------------------------------------------------------------
    // Platform-specific: native file I/O
    // -----------------------------------------------------------------------

    /// Load an STL from a filesystem path (native only).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn import_stl(&mut self, path: &str) {
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Failed to read {path}: {e}");
                return;
            }
        };
        self.load_stl_from_bytes(&data, path);
    }

    /// Open a native save dialog and write G-code to disk (native only).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn export_gcode(&self) {
        let Some(gcode_str) = self.generate_gcode_string() else {
            return;
        };

        let default_name = std::path::Path
            ::new(&self.source_file)
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| format!("{s}.gcode"))
            .unwrap_or_else(|| "output.gcode".to_string());

        let Some(path) = rfd::FileDialog
            ::new()
            .set_file_name(default_name)
            .add_filter("G-code", &["gcode"])
            .save_file() else {
            return;
        };

        if let Err(e) = std::fs::write(path, &gcode_str) {
            eprintln!("Error writing to file! {e}");
        }
    }

    /// Download G-code as a file in the browser (WASM only).
    #[cfg(target_arch = "wasm32")]
    pub fn export_gcode(&self) {
        let Some(gcode_str) = self.generate_gcode_string() else {
            return;
        };

        use wasm_bindgen::JsCast;
        let window = web_sys::window().unwrap();
        let document = window.document().unwrap();

        let array = js_sys::Uint8Array::new_with_length(gcode_str.len() as u32);
        array.copy_from(gcode_str.as_bytes());
        let blob_parts = js_sys::Array::new();
        blob_parts.push(&array);
        let bag = web_sys::BlobPropertyBag::new();
        bag.set_type("text/plain");
        let blob = web_sys::Blob
            ::new_with_u8_array_sequence_and_options(&blob_parts, &bag)
            .unwrap();
        let url = web_sys::Url::create_object_url_with_blob(&blob).unwrap();

        let anchor: web_sys::HtmlAnchorElement = document
            .create_element("a")
            .unwrap()
            .dyn_into()
            .unwrap();
        anchor.set_href(&url);
        anchor.set_download(&format!("{}.gcode", self.source_file));
        anchor.click();
        web_sys::Url::revoke_object_url(&url).unwrap();
    }
}

// ---------------------------------------------------------------------------
// eframe::App
// ---------------------------------------------------------------------------

impl eframe::App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.frame_count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update);
        if elapsed >= Duration::from_secs(1) {
            self.fps = (self.frame_count as f32) / elapsed.as_secs_f32();
            self.frame_time = if self.fps > 0.0 { 1000.0 / self.fps } else { 0.0 };
            self.last_update = now;
            self.frame_count = 0;
        }

        // Check for drag-and-drop files (works on both native and web via egui)
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if let Some(file) = dropped.first() {
            if let Some(bytes) = &file.bytes {
                // Reset slicing state when loading a new file
                self.slicing_progress = 0.0;
                self.slicing_status.clear();
                self.slicing_cancelled = false;
                self.load_stl_from_bytes(bytes, &file.name);
            }
        }

        // Check for a file picked via the web "Import STL" dialog (async).
        // Take out of the RefCell in its own statement so the borrow is released
        // before we call the `&mut self` loader below.
        #[cfg(target_arch = "wasm32")]
        {
            let picked = self.pending_file.borrow_mut().take();
            if let Some((name, bytes)) = picked {
                self.slicing_progress = 0.0;
                self.slicing_status.clear();
                self.slicing_cancelled = false;
                self.load_stl_from_bytes(&bytes, &name);
            }
        }

        match self.phase {
            Phase::Import => self.update_import(ctx),
            Phase::Model => self.update_model(ctx),
            Phase::Processing => self.update_processing(ctx),
            Phase::Sliced => self.update_sliced(ctx),
        }

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
}

// ---------------------------------------------------------------------------
// UI phases
// ---------------------------------------------------------------------------

impl ViewerApp {
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
                        self.open_file_dialog(ctx);
                    }
                });
            });
    }

    fn update_model(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("model_info").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("← Back").clicked() {
                    self.reset_to_import();
                }
                ui.separator();
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

    fn update_sliced(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("info").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("← Back").clicked() {
                    self.clear_slice();
                }
                ui.separator();
                
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
                
                if ui.button("🔄 Reslice").clicked() {
                    self.clear_slice();
                    self.run_slice();
                }
                
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("💾 Export G-code").clicked() {
                        self.export_gcode();
                    }
                });
            });
        });

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

        if self.num_layers > 0 {
            let height = self.layers[self.num_layers.saturating_sub(1)].z;
            let top_z = self.layers[self.max_layer].z;
            let bottom_z = self.layers[self.min_layer].z;
            egui::Window
                ::new("📊 Stats")
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

    fn update_processing(&mut self, ctx: &egui::Context) {
        egui::CentralPanel
            ::default()
            .frame(egui::Frame::NONE.fill(egui::Color32::from_rgb(26, 26, 46)))
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(ui.available_height() / 4.0);
                    
                    ui.heading(
                        egui::RichText
                            ::new("Processing...")
                            .size(32.0)
                            .color(egui::Color32::from_rgb(200, 200, 220))
                    );
                    
                    ui.add_space(24.0);
                    
                    if !self.slicing_status.is_empty() {
                        ui.label(
                            egui::RichText
                                ::new(&self.slicing_status)
                                .size(16.0)
                                .color(egui::Color32::from_rgb(180, 180, 200))
                        );
                    }
                    
                    ui.add_space(16.0);
                    
                    ui.add_sized(
                        [300.0, 24.0],
                        egui::ProgressBar::new(self.slicing_progress).animate(true).text("")
                    );
                    
                    ui.add_space(32.0);
                    
                    let cancel_btn = ui.add_sized(
                        [160.0, 40.0],
                        egui::Button
                            ::new(egui::RichText::new("Cancel").size(16.0))
                            .fill(egui::Color32::from_rgb(120, 50, 50))
                    );
                    if cancel_btn.clicked() {
                        self.cancel_slicing();
                    }
                });
            });
    }

    fn pan_camera(&mut self, delta: egui::Vec2, viewport: egui::Vec2) {
        // build_mvp scales the world by 2·zoom/extent across the smaller viewport
        // dimension, so this is exactly one screen point in world units.
        let scale = self.extent / (self.zoom * viewport.min_elem().max(1.0));
        // Screen y is positive downward; view-space y is up.
        self.pan.x += delta.x * scale;
        self.pan.y -= delta.y * scale;
    }

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
                        self.pan_camera(delta, response.rect.size());
                    } else {
                        self.azimuth -= delta.x * 0.005;
                        self.elevation += delta.y * 0.005;
                    }
                }
                if
                    response.dragged_by(egui::PointerButton::Middle) ||
                    response.dragged_by(egui::PointerButton::Secondary)
                {
                    self.pan_camera(response.drag_delta(), response.rect.size());
                }

                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll != 0.0 {
                    let factor = 1.0 + scroll * 0.002;
                    self.zoom = (self.zoom * factor).clamp(0.1, 50.0);
                }

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

                    if self.max_layer != self.prev_max_layer {
                        self.scrub = 1.0;
                        self.prev_max_layer = self.max_layer;
                    }
                }

                if !self.layers.is_empty() {
                    let mut r = self.renderer.lock().unwrap();
                    r.clip_z_max = self.layers[self.max_layer].z + 0.001;
                    r.clip_z_min = self.layers[self.min_layer].z - 0.001;
                    r.draw_contours = self.slice_view == SliceView::Contours;
                    r.draw_toolpaths = self.slice_view == SliceView::Toolpaths;
                    r.show_travel_moves = self.show_travel_moves;
                    r.show_filaments = self.show_filaments;
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
                    renderer: ThreadSafeRenderer(renderer),
                    mvp,
                    light_dir,
                    bg_mode,
                    width: vw,
                    height: vh,
                });
                painter.add(callback);
            });
    }

    /// Open a file dialog to pick an STL file.
    /// Native: blocking rfd dialog. Web: handled by the web entry point via drag-and-drop or async dialog.
    #[cfg(not(target_arch = "wasm32"))]
    fn open_file_dialog(&mut self, _ctx: &egui::Context) {
        if let Some(path) = rfd::FileDialog::new().add_filter("STL", &["stl"]).pick_file() {
            if let Some(path_str) = path.to_str() {
                self.import_stl(path_str);
            }
        }
    }

    /// On web, the file dialog is async. Spawn a task that opens the browser
    /// file picker, reads the bytes, and stashes them in `pending_file` for the
    /// update loop to consume (the future can't borrow `self`).
    #[cfg(target_arch = "wasm32")]
    fn open_file_dialog(&mut self, ctx: &egui::Context) {
        let pending = self.pending_file.clone();
        let ctx = ctx.clone();
        wasm_bindgen_futures::spawn_local(async move {
            if
                let Some(handle) = rfd::AsyncFileDialog
                    ::new()
                    .add_filter("STL", &["stl"])
                    .pick_file().await
            {
                let name = handle.file_name();
                let bytes = handle.read().await;
                *pending.borrow_mut() = Some((name, bytes));
                ctx.request_repaint();
            }
        });
    }
}

// ---------------------------------------------------------------------------
// WASM entry point
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn start_web() {
    use wasm_bindgen::JsCast;

    console_error_panic_hook::set_once();

    wasm_bindgen_futures::spawn_local(async move {
        // Spin up the rayon thread pool (one Web Worker per hardware thread)
        // before any slicing runs. par_iter() in katana-core uses this global pool.
        let threads = web_sys
            ::window()
            .map(|w| w.navigator().hardware_concurrency() as usize)
            .unwrap_or(1)
            .max(1);
        let _ = wasm_bindgen_futures::JsFuture::from(init_thread_pool(threads)).await;

        let web_options = eframe::WebOptions::default();

        let canvas = web_sys
            ::window()
            .unwrap()
            .document()
            .unwrap()
            .get_element_by_id("katana_canvas")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();

        let runner = eframe::WebRunner::new();
        runner
            .start(
                canvas,
                web_options,
                Box::new(|cc| {
                    let render_state = cc.wgpu_render_state.as_ref().unwrap();
                    let device = render_state.device.clone();
                    let queue = render_state.queue.clone();
                    let target_format = render_state.target_format;

                    let gpu = renderer_wgpu_port::Renderer::new(device, queue, target_format, 1, 1);
                    let renderer = Arc::new(Mutex::new(gpu));

                    Ok(Box::new(ViewerApp::new_web(renderer)))
                })
            ).await
            .unwrap();
    });
}

// ---------------------------------------------------------------------------
// PaintCallback bridge
// ---------------------------------------------------------------------------

/// Wrapper that provides `Send + Sync` for `Renderer` on WASM.
///
/// On WASM, wgpu's `Buffer` contains a `RefCell` which is not `Sync`.
/// However, WASM is single-threaded and our `Mutex` provides the needed
/// synchronization, so this is safe.
struct ThreadSafeRenderer(Arc<Mutex<renderer_wgpu_port::Renderer>>);

// SAFETY: On WASM, everything runs on a single browser thread.
// The Mutex provides the needed synchronization for multi-threaded native.
unsafe impl Send for ThreadSafeRenderer {}
unsafe impl Sync for ThreadSafeRenderer {}

struct ViewerCallback {
    renderer: ThreadSafeRenderer,
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
        self.renderer.0
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
        self.renderer.0.lock().unwrap().paint(render_pass);
    }
}
