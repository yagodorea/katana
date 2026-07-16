//! wgpu-based renderer for the viewer (port of `renderer.rs`).
//!
//! Module layout:
//! - `buffers`   : POD vertex/instance/uniform structs and upload helpers.
//! - `pipelines` : the three render pipelines + shared bind group layout.
//! - this file   : the `Renderer` struct, public API, prepare/paint logic,
//!                 and the offscreen color+depth targets that 3D content
//!                 renders into before being blitted onto egui's pass.

mod buffers;
mod camera;
mod pipelines;

pub use camera::{build_mvp, headlight_dir};

use katana_core::planner::MoveKind;
use wgpu::util::DeviceExt;
use wgpu::*;

// ---------------------------------------------------------------------------
// Offscreen render targets
// ---------------------------------------------------------------------------
//
// egui_wgpu's render pass has color but no depth attachment. To render 3D
// content with depth testing, we own a color+depth texture pair, render
// into them during `prepare`, then blit the color onto egui's pass during
// `paint`. This is the wgpu equivalent of the GL renderer's FBO.

/// Offscreen color + depth, sized to the central panel rect.
pub struct OffscreenTargets {
    pub color_view: TextureView,
    pub depth_view: TextureView,
    pub color_format: TextureFormat,
    pub width: u32,
    pub height: u32,
    // Textures are kept alive so the views remain valid; not used directly outside of resize
    _color_texture: Texture,
    _depth_texture: Texture,
}

impl OffscreenTargets {
    pub fn new(device: &Device, color_format: TextureFormat, width: u32, height: u32) -> Self {
        let color_texture = create_color_texture(device, color_format, width, height);
        let depth_texture = create_depth_texture(device, width, height);
        let color_view = color_texture.create_view(&TextureViewDescriptor::default());
        let depth_view = depth_texture.create_view(&TextureViewDescriptor::default());
        Self {
            color_view,
            depth_view,
            color_format,
            width,
            height,
            _color_texture: color_texture,
            _depth_texture: depth_texture,
        }
    }

    /// Recreate textures if dimensions changed. Returns true if the textures
    /// were actually recreated (so callers can invalidate dependent resources
    /// like sampling bind groups). No-op + returns false otherwise.
    pub fn resize(&mut self, device: &Device, width: u32, height: u32) -> bool {
        if self.width == width && self.height == height {
            return false;
        }
        *self = Self::new(device, self.color_format, width, height);
        true
    }
}

fn create_color_texture(device: &Device, format: TextureFormat, w: u32, h: u32) -> Texture {
    device.create_texture(
        &(TextureDescriptor {
            label: Some("offscreen_color"),
            size: Extent3d {
                width: w.max(1), // never create a 0×0 texture
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1, // no MSAA
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }),
    )
}

fn create_depth_texture(device: &Device, w: u32, h: u32) -> Texture {
    device.create_texture(
        &(TextureDescriptor {
            label: Some("offscreen_depth"),
            size: Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: pipelines::DEPTH_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        }),
    )
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

use std::mem::size_of;

use buffers::{FrameUniforms, GpuBuffer, InstancedBatch, LineBatch};

// ---------------------------------------------------------------------------
// Rhombus vertex table — compile-time constant, 144 bytes, written once.
// ---------------------------------------------------------------------------
//
// Each of 36 vertices is packed into one u32: cross_idx (bits 0-1),
// along_f (bit 2), norm_idx (bits 3-5).
//   cross_idx - which of 4 cross-section offsets {R, T, L, B}
//   along_f   - 0 = start of segment, 1 = end of segment
//   norm_idx  - which of 6 normals {N_RT, N_TL, N_LB, N_BR, N_NEG, N_POS}

const R: u32 = 0;
const T: u32 = 1;
const L: u32 = 2;
const B: u32 = 3;
const N_RT: u32 = 0;
const N_TL: u32 = 1;
const N_LB: u32 = 2;
const N_BR: u32 = 3;
const N_NEG: u32 = 4;
const N_POS: u32 = 5;

const fn pack(cross: u32, along: u32, norm: u32) -> u32 {
    cross | (along << 2) | (norm << 3)
}

#[rustfmt::skip]
const VERTEX_TABLE: [u32; 36] = [
    // Side face right-top (triangles 0-1)
    pack(R, 0, N_RT), pack(T, 0, N_RT), pack(R, 1, N_RT),
    pack(T, 0, N_RT), pack(T, 1, N_RT), pack(R, 1, N_RT),
    // Side face top-left (triangles 2-3)
    pack(T, 0, N_TL), pack(L, 0, N_TL), pack(T, 1, N_TL),
    pack(L, 0, N_TL), pack(L, 1, N_TL), pack(T, 1, N_TL),
    // Side face left-bottom (triangles 4-5)
    pack(L, 0, N_LB), pack(B, 0, N_LB), pack(L, 1, N_LB),
    pack(B, 0, N_LB), pack(B, 1, N_LB), pack(L, 1, N_LB),
    // Side face bottom-right (triangles 6-7)
    pack(B, 0, N_BR), pack(R, 0, N_BR), pack(B, 1, N_BR),
    pack(R, 0, N_BR), pack(R, 1, N_BR), pack(B, 1, N_BR),
    // Start cap (triangles 8-9), normal = -seg_dir
    pack(R, 0, N_NEG), pack(T, 0, N_NEG), pack(L, 0, N_NEG),
    pack(R, 0, N_NEG), pack(L, 0, N_NEG), pack(B, 0, N_NEG),
    // End cap (triangles 10-11), normal = +seg_dir
    pack(R, 1, N_POS), pack(L, 1, N_POS), pack(T, 1, N_POS),
    pack(R, 1, N_POS), pack(B, 1, N_POS), pack(L, 1, N_POS),
];

use crate::renderer_wgpu_port::buffers::{LineVertex, MeshVertex, RhombusInstance};

/// Brightness multiplier applied to layers *below* the one being scrubbed
const SCRUB_DIM: f32 = 0.25;

/// Nozzle radial segments
const NOZZLE_SEGMENTS: u32 = 12;
/// Vertices written for the nozzle each frame: cone side + top cap, 3 per tri.
const NOZZLE_VERT_COUNT: u32 = NOZZLE_SEGMENTS * 3 * 2;
/// Metallic grey, opaque
const NOZZLE_COLOR: [f32; 4] = [0.75, 0.76, 0.8, 1.0];

// ---------------------------------------------------------------------------
// Heatbed (build plate) appearance
// ---------------------------------------------------------------------------
/// Filled plate surface
const BED_PLATE_COLOR: [f32; 4] = [0.14, 0.14, 0.20, 1.0];
/// Interior grid lines
const BED_GRID_COLOR: [f32; 4] = [0.28, 0.28, 0.38, 1.0];
/// Outer border of the plate
const BED_BORDER_COLOR: [f32; 4] = [0.45, 0.45, 0.60, 1.0];
/// Grid spacing in mm.
const BED_GRID_SPACING: f32 = 10.0;

/// Print-head positions over time for a single layer, in print order with travel and extrusion merged.
struct LayerHeadTrack {
    layer_z: f32,
    times: Vec<f32>,
    points: Vec<[f32; 2]>,
}

pub struct Renderer {
    // wgpu's `Device` is cheap to clone (internal Arc); kept here so `paint`
    // can build a transient bind group without threading device through.
    device: Device,

    // We need two uniform buffers and bind groups: one with no z-clipping (for BG),
    // and one with the user's clip range (for FG). That's because queue.write_buffer
    // calls happen before any painting is done (we can't paint BG, change uniforms,
    // then paint FG, which is what we need for clipping)
    frame_uniform_buffer_bg: Buffer,
    frame_bind_group_bg: BindGroup,
    frame_uniform_buffer_fg: Buffer,
    frame_bind_group_fg: BindGroup,

    // Rhombus pipeline bind groups include the frame uniform (binding 0),
    // the static vertex-table (binding 1), and the color palette (binding 2).
    // Lifetime holders — the bind group owns the GPU references.
    _vertex_table_buffer: Buffer,
    _palette_buffer: Buffer,
    _rhombus_bind_group_bg: BindGroup, // reserved for a future BG rhombus pass
    rhombus_bind_group_fg: BindGroup,

    // Six pipelines (opaque + transparent variant per geometry type), built once at startup.
    line_opaque_pipeline: RenderPipeline,
    line_transparent_pipeline: RenderPipeline,
    mesh_opaque_pipeline: RenderPipeline,
    _mesh_transparent_pipeline: RenderPipeline, // unused today, kept for parity
    rhombus_opaque_pipeline: RenderPipeline,
    _rhombus_transparent_pipeline: RenderPipeline, // unused today, kept for parity

    // 4th pipeline + its bind group layout + sampler: copies the offscreen
    // color texture onto egui's pass in `paint`. Sampler is static; bind
    // group must be rebuilt when the offscreen texture is recreated (resize).
    blit_pipeline: RenderPipeline,
    blit_bgl: BindGroupLayout,
    blit_sampler: Sampler,
    // Cached blit bind group. Built once in `new`, invalidated on resize.
    // `paint` reads it without allocating; was previously rebuilt every frame.
    blit_bind_group: BindGroup,

    // Offscreen color+depth that 3D content renders into.
    offscreen: OffscreenTargets,

    // Static geometry uploaded by the public upload_* methods. `None` until
    // the first upload call. Public so main.rs can check for presence if needed.
    pub mesh_buffer: Option<GpuBuffer>,
    pub slices_buffer: Option<LineBatch>,
    pub current_slice_buffer: Option<GpuBuffer>,
    pub toolpath_lines_buffer: Option<LineBatch>, // travel moves
    pub toolpath_path_lines_buffer: Option<LineBatch>, // toolpath as flat lines
    pub toolpath_rhombuses: Option<InstancedBatch>,
    half_height: f32,
    half_width: f32,

    // Public flags driving per-frame draw decisions. Set by the egui side
    // each frame before the callback runs.
    pub clip_z_max: f32,
    pub clip_z_min: f32,
    pub draw_contours: bool,
    pub draw_toolpaths: bool,
    pub show_travel_moves: bool,
    pub show_filaments: bool,

    // Intra-layer scrubber, normalized [0, 1]
    pub scrub_fraction: f32,
    // When true, layers below the scrubbed top layer are dimmed to highlight it.
    pub is_scrubbing: bool,
    // World z of the layer currently being scrubbed
    pub scrub_top_z: f32,

    // Per-layer print-head timelines, ascending by layer_z.
    head_tracks: Vec<LayerHeadTrack>,
    nozzle_buffer: GpuBuffer,

    // Heatbed (build plate): a filled quad + grid/border lines, rebuilt via
    // `upload_bed` whenever a model is (re)loaded so the plate sits under it.
    // Drawn unconditionally in the background pass on every frame.
    bed_plate_buffer: Option<GpuBuffer>,
    bed_grid_buffer: Option<GpuBuffer>,
}

impl Renderer {
    pub fn new(
        device: Device,
        _queue: Queue,
        color_format: TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let frame_bgl = pipelines::build_frame_bgl(&device);
        let rhombus_bgl = pipelines::build_rhombus_bgl(&device);

        // Small factories for uniform buffer and bind group
        let make_uniform_buffer = |label: &str| -> Buffer {
            device.create_buffer(
                &(BufferDescriptor {
                    label: Some(label),
                    size: size_of::<FrameUniforms>() as BufferAddress,
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            )
        };
        let make_bind_group = |label: &str, buf: &Buffer| -> BindGroup {
            device.create_bind_group(
                &(BindGroupDescriptor {
                    label: Some(label),
                    layout: &frame_bgl,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                }),
            )
        };
        let frame_uniform_buffer_bg = make_uniform_buffer("frame_uniform_buffer_bg");
        let frame_uniform_buffer_fg = make_uniform_buffer("frame_uniform_buffer_fg");
        let frame_bind_group_bg = make_bind_group("frame_bind_group_bg", &frame_uniform_buffer_bg);
        let frame_bind_group_fg = make_bind_group("frame_bind_group_fg", &frame_uniform_buffer_fg);

        // Upload the static 36-entry vertex table once; never rewritten.
        let table_bytes: &[u8] = bytemuck::cast_slice(&VERTEX_TABLE);
        let vertex_table_buffer = device.create_buffer_init(
            &(wgpu::util::BufferInitDescriptor {
                label: Some("vertex_table"),
                contents: table_bytes,
                usage: BufferUsages::UNIFORM,
            }),
        );

        // Upload the static color palette (binding 2); never rewritten.
        let palette_uniforms = buffers::PaletteUniforms {
            colors: buffers::COLOR_PALETTE,
        };
        let palette_buffer = device.create_buffer_init(
            &(wgpu::util::BufferInitDescriptor {
                label: Some("palette_buffer"),
                contents: bytemuck::bytes_of(&palette_uniforms),
                usage: BufferUsages::UNIFORM,
            }),
        );

        let make_rhombus_bind_group = |label: &str, frame_buf: &Buffer| -> BindGroup {
            device.create_bind_group(
                &(BindGroupDescriptor {
                    label: Some(label),
                    layout: &rhombus_bgl,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: frame_buf.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: vertex_table_buffer.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: palette_buffer.as_entire_binding(),
                        },
                    ],
                }),
            )
        };
        let rhombus_bind_group_bg =
            make_rhombus_bind_group("rhombus_bind_group_bg", &frame_uniform_buffer_bg);
        let rhombus_bind_group_fg =
            make_rhombus_bind_group("rhombus_bind_group_fg", &frame_uniform_buffer_fg);

        let line_opaque_pipeline =
            pipelines::build_line_opaque_pipeline(&device, &frame_bgl, color_format);
        let line_transparent_pipeline =
            pipelines::build_line_transparent_pipeline(&device, &frame_bgl, color_format);
        let mesh_opaque_pipeline =
            pipelines::build_mesh_opaque_pipeline(&device, &frame_bgl, color_format);
        let mesh_transparent_pipeline =
            pipelines::build_mesh_transparent_pipeline(&device, &frame_bgl, color_format);
        let rhombus_opaque_pipeline =
            pipelines::build_rhombus_opaque_pipeline(&device, &rhombus_bgl, color_format);
        let rhombus_transparent_pipeline =
            pipelines::build_rhombus_transparent_pipeline(&device, &rhombus_bgl, color_format);

        let blit_bgl = pipelines::build_blit_bgl(&device);
        let blit_pipeline = pipelines::build_blit_pipeline(&device, &blit_bgl, color_format);
        let blit_sampler = device.create_sampler(
            &(SamplerDescriptor {
                label: Some("blit_sampler"),
                mag_filter: FilterMode::Nearest,
                min_filter: FilterMode::Nearest,
                mipmap_filter: FilterMode::Nearest,
                ..Default::default()
            }),
        );

        let offscreen = OffscreenTargets::new(&device, color_format, width, height);
        let blit_bind_group =
            build_blit_bind_group(&device, &blit_bgl, &offscreen.color_view, &blit_sampler);

        let nozzle_buffer = GpuBuffer {
            buffer: device.create_buffer(
                &(BufferDescriptor {
                    label: Some("nozzle_vbo"),
                    size: (NOZZLE_VERT_COUNT as u64) * (size_of::<MeshVertex>() as u64),
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            ),
            vertex_count: NOZZLE_VERT_COUNT,
        };

        Self {
            device,
            frame_uniform_buffer_bg,
            frame_bind_group_bg,
            frame_uniform_buffer_fg,
            frame_bind_group_fg,
            _vertex_table_buffer: vertex_table_buffer,
            _palette_buffer: palette_buffer,
            _rhombus_bind_group_bg: rhombus_bind_group_bg,
            rhombus_bind_group_fg,
            line_opaque_pipeline,
            line_transparent_pipeline,
            mesh_opaque_pipeline,
            _mesh_transparent_pipeline: mesh_transparent_pipeline,
            rhombus_opaque_pipeline,
            _rhombus_transparent_pipeline: rhombus_transparent_pipeline,
            blit_pipeline,
            blit_bgl,
            blit_sampler,
            blit_bind_group,
            offscreen,
            mesh_buffer: None,
            slices_buffer: None,
            current_slice_buffer: None,
            toolpath_lines_buffer: None,
            toolpath_path_lines_buffer: None,
            toolpath_rhombuses: None,
            half_height: 0.1,
            half_width: 0.2,
            clip_z_max: 1e30,
            clip_z_min: -1e30,
            draw_contours: false,
            draw_toolpaths: true,
            show_travel_moves: true,
            show_filaments: true,
            scrub_fraction: 1.0,
            is_scrubbing: false,
            scrub_top_z: 0.0,
            head_tracks: Vec::new(),
            nozzle_buffer,
            bed_plate_buffer: None,
            bed_grid_buffer: None,
        }
    }

    /// Build the heatbed geometry: a filled plate quad plus a grid and border,
    /// centered at (`cx`, `cy`) on the `z` plane. `border_color` overrides the
    /// default border (e.g. red while the model spills off the plate).
    pub fn upload_bed(
        &mut self,
        width: f32,
        depth: f32,
        cx: f32,
        cy: f32,
        z: f32,
        border_color: Option<[f32; 4]>,
    ) {
        let border_color = border_color.unwrap_or(BED_BORDER_COLOR);
        let (hw, hd) = (width * 0.5, depth * 0.5);
        let (x0, x1) = (cx - hw, cx + hw);
        let (y0, y1) = (cy - hd, cy + hd);

        // Filled plate: two triangles on the z plane, normal pointing up. The
        // `layer_z` clip key is set to the plate's own z; the background pass
        // uses an unbounded clip range so it always survives the FS z-clip.
        let n = [0.0, 0.0, 1.0];
        let quad = |x: f32, y: f32| MeshVertex {
            pos: [x, y, z],
            normal: n,
            color: BED_PLATE_COLOR,
            layer_z: z,
        };
        let plate_verts = vec![
            quad(x0, y0), quad(x1, y0), quad(x1, y1),
            quad(x0, y0), quad(x1, y1), quad(x0, y1),
        ];
        self.bed_plate_buffer = Some(buffers::upload_mesh(&self.device, &plate_verts));

        // Grid + border, lifted a hair above the plate to avoid z-fighting.
        let gz = z + 0.01;
        let mut lines: Vec<LineVertex> = Vec::new();
        let mut push = |ax: f32, ay: f32, bx: f32, by: f32, c: [f32; 4]| {
            lines.push(LineVertex { pos: [ax, ay, gz], color: c });
            lines.push(LineVertex { pos: [bx, by, gz], color: c });
        };
        // Interior grid lines at multiples of the spacing from the center.
        let steps_x = (hw / BED_GRID_SPACING) as i32;
        for i in -steps_x..=steps_x {
            let x = cx + (i as f32) * BED_GRID_SPACING;
            if x > x0 && x < x1 {
                push(x, y0, x, y1, BED_GRID_COLOR);
            }
        }
        let steps_y = (hd / BED_GRID_SPACING) as i32;
        for i in -steps_y..=steps_y {
            let y = cy + (i as f32) * BED_GRID_SPACING;
            if y > y0 && y < y1 {
                push(x0, y, x1, y, BED_GRID_COLOR);
            }
        }
        // Border loop.
        push(x0, y0, x1, y0, border_color);
        push(x1, y0, x1, y1, border_color);
        push(x1, y1, x0, y1, border_color);
        push(x0, y1, x0, y0, border_color);

        self.bed_grid_buffer = Some(buffers::upload_lines(&self.device, &lines));
    }

    pub fn upload_mesh(&mut self, triangles: &[katana_core::mesh::Triangle]) {
        let mut verts: Vec<buffers::MeshVertex> = Vec::with_capacity(triangles.len() * 3);
        // TODO: move this color to a const or cli param
        let color = [0.35, 0.55, 0.75, 1.0];
        for tri in triangles {
            let n = &tri.normal;
            for v in &tri.vertices {
                verts.push(MeshVertex {
                    pos: [v.x, v.y, v.z],
                    normal: [n.x, n.y, n.z],
                    color,
                    layer_z: -1e30, // mesh always visible - bypasses the FS z-clip
                });
            }
        }
        self.mesh_buffer = Some(buffers::upload_mesh(&self.device, &verts));
    }

    pub fn upload_all_slices(&mut self, layers: &[katana_core::slicer::Layer], stride: usize) {
        // Calculate amount of verts
        let count = layers
            .iter()
            .enumerate()
            .filter(|(i, _)| i % stride == 0) // skip layer for LOD (unused now)
            .flat_map(|(_, l)| l.contours.iter()) // all layers -> all contours
            .filter(|c| c.points.len() >= 2) // skip degenerate contours (min is 2 points)
            .map(|c| c.points.len() * 2) // all contours -> all segments, segment is 2 verts each
            .sum();
        let mut verts: Vec<buffers::LineVertex> = Vec::with_capacity(count);
        // TODO: move this color to a const, config, or cli param
        let color = [0.31, 0.31, 0.47, 0.25];
        for (i, layer) in layers.iter().enumerate() {
            if i % stride != 0 {
                continue;
            }
            for contour in &layer.contours {
                let pts = &contour.points;
                for p in 0..pts.len() {
                    let point = &contour.points[p];
                    let next = &contour.points[(p + 1) % pts.len()];
                    verts.push(LineVertex {
                        pos: [point.x, point.y, layer.z],
                        color,
                    });
                    verts.push(LineVertex {
                        pos: [next.x, next.y, layer.z],
                        color,
                    });
                }
            }
        }
        self.slices_buffer = Some(buffers::upload_lines_batched(&self.device, &verts));
    }

    pub fn upload_current_slice(&mut self, layers: &[katana_core::slicer::Layer]) {
        // Calculate amount of verts
        let count = layers
            .iter()
            .flat_map(|l| l.contours.iter()) // all layers -> all contours
            .filter(|c| c.points.len() >= 2) // skip degenerate contours (min is 2 points)
            .map(|c| c.points.len() * 2) // all contours -> all segments, segment is 2 verts each
            .sum();
        let mut verts: Vec<buffers::LineVertex> = Vec::with_capacity(count);
        // TODO: move this color to a const, config, or cli param
        let color = [0.91, 0.27, 0.38, 1.0];
        for layer in layers {
            for contour in &layer.contours {
                let pts = &contour.points;
                for p in 0..pts.len() {
                    let point = &contour.points[p];
                    let next = &contour.points[(p + 1) % pts.len()];
                    verts.push(LineVertex {
                        pos: [point.x, point.y, layer.z],
                        color,
                    });
                    verts.push(LineVertex {
                        pos: [next.x, next.y, layer.z],
                        color,
                    });
                }
            }
        }
        self.current_slice_buffer = Some(buffers::upload_lines(&self.device, &verts));
    }

    pub fn upload_planned_toolpath(
        &mut self,
        planned_layers: &[katana_core::planner::PlannedLayer],
        nozzle_width: f32,
        layer_height: f32,
    ) {
        self.half_height = layer_height * 0.5;
        self.half_width = nozzle_width * 0.5;

        // Three concatenated buffers, each built in print order across all
        // layers, plus the parallel cumulative-time arrays that drive scrubbing.
        let mut travel_verts: Vec<LineVertex> = Vec::new();
        let mut travel_seg_times: Vec<f32> = Vec::new();
        let mut travel_entries: Vec<buffers::LineLayerEntry> = Vec::new();

        let mut path_verts: Vec<LineVertex> = Vec::new();
        let mut path_seg_times: Vec<f32> = Vec::new();
        let mut path_entries: Vec<buffers::LineLayerEntry> = Vec::new();

        let mut rhombus_instances: Vec<RhombusInstance> = Vec::new();
        let mut rhombus_times: Vec<f32> = Vec::new();
        let mut rhombus_start_times: Vec<f32> = Vec::new();
        let mut rhombus_entries: Vec<buffers::LayerEntry> = Vec::new();

        let mut head_tracks: Vec<LayerHeadTrack> = Vec::new();

        for pl in planned_layers {
            // One timeline per layer, accumulating travel + extrusion time in
            // the order the nozzle actually prints them.
            let mut elapsed_s = 0.0f32;
            let mut track = LayerHeadTrack {
                layer_z: pl.z,
                times: Vec::new(),
                points: Vec::new(),
            };
            let travel_v0 = travel_verts.len();
            let path_v0 = path_verts.len();
            let rho0 = rhombus_instances.len();

            for mv in &pl.moves {
                let pts = &mv.points;
                if pts.len() < 2 {
                    continue;
                }
                let speed = mv.speed.max(1e-3); // guard against div-by-zero
                let (color_id, flags, line_color): (u8, u8, [f32; 4]) = match mv.kind {
                    MoveKind::Travel => (3, 3, [1.0, 0.8, 0.2, 0.4]),
                    MoveKind::Perimeter => (0, 0, [0.91, 0.27, 0.38, 1.0]),
                    MoveKind::Infill => (1, 1, [0.27, 0.91, 0.38, 1.0]),
                    MoveKind::SurfaceInfill => (2, 2, [0.9, 0.2, 0.7, 1.0]),
                };

                // Perimeters with >= 3 points are closed loops (wrap the last
                // segment back to the start); everything else is sequential.
                let is_closed = mv.kind == MoveKind::Perimeter && pts.len() >= 3;
                let seg_count = if is_closed { pts.len() } else { pts.len() - 1 };

                for s in 0..seg_count {
                    let a = pts[s];
                    let b = pts[(s + 1) % pts.len()];
                    let dx = b.x - a.x;
                    let dy = b.y - a.y;
                    let seg_len = (dx * dx + dy * dy).sqrt();
                    if seg_len < 1e-9 {
                        continue;
                    }
                    // Whole segment, stamped with the print time at its end. Time
                    // (not segment count) drives scrubbing: a long fast infill
                    // run advances the same wall-clock as a short slow perimeter.
                    // Mid-segment smoothing is a top-layer-only concern, handled
                    // separately, so lower layers stay one-primitive-per-segment.
                    // Seed the timeline with the layer's very first point at t=0,
                    // then stamp each segment's endpoint at its arrival time.
                    if track.points.is_empty() {
                        track.times.push(0.0);
                        track.points.push([a.x, a.y]);
                    }
                    elapsed_s += seg_len / speed;
                    track.times.push(elapsed_s);
                    track.points.push([b.x, b.y]);

                    if mv.kind == MoveKind::Travel {
                        travel_verts.push(LineVertex {
                            pos: [a.x, a.y, pl.z],
                            color: line_color,
                        });
                        travel_verts.push(LineVertex {
                            pos: [b.x, b.y, pl.z],
                            color: line_color,
                        });
                        travel_seg_times.push(elapsed_s);
                    } else {
                        // Extrusion: emit both the flat path line and the 3D rhombus.
                        path_verts.push(LineVertex {
                            pos: [a.x, a.y, pl.z],
                            color: line_color,
                        });
                        path_verts.push(LineVertex {
                            pos: [b.x, b.y, pl.z],
                            color: line_color,
                        });
                        path_seg_times.push(elapsed_s);

                        rhombus_instances.push(RhombusInstance {
                            start: [a.x, a.y, pl.z],
                            dir: [dx / seg_len, dy / seg_len],
                            length: seg_len,
                            color_flags: (color_id as u32) | ((flags as u32) << 8),
                        });
                        rhombus_times.push(elapsed_s);
                        rhombus_start_times.push(elapsed_s - seg_len / speed);
                    }
                }
            }

            // Same full-layer time stamped on every buffer's entry, so one
            // threshold cuts travel and extrusion at the same instant.
            let layer_total = elapsed_s;
            if !track.points.is_empty() {
                head_tracks.push(track);
            }
            if travel_verts.len() > travel_v0 {
                travel_entries.push(buffers::LineLayerEntry {
                    layer_z: pl.z,
                    first_vertex: travel_v0 as u32,
                    vertex_count: (travel_verts.len() - travel_v0) as u32,
                    time_total: layer_total,
                });
            }
            if path_verts.len() > path_v0 {
                path_entries.push(buffers::LineLayerEntry {
                    layer_z: pl.z,
                    first_vertex: path_v0 as u32,
                    vertex_count: (path_verts.len() - path_v0) as u32,
                    time_total: layer_total,
                });
            }
            if rhombus_instances.len() > rho0 {
                rhombus_entries.push(buffers::LayerEntry {
                    layer_z: pl.z,
                    instance_count: (rhombus_instances.len() - rho0) as i32,
                    time_total: layer_total,
                });
            }
        }

        // Upload
        self.toolpath_lines_buffer = (!travel_verts.is_empty()).then(|| {
            buffers::make_line_batch(
                &self.device,
                &travel_verts,
                travel_entries,
                travel_seg_times,
            )
        });
        self.toolpath_path_lines_buffer = (!path_verts.is_empty()).then(|| {
            buffers::make_line_batch(&self.device, &path_verts, path_entries, path_seg_times)
        });
        self.toolpath_rhombuses = (!rhombus_instances.is_empty()).then(|| {
            buffers::make_instanced_batch(
                &self.device,
                &rhombus_instances,
                rhombus_entries,
                rhombus_times,
                rhombus_start_times,
            )
        });
        self.head_tracks = head_tracks;
    }

    /// Per-frame: write uniforms, resize offscreen if needed, render the 3D
    /// scene into the offscreen color+depth pair. The blit onto egui's pass
    /// happens later in `paint`.
    pub fn prepare(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        mvp: &[f32; 16],
        light_dir: &[f32; 3],
        bg_mode: super::BgMode,
        viewport_w: u32,
        viewport_h: u32,
    ) {
        // resize offscreen to match the viewport size.
        let resized = self.offscreen.resize(device, viewport_w, viewport_h);
        if resized {
            self.blit_bind_group = build_blit_bind_group(
                device,
                &self.blit_bgl,
                &self.offscreen.color_view,
                &self.blit_sampler,
            );
        }

        // build the per-frame uniform values, twice (once with no clip for BG,
        // and once with the user's clip range (FG layer culling)
        // Remap GL NDC z [-1, +1] → wgpu NDC z [0, +1] by baking
        //   new_z = old_z * 0.5 + 0.5 * w
        // into the matrix's third row (column-major: indices 2/6/10/14).
        // build_mvp was authored for GL conventions; without this remap,
        // ~half the geometry's clip-space z falls outside [0, 1] and gets
        // culled by wgpu's depth-clip stage before the depth test runs.
        let mvp_mat: [[f32; 4]; 4] = [
            [mvp[0], mvp[1], mvp[2] * 0.5, mvp[3]],
            [mvp[4], mvp[5], mvp[6] * 0.5, mvp[7]],
            [mvp[8], mvp[9], mvp[10] * 0.5, mvp[11]],
            [mvp[12], mvp[13], mvp[14] * 0.5 + 0.5 * mvp[15], mvp[15]],
        ];
        let light_dir4 = [light_dir[0], light_dir[1], light_dir[2], 0.0];

        // Dimming is only active while scrubbing
        let fg_scrub_dim = if self.is_scrubbing { SCRUB_DIM } else { 1.0 };

        // In-progress top-layer segment to truncate
        let partial = self.rhombus_partial_cut();
        let (partial_index, partial_frac) = partial.map_or((u32::MAX, 1.0), |(i, f)| (i, f));

        let bg_uniforms = FrameUniforms {
            mvp: mvp_mat,
            light_dir: light_dir4,
            clip_z_max: 1e30,
            clip_z_min: -1e30,
            half_height: self.half_height,
            half_width: self.half_width,
            scrub_top_z: -1e30, // background never dims
            scrub_dim: 1.0,
            scrub_partial_index: u32::MAX, // background never truncates
            scrub_partial_frac: 1.0,
        };
        let fg_uniforms = FrameUniforms {
            mvp: mvp_mat,
            light_dir: light_dir4,
            clip_z_max: self.clip_z_max,
            clip_z_min: self.clip_z_min,
            half_height: self.half_height,
            half_width: self.half_width,
            scrub_top_z: self.scrub_top_z,
            scrub_dim: fg_scrub_dim,
            scrub_partial_index: partial_index,
            scrub_partial_frac: partial_frac,
        };
        queue.write_buffer(
            &self.frame_uniform_buffer_bg,
            0,
            bytemuck::bytes_of(&bg_uniforms),
        );
        queue.write_buffer(
            &self.frame_uniform_buffer_fg,
            0,
            bytemuck::bytes_of(&fg_uniforms),
        );

        // background pass
        {
            let mut pass = encoder.begin_render_pass(
                &(RenderPassDescriptor {
                    label: Some("bg_pass"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &self.offscreen.color_view,
                        resolve_target: None,
                        ops: Operations {
                            // TODO: move this color to a const/config, etc
                            load: LoadOp::Clear(Color {
                                r: 0.102,
                                g: 0.102,
                                b: 0.18,
                                a: 1.0,
                            }),
                            store: StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                        view: &self.offscreen.depth_view,
                        depth_ops: Some(Operations {
                            load: LoadOp::Clear(1.0),
                            store: StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                }),
            );
            pass.set_bind_group(0, &self.frame_bind_group_bg, &[]);

            // Heatbed: drawn first (as the ground plane) on every frame,
            // independent of bg_mode, using the no-clip background uniforms.
            if let Some(plate) = &self.bed_plate_buffer {
                pass.set_pipeline(&self.mesh_opaque_pipeline);
                pass.set_vertex_buffer(0, plate.buffer.slice(..));
                pass.draw(0..plate.vertex_count, 0..1);
            }
            if let Some(grid) = &self.bed_grid_buffer {
                pass.set_pipeline(&self.line_opaque_pipeline);
                pass.set_vertex_buffer(0, grid.buffer.slice(..));
                pass.draw(0..grid.vertex_count, 0..1);
            }

            match bg_mode {
                super::BgMode::Mesh => {
                    if let Some(m) = &self.mesh_buffer {
                        pass.set_pipeline(&self.mesh_opaque_pipeline);
                        pass.set_vertex_buffer(0, m.buffer.slice(..));
                        pass.draw(0..m.vertex_count, 0..1);
                    }
                }
                super::BgMode::Layers => {
                    if let Some(s) = &self.slices_buffer {
                        pass.set_pipeline(&self.line_transparent_pipeline);
                        draw_line_batch(&mut pass, s, -1e30, 1e30, 1.0); // BG: no clip, full
                    }
                }
                super::BgMode::None => {}
            }
        }

        // While scrubbing, place the nozzle over the current print-head position
        let head = if self.is_scrubbing {
            self.nozzle_head_position()
        } else {
            None
        };
        let draw_nozzle = head.is_some();
        if let Some([hx, hy]) = head {
            let tip = [hx, hy, self.scrub_top_z + self.half_height];
            let radius = self.half_width * 6.0;
            let height = radius * 4.0;
            let verts = build_nozzle_verts(tip, radius, height, self.scrub_top_z);
            queue.write_buffer(&self.nozzle_buffer.buffer, 0, bytemuck::cast_slice(&verts));
        }

        // Foreground pass: load color, clear depth
        {
            let mut pass = encoder.begin_render_pass(
                &(RenderPassDescriptor {
                    label: Some("fg_pass"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &self.offscreen.color_view,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Load,
                            store: StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                        view: &self.offscreen.depth_view,
                        depth_ops: Some(Operations {
                            load: LoadOp::Clear(1.0),
                            store: StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                }),
            );
            pass.set_bind_group(0, &self.frame_bind_group_fg, &[]);

            // 2 view modes: `Contours` and `Toolpaths`
            if self.draw_contours {
                if let Some(c) = &self.current_slice_buffer {
                    pass.set_pipeline(&self.line_opaque_pipeline);
                    pass.set_vertex_buffer(0, c.buffer.slice(..));
                    pass.draw(0..c.vertex_count, 0..1);
                }
            } else if self.draw_toolpaths {
                // Conditionally draw filament rhombuses
                if self.show_filaments {
                    if let Some(rhombuses) = &self.toolpath_rhombuses {
                        draw_rhombus_batch(
                            &mut pass,
                            &self.rhombus_opaque_pipeline,
                            &self.rhombus_bind_group_fg,
                            rhombuses,
                            self.clip_z_max,
                            self.clip_z_min,
                            self.scrub_fraction,
                            partial_index,
                        );
                    }
                } else {
                    // Otherwise draw simple line toolpaths
                    if let Some(plb) = &self.toolpath_path_lines_buffer {
                        pass.set_pipeline(&self.line_transparent_pipeline);
                        draw_line_batch(
                            &mut pass,
                            plb,
                            self.clip_z_min,
                            self.clip_z_max,
                            self.scrub_fraction,
                        );
                    }
                }

                // Conditionally draw travel moves (draw last because they're translucent)
                if self.show_travel_moves {
                    if self.show_filaments {
                        // rhombus_bind_group_fg (2 bindings) was active; line pipeline uses
                        // frame_bgl (1 binding) — must rebind the compatible frame bind group.
                        pass.set_pipeline(&self.line_transparent_pipeline);
                        pass.set_bind_group(0, &self.frame_bind_group_fg, &[]);
                    }
                    if let Some(lb) = &self.toolpath_lines_buffer {
                        draw_line_batch(
                            &mut pass,
                            lb,
                            self.clip_z_min,
                            self.clip_z_max,
                            self.scrub_fraction,
                        );
                    }
                }
            }

            if draw_nozzle {
                pass.set_pipeline(&self.mesh_opaque_pipeline);
                pass.set_bind_group(0, &self.frame_bind_group_fg, &[]);
                pass.set_vertex_buffer(0, self.nozzle_buffer.buffer.slice(..));
                pass.draw(0..self.nozzle_buffer.vertex_count, 0..1);
            }
        }
    }

    /// Per-frame: blit the offscreen color texture onto egui's render pass.
    /// Called from inside the `egui_wgpu::CallbackFn::paint` closure.
    ///
    /// `render_pass` is egui's existing pass, color attachment is already set,
    /// no depth attachment. We add one fullscreen-triangle draw to it.
    pub fn paint(&self, render_pass: &mut RenderPass<'_>) {
        render_pass.set_pipeline(&self.blit_pipeline);
        render_pass.set_bind_group(0, &self.blit_bind_group, &[]);
        render_pass.draw(0..3, 0..1); // 3 verts, 1 instance
    }



    /// XY of the print head at the current scrub instant on the top visible layer, or `None` when there's no toolpath to follow.
    fn nozzle_head_position(&self) -> Option<[f32; 2]> {
        let idx = self
            .head_tracks
            .partition_point(|t| t.layer_z <= self.clip_z_max);
        let track = self.head_tracks.get(idx.checked_sub(1)?)?;
        let total = *track.times.last()?;
        let threshold = self.scrub_fraction * total;
        Some(head_position_at(track, threshold))
    }

    /// Returns the index of the current rhombus being scrubbed and the fraction of it we need to show
    fn rhombus_partial_cut(&self) -> Option<(u32, f32)> {
        if !self.is_scrubbing {
            return None;
        }
        let batch = self.toolpath_rhombuses.as_ref()?;
        let scrubbed_layer = batch
            .layer_entries
            .partition_point(|e| e.layer_z <= self.clip_z_max)
            .checked_sub(1)?;
        let entry = batch.layer_entries.get(scrubbed_layer)?;
        let instance_count = entry.instance_count as usize;
        let instances_below: usize = batch.layer_entries[..scrubbed_layer]
            .iter()
            .map(|e| e.instance_count as usize)
            .sum();

        let threshold = self.scrub_fraction * entry.time_total;
        // Whole segments completed by `threshold`
        let draw_count = batch.instance_times[instances_below..instances_below + instance_count]
            .partition_point(|&t| t <= threshold);
        if draw_count >= instance_count {
            return None; // layer fully printed; no segment in progress
        }
        // instance index
        let gidx = instances_below + draw_count;
        let start_t = batch.instance_start_times[gidx];
        let end_t = batch.instance_times[gidx];
        if threshold <= start_t {
            return None; // head is on a travel move before this extrusion starts
        }
        let span = end_t - start_t;
        // instance fraction
        let frac = if span > 1e-9 {
            (threshold - start_t) / span
        } else {
            1.0
        };
        Some((gidx as u32, frac.clamp(0.0, 1.0)))
    }
}

/// Returns the position to sit the nozzle tip over. `track` is guaranteed non-empty.
fn head_position_at(track: &LayerHeadTrack, threshold: f32) -> [f32; 2] {
    // `reached` = count of samples at or before threshold = index of the first
    // sample after it. The head sits on segment [reached-1, reached], advanced
    // through it by how far `threshold` falls into that segment's time span.
    let reached = track.times.partition_point(|&t| t <= threshold);
    if reached == 0 {
        return track.points[0]; // before the first sample (times[0] == 0)
    }
    if reached >= track.points.len() {
        return track.points[track.points.len() - 1]; // at/past the final sample
    }
    let (p0, p1) = (track.points[reached - 1], track.points[reached]);
    let (t0, t1) = (track.times[reached - 1], track.times[reached]);
    let span = t1 - t0;
    let f = if span > 1e-9 {
        (threshold - t0) / span
    } else {
        0.0
    };
    [p0[0] + (p1[0] - p0[0]) * f, p0[1] + (p1[1] - p0[1]) * f]
}

/// Build a downward-pointing cone (the nozzle): tip at `tip`, opening upward to
/// a circle of `radius` at z = `tip.z + height`. Flat-shaded (one face normal
/// per triangle). Every vertex carries `clip_z` as its `layer_z` clip key so it
/// survives the mesh shader's fragment z-clip — pass a value inside the active
/// [clip_z_min, clip_z_max] range. Returns exactly `NOZZLE_VERT_COUNT` verts.
fn build_nozzle_verts(tip: [f32; 3], radius: f32, height: f32, clip_z: f32) -> Vec<MeshVertex> {
    let n = NOZZLE_SEGMENTS;
    let base_z = tip[2] + height;
    let center_top = [tip[0], tip[1], base_z];

    // i-th point on the base ring (the wide, upper end of the cone).
    let ring = |i: u32| -> [f32; 3] {
        let a = (((i % n) as f32) / (n as f32)) * std::f32::consts::TAU;
        [tip[0] + radius * a.cos(), tip[1] + radius * a.sin(), base_z]
    };

    let mut verts: Vec<MeshVertex> = Vec::with_capacity(NOZZLE_VERT_COUNT as usize);
    let mut push_tri = |p0: [f32; 3], p1: [f32; 3], p2: [f32; 3]| {
        // Face normal from two edges (cull_mode is None, so winding is cosmetic).
        let u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let v = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let mut nrm = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let len = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2])
            .sqrt()
            .max(1e-6);
        nrm = [nrm[0] / len, nrm[1] / len, nrm[2] / len];
        for p in [p0, p1, p2] {
            verts.push(MeshVertex {
                pos: p,
                normal: nrm,
                color: NOZZLE_COLOR,
                layer_z: clip_z,
            });
        }
    };

    for i in 0..n {
        let a = ring(i);
        let b = ring(i + 1);
        push_tri(tip, a, b); // cone side
        push_tri(center_top, b, a); // top cap closes the wide end
    }
    verts
}

// ---------------------------------------------------------------------------
// Rhombus batch draw with run merging.
// ---------------------------------------------------------------------------
//
// All instances live in one buffer, sorted by layer_z. We draw the visible
// layers in [clip_z_min, clip_z_max], merging contiguous layers into a single
// `pass.draw(0..36, first_instance..first_instance + total)` — the wgpu
// equivalent of glDrawArraysInstancedBaseInstance with no per-layer VAOs.
fn draw_rhombus_batch(
    pass: &mut RenderPass<'_>,
    pipeline: &RenderPipeline,
    bind_group: &BindGroup,
    batch: &InstancedBatch,
    clip_z_max: f32,
    clip_z_min: f32,
    scrub_fraction: f32,
    partial_index: u32,
) {
    // Binary-search the visible layer range. layer_entries is sorted ascending.
    let start_idx = batch
        .layer_entries
        .partition_point(|e| e.layer_z < clip_z_min);
    let end_idx = batch
        .layer_entries
        .partition_point(|e| e.layer_z <= clip_z_max);
    if start_idx >= end_idx {
        return;
    }

    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_vertex_buffer(0, batch.buffer.slice(..));

    // Cumulative offset into the instance buffer for the first visible layer.
    let mut layer_first_instance: u32 = batch.layer_entries[..start_idx]
        .iter()
        .map(|e| e.instance_count as u32)
        .sum();

    let mut run_first: Option<u32> = None;
    let mut run_total: u32 = 0;

    // Instances are stored in planner move order within each layer, so the
    // scrubber just shortens the draw range of the *top* visible layer.
    let top_idx = end_idx - 1;

    for (i, entry) in batch.layer_entries[start_idx..end_idx].iter().enumerate() {
        let full = entry.instance_count as u32;
        // Only the topmost visible layer is scrubbed; layers below are full.
        // The cutoff is time-based: keep instances whose cumulative layer-time
        // is within `scrub_fraction` of the layer's total print time.
        let mut draw_count = if start_idx + i == top_idx && scrub_fraction < 1.0 {
            let threshold = scrub_fraction * entry.time_total;
            let lo = layer_first_instance as usize;
            let times = &batch.instance_times[lo..lo + (full as usize)];
            times.partition_point(|&t| t <= threshold) as u32
        } else {
            full
        };
        // Include the in-progress segment (top layer only; the shader shrinks it to its printed fraction)
        // Guarding on top_idx avoids a false match at a layer boundary when the in-progress segment is the top layer's first.
        if start_idx + i == top_idx && partial_index == layer_first_instance + draw_count {
            draw_count += 1;
        }

        if draw_count == 0 {
            // Flush the in-progress run, if any. (A zero-count top layer also
            // breaks the run — there's nothing after it to merge with anyway.)
            if let Some(rf) = run_first.take() {
                pass.draw(0..36, rf..rf + run_total);
                run_total = 0;
            }
        } else {
            // Extend (or start) the current run.
            if run_first.is_none() {
                run_first = Some(layer_first_instance);
            }
            run_total += draw_count;
        }
        // Advance by the FULL count so later layers' offsets stay correct even
        // when the top layer drew only a prefix of its instances.
        layer_first_instance += full;
    }
    // Flush any trailing run.
    if let Some(rf) = run_first {
        pass.draw(0..36, rf..rf + run_total);
    }
}

// ---------------------------------------------------------------------------
// Line batch draw with CPU-side layer culling.
// ---------------------------------------------------------------------------
//
// Mirrors draw_rhombus_batch but for line geometry. Pipeline and bind group
// must be set by the caller before invoking this function.
fn draw_line_batch(
    pass: &mut RenderPass<'_>,
    batch: &LineBatch,
    clip_z_min: f32,
    clip_z_max: f32,
    scrub_fraction: f32,
) {
    // Binary search the visible layers
    let start_idx = batch
        .layer_entries
        .partition_point(|e| e.layer_z < clip_z_min);
    let end_idx = batch
        .layer_entries
        .partition_point(|e| e.layer_z <= clip_z_max);
    if start_idx >= end_idx {
        return;
    }

    pass.set_vertex_buffer(0, batch.buffer.slice(..));

    // Verts come in pairs (one line segment = 2 verts), in planner move order.
    // The scrubber shortens only the top visible layer, cutting on cumulative
    // print time so travel and extrusion advance together. `segment_times` is
    // empty for non-scrubbed batches (background slices) → always draw full.
    let top_idx = end_idx - 1;
    for (i, entry) in batch.layer_entries[start_idx..end_idx].iter().enumerate() {
        let scrub_top =
            start_idx + i == top_idx && scrub_fraction < 1.0 && !batch.segment_times.is_empty();
        let vcount = if scrub_top {
            let threshold = scrub_fraction * entry.time_total;
            let seg0 = (entry.first_vertex / 2) as usize;
            let seg_count = (entry.vertex_count / 2) as usize;
            let times = &batch.segment_times[seg0..seg0 + seg_count];
            (times.partition_point(|&t| t <= threshold) as u32) * 2
        } else {
            entry.vertex_count
        };
        if vcount == 0 {
            continue;
        }
        pass.draw(entry.first_vertex..entry.first_vertex + vcount, 0..1);
    }
}

/// Build the bind group used by the blit pipeline. Re-call whenever the
/// offscreen color view is recreated (i.e. after `OffscreenTargets::resize`
/// returns true).
fn build_blit_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    color_view: &TextureView,
    sampler: &Sampler,
) -> BindGroup {
    device.create_bind_group(
        &(BindGroupDescriptor {
            label: Some("blit_bind_group"),
            layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(color_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(sampler),
                },
            ],
        }),
    )
}
