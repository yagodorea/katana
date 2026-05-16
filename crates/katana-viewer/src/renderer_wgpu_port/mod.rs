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

pub use camera::{ build_mvp, headlight_dir };

use katana_core::planner::{ MoveKind, PointXY };
use wgpu::*;
use wgpu::util::DeviceExt;

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
        })
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
        })
    )
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

use std::mem::size_of;

use buffers::{ FrameUniforms, GpuBuffer, InstancedBatch };

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

use crate::renderer_wgpu_port::buffers::{ LineVertex, MeshVertex, RhombusInstance };

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

    // Rhombus pipeline bind groups include both the frame uniform (binding 0)
    // and the static vertex-table buffer (binding 1). Kept separate from the
    // shared frame bind groups because line/mesh pipelines don't have binding 1.
    // `_vertex_table_buffer` is held only for its lifetime; the bind group owns the GPU reference.
    _vertex_table_buffer: Buffer,
    _rhombus_bind_group_bg: BindGroup, // reserved for a future BG rhombus pass
    rhombus_bind_group_fg: BindGroup,

    // The three pipelines, built once at startup.
    line_pipeline: RenderPipeline,
    mesh_pipeline: RenderPipeline,
    rhombus_pipeline: RenderPipeline,

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
    pub slices_buffer: Option<GpuBuffer>,
    pub current_slice_buffer: Option<GpuBuffer>,
    pub toolpath_lines_buffer: Option<GpuBuffer>, // travel moves
    pub toolpath_path_lines_buffer: Option<GpuBuffer>, // toolpath as flat lines
    pub toolpath_rhombuses: Option<InstancedBatch>,
    half_height: f32,

    // Public flags driving per-frame draw decisions. Set by the egui side
    // each frame before the callback runs.
    pub clip_z_max: f32,
    pub clip_z_min: f32,
    pub draw_contours: bool,
    pub draw_toolpaths: bool,
    pub show_travel_moves: bool,
    pub show_filaments: bool,
}

impl Renderer {
    pub fn new(
        device: Device,
        _queue: Queue,
        color_format: TextureFormat,
        width: u32,
        height: u32
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
                })
            )
        };
        let make_bind_group = |label: &str, buf: &Buffer| -> BindGroup {
            device.create_bind_group(
                &(BindGroupDescriptor {
                    label: Some(label),
                    layout: &frame_bgl,
                    entries: &[BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
                })
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
            })
        );
        let make_rhombus_bind_group = |label: &str, frame_buf: &Buffer| -> BindGroup {
            device.create_bind_group(
                &(BindGroupDescriptor {
                    label: Some(label),
                    layout: &rhombus_bgl,
                    entries: &[
                        BindGroupEntry { binding: 0, resource: frame_buf.as_entire_binding() },
                        BindGroupEntry {
                            binding: 1,
                            resource: vertex_table_buffer.as_entire_binding(),
                        },
                    ],
                })
            )
        };
        let rhombus_bind_group_bg = make_rhombus_bind_group(
            "rhombus_bind_group_bg",
            &frame_uniform_buffer_bg
        );
        let rhombus_bind_group_fg = make_rhombus_bind_group(
            "rhombus_bind_group_fg",
            &frame_uniform_buffer_fg
        );

        let line_pipeline = pipelines::build_line_pipeline(&device, &frame_bgl, color_format);
        let mesh_pipeline = pipelines::build_mesh_pipeline(&device, &frame_bgl, color_format);
        let rhombus_pipeline = pipelines::build_rhombus_pipeline(
            &device,
            &rhombus_bgl,
            color_format
        );

        let blit_bgl = pipelines::build_blit_bgl(&device);
        let blit_pipeline = pipelines::build_blit_pipeline(&device, &blit_bgl, color_format);
        let blit_sampler = device.create_sampler(
            &(SamplerDescriptor {
                label: Some("blit_sampler"),
                mag_filter: FilterMode::Nearest,
                min_filter: FilterMode::Nearest,
                mipmap_filter: FilterMode::Nearest,
                ..Default::default()
            })
        );

        let offscreen = OffscreenTargets::new(&device, color_format, width, height);
        let blit_bind_group = build_blit_bind_group(
            &device,
            &blit_bgl,
            &offscreen.color_view,
            &blit_sampler
        );

        Self {
            device,
            frame_uniform_buffer_bg,
            frame_bind_group_bg,
            frame_uniform_buffer_fg,
            frame_bind_group_fg,
            _vertex_table_buffer: vertex_table_buffer,
            _rhombus_bind_group_bg: rhombus_bind_group_bg,
            rhombus_bind_group_fg,
            line_pipeline,
            mesh_pipeline,
            rhombus_pipeline,
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
            clip_z_max: 1e30,
            clip_z_min: -1e30,
            draw_contours: false,
            draw_toolpaths: true,
            show_travel_moves: true,
            show_filaments: true,
        }
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
                    verts.push(LineVertex { pos: [point.x, point.y, layer.z], color });
                    verts.push(LineVertex { pos: [next.x, next.y, layer.z], color });
                }
            }
        }
        self.slices_buffer = Some(buffers::upload_lines(&self.device, &verts));
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
                    verts.push(LineVertex { pos: [point.x, point.y, layer.z], color });
                    verts.push(LineVertex { pos: [next.x, next.y, layer.z], color });
                }
            }
        }
        self.current_slice_buffer = Some(buffers::upload_lines(&self.device, &verts));
    }

    pub fn upload_planned_toolpath(
        &mut self,
        planned_layers: &[katana_core::planner::PlannedLayer],
        nozzle_width: f32,
        layer_height: f32
    ) {
        self.half_height = layer_height * 0.5;
        // TODO: Evaluate the trade-off of estimating the count instead of actually doing it (trade memory for CPU)
        let mut line_count = 0 as usize;
        let mut path_line_count = 0 as usize;
        // First pass, count verts
        for pl in planned_layers {
            for mv in &pl.moves {
                let pts = mv.points.len();
                if pts < 2 {
                    continue;
                }
                match mv.kind {
                    MoveKind::Travel => {
                        line_count += 2;
                    }
                    MoveKind::Perimeter if pts >= 3 => {
                        path_line_count += pts * 2;
                    }
                    MoveKind::Perimeter | MoveKind::Infill | MoveKind::SurfaceInfill => {
                        path_line_count += 2;
                    }
                }
            }
        }
        let mut line_verts: Vec<buffers::LineVertex> = Vec::with_capacity(line_count);
        let mut path_line_verts: Vec<buffers::LineVertex> = Vec::with_capacity(path_line_count);
        let mut rhombus_instances: Vec<buffers::RhombusInstance> = Vec::with_capacity(
            path_line_count / 2
        );

        // Second pass, push verts
        for pl in planned_layers {
            for mv in &pl.moves {
                let pts = &mv.points;
                if pts.len() < 2 {
                    continue;
                }
                // TODO: move this color to a const or cli param
                let color = match mv.kind {
                    MoveKind::Travel => [1.0, 0.8, 0.2, 0.4],
                    MoveKind::Perimeter => [0.91, 0.27, 0.38, 1.0],
                    MoveKind::Infill => [0.27, 0.91, 0.38, 0.8],
                    MoveKind::SurfaceInfill => [0.9, 0.2, 0.7, 0.9],
                };
                match mv.kind {
                    MoveKind::Travel => {
                        line_verts.push(LineVertex { pos: [pts[0].x, pts[0].y, pl.z], color });
                        line_verts.push(LineVertex { pos: [pts[1].x, pts[1].y, pl.z], color });
                    }
                    MoveKind::Infill | MoveKind::SurfaceInfill => {
                        self.push_segment(
                            &pts[0],
                            &pts[1],
                            pl.z,
                            color,
                            nozzle_width,
                            &mut path_line_verts,
                            &mut rhombus_instances
                        );
                    }
                    MoveKind::Perimeter => {
                        if pts.len() == 2 {
                            // Open segment
                            self.push_segment(
                                &pts[0],
                                &pts[1],
                                pl.z,
                                color,
                                nozzle_width,
                                &mut path_line_verts,
                                &mut rhombus_instances
                            );
                        } else {
                            // closed loop
                            for i in 0..pts.len() {
                                let a = &pts[i];
                                let b = &pts[(i + 1) % pts.len()];
                                self.push_segment(
                                    a,
                                    b,
                                    pl.z,
                                    color,
                                    nozzle_width,
                                    &mut path_line_verts,
                                    &mut rhombus_instances
                                );
                            }
                        }
                    }
                }
            }
        }

        // Upload
        self.toolpath_lines_buffer = (!line_verts.is_empty()).then(||
            buffers::upload_lines(&self.device, &line_verts)
        );
        self.toolpath_path_lines_buffer = (!path_line_verts.is_empty()).then(||
            buffers::upload_lines(&self.device, &path_line_verts)
        );
        self.toolpath_rhombuses = (!rhombus_instances.is_empty()).then(||
            buffers::upload_rhombus_batch(&self.device, &rhombus_instances)
        );
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
        viewport_h: u32
    ) {
        // resize offscreen to match the viewport size.
        let resized = self.offscreen.resize(device, viewport_w, viewport_h);
        if resized {
            self.blit_bind_group = build_blit_bind_group(
                device,
                &self.blit_bgl,
                &self.offscreen.color_view,
                &self.blit_sampler
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

        let bg_uniforms = FrameUniforms {
            mvp: mvp_mat,
            light_dir: light_dir4,
            clip_z_max: 1e30,
            clip_z_min: -1e30,
            half_height: self.half_height,
            _pad: 0.0,
        };
        let fg_uniforms = FrameUniforms {
            mvp: mvp_mat,
            light_dir: light_dir4,
            clip_z_max: self.clip_z_max,
            clip_z_min: self.clip_z_min,
            half_height: self.half_height,
            _pad: 0.0,
        };
        queue.write_buffer(&self.frame_uniform_buffer_bg, 0, bytemuck::bytes_of(&bg_uniforms));
        queue.write_buffer(&self.frame_uniform_buffer_fg, 0, bytemuck::bytes_of(&fg_uniforms));

        // background pass
        {
            let mut pass = encoder.begin_render_pass(
                &(RenderPassDescriptor {
                    label: Some("bg_pass"),
                    color_attachments: &[
                        Some(RenderPassColorAttachment {
                            view: &self.offscreen.color_view,
                            resolve_target: None,
                            ops: Operations {
                                // TODO: move this color to a const/config, etc
                                load: LoadOp::Clear(Color { r: 0.102, g: 0.102, b: 0.18, a: 1.0 }),
                                store: StoreOp::Store,
                            },
                        }),
                    ],
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
                })
            );
            pass.set_bind_group(0, &self.frame_bind_group_bg, &[]);

            match bg_mode {
                super::BgMode::Mesh => {
                    if let Some(m) = &self.mesh_buffer {
                        pass.set_pipeline(&self.mesh_pipeline);
                        pass.set_vertex_buffer(0, m.buffer.slice(..));
                        pass.draw(0..m.vertex_count, 0..1);
                    }
                }
                super::BgMode::Layers => {
                    if let Some(s) = &self.slices_buffer {
                        pass.set_pipeline(&self.line_pipeline);
                        pass.set_vertex_buffer(0, s.buffer.slice(..));
                        pass.draw(0..s.vertex_count, 0..1);
                    }
                }
                super::BgMode::None => {}
            }
        }

        // Foreground pass: load color, clear depth
        {
            let mut pass = encoder.begin_render_pass(
                &(RenderPassDescriptor {
                    label: Some("fg_pass"),
                    color_attachments: &[
                        Some(RenderPassColorAttachment {
                            view: &self.offscreen.color_view,
                            resolve_target: None,
                            ops: Operations { load: LoadOp::Load, store: StoreOp::Store },
                        }),
                    ],
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
                })
            );
            pass.set_bind_group(0, &self.frame_bind_group_fg, &[]);

            // 2 view modes: `Contours` and `Toolpaths`
            if self.draw_contours {
                if let Some(c) = &self.current_slice_buffer {
                    pass.set_pipeline(&self.line_pipeline);
                    pass.set_vertex_buffer(0, c.buffer.slice(..));
                    pass.draw(0..c.vertex_count, 0..1);
                }
            } else if self.draw_toolpaths {
                // Conditionally draw filament rhombuses
                if self.show_filaments {
                    if let Some(rhombuses) = &self.toolpath_rhombuses {
                        draw_rhombus_batch(
                            &mut pass,
                            &self.rhombus_pipeline,
                            &self.rhombus_bind_group_fg,
                            rhombuses,
                            self.clip_z_max,
                            self.clip_z_min,
                            mvp,
                            self.half_height
                        );
                    }
                } else {
                    // Otherwise draw simple line toolpaths
                    if let Some(plb) = &self.toolpath_path_lines_buffer {
                        pass.set_pipeline(&self.line_pipeline);
                        pass.set_vertex_buffer(0, plb.buffer.slice(..));
                        pass.draw(0..plb.vertex_count, 0..1);
                    }
                }

                // Conditionally draw travel moves (draw last because they're translucent)
                if self.show_travel_moves {
                    if self.show_filaments {
                        // rhombus_bind_group_fg (2 bindings) was active; line pipeline uses
                        // frame_bgl (1 binding) — must rebind the compatible frame bind group.
                        pass.set_pipeline(&self.line_pipeline);
                        pass.set_bind_group(0, &self.frame_bind_group_fg, &[]);
                    }
                    if let Some(lb) = &self.toolpath_lines_buffer {
                        pass.set_vertex_buffer(0, lb.buffer.slice(..));
                        pass.draw(0..lb.vertex_count, 0..1);
                    }
                }
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

    fn push_segment(
        &mut self,
        a: &PointXY<f32>,
        b: &PointXY<f32>,
        z: f32,
        color: [f32; 4],
        nozzle_width: f32,
        path_line_verts: &mut Vec<LineVertex>,
        rhombus_instances: &mut Vec<RhombusInstance>
    ) {
        path_line_verts.push(LineVertex { pos: [a.x, a.y, z], color });
        path_line_verts.push(LineVertex { pos: [b.x, b.y, z], color });

        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let length = (dx * dx + dy * dy).sqrt();
        if length < 1e-9 {
            // skip degenerate segments
            return;
        }

        rhombus_instances.push(RhombusInstance {
            start: [a.x, a.y, z],
            direction: [dx / length, dy / length],
            scale: [length, nozzle_width * 0.5], // (length, half_width)
            color,
            layer_z: z,
        });
    }

    /// Extract 6 frustum planes from MVP matrix using Gribb-Hartmann derivation
    fn extract_frustum_planes(m: &[f32; 16]) -> [[f32; 4]; 6] {
        // m is column-major
        let get_row = |i: usize| -> [f32; 4] { [m[i], m[4 + i], m[8 + i], m[12 + i]] };
        let rows = [get_row(0), get_row(1), get_row(2), get_row(3)];

        let add = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
            [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
        };
        let sub = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
            [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
        };

        [
            add(rows[3], rows[0]), // left
            sub(rows[3], rows[0]), // right
            add(rows[3], rows[1]), // bottom
            sub(rows[3], rows[1]), // top
            add(rows[3], rows[2]), // near
            sub(rows[3], rows[2]), // far
        ]
    }

    /// Returns true if the AABB is entirely outside at least one frustum plane (safe to cull).
    fn aabb_outside_frustum(
        planes: &[[f32; 4]; 6],
        min_x: f32,
        min_y: f32,
        min_z: f32,
        max_x: f32,
        max_y: f32,
        max_z: f32
    ) -> bool {
        for p in planes {
            // P-vertex: AABB corner most aligned with the plane normal
            let px = if p[0] >= 0.0 { max_x } else { min_x };
            let py = if p[1] >= 0.0 { max_y } else { min_y };
            let pz = if p[2] >= 0.0 { max_z } else { min_z };
            if p[0] * px + p[1] * py + p[2] * pz + p[3] < 0.0 {
                return true; // most positive corner is behind this plane → fully outside
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Rhombus batch draw with CPU-side frustum culling + run merging.
// ---------------------------------------------------------------------------
//
// All instances live in one buffer, sorted by layer_z. We draw the visible
// layers in [clip_z_min, clip_z_max], skipping layers whose AABB is outside
// the view frustum, and merging contiguous non-culled layers into a single
// `pass.draw(0..36, first_instance..first_instance + total)` — the wgpu
// equivalent of glDrawArraysInstancedBaseInstance with no per-layer VAOs.

fn draw_rhombus_batch(
    pass: &mut RenderPass<'_>,
    pipeline: &RenderPipeline,
    bind_group: &BindGroup,
    batch: &InstancedBatch,
    clip_z_max: f32,
    clip_z_min: f32,
    mvp: &[f32; 16],
    half_height: f32
) {
    // Binary-search the visible layer range. layer_entries is sorted ascending.
    let start_idx = batch.layer_entries.partition_point(|e| e.layer_z < clip_z_min);
    let end_idx = batch.layer_entries.partition_point(|e| e.layer_z <= clip_z_max);
    if start_idx >= end_idx {
        return;
    }

    let planes = Renderer::extract_frustum_planes(mvp);

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

    for entry in &batch.layer_entries[start_idx..end_idx] {
        let outside = Renderer::aabb_outside_frustum(
            &planes,
            entry.aabb_min_x,
            entry.aabb_min_y,
            entry.layer_z - half_height,
            entry.aabb_max_x,
            entry.aabb_max_y,
            entry.layer_z + half_height
        );
        if outside {
            // Flush the in-progress run, if any.
            if let Some(rf) = run_first.take() {
                pass.draw(0..36, rf..rf + run_total);
                run_total = 0;
            }
        } else {
            // Extend (or start) the current run.
            if run_first.is_none() {
                run_first = Some(layer_first_instance);
            }
            run_total += entry.instance_count as u32;
        }
        layer_first_instance += entry.instance_count as u32;
    }
    // Flush any trailing run.
    if let Some(rf) = run_first {
        pass.draw(0..36, rf..rf + run_total);
    }
}

/// Build the bind group used by the blit pipeline. Re-call whenever the
/// offscreen color view is recreated (i.e. after `OffscreenTargets::resize`
/// returns true).
fn build_blit_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    color_view: &TextureView,
    sampler: &Sampler
) -> BindGroup {
    device.create_bind_group(
        &(BindGroupDescriptor {
            label: Some("blit_bind_group"),
            layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(color_view) },
                BindGroupEntry { binding: 1, resource: BindingResource::Sampler(sampler) },
            ],
        })
    )
}
