use std::sync::Arc;

use glow::HasContext;
use katana_core::planner::{PlannedLayer, MoveKind};

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

const LINE_VS: &str = r#"#version 330 core
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec4 a_color;

uniform mat4 u_mvp;
out vec4 v_color;
out float v_z;

void main() {
    gl_Position = u_mvp * vec4(a_pos, 1.0);
    v_color = a_color;
    v_z = a_pos.z;
}
"#;

const LINE_FS: &str = r#"#version 330 core
in vec4 v_color;
in float v_z;
out vec4 frag_color;

uniform float u_clip_z;

void main() {
    if (v_z < u_clip_z) discard;
    frag_color = v_color;
}
"#;

const MESH_VS: &str = r#"#version 330 core
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in vec4 a_color;
layout (location = 3) in float a_layer_z;

uniform mat4 u_mvp;

out vec3 v_normal;
out vec4 v_color;
out float v_z;

void main() {
    gl_Position = u_mvp * vec4(a_pos, 1.0);
    v_normal = a_normal;
    v_color = a_color;
    v_z = a_layer_z;
}
"#;

const MESH_FS: &str = r#"#version 330 core
in vec3 v_normal;
in vec4 v_color;
in float v_z;
out vec4 frag_color;

uniform vec3 u_light_dir;
uniform float u_clip_z;

void main() {
    if (v_z < u_clip_z) discard;
    vec3 n = normalize(v_normal);
    float diffuse = abs(dot(n, u_light_dir));
    float ambient = 0.15;
    float light = ambient + (1.0 - ambient) * diffuse;
    frag_color = vec4(v_color.rgb * light, v_color.a);
}
"#;

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

const LINE_STRIDE: usize = 7;  // x y z r g b a
const MESH_STRIDE: usize = 11; // x y z nx ny nz r g b a layer_z

pub struct GpuBuffer {
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    vertex_count: i32,
}

pub struct Renderer {
    gl: Arc<glow::Context>,
    line_program: glow::Program,
    mesh_program: glow::Program,
    pub mesh: Option<GpuBuffer>,
    pub slices: Option<GpuBuffer>,
    pub current_slice: Option<GpuBuffer>,
    pub toolpath_quads: Option<GpuBuffer>,
    pub toolpath_lines: Option<GpuBuffer>,
    pub toolpath_path_lines: Option<GpuBuffer>,
    // Our own FBO with a guaranteed depth buffer
    fbo: glow::Framebuffer,
    fbo_color: glow::Texture,
    fbo_depth: glow::Renderbuffer,
    fbo_w: i32,
    fbo_h: i32,
    // Z-clipping: only draw geometry at z >= clip_z
    pub clip_z: f32,
    pub draw_contours: bool,
    pub draw_toolpaths: bool,
    pub show_travel_moves: bool,
    pub show_filaments: bool,
}

impl Renderer {
    pub fn new(gl: Arc<glow::Context>) -> Self {
        let line_program = unsafe { create_program(&gl, LINE_VS, LINE_FS) };
        let mesh_program = unsafe { create_program(&gl, MESH_VS, MESH_FS) };

        // Create FBO with depth buffer (start at 1x1, resized on first draw)
        let (fbo, fbo_color, fbo_depth) = unsafe { create_fbo(&gl, 1, 1) };

        Renderer {
            gl,
            line_program,
            mesh_program,
            mesh: None,
            slices: None,
            current_slice: None,
            toolpath_quads: None,
            toolpath_lines: None,
            toolpath_path_lines: None,
            fbo,
            fbo_color,
            fbo_depth,
            fbo_w: 1,
            fbo_h: 1,
            clip_z: -1e30,
            draw_contours: false,
            draw_toolpaths: true,
            show_travel_moves: true,
            show_filaments: true,
        }
    }

    pub fn upload_mesh(&mut self, triangles: &[katana_core::mesh::Triangle]) {
        let mut verts: Vec<f32> = Vec::with_capacity(triangles.len() * 3 * MESH_STRIDE);

        let (r, g, b, a) = (0.35, 0.55, 0.75, 1.0);

        for tri in triangles {
            let n = &tri.normal;
            for v in &tri.vertices {
                verts.extend_from_slice(&[v.x, v.y, v.z, n.x, n.y, n.z, r, g, b, a, -1e30]);
            }
        }

        let count = (verts.len() / MESH_STRIDE) as i32;
        self.mesh = Some(upload_mesh_buffer(&self.gl, &verts, count));
    }

    pub fn upload_all_slices(&mut self, layers: &[katana_core::slicer::Layer], stride: usize) {
        let mut verts: Vec<f32> = Vec::new();
        let (r, g, b, a) = (0.31, 0.31, 0.47, 0.25);

        for (i, layer) in layers.iter().enumerate() {
            if i % stride != 0 {
                continue;
            }
            for contour in &layer.contours {
                let pts = &contour.points;
                if pts.len() < 2 {
                    continue;
                }
                for j in 0..pts.len() {
                    let k = (j + 1) % pts.len();
                    push_line_vert(&mut verts, pts[j].x, pts[j].y, layer.z, r, g, b, a);
                    push_line_vert(&mut verts, pts[k].x, pts[k].y, layer.z, r, g, b, a);
                }
            }
        }

        let count = (verts.len() / LINE_STRIDE) as i32;
        self.slices = Some(upload_line_buffer(&self.gl, &verts, count));
    }

    pub fn upload_current_slice(&mut self, layers: &[katana_core::slicer::Layer]) {
        let mut verts: Vec<f32> = Vec::new();
        let (r, g, b, a) = (0.91, 0.27, 0.38, 1.0);

        for layer in layers {
            for contour in &layer.contours {
                let pts = &contour.points;
                if pts.len() < 2 {
                    continue;
                }
                for j in 0..pts.len() {
                    let k = (j + 1) % pts.len();
                    push_line_vert(&mut verts, pts[j].x, pts[j].y, layer.z, r, g, b, a);
                    push_line_vert(&mut verts, pts[k].x, pts[k].y, layer.z, r, g, b, a);
                }
            }
        }

        let count = (verts.len() / LINE_STRIDE) as i32;
        self.current_slice = Some(upload_line_buffer(&self.gl, &verts, count));
    }

    /// Upload planned toolpath layers: extrusion moves become lit quads,
    /// travel moves stay as thin lines. All layers are uploaded once;
    /// visibility is controlled by `clip_z` in the shader.
    pub fn upload_planned_toolpath(
        &mut self,
        planned_layers: &[PlannedLayer],
        nozzle_width: f32,
    ) {
        let mut quad_verts: Vec<f32> = Vec::new();
        let mut line_verts: Vec<f32> = Vec::new();
        let mut path_line_verts: Vec<f32> = Vec::new();

        for layer in planned_layers {
            let z = layer.z;

            for move_ in &layer.moves {
                match move_.kind {
                    MoveKind::Travel => {
                        if move_.points.len() >= 2 {
                            let (r, g, b, a) = (1.0, 0.8, 0.2, 0.4);
                            let from = &move_.points[0];
                            let to = &move_.points[1];
                            push_line_vert(&mut line_verts, from.x, from.y, z, r, g, b, a);
                            push_line_vert(&mut line_verts, to.x, to.y, z, r, g, b, a);
                        }
                    }
                    MoveKind::Perimeter => {
                        let (r, g, b, a) = (0.91, 0.27, 0.38, 1.0);
                        let pts = &move_.points;
                        if pts.len() < 2 { continue; }
                        if pts.len() == 2 {
                            // Connection segment between perimeter loops:
                            // render as open tube with sphere joints
                            let from = &pts[0];
                            let to = &pts[1];
                            push_segment_tube(
                                &mut quad_verts,
                                from.x, from.y, to.x, to.y,
                                z, nozzle_width,
                                r, g, b, a,
                            );
                            let radius = nozzle_width * 0.5;
                            push_sphere(&mut quad_verts, from.x, from.y, z, radius, r, g, b, a, z);
                            push_sphere(&mut quad_verts, to.x, to.y, z, radius, r, g, b, a, z);
                            push_line_vert(&mut path_line_verts, from.x, from.y, z, r, g, b, a);
                            push_line_vert(&mut path_line_verts, to.x, to.y, z, r, g, b, a);
                        } else {
                            // Closed perimeter loop
                            let pts_xy: Vec<(f32, f32)> = pts.iter().map(|p| (p.x, p.y)).collect();
                            push_polyline_tube(
                                &mut quad_verts,
                                &pts_xy, z, nozzle_width,
                                r, g, b, a,
                                true, // closed loop
                            );
                            for j in 0..pts.len() {
                                let k = (j + 1) % pts.len();
                                push_line_vert(&mut path_line_verts, pts[j].x, pts[j].y, z, r, g, b, a);
                                push_line_vert(&mut path_line_verts, pts[k].x, pts[k].y, z, r, g, b, a);
                            }
                        }
                    }
                    MoveKind::Infill => {
                        let (r, g, b, a) = (0.27, 0.91, 0.38, 0.8);
                        if move_.points.len() >= 2 {
                            let from = &move_.points[0];
                            let to = &move_.points[1];
                            push_segment_tube(
                                &mut quad_verts,
                                from.x, from.y,
                                to.x, to.y,
                                z, nozzle_width,
                                r, g, b, a,
                            );
                            push_line_vert(&mut path_line_verts, from.x, from.y, z, r, g, b, a);
                            push_line_vert(&mut path_line_verts, to.x, to.y, z, r, g, b, a);
                        }
                    }
                    MoveKind::SurfaceInfill => {
                        let (r, g, b, a) = (0.9, 0.2, 0.7, 0.9);
                        if move_.points.len() >= 2 {
                            let from = &move_.points[0];
                            let to = &move_.points[1];
                            push_segment_tube(
                                &mut quad_verts,
                                from.x, from.y,
                                to.x, to.y,
                                z, nozzle_width,
                                r, g, b, a,
                            );
                            // Sphere joints at endpoints for smooth connections
                            let radius = nozzle_width * 0.5;
                            push_sphere(&mut quad_verts, from.x, from.y, z, radius, r, g, b, a, z);
                            push_sphere(&mut quad_verts, to.x, to.y, z, radius, r, g, b, a, z);
                            push_line_vert(&mut path_line_verts, from.x, from.y, z, r, g, b, a);
                            push_line_vert(&mut path_line_verts, to.x, to.y, z, r, g, b, a);
                        }
                    }
                }
            }
        }

        self.toolpath_quads = if quad_verts.is_empty() {
            None
        } else {
            let count = (quad_verts.len() / MESH_STRIDE) as i32;
            Some(upload_mesh_buffer(&self.gl, &quad_verts, count))
        };
        self.toolpath_lines = if line_verts.is_empty() {
            None
        } else {
            let count = (line_verts.len() / LINE_STRIDE) as i32;
            Some(upload_line_buffer(&self.gl, &line_verts, count))
        };
        self.toolpath_path_lines = if path_line_verts.is_empty() {
            None
        } else {
            let count = (path_line_verts.len() / LINE_STRIDE) as i32;
            Some(upload_line_buffer(&self.gl, &path_line_verts, count))
        };
    }

    /// Draw the scene into our own FBO (with depth buffer), then blit to screen.
    pub fn draw(
        &mut self,
        mvp: &[f32; 16],
        light_dir: &[f32; 3],
        bg_mode: &super::BgMode,
        viewport_w: i32,
        viewport_h: i32,
        screen_x: i32,
        screen_y: i32,
    ) {
        // Resize FBO if viewport changed (must happen before the unsafe block
        // to satisfy the borrow checker — resize_fbo borrows &mut self).
        if viewport_w != self.fbo_w || viewport_h != self.fbo_h {
            unsafe { self.resize_fbo(viewport_w, viewport_h) };
        }

        unsafe {
            let gl = &self.gl;

            // Save egui's framebuffer binding
            let prev_fbo = gl.get_parameter_i32(glow::FRAMEBUFFER_BINDING);

            // --- Render to our FBO ---
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.fbo));
            gl.viewport(0, 0, viewport_w, viewport_h);

            // Clear our FBO (dark background + depth)
            gl.depth_mask(true);
            gl.clear_color(0.102, 0.102, 0.18, 1.0); // #1a1a2e
            gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);

            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LEQUAL);
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);

            let no_clip: f32 = -1e30;

            // Draw background (no z-clipping)
            match bg_mode {
                super::BgMode::Mesh => {
                    if let Some(m) = &self.mesh {
                        gl.use_program(Some(self.mesh_program));
                        let loc = gl.get_uniform_location(self.mesh_program, "u_mvp");
                        gl.uniform_matrix_4_f32_slice(loc.as_ref(), false, mvp);
                        let loc = gl.get_uniform_location(self.mesh_program, "u_light_dir");
                        gl.uniform_3_f32_slice(loc.as_ref(), light_dir);
                        let loc = gl.get_uniform_location(self.mesh_program, "u_clip_z");
                        gl.uniform_1_f32(loc.as_ref(), no_clip);
                        draw_buffer(gl, m, glow::TRIANGLES);
                    }
                }
                super::BgMode::Layers => {
                    if let Some(s) = &self.slices {
                        gl.use_program(Some(self.line_program));
                        let loc = gl.get_uniform_location(self.line_program, "u_mvp");
                        gl.uniform_matrix_4_f32_slice(loc.as_ref(), false, mvp);
                        let loc = gl.get_uniform_location(self.line_program, "u_clip_z");
                        gl.uniform_1_f32(loc.as_ref(), no_clip);
                        draw_buffer(gl, s, glow::LINES);
                    }
                }
                super::BgMode::None => {}
            }

            // Draw foreground with z-clipping (clear depth so BG doesn't
            // occlude it, but keep depth test for inter-layer occlusion).
            gl.clear(glow::DEPTH_BUFFER_BIT);

            let clip = self.clip_z;

            // Contour view (lines)
            if self.draw_contours {
                if let Some(cs) = &self.current_slice {
                    gl.use_program(Some(self.line_program));
                    let loc = gl.get_uniform_location(self.line_program, "u_mvp");
                    gl.uniform_matrix_4_f32_slice(loc.as_ref(), false, mvp);
                    let loc = gl.get_uniform_location(self.line_program, "u_clip_z");
                    gl.uniform_1_f32(loc.as_ref(), clip);
                    draw_buffer(gl, cs, glow::LINES);
                }
            }

            // Toolpath rendering
            if self.draw_toolpaths {
                if self.show_filaments {
                    // 3D filament tubes (lit triangles)
                    if let Some(tq) = &self.toolpath_quads {
                        gl.use_program(Some(self.mesh_program));
                        let loc = gl.get_uniform_location(self.mesh_program, "u_mvp");
                        gl.uniform_matrix_4_f32_slice(loc.as_ref(), false, mvp);
                        let loc = gl.get_uniform_location(self.mesh_program, "u_light_dir");
                        gl.uniform_3_f32_slice(loc.as_ref(), light_dir);
                        let loc = gl.get_uniform_location(self.mesh_program, "u_clip_z");
                        gl.uniform_1_f32(loc.as_ref(), clip);
                        draw_buffer(gl, tq, glow::TRIANGLES);
                    }
                } else {
                    // Toolpath lines (flat lines for extrusion paths)
                    if let Some(pl) = &self.toolpath_path_lines {
                        gl.use_program(Some(self.line_program));
                        let loc = gl.get_uniform_location(self.line_program, "u_mvp");
                        gl.uniform_matrix_4_f32_slice(loc.as_ref(), false, mvp);
                        let loc = gl.get_uniform_location(self.line_program, "u_clip_z");
                        gl.uniform_1_f32(loc.as_ref(), clip);
                        draw_buffer(gl, pl, glow::LINES);
                    }
                }

                // Toolpath travel lines
                if self.show_travel_moves {
                    if let Some(tl) = &self.toolpath_lines {
                        gl.use_program(Some(self.line_program));
                        let loc = gl.get_uniform_location(self.line_program, "u_mvp");
                        gl.uniform_matrix_4_f32_slice(loc.as_ref(), false, mvp);
                        let loc = gl.get_uniform_location(self.line_program, "u_clip_z");
                        gl.uniform_1_f32(loc.as_ref(), clip);
                        draw_buffer(gl, tl, glow::LINES);
                    }
                }
            }

            gl.disable(glow::BLEND);
            gl.use_program(None);

            // --- Blit our FBO to egui's framebuffer ---
            let prev_fbo_id = if prev_fbo == 0 {
                None
            } else {
                // Re-wrap the raw ID into a glow framebuffer handle
                Some(glow::NativeFramebuffer(std::num::NonZeroU32::new(prev_fbo as u32).unwrap()))
            };
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(self.fbo));
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, prev_fbo_id);
            gl.blit_framebuffer(
                0,
                0,
                viewport_w,
                viewport_h,
                screen_x,
                screen_y,
                screen_x + viewport_w,
                screen_y + viewport_h,
                glow::COLOR_BUFFER_BIT,
                glow::NEAREST,
            );

            // Restore egui's state
            gl.bind_framebuffer(glow::FRAMEBUFFER, prev_fbo_id);
        }
    }

    unsafe fn resize_fbo(&mut self, w: i32, h: i32) {
        let gl = &self.gl;

        gl.bind_texture(glow::TEXTURE_2D, Some(self.fbo_color));
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA8 as i32,
            w,
            h,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelUnpackData::Slice(None),
        );

        gl.bind_renderbuffer(glow::RENDERBUFFER, Some(self.fbo_depth));
        gl.renderbuffer_storage(glow::RENDERBUFFER, glow::DEPTH_COMPONENT24, w, h);

        gl.bind_texture(glow::TEXTURE_2D, None);
        gl.bind_renderbuffer(glow::RENDERBUFFER, None);

        self.fbo_w = w;
        self.fbo_h = h;
    }

    pub fn destroy(&self) {
        unsafe {
            let gl = &self.gl;
            gl.delete_program(self.line_program);
            gl.delete_program(self.mesh_program);
            gl.delete_framebuffer(self.fbo);
            gl.delete_texture(self.fbo_color);
            gl.delete_renderbuffer(self.fbo_depth);
            for buf in [&self.mesh, &self.slices, &self.current_slice, &self.toolpath_quads, &self.toolpath_lines, &self.toolpath_path_lines]
                .into_iter()
                .flatten()
            {
                gl.delete_vertex_array(buf.vao);
                gl.delete_buffer(buf.vbo);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FBO creation
// ---------------------------------------------------------------------------

unsafe fn create_fbo(
    gl: &glow::Context,
    w: i32,
    h: i32,
) -> (glow::Framebuffer, glow::Texture, glow::Renderbuffer) {
    let fbo = gl.create_framebuffer().unwrap();
    gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));

    // Color attachment (texture)
    let color = gl.create_texture().unwrap();
    gl.bind_texture(glow::TEXTURE_2D, Some(color));
    gl.tex_image_2d(
        glow::TEXTURE_2D,
        0,
        glow::RGBA8 as i32,
        w,
        h,
        0,
        glow::RGBA,
        glow::UNSIGNED_BYTE,
        glow::PixelUnpackData::Slice(None),
    );
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
    gl.framebuffer_texture_2d(
        glow::FRAMEBUFFER,
        glow::COLOR_ATTACHMENT0,
        glow::TEXTURE_2D,
        Some(color),
        0,
    );

    // Depth attachment (24-bit renderbuffer)
    let depth = gl.create_renderbuffer().unwrap();
    gl.bind_renderbuffer(glow::RENDERBUFFER, Some(depth));
    gl.renderbuffer_storage(glow::RENDERBUFFER, glow::DEPTH_COMPONENT24, w, h);
    gl.framebuffer_renderbuffer(
        glow::FRAMEBUFFER,
        glow::DEPTH_ATTACHMENT,
        glow::RENDERBUFFER,
        Some(depth),
    );

    let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
    if status != glow::FRAMEBUFFER_COMPLETE {
        panic!("FBO incomplete: {status:#x}");
    }

    gl.bind_framebuffer(glow::FRAMEBUFFER, None);
    gl.bind_texture(glow::TEXTURE_2D, None);
    gl.bind_renderbuffer(glow::RENDERBUFFER, None);

    (fbo, color, depth)
}

// ---------------------------------------------------------------------------
// Vertex helpers
// ---------------------------------------------------------------------------

fn push_line_vert(buf: &mut Vec<f32>, x: f32, y: f32, z: f32, r: f32, g: f32, b: f32, a: f32) {
    buf.extend_from_slice(&[x, y, z, r, g, b, a]);
}

fn push_mesh_vert(
    buf: &mut Vec<f32>,
    x: f32, y: f32, z: f32,
    nx: f32, ny: f32, nz: f32,
    r: f32, g: f32, b: f32, a: f32,
    layer_z: f32,
) {
    buf.extend_from_slice(&[x, y, z, nx, ny, nz, r, g, b, a, layer_z]);
}

const TUBE_SIDES: usize = 8;

/// Compute a ring of TUBE_SIDES vertices around `center` at height `z`,
/// oriented perpendicular to the XY direction `(dx, dy)` (need not be unit).
/// Returns [(x, y, z, nx, ny, nz); TUBE_SIDES].
fn compute_ring(
    cx: f32, cy: f32, z: f32,
    dx: f32, dy: f32,
    radius: f32,
) -> [(f32, f32, f32, f32, f32, f32); TUBE_SIDES] {
    let len = (dx * dx + dy * dy).sqrt();
    let (p1x, p1y) = if len < 1e-9 { (1.0, 0.0) } else { (-dy / len, dx / len) };

    let mut ring = [(0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0); TUBE_SIDES];
    for i in 0..TUBE_SIDES {
        let theta = std::f32::consts::TAU * (i as f32) / (TUBE_SIDES as f32);
        let (c, s) = (theta.cos(), theta.sin());
        ring[i] = (
            cx + radius * (c * p1x),
            cy + radius * (c * p1y),
            z  + radius * s,
            c * p1x,
            c * p1y,
            s,
        );
    }
    ring
}

/// Connect two rings of TUBE_SIDES vertices with a triangle strip (2N tris).
fn connect_rings(
    buf: &mut Vec<f32>,
    ring_a: &[(f32, f32, f32, f32, f32, f32); TUBE_SIDES],
    ring_b: &[(f32, f32, f32, f32, f32, f32); TUBE_SIDES],
    r: f32, g: f32, b: f32, a: f32,
    layer_z: f32,
) {
    for i in 0..TUBE_SIDES {
        let j = (i + 1) % TUBE_SIDES;
        let (a0x, a0y, a0z, an0x, an0y, an0z) = ring_a[i];
        let (a1x, a1y, a1z, an1x, an1y, an1z) = ring_a[j];
        let (b0x, b0y, b0z, bn0x, bn0y, bn0z) = ring_b[i];
        let (b1x, b1y, b1z, bn1x, bn1y, bn1z) = ring_b[j];

        push_mesh_vert(buf, a0x, a0y, a0z, an0x, an0y, an0z, r, g, b, a, layer_z);
        push_mesh_vert(buf, b0x, b0y, b0z, bn0x, bn0y, bn0z, r, g, b, a, layer_z);
        push_mesh_vert(buf, a1x, a1y, a1z, an1x, an1y, an1z, r, g, b, a, layer_z);

        push_mesh_vert(buf, a1x, a1y, a1z, an1x, an1y, an1z, r, g, b, a, layer_z);
        push_mesh_vert(buf, b0x, b0y, b0z, bn0x, bn0y, bn0z, r, g, b, a, layer_z);
        push_mesh_vert(buf, b1x, b1y, b1z, bn1x, bn1y, bn1z, r, g, b, a, layer_z);
    }
}

/// Emit a disc cap (triangle fan) for a ring, facing `sign` direction along the tube.
fn push_disc_cap(
    buf: &mut Vec<f32>,
    cx: f32, cy: f32, cz: f32,
    nx: f32, ny: f32, nz: f32,
    ring: &[(f32, f32, f32, f32, f32, f32); TUBE_SIDES],
    r: f32, g: f32, b: f32, a: f32,
    layer_z: f32,
) {
    for i in 0..TUBE_SIDES {
        let j = (i + 1) % TUBE_SIDES;
        push_mesh_vert(buf, cx, cy, cz, nx, ny, nz, r, g, b, a, layer_z);
        push_mesh_vert(buf, ring[i].0, ring[i].1, ring[i].2, nx, ny, nz, r, g, b, a, layer_z);
        push_mesh_vert(buf, ring[j].0, ring[j].1, ring[j].2, nx, ny, nz, r, g, b, a, layer_z);
    }
}

/// Emit a UV sphere at (cx, cy, cz) for round joints between tube segments.
fn push_sphere(
    buf: &mut Vec<f32>,
    cx: f32, cy: f32, cz: f32,
    radius: f32,
    r: f32, g: f32, b: f32, a: f32,
    layer_z: f32,
) {
    let stacks = TUBE_SIDES;
    let slices = TUBE_SIDES;

    for i in 0..stacks {
        let phi0 = std::f32::consts::PI * (i as f32) / (stacks as f32);
        let phi1 = std::f32::consts::PI * ((i + 1) as f32) / (stacks as f32);
        let (cp0, sp0) = (phi0.cos(), phi0.sin());
        let (cp1, sp1) = (phi1.cos(), phi1.sin());

        for j in 0..slices {
            let t0 = std::f32::consts::TAU * (j as f32) / (slices as f32);
            let t1 = std::f32::consts::TAU * ((j + 1) as f32) / (slices as f32);
            let (ct0, st0) = (t0.cos(), t0.sin());
            let (ct1, st1) = (t1.cos(), t1.sin());

            let p00 = (sp0 * ct0, sp0 * st0, cp0);
            let p01 = (sp0 * ct1, sp0 * st1, cp0);
            let p10 = (sp1 * ct0, sp1 * st0, cp1);
            let p11 = (sp1 * ct1, sp1 * st1, cp1);

            push_mesh_vert(buf, cx + radius * p00.0, cy + radius * p00.1, cz + radius * p00.2, p00.0, p00.1, p00.2, r, g, b, a, layer_z);
            push_mesh_vert(buf, cx + radius * p10.0, cy + radius * p10.1, cz + radius * p10.2, p10.0, p10.1, p10.2, r, g, b, a, layer_z);
            push_mesh_vert(buf, cx + radius * p01.0, cy + radius * p01.1, cz + radius * p01.2, p01.0, p01.1, p01.2, r, g, b, a, layer_z);

            push_mesh_vert(buf, cx + radius * p01.0, cy + radius * p01.1, cz + radius * p01.2, p01.0, p01.1, p01.2, r, g, b, a, layer_z);
            push_mesh_vert(buf, cx + radius * p10.0, cy + radius * p10.1, cz + radius * p10.2, p10.0, p10.1, p10.2, r, g, b, a, layer_z);
            push_mesh_vert(buf, cx + radius * p11.0, cy + radius * p11.1, cz + radius * p11.2, p11.0, p11.1, p11.2, r, g, b, a, layer_z);
        }
    }
}

/// Emit a continuous tube along a polyline using per-segment tubes with
/// round (sphere) joints at each vertex to cover gaps between segments.
fn push_polyline_tube(
    buf: &mut Vec<f32>,
    pts_xy: &[(f32, f32)],
    z: f32,
    w: f32,
    r: f32, g: f32, b: f32, a: f32,
    closed: bool,
) {
    let n = pts_xy.len();
    if n < 2 { return; }

    let radius = w * 0.5;
    let num_segs = if closed { n } else { n - 1 };

    // Emit tube body for each segment (both rings share the same perp — no twist)
    for s in 0..num_segs {
        let next = (s + 1) % n;
        let dx = pts_xy[next].0 - pts_xy[s].0;
        let dy = pts_xy[next].1 - pts_xy[s].1;

        let ring_a = compute_ring(pts_xy[s].0, pts_xy[s].1, z, dx, dy, radius);
        let ring_b = compute_ring(pts_xy[next].0, pts_xy[next].1, z, dx, dy, radius);
        connect_rings(buf, &ring_a, &ring_b, r, g, b, a, z);
    }

    // Emit sphere at each vertex to cover joint gaps (Z-buffer handles overlap)
    for i in 0..n {
        push_sphere(buf, pts_xy[i].0, pts_xy[i].1, z, radius, r, g, b, a, z);
    }
}

/// Emit a single capped tube segment for a standalone line (infill).
fn push_segment_tube(
    buf: &mut Vec<f32>,
    ax: f32, ay: f32,
    bx: f32, by: f32,
    z: f32,
    w: f32,
    r: f32, g: f32, b: f32, a: f32,
) {
    let dx = bx - ax;
    let dy = by - ay;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-9 { return; }

    let radius = w * 0.5;

    let ring_a = compute_ring(ax, ay, z, dx, dy, radius);
    let ring_b = compute_ring(bx, by, z, dx, dy, radius);

    connect_rings(buf, &ring_a, &ring_b, r, g, b, a, z);

    // Cap both ends
    let ndx = dx / len;
    let ndy = dy / len;
    push_disc_cap(buf, ax, ay, z, -ndx, -ndy, 0.0, &ring_a, r, g, b, a, z);
    push_disc_cap(buf, bx, by, z,  ndx,  ndy, 0.0, &ring_b, r, g, b, a, z);
}

fn upload_line_buffer(gl: &glow::Context, data: &[f32], vertex_count: i32) -> GpuBuffer {
    unsafe {
        let vao = gl.create_vertex_array().unwrap();
        let vbo = gl.create_buffer().unwrap();

        gl.bind_vertex_array(Some(vao));
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, cast_f32_u8(data), glow::STATIC_DRAW);

        let stride = (LINE_STRIDE * 4) as i32;
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, stride, 0);
        gl.enable_vertex_attrib_array(1);
        gl.vertex_attrib_pointer_f32(1, 4, glow::FLOAT, false, stride, 3 * 4);

        gl.bind_vertex_array(None);
        GpuBuffer { vao, vbo, vertex_count }
    }
}

fn upload_mesh_buffer(gl: &glow::Context, data: &[f32], vertex_count: i32) -> GpuBuffer {
    unsafe {
        let vao = gl.create_vertex_array().unwrap();
        let vbo = gl.create_buffer().unwrap();

        gl.bind_vertex_array(Some(vao));
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, cast_f32_u8(data), glow::STATIC_DRAW);

        let stride = (MESH_STRIDE * 4) as i32;
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, stride, 0);
        gl.enable_vertex_attrib_array(1);
        gl.vertex_attrib_pointer_f32(1, 3, glow::FLOAT, false, stride, 3 * 4);
        gl.enable_vertex_attrib_array(2);
        gl.vertex_attrib_pointer_f32(2, 4, glow::FLOAT, false, stride, 6 * 4);
        gl.enable_vertex_attrib_array(3);
        gl.vertex_attrib_pointer_f32(3, 1, glow::FLOAT, false, stride, 10 * 4);

        gl.bind_vertex_array(None);
        GpuBuffer { vao, vbo, vertex_count }
    }
}

unsafe fn draw_buffer(gl: &glow::Context, buf: &GpuBuffer, mode: u32) {
    gl.bind_vertex_array(Some(buf.vao));
    gl.draw_arrays(mode, 0, buf.vertex_count);
    gl.bind_vertex_array(None);
}

fn cast_f32_u8(data: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
}

unsafe fn create_program(gl: &glow::Context, vs_src: &str, fs_src: &str) -> glow::Program {
    let program = gl.create_program().unwrap();
    let vs = compile_shader(gl, glow::VERTEX_SHADER, vs_src);
    let fs = compile_shader(gl, glow::FRAGMENT_SHADER, fs_src);

    gl.attach_shader(program, vs);
    gl.attach_shader(program, fs);
    gl.link_program(program);

    if !gl.get_program_link_status(program) {
        panic!("Shader link error: {}", gl.get_program_info_log(program));
    }

    gl.detach_shader(program, vs);
    gl.detach_shader(program, fs);
    gl.delete_shader(vs);
    gl.delete_shader(fs);
    program
}

unsafe fn compile_shader(gl: &glow::Context, shader_type: u32, source: &str) -> glow::Shader {
    let shader = gl.create_shader(shader_type).unwrap();
    gl.shader_source(shader, source);
    gl.compile_shader(shader);

    if !gl.get_shader_compile_status(shader) {
        panic!("Shader compile error: {}", gl.get_shader_info_log(shader));
    }
    shader
}

// ---------------------------------------------------------------------------
// Camera math
// ---------------------------------------------------------------------------

pub fn build_mvp(
    center: [f32; 3],
    azimuth: f32,
    elevation: f32,
    zoom: f32,
    extent: f32,
    aspect: f32,
    _pan: (f32, f32),  // Currently unused; panning handled by updating center directly
) -> [f32; 16] {
    let s = 2.0 * zoom / extent;
    let sx = if aspect > 1.0 { s / aspect } else { s };
    let sy = if aspect > 1.0 { s } else { s * aspect };
    let sz = 1.0 / (extent * 0.87);

    let ca = azimuth.cos();
    let sa = azimuth.sin();
    let ce = elevation.cos();
    let se = elevation.sin();
    let (tx, ty, tz) = (center[0], center[1], center[2]);

    let r00 = sx * ca;
    let r01 = sx * (-sa);
    let r02 = 0.0;
    let r10 = sy * se * sa;
    let r11 = sy * se * ca;
    let r12 = sy * (-ce);
    let r20 = sz * ce * sa;
    let r21 = sz * ce * ca;
    let r22 = sz * se;

    // Translation to center on the target point
    // (pan is now handled by updating center directly in world space)
    let t0 = -(r00 * tx + r01 * ty + r02 * tz);
    let t1 = -(r10 * tx + r11 * ty + r12 * tz);
    let t2 = -(r20 * tx + r21 * ty + r22 * tz);

    [
        r00, r10, r20, 0.0, r01, r11, r21, 0.0, r02, r12, r22, 0.0, t0, t1, t2, 1.0,
    ]
}

pub fn headlight_dir(azimuth: f32, elevation: f32) -> [f32; 3] {
    let ca = azimuth.cos();
    let sa = azimuth.sin();
    let ce = elevation.cos();
    let se = elevation.sin();
    [-sa * ce, ca * ce, se]
}
