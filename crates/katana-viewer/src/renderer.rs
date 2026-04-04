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
    if (v_z > u_clip_z) discard;
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
    if (v_z > u_clip_z) discard;
    vec3 n = normalize(v_normal);
    float diffuse = abs(dot(n, u_light_dir));
    float ambient = 0.15;
    float light = ambient + (1.0 - ambient) * diffuse;
    frag_color = vec4(v_color.rgb * light, v_color.a);
}
"#;

// Extruded rhombus vertex shader: generates a 12-triangle prism per instance from gl_VertexID.
// Cross-section is a rhombus: width = nozzle_width (horizontal), height = layer_height (vertical).
// 4 side faces (2 triangles each) + 2 end caps (2 triangles each) = 12 triangles = 36 vertices.
const RHOMBUS_VS: &str = r#"#version 330 core
layout (location = 0) in vec3  a_inst_start;
layout (location = 1) in vec2  a_inst_dir;
layout (location = 2) in vec2  a_inst_scale;     // (length, half_width)
layout (location = 3) in vec4  a_inst_color;
layout (location = 4) in float a_inst_layer_z;

uniform mat4  u_mvp;
uniform float u_clip_z;
uniform float u_half_height;                      // layer_height / 2 (constant for whole print)

flat out vec3 v_normal;
out vec4  v_color;
out float v_z;

void main() {
    float seg_len = a_inst_scale.x;
    float half_w  = a_inst_scale.y;
    float half_h  = u_half_height;

    vec3 seg_dir = vec3(a_inst_dir, 0.0);
    vec3 perp    = vec3(-a_inst_dir.y, a_inst_dir.x, 0.0);
    vec3 up      = vec3(0.0, 0.0, 1.0);

    // 4 cross-section offsets
    vec3 r_off = perp * half_w;    // right
    vec3 t_off = up   * half_h;    // top
    vec3 l_off = -perp * half_w;   // left
    vec3 b_off = -up   * half_h;   // bottom

    // 4 side face normals (normalized bisectors of adjacent edges)
    vec3 n_rt = normalize(perp + up);     // right-top face
    vec3 n_tl = normalize(-perp + up);    // top-left face
    vec3 n_lb = normalize(-perp - up);    // left-bottom face
    vec3 n_br = normalize(perp - up);     // bottom-right face

    vec3 cross_off;
    vec3 along_off;
    vec3 norm;

    int vid = gl_VertexID;

    // Side face 0: right-top (triangles 0, 1)
    if      (vid == 0)  { cross_off = r_off; along_off = vec3(0);              norm = n_rt; }
    else if (vid == 1)  { cross_off = t_off; along_off = vec3(0);              norm = n_rt; }
    else if (vid == 2)  { cross_off = r_off; along_off = seg_dir * seg_len;    norm = n_rt; }
    else if (vid == 3)  { cross_off = t_off; along_off = vec3(0);              norm = n_rt; }
    else if (vid == 4)  { cross_off = t_off; along_off = seg_dir * seg_len;    norm = n_rt; }
    else if (vid == 5)  { cross_off = r_off; along_off = seg_dir * seg_len;    norm = n_rt; }
    // Side face 1: top-left (triangles 2, 3)
    else if (vid == 6)  { cross_off = t_off; along_off = vec3(0);              norm = n_tl; }
    else if (vid == 7)  { cross_off = l_off; along_off = vec3(0);              norm = n_tl; }
    else if (vid == 8)  { cross_off = t_off; along_off = seg_dir * seg_len;    norm = n_tl; }
    else if (vid == 9)  { cross_off = l_off; along_off = vec3(0);              norm = n_tl; }
    else if (vid == 10) { cross_off = l_off; along_off = seg_dir * seg_len;    norm = n_tl; }
    else if (vid == 11) { cross_off = t_off; along_off = seg_dir * seg_len;    norm = n_tl; }
    // Side face 2: left-bottom (triangles 4, 5)
    else if (vid == 12) { cross_off = l_off; along_off = vec3(0);              norm = n_lb; }
    else if (vid == 13) { cross_off = b_off; along_off = vec3(0);              norm = n_lb; }
    else if (vid == 14) { cross_off = l_off; along_off = seg_dir * seg_len;    norm = n_lb; }
    else if (vid == 15) { cross_off = b_off; along_off = vec3(0);              norm = n_lb; }
    else if (vid == 16) { cross_off = b_off; along_off = seg_dir * seg_len;    norm = n_lb; }
    else if (vid == 17) { cross_off = l_off; along_off = seg_dir * seg_len;    norm = n_lb; }
    // Side face 3: bottom-right (triangles 6, 7)
    else if (vid == 18) { cross_off = b_off; along_off = vec3(0);              norm = n_br; }
    else if (vid == 19) { cross_off = r_off; along_off = vec3(0);              norm = n_br; }
    else if (vid == 20) { cross_off = b_off; along_off = seg_dir * seg_len;    norm = n_br; }
    else if (vid == 21) { cross_off = r_off; along_off = vec3(0);              norm = n_br; }
    else if (vid == 22) { cross_off = r_off; along_off = seg_dir * seg_len;    norm = n_br; }
    else if (vid == 23) { cross_off = b_off; along_off = seg_dir * seg_len;    norm = n_br; }
    // Start cap (triangles 8, 9), normal = -seg_dir
    else if (vid == 24) { cross_off = r_off; along_off = vec3(0);              norm = -seg_dir; }
    else if (vid == 25) { cross_off = t_off; along_off = vec3(0);              norm = -seg_dir; }
    else if (vid == 26) { cross_off = l_off; along_off = vec3(0);              norm = -seg_dir; }
    else if (vid == 27) { cross_off = r_off; along_off = vec3(0);              norm = -seg_dir; }
    else if (vid == 28) { cross_off = l_off; along_off = vec3(0);              norm = -seg_dir; }
    else if (vid == 29) { cross_off = b_off; along_off = vec3(0);              norm = -seg_dir; }
    // End cap (triangles 10, 11), normal = +seg_dir
    else if (vid == 30) { cross_off = r_off; along_off = seg_dir * seg_len;    norm = seg_dir; }
    else if (vid == 31) { cross_off = l_off; along_off = seg_dir * seg_len;    norm = seg_dir; }
    else if (vid == 32) { cross_off = t_off; along_off = seg_dir * seg_len;    norm = seg_dir; }
    else if (vid == 33) { cross_off = r_off; along_off = seg_dir * seg_len;    norm = seg_dir; }
    else if (vid == 34) { cross_off = b_off; along_off = seg_dir * seg_len;    norm = seg_dir; }
    else                { cross_off = l_off; along_off = seg_dir * seg_len;    norm = seg_dir; }

    vec3 world_pos = a_inst_start + along_off + cross_off;

    gl_Position = u_mvp * vec4(world_pos, 1.0);
    gl_ClipDistance[0] = u_clip_z - a_inst_layer_z;

    v_normal = norm;
    v_color  = a_inst_color;
    v_z      = a_inst_layer_z;
}
"#;

const RHOMBUS_FS: &str = r#"#version 330 core
flat in vec3  v_normal;
in vec4  v_color;
in float v_z;

uniform vec3  u_light_dir;
uniform float u_clip_z;

out vec4 frag_color;

void main() {
    float diffuse = abs(dot(v_normal, u_light_dir));
    float ambient = 0.15;
    float light   = ambient + (1.0 - ambient) * diffuse;
    frag_color    = vec4(v_color.rgb * light, v_color.a);
}
"#;

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

const LINE_STRIDE: usize = 7;  // x y z r g b a
const MESH_STRIDE: usize = 11; // x y z nx ny nz r g b a layer_z
const RHOMBUS_INSTANCE_STRIDE: usize = 12;  // start(3) + dir(2) + scale(2) + color(4) + layer_z(1)

pub struct GpuBuffer {
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    vertex_count: i32,
}

#[allow(dead_code)]
enum BatchKind { Rhombus }

/// Per-layer index entry: Z position, instance count, and XY AABB for frustum culling.
struct LayerEntry {
    layer_z: f32,
    instance_count: i32,
    aabb_min_x: f32,
    aabb_min_y: f32,
    aabb_max_x: f32,
    aabb_max_y: f32,
}

struct LineUniforms {
    mvp:    Option<glow::UniformLocation>,
    clip_z: Option<glow::UniformLocation>,
}

struct MeshUniforms {
    mvp:       Option<glow::UniformLocation>,
    light_dir: Option<glow::UniformLocation>,
    clip_z:    Option<glow::UniformLocation>,
}

struct RhombusUniforms {
    mvp:         Option<glow::UniformLocation>,
    light_dir:   Option<glow::UniformLocation>,
    clip_z:      Option<glow::UniformLocation>,
    half_height: Option<glow::UniformLocation>,
}

struct InstancedBatch {
    /// One VAO per layer; attrib pointers pre-offset to that layer's region of the shared VBO.
    /// Emulates glDrawArraysInstancedBaseInstance (GL 4.2) on GL 3.3 / macOS GL 4.1.
    layer_vaos: Vec<glow::VertexArray>,
    instance_vbo: glow::Buffer,
    #[allow(dead_code)]
    instance_count: i32,
    prototype_vertex_count: i32,
    /// Sorted by layer_z; used for both clip_z culling and frustum culling.
    layer_entries: Vec<LayerEntry>,
}


pub struct Renderer {
    gl: Arc<glow::Context>,
    line_program: glow::Program,
    mesh_program: glow::Program,
    line_uniforms: LineUniforms,
    mesh_uniforms: MeshUniforms,
    rhombus_uniforms: RhombusUniforms,
    pub mesh: Option<GpuBuffer>,
    pub slices: Option<GpuBuffer>,
    pub current_slice: Option<GpuBuffer>,
    // Extruded rhombus toolpath rendering
    rhombus_program: glow::Program,
    toolpath_rhombuses: Option<InstancedBatch>,
    half_height: f32,
    pub toolpath_lines: Option<GpuBuffer>,
    pub toolpath_path_lines: Option<GpuBuffer>,
    // Our own FBO with a guaranteed depth buffer
    fbo: glow::Framebuffer,
    fbo_color: glow::Texture,
    fbo_depth: glow::Renderbuffer,
    fbo_w: i32,
    fbo_h: i32,
    // Z-clipping: only draw geometry at z <= clip_z
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
        let rhombus_program = unsafe { create_program(&gl, RHOMBUS_VS, RHOMBUS_FS) };

        let line_uniforms = unsafe { LineUniforms {
            mvp:    gl.get_uniform_location(line_program, "u_mvp"),
            clip_z: gl.get_uniform_location(line_program, "u_clip_z"),
        }};
        let mesh_uniforms = unsafe { MeshUniforms {
            mvp:       gl.get_uniform_location(mesh_program, "u_mvp"),
            light_dir: gl.get_uniform_location(mesh_program, "u_light_dir"),
            clip_z:    gl.get_uniform_location(mesh_program, "u_clip_z"),
        }};
        let rhombus_uniforms = unsafe { RhombusUniforms {
            mvp:         gl.get_uniform_location(rhombus_program, "u_mvp"),
            light_dir:   gl.get_uniform_location(rhombus_program, "u_light_dir"),
            clip_z:      gl.get_uniform_location(rhombus_program, "u_clip_z"),
            half_height: gl.get_uniform_location(rhombus_program, "u_half_height"),
        }};

        // Create FBO with depth buffer (start at 1x1, resized on first draw)
        let (fbo, fbo_color, fbo_depth) = unsafe { create_fbo(&gl, 1, 1) };

        Renderer {
            gl,
            line_program,
            mesh_program,
            line_uniforms,
            mesh_uniforms,
            rhombus_uniforms,
            mesh: None,
            slices: None,
            current_slice: None,
            rhombus_program,
            toolpath_rhombuses: None,
            half_height: 0.1,
            toolpath_lines: None,
            toolpath_path_lines: None,
            fbo,
            fbo_color,
            fbo_depth,
            fbo_w: 1,
            fbo_h: 1,
            clip_z: 1e30,
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

    /// Upload planned toolpath layers as extruded rhombus segments.
    /// Each segment is a prism with rhombus cross-section (width = nozzle_width, height = layer_height).
    pub fn upload_planned_toolpath(
        &mut self,
        planned_layers: &[PlannedLayer],
        nozzle_width: f32,
        layer_height: f32,
    ) {
        self.half_height = layer_height * 0.5;
        let mut rhombus_instances: Vec<f32> = Vec::new();
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
                            let from = &pts[0];
                            let to = &pts[1];
                            push_rhombus_instance(&mut rhombus_instances,
                                from.x, from.y, to.x, to.y,
                                z, nozzle_width,
                                r, g, b, a,
                                z,
                            );
                            push_line_vert(&mut path_line_verts, from.x, from.y, z, r, g, b, a);
                            push_line_vert(&mut path_line_verts, to.x, to.y, z, r, g, b, a);
                        } else {
                            let n = pts.len();
                            for s in 0..n {
                                let next = (s + 1) % n;
                                push_rhombus_instance(&mut rhombus_instances,
                                    pts[s].x, pts[s].y, pts[next].x, pts[next].y,
                                    z, nozzle_width,
                                    r, g, b, a,
                                    z,
                                );
                            }
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
                            push_rhombus_instance(&mut rhombus_instances,
                                from.x, from.y, to.x, to.y,
                                z, nozzle_width,
                                r, g, b, a,
                                z,
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
                            push_rhombus_instance(&mut rhombus_instances,
                                from.x, from.y, to.x, to.y,
                                z, nozzle_width,
                                r, g, b, a,
                                z,
                            );
                            push_line_vert(&mut path_line_verts, from.x, from.y, z, r, g, b, a);
                            push_line_vert(&mut path_line_verts, to.x, to.y, z, r, g, b, a);
                        }
                    }
                }
            }
        }

        // Upload rhombus batch
        self.toolpath_rhombuses = if rhombus_instances.is_empty() {
            None
        } else {
            let count = (rhombus_instances.len() / RHOMBUS_INSTANCE_STRIDE) as i32;
            Some(unsafe { upload_impostor_batch(&self.gl, &rhombus_instances, count, BatchKind::Rhombus) })
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

            let no_clip: f32 = 1e30;

            // Draw background (no z-clipping)
            match bg_mode {
                super::BgMode::Mesh => {
                    if let Some(m) = &self.mesh {
                        gl.use_program(Some(self.mesh_program));
                        gl.uniform_matrix_4_f32_slice(self.mesh_uniforms.mvp.as_ref(), false, mvp);
                        gl.uniform_3_f32_slice(self.mesh_uniforms.light_dir.as_ref(), light_dir);
                        gl.uniform_1_f32(self.mesh_uniforms.clip_z.as_ref(), no_clip);
                        draw_buffer(gl, m, glow::TRIANGLES);
                    }
                }
                super::BgMode::Layers => {
                    if let Some(s) = &self.slices {
                        gl.use_program(Some(self.line_program));
                        gl.uniform_matrix_4_f32_slice(self.line_uniforms.mvp.as_ref(), false, mvp);
                        gl.uniform_1_f32(self.line_uniforms.clip_z.as_ref(), no_clip);
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
                    gl.uniform_matrix_4_f32_slice(self.line_uniforms.mvp.as_ref(), false, mvp);
                    gl.uniform_1_f32(self.line_uniforms.clip_z.as_ref(), clip);
                    draw_buffer(gl, cs, glow::LINES);
                }
            }

            // Toolpath rendering
            if self.draw_toolpaths {
                if self.show_filaments {
                    // Enable hardware clipping for layer culling
                    gl.enable(glow::CLIP_DISTANCE0);

                    // Extruded rhombus filament rendering
                    if let Some(rhombuses) = &self.toolpath_rhombuses {
                        gl.use_program(Some(self.rhombus_program));
                        gl.uniform_matrix_4_f32_slice(self.rhombus_uniforms.mvp.as_ref(), false, mvp);
                        gl.uniform_3_f32_slice(self.rhombus_uniforms.light_dir.as_ref(), light_dir);
                        gl.uniform_1_f32(self.rhombus_uniforms.clip_z.as_ref(), clip);
                        gl.uniform_1_f32(self.rhombus_uniforms.half_height.as_ref(), self.half_height);
                        draw_rhombus_batch(gl, rhombuses, clip, mvp, self.half_height);
                    }

                    gl.disable(glow::CLIP_DISTANCE0);
                } else {
                    // Toolpath lines (flat lines for extrusion paths)
                    if let Some(pl) = &self.toolpath_path_lines {
                        gl.use_program(Some(self.line_program));
                        gl.uniform_matrix_4_f32_slice(self.line_uniforms.mvp.as_ref(), false, mvp);
                        gl.uniform_1_f32(self.line_uniforms.clip_z.as_ref(), clip);
                        draw_buffer(gl, pl, glow::LINES);
                    }
                }

                // Toolpath travel lines
                if self.show_travel_moves {
                    if let Some(tl) = &self.toolpath_lines {
                        gl.use_program(Some(self.line_program));
                        gl.uniform_matrix_4_f32_slice(self.line_uniforms.mvp.as_ref(), false, mvp);
                        gl.uniform_1_f32(self.line_uniforms.clip_z.as_ref(), clip);
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
            gl.delete_program(self.rhombus_program);
            gl.delete_framebuffer(self.fbo);
            gl.delete_texture(self.fbo_color);
            gl.delete_renderbuffer(self.fbo_depth);

            // Delete regular GpuBuffers
            for buf in [&self.mesh, &self.slices, &self.current_slice, &self.toolpath_lines, &self.toolpath_path_lines]
                .into_iter()
                .flatten()
            {
                gl.delete_vertex_array(buf.vao);
                gl.delete_buffer(buf.vbo);
            }

            // Delete instanced batches
            if let Some(batch) = &self.toolpath_rhombuses {
                for &vao in &batch.layer_vaos {
                    gl.delete_vertex_array(vao);
                }
                gl.delete_buffer(batch.instance_vbo);
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

/// Push a rhombus instance: segment from (ax, ay, z) to (bx, by, z) with given nozzle width and color.
fn push_rhombus_instance(
    buf: &mut Vec<f32>,
    ax: f32, ay: f32, bx: f32, by: f32,
    z: f32, nozzle_width: f32,
    r: f32, g: f32, b: f32, a: f32,
    layer_z: f32,
) {
    let dx = bx - ax;
    let dy = by - ay;
    let length = (dx * dx + dy * dy).sqrt();
    if length < 1e-9 { return; }
    let dir_x = dx / length;
    let dir_y = dy / length;
    let half_width = nozzle_width * 0.5;
    buf.extend_from_slice(&[ax, ay, z, dir_x, dir_y, length, half_width, r, g, b, a, layer_z]);
}

/// Compute a conservative XY AABB for `count` rhombus instances starting at `first`.
/// Each segment's footprint is its endpoint pair expanded by nozzle half-width.
/// Returns (min_x, min_y, max_x, max_y).
fn layer_aabb_xy(instance_data: &[f32], first: usize, count: usize) -> (f32, f32, f32, f32) {
    let (mut mn_x, mut mn_y) = (f32::MAX, f32::MAX);
    let (mut mx_x, mut mx_y) = (f32::MIN, f32::MIN);
    for i in first..first + count {
        let idx = i * RHOMBUS_INSTANCE_STRIDE;
        let ax = instance_data[idx];
        let ay = instance_data[idx + 1];
        let dir_x  = instance_data[idx + 3];
        let dir_y  = instance_data[idx + 4];
        let len    = instance_data[idx + 5];
        let half_w = instance_data[idx + 6];
        let bx = ax + dir_x * len;
        let by = ay + dir_y * len;
        mn_x = mn_x.min(ax.min(bx) - half_w);
        mn_y = mn_y.min(ay.min(by) - half_w);
        mx_x = mx_x.max(ax.max(bx) + half_w);
        mx_y = mx_y.max(ay.max(by) + half_w);
    }
    (mn_x, mn_y, mx_x, mx_y)
}

/// Upload an instanced batch: instance data only, no prototype mesh.
/// Rhombus geometry is generated from gl_VertexID in the vertex shader.
unsafe fn upload_impostor_batch(
    gl: &glow::Context,
    instance_data: &[f32],
    instance_count: i32,
    _kind: BatchKind,
) -> InstancedBatch {
    let stride = (RHOMBUS_INSTANCE_STRIDE * 4) as i32; // 48 bytes

    // Upload all instance data into a single VBO (no VAO yet)
    let inst_vbo = gl.create_buffer().unwrap();
    gl.bind_buffer(glow::ARRAY_BUFFER, Some(inst_vbo));
    gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, cast_f32_u8(instance_data), glow::STATIC_DRAW);

    // Pass 1: find layer boundaries
    let mut layer_starts: Vec<(f32, usize)> = Vec::new();
    let mut last_layer_z: Option<f32> = None;
    for i in 0..instance_count as usize {
        let layer_z = instance_data[i * RHOMBUS_INSTANCE_STRIDE + 11];
        if last_layer_z != Some(layer_z) {
            layer_starts.push((layer_z, i));
            last_layer_z = Some(layer_z);
        }
    }

    // Pass 2: compute AABB per layer and build per-layer VAOs.
    // Each VAO has its attrib pointers pre-offset to that layer's byte range in inst_vbo,
    // emulating glDrawArraysInstancedBaseInstance without requiring GL 4.2.
    let n = layer_starts.len();
    let mut layer_entries: Vec<LayerEntry> = Vec::with_capacity(n);
    let mut layer_vaos: Vec<glow::VertexArray> = Vec::with_capacity(n);

    for i in 0..n {
        let (layer_z, first) = layer_starts[i];
        let count = if i + 1 < n { layer_starts[i + 1].1 - first } else { instance_count as usize - first };
        let base = (first as i32) * stride;  // byte offset into inst_vbo for this layer

        // Create a VAO whose attrib pointers start at this layer's byte offset
        let vao = gl.create_vertex_array().unwrap();
        gl.bind_vertex_array(Some(vao));
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(inst_vbo));
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, stride, base + 0);
        gl.vertex_attrib_divisor(0, 1);
        gl.enable_vertex_attrib_array(1);
        gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, stride, base + 3 * 4);
        gl.vertex_attrib_divisor(1, 1);
        gl.enable_vertex_attrib_array(2);
        gl.vertex_attrib_pointer_f32(2, 2, glow::FLOAT, false, stride, base + 5 * 4);
        gl.vertex_attrib_divisor(2, 1);
        gl.enable_vertex_attrib_array(3);
        gl.vertex_attrib_pointer_f32(3, 4, glow::FLOAT, false, stride, base + 7 * 4);
        gl.vertex_attrib_divisor(3, 1);
        gl.enable_vertex_attrib_array(4);
        gl.vertex_attrib_pointer_f32(4, 1, glow::FLOAT, false, stride, base + 11 * 4);
        gl.vertex_attrib_divisor(4, 1);
        gl.bind_vertex_array(None);
        layer_vaos.push(vao);

        let (mn_x, mn_y, mx_x, mx_y) = layer_aabb_xy(instance_data, first, count);
        layer_entries.push(LayerEntry {
            layer_z,
            instance_count: count as i32,
            aabb_min_x: mn_x, aabb_min_y: mn_y,
            aabb_max_x: mx_x, aabb_max_y: mx_y,
        });
    }

    gl.bind_buffer(glow::ARRAY_BUFFER, None);

    InstancedBatch {
        layer_vaos,
        instance_vbo: inst_vbo,
        instance_count,
        prototype_vertex_count: 36,
        layer_entries,
    }
}

/// Extract 6 frustum planes from MVP matrix using Gribb-Hartmann derivation
fn extract_frustum_planes(m: &[f32; 16]) -> [[f32; 4]; 6] {
    // m is column-major
    let get_row = |i: usize| -> [f32; 4] {
        [m[i], m[4 + i], m[8 + i], m[12 + i]]
    };
    let rows = [
        get_row(0),
        get_row(1),
        get_row(2),
        get_row(3),
    ];

    let add = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
        [a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]]
    };
    let sub = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
        [a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3]]
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
    min_x: f32, min_y: f32, min_z: f32,
    max_x: f32, max_y: f32, max_z: f32,
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

/// Draw a rhombus batch with CPU-side frustum culling per layer.
/// Layers are skipped if layer_z > clip_z or their AABB is outside the view frustum.
/// Adjacent visible layers are merged into contiguous BaseInstance draw calls to minimise draw-call count.
unsafe fn draw_rhombus_batch(
    gl: &glow::Context,
    batch: &InstancedBatch,
    clip_z: f32,
    mvp: &[f32; 16],
    half_height: f32,
) {
    let visible = batch.layer_entries.partition_point(|e| e.layer_z <= clip_z);
    if visible == 0 { return; }

    let planes = extract_frustum_planes(mvp);

    // Walk visible layers and merge contiguous non-culled ranges into single draw calls.
    // Each layer has its own VAO whose attrib pointers start at that layer's VBO offset,
    // so drawing `run_total` instances from layer_vaos[run_first_idx] reads layers
    // run_first_idx, run_first_idx+1, ... contiguously from the shared VBO.
    let mut run_vao_idx: Option<usize> = None;
    let mut run_total: i32 = 0;

    for (i, entry) in batch.layer_entries[..visible].iter().enumerate() {
        let outside = aabb_outside_frustum(
            &planes,
            entry.aabb_min_x, entry.aabb_min_y, entry.layer_z - half_height,
            entry.aabb_max_x, entry.aabb_max_y, entry.layer_z + half_height,
        );
        if outside {
            if let Some(vao_idx) = run_vao_idx.take() {
                gl.bind_vertex_array(Some(batch.layer_vaos[vao_idx]));
                gl.draw_arrays_instanced(glow::TRIANGLES, 0, batch.prototype_vertex_count, run_total);
                run_total = 0;
            }
        } else {
            if run_vao_idx.is_none() { run_vao_idx = Some(i); }
            run_total += entry.instance_count;
        }
    }
    // Flush any remaining run
    if let Some(vao_idx) = run_vao_idx {
        gl.bind_vertex_array(Some(batch.layer_vaos[vao_idx]));
        gl.draw_arrays_instanced(glow::TRIANGLES, 0, batch.prototype_vertex_count, run_total);
    }

    gl.bind_vertex_array(None);
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
