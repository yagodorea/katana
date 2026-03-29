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

// Impostor tube vertex shader: billboard quad per segment, cylinder shading in FS.
// No prototype mesh — quad corners computed from gl_VertexID (TRIANGLE_STRIP, 4 verts).
const IMPOSTOR_TUBE_VS: &str = r#"#version 330 core
// Per-instance data only — no prototype mesh
layout (location = 0) in vec3 a_inst_start;
layout (location = 1) in vec2 a_inst_dir;
layout (location = 2) in vec2 a_inst_scale;     // (length, radius)
layout (location = 3) in vec4 a_inst_color;
layout (location = 4) in float a_inst_layer_z;

uniform mat4 u_mvp;
uniform float u_clip_z;

flat out vec3 v_perp;
flat out vec3 v_perp2;
out float v_v;          // cross-tube coordinate, interpolates -1..+1
out vec4 v_color;
out float v_z;

void main() {
    float seg_len = a_inst_scale.x;
    float radius  = a_inst_scale.y;

    vec3 axis    = vec3(a_inst_dir, 0.0);
    vec3 seg_end = a_inst_start + axis * seg_len;

    // Camera forward extracted from orthographic MVP matrix
    vec3 cam_fwd = normalize(vec3(u_mvp[0][2], u_mvp[1][2], u_mvp[2][2]));

    // Billboard perpendicular: maximises visible tube width from camera
    vec3 perp = cross(axis, cam_fwd);
    float perp_len = length(perp);
    if (perp_len < 0.001) {
        // Degenerate: looking along tube axis — use lateral direction
        perp = vec3(-a_inst_dir.y, a_inst_dir.x, 0.0);
    } else {
        perp /= perp_len;
    }
    vec3 p2 = cross(axis, perp);   // always faces away from camera

    // Quad corners (TRIANGLE_STRIP: 0-1-2-3)
    vec3  base = (gl_VertexID < 2) ? a_inst_start : seg_end;
    float side = ((gl_VertexID & 1) == 0) ? -1.0 : 1.0;

    vec3 world_pos = base + perp * (radius * side);

    gl_Position      = u_mvp * vec4(world_pos, 1.0);
    gl_ClipDistance[0] = a_inst_layer_z - u_clip_z;

    v_perp  = perp;
    v_perp2 = p2;
    v_v     = side;
    v_color = a_inst_color;
    v_z     = a_inst_layer_z;
}
"#;

const IMPOSTOR_TUBE_FS: &str = r#"#version 330 core
flat in vec3 v_perp;
flat in vec3 v_perp2;
in float v_v;
in vec4 v_color;
in float v_z;

uniform vec3  u_light_dir;
uniform float u_clip_z;

out vec4 frag_color;

void main() {
    if (v_z < u_clip_z) discard;

    float cos_t = v_v;
    float sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));
    vec3 n = normalize(cos_t * v_perp + sin_t * v_perp2);

    float diffuse = abs(dot(n, u_light_dir));
    float ambient = 0.15;
    float light   = ambient + (1.0 - ambient) * diffuse;
    frag_color    = vec4(v_color.rgb * light, v_color.a);
}
"#;

// Impostor sphere vertex shader: camera-facing billboard quad, sphere shading in FS.
const IMPOSTOR_SPHERE_VS: &str = r#"#version 330 core
layout (location = 0) in vec3  a_inst_center;
layout (location = 1) in float a_inst_radius;
layout (location = 2) in vec4  a_inst_color;
layout (location = 3) in float a_inst_layer_z;

uniform mat4  u_mvp;
uniform float u_clip_z;

out vec2 v_uv;                 // billboard coords [-1,1]
flat out vec3 v_cam_right;
flat out vec3 v_cam_up;
flat out vec3 v_cam_fwd;
out vec4 v_color;
out float v_z;

void main() {
    // Camera basis from orthographic MVP matrix
    vec3 cam_right = normalize(vec3(u_mvp[0][0], u_mvp[1][0], u_mvp[2][0]));
    vec3 cam_up    = normalize(vec3(u_mvp[0][1], u_mvp[1][1], u_mvp[2][1]));

    // Quad corners via gl_VertexID (TRIANGLE_STRIP)
    float u = ((gl_VertexID & 1) == 0) ? -1.0 : 1.0;
    float v = ((gl_VertexID & 2) == 0) ? -1.0 : 1.0;

    vec3 world_pos = a_inst_center
        + cam_right * (u * a_inst_radius)
        + cam_up    * (v * a_inst_radius);

    gl_Position      = u_mvp * vec4(world_pos, 1.0);
    gl_ClipDistance[0] = a_inst_layer_z - u_clip_z;

    v_uv        = vec2(u, v);
    v_cam_right = cam_right;
    v_cam_up    = cam_up;
    v_cam_fwd   = normalize(vec3(u_mvp[0][2], u_mvp[1][2], u_mvp[2][2]));
    v_color     = a_inst_color;
    v_z         = a_inst_layer_z;
}
"#;

const IMPOSTOR_SPHERE_FS: &str = r#"#version 330 core
in vec2 v_uv;
flat in vec3 v_cam_right;
flat in vec3 v_cam_up;
flat in vec3 v_cam_fwd;
in vec4  v_color;
in float v_z;

uniform vec3  u_light_dir;
uniform float u_clip_z;

out vec4 frag_color;

void main() {
    if (v_z < u_clip_z) discard;

    float r2 = dot(v_uv, v_uv);
    if (r2 > 1.0) discard;          // outside sphere silhouette

    float nz = sqrt(1.0 - r2);
    vec3 n = normalize(v_uv.x * v_cam_right + v_uv.y * v_cam_up - nz * v_cam_fwd);

    float diffuse = abs(dot(n, u_light_dir));
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
const TUBE_INSTANCE_STRIDE: usize = 12;   // start(3) + dir(2) + scale(2) + color(4) + layer_z(1)
const SPHERE_INSTANCE_STRIDE: usize = 9;  // center(3) + radius(1) + color(4) + layer_z(1)

pub struct GpuBuffer {
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    vertex_count: i32,
}

enum BatchKind { Tube, Sphere }

struct InstancedBatch {
    vao: glow::VertexArray,
    instance_vbo: glow::Buffer,
    instance_count: i32,
    prototype_vertex_count: i32,
    // Layer index: sorted list of (layer_z, first_instance_index) for skipping invisible layers
    layer_starts: Vec<(f32, i32)>,  // (layer_z, first_instance_index)
}


pub struct Renderer {
    gl: Arc<glow::Context>,
    line_program: glow::Program,
    mesh_program: glow::Program,
    pub mesh: Option<GpuBuffer>,
    pub slices: Option<GpuBuffer>,
    pub current_slice: Option<GpuBuffer>,
    // Impostor toolpath rendering
    impostor_tube_program: glow::Program,
    impostor_sphere_program: glow::Program,
    toolpath_tubes: Option<InstancedBatch>,
    toolpath_spheres: Option<InstancedBatch>,
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
        let impostor_tube_program = unsafe { create_program(&gl, IMPOSTOR_TUBE_VS, IMPOSTOR_TUBE_FS) };
        let impostor_sphere_program = unsafe { create_program(&gl, IMPOSTOR_SPHERE_VS, IMPOSTOR_SPHERE_FS) };

        // Create FBO with depth buffer (start at 1x1, resized on first draw)
        let (fbo, fbo_color, fbo_depth) = unsafe { create_fbo(&gl, 1, 1) };

        Renderer {
            gl,
            line_program,
            mesh_program,
            mesh: None,
            slices: None,
            current_slice: None,
            impostor_tube_program,
            impostor_sphere_program,
            toolpath_tubes: None,
            toolpath_spheres: None,
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

    /// Upload planned toolpath layers using impostor rendering.
    /// Tube segments and sphere joints are rendered as billboard quads with analytical shading.
    pub fn upload_planned_toolpath(
        &mut self,
        planned_layers: &[PlannedLayer],
        nozzle_width: f32,
    ) {
        let mut tube_instances: Vec<f32> = Vec::new();
        let mut sphere_instances: Vec<f32> = Vec::new();
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
                            // Connection segment between perimeter loops
                            let from = &pts[0];
                            let to = &pts[1];
                            push_tube_instance(&mut tube_instances,
                                from.x, from.y, to.x, to.y,
                                z, nozzle_width,
                                r, g, b, a,
                                z,
                            );
                            let radius = nozzle_width * 0.5;
                            push_sphere_instance(&mut sphere_instances, from.x, from.y, z, radius, r, g, b, a, z);
                            push_sphere_instance(&mut sphere_instances, to.x, to.y, z, radius, r, g, b, a, z);
                            push_line_vert(&mut path_line_verts, from.x, from.y, z, r, g, b, a);
                            push_line_vert(&mut path_line_verts, to.x, to.y, z, r, g, b, a);
                        } else {
                            // Closed perimeter loop
                            let n = pts.len();
                            for s in 0..n {
                                let next = (s + 1) % n;
                                push_tube_instance(&mut tube_instances,
                                    pts[s].x, pts[s].y, pts[next].x, pts[next].y,
                                    z, nozzle_width,
                                    r, g, b, a,
                                    z,
                                );
                            }
                            let radius = nozzle_width * 0.5;
                            for p in pts {
                                push_sphere_instance(&mut sphere_instances, p.x, p.y, z, radius, r, g, b, a, z);
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
                            push_tube_instance(&mut tube_instances,
                                from.x, from.y, to.x, to.y,
                                z, nozzle_width,
                                r, g, b, a,
                                z,
                            );
                            let radius = nozzle_width * 0.5;
                            push_sphere_instance(&mut sphere_instances, from.x, from.y, z, radius, r, g, b, a, z);
                            push_sphere_instance(&mut sphere_instances, to.x, to.y, z, radius, r, g, b, a, z);
                            push_line_vert(&mut path_line_verts, from.x, from.y, z, r, g, b, a);
                            push_line_vert(&mut path_line_verts, to.x, to.y, z, r, g, b, a);
                        }
                    }
                    MoveKind::SurfaceInfill => {
                        let (r, g, b, a) = (0.9, 0.2, 0.7, 0.9);
                        if move_.points.len() >= 2 {
                            let from = &move_.points[0];
                            let to = &move_.points[1];
                            push_tube_instance(&mut tube_instances,
                                from.x, from.y, to.x, to.y,
                                z, nozzle_width,
                                r, g, b, a,
                                z,
                            );
                            let radius = nozzle_width * 0.5;
                            push_sphere_instance(&mut sphere_instances, from.x, from.y, z, radius, r, g, b, a, z);
                            push_sphere_instance(&mut sphere_instances, to.x, to.y, z, radius, r, g, b, a, z);
                            push_line_vert(&mut path_line_verts, from.x, from.y, z, r, g, b, a);
                            push_line_vert(&mut path_line_verts, to.x, to.y, z, r, g, b, a);
                        }
                    }
                }
            }
        }

        // Upload impostor batches (instance data only, no prototype meshes)
        self.toolpath_tubes = if tube_instances.is_empty() {
            None
        } else {
            let tube_count = (tube_instances.len() / TUBE_INSTANCE_STRIDE) as i32;
            Some(unsafe { upload_impostor_batch(&self.gl, &tube_instances, tube_count, BatchKind::Tube) })
        };

        self.toolpath_spheres = if sphere_instances.is_empty() {
            None
        } else {
            let sphere_count = (sphere_instances.len() / SPHERE_INSTANCE_STRIDE) as i32;
            Some(unsafe { upload_impostor_batch(&self.gl, &sphere_instances, sphere_count, BatchKind::Sphere) })
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
                    // Enable hardware clipping for layer culling (more efficient than fragment discard)
                    gl.enable(glow::CLIP_DISTANCE0);
                    
                    // 3D filament impostor rendering (billboard quads)

                    // Draw tube impostors
                    if let Some(tubes) = &self.toolpath_tubes {
                        gl.use_program(Some(self.impostor_tube_program));
                        let loc = gl.get_uniform_location(self.impostor_tube_program, "u_mvp");
                        gl.uniform_matrix_4_f32_slice(loc.as_ref(), false, mvp);
                        let loc = gl.get_uniform_location(self.impostor_tube_program, "u_light_dir");
                        gl.uniform_3_f32_slice(loc.as_ref(), light_dir);
                        let loc = gl.get_uniform_location(self.impostor_tube_program, "u_clip_z");
                        gl.uniform_1_f32(loc.as_ref(), clip);
                        draw_impostor_batch(gl, tubes, clip);
                    }

                    // Draw sphere impostors
                    if let Some(spheres) = &self.toolpath_spheres {
                        gl.use_program(Some(self.impostor_sphere_program));
                        let loc = gl.get_uniform_location(self.impostor_sphere_program, "u_mvp");
                        gl.uniform_matrix_4_f32_slice(loc.as_ref(), false, mvp);
                        let loc = gl.get_uniform_location(self.impostor_sphere_program, "u_light_dir");
                        gl.uniform_3_f32_slice(loc.as_ref(), light_dir);
                        let loc = gl.get_uniform_location(self.impostor_sphere_program, "u_clip_z");
                        gl.uniform_1_f32(loc.as_ref(), clip);
                        draw_impostor_batch(gl, spheres, clip);
                    }
                    
                    // Disable clipping after instanced drawing
                    gl.disable(glow::CLIP_DISTANCE0);
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
            gl.delete_program(self.impostor_tube_program);
            gl.delete_program(self.impostor_sphere_program);
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
            for batch in [&self.toolpath_tubes, &self.toolpath_spheres].into_iter().flatten() {
                gl.delete_vertex_array(batch.vao);
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

/// Push a tube instance: segment from (ax, ay, z) to (bx, by, z) with given nozzle width and color.
fn push_tube_instance(
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
    let radius = nozzle_width * 0.5;
    buf.extend_from_slice(&[ax, ay, z, dir_x, dir_y, length, radius, r, g, b, a, layer_z]);
}

/// Push a sphere instance at (cx, cy, cz) with given radius and color.
fn push_sphere_instance(
    buf: &mut Vec<f32>,
    cx: f32, cy: f32, cz: f32,
    radius: f32,
    r: f32, g: f32, b: f32, a: f32,
    layer_z: f32,
) {
    buf.extend_from_slice(&[cx, cy, cz, radius, r, g, b, a, layer_z]);
}

/// Upload an impostor batch: instance data only, no prototype mesh.
/// Quad corners are computed from gl_VertexID in the vertex shader.
unsafe fn upload_impostor_batch(
    gl: &glow::Context,
    instance_data: &[f32],
    instance_count: i32,
    kind: BatchKind,
) -> InstancedBatch {
    let vao = gl.create_vertex_array().unwrap();
    gl.bind_vertex_array(Some(vao));

    // Create and bind instance VBO (attributes start at location 0)
    let inst_vbo = gl.create_buffer().unwrap();
    gl.bind_buffer(glow::ARRAY_BUFFER, Some(inst_vbo));
    gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, cast_f32_u8(instance_data), glow::STATIC_DRAW);

    match kind {
        BatchKind::Tube => {
            let stride = (TUBE_INSTANCE_STRIDE * 4) as i32; // 48 bytes
            // slot 0: vec3 start (offset 0)
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, stride, 0);
            gl.vertex_attrib_divisor(0, 1);
            // slot 1: vec2 dir (offset 12)
            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, stride, 3 * 4);
            gl.vertex_attrib_divisor(1, 1);
            // slot 2: vec2 scale (length, radius) (offset 20)
            gl.enable_vertex_attrib_array(2);
            gl.vertex_attrib_pointer_f32(2, 2, glow::FLOAT, false, stride, 5 * 4);
            gl.vertex_attrib_divisor(2, 1);
            // slot 3: vec4 color (offset 28)
            gl.enable_vertex_attrib_array(3);
            gl.vertex_attrib_pointer_f32(3, 4, glow::FLOAT, false, stride, 7 * 4);
            gl.vertex_attrib_divisor(3, 1);
            // slot 4: float layer_z (offset 44)
            gl.enable_vertex_attrib_array(4);
            gl.vertex_attrib_pointer_f32(4, 1, glow::FLOAT, false, stride, 11 * 4);
            gl.vertex_attrib_divisor(4, 1);
        }
        BatchKind::Sphere => {
            let stride = (SPHERE_INSTANCE_STRIDE * 4) as i32; // 36 bytes
            // slot 0: vec3 center (offset 0)
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, stride, 0);
            gl.vertex_attrib_divisor(0, 1);
            // slot 1: float radius (offset 12)
            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_f32(1, 1, glow::FLOAT, false, stride, 3 * 4);
            gl.vertex_attrib_divisor(1, 1);
            // slot 2: vec4 color (offset 16)
            gl.enable_vertex_attrib_array(2);
            gl.vertex_attrib_pointer_f32(2, 4, glow::FLOAT, false, stride, 4 * 4);
            gl.vertex_attrib_divisor(2, 1);
            // slot 3: float layer_z (offset 32)
            gl.enable_vertex_attrib_array(3);
            gl.vertex_attrib_pointer_f32(3, 1, glow::FLOAT, false, stride, 8 * 4);
            gl.vertex_attrib_divisor(3, 1);
        }
    }

    gl.bind_vertex_array(None);

    // Build layer index: for each unique layer_z, record (layer_z, first_instance_index)
    let mut layer_starts: Vec<(f32, i32)> = Vec::new();
    let mut last_layer_z: Option<f32> = None;
    let layer_z_offset = match kind {
        BatchKind::Tube => 11,    // layer_z is at index 11 in tube instance data
        BatchKind::Sphere => 8,   // layer_z is at index 8 in sphere instance data
    };

    for i in 0..instance_count as isize {
        let idx = i as usize * match kind {
            BatchKind::Tube => TUBE_INSTANCE_STRIDE,
            BatchKind::Sphere => SPHERE_INSTANCE_STRIDE,
        };
        let layer_z = instance_data[idx + layer_z_offset];
        if last_layer_z != Some(layer_z) {
            layer_starts.push((layer_z, i as i32));
            last_layer_z = Some(layer_z);
        }
    }

    InstancedBatch {
        vao,
        instance_vbo: inst_vbo,
        instance_count,
        prototype_vertex_count: 4,  // impostor quad = 4 vertices (TRIANGLE_STRIP)
        layer_starts,
    }
}

/// Draw an impostor batch. Uses TRIANGLE_STRIP with 4 vertices per instance.
/// Hardware clip distance culls invisible layers efficiently.
unsafe fn draw_impostor_batch(gl: &glow::Context, batch: &InstancedBatch, _clip_z: f32) {
    gl.bind_vertex_array(Some(batch.vao));
    gl.draw_arrays_instanced(
        glow::TRIANGLE_STRIP,
        0,
        batch.prototype_vertex_count,  // 4
        batch.instance_count,
    );
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
