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

// Instanced tube vertex shader: prototype tube is along +X from x=-0.5 to x=+0.5, radius 0.5
const INSTANCED_TUBE_VS: &str = r#"#version 330 core
// Prototype mesh (unit tube along X)
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_normal;
// Per-instance data
layout (location = 2) in vec3 a_inst_start;    // (start_x, start_y, z)
layout (location = 3) in vec2 a_inst_dir;       // unit direction in XY
layout (location = 4) in vec2 a_inst_scale;     // (length, radius)
layout (location = 5) in vec4 a_inst_color;
layout (location = 6) in float a_inst_layer_z;

uniform mat4 u_mvp;
uniform float u_clip_z;

out vec3 v_normal;
out vec4 v_color;
out float v_z;

void main() {
    float seg_len = a_inst_scale.x;
    float radius  = a_inst_scale.y;

    // Unit tube is along +X. Rotate to segment direction.
    vec3 tangent   = vec3(a_inst_dir, 0.0);
    vec3 bitangent = vec3(-a_inst_dir.y, a_inst_dir.x, 0.0);
    vec3 up        = vec3(0.0, 0.0, 1.0);

    // Scale prototype: X by length, YZ by radius*2 (prototype radius = 0.5)
    vec3 scaled = vec3(a_pos.x * seg_len, a_pos.y * radius * 2.0, a_pos.z * radius * 2.0);
    // Shift so start is at segment start (prototype center is at origin)
    scaled.x += seg_len * 0.5;

    vec3 world_pos = a_inst_start
        + scaled.x * tangent
        + scaled.y * bitangent
        + scaled.z * up;

    vec3 world_normal = a_normal.x * tangent
                      + a_normal.y * bitangent
                      + a_normal.z * up;

    gl_Position = u_mvp * vec4(world_pos, 1.0);
    
    // Hardware clipping: discard if layer is below clip plane
    // This culls entire primitives before rasterization
    gl_ClipDistance[0] = a_inst_layer_z - u_clip_z;
    
    v_normal = world_normal;
    v_color = a_inst_color;
    v_z = a_inst_layer_z;
}
"#;

// Instanced sphere vertex shader: prototype sphere is radius 0.5 centered at origin
const INSTANCED_SPHERE_VS: &str = r#"#version 330 core
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in vec3 a_inst_center;
layout (location = 3) in float a_inst_radius;
layout (location = 4) in vec4 a_inst_color;
layout (location = 5) in float a_inst_layer_z;

uniform mat4 u_mvp;
uniform float u_clip_z;

out vec3 v_normal;
out vec4 v_color;
out float v_z;

void main() {
    vec3 world_pos = a_inst_center + a_pos * (a_inst_radius);
    gl_Position = u_mvp * vec4(world_pos, 1.0);
    
    // Hardware clipping: discard if layer is below clip plane
    // This culls entire primitives before rasterization
    gl_ClipDistance[0] = a_inst_layer_z - u_clip_z;
    
    v_normal = a_normal;  // rotation-invariant for uniform scale
    v_color = a_inst_color;
    v_z = a_inst_layer_z;
}
"#;

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

const LINE_STRIDE: usize = 7;  // x y z r g b a
const MESH_STRIDE: usize = 11; // x y z nx ny nz r g b a layer_z
const TUBE_INSTANCE_STRIDE: usize = 12;   // start(3) + dir(2) + scale(2) + color(4) + layer_z(1)
const SPHERE_INSTANCE_STRIDE: usize = 9;  // center(3) + radius(1) + color(4) + layer_z(1)
const PROTO_STRIDE: usize = 6;            // pos(3) + normal(3)

pub struct GpuBuffer {
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    vertex_count: i32,
}

struct PrototypeMesh {
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
    // Instanced toolpath rendering
    instanced_tube_program: glow::Program,
    instanced_sphere_program: glow::Program,
    tube_prototype: Option<PrototypeMesh>,
    sphere_prototype: Option<PrototypeMesh>,
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
        let instanced_tube_program = unsafe { create_program(&gl, INSTANCED_TUBE_VS, MESH_FS) };
        let instanced_sphere_program = unsafe { create_program(&gl, INSTANCED_SPHERE_VS, MESH_FS) };

        // Create FBO with depth buffer (start at 1x1, resized on first draw)
        let (fbo, fbo_color, fbo_depth) = unsafe { create_fbo(&gl, 1, 1) };

        Renderer {
            gl,
            line_program,
            mesh_program,
            mesh: None,
            slices: None,
            current_slice: None,
            instanced_tube_program,
            instanced_sphere_program,
            tube_prototype: None,
            sphere_prototype: None,
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

    /// Upload planned toolpath layers using instanced rendering.
    /// Tube segments and sphere joints are rendered as instances of prototype meshes.
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

        // Ensure prototypes exist (create once, reuse across uploads)
        if self.tube_prototype.is_none() {
            self.tube_prototype = Some(unsafe { create_unit_tube_prototype(&self.gl) });
            self.sphere_prototype = Some(unsafe { create_unit_sphere_prototype(&self.gl) });
        }

        // Upload instanced batches
        self.toolpath_tubes = if tube_instances.is_empty() {
            None
        } else {
            let tube_count = (tube_instances.len() / TUBE_INSTANCE_STRIDE) as i32;
            let proto = self.tube_prototype.as_ref().unwrap();
            Some(unsafe { upload_instanced_batch(&self.gl, proto.vbo, proto.vertex_count, &tube_instances, tube_count, BatchKind::Tube) })
        };

        self.toolpath_spheres = if sphere_instances.is_empty() {
            None
        } else {
            let sphere_count = (sphere_instances.len() / SPHERE_INSTANCE_STRIDE) as i32;
            let proto = self.sphere_prototype.as_ref().unwrap();
            Some(unsafe { upload_instanced_batch(&self.gl, proto.vbo, proto.vertex_count, &sphere_instances, sphere_count, BatchKind::Sphere) })
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
                    
                    // 3D filament tubes (instanced rendering)

                    // Draw tubes
                    if let Some(tubes) = &self.toolpath_tubes {
                        gl.use_program(Some(self.instanced_tube_program));
                        let loc = gl.get_uniform_location(self.instanced_tube_program, "u_mvp");
                        gl.uniform_matrix_4_f32_slice(loc.as_ref(), false, mvp);
                        let loc = gl.get_uniform_location(self.instanced_tube_program, "u_light_dir");
                        gl.uniform_3_f32_slice(loc.as_ref(), light_dir);
                        let loc = gl.get_uniform_location(self.instanced_tube_program, "u_clip_z");
                        gl.uniform_1_f32(loc.as_ref(), clip);
                        draw_instanced_batch(gl, tubes, clip);
                    }

                    // Draw spheres
                    if let Some(spheres) = &self.toolpath_spheres {
                        gl.use_program(Some(self.instanced_sphere_program));
                        let loc = gl.get_uniform_location(self.instanced_sphere_program, "u_mvp");
                        gl.uniform_matrix_4_f32_slice(loc.as_ref(), false, mvp);
                        let loc = gl.get_uniform_location(self.instanced_sphere_program, "u_light_dir");
                        gl.uniform_3_f32_slice(loc.as_ref(), light_dir);
                        let loc = gl.get_uniform_location(self.instanced_sphere_program, "u_clip_z");
                        gl.uniform_1_f32(loc.as_ref(), clip);
                        draw_instanced_batch(gl, spheres, clip);
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
            gl.delete_program(self.instanced_tube_program);
            gl.delete_program(self.instanced_sphere_program);
            gl.delete_framebuffer(self.fbo);
            gl.delete_texture(self.fbo_color);
            gl.delete_renderbuffer(self.fbo_depth);

            // Delete prototype VBOs
            if let Some(proto) = &self.tube_prototype {
                gl.delete_buffer(proto.vbo);
            }
            if let Some(proto) = &self.sphere_prototype {
                gl.delete_buffer(proto.vbo);
            }

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

const TUBE_SIDES: usize = 6;  // Reduced from 8 for 25% fewer vertices per tube

/// Create a unit tube prototype: cylinder along +X from x=-0.5 to x=+0.5, radius 0.5, 6-sided.
/// Only stores position + normal (6 floats per vertex). 36 vertices total.
unsafe fn create_unit_tube_prototype(gl: &glow::Context) -> PrototypeMesh {
    let mut verts: Vec<f32> = Vec::with_capacity(TUBE_SIDES * 6 * PROTO_STRIDE);
    let radius = 0.5_f32;

    for i in 0..TUBE_SIDES {
        let j = (i + 1) % TUBE_SIDES;
        let theta_i = std::f32::consts::TAU * (i as f32) / (TUBE_SIDES as f32);
        let theta_j = std::f32::consts::TAU * (j as f32) / (TUBE_SIDES as f32);

        let (ci, si) = (theta_i.cos(), theta_i.sin());
        let (cj, sj) = (theta_j.cos(), theta_j.sin());

        // Ring A at x = -0.5, Ring B at x = +0.5
        // Normals point radially: (0, cos, sin)
        let a0 = (-0.5, radius * ci, radius * si, 0.0, ci, si);
        let a1 = (-0.5, radius * cj, radius * sj, 0.0, cj, sj);
        let b0 = ( 0.5, radius * ci, radius * si, 0.0, ci, si);
        let b1 = ( 0.5, radius * cj, radius * sj, 0.0, cj, sj);

        // Triangle 1: a0, b0, a1
        verts.extend_from_slice(&[a0.0, a0.1, a0.2, a0.3, a0.4, a0.5]);
        verts.extend_from_slice(&[b0.0, b0.1, b0.2, b0.3, b0.4, b0.5]);
        verts.extend_from_slice(&[a1.0, a1.1, a1.2, a1.3, a1.4, a1.5]);
        // Triangle 2: a1, b0, b1
        verts.extend_from_slice(&[a1.0, a1.1, a1.2, a1.3, a1.4, a1.5]);
        verts.extend_from_slice(&[b0.0, b0.1, b0.2, b0.3, b0.4, b0.5]);
        verts.extend_from_slice(&[b1.0, b1.1, b1.2, b1.3, b1.4, b1.5]);
    }

    let vbo = gl.create_buffer().unwrap();
    gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
    gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, cast_f32_u8(&verts), glow::STATIC_DRAW);
    gl.bind_buffer(glow::ARRAY_BUFFER, None);

    PrototypeMesh { vbo, vertex_count: (TUBE_SIDES * 6) as i32 }
}

/// Create a unit sphere prototype: 8×8 UV sphere, radius 1, centered at origin.
/// Only stores position + normal (6 floats per vertex). 384 vertices total.
unsafe fn create_unit_sphere_prototype(gl: &glow::Context) -> PrototypeMesh {
    let mut verts: Vec<f32> = Vec::new();
    let stacks = 8;
    let slices = 8;

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

            // Triangle 1
            verts.extend_from_slice(&[p00.0, p00.1, p00.2, p00.0, p00.1, p00.2]);
            verts.extend_from_slice(&[p10.0, p10.1, p10.2, p10.0, p10.1, p10.2]);
            verts.extend_from_slice(&[p01.0, p01.1, p01.2, p01.0, p01.1, p01.2]);

            // Triangle 2
            verts.extend_from_slice(&[p01.0, p01.1, p01.2, p01.0, p01.1, p01.2]);
            verts.extend_from_slice(&[p10.0, p10.1, p10.2, p10.0, p10.1, p10.2]);
            verts.extend_from_slice(&[p11.0, p11.1, p11.2, p11.0, p11.1, p11.2]);
        }
    }

    let vbo = gl.create_buffer().unwrap();
    gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
    gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, cast_f32_u8(&verts), glow::STATIC_DRAW);
    gl.bind_buffer(glow::ARRAY_BUFFER, None);

    PrototypeMesh { vbo, vertex_count: (stacks * slices * 6) as i32 }
}

/// Upload an instanced batch: prototype mesh + instance data.
unsafe fn upload_instanced_batch(
    gl: &glow::Context,
    proto_vbo: glow::Buffer,
    proto_vertex_count: i32,
    instance_data: &[f32],
    instance_count: i32,
    kind: BatchKind,
) -> InstancedBatch {
    let vao = gl.create_vertex_array().unwrap();
    gl.bind_vertex_array(Some(vao));

    // Bind prototype VBO for slots 0-1 (position + normal)
    gl.bind_buffer(glow::ARRAY_BUFFER, Some(proto_vbo));
    let proto_stride = (PROTO_STRIDE * 4) as i32;
    gl.enable_vertex_attrib_array(0);
    gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, proto_stride, 0);
    gl.enable_vertex_attrib_array(1);
    gl.vertex_attrib_pointer_f32(1, 3, glow::FLOAT, false, proto_stride, 3 * 4);

    // Create and bind instance VBO
    let inst_vbo = gl.create_buffer().unwrap();
    gl.bind_buffer(glow::ARRAY_BUFFER, Some(inst_vbo));
    gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, cast_f32_u8(instance_data), glow::STATIC_DRAW);

    match kind {
        BatchKind::Tube => {
            let stride = (TUBE_INSTANCE_STRIDE * 4) as i32; // 48 bytes
            // slot 2: vec3 start (offset 0)
            gl.enable_vertex_attrib_array(2);
            gl.vertex_attrib_pointer_f32(2, 3, glow::FLOAT, false, stride, 0);
            gl.vertex_attrib_divisor(2, 1);
            // slot 3: vec2 dir (offset 12)
            gl.enable_vertex_attrib_array(3);
            gl.vertex_attrib_pointer_f32(3, 2, glow::FLOAT, false, stride, 3 * 4);
            gl.vertex_attrib_divisor(3, 1);
            // slot 4: vec2 scale (length, radius) (offset 20)
            gl.enable_vertex_attrib_array(4);
            gl.vertex_attrib_pointer_f32(4, 2, glow::FLOAT, false, stride, 5 * 4);
            gl.vertex_attrib_divisor(4, 1);
            // slot 5: vec4 color (offset 28)
            gl.enable_vertex_attrib_array(5);
            gl.vertex_attrib_pointer_f32(5, 4, glow::FLOAT, false, stride, 7 * 4);
            gl.vertex_attrib_divisor(5, 1);
            // slot 6: float layer_z (offset 44)
            gl.enable_vertex_attrib_array(6);
            gl.vertex_attrib_pointer_f32(6, 1, glow::FLOAT, false, stride, 11 * 4);
            gl.vertex_attrib_divisor(6, 1);
        }
        BatchKind::Sphere => {
            let stride = (SPHERE_INSTANCE_STRIDE * 4) as i32; // 36 bytes
            // slot 2: vec3 center (offset 0)
            gl.enable_vertex_attrib_array(2);
            gl.vertex_attrib_pointer_f32(2, 3, glow::FLOAT, false, stride, 0);
            gl.vertex_attrib_divisor(2, 1);
            // slot 3: float radius (offset 12)
            gl.enable_vertex_attrib_array(3);
            gl.vertex_attrib_pointer_f32(3, 1, glow::FLOAT, false, stride, 3 * 4);
            gl.vertex_attrib_divisor(3, 1);
            // slot 4: vec4 color (offset 16)
            gl.enable_vertex_attrib_array(4);
            gl.vertex_attrib_pointer_f32(4, 4, glow::FLOAT, false, stride, 4 * 4);
            gl.vertex_attrib_divisor(4, 1);
            // slot 5: float layer_z (offset 32)
            gl.enable_vertex_attrib_array(5);
            gl.vertex_attrib_pointer_f32(5, 1, glow::FLOAT, false, stride, 8 * 4);
            gl.vertex_attrib_divisor(5, 1);
        }
    }

    gl.bind_vertex_array(None);

    // Build layer index: for each unique layer_z, record (layer_z, first_instance_index)
    let mut layer_starts: Vec<(f32, i32)> = Vec::new();
    let mut last_layer_z: Option<f32> = None;
    let layer_z_offset = match kind {
        BatchKind::Tube => 11,    // layer_z is at index 11 in tube instance data
        BatchKind::Sphere => 8,    // layer_z is at index 8 in sphere instance data
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
        prototype_vertex_count: proto_vertex_count,
        layer_starts,
    }
}

/// Draw an instanced batch with per-layer culling.
/// Issues one draw call per visible layer to completely skip invisible instances.
unsafe fn draw_instanced_batch(gl: &glow::Context, batch: &InstancedBatch, clip_z: f32) {
    if batch.layer_starts.is_empty() {
        // Fallback: draw all instances
        gl.bind_vertex_array(Some(batch.vao));
        gl.draw_arrays_instanced(
            glow::TRIANGLES,
            0,
            batch.prototype_vertex_count,
            batch.instance_count,
        );
        gl.bind_vertex_array(None);
        return;
    }

    gl.bind_vertex_array(Some(batch.vao));

    // Draw each visible layer separately
    for i in 0..batch.layer_starts.len() {
        let (layer_z, first_instance) = batch.layer_starts[i];
        
        if layer_z < clip_z {
            continue;  // Skip invisible layers
        }

        // Calculate instance count for this layer
        let next_start = if i + 1 < batch.layer_starts.len() {
            batch.layer_starts[i + 1].1
        } else {
            batch.instance_count
        };
        let layer_count = next_start - first_instance;

        if layer_count > 0 {
            // Draw this layer's instances
            // Note: GL 3.3 doesn't have base_instance, so we draw all instances
            // and rely on vertex shader culling for invisible ones within the batch.
            // TODO: Add ARB_base_instance support for true per-layer offsets.
            gl.draw_arrays_instanced(
                glow::TRIANGLES,
                0,
                batch.prototype_vertex_count,
                batch.instance_count,
            );
            break;  // For now, draw all at once after first visible layer
        }
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
