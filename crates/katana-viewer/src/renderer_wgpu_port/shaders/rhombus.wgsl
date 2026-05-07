// Per-frame uniforms shared across line, mesh, and rhombus shaders.
// Layout MUST match the Rust `FrameUniforms` POD struct (buffers.rs).
// Total size: 96 bytes (multiple of 16).
struct Uniforms {
    mvp:         mat4x4<f32>, // 64 B, offset  0
    light_dir:   vec4<f32>,   // 16 B, offset 64  (.xyz used; .w unused — vec4 avoids vec3 alignment trap)
    clip_z_max:  f32,         //  4 B, offset 80
    clip_z_min:  f32,         //  4 B, offset 84
    half_height: f32,         //  4 B, offset 88
    _pad:        f32,         //  4 B, offset 92 — round struct size to 96
}

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsIn {
    @location(0) start:      vec3<f32>,
    @location(1) direction:  vec2<f32>,
    @location(2) scale:      vec2<f32>, // (length, half_width)
    @location(3) color:      vec4<f32>,
    @location(4) layer_z:    f32,
}

struct VsOut {
    @builtin(position)              clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) normal:   vec3<f32>,
    @location(1)                    color:    vec4<f32>,
    @location(2)                    layer_z:  f32,
}

@vertex fn vs_main(in: VsIn, @builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    let seg_len = in.scale.x;
    let half_w = in.scale.y;
    let half_h = u.half_height;

    let seg_dir = vec3(in.direction, 0.0);
    let perp = vec3(-in.direction.y, in.direction.x, 0.0);
    let up = vec3(0.0, 0.0, 1.0);

    // 4 cross-section offsets
    let r_off = perp * half_w;    // right
    let t_off = up   * half_h;    // top
    let l_off = -perp * half_w;   // left
    let b_off = -up   * half_h;   // bottom

    // 4 side face normals (normalized bisectors of adjacent edges)
    let n_rt = normalize(perp + up);     // right-top face
    let n_tl = normalize(-perp + up);    // top-left face
    let n_lb = normalize(-perp - up);    // left-bottom face
    let n_br = normalize(perp - up);     // bottom-right face

    var cross_off: vec3<f32>;
    var along_off: vec3<f32>;
    var norm: vec3<f32>;

    switch (vid) {
        // Side face 0: right-top (triangles 0, 1)
        case 0u:  { cross_off = r_off; along_off = vec3(0.0);              norm = n_rt; }
        case 1u:  { cross_off = t_off; along_off = vec3(0.0);              norm = n_rt; }
        case 2u:  { cross_off = r_off; along_off = seg_dir * seg_len;    norm = n_rt; }
        case 3u:  { cross_off = t_off; along_off = vec3(0.0);              norm = n_rt; }
        case 4u:  { cross_off = t_off; along_off = seg_dir * seg_len;    norm = n_rt; }
        case 5u:  { cross_off = r_off; along_off = seg_dir * seg_len;    norm = n_rt; }
        // Side face 1: top-left (triangles 2, 3)
        case 6u:  { cross_off = t_off; along_off = vec3(0.0);              norm = n_tl; }
        case 7u:  { cross_off = l_off; along_off = vec3(0.0);              norm = n_tl; }
        case 8u:  { cross_off = t_off; along_off = seg_dir * seg_len;    norm = n_tl; }
        case 9u:  { cross_off = l_off; along_off = vec3(0.0);              norm = n_tl; }
        case 10u: { cross_off = l_off; along_off = seg_dir * seg_len;    norm = n_tl; }
        case 11u: { cross_off = t_off; along_off = seg_dir * seg_len;    norm = n_tl; }
        // Side face 2: left-bottom (triangles 4, 5)
        case 12u: { cross_off = l_off; along_off = vec3(0.0);              norm = n_lb; }
        case 13u: { cross_off = b_off; along_off = vec3(0.0);              norm = n_lb; }
        case 14u: { cross_off = l_off; along_off = seg_dir * seg_len;    norm = n_lb; }
        case 15u: { cross_off = b_off; along_off = vec3(0.0);              norm = n_lb; }
        case 16u: { cross_off = b_off; along_off = seg_dir * seg_len;    norm = n_lb; }
        case 17u: { cross_off = l_off; along_off = seg_dir * seg_len;    norm = n_lb; }
        // Side face 3: bottom-right (triangles 6, 7)
        case 18u: { cross_off = b_off; along_off = vec3(0.0);              norm = n_br; }
        case 19u: { cross_off = r_off; along_off = vec3(0.0);              norm = n_br; }
        case 20u: { cross_off = b_off; along_off = seg_dir * seg_len;    norm = n_br; }
        case 21u: { cross_off = r_off; along_off = vec3(0.0);              norm = n_br; }
        case 22u: { cross_off = r_off; along_off = seg_dir * seg_len;    norm = n_br; }
        case 23u: { cross_off = b_off; along_off = seg_dir * seg_len;    norm = n_br; }
        // Start cap (triangles 8, 9), normal = -seg_dir
        case 24u: { cross_off = r_off; along_off = vec3(0.0);              norm = -seg_dir; }
        case 25u: { cross_off = t_off; along_off = vec3(0.0);              norm = -seg_dir; }
        case 26u: { cross_off = l_off; along_off = vec3(0.0);              norm = -seg_dir; }
        case 27u: { cross_off = r_off; along_off = vec3(0.0);              norm = -seg_dir; }
        case 28u: { cross_off = l_off; along_off = vec3(0.0);              norm = -seg_dir; }
        case 29u: { cross_off = b_off; along_off = vec3(0.0);              norm = -seg_dir; }
        // End cap (triangles 10, 11), normal = +seg_dir
        case 30u: { cross_off = r_off; along_off = seg_dir * seg_len;    norm = seg_dir; }
        case 31u: { cross_off = l_off; along_off = seg_dir * seg_len;    norm = seg_dir; }
        case 32u: { cross_off = t_off; along_off = seg_dir * seg_len;    norm = seg_dir; }
        case 33u: { cross_off = r_off; along_off = seg_dir * seg_len;    norm = seg_dir; }
        case 34u: { cross_off = b_off; along_off = seg_dir * seg_len;    norm = seg_dir; }
        default:  { cross_off = l_off; along_off = seg_dir * seg_len;    norm = seg_dir; }
    }

    let world_pos = in.start + along_off + cross_off;
    out.clip_pos = u.mvp * vec4(world_pos, 1.0);
    out.normal = norm;
    out.color = in.color;
    out.layer_z = in.layer_z;

    return out;
}

@fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if (in.layer_z > u.clip_z_max || in.layer_z < u.clip_z_min) {
        discard;
    }

    let diffuse = abs(dot(in.normal, u.light_dir.xyz));
    let ambient = 0.15;
    let light = ambient + (1.0 - ambient) * diffuse;
    return vec4(in.color.rgb * light, in.color.a);
}
