// Per-frame uniforms shared across line, mesh, and rhombus shaders.
// Layout MUST match the Rust `FrameUniforms` POD struct (buffers.rs).
// Total size: 96 bytes (multiple of 16).
struct Uniforms {
    mvp:         mat4x4<f32>, // 64 B, offset  0
    light_dir:   vec4<f32>,   // 16 B, offset 64  (.xyz used; .w unused — vec4 avoids vec3 alignment trap)
    clip_z_max:  f32,         //  4 B, offset 80
    clip_z_min:  f32,         //  4 B, offset 84
    half_height: f32,         //  4 B, offset 88
    half_width:  f32,         //  4 B, offset 92
    scrub_top_z: f32,         //  4 B, offset 96
    scrub_dim:   f32,         //  4 B, offset 100
    _pad0:       f32,         //  4 B, offset 104
    _pad1:       f32,         //  4 B, offset 108 — round struct size to 112
}

@group(0) @binding(0) var<uniform> u: Uniforms;

// Uniform arrays require 16-byte element stride; pack 4 u32s per vec4.
// 36 entries = 9 vec4<u32>s. Access: entries[vid >> 2u][vid & 3u].
struct VertexTable {
    entries: array<vec4<u32>, 9>,
}
@group(0) @binding(1) var<uniform> vertex_table: VertexTable;

struct Palette {
    colors: array<vec4<f32>, 16>,
}
@group(0) @binding(2) var<uniform> palette: Palette;

struct VsIn {
    @location(0) start:       vec3<f32>,  // (Float32x3, offset 0)
    @location(1) dir:         vec2<f32>,  // (Snorm16x2, offset 12; hardware divides i16 by 32767)
    @location(2) length:      f32,        // (Float32, offset 16)
    @location(3) color_flags: u32,        // (Uint32, offset 20; color_id in bits 0-7)
}

struct VsOut {
    @builtin(position)              clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) normal:   vec3<f32>,
    @location(1)                    color:    vec4<f32>,
    @location(2)                    layer_z:  f32,
}

@vertex fn vs_main(in: VsIn, @builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;

    let color_id = in.color_flags & 0xFFu;

    // ----- begin geometry -----
    let half_h  = u.half_height;
    let half_w  = u.half_width;
    let seg_dir = vec3(in.dir, 0.0);
    let perp    = vec3(-seg_dir.y, seg_dir.x, 0.0);
    let seg_len = in.length;
    let up = vec3(0.0, 0.0, 1.0);

    let cross_offsets = array<vec3<f32>, 4>(
        perp * half_w, up * half_h, -(perp * half_w), -(up * half_h));

    let normals = array<vec3<f32>, 6>(
        normalize(perp + up), normalize(-perp + up),
        normalize(-perp - up), normalize(perp - up),
        -seg_dir, seg_dir);

    let packed    = vertex_table.entries[vid >> 2u][vid & 3u];
    let cross_idx = packed & 3u;
    let along_f   = (packed >> 2u) & 1u;
    let norm_idx  = (packed >> 3u) & 7u;

    let cross_off = cross_offsets[cross_idx];
    let along_off = seg_dir * (seg_len * f32(along_f));
    let norm      = normals[norm_idx];

    let world_pos = in.start + along_off + cross_off;
    out.clip_pos = u.mvp * vec4(world_pos, 1.0);
    out.normal = norm;
    out.color  = palette.colors[color_id];
    out.layer_z = in.start.z;
    // ----- end geometry -----

    return out;
}

@fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let diffuse = abs(dot(in.normal, u.light_dir.xyz));
    let ambient = 0.15;
    let light = ambient + (1.0 - ambient) * diffuse;

    let dim = select(u.scrub_dim, 1.0, in.layer_z >= u.scrub_top_z);

    return vec4(in.color.rgb * light * dim, in.color.a);
}
