// Per-frame uniforms shared across line, mesh, and rhombus shaders.
// Layout MUST match the Rust `FrameUniforms` POD struct (buffers.rs).
// Total size: 96 bytes (multiple of 16).
struct Uniforms {
    mvp:         mat4x4<f32>, // 64 B, offset  0
    light_dir:   vec4<f32>,   // 16 B, offset 64  (.xyz used; .w unused — vec4 avoids vec3 alignment trap)
    clip_z_max:  f32,         //  4 B, offset 80
    clip_z_min:  f32,         //  4 B, offset 84
    half_height: f32,         //  4 B, offset 88
    half_width:  f32,         //  4 B, offset 92 (unused here; kept for layout parity)
    scrub_top_z: f32,         //  4 B, offset 96
    scrub_dim:   f32,         //  4 B, offset 100
    _pad0:       f32,         //  4 B, offset 104
    _pad1:       f32,         //  4 B, offset 108 — round struct size to 112
}

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsIn {
    @location(0) pos:   vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       color:    vec4<f32>,
    @location(1)       z:        f32,
}

@vertex fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    // Convert pos from world space to clip space using orthographic mvp
    out.clip_pos = u.mvp * vec4(in.pos, 1.0);
    out.color = in.color;
    out.z = in.pos.z;
    return out;
}

@fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // select(if_false, if_true, condition) <- pretty weird notation
    let dim = select(u.scrub_dim, 1.0, in.z >= u.scrub_top_z);
    return vec4(in.color.rgb * dim, in.color.a);
}
