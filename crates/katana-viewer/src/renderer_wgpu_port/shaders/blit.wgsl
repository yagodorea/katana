// Fullscreen blit: copies the offscreen color texture onto egui's pass.
//
// Uses the "fullscreen triangle" trick — a single oversize triangle covering
// the [-1,1] viewport — to avoid the wasted vertex on a quad. Three vertices,
// no vertex buffer.

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       uv:       vec2<f32>,
}

@vertex fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Three positions covering the screen with a single triangle.
    // vid 0: (-1, -1), vid 1: (3, -1), vid 2: (-1, 3)
    let x = f32(i32(vid & 1u) * 4 - 1);   //  -1, 3, -1
    let y = f32(i32(vid >> 1u) * 4 - 1);  //  -1, -1, 3

    var out: VsOut;
    out.clip_pos = vec4<f32>(x, y, 0.0, 1.0);
    // Y-flip: clip-space y up, texture-space y down.
    out.uv = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
    return out;
}

@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

@fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(t, s, in.uv);
}
