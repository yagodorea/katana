// Rhombus-prism billboard impostor.
//
// VS: emits a 6-vertex segment-axis-aligned quad (2 triangles) per instance.
// FS: intersects the world-space ray with the rhombus prism analytically,
//     writes frag_depth from the hit point, shades via Lambertian.
//
// Layout MUST match Rust FrameUniforms (buffers.rs). Total: 112 bytes.
struct Uniforms {
    mvp:         mat4x4<f32>, // 64 B, offset  0
    light_dir:   vec4<f32>,   // 16 B, offset 64  (.xyz = world-space headlight dir)
    clip_z_max:  f32,         //  4 B, offset 80
    clip_z_min:  f32,         //  4 B, offset 84
    half_height: f32,         //  4 B, offset 88
    half_width:  f32,         //  4 B, offset 92
    cam_forward: vec4<f32>,   // 16 B, offset 96  (.xyz = ortho ray direction, world space)
}
@group(0) @binding(0) var<uniform> u: Uniforms;

struct Palette { colors: array<vec4<f32>, 16>, }
@group(0) @binding(1) var<uniform> palette: Palette;

// ---------------------------------------------------------------------------
// Vertex shader — billboard construction
// ---------------------------------------------------------------------------

struct VsIn {
    @location(0) start:       vec3<f32>,
    @location(1) dir:         vec2<f32>,
    @location(2) length:      f32,
    @location(3) color_flags: u32,       // color_id in bits 0-7
}

struct VsOut {
    @builtin(position)             clip_pos:   vec4<f32>,
    @location(0)                   world_pos:  vec3<f32>, // billboard surface pt → ortho ray origin
    @location(1) @interpolate(flat) seg_start: vec3<f32>,
    @location(2) @interpolate(flat) axis_long: vec3<f32>,
    @location(3) @interpolate(flat) seg_length: f32,
    @location(4) @interpolate(flat) color:     vec4<f32>,
}

fn get_axis_short(axis_long: vec3<f32>) -> vec3<f32> {
    let axis_short = normalize(cross(axis_long, u.cam_forward.xyz));
    if length(axis_short) >= 0.01 {
        return axis_short;
    }
    // Fallback 1
    let axis_short_f1 = normalize(cross(axis_long, vec3<f32>(0.0, 0.0, 1.0)));
    if length(axis_short_f1) >= 0.01 {
        return axis_short_f1;
    }
    // Fallback 2
    return normalize(cross(axis_long, vec3<f32>(1.0, 0.0, 0.0)));
}

@vertex fn vs_main(in: VsIn, @builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    let half_w   = u.half_width;
    let half_h   = u.half_height;
    let color_id = in.color_flags & 0xFFu;

    let axis_long = vec3<f32>(in.dir, 0.0); // placeholder (correct direction, wrong position)
    let r = sqrt(half_h * half_h + half_h * half_h);
    let axis_short = get_axis_short(axis_long);

    /**
     * Billboard calc
     * A ------- B
     * |  (0,0)  |
     * D ------- C
     * A = (-1, +1), B = (+1, +1), C = (+1, -1), D = (-1, -1)
     * start side: A, D
     * end side: C, B
     * Triangle 0 = A-D-C = (−1,+1) (−1,−1) (+1,−1)
     * Triangle 1 = A-C-B = (−1,+1) (+1,−1) (+1,+1)
     */
    // sign_along/sign_short lookup tables
    // ------------------------------------------ A     D     C     A     C    B
    let sa_lookup = array<f32, 6>(-1.0, -1.0,  1.0, -1.0,  1.0, 1.0);
    let ss_lookup = array<f32, 6>( 1.0, -1.0, -1.0,  1.0, -1.0, 1.0);
    let sa = sa_lookup[vid];
    let ss = ss_lookup[vid];

    let base_along = select(0.0, in.length, sa > 0.0);
    let world_pos = in.start
        + axis_long * (base_along + sa * r)
        + axis_short * (ss * r);

    out.clip_pos   = u.mvp * vec4<f32>(world_pos, 1.0);
    out.world_pos  = world_pos;
    out.seg_start  = in.start;
    out.axis_long  = axis_long;
    out.seg_length = in.length;
    out.color      = palette.colors[color_id];
    return out;
}

// ---------------------------------------------------------------------------
// Fragment shader — ray-rhombus-prism intersection
// ---------------------------------------------------------------------------

@fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let half_w = u.half_width;
    let half_h = u.half_height;

    // Segment-local coordinate frame.
    // axis_x = along segment, axis_y = XY-plane perpendicular, axis_z = world up.
    let axis_x = in.axis_long;
    let axis_y = normalize(vec3<f32>(-axis_x.y, axis_x.x, 0.0));
    let axis_z = vec3<f32>(0.0, 0.0, 1.0);

    // Orthographic ray: direction is constant (cam_forward), origin on billboard.
    let ray_o = in.world_pos;
    let ray_d = u.cam_forward.xyz;

    // Transform ray into segment-local space.
    let rel     = ray_o - in.seg_start;
    let local_o = vec3<f32>(dot(rel, axis_x),   dot(rel, axis_y),   dot(rel, axis_z));
    let local_d = vec3<f32>(dot(ray_d, axis_x), dot(ray_d, axis_y), dot(ray_d, axis_z));

    // ---- X slab: x ∈ [0, seg_length] (start/end caps) ----
    // enter_face: 4 = start cap (−x normal), 5 = end cap (+x normal), 0-3 = side faces
    var t_enter: f32 = -1e30;
    var t_exit:  f32 =  1e30;
    var enter_face: i32 = -1;

    let dx = local_d.x;
    let ox = local_o.x;
    if abs(dx) < 1e-9 {
        if ox < 0.0 || ox > in.seg_length { discard; }
    } else {
        let t0 = -ox / dx;
        let t1 = (in.seg_length - ox) / dx;
        if t0 < t1 {
            if t0 > t_enter { t_enter = t0; enter_face = 4; }
            if t1 < t_exit  { t_exit  = t1; }
        } else {
            if t1 > t_enter { t_enter = t1; enter_face = 5; }
            if t0 < t_exit  { t_exit  = t0; }
        }
    }

    // ---- 4 rhombus side-face planes: sy*y/hw + sz*z/hh = 1 ----
    // Face i: sy = (-1)^(~i&1), sz = (-1)^(~(i>>1)&1)
    // d_den < 0 → ray entering the half-space → t is enter candidate
    // d_den > 0 → ray exiting the half-space  → t is exit candidate
    for (var i = 0u; i < 4u; i = i + 1u) {
        let sy = select(-1.0, 1.0, (i & 1u) == 1u);
        let sz = select(-1.0, 1.0, (i & 2u) == 2u);
        let ny = sy / half_w;
        let nz = sz / half_h;
        let d_num = 1.0 - (ny * local_o.y + nz * local_o.z);
        let d_den = ny * local_d.y + nz * local_d.z;

        if abs(d_den) < 1e-9 {
            if d_num < 0.0 { discard; } // parallel and outside this half-space
        } else if d_den < 0.0 {
            let t = d_num / d_den;
            if t > t_enter { t_enter = t; enter_face = i32(i); }
        } else {
            let t = d_num / d_den;
            if t < t_exit { t_exit = t; }
        }
    }

    if t_enter > t_exit || t_exit < 0.0 { discard; }
    let t_hit = max(t_enter, 0.0);

    // Reconstruct world-space hit point.
    let lh       = local_o + t_hit * local_d;
    let world_hit = in.seg_start + lh.x * axis_x + lh.y * axis_y + lh.z * axis_z;

    // Normal from enter_face → world space.
    var n_local: vec3<f32>;
    switch enter_face {
        case 0:      { n_local = normalize(vec3<f32>(0.0, -1.0/half_w, -1.0/half_h)); }
        case 1:      { n_local = normalize(vec3<f32>(0.0,  1.0/half_w, -1.0/half_h)); }
        case 2:      { n_local = normalize(vec3<f32>(0.0, -1.0/half_w,  1.0/half_h)); }
        case 3:      { n_local = normalize(vec3<f32>(0.0,  1.0/half_w,  1.0/half_h)); }
        case 4:      { n_local = vec3<f32>(-1.0, 0.0, 0.0); }
        case 5:      { n_local = vec3<f32>( 1.0, 0.0, 0.0); }
        default:     { n_local = vec3<f32>( 0.0, 0.0, 1.0); }
    }
    let n_world  = n_local.x * axis_x + n_local.y * axis_y + n_local.z * axis_z;
    let diffuse  = abs(dot(n_world, u.light_dir.xyz));
    let light    = 0.15 + 0.85 * diffuse;

    return vec4<f32>(in.color.rgb * light, in.color.a);
}
