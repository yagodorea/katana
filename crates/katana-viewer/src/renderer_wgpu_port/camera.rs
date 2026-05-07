//! Camera math: orbit MVP and headlight direction.
//!
//! Output convention is GL-style clip space (z in [-1, +1]); the renderer
//! remaps to wgpu's [0, +1] when writing the per-frame uniform.

pub fn build_mvp(
    center: [f32; 3],
    azimuth: f32,
    elevation: f32,
    zoom: f32,
    extent: f32,
    aspect: f32,
    _pan: (f32, f32),
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

    // Orbit camera: positive elevation places the camera above the target.
    // Right = (ca, -sa, 0); Up = (-se·sa, -se·ca, ce);
    // Forward (into scene) = -(ce·sa, ce·ca, se).
    let r00 = sx * ca;
    let r01 = sx * (-sa);
    let r02 = 0.0;
    let r10 = sy * (-se) * sa;
    let r11 = sy * (-se) * ca;
    let r12 = sy * ce;
    let r20 = -sz * ce * sa;
    let r21 = -sz * ce * ca;
    let r22 = -sz * se;

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
