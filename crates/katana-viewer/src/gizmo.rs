//! Viewport transform gizmo (Move / Rotate / Scale), drawn as a screen-space
//! egui overlay on top of the 3D scene.
//!
//! Everything works off the same column-major MVP that `build_mvp` produces:
//! world points are projected to screen once per frame, and all hit-testing
//! and drag math happen in 2D screen space. Drag math is snapshot-based —
//! new values are always recomputed from the state captured at drag start and
//! the absolute cursor position (never integrated per-frame deltas).

use eframe::egui::{ self, Color32, Painter, Pos2, Rect, Shape, Stroke, Vec2 };
use nalgebra::{ UnitQuaternion, Vector2, Vector3 };

#[derive(PartialEq, Clone, Copy)]
pub enum GizmoMode {
    Move,
    Rotate,
    Scale,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GizmoAxis {
    X,
    Y,
    Z,
    /// Center handle: uniform scale (Scale mode only).
    Uniform,
}

/// Snapshot taken on pointer-down over a handle; drag frames recompute the
/// transform from this + the current cursor position.
pub struct GizmoDrag {
    pub mode: GizmoMode,
    pub axis: GizmoAxis,
    pub start_cursor: Pos2,
    // Transform values at drag start
    pub start_translation: Vector2<f32>,
    pub start_rotation: UnitQuaternion<f32>,
    pub start_scale: Vector3<f32>,
    // Screen-space geometry at drag start
    pub center_px: Pos2, // projected anchor
    pub axis_screen_dir: Vec2, // normalized screen direction of the axis
    pub px_per_mm: f32, // screen px per world mm along that axis
    pub start_angle: f32, // rotate: cursor angle around center at start
    pub winding: f32, // rotate: +1/-1 mapping screen angle → world angle
}

/// What a drag frame wants to set on the app's transform state.
pub enum GizmoUpdate {
    /// New XY translation + a temporary Z lift (Move-Z only; settles to 0 on release).
    Translate(Vector2<f32>, f32),
    Rotate(UnitQuaternion<f32>),
    Scale(Vector3<f32>),
}

/// Per-frame projection context for the gizmo.
pub struct GizmoCtx<'a> {
    pub mvp: &'a [f32; 16],
    pub rect: Rect,
    /// World anchor: transformed model bbox center.
    pub anchor: [f32; 3],
    /// Arrow/handle length in world mm.
    pub handle_len: f32,
}

const AXIS_COLORS: [Color32; 3] = [
    Color32::from_rgb(225, 90, 90), // X
    Color32::from_rgb(90, 200, 90), // Y
    Color32::from_rgb(90, 140, 225), // Z
];
const UNIFORM_COLOR: Color32 = Color32::from_rgb(190, 190, 200);
const HIT_THRESHOLD_PX: f32 = 8.0;
const HANDLE_HALF_PX: f32 = 5.0; // scale handle square half-size
const RING_SEGMENTS: usize = 64;
/// Axes whose screen footprint is shorter than this are pointing at the
/// camera — undraggable, so they are skipped by hit-testing.
const MIN_AXIS_PX: f32 = 12.0;

// ---------------------------------------------------------------------------
// Projection
// ---------------------------------------------------------------------------

/// World point → egui screen position within `rect` (points), or None if w≈0.
/// `mvp` is the raw column-major output of `build_mvp` (GL-style NDC; the
/// wgpu z remap only affects depth, not screen XY).
pub fn project(mvp: &[f32; 16], rect: Rect, p: [f32; 3]) -> Option<Pos2> {
    let x = mvp[0] * p[0] + mvp[4] * p[1] + mvp[8] * p[2] + mvp[12];
    let y = mvp[1] * p[0] + mvp[5] * p[1] + mvp[9] * p[2] + mvp[13];
    let w = mvp[3] * p[0] + mvp[7] * p[1] + mvp[11] * p[2] + mvp[15];
    if w.abs() < 1e-9 {
        return None;
    }
    let (ndc_x, ndc_y) = (x / w, y / w);
    Some(
        egui::pos2(
            rect.left() + (ndc_x + 1.0) * 0.5 * rect.width(),
            rect.top() + (1.0 - ndc_y) * 0.5 * rect.height()
        )
    )
}

pub fn axis_unit(axis: GizmoAxis) -> Vector3<f32> {
    match axis {
        GizmoAxis::X => Vector3::x(),
        GizmoAxis::Y => Vector3::y(),
        GizmoAxis::Z => Vector3::z(),
        GizmoAxis::Uniform => Vector3::zeros(),
    }
}

fn axis_color(axis: GizmoAxis) -> Color32 {
    match axis {
        GizmoAxis::X => AXIS_COLORS[0],
        GizmoAxis::Y => AXIS_COLORS[1],
        GizmoAxis::Z => AXIS_COLORS[2],
        GizmoAxis::Uniform => UNIFORM_COLOR,
    }
}

fn brighten(c: Color32) -> Color32 {
    Color32::from_rgb(
        c.r().saturating_add(40),
        c.g().saturating_add(40),
        c.b().saturating_add(40)
    )
}

/// Screen direction + px-per-mm scale of a world axis at the anchor.
/// None if the axis is degenerate on screen (pointing at the camera).
pub fn axis_screen(ctx: &GizmoCtx, axis: GizmoAxis) -> Option<(Vec2, f32)> {
    let a = ctx.anchor;
    let u = axis_unit(axis);
    let p0 = project(ctx.mvp, ctx.rect, a)?;
    let p1 = project(ctx.mvp, ctx.rect, [a[0] + u.x, a[1] + u.y, a[2] + u.z])?;
    let d = p1 - p0;
    let px_per_mm = d.length();
    if px_per_mm * ctx.handle_len < MIN_AXIS_PX {
        return None;
    }
    Some((d / px_per_mm, px_per_mm))
}

// ---------------------------------------------------------------------------
// Screen geometry per mode
// ---------------------------------------------------------------------------

/// Arrow segment (Move) or handle stem (Scale) endpoints on screen.
fn axis_segment(ctx: &GizmoCtx, axis: GizmoAxis, len_frac: f32) -> Option<(Pos2, Pos2)> {
    let a = ctx.anchor;
    let u = axis_unit(axis) * ctx.handle_len * len_frac;
    let p0 = project(ctx.mvp, ctx.rect, a)?;
    let p1 = project(ctx.mvp, ctx.rect, [a[0] + u.x, a[1] + u.y, a[2] + u.z])?;
    if (p1 - p0).length() < MIN_AXIS_PX {
        return None;
    }
    Some((p0, p1))
}

/// Orthonormal basis (u, v) perpendicular to a ring's axis.
fn ring_basis(axis: GizmoAxis) -> (Vector3<f32>, Vector3<f32>) {
    match axis {
        GizmoAxis::X => (Vector3::y(), Vector3::z()),
        GizmoAxis::Y => (Vector3::z(), Vector3::x()),
        _ => (Vector3::x(), Vector3::y()),
    }
}

/// Projected points of a rotation ring (world circle around `axis`).
fn ring_points(ctx: &GizmoCtx, axis: GizmoAxis) -> Option<Vec<Pos2>> {
    let (u, v) = ring_basis(axis);
    let a = ctx.anchor;
    let r = ctx.handle_len;
    let mut pts = Vec::with_capacity(RING_SEGMENTS);
    for i in 0..RING_SEGMENTS {
        let t = ((i as f32) / (RING_SEGMENTS as f32)) * std::f32::consts::TAU;
        let w = u * (r * t.cos()) + v * (r * t.sin());
        pts.push(project(ctx.mvp, ctx.rect, [a[0] + w.x, a[1] + w.y, a[2] + w.z])?);
    }
    Some(pts)
}

/// +1 when increasing world angle around `axis` appears counter-clockwise on
/// screen (in egui's y-down coords), -1 otherwise. Maps screen-angle deltas
/// back to world-angle deltas during ring drags.
pub fn ring_winding(ctx: &GizmoCtx, axis: GizmoAxis) -> f32 {
    let (u, v) = ring_basis(axis);
    let a = ctx.anchor;
    let r = ctx.handle_len;
    let at = |t: f32| {
        let w = u * (r * t.cos()) + v * (r * t.sin());
        project(ctx.mvp, ctx.rect, [a[0] + w.x, a[1] + w.y, a[2] + w.z])
    };
    let (Some(c), Some(p0), Some(p1)) = (project(ctx.mvp, ctx.rect, a), at(0.0), at(0.2)) else {
        return 1.0;
    };
    let d0 = p0 - c;
    let d1 = p1 - c;
    let cross = d0.x * d1.y - d0.y * d1.x;
    if cross >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

fn dist_to_segment(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let len2 = ab.length_sq();
    if len2 < 1e-9 {
        return (p - a).length();
    }
    let t = (((p - a).dot(ab)) / len2).clamp(0.0, 1.0);
    (p - (a + ab * t)).length()
}

fn dist_to_polyline(p: Pos2, pts: &[Pos2]) -> f32 {
    let mut best = f32::INFINITY;
    for i in 0..pts.len() {
        let a = pts[i];
        let b = pts[(i + 1) % pts.len()];
        best = best.min(dist_to_segment(p, a, b));
    }
    best
}

// ---------------------------------------------------------------------------
// Hit-testing
// ---------------------------------------------------------------------------

/// Which handle (if any) is under `cursor` for the given mode.
pub fn hit_test(ctx: &GizmoCtx, mode: GizmoMode, cursor: Pos2) -> Option<GizmoAxis> {
    let axes = [GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z];
    match mode {
        GizmoMode::Move => {
            let mut best: Option<(f32, GizmoAxis)> = None;
            for axis in axes {
                let Some((a, b)) = axis_segment(ctx, axis, 1.0) else {
                    continue;
                };
                let d = dist_to_segment(cursor, a, b);
                if d < HIT_THRESHOLD_PX && best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, axis));
                }
            }
            best.map(|(_, axis)| axis)
        }
        GizmoMode::Rotate => {
            let mut best: Option<(f32, GizmoAxis)> = None;
            for axis in axes {
                let Some(pts) = ring_points(ctx, axis) else {
                    continue;
                };
                let d = dist_to_polyline(cursor, &pts);
                if d < HIT_THRESHOLD_PX && best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, axis));
                }
            }
            best.map(|(_, axis)| axis)
        }
        GizmoMode::Scale => {
            // Center (uniform) handle wins first.
            if let Some(c) = project(ctx.mvp, ctx.rect, ctx.anchor) {
                if (cursor - c).length() <= HANDLE_HALF_PX * 2.0 {
                    return Some(GizmoAxis::Uniform);
                }
            }
            let mut best: Option<(f32, GizmoAxis)> = None;
            for axis in axes {
                let Some((a, b)) = axis_segment(ctx, axis, 0.8) else {
                    continue;
                };
                // Tip square first, then the stem.
                let d = ((cursor - b).length() - HANDLE_HALF_PX).max(0.0).min(
                    dist_to_segment(cursor, a, b)
                );
                if d < HIT_THRESHOLD_PX && best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, axis));
                }
            }
            best.map(|(_, axis)| axis)
        }
    }
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

/// Draw the gizmo for `mode`; `hot` (hovered or dragged handle) is highlighted.
pub fn draw(painter: &Painter, ctx: &GizmoCtx, mode: GizmoMode, hot: Option<GizmoAxis>) {
    let axes = [GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z];
    let stroke_for = |axis: GizmoAxis| {
        let is_hot = hot == Some(axis);
        let color = if is_hot { brighten(axis_color(axis)) } else { axis_color(axis) };
        Stroke::new(if is_hot { 3.5_f32 } else { 2.0_f32 }, color)
    };

    match mode {
        GizmoMode::Move => {
            for axis in axes {
                let Some((a, b)) = axis_segment(ctx, axis, 1.0) else {
                    continue;
                };
                let stroke = stroke_for(axis);
                let dir = (b - a).normalized();
                let head_base = b - dir * 12.0;
                painter.line_segment([a, head_base], stroke);
                let perp = dir.rot90() * 5.0;
                painter.add(
                    Shape::convex_polygon(
                        vec![b, head_base + perp, head_base - perp],
                        stroke.color,
                        Stroke::NONE
                    )
                );
            }
        }
        GizmoMode::Rotate => {
            for axis in axes {
                let Some(pts) = ring_points(ctx, axis) else {
                    continue;
                };
                painter.add(Shape::closed_line(pts, stroke_for(axis)));
            }
        }
        GizmoMode::Scale => {
            for axis in axes {
                let Some((a, b)) = axis_segment(ctx, axis, 0.8) else {
                    continue;
                };
                let stroke = stroke_for(axis);
                painter.line_segment([a, b], stroke);
                painter.rect_filled(
                    Rect::from_center_size(b, egui::vec2(HANDLE_HALF_PX * 2.0, HANDLE_HALF_PX * 2.0)),
                    1.0,
                    stroke.color
                );
            }
            if let Some(c) = project(ctx.mvp, ctx.rect, ctx.anchor) {
                let is_hot = hot == Some(GizmoAxis::Uniform);
                let color = if is_hot { brighten(UNIFORM_COLOR) } else { UNIFORM_COLOR };
                let half = if is_hot { HANDLE_HALF_PX + 2.0 } else { HANDLE_HALF_PX + 1.0 };
                painter.rect_filled(
                    Rect::from_center_size(c, egui::vec2(half * 2.0, half * 2.0)),
                    1.0,
                    color
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Drag math
// ---------------------------------------------------------------------------

/// Build the drag snapshot when the pointer lands on `axis`.
pub fn begin_drag(
    ctx: &GizmoCtx,
    mode: GizmoMode,
    axis: GizmoAxis,
    cursor: Pos2,
    start_translation: Vector2<f32>,
    start_rotation: UnitQuaternion<f32>,
    start_scale: Vector3<f32>
) -> Option<GizmoDrag> {
    let center_px = project(ctx.mvp, ctx.rect, ctx.anchor)?;
    let (axis_screen_dir, px_per_mm) = if axis == GizmoAxis::Uniform {
        (Vec2::ZERO, 1.0)
    } else {
        axis_screen(ctx, axis)?
    };
    Some(GizmoDrag {
        mode,
        axis,
        start_cursor: cursor,
        start_translation,
        start_rotation,
        start_scale,
        center_px,
        axis_screen_dir,
        px_per_mm,
        start_angle: (cursor - center_px).angle(),
        winding: ring_winding(ctx, axis),
    })
}

/// Recompute the transform for the current cursor position. `shift` enables
/// snapping (1 mm translations, 15° rotations).
pub fn apply_drag(drag: &GizmoDrag, cursor: Pos2, shift: bool) -> GizmoUpdate {
    match drag.mode {
        GizmoMode::Move => {
            let mut delta_mm = (cursor - drag.start_cursor).dot(drag.axis_screen_dir)
                / drag.px_per_mm;
            if shift {
                delta_mm = delta_mm.round();
            }
            match drag.axis {
                GizmoAxis::X =>
                    GizmoUpdate::Translate(
                        drag.start_translation + Vector2::new(delta_mm, 0.0),
                        0.0
                    ),
                GizmoAxis::Y =>
                    GizmoUpdate::Translate(
                        drag.start_translation + Vector2::new(0.0, delta_mm),
                        0.0
                    ),
                // Z: temporary lift only — the snap-to-bed rule settles it on release.
                _ => GizmoUpdate::Translate(drag.start_translation, delta_mm.max(0.0)),
            }
        }
        GizmoMode::Rotate => {
            let angle = (cursor - drag.center_px).angle();
            let mut delta = angle - drag.start_angle;
            // Wrap to (-π, π] so crossing the atan2 seam doesn't jump.
            while delta > std::f32::consts::PI {
                delta -= std::f32::consts::TAU;
            }
            while delta <= -std::f32::consts::PI {
                delta += std::f32::consts::TAU;
            }
            let mut world_delta = drag.winding * delta;
            if shift {
                let step = (15.0f32).to_radians();
                world_delta = (world_delta / step).round() * step;
            }
            let world_axis = nalgebra::Unit::new_normalize(axis_unit(drag.axis));
            GizmoUpdate::Rotate(
                UnitQuaternion::from_axis_angle(&world_axis, world_delta) * drag.start_rotation
            )
        }
        GizmoMode::Scale => {
            let start_r = (drag.start_cursor - drag.center_px).length().max(1e-3);
            let factor = (cursor - drag.center_px).length() / start_r;
            let clamp = |v: f32| v.clamp(0.01, 100.0);
            let mut scale = drag.start_scale;
            match drag.axis {
                GizmoAxis::X => {
                    scale.x = clamp(scale.x * factor);
                }
                GizmoAxis::Y => {
                    scale.y = clamp(scale.y * factor);
                }
                GizmoAxis::Z => {
                    scale.z = clamp(scale.z * factor);
                }
                GizmoAxis::Uniform => {
                    scale.x = clamp(scale.x * factor);
                    scale.y = clamp(scale.y * factor);
                    scale.z = clamp(scale.z * factor);
                }
            }
            GizmoUpdate::Scale(scale)
        }
    }
}

/// Human-readable readout for the floating label near the cursor.
pub fn drag_readout(drag: &GizmoDrag, cursor: Pos2) -> String {
    match apply_drag(drag, cursor, false) {
        GizmoUpdate::Translate(t, lift) => {
            if drag.axis == GizmoAxis::Z {
                format!("+{lift:.1} mm")
            } else {
                let d = t - drag.start_translation;
                format!("{:+.1} mm", if drag.axis == GizmoAxis::X { d.x } else { d.y })
            }
        }
        GizmoUpdate::Rotate(r) => {
            let delta = drag.start_rotation.angle_to(&r).to_degrees();
            format!("{delta:.1}°")
        }
        GizmoUpdate::Scale(s) => {
            let f = match drag.axis {
                GizmoAxis::X => s.x / drag.start_scale.x.max(1e-6),
                GizmoAxis::Y => s.y / drag.start_scale.y.max(1e-6),
                _ => s.z / drag.start_scale.z.max(1e-6),
            };
            format!("×{f:.2}")
        }
    }
}
