//! Arc fitting: collapse runs of short line segments that approximate a
//! circular arc into single `G2`/`G3` moves.

use nalgebra::Point2;

#[derive(Debug, Clone, PartialEq)]
pub enum PathPrimitive {
    Line {
        to: Point2<f32>,
    },
    /// Circular arc ending at `to`, centered at `center`.
    /// `cw == true` = clockwise (`G2`); `cw == false` = counter-clockwise (`G3`).
    Arc {
        to: Point2<f32>,
        center: Point2<f32>,
        cw: bool,
    },
}

/// Center and radius of the circle through three points.
/// Returns `None` when the points are close to collinear
pub fn circle_from_three_points(
    a: &Point2<f32>,
    b: &Point2<f32>,
    c: &Point2<f32>
) -> Option<(Point2<f32>, f32)> {
    let (ax, ay) = (a.x as f64, a.y as f64);
    let (bx, by) = (b.x as f64, b.y as f64);
    let (cx, cy) = (c.x as f64, c.y as f64);

    // d = 2 * the signed area determinant of the triangle.
    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));

    if d.abs() < 1e-9 {
        return None; // collinear
    }

    // Precompute x²+y² for each vertex.
    let ax2ay2 = ax * ax + ay * ay;
    let bx2by2 = bx * bx + by * by;
    let cx2cy2 = cx * cx + cy * cy;

    // circumcenter X coordinate. Closed-form solution for the intersection of the
    // perpendicular bisectors of the triangle's sides.
    let ux = (ax2ay2 * (by - cy) + bx2by2 * (cy - ay) + cx2cy2 * (ay - by)) / d;

    // circumcenter Y coordinate.
    let uy = (ax2ay2 * (cx - bx) + bx2by2 * (ax - cx) + cx2cy2 * (bx - ax)) / d;

    // Radius = distance from the computed center to any of the three points.
    let r = (ax - ux).hypot(ay - uy);
    Some((Point2::new(ux as f32, uy as f32), r as f32))
}

fn radial_error(p: &Point2<f32>, center: &Point2<f32>, radius: f32) -> f32 {
    let dist = ((p.x - center.x).powi(2) + (p.y - center.y).powi(2)).sqrt();
    (dist - radius).abs()
}

/// How far the midpoint of chord `a`-`b` sits from the circle. A chord spanning
/// a large arc bows inward (large sagitta) even when both endpoints are exactly
/// on the circle — this is what distinguishes a real arc from a sharp corner
/// whose vertices happen to be concyclic (e.g. the corners of a square).
fn midpoint_error(a: &Point2<f32>, b: &Point2<f32>, center: &Point2<f32>, radius: f32) -> f32 {
    let mx = (a.x + b.x) / 2.0;
    let my = (a.y + b.y) / 2.0;
    let dist = ((mx - center.x).powi(2) + (my - center.y).powi(2)).sqrt();
    (radius - dist).abs()
}

/// Signed angle (radians, in (-PI, PI]) swept going from spoke `a` to spoke `b`
/// around `center`. Positive is counter-clockwise. Accumulating this across a
/// run gives the *directed* sweep: its sign is the arc direction (no separate
/// winding guess) and its magnitude is the true swept angle, even past 180 deg —
/// which an unsigned `acos` cannot represent (it folds 348 deg down to 12 deg).
fn signed_step(a: &Point2<f32>, b: &Point2<f32>, center: &Point2<f32>) -> f32 {
    let (ax, ay) = (a.x - center.x, a.y - center.y);
    let (bx, by) = (b.x - center.x, b.y - center.y);
    let cross = ax * by - ay * bx;
    let dot = ax * bx + ay * by;
    cross.atan2(dot)
}

/// Below this radius an "arc" is sub-nozzle noise; emit straight segments.
const MIN_ARC_RADIUS: f32 = 0.5;
/// Above this radius the run is effectively straight; a line is safer/smaller.
const MAX_ARC_RADIUS: f32 = 1000.0;
/// Don't bother emitting an arc that sweeps less than this (avoids flooding the
/// file with marginal micro-arcs that stress downstream slicers).
const MIN_ARC_SWEEP: f32 = 0.20; // ~11 degrees
/// Cap a single arc's sweep, splitting longer curves into multiple arcs. Keeps
/// every arc clear of the ~180-degree antipodal ambiguity that makes slicers
/// mis-interpret the direction and tessellate near-full circles.
const MAX_ARC_SWEEP: f32 = 2.0; // ~114 degrees

/// Fit arcs greedily over a polyline, returning one primitive per emitted move.
/// `tolerance` is the max perpendicular deviation (mm) a point may have from a
/// candidate arc's circle and still be folded into it.
pub fn fit_arcs(points: &[Point2<f32>], tolerance: f32) -> Vec<PathPrimitive> {
    let mut result = Vec::new();
    let mut p = 0;
    while p + 2 < points.len() {
        let circle = circle_from_three_points(&points[p], &points[p + 1], &points[p + 2]);

        // The seed must be a physically sensible circle whose *edges* (not just
        // vertices) hug it — otherwise sharp corners with concyclic vertices, or
        // sub-nozzle / quasi-straight noise, would masquerade as arcs.
        let seed_fits = circle.is_some_and(|(center, radius)| {
            radius >= MIN_ARC_RADIUS &&
                radius <= MAX_ARC_RADIUS &&
                midpoint_error(&points[p], &points[p + 1], &center, radius) <= tolerance &&
                midpoint_error(&points[p + 1], &points[p + 2], &center, radius) <= tolerance
        });

        if let (true, Some((center, radius))) = (seed_fits, circle) {
            // Grow while each new point lands on the circle, the new segment
            // still hugs it, and the *directed* sweep stays under the cap.
            let mut end = p + 2;
            let mut sweep =
                signed_step(&points[p], &points[p + 1], &center) +
                signed_step(&points[p + 1], &points[p + 2], &center);
            while end + 1 < points.len() {
                let next = signed_step(&points[end], &points[end + 1], &center);
                if
                    (sweep + next).abs() > MAX_ARC_SWEEP ||
                    radial_error(&points[end + 1], &center, radius) > tolerance ||
                    midpoint_error(&points[end], &points[end + 1], &center, radius) > tolerance
                {
                    break;
                }
                sweep += next;
                end += 1;
            }
            if end - p >= 3 && sweep.abs() >= MIN_ARC_SWEEP {
                // Negative directed sweep is clockwise => G2.
                let cw = sweep < 0.0;
                result.push(PathPrimitive::Arc { to: points[end], center, cw });
                p = end; // the run's end seeds the next run
                continue;
            }
        }

        // No qualifying arc here: emit one straight segment and step on.
        result.push(PathPrimitive::Line { to: points[p + 1] });
        p += 1;
    }
    // Flush any trailing segment(s) too short to seed a circle.
    while p + 1 < points.len() {
        result.push(PathPrimitive::Line { to: points[p + 1] });
        p += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f32, y: f32) -> Point2<f32> {
        Point2::new(x, y)
    }

    /// Sample `n` points along an arc of `center`/`r` from `a0` to `a1` degrees.
    fn arc_points(center: Point2<f32>, r: f32, a0: f32, a1: f32, n: usize) -> Vec<Point2<f32>> {
        (0..n)
            .map(|k| {
                let t = (a0 + ((a1 - a0) * (k as f32)) / ((n - 1) as f32)).to_radians();
                p(center.x + r * t.cos(), center.y + r * t.sin())
            })
            .collect()
    }

    #[test]
    fn circle_from_three_points_basic() {
        let (center, r) = circle_from_three_points(
            &p(1.0, 0.0),
            &p(0.0, 1.0),
            &p(-1.0, 0.0)
        ).expect("three non-collinear points define a circle");
        assert!(center.coords.norm() < 1e-4, "center should be the origin");
        assert!((r - 1.0).abs() < 1e-4, "radius should be 1");
    }

    #[test]
    fn circle_from_three_collinear_points_is_none() {
        assert!(circle_from_three_points(&p(0.0, 0.0), &p(1.0, 1.0), &p(2.0, 2.0)).is_none());
    }

    #[test]
    fn two_points_stay_a_single_line() {
        let out = fit_arcs(&[p(0.0, 0.0), p(3.0, 4.0)], 0.05);
        assert_eq!(out, vec![PathPrimitive::Line { to: p(3.0, 4.0) }]);
    }

    #[test]
    fn collinear_polyline_has_no_arcs() {
        let line = vec![p(0.0, 0.0), p(1.0, 0.0), p(2.0, 0.0), p(3.0, 0.0)];
        let out = fit_arcs(&line, 0.05);
        assert!(out.iter().all(|prim| matches!(prim, PathPrimitive::Line { .. })));
    }

    #[test]
    fn ccw_quarter_circle_is_one_g3_arc() {
        // Sweep 0deg -> 90deg around the origin: angle increases => CCW => G3.
        let pts = arc_points(p(0.0, 0.0), 10.0, 0.0, 90.0, 9);
        let out = fit_arcs(&pts, 0.05);
        assert_eq!(out.len(), 1, "the whole quarter circle should be one arc");
        match &out[0] {
            PathPrimitive::Arc { to, center, cw } => {
                assert!(!cw, "increasing angle is counter-clockwise (G3)");
                assert!(center.coords.norm() < 1e-2, "center near origin");
                assert!((to - p(0.0, 10.0)).norm() < 1e-2, "ends at (0, 10)");
            }
            other => panic!("expected a single arc, got {other:?}"),
        }
    }

    #[test]
    fn cw_quarter_circle_is_one_g2_arc() {
        // Sweep 90deg -> 0deg around the origin: angle decreases => CW => G2.
        let pts = arc_points(p(0.0, 0.0), 10.0, 90.0, 0.0, 9);
        let out = fit_arcs(&pts, 0.05);
        assert_eq!(out.len(), 1, "the whole quarter circle should be one arc");
        match &out[0] {
            PathPrimitive::Arc { to, center, cw } => {
                assert!(cw, "decreasing angle is clockwise (G2)");
                assert!(center.coords.norm() < 1e-2, "center near origin");
                assert!((to - p(10.0, 0.0)).norm() < 1e-2, "ends at (10, 0)");
            }
            other => panic!("expected a single arc, got {other:?}"),
        }
    }
}
