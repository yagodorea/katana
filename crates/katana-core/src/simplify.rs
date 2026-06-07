use nalgebra::Point2;

/// Perpendicular distance from a point to a line defined by `a` and `b`.
/// If `a ~= b`, falls back to the plain distance from `p` to `a`.
pub fn perpendicular_distance(p: &Point2<f32>, a: &Point2<f32>, b: &Point2<f32>) -> f32 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-9 {
        let ex = p.x - a.x;
        let ey = p.y - a.y;
        return (ex * ex + ey * ey).sqrt();
    }
    // |cross((b - a), (p - a))| / |b - a|
    ((p.x - a.x) * dy - (p.y - a.y) * dx).abs() / len
}

/// Simplify an open polyline using RDP recursive algo, keeping the first and last points fixed.
pub fn douglas_peucker(points: &[Point2<f32>], epsilon: f32) -> Vec<Point2<f32>> {
    // If there's only 2 points, return them
    if points.len() < 3 {
        return points.to_vec();
    }
    // get point farthest from line
    let mut max_dist: f32 = 0.0;
    let mut farthest: usize = 0;
    for i in 1..points.len() - 1 {
        let p = points[i];
        let dist = perpendicular_distance(&p, &points[0], &points[points.len() - 1]);
        if dist > max_dist {
            max_dist = dist;
            farthest = i;
        }
    }
    // Cull points in between if the farthest one is already less than epsilon
    if max_dist < epsilon {
        return Vec::from([points[0], points[points.len() - 1]]);
    }
    // Otherwise, call recursively on the 2 slices defined by the farthest point
    let mut seg1 = douglas_peucker(&points[0..=farthest], epsilon);
    let seg2 = douglas_peucker(&points[farthest..points.len()], epsilon);
    seg1.pop(); // remove shared point
    seg1.extend(seg2);
    seg1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f32, y: f32) -> Point2<f32> {
        Point2::new(x, y)
    }

    #[test]
    fn collinear_run_collapses_to_endpoints() {
        // Five points exactly on a straight line: only the ends should remain.
        let line = vec![p(0.0, 0.0), p(1.0, 0.0), p(2.0, 0.0), p(3.0, 0.0), p(4.0, 0.0)];
        let out = douglas_peucker(&line, 0.01);
        assert_eq!(out, vec![p(0.0, 0.0), p(4.0, 0.0)]);
    }

    #[test]
    fn significant_corner_is_kept() {
        // A clear corner well above tolerance must survive.
        let path = vec![p(0.0, 0.0), p(2.0, 2.0), p(4.0, 0.0)];
        let out = douglas_peucker(&path, 0.01);
        assert_eq!(out, path);
    }

    #[test]
    fn deviation_below_epsilon_is_dropped() {
        // The middle point bows out by only 0.005mm — under tolerance, so it goes.
        let path = vec![p(0.0, 0.0), p(2.0, 0.005), p(4.0, 0.0)];
        let out = douglas_peucker(&path, 0.01);
        assert_eq!(out, vec![p(0.0, 0.0), p(4.0, 0.0)]);
    }

    #[test]
    fn endpoints_always_survive() {
        let path = vec![p(0.0, 0.0), p(1.0, 0.0)];
        let out = douglas_peucker(&path, 1000.0);
        assert_eq!(out, path);
    }
    #[test]
    fn collinear_run_between_corners_is_pruned() {
        // (1,0) is redundant on the flat run; the corner at (2,0) must stay.
        let path = vec![p(0.0, 0.0), p(1.0, 0.0), p(2.0, 0.0), p(2.0, 2.0)];
        let out = douglas_peucker(&path, 0.01);
        assert_eq!(out, vec![p(0.0, 0.0), p(2.0, 0.0), p(2.0, 2.0)]);
    }
}
