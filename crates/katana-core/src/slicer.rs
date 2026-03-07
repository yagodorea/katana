use nalgebra::Point2;

use crate::mesh::Mesh;

/// Small offset to avoid slicing exactly on vertices/edges.
const EPSILON: f32 = 1e-4;

/// Pre-computed z-range for a triangle, used for sweep-line acceleration.
struct TriZRange {
    z_min: f32,
    z_max: f32,
    tri_idx: usize,
}

/// A closed polygon contour from slicing at a single Z height.
#[derive(Debug, Clone)]
pub struct Contour {
    pub points: Vec<Point2<f32>>,
}

/// All contours at a single Z height.
#[derive(Debug, Clone)]
pub struct Layer {
    pub z: f32,
    pub contours: Vec<Contour>,
    /// Minimum surface angle from horizontal (radians) among triangles
    /// intersecting this layer. 0 = horizontal overhang, π/2 = vertical wall.
    /// Used to determine extra perimeters on shallow slopes.
    pub min_surface_angle: f32,
}

/// Result of slicing a mesh into layers.
#[derive(Debug)]
pub struct SliceResult {
    pub layers: Vec<Layer>,
}

/// Slice a mesh into horizontal layers.
///
/// `layer_height` is the distance between slicing planes (e.g. 0.2 mm).
/// Returns one `Layer` per Z height, each containing closed contour polygons.
pub fn slice_mesh(mesh: &Mesh, layer_height: f32) -> SliceResult {
    use rayon::prelude::*;

    let (min, max) = mesh.bounding_box();
    let z_min = min.z + EPSILON;
    let z_max = max.z - EPSILON;

    // Build z-range index sorted by z_min for sweep-line acceleration
    let mut z_index: Vec<TriZRange> = mesh
        .triangles
        .iter()
        .enumerate()
        .map(|(i, tri)| {
            let zs = [tri.vertices[0].z, tri.vertices[1].z, tri.vertices[2].z];
            TriZRange {
                z_min: zs[0].min(zs[1]).min(zs[2]),
                z_max: zs[0].max(zs[1]).max(zs[2]),
                tri_idx: i,
            }
        })
        .collect();
    z_index.sort_unstable_by(|a, b| a.z_min.partial_cmp(&b.z_min).unwrap());

    // Collect Z heights first, then process in parallel
    let mut z_heights = Vec::new();
    let mut z = z_min + layer_height;
    while z < z_max {
        z_heights.push(z);
        z += layer_height;
    }

    let layers: Vec<Layer> = z_heights
        .par_iter()
        .map(|&z| {
            let (segments, min_surface_angle) = intersect_plane_indexed(mesh, z, &z_index);
            let contours = assemble_contours(segments);
            Layer { z, contours, min_surface_angle }
        })
        .collect();

    SliceResult { layers }
}

// ---------------------------------------------------------------------------
// Plane-triangle intersection
// ---------------------------------------------------------------------------

/// A 2D line segment (Z is implicit — it's the slice height).
#[derive(Debug, Clone)]
struct Segment {
    a: Point2<f32>,
    b: Point2<f32>,
}

/// Intersect mesh triangles with a horizontal plane at height `z`, using a
/// pre-sorted z-range index to skip triangles that cannot intersect.
///
/// Returns the intersection segments and the minimum surface angle from
/// horizontal (radians) among all intersecting triangles.
fn intersect_plane_indexed(mesh: &Mesh, z: f32, z_index: &[TriZRange]) -> (Vec<Segment>, f32) {
    let mut segments = Vec::new();
    let mut min_angle = std::f32::consts::FRAC_PI_2; // start at vertical (best case)

    // Binary-search to find the first triangle whose z_min > z — all triangles
    // before this point *may* intersect (their z_min <= z). We then filter by
    // z_max >= z within the loop.
    let upper = z_index.partition_point(|t| t.z_min <= z);

    for entry in &z_index[..upper] {
        if entry.z_max < z {
            continue; // z-range doesn't span this plane
        }

        let tri = &mesh.triangles[entry.tri_idx];
        let v = &tri.vertices;
        let d = [v[0].z - z, v[1].z - z, v[2].z - z];

        let above = d.iter().filter(|&&d| d > 0.0).count();
        let below = d.iter().filter(|&&d| d < 0.0).count();

        if above == 0 || below == 0 {
            continue;
        }

        // Surface angle from horizontal: the normal is perpendicular to the
        // surface, so if the surface makes angle θ with horizontal, the normal
        // makes angle θ with vertical. Thus θ = acos(|normal.z|).
        //   vertical wall: |normal.z| ≈ 0 → θ = π/2 (no extra perimeters)
        //   horizontal:    |normal.z| ≈ 1 → θ = 0   (max extra perimeters)
        let nz_abs = tri.normal.z.abs();
        let angle = nz_abs.clamp(0.0, 1.0).acos();
        if angle < min_angle {
            min_angle = angle;
        }

        let (lone, pair0, pair1) = if (d[0] > 0.0) != (d[1] > 0.0) && (d[0] > 0.0) != (d[2] > 0.0)
        {
            (0, 1, 2)
        } else if (d[1] > 0.0) != (d[0] > 0.0) && (d[1] > 0.0) != (d[2] > 0.0) {
            (1, 2, 0)
        } else {
            (2, 0, 1)
        };

        let p0 = edge_intersect_z(&v[lone], &v[pair0], z);
        let p1 = edge_intersect_z(&v[lone], &v[pair1], z);

        segments.push(Segment { a: p0, b: p1 });
    }

    (segments, min_angle)
}

/// Linearly interpolate along an edge to find where it crosses height `z`.
/// Returns the 2D (x, y) intersection point.
fn edge_intersect_z(
    a: &nalgebra::Point3<f32>,
    b: &nalgebra::Point3<f32>,
    z: f32,
) -> Point2<f32> {
    // what fraction of the way from a.z to b.z do I need to go from a to z?
    let t = (z - a.z) / (b.z - a.z);
    Point2::new(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y))
}

// ---------------------------------------------------------------------------
// Contour assembly
// ---------------------------------------------------------------------------

/// Quantize a coordinate to an integer key for HashMap lookups.
/// Matches the 1e-3 tolerance used by `points_close`.
fn quantize(v: f32) -> i64 {
    (v * 1000.0).round() as i64
}

fn quantize_point(p: &Point2<f32>) -> (i64, i64) {
    (quantize(p.x), quantize(p.y))
}

/// Connect line segments into closed polygon contours.
///
/// Uses a HashMap keyed by quantized endpoint coordinates for O(1) neighbor
/// lookup instead of O(s) linear scans. Checks a 3×3 neighborhood of cells
/// to handle points that straddle quantization bucket boundaries.
fn assemble_contours(segments: Vec<Segment>) -> Vec<Contour> {
    use std::collections::HashMap;

    if segments.is_empty() {
        return Vec::new();
    }

    let n = segments.len();

    // Build adjacency map: quantized point → list of (segment_index, is_endpoint_a)
    let mut map: HashMap<(i64, i64), Vec<(usize, bool)>> = HashMap::with_capacity(n * 2);
    for (i, seg) in segments.iter().enumerate() {
        map.entry(quantize_point(&seg.a))
            .or_default()
            .push((i, true));
        map.entry(quantize_point(&seg.b))
            .or_default()
            .push((i, false));
    }

    let mut used = vec![false; n];
    let mut contours = Vec::new();

    for start_idx in 0..n {
        if used[start_idx] {
            continue;
        }

        used[start_idx] = true;
        let mut contour_points = vec![segments[start_idx].a];
        let mut current_end = segments[start_idx].b;
        let chain_start = segments[start_idx].a;

        loop {
            if points_close(current_end, chain_start) {
                break;
            }

            let (qx, qy) = quantize_point(&current_end);
            let mut found = false;

            // Check the 3×3 neighborhood to handle bucket boundary straddling
            'outer: for dx in -1i64..=1 {
                for dy in -1i64..=1 {
                    let key = (qx + dx, qy + dy);
                    if let Some(candidates) = map.get(&key) {
                        for &(seg_idx, is_a) in candidates {
                            if used[seg_idx] {
                                continue;
                            }
                            let seg = &segments[seg_idx];
                            if is_a && points_close(seg.a, current_end) {
                                contour_points.push(seg.a);
                                current_end = seg.b;
                                used[seg_idx] = true;
                                found = true;
                                break 'outer;
                            } else if !is_a && points_close(seg.b, current_end) {
                                contour_points.push(seg.b);
                                current_end = seg.a;
                                used[seg_idx] = true;
                                found = true;
                                break 'outer;
                            }
                        }
                    }
                }
            }

            if !found {
                break;
            }
        }

        contours.push(Contour {
            points: contour_points,
        });
    }

    contours
}

/// Check if two points are close enough to be considered the same.
///
/// STL stores vertices per-triangle with no sharing, so the "same" vertex
/// in adjacent triangles may have slightly different f32 values. We need a
/// tolerance generous enough to bridge these gaps. 1e-3 distance (1e-6
/// squared) is well below any meaningful print resolution.
fn points_close(a: Point2<f32>, b: Point2<f32>) -> bool {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    dx * dx + dy * dy < 1e-6
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stl;
    use std::path::Path;

    fn stl_path(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../stls")
            .join(name)
    }

    #[test]
    fn slice_cube_produces_square_contours() {
        let data = std::fs::read(stl_path("cube.stl")).unwrap();
        let mesh = stl::load_stl(&data).unwrap();

        // cube.stl: bounding box (-0.5,-0.5,0) to (0.5,0.5,1)
        let result = slice_mesh(&mesh, 0.2);

        // With height 1.0 and layer_height 0.2, expect ~4 layers
        // (z = 0.2, 0.4, 0.6, 0.8 approximately, offset by epsilon)
        assert!(
            result.layers.len() >= 4,
            "Expected at least 4 layers, got {}",
            result.layers.len()
        );

        // Each layer of a cube should produce exactly 1 contour (a square).
        // The cube has 2 triangles per face, so each side of the square is
        // split into 2 segments → 8 points total.
        for layer in &result.layers {
            assert_eq!(
                layer.contours.len(),
                1,
                "Expected 1 contour at z={}, got {}",
                layer.z,
                layer.contours.len()
            );
            assert_eq!(
                layer.contours[0].points.len(),
                8,
                "Expected 8 points at z={}, got {}",
                layer.z,
                layer.contours[0].points.len()
            );
        }
    }

    #[test]
    fn slice_sphere_produces_closed_contours() {
        let data = std::fs::read(stl_path("sphere.stl")).unwrap();
        let mesh = stl::load_stl(&data).unwrap();

        // sphere.stl: bounding box z = 5.0 to 45.0
        let result = slice_mesh(&mesh, 1.0);

        assert!(!result.layers.is_empty(), "Expected some layers");

        // Check the middle layers (away from poles where tessellation is coarse)
        let mid_layers: Vec<_> = result
            .layers
            .iter()
            .filter(|l| l.z > 15.0 && l.z < 35.0)
            .collect();

        assert!(!mid_layers.is_empty(), "Expected mid-range layers");

        for layer in &mid_layers {
            assert_eq!(
                layer.contours.len(),
                1,
                "Expected 1 contour at z={}, got {}",
                layer.z,
                layer.contours.len()
            );
            assert!(
                layer.contours[0].points.len() >= 6,
                "Contour at z={} has only {} points, expected a well-formed polygon",
                layer.z,
                layer.contours[0].points.len()
            );
        }
    }
}
