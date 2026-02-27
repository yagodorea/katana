use nalgebra::Point2;

use crate::mesh::Mesh;

/// Small offset to avoid slicing exactly on vertices/edges.
const EPSILON: f32 = 1e-4;

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
            let segments = intersect_plane(mesh, z);
            let contours = assemble_contours(segments);
            Layer { z, contours }
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

/// Intersect all mesh triangles with a horizontal plane at height `z`.
// TODO: currently iterates every triangle for every layer (O(triangles × layers)).
// Optimize with a sorted sweep: precompute min_z/max_z per triangle, sort by
// min_z, then maintain an active window as layers advance. This reduces total
// work to O(triangles + layers).
fn intersect_plane(mesh: &Mesh, z: f32) -> Vec<Segment> {
    let mut segments = Vec::new();

    for tri in &mesh.triangles {
        let v = &tri.vertices;
        // Classify each vertex relative to the plane
        let d = [v[0].z - z, v[1].z - z, v[2].z - z];

        // Count vertices above and below
        let above = d.iter().filter(|&&d| d > 0.0).count();
        let below = d.iter().filter(|&&d| d < 0.0).count();

        // No intersection if all vertices are on the same side
        if above == 0 || below == 0 {
            continue;
        }

        // Find the vertex that's alone on its side of the plane.
        // With epsilon offset we won't have d == 0, so one vertex is alone
        // on one side and two are on the other.
        let (lone, pair0, pair1) = if (d[0] > 0.0) != (d[1] > 0.0) && (d[0] > 0.0) != (d[2] > 0.0)
        {
            // v[0] is alone
            (0, 1, 2)
        } else if (d[1] > 0.0) != (d[0] > 0.0) && (d[1] > 0.0) != (d[2] > 0.0) {
            // v[1] is alone
            (1, 2, 0)
        } else {
            // v[2] is alone
            (2, 0, 1)
        };

        // Interpolate along edges from the lone vertex to each of the pair
        let p0 = edge_intersect_z(&v[lone], &v[pair0], z);
        let p1 = edge_intersect_z(&v[lone], &v[pair1], z);

        segments.push(Segment { a: p0, b: p1 });
    }

    segments
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

/// Connect line segments into closed polygon contours.
///
/// For a watertight mesh, every segment endpoint is shared with exactly one
/// other segment. We follow the chain from segment to segment until we loop
/// back to the start.
fn assemble_contours(segments: Vec<Segment>) -> Vec<Contour> {
    if segments.is_empty() {
        return Vec::new();
    }

    // Build an adjacency structure: for each segment, store both endpoints.
    // We'll consume segments as we chain them together.
    let mut remaining: Vec<Option<Segment>> = segments.into_iter().map(Some).collect();
    let mut contours = Vec::new();

    loop {
        // Find the next unconsumed segment
        let start_idx = match remaining.iter().position(|s| s.is_some()) {
            Some(i) => i,
            None => break,
        };

        let start_seg = remaining[start_idx].take().unwrap();
        let mut contour_points = vec![start_seg.a];
        let mut current_end = start_seg.b;
        let chain_start = start_seg.a;

        // Follow the chain
        loop {
            // Are we back to the start?
            if points_close(current_end, chain_start) {
                break;
            }

            // Find a segment that connects to current_end
            let mut found = false;
            for slot in remaining.iter_mut() {
                if let Some(seg) = slot {
                    if points_close(seg.a, current_end) {
                        contour_points.push(seg.a);
                        current_end = seg.b;
                        *slot = None;
                        found = true;
                        break;
                    } else if points_close(seg.b, current_end) {
                        contour_points.push(seg.b);
                        current_end = seg.a;
                        *slot = None;
                        found = true;
                        break;
                    }
                }
            }

            if !found {
                // Open contour — mesh probably isn't watertight. Stop this chain.
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
