use nalgebra::Point2;

use crate::slicer::{Contour, Layer, SliceResult};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for perimeter generation.
#[derive(Debug, Clone)]
pub struct PerimeterConfig {
    /// Nozzle diameter in mm (e.g. 0.4).
    pub nozzle_width: f32,
    /// Number of perimeter walls to generate (e.g. 3).
    pub perimeter_count: usize,
}

/// A single closed toolpath loop the nozzle follows.
#[derive(Debug, Clone)]
pub struct Perimeter {
    pub points: Vec<Point2<f32>>,
}

/// All perimeters derived from one shape (outer contour + its holes).
#[derive(Debug, Clone)]
pub struct PerimeterSet {
    /// Ordered from outermost (index 0) to innermost.
    /// Each level may contain multiple disjoint loops (if geometry pinches off).
    pub perimeters: Vec<Vec<Perimeter>>,
    /// The innermost offset boundary — defines the region available for infill.
    /// Empty if the geometry shrank to nothing.
    pub infill_boundary: Vec<Contour>,
}

/// A fully processed layer with perimeter toolpaths.
#[derive(Debug, Clone)]
pub struct ToolpathLayer {
    pub z: f32,
    pub perimeter_sets: Vec<PerimeterSet>,
}

/// Result of generating toolpaths for all layers.
#[derive(Debug)]
pub struct ToolpathResult {
    pub layers: Vec<ToolpathLayer>,
}

// ---------------------------------------------------------------------------
// Conversion helpers: nalgebra Point2<f32> <-> i_overlay [f32; 2]
// ---------------------------------------------------------------------------

fn to_overlay(p: &Point2<f32>) -> [f32; 2] {
    [p.x, p.y]
}

fn from_overlay(p: &[f32; 2]) -> Point2<f32> {
    Point2::new(p[0], p[1])
}

fn contour_to_overlay(contour: &Contour) -> Vec<[f32; 2]> {
    contour.points.iter().map(to_overlay).collect()
}

fn overlay_to_points(ring: &[[f32; 2]]) -> Vec<Point2<f32>> {
    ring.iter().map(from_overlay).collect()
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/// Signed area of a polygon. Positive = CCW (outer), negative = CW (hole).
fn signed_area(points: &[Point2<f32>]) -> f32 {
    let n = points.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0f32;
    for i in 0..n {
        let j = (i + 1) % n;
        area += points[i].x * points[j].y;
        area -= points[j].x * points[i].y;
    }
    area / 2.0
}

/// Check if a point is inside a polygon using the ray casting algorithm.
fn point_in_polygon(point: &Point2<f32>, polygon: &[Point2<f32>]) -> bool {
    let mut inside = false;
    let n = polygon.len();
    let mut j = n - 1;
    for i in 0..n {
        let pi = &polygon[i];
        let pj = &polygon[j];
        if ((pi.y > point.y) != (pj.y > point.y))
            && (point.x < (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x)
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

// ---------------------------------------------------------------------------
// Contour classification: group flat contours into shapes (outer + holes)
// ---------------------------------------------------------------------------

/// An i_overlay "shape": first ring is the outer boundary (CCW),
/// remaining rings are holes (CW).
type OverlayShape = Vec<Vec<[f32; 2]>>;

/// Group a layer's flat contours into i_overlay shapes.
///
/// Uses containment depth to classify contours: a contour nested inside an
/// even number of other contours is an outer boundary; odd = hole. This is
/// robust against the slicer producing inconsistent winding orders.
fn classify_contours(contours: &[Contour]) -> Vec<OverlayShape> {
    if contours.is_empty() {
        return Vec::new();
    }

    let n = contours.len();

    // For each contour, count how many other contours contain its first point.
    // Even depth = outer boundary; odd depth = hole.
    let mut depth = vec![0usize; n];
    for i in 0..n {
        if let Some(test_pt) = contours[i].points.first() {
            for j in 0..n {
                if i != j && point_in_polygon(test_pt, &contours[j].points) {
                    depth[i] += 1;
                }
            }
        }
    }

    let mut outers: Vec<(usize, f32)> = Vec::new(); // (index, abs_area)
    let mut holes: Vec<(usize, f32)> = Vec::new();  // (index, signed_area)

    for i in 0..n {
        let area = signed_area(&contours[i].points);
        if area.abs() < 1e-10 {
            continue; // degenerate
        }
        if depth[i] % 2 == 0 {
            outers.push((i, area.abs()));
        } else {
            holes.push((i, area));
        }
    }

    // Build shapes: each outer gets a vec, then we assign holes
    let mut shapes: Vec<OverlayShape> = Vec::new();
    let mut outer_indices: Vec<(usize, f32)> = Vec::new(); // (contour_idx, abs_area)

    for &(idx, abs_area) in &outers {
        let area = signed_area(&contours[idx].points);
        let mut ring = contour_to_overlay(&contours[idx]);
        // Ensure CCW winding for outer (positive area)
        if area < 0.0 {
            ring.reverse();
        }
        shapes.push(vec![ring]);
        outer_indices.push((idx, abs_area));
    }

    // Assign each hole to the smallest outer that contains it
    for &(hole_idx, hole_area) in &holes {
        let hole_contour = &contours[hole_idx];
        if let Some(test_point) = hole_contour.points.first() {
            let mut best_outer: Option<usize> = None;
            let mut best_area = f32::INFINITY;

            for (shape_idx, &(outer_idx, outer_abs_area)) in outer_indices.iter().enumerate() {
                if outer_abs_area < best_area
                    && point_in_polygon(test_point, &contours[outer_idx].points)
                {
                    best_outer = Some(shape_idx);
                    best_area = outer_abs_area;
                }
            }

            if let Some(shape_idx) = best_outer {
                let mut ring = contour_to_overlay(hole_contour);
                // Ensure CW winding for hole (negative area)
                if hole_area > 0.0 {
                    ring.reverse();
                }
                shapes[shape_idx].push(ring);
            }
        }
    }

    shapes
}

// ---------------------------------------------------------------------------
// Core offset logic
// ---------------------------------------------------------------------------

use i_overlay::mesh::outline::offset::OutlineOffset;
use i_overlay::mesh::style::{LineJoin, OutlineStyle};

/// Generate perimeter toolpaths for a single layer.
pub fn generate_perimeters(layer: &Layer, config: &PerimeterConfig) -> ToolpathLayer {
    let shapes = classify_contours(&layer.contours);
    let mut perimeter_sets = Vec::new();

    for shape in &shapes {
        let mut all_perimeters: Vec<Vec<Perimeter>> = Vec::new();
        // Wrap in an outer vec to form "Shapes" (Vec<Shape>)
        let mut current_shapes: Vec<OverlayShape> = vec![shape.clone()];

        for i in 0..config.perimeter_count {
            // First perimeter: inset by nozzle_width/2 so the outer edge of
            // the extruded line aligns with the original contour.
            // Subsequent perimeters: inset by a full nozzle_width.
            let inset = if i == 0 {
                config.nozzle_width / 2.0
            } else {
                config.nozzle_width
            };

            // Negative offset = shrink inward
            let style = OutlineStyle::new(-inset)
                .line_join(LineJoin::Miter(2.0));

            let mut level_perimeters = Vec::new();
            let mut next_shapes = Vec::new();

            for s in &current_shapes {
                let offset_result: Vec<OverlayShape> = vec![s.clone()].outline(&style);

                for offset_shape in &offset_result {
                    // Each ring in the offset shape is a perimeter path
                    for ring in offset_shape {
                        if ring.len() >= 3 {
                            level_perimeters.push(Perimeter {
                                points: overlay_to_points(ring),
                            });
                        }
                    }
                    next_shapes.push(offset_shape.clone());
                }
            }

            if level_perimeters.is_empty() {
                break; // Geometry shrank to nothing
            }

            all_perimeters.push(level_perimeters);
            current_shapes = next_shapes;
        }

        // The last valid offset result defines the infill boundary
        let infill_boundary = current_shapes
            .iter()
            .flat_map(|s| {
                s.iter().filter(|ring| ring.len() >= 3).map(|ring| Contour {
                    points: overlay_to_points(ring),
                })
            })
            .collect();

        perimeter_sets.push(PerimeterSet {
            perimeters: all_perimeters,
            infill_boundary,
        });
    }

    ToolpathLayer {
        z: layer.z,
        perimeter_sets,
    }
}

/// Generate toolpaths for all layers (parallelized across cores).
pub fn generate_toolpaths(
    slice_result: &SliceResult,
    config: &PerimeterConfig,
) -> ToolpathResult {
    use rayon::prelude::*;

    let layers = slice_result
        .layers
        .par_iter()
        .map(|layer| generate_perimeters(layer, config))
        .collect();
    ToolpathResult { layers }
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
    fn signed_area_ccw_is_positive() {
        // Unit square, CCW
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        assert!(signed_area(&pts) > 0.0);
    }

    #[test]
    fn signed_area_cw_is_negative() {
        // Unit square, CW
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(0.0, 1.0),
            Point2::new(1.0, 1.0),
            Point2::new(1.0, 0.0),
        ];
        assert!(signed_area(&pts) < 0.0);
    }

    #[test]
    fn point_in_polygon_works() {
        let square = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        assert!(point_in_polygon(&Point2::new(0.5, 0.5), &square));
        assert!(!point_in_polygon(&Point2::new(2.0, 2.0), &square));
    }

    #[test]
    fn debug_outline_api() {
        // 100x100 square, try various offset approaches
        let shape: Vec<Vec<[f32; 2]>> = vec![vec![
            [0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0],
        ]];
        let shapes: Vec<Vec<Vec<[f32; 2]>>> = vec![shape];

        // Approach: negative offset to shrink
        let style = OutlineStyle::new(-10.0)
            .line_join(LineJoin::Miter(2.0));

        let result: Vec<Vec<Vec<[f32; 2]>>> = shapes.outline(&style);
        println!("Negative offset result:");
        for (i, shape) in result.iter().enumerate() {
            println!("  Shape {i}: {} rings", shape.len());
            for (j, ring) in shape.iter().enumerate() {
                println!("    Ring {j}: {} pts: {:?}", ring.len(), ring);
            }
        }

        assert!(!result.is_empty(), "Expected non-empty result");
        // The inset square should be 10..90
        let ring = &result[0][0];
        for p in ring {
            assert!(p[0] >= 9.0 && p[0] <= 91.0, "x={} not in expected inset range", p[0]);
            assert!(p[1] >= 9.0 && p[1] <= 91.0, "y={} not in expected inset range", p[1]);
        }
    }

    #[test]
    fn cube_single_perimeter() {
        let data = std::fs::read(stl_path("cube.stl")).unwrap();
        let mesh = stl::load_stl(&data).unwrap();
        let result = crate::slicer::slice_mesh(&mesh, 0.2);
        let layer = &result.layers[2]; // middle layer

        let config = PerimeterConfig {
            nozzle_width: 0.2,
            perimeter_count: 1,
        };
        let toolpath = generate_perimeters(layer, &config);

        assert_eq!(toolpath.perimeter_sets.len(), 1, "Cube should produce 1 shape");
        assert_eq!(
            toolpath.perimeter_sets[0].perimeters.len(),
            1,
            "Should have 1 perimeter level"
        );
        assert!(
            !toolpath.perimeter_sets[0].perimeters[0].is_empty(),
            "Perimeter level should have at least 1 loop"
        );
    }
}
