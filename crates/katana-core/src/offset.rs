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

/// A single open infill line segment.
#[derive(Debug, Clone)]
pub struct InfillLine {
    pub start: Point2<f32>,
    pub end: Point2<f32>,
}

/// Configuration for infill generation.
#[derive(Debug, Clone)]
pub struct InfillConfig {
    /// Infill density from 0.0 (empty) to 1.0 (solid).
    pub density: f32,
    /// Nozzle width in mm — determines line spacing at 100% density.
    pub nozzle_width: f32,
}

/// A fully processed layer with perimeter toolpaths and infill.
#[derive(Debug, Clone)]
pub struct ToolpathLayer {
    pub z: f32,
    pub layer_index: usize,
    pub perimeter_sets: Vec<PerimeterSet>,
    pub infill_lines: Vec<InfillLine>,
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

use i_overlay::core::fill_rule::FillRule;
use i_overlay::float::clip::FloatClip;
use i_overlay::mesh::outline::offset::OutlineOffset;
use i_overlay::mesh::style::{LineJoin, OutlineStyle};
use i_overlay::string::clip::ClipRule;

/// Generate perimeter toolpaths and infill for a single layer.
pub fn generate_perimeters(
    layer: &Layer,
    layer_index: usize,
    perim_config: &PerimeterConfig,
    infill_config: &InfillConfig,
) -> ToolpathLayer {
    let shapes = classify_contours(&layer.contours);
    let mut perimeter_sets = Vec::new();

    for shape in &shapes {
        let mut all_perimeters: Vec<Vec<Perimeter>> = Vec::new();
        // Wrap in an outer vec to form "Shapes" (Vec<Shape>)
        let mut current_shapes: Vec<OverlayShape> = vec![shape.clone()];

        for i in 0..perim_config.perimeter_count {
            // First perimeter: inset by nozzle_width/2 so the outer edge of
            // the extruded line aligns with the original contour.
            // Subsequent perimeters: inset by a full nozzle_width.
            let inset = if i == 0 {
                perim_config.nozzle_width / 2.0
            } else {
                perim_config.nozzle_width
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

    // Generate infill
    let infill_lines = generate_infill(&perimeter_sets, infill_config);

    ToolpathLayer {
        z: layer.z,
        layer_index,
        perimeter_sets,
        infill_lines,
    }
}

/// Generate rectilinear grid infill lines clipped to the infill boundaries.
///
/// Produces both horizontal and vertical lines on every layer, forming a grid
/// so each line is deposited on top of the previous layer's perpendicular line.
fn generate_infill(
    perimeter_sets: &[PerimeterSet],
    config: &InfillConfig,
) -> Vec<InfillLine> {
    if config.density <= 0.0 {
        return Vec::new();
    }

    let spacing = config.nozzle_width / config.density.min(1.0);

    let mut all_lines = Vec::new();

    for pset in perimeter_sets {
        if pset.infill_boundary.is_empty() {
            continue;
        }

        // Build i_overlay shape from infill boundary contours.
        // First contour with positive area is outer, rest assigned as holes.
        let boundary_shape: Vec<Vec<[f32; 2]>> = pset
            .infill_boundary
            .iter()
            .map(|c| contour_to_overlay(c))
            .collect();

        // Compute bounding box of the boundary
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for ring in &boundary_shape {
            for p in ring {
                min_x = min_x.min(p[0]);
                min_y = min_y.min(p[1]);
                max_x = max_x.max(p[0]);
                max_y = max_y.max(p[1]);
            }
        }

        // Generate a grid: both horizontal and vertical lines so each
        // line is deposited on top of the previous layer's perpendicular line.
        let mut scan_lines: Vec<Vec<[f32; 2]>> = Vec::new();

        // Horizontal lines
        let mut y = min_y + spacing;
        while y < max_y {
            scan_lines.push(vec![[min_x, y], [max_x, y]]);
            y += spacing;
        }

        // Vertical lines
        let mut x = min_x + spacing;
        while x < max_x {
            scan_lines.push(vec![[x, min_y], [x, max_y]]);
            x += spacing;
        }

        if scan_lines.is_empty() {
            continue;
        }

        // Clip each scan line individually against the infill boundary.
        // clip_by treats a Vec<Vec<>> as a single connected polyline, so we
        // must clip one line at a time.
        let clip_rule = ClipRule {
            invert: false,
            boundary_included: true,
        };
        let shapes = vec![boundary_shape];

        for line in &scan_lines {
            let clipped: Vec<Vec<[f32; 2]>> =
                vec![line.clone()].clip_by(&shapes, FillRule::EvenOdd, clip_rule);

            for segment in &clipped {
                if segment.len() >= 2 {
                    let start = from_overlay(&segment[0]);
                    let end = from_overlay(segment.last().unwrap());
                    all_lines.push(InfillLine { start, end });
                }
            }
        }
    }

    all_lines
}

/// Generate toolpaths for all layers (parallelized across cores).
pub fn generate_toolpaths(
    slice_result: &SliceResult,
    perim_config: &PerimeterConfig,
    infill_config: &InfillConfig,
) -> ToolpathResult {
    use rayon::prelude::*;

    let layers = slice_result
        .layers
        .par_iter()
        .enumerate()
        .map(|(i, layer)| generate_perimeters(layer, i, perim_config, infill_config))
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

    fn no_infill() -> InfillConfig {
        InfillConfig { density: 0.0, nozzle_width: 0.4 }
    }

    #[test]
    fn cube_single_perimeter() {
        let data = std::fs::read(stl_path("cube.stl")).unwrap();
        let mesh = stl::load_stl(&data).unwrap();
        let result = crate::slicer::slice_mesh(&mesh, 0.2);
        let layer = &result.layers[2];

        let config = PerimeterConfig {
            nozzle_width: 0.2,
            perimeter_count: 1,
        };
        let toolpath = generate_perimeters(layer, 2, &config, &no_infill());

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

    #[test]
    fn block_infill_generates_grid() {
        let data = std::fs::read(stl_path("block100.stl")).unwrap();
        let mesh = stl::load_stl(&data).unwrap();
        let result = crate::slicer::slice_mesh(&mesh, 10.0);

        let perim_config = PerimeterConfig {
            nozzle_width: 5.0,
            perimeter_count: 1,
        };
        let infill_config = InfillConfig {
            density: 0.5,
            nozzle_width: 5.0,
        };

        let tp = generate_perimeters(&result.layers[4], 0, &perim_config, &infill_config);

        assert!(
            !tp.infill_lines.is_empty(),
            "Expected infill lines"
        );

        // Grid should have both horizontal and vertical lines.
        // Use a tolerance relative to the boundary size since clipping
        // may introduce tiny floating-point deviations.
        let tol = 0.1;
        let horizontal = tp.infill_lines.iter()
            .filter(|l| (l.start.y - l.end.y).abs() < tol)
            .count();
        let vertical = tp.infill_lines.iter()
            .filter(|l| (l.start.x - l.end.x).abs() < tol)
            .count();

        assert!(horizontal > 0, "Expected horizontal infill lines, got 0");
        assert!(vertical > 0, "Expected vertical infill lines, got 0");
        println!("Infill: {horizontal} horizontal + {vertical} vertical = {} total", tp.infill_lines.len());
    }
}
