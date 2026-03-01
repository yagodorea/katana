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

/// Configuration for top/bottom surface layers.
#[derive(Debug, Clone)]
pub struct SurfaceConfig {
    /// Number of solid bottom layers (first N layers from build plate).
    pub bottom_layers: usize,
    /// Number of solid top layers (last N layers before top of model).
    pub top_layers: usize,
}

/// A fully processed layer with perimeter toolpaths and infill.
#[derive(Debug, Clone)]
pub struct ToolpathLayer {
    pub z: f32,
    pub layer_index: usize,
    pub perimeter_sets: Vec<PerimeterSet>,
    pub infill_lines: Vec<InfillLine>,
    /// Solid surface infill for top/bottom layers (monotonic unidirectional).
    /// Only present on layers configured as top or bottom surfaces.
    pub surface_infill_lines: Vec<InfillLine>,
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
/// Axis-aligned bounding box for fast containment pre-checks.
struct Aabb {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
}

impl Aabb {
    fn from_points(points: &[Point2<f32>]) -> Self {
        let mut bb = Aabb {
            min_x: f32::INFINITY,
            min_y: f32::INFINITY,
            max_x: f32::NEG_INFINITY,
            max_y: f32::NEG_INFINITY,
        };
        for p in points {
            bb.min_x = bb.min_x.min(p.x);
            bb.min_y = bb.min_y.min(p.y);
            bb.max_x = bb.max_x.max(p.x);
            bb.max_y = bb.max_y.max(p.y);
        }
        bb
    }

    fn contains(&self, p: &Point2<f32>) -> bool {
        p.x >= self.min_x && p.x <= self.max_x && p.y >= self.min_y && p.y <= self.max_y
    }
}

fn classify_contours(contours: &[Contour]) -> Vec<OverlayShape> {
    if contours.is_empty() {
        return Vec::new();
    }

    let n = contours.len();

    // Precompute AABBs and signed areas for all contours
    let aabbs: Vec<Aabb> = contours.iter().map(|c| Aabb::from_points(&c.points)).collect();
    let areas: Vec<f32> = contours.iter().map(|c| signed_area(&c.points)).collect();

    // For each contour, count how many other contours contain its first point.
    // Even depth = outer boundary; odd depth = hole.
    // AABB pre-check skips expensive PIP tests when point is outside bounding box.
    let mut depth = vec![0usize; n];
    for i in 0..n {
        if let Some(test_pt) = contours[i].points.first() {
            for j in 0..n {
                if i != j
                    && aabbs[j].contains(test_pt)
                    && point_in_polygon(test_pt, &contours[j].points)
                {
                    depth[i] += 1;
                }
            }
        }
    }

    let mut outers: Vec<(usize, f32)> = Vec::new(); // (index, abs_area)
    let mut holes: Vec<(usize, f32)> = Vec::new();  // (index, signed_area)

    for i in 0..n {
        let area = areas[i];
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
        let mut ring = contour_to_overlay(&contours[idx]);
        // Ensure CCW winding for outer (positive area)
        if areas[idx] < 0.0 {
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
                    && aabbs[outer_idx].contains(test_point)
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

/// Generate perimeter toolpaths and infill for a single layer.
pub fn generate_perimeters(
    layer: &Layer,
    layer_index: usize,
    total_layers: usize,
    perim_config: &PerimeterConfig,
    infill_config: &InfillConfig,
    surface_config: &SurfaceConfig,
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

            // Pass all current shapes as a batch to .outline()
            let offset_result: Vec<OverlayShape> = current_shapes.outline(&style);

            for offset_shape in &offset_result {
                for ring in offset_shape {
                    if ring.len() >= 3 {
                        level_perimeters.push(Perimeter {
                            points: overlay_to_points(ring),
                        });
                    }
                }
            }

            if level_perimeters.is_empty() {
                break; // Geometry shrank to nothing
            }

            all_perimeters.push(level_perimeters);
            current_shapes = offset_result;
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

    // Determine if this is a top or bottom surface layer
    let is_bottom_layer = layer_index < surface_config.bottom_layers;
    let is_top_layer = layer_index >= total_layers.saturating_sub(surface_config.top_layers);
    let is_surface_layer = is_bottom_layer || is_top_layer;

    // Generate infill
    let infill_lines = if is_surface_layer {
        // Surface layers get monotonic solid infill instead of regular infill
        Vec::new()
    } else {
        generate_infill(&perimeter_sets, infill_config)
    };

    // Generate monotonic surface infill for top/bottom layers
    let surface_infill_lines = if is_surface_layer {
        // Alternate between 45° and 135° each layer for cross-hatched strength
        let angle = if layer_index % 2 == 0 {
            std::f32::consts::FRAC_PI_4
        } else {
            std::f32::consts::FRAC_PI_4 + std::f32::consts::FRAC_PI_2
        };
        generate_monotonic_surface_infill(&perimeter_sets, perim_config.nozzle_width, angle)
    } else {
        Vec::new()
    };

    ToolpathLayer {
        z: layer.z,
        layer_index,
        perimeter_sets,
        infill_lines,
        surface_infill_lines,
    }
}

/// Clip a horizontal scan line (at y = `y`) against polygon edges using
/// even-odd ray intersection. Returns pairs of x-coordinates representing
/// inside segments.
fn clip_horizontal_line(y: f32, edges: &[([f32; 2], [f32; 2])], min_x: f32, max_x: f32) -> Vec<(f32, f32)> {
    let mut intersections = Vec::new();

    for &(p0, p1) in edges {
        let (y0, y1) = (p0[1], p1[1]);
        // Check if edge spans this y (exclusive of endpoints to avoid
        // double-counting at vertices).
        if (y0 < y && y1 >= y) || (y1 < y && y0 >= y) {
            let t = (y - y0) / (y1 - y0);
            let x = p0[0] + t * (p1[0] - p0[0]);
            intersections.push(x);
        }
    }

    intersections.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let mut segments = Vec::new();
    // Even-odd: pairs of intersections define inside segments
    for pair in intersections.chunks_exact(2) {
        let x0 = pair[0].max(min_x);
        let x1 = pair[1].min(max_x);
        if x0 < x1 {
            segments.push((x0, x1));
        }
    }
    segments
}

/// Clip a vertical scan line (at x = `x`) against polygon edges using
/// even-odd ray intersection. Returns pairs of y-coordinates.
fn clip_vertical_line(x: f32, edges: &[([f32; 2], [f32; 2])], min_y: f32, max_y: f32) -> Vec<(f32, f32)> {
    let mut intersections = Vec::new();

    for &(p0, p1) in edges {
        let (x0, x1) = (p0[0], p1[0]);
        if (x0 < x && x1 >= x) || (x1 < x && x0 >= x) {
            let t = (x - x0) / (x1 - x0);
            let y = p0[1] + t * (p1[1] - p0[1]);
            intersections.push(y);
        }
    }

    intersections.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let mut segments = Vec::new();
    for pair in intersections.chunks_exact(2) {
        let y0 = pair[0].max(min_y);
        let y1 = pair[1].min(max_y);
        if y0 < y1 {
            segments.push((y0, y1));
        }
    }
    segments
}

/// Generate rectilinear grid infill lines clipped to the infill boundaries.
///
/// Uses direct scanline-polygon intersection (even-odd rule) instead of
/// i_overlay's full clip_by, which is orders of magnitude faster for
/// axis-aligned lines.
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

        // Build edge list from all boundary contours (outer + holes).
        // Even-odd fill rule handles holes naturally.
        let mut edges: Vec<([f32; 2], [f32; 2])> = Vec::new();
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        for contour in &pset.infill_boundary {
            let pts = &contour.points;
            let n = pts.len();
            if n < 3 {
                continue;
            }
            for i in 0..n {
                let j = (i + 1) % n;
                let p0 = [pts[i].x, pts[i].y];
                let p1 = [pts[j].x, pts[j].y];
                edges.push((p0, p1));
                min_x = min_x.min(p0[0]);
                min_y = min_y.min(p0[1]);
                max_x = max_x.max(p0[0]);
                max_y = max_y.max(p0[1]);
            }
        }

        // Horizontal scan lines at absolute grid positions (aligned to origin)
        let mut y = (min_y / spacing).ceil() * spacing;
        while y < max_y {
            for (x0, x1) in clip_horizontal_line(y, &edges, min_x, max_x) {
                all_lines.push(InfillLine {
                    start: Point2::new(x0, y),
                    end: Point2::new(x1, y),
                });
            }
            y += spacing;
        }

        // Vertical scan lines at absolute grid positions (aligned to origin)
        let mut x = (min_x / spacing).ceil() * spacing;
        while x < max_x {
            for (y0, y1) in clip_vertical_line(x, &edges, min_y, max_y) {
                all_lines.push(InfillLine {
                    start: Point2::new(x, y0),
                    end: Point2::new(x, y1),
                });
            }
            x += spacing;
        }
    }

    all_lines
}

/// Generate monotonic surface infill for solid top/bottom layers at an
/// arbitrary angle. Rotates the boundary into axis-aligned space, runs
/// horizontal scan lines, then rotates results back.
fn generate_monotonic_surface_infill(
    perimeter_sets: &[PerimeterSet],
    nozzle_width: f32,
    angle: f32,
) -> Vec<InfillLine> {
    let cos = angle.cos();
    let sin = angle.sin();

    // Rotate a point by -angle (into scan-line space)
    let rotate_fwd = |x: f32, y: f32| -> [f32; 2] {
        [x * cos + y * sin, -x * sin + y * cos]
    };
    // Rotate a point by +angle (back to world space)
    let rotate_inv = |u: f32, v: f32| -> Point2<f32> {
        Point2::new(u * cos - v * sin, u * sin + v * cos)
    };

    let spacing = nozzle_width;
    let mut all_lines = Vec::new();

    for pset in perimeter_sets {
        if pset.infill_boundary.is_empty() {
            continue;
        }

        // Build rotated edge list and bounding box
        let mut edges: Vec<([f32; 2], [f32; 2])> = Vec::new();
        let mut min_u = f32::INFINITY;
        let mut min_v = f32::INFINITY;
        let mut max_u = f32::NEG_INFINITY;
        let mut max_v = f32::NEG_INFINITY;

        for contour in &pset.infill_boundary {
            let pts = &contour.points;
            let n = pts.len();
            if n < 3 {
                continue;
            }
            for i in 0..n {
                let j = (i + 1) % n;
                let p0 = rotate_fwd(pts[i].x, pts[i].y);
                let p1 = rotate_fwd(pts[j].x, pts[j].y);
                edges.push((p0, p1));
                min_u = min_u.min(p0[0]).min(p1[0]);
                min_v = min_v.min(p0[1]).min(p1[1]);
                max_u = max_u.max(p0[0]).max(p1[0]);
                max_v = max_v.max(p0[1]).max(p1[1]);
            }
        }

        // Horizontal scan lines in rotated space at absolute grid positions
        let half = spacing / 2.0;
        let mut v = ((min_v - half) / spacing).ceil() * spacing + half;
        while v < max_v {
            for (u0, u1) in clip_horizontal_line(v, &edges, min_u, max_u) {
                all_lines.push(InfillLine {
                    start: rotate_inv(u0, v),
                    end: rotate_inv(u1, v),
                });
            }
            v += spacing;
        }
    }

    all_lines
}

/// Generate toolpaths for all layers (parallelized across cores).
pub fn generate_toolpaths(
    slice_result: &SliceResult,
    perim_config: &PerimeterConfig,
    infill_config: &InfillConfig,
    surface_config: &SurfaceConfig,
) -> ToolpathResult {
    use rayon::prelude::*;

    let total_layers = slice_result.layers.len();
    let layers = slice_result
        .layers
        .par_iter()
        .enumerate()
        .map(|(i, layer)| generate_perimeters(layer, i, total_layers, perim_config, infill_config, surface_config))
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
        let toolpath = generate_perimeters(
            layer,
            2,
            result.layers.len(),
            &config,
            &no_infill(),
            &SurfaceConfig { bottom_layers: 0, top_layers: 0 },
        );

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

        let tp = generate_perimeters(
            &result.layers[4],
            0,
            result.layers.len(),
            &perim_config,
            &infill_config,
            &SurfaceConfig { bottom_layers: 0, top_layers: 0 },
        );

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

    #[test]
    fn infill_lines_are_axis_aligned_and_inside_boundary() {
        // Verify that every infill line is either horizontal or vertical
        // and that all endpoints lie within the infill boundary.
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

        let tp = generate_perimeters(
            &result.layers[4],
            0,
            result.layers.len(),
            &perim_config,
            &infill_config,
            &SurfaceConfig { bottom_layers: 0, top_layers: 0 },
        );
        assert!(!tp.infill_lines.is_empty(), "Expected infill lines");

        let tol = 0.1;

        // Every infill line must be either horizontal or vertical
        for (i, line) in tp.infill_lines.iter().enumerate() {
            let is_horizontal = (line.start.y - line.end.y).abs() < tol;
            let is_vertical = (line.start.x - line.end.x).abs() < tol;
            assert!(
                is_horizontal || is_vertical,
                "Infill line {} is neither horizontal nor vertical: ({:.3},{:.3}) -> ({:.3},{:.3})",
                i, line.start.x, line.start.y, line.end.x, line.end.y
            );
        }

        // Collect the infill boundary polygon(s)
        let boundary_polys: Vec<&[Point2<f32>]> = tp
            .perimeter_sets
            .iter()
            .flat_map(|ps| ps.infill_boundary.iter().map(|c| c.points.as_slice()))
            .collect();

        assert!(!boundary_polys.is_empty(), "Expected infill boundary");

        // Every infill endpoint must be inside or on the boundary of at least
        // one infill boundary polygon (with a small tolerance for clipping).
        let margin = 0.5; // generous margin for clipping tolerance
        for (i, line) in tp.infill_lines.iter().enumerate() {
            for (label, pt) in [("start", &line.start), ("end", &line.end)] {
                let inside = boundary_polys.iter().any(|poly| {
                    // Check if point is within the bounding box + margin
                    let (mut bmin_x, mut bmin_y) = (f32::INFINITY, f32::INFINITY);
                    let (mut bmax_x, mut bmax_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
                    for p in *poly {
                        bmin_x = bmin_x.min(p.x);
                        bmin_y = bmin_y.min(p.y);
                        bmax_x = bmax_x.max(p.x);
                        bmax_y = bmax_y.max(p.y);
                    }
                    pt.x >= bmin_x - margin
                        && pt.x <= bmax_x + margin
                        && pt.y >= bmin_y - margin
                        && pt.y <= bmax_y + margin
                });
                assert!(
                    inside,
                    "Infill line {} {} ({:.3},{:.3}) is outside all boundaries",
                    i, label, pt.x, pt.y
                );
            }
        }
    }

    #[test]
    fn sphere_infill_generates_lines() {
        // Sphere has curved contours — verify infill works on non-rectangular geometry.
        let data = std::fs::read(stl_path("sphere.stl")).unwrap();
        let mesh = stl::load_stl(&data).unwrap();
        let result = crate::slicer::slice_mesh(&mesh, 2.0);

        let perim_config = PerimeterConfig {
            nozzle_width: 0.4,
            perimeter_count: 2,
        };
        let infill_config = InfillConfig {
            density: 0.2,
            nozzle_width: 0.4,
        };

        // Test a mid-height layer (away from poles)
        let (mid_layer_idx, mid_layer) = result
            .layers
            .iter()
            .enumerate()
            .find(|(_, l)| l.z > 20.0 && l.z < 30.0)
            .expect("Expected mid-range layer");

        let tp = generate_perimeters(
            mid_layer,
            mid_layer_idx,
            result.layers.len(),
            &perim_config,
            &infill_config,
            &SurfaceConfig { bottom_layers: 0, top_layers: 0 },
        );
        assert!(
            !tp.infill_lines.is_empty(),
            "Expected infill lines on sphere mid-layer at z={}",
            mid_layer.z
        );

        // All lines should be axis-aligned
        let tol = 0.1;
        for line in &tp.infill_lines {
            let is_horizontal = (line.start.y - line.end.y).abs() < tol;
            let is_vertical = (line.start.x - line.end.x).abs() < tol;
            assert!(
                is_horizontal || is_vertical,
                "Non-axis-aligned infill on sphere: ({:.3},{:.3}) -> ({:.3},{:.3})",
                line.start.x, line.start.y, line.end.x, line.end.y
            );
        }
    }
}
