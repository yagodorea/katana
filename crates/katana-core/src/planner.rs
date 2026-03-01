use nalgebra::Point2;
use crate::offset::{ToolpathLayer, ToolpathResult, Perimeter, InfillLine};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The type of move the nozzle makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveKind {
    /// Non-extrusion repositioning.
    Travel,
    /// Extrude along a closed perimeter loop.
    Perimeter,
    /// Extrude a single infill line segment.
    Infill,
    /// Extrude a monotonic surface infill line (solid top/bottom layers).
    SurfaceInfill,
}

/// A single move: an ordered list of points the nozzle follows.
#[derive(Debug, Clone)]
pub struct Move {
    pub kind: MoveKind,
    /// Ordered points the nozzle follows for this move.
    /// - Travel: [from, to] (2 points)
    /// - Perimeter: [p0, p1, ..., pN] closed loop (last connects back to first)
    /// - Infill: [start, end] (2 points)
    pub points: Vec<Point2<f32>>,
}

/// A layer with a fully ordered sequence of moves.
#[derive(Debug, Clone)]
pub struct PlannedLayer {
    pub z: f32,
    pub layer_index: usize,
    pub moves: Vec<Move>,
}

/// Result of planning all layers.
#[derive(Debug)]
pub struct PlannedResult {
    pub layers: Vec<PlannedLayer>,
}

// ---------------------------------------------------------------------------
// Planning logic
// ---------------------------------------------------------------------------

/// Plan all toolpath layers, adding travel moves and optimizing print order.
/// Layers are planned in parallel; each starts from the origin.
pub fn plan_toolpaths(toolpath_result: &ToolpathResult) -> PlannedResult {
    use rayon::prelude::*;

    let origin = Point2::new(0.0, 0.0);
    let layers: Vec<PlannedLayer> = toolpath_result
        .layers
        .par_iter()
        .map(|layer| plan_layer(layer, origin))
        .collect();

    PlannedResult { layers }
}

/// Plan a single layer: order segments and insert travel moves.
fn plan_layer(layer: &ToolpathLayer, start_pos: Point2<f32>) -> PlannedLayer {
    let mut moves = Vec::new();
    let mut current_pos = start_pos;

    // Order perimeters and infill using nearest-neighbor
    let ordered_perimeter_sets = order_perimeter_sets_nearest(&layer.perimeter_sets, &current_pos);

    // Process each perimeter set in order
    for ordered_set in &ordered_perimeter_sets {
        // Get the PerimeterSet using the stored pset_idx
        let pset = &layer.perimeter_sets[ordered_set.pset_idx];

        // Print perimeters from innermost to outermost (reverse order)
        for level_idx in ordered_set.level_indices.iter().rev() {
            let perimeters = &pset.perimeters[*level_idx];

            // Within this level, order disjoint loops by nearest-neighbor
            let mut perimeter_order: Vec<usize> = (0..perimeters.len()).collect();
            order_perimeter_loops_nearest(&mut perimeter_order, perimeters, &current_pos);

            // Process each perimeter loop
            for loop_idx in perimeter_order {
                let perimeter = &perimeters[loop_idx];
                let mut points = perimeter.points.clone();

                // Rotate points so start is nearest to current position
                rotate_to_nearest(&mut points, &current_pos);

                // Capture the rotated start point (loop closes back here)
                let loop_start = points[0];

                // Add travel move if needed
                if !points_equal(&current_pos, &loop_start) {
                    moves.push(Move {
                        kind: MoveKind::Travel,
                        points: vec![current_pos, loop_start],
                    });
                }

                // Add perimeter move
                moves.push(Move {
                    kind: MoveKind::Perimeter,
                    points,
                });

                // Nozzle returns to loop start after closing the perimeter
                current_pos = loop_start;
            }
        }
    }

    // Process infill lines using nearest-neighbor with direction optimization
    if !layer.infill_lines.is_empty() {
        let ordered_infill = order_infill_nearest(&layer.infill_lines, &current_pos);

        for (line_idx, flipped) in ordered_infill {
            let line = &layer.infill_lines[line_idx];
            let (start, end) = if flipped {
                (line.end, line.start)
            } else {
                (line.start, line.end)
            };

            // Add travel move if needed
            if !points_equal(&current_pos, &start) {
                moves.push(Move {
                    kind: MoveKind::Travel,
                    points: vec![current_pos, start],
                });
            }

            // Add infill move
            moves.push(Move {
                kind: MoveKind::Infill,
                points: vec![start, end],
            });

            current_pos = end;
        }
    }

    // Process surface infill lines (serpentine/boustrophedon pattern)
    // Works for any line angle by deriving direction from the lines themselves.
    if !layer.surface_infill_lines.is_empty() {
        let lines = &layer.surface_infill_lines;

        // Derive line direction and perpendicular from the first line
        let first = &lines[0];
        let dx = first.end.x - first.start.x;
        let dy = first.end.y - first.start.y;
        let len = (dx * dx + dy * dy).sqrt();
        let dir_x = dx / len;
        let dir_y = dy / len;
        // Perpendicular to the line direction (used for scanline ordering)
        let perp_x = -dir_y;
        let perp_y = dir_x;

        // Sort lines by perpendicular projection (scanline order)
        let mut indices: Vec<usize> = (0..lines.len()).collect();
        indices.sort_unstable_by(|&a, &b| {
            let la = &lines[a];
            let lb = &lines[b];
            let mid_a = (la.start.x + la.end.x) * perp_x + (la.start.y + la.end.y) * perp_y;
            let mid_b = (lb.start.x + lb.end.x) * perp_x + (lb.start.y + lb.end.y) * perp_y;
            mid_a.partial_cmp(&mid_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Serpentine: alternate direction each scanline row
        let mut forward = true;
        for &idx in &indices {
            let line = &lines[idx];

            // Project endpoints onto the line direction to determine ordering
            let proj_s = line.start.x * dir_x + line.start.y * dir_y;
            let proj_e = line.end.x * dir_x + line.end.y * dir_y;

            let (start, end) = if (proj_s <= proj_e) == forward {
                (line.start, line.end)
            } else {
                (line.end, line.start)
            };

            // Add travel move if needed
            if !points_equal(&current_pos, &start) {
                moves.push(Move {
                    kind: MoveKind::Travel,
                    points: vec![current_pos, start],
                });
            }

            // Add surface infill move
            moves.push(Move {
                kind: MoveKind::SurfaceInfill,
                points: vec![start, end],
            });

            current_pos = end;
            forward = !forward;
        }
    }

    PlannedLayer {
        z: layer.z,
        layer_index: layer.layer_index,
        moves,
    }
}

/// Helper to compare points with tolerance.
fn points_equal(a: &Point2<f32>, b: &Point2<f32>) -> bool {
    (a.x - b.x).abs() < 1e-6 && (a.y - b.y).abs() < 1e-6
}

/// Rotate points so the closest vertex to `from` becomes index 0.
fn rotate_to_nearest(points: &mut Vec<Point2<f32>>, from: &Point2<f32>) {
    if points.is_empty() {
        return;
    }

    // Find the index of the point closest to `from`
    let nearest_idx = points
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = distance_squared(from, a);
            let db = distance_squared(from, b);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Rotate the vector so the nearest point is at index 0
    if nearest_idx != 0 {
        let n = points.len();
        let mut rotated = Vec::with_capacity(n);
        for i in 0..n {
            rotated.push(points[(nearest_idx + i) % n]);
        }
        *points = rotated;
    }
}

/// Squared distance between two points (avoids sqrt for comparisons).
fn distance_squared(a: &Point2<f32>, b: &Point2<f32>) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    dx * dx + dy * dy
}

/// Structure to hold ordered perimeter set information.
#[derive(Clone)]
struct OrderedPerimeterSet {
    pset_idx: usize,
    level_indices: Vec<usize>, // All levels present in this set
}

/// Order perimeter sets by nearest-neighbor from current position.
fn order_perimeter_sets_nearest(
    perimeter_sets: &[crate::offset::PerimeterSet],
    from: &Point2<f32>,
) -> Vec<OrderedPerimeterSet> {
    let sets: Vec<OrderedPerimeterSet> = perimeter_sets
        .iter()
        .enumerate()
        .map(|(i, pset)| OrderedPerimeterSet {
            pset_idx: i,
            level_indices: (0..pset.perimeters.len()).collect(),
        })
        .collect();

    // Find the start point for each set (innermost perimeter's first point)
    let mut set_starts: Vec<(usize, Point2<f32>)> = sets
        .iter()
        .filter_map(|s| {
            let pset = &perimeter_sets[s.pset_idx];
            // Innermost is the last level
            let innermost = pset.perimeters.last()?;
            // Get the first perimeter loop at this level, and its first point
            innermost.first()?.points.first().map(|pt| (s.pset_idx, *pt))
        })
        .collect();

    // Greedy nearest-neighbor ordering
    let mut ordered = Vec::new();
    let mut current = *from;

    while !set_starts.is_empty() {
        // Find the nearest set
        let nearest_idx = set_starts
            .iter()
            .enumerate()
            .min_by(|a, b| {
                let dist_a = distance_squared(&current, &a.1 .1);
                let dist_b = distance_squared(&current, &b.1 .1);
                dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let (pset_idx, start_pt) = set_starts.remove(nearest_idx);

        // Find the corresponding OrderedPerimeterSet
        let set_info = sets.iter().find(|s| s.pset_idx == pset_idx).cloned().unwrap();
        ordered.push(set_info);

        current = start_pt;
    }

    ordered
}

/// Order perimeter loops within a level by nearest-neighbor.
fn order_perimeter_loops_nearest(
    loop_indices: &mut Vec<usize>,
    perimeters: &[Perimeter],
    from: &Point2<f32>,
) {
    if loop_indices.len() <= 1 {
        return;
    }

    // Get start points for all loops
    let mut loop_starts: Vec<(usize, Point2<f32>)> = loop_indices
        .iter()
        .filter_map(|&i| perimeters.get(i).and_then(|p| p.points.first().map(|pt| (i, *pt))))
        .collect();

    let mut ordered = Vec::new();
    let mut current = *from;

    while !loop_starts.is_empty() {
        let nearest_idx = loop_starts
            .iter()
            .enumerate()
            .min_by(|a, b| {
                let dist_a = distance_squared(&current, &a.1 .1);
                let dist_b = distance_squared(&current, &b.1 .1);
                dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let (loop_idx, start_pt) = loop_starts.remove(nearest_idx);
        ordered.push(loop_idx);
        current = start_pt;
    }

    *loop_indices = ordered;
}

/// Order infill lines by nearest-neighbor, also considering direction.
/// Returns Vec of (line_index, flipped) tuples.
fn order_infill_nearest(
    infill_lines: &[InfillLine],
    from: &Point2<f32>,
) -> Vec<(usize, bool)> {
    if infill_lines.is_empty() {
        return Vec::new();
    }

    let mut remaining: Vec<usize> = (0..infill_lines.len()).collect();
    let mut ordered = Vec::with_capacity(remaining.len());
    let mut current = *from;

    while !remaining.is_empty() {
        // Find the nearest line, considering both directions
        let (best_pos, best_line_idx, best_flipped) = remaining
            .iter()
            .enumerate()
            .map(|(pos, &idx)| {
                let line = &infill_lines[idx];
                let dist_start = distance_squared(&current, &line.start);
                let dist_end = distance_squared(&current, &line.end);

                if dist_start <= dist_end {
                    (pos, idx, false, dist_start)
                } else {
                    (pos, idx, true, dist_end)
                }
            })
            .min_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(pos, idx, flipped, _)| (pos, idx, flipped))
            .unwrap();

        // Update current position
        let line = &infill_lines[best_line_idx];
        current = if best_flipped {
            line.start // We're printing from end to start
        } else {
            line.end // We're printing from start to end
        };

        // O(1) removal via swap_remove instead of O(n) retain
        remaining.swap_remove(best_pos);
        ordered.push((best_line_idx, best_flipped));
    }

    ordered
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotate_to_nearest_moves_closest_to_front() {
        // Use a reference point that's closer to one corner than others
        let from = Point2::new(8.0, 8.0);
        let mut points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ];

        rotate_to_nearest(&mut points, &from);

        // (10, 10) should be first (closest to (8, 8))
        assert_eq!(points[0], Point2::new(10.0, 10.0));
        assert_eq!(points[1], Point2::new(0.0, 10.0));
        assert_eq!(points[2], Point2::new(0.0, 0.0));
        assert_eq!(points[3], Point2::new(10.0, 0.0));
    }

    #[test]
    fn distance_squared_works() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(3.0, 4.0);
        assert!((distance_squared(&a, &b) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn order_infill_nearest_considers_direction() {
        let from = Point2::new(5.0, 0.0);
        let lines = vec![
            InfillLine { start: Point2::new(0.0, 0.0), end: Point2::new(10.0, 0.0) },
            InfillLine { start: Point2::new(0.0, 10.0), end: Point2::new(10.0, 10.0) },
        ];

        let ordered = order_infill_nearest(&lines, &from);

        // First line should be the bottom one, from start (0,0) to end (10,0)
        // because (10, 0) is closer to (5, 0) than any point on the top line
        assert_eq!(ordered[0].0, 0);
        assert!(!ordered[0].1); // Not flipped
    }

    #[test]
    fn points_equal_with_tolerance() {
        let a = Point2::new(1.0, 2.0);
        let b = Point2::new(1.000001, 2.000001);
        assert!(points_equal(&a, &b));

        let c = Point2::new(1.01, 2.0);
        assert!(!points_equal(&a, &c));
    }
}
