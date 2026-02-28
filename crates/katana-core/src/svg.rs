use std::fmt::Write;

use nalgebra::Point2;

use crate::offset::ToolpathLayer;
use crate::slicer::Layer;

/// Render a single layer's contours as an SVG string.
///
/// The SVG is sized to fit the contours with some padding. Coordinates are
/// flipped vertically (SVG has Y-down, we want Y-up like a print bed).
pub fn layer_to_svg(layer: &Layer, padding: f32) -> String {
    if layer.contours.is_empty() {
        return String::from(r#"<svg xmlns="http://www.w3.org/2000/svg"/>"#);
    }

    // Compute bounding box of all contour points
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for contour in &layer.contours {
        for p in &contour.points {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }
    }

    let width = max_x - min_x + 2.0 * padding;
    let height = max_y - min_y + 2.0 * padding;

    // Scale so the SVG is a reasonable pixel size (aim for ~800px on the long side)
    let scale = 800.0 / width.max(height);
    let svg_w = width * scale;
    let svg_h = height * scale;

    let mut svg = String::new();
    writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w:.0}" height="{svg_h:.0}" viewBox="0 0 {svg_w:.2} {svg_h:.2}">"#,
    )
    .unwrap();

    // Background
    writeln!(
        svg,
        r##"  <rect width="100%" height="100%" fill="#1a1a2e"/>"##,
    )
    .unwrap();

    // Layer label
    writeln!(
        svg,
        r##"  <text x="10" y="20" font-family="monospace" font-size="14" fill="#888">z = {:.3}</text>"##,
        layer.z,
    )
    .unwrap();

    // Draw contours
    for contour in &layer.contours {
        if contour.points.is_empty() {
            continue;
        }

        let mut d = String::new();
        for (i, p) in contour.points.iter().enumerate() {
            // Transform: translate so min is at padding, flip Y, scale
            let x = (p.x - min_x + padding) * scale;
            let y = (max_y - p.y + padding) * scale; // flip Y

            if i == 0 {
                write!(d, "M {x:.2} {y:.2}").unwrap();
            } else {
                write!(d, " L {x:.2} {y:.2}").unwrap();
            }
        }
        d.push_str(" Z"); // close the path

        writeln!(
            svg,
            r##"  <path d="{d}" fill="none" stroke="#e94560" stroke-width="1.5"/>"##,
        )
        .unwrap();
    }

    writeln!(svg, "</svg>").unwrap();
    svg
}

/// Render a toolpath layer as an SVG, showing perimeters in color-coded shades.
///
/// - Original contour: dim gray dashed line
/// - Perimeters: bright red (outer) to dim red (inner)
/// - Infill boundary: blue dashed line
pub fn toolpath_layer_to_svg(layer: &ToolpathLayer, original: &Layer, padding: f32) -> String {
    // Collect all points for bounding box
    let mut all_points: Vec<&Point2<f32>> = Vec::new();
    for c in &original.contours {
        all_points.extend(c.points.iter());
    }
    for pset in &layer.perimeter_sets {
        for level in &pset.perimeters {
            for p in level {
                all_points.extend(p.points.iter());
            }
        }
    }

    if all_points.is_empty() {
        return String::from(r#"<svg xmlns="http://www.w3.org/2000/svg"/>"#);
    }

    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for p in &all_points {
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }

    let width = max_x - min_x + 2.0 * padding;
    let height = max_y - min_y + 2.0 * padding;
    let scale = 800.0 / width.max(height);
    let svg_w = width * scale;
    let svg_h = height * scale;

    let mut svg = String::new();
    writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w:.0}" height="{svg_h:.0}" viewBox="0 0 {svg_w:.2} {svg_h:.2}">"#,
    )
    .unwrap();

    writeln!(
        svg,
        r##"  <rect width="100%" height="100%" fill="#1a1a2e"/>"##,
    )
    .unwrap();

    writeln!(
        svg,
        r##"  <text x="10" y="20" font-family="monospace" font-size="14" fill="#888">z = {:.3}</text>"##,
        layer.z,
    )
    .unwrap();

    let to_svg = |p: &Point2<f32>| -> (f32, f32) {
        let x = (p.x - min_x + padding) * scale;
        let y = (max_y - p.y + padding) * scale;
        (x, y)
    };

    let write_path = |svg: &mut String, points: &[Point2<f32>], stroke: &str, width: f32, extra: &str| {
        if points.is_empty() {
            return;
        }
        let mut d = String::new();
        for (i, p) in points.iter().enumerate() {
            let (x, y) = to_svg(p);
            if i == 0 {
                write!(d, "M {x:.2} {y:.2}").unwrap();
            } else {
                write!(d, " L {x:.2} {y:.2}").unwrap();
            }
        }
        d.push_str(" Z");
        writeln!(
            svg,
            r#"  <path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}"{extra}/>"#,
        )
        .unwrap();
    };

    // Original contours: dim gray dashed
    for c in &original.contours {
        write_path(&mut svg, &c.points, "#555", 0.8, r#" stroke-dasharray="4 3""#);
    }

    // Perimeters: color-coded from bright red (outer) to dim red (inner)
    let perimeter_colors = ["#e94560", "#c93850", "#a92b40", "#892030", "#691520"];

    for pset in &layer.perimeter_sets {
        for (level_idx, level) in pset.perimeters.iter().enumerate() {
            let color = perimeter_colors[level_idx.min(perimeter_colors.len() - 1)];
            let w = 1.5 - (level_idx as f32 * 0.2).min(0.8);
            for perimeter in level {
                write_path(&mut svg, &perimeter.points, color, w, "");
            }
        }

        // Infill boundary: blue dashed
        for boundary in &pset.infill_boundary {
            write_path(&mut svg, &boundary.points, "#4560e9", 0.8, r#" stroke-dasharray="3 2""#);
        }
    }

    // Infill lines: green, thin
    for line in &layer.infill_lines {
        let (x1, y1) = to_svg(&line.start);
        let (x2, y2) = to_svg(&line.end);
        writeln!(
            svg,
            r##"  <line x1="{x1:.2}" y1="{y1:.2}" x2="{x2:.2}" y2="{y2:.2}" stroke="#45e960" stroke-width="0.6"/>"##,
        )
        .unwrap();
    }

    writeln!(svg, "</svg>").unwrap();
    svg
}
