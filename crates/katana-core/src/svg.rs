use std::fmt::Write;

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
