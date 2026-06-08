//! Integration tests for G-code export.
//!
//! These exercise the public pipeline a real consumer (the CLI) uses:
//!   STL -> slice_mesh -> generate_toolpaths -> plan_toolpaths -> Gcode::export
//!
//! Most assertions are *invariants* of valid G-code (E never decreases, Z
//! increases, travels don't extrude) rather than golden strings, so they only
//! fail when behavior is wrong, not when a speed default changes.

use std::f32::consts::PI;

use nalgebra::Point2;

use katana_core::gcode::{ Gcode, GcodeConfig };
use katana_core::offset::{ generate_toolpaths, InfillConfig, PerimeterConfig, SurfaceConfig };
use katana_core::planner::{
    plan_toolpaths,
    Move,
    MoveKind,
    PlannedLayer,
    PlannedResult,
    SpeedConfig,
};
use katana_core::{ slicer, stl };

const NOZZLE: f32 = 0.4;
const LAYER_H: f32 = 0.2;
const FILAMENT: f32 = 1.75;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Run a plan through a fresh exporter with the shared test config.
fn export(plan: &PlannedResult) -> String {
    let mut g = Gcode {
        e: 0.0,
        config: GcodeConfig {
            filament_diameter: FILAMENT,
            nozzle_width: NOZZLE,
            layer_height: LAYER_H,
        },
        // Zero offset keeps emitted coordinates in model space for assertions.
        offset: nalgebra::Vector2::zeros(),
        out: String::new(),
    };
    g.export(plan)
}

/// Slice the bundled cube STL all the way to G-code via the real pipeline.
fn cube_gcode() -> String {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../stls/cube.stl");
    let data = std::fs::read(path).expect("read cube.stl");
    let mesh = stl::load_stl(&data).expect("parse cube.stl");

    let slices = slicer::slice_mesh(&mesh, LAYER_H);
    let toolpaths = generate_toolpaths(
        &slices,
        &(PerimeterConfig {
            nozzle_width: NOZZLE,
            perimeter_count: 3,
            layer_height: LAYER_H,
        }),
        &(InfillConfig {
            density: 0.2,
            nozzle_width: NOZZLE,
        }),
        &(SurfaceConfig {
            bottom_layers: 3,
            top_layers: 3,
        })
    );
    let plan = plan_toolpaths(&toolpaths, &SpeedConfig::default());
    export(&plan)
}

/// Parse the numeric value of a single-letter G-code word (e.g. 'X', 'E') from
/// a line, if present. Returns None if the word is absent or unparseable.
fn word(line: &str, letter: char) -> Option<f32> {
    line.split_whitespace()
        .find(|tok| tok.starts_with(letter))
        .and_then(|tok| tok[1..].parse::<f32>().ok())
}

// ---------------------------------------------------------------------------
// Structural invariants (real cube pipeline)
// ---------------------------------------------------------------------------

#[test]
fn emits_start_and_end_sequences() {
    let g = cube_gcode();
    // Start: home, heat nozzle (wait), heat bed (wait), zero the extruder.
    for needle in ["G28", "M109 S", "M190 S", "G92 E0"] {
        assert!(g.contains(needle), "start sequence missing `{needle}`");
    }
    // End: cooldown + steppers off.
    for needle in ["M104 S0", "M140 S0", "M84"] {
        assert!(g.contains(needle), "end sequence missing `{needle}`");
    }
}

#[test]
fn one_z_move_per_layer_and_z_strictly_increases() {
    let g = cube_gcode();

    // Restrict to the layer body so the end-sequence Z lift isn't counted.
    let body = g.split("; --- Ending sequence").next().unwrap();

    let layer_markers = body
        .lines()
        .filter(|l| l.starts_with("; CHANGE_LAYER"))
        .count();
    let z_moves: Vec<f32> = body
        .lines()
        .filter(|l| l.starts_with("G1 Z"))
        .filter_map(|l| word(l, 'Z'))
        .collect();

    assert!(layer_markers > 1, "cube should produce multiple layers");
    assert_eq!(
        layer_markers,
        z_moves.len(),
        "every layer marker should be paired with exactly one Z move"
    );
    for pair in z_moves.windows(2) {
        assert!(
            pair[1] > pair[0],
            "Z must strictly increase between layers: {} -> {}",
            pair[0],
            pair[1]
        );
    }
}

#[test]
fn extrusion_is_monotonic_nondecreasing() {
    // Absolute E mode with no retraction yet, so cumulative E must never drop.
    let g = cube_gcode();
    let mut last = 0.0f32;
    for line in g.lines().filter(|l| l.starts_with("G1")) {
        if let Some(e) = word(line, 'E') {
            assert!(e >= last - 1e-6, "cumulative E went backwards: {last} -> {e} on `{line}`");
            last = e;
        }
    }
    assert!(last > 0.0, "no extrusion happened at all");
}

#[test]
fn travels_never_extrude() {
    let g = cube_gcode();
    for line in g.lines().filter(|l| l.starts_with("G0")) {
        assert!(word(line, 'E').is_none(), "travel move carried an E word: `{line}`");
    }
}

#[test]
fn all_coordinates_finite_and_feedrates_positive() {
    let g = cube_gcode();
    for line in g.lines().filter(|l| (l.starts_with("G0") || l.starts_with("G1"))) {
        for letter in ['X', 'Y', 'Z', 'E'] {
            if let Some(v) = word(line, letter) {
                assert!(v.is_finite(), "non-finite {letter} on `{line}`");
            }
        }
        if let Some(f) = word(line, 'F') {
            assert!(f > 0.0, "non-positive feedrate on `{line}`");
        }
    }
}

// ---------------------------------------------------------------------------
// Numeric correctness (synthetic plan with known geometry)
// ---------------------------------------------------------------------------

/// A one-layer plan with a single closed square perimeter of side `s`.
/// The emitter adds the implicit closing segment, so the deposited path length
/// is the full perimeter: `4 * s`.
fn square_perimeter_plan(s: f32) -> PlannedResult {
    let points = vec![
        Point2::new(0.0, 0.0),
        Point2::new(s, 0.0),
        Point2::new(s, s),
        Point2::new(0.0, s)
    ];
    let mv = Move {
        kind: MoveKind::Perimeter,
        points,
        speed: 30.0,
        flow: 0.0,
    };
    PlannedResult {
        layers: vec![PlannedLayer {
            z: LAYER_H,
            layer_index: 0,
            moves: vec![mv],
        }],
    }
}

#[test]
fn extrusion_volume_matches_bead_geometry() {
    let side = 10.0f32;
    let g = export(&square_perimeter_plan(side));

    // Final cumulative E = the last E word emitted in the file.
    let final_e = g
        .lines()
        .filter(|l| l.starts_with("G1"))
        .filter_map(|l| word(l, 'E'))
        .last()
        .expect("the perimeter should emit at least one extruding move");

    let bead_area = NOZZLE * LAYER_H;
    let filament_area = PI * (FILAMENT / 2.0) * (FILAMENT / 2.0);
    let len = 4.0 * side;
    let expected_e = bead_area * (len / filament_area);
    let diff = final_e - expected_e;
    assert!(diff.abs() < 0.01, "Final e doesn't match bead geometry!")
}
