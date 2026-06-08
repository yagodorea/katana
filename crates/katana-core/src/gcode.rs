//
// G-code reference
// -- Movement
// G0 X Y - fast movement (travel)
// G1 X Y E F - linear movement to (X,Y), with extrusion rate E, and feed rate (speed) F
// G28 - home
// G92 E0 - redefine the current extruder position to zero (without moving it)
//     ("consider the current amount of filament pushed as zero")
// G92 Z0 - redefine current Z position to zero (without moving it)
// -- Other
// M84 - disable steppers
// M104 S - set temperature of nozzle without waiting for it to reach the temp (async)
// M109 S - set temperature of nozzle while waiting for it to reach the temp (sync)
// M140 S - set temperature of hotbed without waiting for it to reach the temp (async)
// M190 S - set temperature of hotbed while waiting for it to reach the temp (sync)

use std::{ f32::consts::PI, fmt::Write };

use nalgebra::{ Point2, Vector2 };

use crate::arc::{ self, PathPrimitive };
use crate::planner::{ Move, MoveKind, PlannedResult };

const ARC_FIT_TOLERANCE: f32 = 0.05;

pub struct GcodeConfig {
    pub filament_diameter: f32,
    pub nozzle_width: f32,
    pub layer_height: f32,
}

/// Build-plate dimensions in mm (origin at the front-left corner)
#[derive(Clone, Copy)]
pub struct BedConfig {
    pub width: f32, // X
    pub depth: f32, // Y
}

pub struct Gcode {
    pub config: GcodeConfig,
    /// Translation added to every absolute XY coordinate before it is emitted.
    pub offset: Vector2<f32>,
    pub e: f32, // cumulative absolute-E total
    pub out: String,
}

/// Compute the XY translation that places a model onto the bed
pub fn bed_offset(bed: BedConfig, model_min: Point2<f32>, model_max: Point2<f32>) -> Vector2<f32> {
    let model_mid = Point2::new(
        (model_max.x + model_min.x) / 2.0,
        (model_max.y + model_min.y) / 2.0
    );
    let bed_mid = Point2::new(bed.width / 2.0, bed.depth / 2.0);
    Vector2::new(bed_mid.x - model_mid.x, bed_mid.y - model_mid.y)
}

impl Gcode {
    pub fn export(&mut self, plan: &PlannedResult) -> String {
        self.out = String::new();
        self.emit_header(plan);
        // For Bambu compatibility
        writeln!(self.out, "; EXECUTABLE_BLOCK_START").unwrap();
        self.emit_start();
        for layer in &plan.layers {
            let idx = layer.layer_index;
            let z = layer.z;
            // For Bambu compatibility
            writeln!(self.out, "; CHANGE_LAYER").unwrap();
            writeln!(self.out, "; Z_HEIGHT: {z:.3}").unwrap();
            writeln!(self.out, "; LAYER_HEIGHT: {:.3}", self.config.layer_height).unwrap();
            writeln!(self.out, "G1 Z{z:.3} F60").unwrap();
            // Each layer re-announces its first feature after the CHANGE_LAYER
            // marker, matching Bambu; tracked fresh per layer.
            let mut current_feature: Option<&'static str> = None;
            for mv in &layer.moves {
                self.emit_move(mv, idx, &mut current_feature);
            }
        }
        self.emit_end();
        writeln!(self.out, "; EXECUTABLE_BLOCK_END").unwrap();
        std::mem::take(&mut self.out)
    }

    fn emit_move(&mut self, mv: &Move, idx: usize, current_feature: &mut Option<&'static str>) {
        if mv.points.len() < 2 {
            println!("Empty move on layer {idx}!");
            return;
        }
        // Announce the feature only when it actually changes
        if let Some(label) = feature_label(mv.kind) {
            if *current_feature != Some(label) {
                writeln!(self.out, "; FEATURE: {label}").unwrap();
                *current_feature = Some(label);
            }
        }
        let mut cmd = String::new();
        let spd = mv.speed * 60.0;

        if mv.kind == MoveKind::Travel {
            let to = mv.points[1] + self.offset;
            writeln!(cmd, "G0 X{:.3} Y{:.3} F{spd:.0}", to.x, to.y).unwrap();
        } else {
            // Fit arcs over the simplified polyline, then emit each primitive.
            // The nozzle already sits at points[0] (a travel preceded this).
            let mut from = mv.points[0];
            for prim in arc::fit_arcs(&mv.points, ARC_FIT_TOLERANCE) {
                from = self.emit_primitive(&mut cmd, from, prim, spd);
            }
            if mv.kind == MoveKind::Perimeter {
                // Close the loop back to its start with a straight move.
                let to = mv.points[0];
                // Extrusion length is computed in model space; only the emitted
                // coordinate is shifted onto the bed.
                let e = self.calc_extrusion((to - from).norm());
                let to = to + self.offset;
                writeln!(cmd, "G1 X{:.3} Y{:.3} E{e:.5} F{spd:.0}", to.x, to.y).unwrap();
            }
        }
        write!(self.out, "{cmd}").unwrap();
    }

    /// Emit one fitted primitive, returning the nozzle's new position.
    fn emit_primitive(
        &mut self,
        cmd: &mut String,
        from: Point2<f32>,
        prim: PathPrimitive,
        spd: f32
    ) -> Point2<f32> {
        match prim {
            PathPrimitive::Line { to } => {
                let e = self.calc_extrusion((to - from).norm());
                let at = to + self.offset;
                writeln!(cmd, "G1 X{:.3} Y{:.3} E{e:.5} F{spd:.0}", at.x, at.y).unwrap();
                to
            }
            PathPrimitive::Arc { to, center, cw } => {
                // Extrusion follows the *arc length* (r * sweep), not the chord —
                // using the chord here would under-extrude every curve.
                let radius = (from - center).norm();
                let sweep = sweep_angle(from, to, center, cw);
                let e = self.calc_extrusion(radius * sweep);
                // I/J are the center offset relative to the start point. Emit them
                // with extra precision so the firmware's radius check agrees.
                // I/J are relative to the start point, so the bed offset
                // cancels out and they need no shifting.
                let i = center.x - from.x;
                let j = center.y - from.y;
                let g = if cw { "G2" } else { "G3" };
                let at = to + self.offset;
                writeln!(
                    cmd,
                    "{g} X{:.3} Y{:.3} I{i:.4} J{j:.4} E{e:.5} F{spd:.0}",
                    at.x,
                    at.y
                ).unwrap();
                to
            }
        }
    }

    fn calc_extrusion(&mut self, seg_len: f32) -> f32 {
        let bead_area = self.config.nozzle_width * self.config.layer_height;
        let filament_area =
            PI * (self.config.filament_diameter / 2.0) * (self.config.filament_diameter / 2.0);
        self.e += bead_area * (seg_len / filament_area);
        self.e
    }

    fn emit_header(&mut self, plan: &PlannedResult) {
        writeln!(self.out, "; HEADER_BLOCK_START").unwrap();
        writeln!(self.out, "; G-code generated with Katana slicer").unwrap();
        writeln!(self.out, "; Check out https://github.com/yagodorea/katana").unwrap();
        writeln!(self.out, "; total layer number: {}", plan.layers.len()).unwrap();
        writeln!(self.out, "; HEADER_BLOCK_END").unwrap();
        writeln!(self.out).unwrap();
    }

    fn emit_start(&mut self) {
        // TODO: drill filename into here to print metadata in comments
        writeln!(self.out, "; G-code generated with Katana slicer").unwrap();
        writeln!(self.out, "; Check out https://github.com/yagodorea/katana").unwrap();
        writeln!(self.out, ";").unwrap();
        writeln!(self.out, "; --- Starting sequence").unwrap();
        writeln!(self.out, "G28 ; home").unwrap();
        writeln!(self.out, "M104 S210 ; set nozzle temp (hardcoded to 210 for now)").unwrap();
        writeln!(self.out, "M140 S60 ; set bed temp (hardcoded to 60 for now)").unwrap();
        writeln!(self.out, "M109 S210 ; wait for nozzle temp").unwrap();
        writeln!(self.out, "M190 S60 ; wait for bed temp").unwrap();
        writeln!(self.out, "G92 E0 ; set absolute extruder offset to zero").unwrap();
        writeln!(self.out, "; --- Finish starting sequence").unwrap();
        writeln!(self.out).unwrap();
    }

    fn emit_end(&mut self) {
        writeln!(self.out, "; --- Ending sequence").unwrap();
        writeln!(self.out, "M104 S0 ; reset nozzle temp async").unwrap();
        writeln!(self.out, "M140 S0 ; reset bed temp async").unwrap();
        writeln!(self.out, "G92 Z0 ; redefine z").unwrap();
        writeln!(self.out, "G1 Z10 ; retract").unwrap();
        writeln!(self.out, "G28 ; home").unwrap();
        writeln!(self.out, "M84 ; disable steppers").unwrap();
        writeln!(self.out, "; --- Finish ending sequence").unwrap();
    }
}

/// Map an internal [`MoveKind`] to the Bambu Studio "FEATURE" label that the
/// slicer and printer firmware use to classify each extrusion run for the
/// Line-Type preview. Returns `None` for moves that should *not* start a new feature block
fn feature_label(kind: MoveKind) -> Option<&'static str> {
    match kind {
        MoveKind::Infill => Some("Internal solid infill"),
        MoveKind::Perimeter => Some("Outer wall"),
        MoveKind::SurfaceInfill => Some("Top surface"),
        MoveKind::Travel => None,
    }
}

/// Angle swept (radians, always positive) going from `from` to `to` around
/// `center` in the given direction. Used to turn an arc into its true length.
fn sweep_angle(from: Point2<f32>, to: Point2<f32>, center: Point2<f32>, cw: bool) -> f32 {
    use std::f32::consts::TAU;
    let a0 = (from.y - center.y).atan2(from.x - center.x);
    let a1 = (to.y - center.y).atan2(to.x - center.x);
    let mut sweep = if cw { a0 - a1 } else { a1 - a0 };
    while sweep < 0.0 {
        sweep += TAU;
    }
    while sweep >= TAU {
        sweep -= TAU;
    }
    sweep
}
