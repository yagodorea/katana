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

use crate::planner::{ Move, MoveKind, PlannedResult };

pub struct GcodeConfig {
    pub filament_diameter: f32,
    pub nozzle_width: f32,
    pub layer_height: f32,
}

pub struct Gcode {
    pub config: GcodeConfig,
    pub e: f32, // cumulative absolute-E total
    pub out: String,
}

impl Gcode {
    pub fn export(&mut self, plan: &PlannedResult) -> String {
        self.out = String::new();
        self.emit_start();
        for layer in &plan.layers {
            let idx = layer.layer_index;
            let z = layer.z;
            writeln!(self.out, "; LAYER {idx}").unwrap();
            writeln!(self.out, "G1 Z{z:.3} F60").unwrap();
            for mv in &layer.moves {
                self.emit_move(mv, idx);
            }
        }
        self.emit_end();
        std::mem::take(&mut self.out)
    }

    fn emit_move(&mut self, mv: &Move, idx: usize) {
        if mv.points.len() < 2 {
            println!("Empty move on layer {idx}!");
            return;
        }
        let mut cmd = String::new();
        let mut x1 = mv.points[1].x;
        let mut y1 = mv.points[1].y;
        let spd = mv.speed * 60.0;
        if mv.kind == MoveKind::Travel {
            writeln!(cmd, "G0 X{x1:.3} Y{y1:.3} F{spd:.0}").unwrap();
        } else {
            let mut x0 = mv.points[0].x;
            let mut y0 = mv.points[0].y;
            for i in 1..mv.points.len() {
                x1 = mv.points[i].x;
                y1 = mv.points[i].y;

                let dx = x1 - x0;
                let dy = y1 - y0;
                let seg_len = (dx * dx + dy * dy).sqrt();
                let e = self.calc_extrusion(seg_len);
                writeln!(cmd, "G1 X{x1:.3} Y{y1:.3} E{e:.5} F{spd:.0}").unwrap();

                // Advance
                x0 = mv.points[i].x;
                y0 = mv.points[i].y;
            }
            if mv.kind == MoveKind::Perimeter {
                // Closing movement is implicit, need to emit
                x1 = mv.points[0].x;
                y1 = mv.points[0].y;

                let dx = x1 - x0;
                let dy = y1 - y0;
                let seg_len = (dx * dx + dy * dy).sqrt();
                let e = self.calc_extrusion(seg_len);
                writeln!(cmd, "G1 X{x1:.3} Y{y1:.3} E{e:.5} F{spd:.0}").unwrap();
            }
        }
        write!(self.out, "{cmd}").unwrap();
    }

    fn calc_extrusion(&mut self, seg_len: f32) -> f32 {
        let bead_area = self.config.nozzle_width * self.config.layer_height;
        let filament_area =
            PI * (self.config.filament_diameter / 2.0) * (self.config.filament_diameter / 2.0);
        self.e += bead_area * (seg_len / filament_area);
        self.e
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
