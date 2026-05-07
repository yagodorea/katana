//! katana-bench-runner: external, implementation-agnostic benchmark for katana-viewer.
//!
//! Measures two things:
//!   1. Init time  – parsed from the two lines katana-viewer prints before its GUI loop.
//!   2. Render FPS – counts unique frames (via pixel-hash) while orbit_camera() pans
//!                   the camera.  Runs in parallel with the orbit so no frame is missed.

use std::io::{ BufRead, BufReader };
use std::process::{ Child, Command, Stdio };
use std::sync::mpsc;
use std::thread;
use std::time::{ Duration, Instant };

use enigo::{ Button, Coordinate, Direction, Enigo, Mouse, Settings };
use screenshots::Screen;

// ─── CLI ────────────────────────────────────────────────────────────────────

fn usage() -> ! {
    eprintln!("Usage: katana-bench-runner --viewer <path> --stl <path>");
    eprintln!(
        "       katana-bench-runner --viewer <path> --stl <path> --warmup <secs> --measure <secs>"
    );
    std::process::exit(1);
}

struct Args {
    viewer: String,
    stl: String,
    warmup_secs: u64,
    measure_secs: u64,
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let get = |flag: &str| -> Option<String> {
        raw.windows(2)
            .find(|w| w[0] == flag)
            .map(|w| w[1].clone())
    };
    Args {
        viewer: get("--viewer").unwrap_or_else(|| usage()),
        stl: get("--stl").unwrap_or_else(|| usage()),
        warmup_secs: get("--warmup")
            .and_then(|s| s.parse().ok())
            .unwrap_or(3),
        measure_secs: get("--measure")
            .and_then(|s| s.parse().ok())
            .unwrap_or(10),
    }
}

// ─── Init timing ────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct InitTimings {
    load_ms: f64,
    slice_ms: f64,
    offset_ms: f64,
    plan_ms: f64,
}

/// Launch the viewer, then read its stdout on a background thread until we've
/// captured the two timing lines.  Returns the child process (still running).
fn launch_viewer(viewer: &str, stl: &str) -> (Child, InitTimings) {
    eprintln!("[bench] spawning: {viewer} {stl}");
    let mut child = Command::new(viewer)
        .arg(stl)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit()) // show viewer errors so failures are visible
        .spawn()
        .unwrap_or_else(|e| {
            eprintln!("Failed to launch {viewer}: {e}");
            std::process::exit(1)
        });

    let stdout = child.stdout.take().unwrap();
    let (tx, rx) = mpsc::channel::<InitTimings>();

    // Echoes every viewer stdout line so we can see exactly what it prints.
    // If the viewer exits early the pipe closes; the thread exits without
    // sending and recv_timeout below prints a clear diagnosis.
    thread::spawn(move || {
        let reader = BufReader::new(stdout);
        let mut t = InitTimings::default();
        for line in reader.lines().flatten() {
            eprintln!("[viewer] {line}");
            if let Some(ms) = extract_ms(&line, "in ") {
                t.load_ms = ms;
            }
            if line.starts_with("Sliced") {
                t.slice_ms = extract_ms(&line, "in ").unwrap_or(0.0);
                t.offset_ms = extract_ms_nth(&line, "in ", 1).unwrap_or(0.0);
                t.plan_ms = extract_ms_nth(&line, "in ", 2).unwrap_or(0.0);
                let _ = tx.send(t);
                return;
            }
        }
        eprintln!("[bench] viewer stdout closed without printing init timings");
    });

    let timings = rx.recv_timeout(Duration::from_secs(60)).unwrap_or_else(|_| {
        eprintln!(
            "[bench] timed out after 60 s — viewer likely crashed or printed unexpected output"
        );
        std::process::exit(1)
    });

    (child, timings)
}

fn extract_ms(line: &str, prefix: &str) -> Option<f64> {
    extract_ms_nth(line, prefix, 0)
}

fn extract_ms_nth(line: &str, prefix: &str, n: usize) -> Option<f64> {
    let mut count = 0;
    let mut rest = line;
    loop {
        let idx = rest.find(prefix)?;
        rest = &rest[idx + prefix.len()..];
        if count == n {
            let end = rest.find(|c: char| !c.is_ascii_digit() && c != '.').unwrap_or(rest.len());
            return rest[..end].parse().ok();
        }
        count += 1;
    }
}

// ─── Window detection ───────────────────────────────────────────────────────

// Uses CGWindowListCopyWindowInfo via a tiny Swift script piped to `swift -`.
// Swift ships with Xcode CLT (required for cargo on macOS) — zero extra installs.
// Searches by PID so it's immune to process-name quirks.

fn find_window_by_pid(pid: u32) -> Option<(i32, i32, u32, u32)> {
    use std::io::Write;
    let src = format!(
        r#"import CoreGraphics
let target = Int32({pid})
let opts = CGWindowListOption(rawValue:
    CGWindowListOption.optionOnScreenOnly.rawValue |
    CGWindowListOption.excludeDesktopElements.rawValue)
if let list = CGWindowListCopyWindowInfo(opts, kCGNullWindowID) as? [[String:Any]] {{
    for w in list {{
        guard let owner = w["kCGWindowOwnerPID"] as? Int32, owner == target,
              let b = w["kCGWindowBounds"] as? [String:Any],
              let x = b["X"] as? CGFloat, let y = b["Y"] as? CGFloat,
              let ww = b["Width"] as? CGFloat, let wh = b["Height"] as? CGFloat,
              ww > 0, wh > 0 else {{ continue }}
        print("\(Int(x)) \(Int(y)) \(Int(ww)) \(Int(wh))")
        exit(0)
    }}
}}
exit(1)
"#
    );
    let mut child = Command::new("swift")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    child.stdin.take()?.write_all(src.as_bytes()).ok()?;
    let out = child.wait_with_output().ok()?;
    if !out.status.success() { return None; }

    let s = String::from_utf8_lossy(&out.stdout);
    let nums: Vec<i32> = s.split_whitespace().filter_map(|n| n.parse().ok()).collect();
    if nums.len() != 4 { return None; }
    Some((nums[0], nums[1], nums[2].unsigned_abs(), nums[3].unsigned_abs()))
}

fn wait_for_window(pid: u32, timeout: Duration) -> (i32, i32, u32, u32) {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(b) = find_window_by_pid(pid) {
            return b;
        }
        if Instant::now() >= deadline {
            eprintln!("[bench] timed out waiting for window of pid {pid}");
            std::process::exit(1);
        }
        thread::sleep(Duration::from_millis(500));
    }
}

// ─── Window focus ───────────────────────────────────────────────────────────

/// Bring the named process's window to the front and wait for it to activate.
fn focus_window(process_name: &str) {
    let script = format!(
        "tell application \"System Events\" to set frontmost of process \"{process_name}\" to true"
    );
    let _ = Command::new("osascript").arg("-e").arg(&script).output();
    thread::sleep(Duration::from_millis(400));
}

// ─── Camera orbit ───────────────────────────────────────────────────────────

/// Animate a slow camera orbit across the viewer window for `duration`.
/// `cx/cy` is the window centre; `radius` is half the sweep width in pixels.
fn orbit_camera(enigo: &mut Enigo, cx: i32, cy: i32, radius: i32, duration: Duration) {
    println!("Start orbit camera");
    let step_px = 4;
    let step_delay = Duration::from_millis(12); // ~83 steps/s → smooth at 60 fps
    enigo.move_mouse(cx - radius, cy, Coordinate::Abs).unwrap();
    enigo.button(Button::Left, Direction::Press).unwrap();
    let start = Instant::now();
    let mut x = cx - radius;
    let mut dir = 1i32;
    while start.elapsed() < duration {
        x += dir * step_px;
        if x >= cx + radius || x <= cx - radius {
            dir = -dir;
        }
        enigo.move_mouse(x, cy, Coordinate::Abs).unwrap();
        thread::sleep(step_delay);
    }
    enigo.button(Button::Left, Direction::Release).unwrap();
}

// ─── FPS measurement ────────────────────────────────────────────────────────

/// Capture unique frames while orbiting; returns average FPS over `duration`.
fn measure_fps(bounds: (i32, i32, u32, u32), duration: Duration) -> f64 {
    let (wx, wy, ww, wh) = bounds;

    // Capture only the centre quarter of the window — faster and avoids UI chrome.
    let cap_x = wx + (ww as i32) / 4;
    let cap_y = wy + (wh as i32) / 4;
    let cap_w = ww / 2;
    let cap_h = wh / 2;

    // Orbit runs in its own thread so frame capture isn't gated on mouse moves.
    let orbit_duration = duration + Duration::from_millis(500);
    let orbit_thread = thread::spawn(move || {
        let mut enigo = Enigo::new(&Settings::default()).expect(
            "Failed to create Enigo — check Accessibility permission"
        );
        let cx = wx + (ww as i32) / 2;
        let cy = wy + (wh as i32) / 2;
        orbit_camera(&mut enigo, cx, cy, (ww as i32) / 3, orbit_duration);
    });

    let screens = Screen::all().expect("screenshots: failed to list screens");
    let screen = screens.into_iter().next().expect("No screens found");

    let mut prev_hash: u64 = 0;
    let mut unique = 0u64;
    let start = Instant::now();

    while start.elapsed() < duration {
        if let Ok(img) = screen.capture_area(cap_x, cap_y, cap_w, cap_h) {
            let h = hash_pixels(img.as_raw());
            if h != prev_hash {
                unique += 1;
                prev_hash = h;
            }
        }
        thread::sleep(Duration::from_millis(4)); // aim for ~250 Hz capture ceiling
    }

    let _ = orbit_thread.join();
    (unique as f64) / duration.as_secs_f64()
}

/// Fast non-cryptographic hash of raw RGBA pixels (FNV-1a 64-bit).
fn hash_pixels(pixels: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    // Sample every 64th byte for speed on large captures.
    for &b in pixels.iter().step_by(64) {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ─── Entry point ────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

    println!("katana-bench-runner");
    println!("  viewer : {}", args.viewer);
    println!("  stl    : {}", args.stl);

    // ── 1. Launch and time init ──────────────────────────────────────────
    let (mut child, t) = launch_viewer(&args.viewer, &args.stl);
    let total_init = t.load_ms + t.slice_ms + t.offset_ms + t.plan_ms;

    println!("\n[init]");
    println!("  load   : {:>7.1} ms", t.load_ms);
    println!("  slice  : {:>7.1} ms", t.slice_ms);
    println!("  offset : {:>7.1} ms", t.offset_ms);
    println!("  plan   : {:>7.1} ms", t.plan_ms);
    println!("  total  : {:>7.1} ms", total_init);

    // ── 2. Wait for window ───────────────────────────────────────────────
    println!("\nWaiting for window...");
    let bounds = wait_for_window(child.id(), Duration::from_secs(30));
    println!("found {}×{} at ({},{})", bounds.2, bounds.3, bounds.0, bounds.1);

    print!("Focusing window...");
    focus_window("katana-viewer"); // still try by process name; harmless if it fails
    println!(" done");

    let (wx, wy, ww, wh) = bounds;
    let cx = wx + (ww as i32) / 2;
    let cy = wy + (wh as i32) / 2;
    let radius = (ww as i32) / 3;
    println!("  orbit centre ({cx},{cy}), sweep x: {}..{}", cx - radius, cx + radius);

    // ── 3. Warm-up ───────────────────────────────────────────────────────
    println!("Warming up ({} s)...", args.warmup_secs);
    {
        let mut enigo = Enigo::new(&Settings::default()).unwrap();
        orbit_camera(&mut enigo, cx, cy, radius, Duration::from_secs(args.warmup_secs));
    }

    // ── 4. Measure FPS ───────────────────────────────────────────────────
    println!("Measuring FPS ({} s)...", args.measure_secs);
    let fps = measure_fps(bounds, Duration::from_secs(args.measure_secs));

    println!("\n[fps]");
    println!("  avg fps : {fps:.1}");
    println!("  avg ms  : {:.2}", 1000.0 / fps.max(0.001));

    // ── 5. Clean up ──────────────────────────────────────────────────────
    child.kill().ok();
    child.wait().ok();
}
