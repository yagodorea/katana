# Katana - Rust-based 3D slicer

This workspace contains 3 programs:
1. `katana-core` - Application that processes STLs and slices them
2. `katana-cli` - CLI interface for katana-core
3. `katana-viewer` - GUI application based on `eframe` (for the interface) and `glow` (for GPU rendering with OpenGL) to visualize the slices.

## Getting started

- `cargo build`
- `cargo run -p katana-viewer -- stls/liver.stl`

