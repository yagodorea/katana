//! POD structs and upload helpers for the wgpu renderer.
//!
//! These structs are the Rust mirrors of the WGSL `VsIn` and `Uniforms`
//! declarations in `shaders/*.wgsl`. They MUST match the WGSL layout
//! byte-for-byte, `bytemuck::cast_slice` reinterprets the raw memory.

use bytemuck::{ Pod, Zeroable };
use wgpu::{ util::DeviceExt, BufferUsages, Device };

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FrameUniforms {
    pub mvp: [[f32; 4]; 4],
    pub light_dir: [f32; 4],
    pub clip_z_max: f32,
    pub clip_z_min: f32,
    pub half_height: f32,
    pub half_width: f32,
    // Scrubber highlight
    pub scrub_top_z: f32,
    pub scrub_dim: f32,
    // Mid-segment scrub smoothing
    pub scrub_partial_index: u32,
    pub scrub_partial_frac: f32,
}
const _: () = assert!(std::mem::size_of::<FrameUniforms>() == 112);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LineVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<LineVertex>() == 28);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MeshVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
    pub layer_z: f32,
}
const _: () = assert!(std::mem::size_of::<MeshVertex>() == 44);

/// Color palette for the rhombus pipeline
pub const COLOR_PALETTE: [[f32; 4]; 16] = [
    [0.91, 0.27, 0.38, 1.0], // Perimeter
    [0.27, 0.91, 0.38, 1.0], // Infill
    [0.9, 0.2, 0.7, 1.0], // SurfaceInfill
    [1.0, 0.8, 0.2, 0.4], // Travel
    [0.0, 0.0, 0.0, 0.0], // Reserved...
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PaletteUniforms {
    pub colors: [[f32; 4]; 16],
}
const _: () = assert!(std::mem::size_of::<PaletteUniforms>() == 256);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RhombusInstance {
    pub start: [f32; 3], // 12B
    pub dir: [f32; 2], //  8B, unit direction vector
    pub length: f32, //  4B, segment length; half_width lives in FrameUniforms
    pub color_flags: u32, //  4B, color_id in bits 0-7, kind flags in bits 8-15
}
const _: () = assert!(std::mem::size_of::<RhombusInstance>() == 28);

pub struct GpuBuffer {
    pub buffer: wgpu::Buffer,
    pub vertex_count: u32,
}

pub struct LayerEntry {
    pub layer_z: f32,
    pub instance_count: i32,
    /// Full print time of this layer
    pub time_total: f32,
}

pub struct InstancedBatch {
    pub buffer: wgpu::Buffer,
    pub layer_entries: Vec<LayerEntry>,
    /// Cumulative layer-time (s) at the end of each instance, ascending within each layer. Empty for non-scrubbed batches.
    pub instance_times: Vec<f32>,
    /// Cumulative layer-time (s) at the *start* of each instance (parallel to
    /// `instance_times`). Lets the scrubber truncate the in-progress segment.
    pub instance_start_times: Vec<f32>,
}

pub struct LineLayerEntry {
    pub layer_z: f32,
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub time_total: f32,
}

pub struct LineBatch {
    pub buffer: wgpu::Buffer,
    pub layer_entries: Vec<LineLayerEntry>,
    pub segment_times: Vec<f32>,
}

pub fn upload_lines(device: &Device, verts: &[LineVertex]) -> GpuBuffer {
    let buffer = device.create_buffer_init(
        &(wgpu::util::BufferInitDescriptor {
            label: Some("line_vbo"),
            contents: bytemuck::cast_slice(verts),
            usage: BufferUsages::VERTEX,
        })
    );
    let vertex_count = verts.len() as u32;
    GpuBuffer {
        buffer,
        vertex_count,
    }
}

pub fn upload_lines_batched(device: &Device, verts: &[LineVertex]) -> LineBatch {
    let buffer = device.create_buffer_init(
        &(wgpu::util::BufferInitDescriptor {
            label: Some("line_batch_vbo"),
            contents: bytemuck::cast_slice(verts),
            usage: BufferUsages::VERTEX,
        })
    );
    let mut layer_entries: Vec<LineLayerEntry> = Vec::new();
    if verts.is_empty() {
        return LineBatch {
            buffer,
            layer_entries,
            segment_times: Vec::new(),
        };
    }
    let mut current_z = verts[0].pos[2];
    let mut layer_start = 0usize;
    for i in 1..=verts.len() {
        let at_end = i == verts.len();
        let z_changed = !at_end && verts[i].pos[2] != current_z;
        if at_end || z_changed {
            layer_entries.push(LineLayerEntry {
                layer_z: current_z,
                first_vertex: layer_start as u32,
                vertex_count: (i - layer_start) as u32,
                time_total: 0.0, // not scrubbed (background slices)
            });
            if !at_end {
                current_z = verts[i].pos[2];
                layer_start = i;
            }
        }
    }
    LineBatch {
        buffer,
        layer_entries,
        segment_times: Vec::new(),
    }
}

/// Wrap pre-built line geometry + per-layer entries + per-segment cumulative times into a `LineBatch`
pub fn make_line_batch(
    device: &Device,
    verts: &[LineVertex],
    layer_entries: Vec<LineLayerEntry>,
    segment_times: Vec<f32>
) -> LineBatch {
    let buffer = device.create_buffer_init(
        &(wgpu::util::BufferInitDescriptor {
            label: Some("line_batch_timed_vbo"),
            contents: bytemuck::cast_slice(verts),
            usage: BufferUsages::VERTEX,
        })
    );
    LineBatch {
        buffer,
        layer_entries,
        segment_times,
    }
}

/// Wrap pre-built rhombus instances + per-layer entries + per-instance cumulative times into an `InstancedBatch`
pub fn make_instanced_batch(
    device: &Device,
    instances: &[RhombusInstance],
    layer_entries: Vec<LayerEntry>,
    instance_times: Vec<f32>,
    instance_start_times: Vec<f32>
) -> InstancedBatch {
    let buffer = device.create_buffer_init(
        &(wgpu::util::BufferInitDescriptor {
            label: Some("rhombus_instance_buffer"),
            contents: bytemuck::cast_slice(instances),
            usage: BufferUsages::VERTEX,
        })
    );
    InstancedBatch {
        buffer,
        layer_entries,
        instance_times,
        instance_start_times,
    }
}

pub fn upload_mesh(device: &Device, verts: &[MeshVertex]) -> GpuBuffer {
    let buffer = device.create_buffer_init(
        &(wgpu::util::BufferInitDescriptor {
            label: Some("mesh_vbo"),
            contents: bytemuck::cast_slice(verts),
            usage: BufferUsages::VERTEX,
        })
    );
    let vertex_count = verts.len() as u32;
    GpuBuffer {
        buffer,
        vertex_count,
    }
}


