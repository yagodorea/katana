//! POD structs and upload helpers for the wgpu renderer.
//!
//! These structs are the Rust mirrors of the WGSL `VsIn` and `Uniforms`
//! declarations in `shaders/*.wgsl`. They MUST match the WGSL layout
//! byte-for-byte, `bytemuck::cast_slice` reinterprets the raw memory.

use bytemuck::{ Pod, Zeroable };
use wgpu::{ BufferUsages, Device, util::DeviceExt };

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FrameUniforms {
    pub mvp: [[f32; 4]; 4],
    pub light_dir: [f32; 4],
    pub clip_z_max: f32,
    pub clip_z_min: f32,
    pub half_height: f32,
    pub half_width: f32,
}
const _: () = assert!(std::mem::size_of::<FrameUniforms>() == 96);

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
    pub aabb_min_x: f32,
    pub aabb_min_y: f32,
    pub aabb_max_x: f32,
    pub aabb_max_y: f32,
}

pub struct InstancedBatch {
    pub buffer: wgpu::Buffer,
    pub layer_entries: Vec<LayerEntry>,
}

pub struct LineLayerEntry {
    pub layer_z: f32,
    pub first_vertex: u32,
    pub vertex_count: u32,
}

pub struct LineBatch {
    pub buffer: wgpu::Buffer,
    pub layer_entries: Vec<LineLayerEntry>,
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
    GpuBuffer { buffer, vertex_count }
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
        return LineBatch { buffer, layer_entries };
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
            });
            if !at_end {
                current_z = verts[i].pos[2];
                layer_start = i;
            }
        }
    }
    LineBatch { buffer, layer_entries }
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
    GpuBuffer { buffer, vertex_count }
}

pub fn upload_rhombus_batch(device: &Device, instances: &[RhombusInstance]) -> InstancedBatch {
    use wgpu::util::DeviceExt;

    let buffer = device.create_buffer_init(
        &(wgpu::util::BufferInitDescriptor {
            label: Some("rhombus_instance_buffer"),
            contents: bytemuck::cast_slice(instances),
            usage: BufferUsages::VERTEX,
        })
    );

    let mut layer_entries: Vec<LayerEntry> = Vec::new();
    if instances.is_empty() {
        return InstancedBatch { buffer, layer_entries };
    }

    // Find layer boundaries and compute AABB per layer
    let mut current_z = instances[0].start[2];
    let mut layer_start = 0 as usize;
    for i in 1..=instances.len() {
        let at_end = i == instances.len();
        let z_changed = !at_end && instances[i].start[2] != current_z;
        if at_end || z_changed {
            let aabb = compute_layer_aabb(&instances[layer_start..i]);
            layer_entries.push(LayerEntry {
                layer_z: current_z,
                instance_count: (i - layer_start) as i32,
                aabb_min_x: aabb.0,
                aabb_min_y: aabb.1,
                aabb_max_x: aabb.2,
                aabb_max_y: aabb.3,
            });
            if !at_end {
                current_z = instances[i].start[2];
                layer_start = i;
            }
        }
    }
    InstancedBatch { buffer, layer_entries }
}

fn compute_layer_aabb(slice: &[RhombusInstance]) -> (f32, f32, f32, f32) {
    let (mut mn_x, mut mn_y) = (f32::MAX, f32::MAX);
    let (mut mx_x, mut mx_y) = (f32::MIN, f32::MIN);
    for inst in slice {
        let [ax, ay, _] = inst.start;
        let dx = inst.dir[0];
        let dy = inst.dir[1];
        let len = inst.length;
        let half_w = 0.0_f32; // half_width is in FrameUniforms; conservative AABB is fine
        let bx = ax + dx * len;
        let by = ay + dy * len;
        mn_x = mn_x.min(ax.min(bx) - half_w);
        mn_y = mn_y.min(ay.min(by) - half_w);
        mx_x = mx_x.max(ax.max(bx) + half_w);
        mx_y = mx_y.max(ay.max(by) + half_w);
    }
    (mn_x, mn_y, mx_x, mx_y)
}
