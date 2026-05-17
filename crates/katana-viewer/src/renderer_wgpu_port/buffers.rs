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
    pub _pad: f32,
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RhombusInstance {
    pub start: [f32; 3],
    pub direction: [f32; 2],
    pub scale: [f32; 2], // (length, half_width)
    pub color: [f32; 4],
    pub layer_z: f32,
}
const _: () = assert!(std::mem::size_of::<RhombusInstance>() == 48);

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
    let mut current_z = instances[0].layer_z;
    let mut layer_start = 0 as usize;
    for i in 1..=instances.len() {
        let at_end = i == instances.len();
        let z_changed = !at_end && instances[i].layer_z != current_z;
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
                current_z = instances[i].layer_z;
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
        let [dx, dy] = inst.direction;
        let [len, half_w] = inst.scale;
        let bx = ax + dx * len;
        let by = ay + dy * len;
        mn_x = mn_x.min(ax.min(bx) - half_w);
        mn_y = mn_y.min(ay.min(by) - half_w);
        mx_x = mx_x.max(ax.max(bx) + half_w);
        mx_y = mx_y.max(ay.max(by) + half_w);
    }
    (mn_x, mn_y, mx_x, mx_y)
}
