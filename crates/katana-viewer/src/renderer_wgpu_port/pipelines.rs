//! Render pipeline builders for the three draw kinds: line, mesh, rhombus.
//!
//! Each builder is a pure function `(device, &bgl, color_format) -> RenderPipeline`.
//! Pipelines are created once at `Renderer::new` and reused every frame.
//!
//! All three pipelines share:
//!   - Bind group layout (`build_frame_bgl`): group 0, binding 0 = uniform buffer (FrameUniforms)
//!   - Depth target: `DEPTH_FORMAT`, depth-write enabled, `LessEqual` compare
//!   - Color blend: standard SrcAlpha / OneMinusSrcAlpha
//!   - Cull mode: None (render both sides; matches GL behavior)
//!   - Multisample: 1 (no MSAA for this port; Plan 04 will turn it on)

use std::mem::size_of;
use wgpu::*;

use super::buffers::{ LineVertex, MeshVertex, RhombusInstance };

/// Depth buffer format
pub const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth24Plus;

/// Bind group layout shared by line + mesh pipelines.
/// Binding 0 = `FrameUniforms` uniform buffer, visible to VS + FS.
pub fn build_frame_bgl(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(
        &(BindGroupLayoutDescriptor {
            label: Some("frame_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    )
}

/// Bind group layout for the rhombus pipeline.
/// binding 0: FrameUniforms (VS+FS), binding 1: VertexTable (VS), binding 2: Palette (VS).
pub fn build_rhombus_bgl(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(
        &(BindGroupLayoutDescriptor {
            label: Some("rhombus_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    )
}

fn color_target_for(format: TextureFormat, blend: bool) -> ColorTargetState {
    ColorTargetState {
        format,
        blend: if blend {
            Some(BlendState::ALPHA_BLENDING)
        } else {
            None
        },
        write_mask: ColorWrites::ALL,
    }
}

fn depth_state_for(depth_write: bool) -> DepthStencilState {
    DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: depth_write,
        depth_compare: CompareFunction::LessEqual,
        stencil: StencilState::default(),
        bias: DepthBiasState::default(),
    }
}

fn build_line_pipeline_inner(
    device: &Device,
    frame_bgl: &BindGroupLayout,
    color_format: TextureFormat,
    blend: bool,
    depth_write: bool
) -> RenderPipeline {
    // Load shader module
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("line_shader"),
        source: ShaderSource::Wgsl(include_str!("shaders/line.wgsl").into()),
    });

    // Build pipeline layout referencing frame_bgl
    let layout = device.create_pipeline_layout(
        &(PipelineLayoutDescriptor {
            label: Some("line_pipeline_layout"),
            bind_group_layouts: &[frame_bgl],
            push_constant_ranges: &[],
        })
    );

    // Declare vertex buffer layout for LineVertex
    const ATTRS: [VertexAttribute; 2] = vertex_attr_array![0 => Float32x3, 1 => Float32x4];
    let vbuf_layout = VertexBufferLayout {
        array_stride: size_of::<LineVertex>() as BufferAddress,
        step_mode: VertexStepMode::Vertex, // per-vertex, not per-instance
        attributes: &ATTRS,
    };

    // Build the render pipeline
    device.create_render_pipeline(
        &(RenderPipelineDescriptor {
            label: Some("line_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[vbuf_layout],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(color_target_for(color_format, blend))],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(depth_state_for(depth_write)),
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    )
}

pub fn build_line_opaque_pipeline(
    device: &Device,
    frame_bgl: &BindGroupLayout,
    color_format: TextureFormat
) -> RenderPipeline {
    build_line_pipeline_inner(device, frame_bgl, color_format, false, true)
}
pub fn build_line_transparent_pipeline(
    device: &Device,
    frame_bgl: &BindGroupLayout,
    color_format: TextureFormat
) -> RenderPipeline {
    build_line_pipeline_inner(device, frame_bgl, color_format, true, false)
}

fn build_mesh_pipeline_inner(
    device: &Device,
    frame_bgl: &BindGroupLayout,
    color_format: TextureFormat,
    blend: bool,
    depth_write: bool
) -> RenderPipeline {
    // Load shader module
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("mesh_shader"),
        source: ShaderSource::Wgsl(include_str!("shaders/mesh.wgsl").into()),
    });

    // Build pipeline layout referencing frame_bgl
    let layout = device.create_pipeline_layout(
        &(PipelineLayoutDescriptor {
            label: Some("mesh_pipeline_layout"),
            bind_group_layouts: &[frame_bgl],
            push_constant_ranges: &[],
        })
    );

    // Declare vertex buffer layout for MeshVertex
    const ATTRS: [VertexAttribute; 4] =
        vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4, 3 => Float32];
    let vbuf_layout = VertexBufferLayout {
        array_stride: size_of::<MeshVertex>() as BufferAddress,
        step_mode: VertexStepMode::Vertex, // per-vertex, not per-instance
        attributes: &ATTRS,
    };

    // Build the render pipeline
    device.create_render_pipeline(
        &(RenderPipelineDescriptor {
            label: Some("mesh_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[vbuf_layout],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(color_target_for(color_format, blend))],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(depth_state_for(depth_write)),
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    )
}

pub fn build_mesh_opaque_pipeline(
    device: &Device,
    frame_bgl: &BindGroupLayout,
    color_format: TextureFormat
) -> RenderPipeline {
    build_mesh_pipeline_inner(device, frame_bgl, color_format, false, true)
}
pub fn build_mesh_transparent_pipeline(
    device: &Device,
    frame_bgl: &BindGroupLayout,
    color_format: TextureFormat
) -> RenderPipeline {
    build_mesh_pipeline_inner(device, frame_bgl, color_format, true, false)
}

fn build_rhombus_pipeline_inner(
    device: &Device,
    rhombus_bgl: &BindGroupLayout,
    color_format: TextureFormat,
    blend: bool,
    depth_write: bool
) -> RenderPipeline {
    // Load shader module
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("rhombus_shader"),
        source: ShaderSource::Wgsl(include_str!("shaders/rhombus.wgsl").into()),
    });

    // Build pipeline layout referencing rhombus_bgl (frame uniform + vertex table)
    let layout = device.create_pipeline_layout(
        &(PipelineLayoutDescriptor {
            label: Some("rhombus_pipeline_layout"),
            bind_group_layouts: &[rhombus_bgl],
            push_constant_ranges: &[],
        })
    );

    // Declare vertex buffer layout for RhombusInstance (packed: 24 B stride)
    const ATTRS: [VertexAttribute; 4] =
        vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32, 3 => Uint32];
    let vbuf_layout = VertexBufferLayout {
        array_stride: size_of::<RhombusInstance>() as BufferAddress,
        step_mode: VertexStepMode::Instance, // per-instance
        attributes: &ATTRS,
    };

    // Build the render pipeline
    device.create_render_pipeline(
        &(RenderPipelineDescriptor {
            label: Some("rhombus_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[vbuf_layout],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(color_target_for(color_format, blend))],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(depth_state_for(depth_write)),
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    )
}

pub fn build_rhombus_opaque_pipeline(
    device: &Device,
    rhombus_bgl: &BindGroupLayout,
    color_format: TextureFormat
) -> RenderPipeline {
    build_rhombus_pipeline_inner(device, rhombus_bgl, color_format, false, true)
}
pub fn build_rhombus_transparent_pipeline(
    device: &Device,
    rhombus_bgl: &BindGroupLayout,
    color_format: TextureFormat
) -> RenderPipeline {
    build_rhombus_pipeline_inner(device, rhombus_bgl, color_format, true, false)
}

// ---------------------------------------------------------------------------
// Blit pipeline (offscreen color → egui's pass)
// ---------------------------------------------------------------------------
//
// Different from the three above: takes a sampled texture + sampler instead
// of a uniform buffer, has no vertex buffer (geometry generated from
// vertex_index), no depth test (egui's pass has no depth attachment), and
// targets the egui surface format (we set it equal to the offscreen format
// at startup, so a straight copy works).

/// Bind group layout for the blit: texture at binding 0, sampler at binding 1.
pub fn build_blit_bgl(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(
        &(BindGroupLayoutDescriptor {
            label: Some("blit_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    )
}

pub fn build_blit_pipeline(
    device: &Device,
    blit_bgl: &BindGroupLayout,
    color_format: TextureFormat
) -> RenderPipeline {
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("blit_shader"),
        source: ShaderSource::Wgsl(include_str!("shaders/blit.wgsl").into()),
    });

    let layout = device.create_pipeline_layout(
        &(PipelineLayoutDescriptor {
            label: Some("blit_pipeline_layout"),
            bind_group_layouts: &[blit_bgl],
            push_constant_ranges: &[],
        })
    );

    device.create_render_pipeline(
        &(RenderPipelineDescriptor {
            label: Some("blit_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[], // no vertex buffer; geometry from vertex_index
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                // No alpha blend — we own every pixel of the central panel rect.
                targets: &[
                    Some(ColorTargetState {
                        format: color_format,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                ],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None, // egui's pass has no depth attachment
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    )
}
