use bytemuck::{Pod, Zeroable};
use egui_wgpu::{CallbackResources, CallbackTrait, ScreenDescriptor};
use wgpu::util::DeviceExt;

const SHADER: &str = r#"
struct ViewportUniform {
    world_min:   vec2<f32>,
    world_size:  vec2<f32>,
    screen_min:  vec2<f32>,  // canvas top-left in physical pixels
    screen_size: vec2<f32>,  // canvas size in physical pixels
}

@group(0) @binding(0)
var<uniform> viewport: ViewportUniform;

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(
    @location(0) corner:       vec2<f32>,
    @location(1) world_pos:    vec2<f32>,
    @location(2) packed_color: u32,
    @location(3) size:         f32,
) -> VertexOut {
    let normalized = (world_pos - viewport.world_min) / viewport.world_size;
    // Y is flipped: world Y increases upward, screen Y increases downward.
    let screen_center = vec2<f32>(
        viewport.screen_min.x + normalized.x * viewport.screen_size.x,
        viewport.screen_min.y + (1.0 - normalized.y) * viewport.screen_size.y,
    );
    let screen_pos = screen_center + corner * size;
    // egui sets the render-pass viewport to the canvas rect before calling paint(),
    // so NDC must be relative to that canvas rect (not the full surface).
    let local = screen_pos - viewport.screen_min;
    let ndc = vec2<f32>(
        local.x / viewport.screen_size.x * 2.0 - 1.0,
        1.0 - local.y / viewport.screen_size.y * 2.0,
    );

    let r = f32((packed_color)        & 0xFFu) / 255.0;
    let g = f32((packed_color >> 8u)  & 0xFFu) / 255.0;
    let b = f32((packed_color >> 16u) & 0xFFu) / 255.0;
    let a = f32((packed_color >> 24u) & 0xFFu) / 255.0;

    var out: VertexOut;
    out.clip_pos = vec4<f32>(ndc.x, ndc.y, 0.0, 1.0);
    out.color    = vec4<f32>(r, g, b, a);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuPoint {
    pub position: [f32; 2],
    pub color: u32,
    pub size: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ViewportUniform {
    world_min: [f32; 2],
    world_size: [f32; 2],
    screen_min: [f32; 2],
    screen_size: [f32; 2],
}

pub struct PointCloudPipeline {
    pipeline: wgpu::RenderPipeline,
    quad_vertex_buffer: wgpu::Buffer,
    pub instance_buffer: wgpu::Buffer,
    pub uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pub point_count: u32,
}

impl PointCloudPipeline {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("point cloud shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("point cloud uniform"),
            size: std::mem::size_of::<ViewportUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("point cloud bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("point cloud bind group"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("point cloud pipeline layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let quad_verts: [[f32; 2]; 6] = [
            [-0.5, -0.5],
            [0.5, -0.5],
            [0.5, 0.5],
            [-0.5, -0.5],
            [0.5, 0.5],
            [-0.5, 0.5],
        ];
        let quad_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad vertex buffer"),
            contents: bytemuck::cast_slice(&quad_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("point cloud instances"),
            size: 64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("point cloud render pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[
                    // slot 0 — one quad corner per vertex
                    wgpu::VertexBufferLayout {
                        array_stride: 8,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        }],
                    },
                    // slot 1 — one GpuPoint per instance
                    wgpu::VertexBufferLayout {
                        array_stride: 16,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 0,
                                shader_location: 1,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Uint32,
                                offset: 8,
                                shader_location: 2,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 12,
                                shader_location: 3,
                            },
                        ],
                    },
                ],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            quad_vertex_buffer,
            instance_buffer,
            uniform_buffer,
            bind_group,
            point_count: 0,
        }
    }

    pub fn upload_points(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        points: &[GpuPoint],
    ) {
        // wgpu hard limit is 256 MiB; clamp silently until viewport culling lands.
        const MAX_POINTS: usize = (256 * 1024 * 1024) / std::mem::size_of::<GpuPoint>();
        let points = if points.len() > MAX_POINTS {
            &points[..MAX_POINTS]
        } else {
            points
        };

        self.point_count = points.len() as u32;
        if points.is_empty() {
            return;
        }

        let data: &[u8] = bytemuck::cast_slice(points);
        let needed = data.len() as u64;

        if self.instance_buffer.size() < needed {
            self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("point cloud instances"),
                size: needed,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        queue.write_buffer(&self.instance_buffer, 0, data);
    }
}

pub struct PointCloudCallback {
    /// World bounding box of the visible canvas: [xmin, ymin].
    pub world_min: [f32; 2],
    /// World extent of the visible canvas: [width, height].
    pub world_size: [f32; 2],
    /// Canvas top-left in logical (egui point) coordinates.
    pub screen_min: [f32; 2],
    /// Canvas size in logical (egui point) coordinates.
    pub screen_size: [f32; 2],
}

impl CallbackTrait for PointCloudCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_descriptor: &ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let info = _device.adapter_info();
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(
            &format!(
                "Backend: {:?} | Device: {} | Driver: {}",
                info.backend, info.name, info.driver_info
            )
            .into(),
        );
        let ppp = screen_descriptor.pixels_per_point;
        let uniform = ViewportUniform {
            world_min: self.world_min,
            world_size: self.world_size,
            screen_min: [self.screen_min[0] * ppp, self.screen_min[1] * ppp],
            screen_size: [self.screen_size[0] * ppp, self.screen_size[1] * ppp],
        };

        if let Some(pipeline) = callback_resources.get::<PointCloudPipeline>() {
            queue.write_buffer(&pipeline.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        }

        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &CallbackResources,
    ) {
        let Some(pipeline) = callback_resources.get::<PointCloudPipeline>() else {
            return;
        };
        if pipeline.point_count == 0 {
            return;
        }

        render_pass.set_pipeline(&pipeline.pipeline);
        render_pass.set_bind_group(0, &pipeline.bind_group, &[]);
        render_pass.set_vertex_buffer(0, pipeline.quad_vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, pipeline.instance_buffer.slice(..));
        render_pass.draw(0..6, 0..pipeline.point_count);
    }
}
