use bytemuck::{Pod, Zeroable};
use egui_wgpu::{CallbackResources, CallbackTrait, ScreenDescriptor};
use wgpu::util::DeviceExt;

const POINT_SHADER: &str = r#"
struct ViewportUniform {
    world_min:   vec2<f32>,
    world_size:  vec2<f32>,
    screen_min:  vec2<f32>,
    screen_size: vec2<f32>,
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
    let screen_center = vec2<f32>(
        viewport.screen_min.x + normalized.x * viewport.screen_size.x,
        viewport.screen_min.y + (1.0 - normalized.y) * viewport.screen_size.y,
    );
    let screen_pos = screen_center + corner * size;
    // screen_min cancels in local so NDC is identical whether rendering to
    // the surface or to the offscreen texture of the same canvas dimensions.
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

const BLIT_SHADER: &str = r#"
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

struct V { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }

@vertex
fn vs(@builtin(vertex_index) i: u32) -> V {
    var p = array<vec2<f32>, 6>(
        vec2<f32>(-1., -1.), vec2<f32>(1., -1.), vec2<f32>(1.,  1.),
        vec2<f32>(-1., -1.), vec2<f32>(1.,  1.), vec2<f32>(-1., 1.)
    );
    var u = array<vec2<f32>, 6>(
        vec2<f32>(0., 1.), vec2<f32>(1., 1.), vec2<f32>(1., 0.),
        vec2<f32>(0., 1.), vec2<f32>(1., 0.), vec2<f32>(0., 0.)
    );
    return V(vec4<f32>(p[i], 0., 1.), u[i]);
}

@fragment
fn fs(in: V) -> @location(0) vec4<f32> {
    return textureSample(t, s, in.uv);
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

struct OffscreenTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    blit_bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

pub struct PointCloudPipeline {
    render_pipeline: wgpu::RenderPipeline,
    blit_pipeline: wgpu::RenderPipeline,
    blit_bgl: wgpu::BindGroupLayout,
    blit_sampler: wgpu::Sampler,
    quad_vertex_buffer: wgpu::Buffer,
    pub instance_buffer: wgpu::Buffer,
    pub uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pub point_count: u32,
    target_format: wgpu::TextureFormat,
    offscreen: Option<OffscreenTarget>,
    pub render_dirty: bool,
}

impl PointCloudPipeline {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let point_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("point cloud shader"),
            source: wgpu::ShaderSource::Wgsl(POINT_SHADER.into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("point cloud uniform"),
            size: std::mem::size_of::<ViewportUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let point_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            layout: &point_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let point_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("point cloud pipeline layout"),
            bind_group_layouts: &[Some(&point_bgl)],
            immediate_size: 0,
        });

        let quad_verts: [[f32; 2]; 6] = [
            [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5],
            [-0.5, -0.5], [0.5, 0.5],  [-0.5, 0.5],
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

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("point cloud render pipeline"),
            layout: Some(&point_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &point_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: 8,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        }],
                    },
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
            multisample: wgpu::MultisampleState { count: 1, ..Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &point_shader,
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

        // ── Blit pipeline ─────────────────────────────────────────────────────
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("point cloud blit shader"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });

        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit pipeline layout"),
            bind_group_layouts: &[Some(&blit_bgl)],
            immediate_size: 0,
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("point cloud blit pipeline"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState { count: 1, ..Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blit sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            render_pipeline,
            blit_pipeline,
            blit_bgl,
            blit_sampler,
            quad_vertex_buffer,
            instance_buffer,
            uniform_buffer,
            bind_group,
            point_count: 0,
            target_format,
            offscreen: None,
            render_dirty: false,
        }
    }

    fn ensure_offscreen(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.offscreen.as_ref().map(|o| o.width == width && o.height == height).unwrap_or(false) {
            return;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("point cloud offscreen"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("point cloud blit bind group"),
            layout: &self.blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.blit_sampler),
                },
            ],
        });
        self.offscreen = Some(OffscreenTarget { _texture: texture, view, blit_bind_group, width, height });
        self.render_dirty = true;
    }

    pub fn upload_points(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        points: &[GpuPoint],
    ) {
        const MAX_POINTS: usize = (256 * 1024 * 1024) / std::mem::size_of::<GpuPoint>();
        let points = if points.len() > MAX_POINTS { &points[..MAX_POINTS] } else { points };

        self.point_count = points.len() as u32;
        self.render_dirty = true;

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
    pub world_min: [f32; 2],
    pub world_size: [f32; 2],
    pub screen_min: [f32; 2],
    pub screen_size: [f32; 2],
    /// True when viewport or point data changed this frame — triggers offscreen re-render.
    pub render_dirty: bool,
}

impl CallbackTrait for PointCloudCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_descriptor: &ScreenDescriptor,
        egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let ppp = screen_descriptor.pixels_per_point;

        let Some(pipeline) = callback_resources.get_mut::<PointCloudPipeline>() else {
            return vec![];
        };

        let w = (self.screen_size[0] * ppp).round() as u32;
        let h = (self.screen_size[1] * ppp).round() as u32;
        if w == 0 || h == 0 {
            return vec![];
        }
        pipeline.ensure_offscreen(device, w, h);

        // Always update viewport uniform (cheap; needed if viewport changed).
        let uniform = ViewportUniform {
            world_min: self.world_min,
            world_size: self.world_size,
            screen_min: [self.screen_min[0] * ppp, self.screen_min[1] * ppp],
            screen_size: [self.screen_size[0] * ppp, self.screen_size[1] * ppp],
        };
        queue.write_buffer(&pipeline.uniform_buffer, 0, bytemuck::bytes_of(&uniform));

        if self.render_dirty || pipeline.render_dirty {
            if let Some(offscreen) = &pipeline.offscreen {
                let mut rp = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("point cloud offscreen pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &offscreen.view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                if pipeline.point_count > 0 {
                    rp.set_pipeline(&pipeline.render_pipeline);
                    rp.set_bind_group(0, &pipeline.bind_group, &[]);
                    rp.set_vertex_buffer(0, pipeline.quad_vertex_buffer.slice(..));
                    rp.set_vertex_buffer(1, pipeline.instance_buffer.slice(..));
                    rp.draw(0..6, 0..pipeline.point_count);
                }
            }
            pipeline.render_dirty = false;
        }

        vec![]
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
        let Some(offscreen) = &pipeline.offscreen else {
            return;
        };

        render_pass.set_pipeline(&pipeline.blit_pipeline);
        render_pass.set_bind_group(0, &offscreen.blit_bind_group, &[]);
        render_pass.draw(0..6, 0..1);
    }
}
