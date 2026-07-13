use bytemuck::{Pod, Zeroable};
use egui_wgpu::{CallbackResources, CallbackTrait, ScreenDescriptor};

// Same viewport-normalize-to-NDC math as point_cloud.rs's POINT_SHADER, minus
// the per-instance billboard-quad logic — vertices here are literal world
// positions, drawn with VertexStepMode::Vertex (no instancing).
const VECTOR_SHADER: &str = r#"
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
    @location(0) world_pos:    vec2<f32>,
    @location(1) packed_color: u32,
) -> VertexOut {
    let normalized = (world_pos - viewport.world_min) / viewport.world_size;
    let screen_pos = vec2<f32>(
        viewport.screen_min.x + normalized.x * viewport.screen_size.x,
        viewport.screen_min.y + (1.0 - normalized.y) * viewport.screen_size.y,
    );
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
pub struct GpuVertex {
    pub position: [f32; 2],
    pub color: u32,
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

fn vertex_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: 12,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32,
                offset: 8,
                shader_location: 1,
            },
        ],
    }
}

pub struct VectorPipeline {
    fill_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    blit_pipeline: wgpu::RenderPipeline,
    blit_bgl: wgpu::BindGroupLayout,
    blit_sampler: wgpu::Sampler,
    pub fill_vertex_buffer: wgpu::Buffer,
    pub fill_index_buffer: wgpu::Buffer,
    pub line_vertex_buffer: wgpu::Buffer,
    pub uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pub fill_index_count: u32,
    pub line_vertex_count: u32,
    target_format: wgpu::TextureFormat,
    offscreen: Option<OffscreenTarget>,
    pub render_dirty: bool,
}

impl VectorPipeline {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vector shader"),
            source: wgpu::ShaderSource::Wgsl(VECTOR_SHADER.into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vector uniform"),
            size: std::mem::size_of::<ViewportUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vector bgl"),
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
            label: Some("vector bind group"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vector pipeline layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let make_pipeline = |topology: wgpu::PrimitiveTopology, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[vertex_buffer_layout()],
                },
                primitive: wgpu::PrimitiveState { topology, ..Default::default() },
                depth_stencil: None,
                multisample: wgpu::MultisampleState { count: 1, ..Default::default() },
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
            })
        };
        let fill_pipeline = make_pipeline(wgpu::PrimitiveTopology::TriangleList, "vector fill pipeline");
        let line_pipeline = make_pipeline(wgpu::PrimitiveTopology::LineList, "vector line pipeline");

        // ── Blit pipeline — identical to point_cloud.rs's ───────────────────────
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vector blit shader"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });

        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vector blit bgl"),
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
            label: Some("vector blit pipeline layout"),
            bind_group_layouts: &[Some(&blit_bgl)],
            immediate_size: 0,
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vector blit pipeline"),
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
            label: Some("vector blit sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let fill_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vector fill vertices"),
            size: 64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let fill_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vector fill indices"),
            size: 64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let line_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vector line vertices"),
            size: 64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            fill_pipeline,
            line_pipeline,
            blit_pipeline,
            blit_bgl,
            blit_sampler,
            fill_vertex_buffer,
            fill_index_buffer,
            line_vertex_buffer,
            uniform_buffer,
            bind_group,
            fill_index_count: 0,
            line_vertex_count: 0,
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
            label: Some("vector offscreen"),
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
            label: Some("vector blit bind group"),
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

    /// Uploads the fully-flattened fill/line buffers built by
    /// `gpu_collect::collect_gpu_vector_mesh`. Grows GPU buffers on demand,
    /// same resize-on-demand strategy as `PointCloudPipeline::upload_points`.
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        fill_verts: &[GpuVertex],
        fill_indices: &[u32],
        line_verts: &[GpuVertex],
    ) {
        self.render_dirty = true;
        self.fill_index_count = fill_indices.len() as u32;
        self.line_vertex_count = line_verts.len() as u32;

        if !fill_verts.is_empty() {
            let data: &[u8] = bytemuck::cast_slice(fill_verts);
            if self.fill_vertex_buffer.size() < data.len() as u64 {
                self.fill_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("vector fill vertices"),
                    size: data.len() as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            queue.write_buffer(&self.fill_vertex_buffer, 0, data);
        }
        if !fill_indices.is_empty() {
            let data: &[u8] = bytemuck::cast_slice(fill_indices);
            if self.fill_index_buffer.size() < data.len() as u64 {
                self.fill_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("vector fill indices"),
                    size: data.len() as u64,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            queue.write_buffer(&self.fill_index_buffer, 0, data);
        }
        if !line_verts.is_empty() {
            let data: &[u8] = bytemuck::cast_slice(line_verts);
            if self.line_vertex_buffer.size() < data.len() as u64 {
                self.line_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("vector line vertices"),
                    size: data.len() as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            queue.write_buffer(&self.line_vertex_buffer, 0, data);
        }
    }
}

pub struct VectorCallback {
    pub world_min: [f32; 2],
    pub world_size: [f32; 2],
    pub screen_min: [f32; 2],
    pub screen_size: [f32; 2],
    pub render_dirty: bool,
}

impl CallbackTrait for VectorCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_descriptor: &ScreenDescriptor,
        egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let ppp = screen_descriptor.pixels_per_point;

        let Some(pipeline) = callback_resources.get_mut::<VectorPipeline>() else {
            return vec![];
        };

        let w = (self.screen_size[0] * ppp).round() as u32;
        let h = (self.screen_size[1] * ppp).round() as u32;
        if w == 0 || h == 0 {
            return vec![];
        }
        pipeline.ensure_offscreen(device, w, h);

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
                    label: Some("vector offscreen pass"),
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

                // Fill first, outlines on top -- matches the CPU draw order.
                if pipeline.fill_index_count > 0 {
                    rp.set_pipeline(&pipeline.fill_pipeline);
                    rp.set_bind_group(0, &pipeline.bind_group, &[]);
                    rp.set_vertex_buffer(0, pipeline.fill_vertex_buffer.slice(..));
                    rp.set_index_buffer(pipeline.fill_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    rp.draw_indexed(0..pipeline.fill_index_count, 0, 0..1);
                }
                if pipeline.line_vertex_count > 0 {
                    rp.set_pipeline(&pipeline.line_pipeline);
                    rp.set_bind_group(0, &pipeline.bind_group, &[]);
                    rp.set_vertex_buffer(0, pipeline.line_vertex_buffer.slice(..));
                    rp.draw(0..pipeline.line_vertex_count, 0..1);
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
        let Some(pipeline) = callback_resources.get::<VectorPipeline>() else {
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
