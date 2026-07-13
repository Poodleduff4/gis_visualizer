/// Globe renderer: shaded UV sphere + 3D billboard points.
/// EPSG:4326 lon/lat → unit-sphere Cartesian, rendered with an orbital camera.
use bytemuck::{Pod, Zeroable};
use egui::Color32;
use egui_wgpu::{CallbackResources, CallbackTrait, ScreenDescriptor};
use wgpu::util::DeviceExt;

use crate::gis_layer::{bake_raster_rgba, LayerEntry, LayerKind, RasterData};

const EARTH_PNG: &[u8] = include_bytes!("../assets/earth.png");

// ── Shaders ───────────────────────────────────────────────────────────────────

const SPHERE_SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    eye_dir:   vec3<f32>,
    _pad:      f32,
}
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var earth_tex: texture_2d<f32>;
@group(0) @binding(2) var earth_smp: sampler;
@group(0) @binding(3) var raster_tex: texture_2d<f32>;
@group(0) @binding(4) var raster_smp: sampler;

struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) n:          vec3<f32>,
    @location(1) uv:         vec2<f32>,
}

@vertex fn vs(
    @location(0) pos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) uv:  vec2<f32>,
) -> Out {
    var o: Out;
    o.clip = camera.view_proj * vec4<f32>(pos, 1.0);
    o.n  = nrm;
    o.uv = uv;
    return o;
}

@fragment fn fs(in: Out) -> @location(0) vec4<f32> {
    let earth_col  = textureSample(earth_tex, earth_smp, in.uv);
    let raster_col = textureSample(raster_tex, raster_smp, in.uv);
    let rgb = mix(earth_col.rgb, raster_col.rgb, raster_col.a);
    return vec4<f32>(rgb, 1.0);
}
"#;

const POINT_SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    eye_dir:   vec3<f32>,
    _pad:      f32,
}
struct Screen { size: vec2<f32>, _pad: vec2<f32> }

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> screen: Screen;

struct Out { @builtin(position) clip: vec4<f32>, @location(0) color: vec4<f32> }

@vertex fn vs(
    @location(0) corner:       vec2<f32>,
    @location(1) world_pos:    vec3<f32>,
    @location(2) packed_color: u32,
    @location(3) size:         f32,
) -> Out {
    let vis = dot(normalize(world_pos), normalize(camera.eye_dir));
    var clip = camera.view_proj * vec4<f32>(world_pos, 1.0);
    if vis < 0.02 {
        clip = vec4<f32>(0.0, 0.0, 10.0, 1.0);
    } else {
        let off = corner * (size * 2.0 / screen.size) * clip.w;
        clip = vec4<f32>(clip.xy + off, clip.z, clip.w);
    }
    let r = f32(packed_color        & 0xFFu) / 255.0;
    let g = f32((packed_color >> 8u)  & 0xFFu) / 255.0;
    let b = f32((packed_color >> 16u) & 0xFFu) / 255.0;
    let a = f32((packed_color >> 24u) & 0xFFu) / 255.0;
    var o: Out; o.clip = clip; o.color = vec4<f32>(r, g, b, a); return o;
}

@fragment fn fs(in: Out) -> @location(0) vec4<f32> { return in.color; }
"#;

const BLIT_SHADER: &str = r#"
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
struct V { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) i: u32) -> V {
    var p = array<vec2<f32>,6>(vec2(-1.,-1.),vec2(1.,-1.),vec2(1.,1.),vec2(-1.,-1.),vec2(1.,1.),vec2(-1.,1.));
    var u = array<vec2<f32>,6>(vec2(0.,1.),vec2(1.,1.),vec2(1.,0.),vec2(0.,1.),vec2(1.,0.),vec2(0.,0.));
    return V(vec4<f32>(p[i],0.,1.), u[i]);
}
@fragment fn fs(in: V) -> @location(0) vec4<f32> { return textureSample(t, s, in.uv); }
"#;

// ── GPU types ─────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GlobeCameraUniform {
    view_proj: [[f32; 4]; 4],
    eye_dir:   [f32; 3],
    _pad:      f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ScreenUniform {
    size: [f32; 2],
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GlobePoint {
    pub position: [f32; 3],
    pub color:    u32,
    pub size:     f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SphereVert {
    pos: [f32; 3],
    nrm: [f32; 3],
    uv:  [f32; 2],
}

// ── Orbital camera ────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct GlobeCamera {
    pub yaw:    f32,
    pub pitch:  f32,
    pub radius: f32,
}

impl Default for GlobeCamera {
    fn default() -> Self {
        Self { yaw: 0.0, pitch: 0.3, radius: 2.5 }
    }
}

impl GlobeCamera {
    fn eye(&self) -> [f32; 3] {
        let cp = self.pitch.cos();
        [
            self.radius * cp * self.yaw.sin(),
            self.radius * self.pitch.sin(),
            self.radius * cp * self.yaw.cos(),
        ]
    }

    fn eye_dir(&self) -> [f32; 3] {
        use glam::Vec3;
        Vec3::from_array(self.eye()).normalize().to_array()
    }

    fn view_proj(&self, aspect: f32) -> [[f32; 4]; 4] {
        use glam::{Mat4, Vec3};
        let eye = Vec3::from_array(self.eye());
        let view = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(45_f32.to_radians(), aspect, 0.05, 20.0);
        (proj * view).to_cols_array_2d()
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw  -= dx * 0.005;
        // positive dy = drag down = camera moves up = more north visible
        self.pitch = (self.pitch + dy * 0.005).clamp(-1.4, 1.4);
    }

    pub fn zoom(&mut self, delta: f32) {
        self.radius = (self.radius - delta * 0.2).clamp(1.15, 8.0);
    }
}

// ── Offscreen target ──────────────────────────────────────────────────────────

struct Offscreen {
    _tex:            wgpu::Texture,
    view:            wgpu::TextureView,
    blit_bind_group: wgpu::BindGroup,
    width:           u32,
    height:          u32,
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

pub struct GlobePipeline {
    // Sphere
    sphere_vbuf:       wgpu::Buffer,
    sphere_ibuf:       wgpu::Buffer,
    sphere_idx_count:  u32,
    sphere_pipeline:   wgpu::RenderPipeline,
    sphere_bind_group: wgpu::BindGroup,

    // Globe points
    quad_vbuf:        wgpu::Buffer,
    point_buf:        wgpu::Buffer,
    pub point_count:  u32,
    point_pipeline:   wgpu::RenderPipeline,
    point_bind_group: wgpu::BindGroup,

    // Shared uniforms
    camera_buf: wgpu::Buffer,
    screen_buf: wgpu::Buffer,

    // Earth texture (keep alive)
    _earth_tex:    wgpu::Texture,
    earth_view:    wgpu::TextureView,
    earth_sampler: wgpu::Sampler,

    // Raster overlay texture
    sphere_bgl:     wgpu::BindGroupLayout,
    raster_tex:     wgpu::Texture,
    raster_view:    wgpu::TextureView,
    raster_sampler: wgpu::Sampler,
    raster_dims:    (u32, u32),

    // Offscreen + blit
    target_format:  wgpu::TextureFormat,
    offscreen:      Option<Offscreen>,
    blit_pipeline:  wgpu::RenderPipeline,
    blit_bgl:       wgpu::BindGroupLayout,
    blit_sampler:   wgpu::Sampler,

    pub render_dirty: bool,
}

impl GlobePipeline {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, target_format: wgpu::TextureFormat) -> Self {
        // ── Earth texture ─────────────────────────────────────────────────────
        let img = image::load_from_memory(EARTH_PNG)
            .expect("embedded earth.png valid")
            .to_rgba8();
        let (tex_w, tex_h) = img.dimensions();
        let earth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("earth texture"),
            size:            wgpu::Extent3d { width: tex_w, height: tex_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Rgba8Unorm,
            usage:           wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats:    &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture:   &earth_tex,
                mip_level: 0,
                origin:    wgpu::Origin3d::ZERO,
                aspect:    wgpu::TextureAspect::All,
            },
            img.as_raw(),
            wgpu::TexelCopyBufferLayout {
                offset:         0,
                bytes_per_row:  Some(4 * tex_w),
                rows_per_image: Some(tex_h),
            },
            wgpu::Extent3d { width: tex_w, height: tex_h, depth_or_array_layers: 1 },
        );
        let earth_view = earth_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let earth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:            Some("earth sampler"),
            address_mode_u:   wgpu::AddressMode::Repeat,
            address_mode_v:   wgpu::AddressMode::ClampToEdge,
            address_mode_w:   wgpu::AddressMode::ClampToEdge,
            mag_filter:       wgpu::FilterMode::Linear,
            min_filter:       wgpu::FilterMode::Linear,
            mipmap_filter:    wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // ── Sphere mesh ───────────────────────────────────────────────────────
        let (verts, indices, sphere_idx_count) = generate_sphere(64, 32);
        let sphere_vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("globe sphere verts"),
            contents: bytemuck::cast_slice(&verts),
            usage:    wgpu::BufferUsages::VERTEX,
        });
        let sphere_ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("globe sphere idx"),
            contents: bytemuck::cast_slice(&indices),
            usage:    wgpu::BufferUsages::INDEX,
        });

        // ── Shared uniforms ───────────────────────────────────────────────────
        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("globe camera uniform"),
            size:               std::mem::size_of::<GlobeCameraUniform>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let screen_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("globe screen uniform"),
            size:               std::mem::size_of::<ScreenUniform>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Raster overlay placeholder (1x1 transparent) ───────────────────────
        let raster_tex = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("raster texture"),
            size:            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Rgba8Unorm,
            usage:           wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats:    &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture:   &raster_tex,
                mip_level: 0,
                origin:    wgpu::Origin3d::ZERO,
                aspect:    wgpu::TextureAspect::All,
            },
            &[0u8, 0, 0, 0],
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let raster_view = raster_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let raster_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:            Some("raster sampler"),
            address_mode_u:   wgpu::AddressMode::Repeat,
            address_mode_v:   wgpu::AddressMode::ClampToEdge,
            address_mode_w:   wgpu::AddressMode::ClampToEdge,
            mag_filter:       wgpu::FilterMode::Linear,
            min_filter:       wgpu::FilterMode::Linear,
            mipmap_filter:    wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // ── Sphere BGL: camera + earth tex/sampler + raster tex/sampler ────────
        let sphere_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("globe sphere bgl"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty:         wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled:   false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty:         wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count:      None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty:         wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled:   false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty:         wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count:      None,
                },
            ],
        });
        let sphere_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("globe sphere bg"),
            layout:  &sphere_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&earth_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&earth_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&raster_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&raster_sampler) },
            ],
        });

        // ── Point BGL + bind group (camera + screen) ──────────────────────────
        let point_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("globe point bgl"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX),
                uniform_entry(1, wgpu::ShaderStages::VERTEX),
            ],
        });
        let point_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("globe point bg"),
            layout:  &point_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: screen_buf.as_entire_binding() },
            ],
        });

        // ── Sphere pipeline ───────────────────────────────────────────────────
        let sphere_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("sphere shader"),
            source: wgpu::ShaderSource::Wgsl(SPHERE_SHADER.into()),
        });
        let sphere_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("sphere layout"),
            bind_group_layouts:   &[Some(&sphere_bgl)],
            immediate_size:       0,
        });
        let sphere_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("sphere pipeline"),
            layout: Some(&sphere_layout),
            vertex: wgpu::VertexState {
                module:              &sphere_shader,
                entry_point:         Some("vs"),
                compilation_options: Default::default(),
                buffers:             &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<SphereVert>() as u64,
                    step_mode:    wgpu::VertexStepMode::Vertex,
                    attributes:   &[
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0,  shader_location: 0 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 24, shader_location: 2 },
                    ],
                }],
            },
            primitive: wgpu::PrimitiveState {
                topology:   wgpu::PrimitiveTopology::TriangleList,
                cull_mode:  Some(wgpu::Face::Back),
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil:  None,
            multisample:    wgpu::MultisampleState { count: 1, ..Default::default() },
            fragment: Some(wgpu::FragmentState {
                module:              &sphere_shader,
                entry_point:         Some("fs"),
                compilation_options: Default::default(),
                targets:             &[Some(wgpu::ColorTargetState {
                    format:     target_format,
                    blend:      None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache:          None,
        });

        // ── Globe point pipeline ──────────────────────────────────────────────
        let point_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("globe point shader"),
            source: wgpu::ShaderSource::Wgsl(POINT_SHADER.into()),
        });
        let point_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("globe point layout"),
            bind_group_layouts: &[Some(&point_bgl)],
            immediate_size:     0,
        });
        let quad_verts: [[f32; 2]; 6] = [
            [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5],
            [-0.5, -0.5], [0.5,  0.5], [-0.5, 0.5],
        ];
        let quad_vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("globe quad verts"),
            contents: bytemuck::cast_slice(&quad_verts),
            usage:    wgpu::BufferUsages::VERTEX,
        });
        let point_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("globe point instances"),
            size:               64,
            usage:              wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let point_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("globe point pipeline"),
            layout: Some(&point_layout),
            vertex: wgpu::VertexState {
                module:              &point_shader,
                entry_point:         Some("vs"),
                compilation_options: Default::default(),
                buffers:             &[
                    wgpu::VertexBufferLayout {
                        array_stride: 8,
                        step_mode:    wgpu::VertexStepMode::Vertex,
                        attributes:   &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0,
                        }],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<GlobePoint>() as u64,
                        step_mode:    wgpu::VertexStepMode::Instance,
                        attributes:   &[
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0,  shader_location: 1 },
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32,    offset: 12, shader_location: 2 },
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32,   offset: 16, shader_location: 3 },
                        ],
                    },
                ],
            },
            primitive:      wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil:  None,
            multisample:    wgpu::MultisampleState { count: 1, ..Default::default() },
            fragment: Some(wgpu::FragmentState {
                module:              &point_shader,
                entry_point:         Some("fs"),
                compilation_options: Default::default(),
                targets:             &[Some(wgpu::ColorTargetState {
                    format:     target_format,
                    blend:      None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache:          None,
        });

        // ── Blit pipeline (offscreen → surface) ───────────────────────────────
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("globe blit shader"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("globe blit bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty:         wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled:   false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty:         wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count:      None,
                },
            ],
        });
        let blit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("globe blit layout"),
            bind_group_layouts: &[Some(&blit_bgl)],
            immediate_size:     0,
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("globe blit pipeline"),
            layout: Some(&blit_layout),
            vertex: wgpu::VertexState {
                module:              &blit_shader,
                entry_point:         Some("vs"),
                compilation_options: Default::default(),
                buffers:             &[],
            },
            primitive:      wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil:  None,
            multisample:    wgpu::MultisampleState { count: 1, ..Default::default() },
            fragment: Some(wgpu::FragmentState {
                module:              &blit_shader,
                entry_point:         Some("fs"),
                compilation_options: Default::default(),
                targets:             &[Some(wgpu::ColorTargetState {
                    format:     target_format,
                    blend:      Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache:          None,
        });
        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:      Some("globe blit sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            sphere_vbuf, sphere_ibuf, sphere_idx_count,
            sphere_pipeline, sphere_bind_group,
            quad_vbuf, point_buf, point_count: 0,
            point_pipeline, point_bind_group,
            camera_buf, screen_buf,
            _earth_tex: earth_tex, earth_view, earth_sampler,
            sphere_bgl, raster_tex, raster_view, raster_sampler, raster_dims: (1, 1),
            target_format, offscreen: None,
            blit_pipeline, blit_bgl, blit_sampler,
            render_dirty: false,
        }
    }

    fn ensure_offscreen(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.offscreen.as_ref().map(|o| o.width == width && o.height == height).unwrap_or(false) {
            return;
        }
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label:               Some("globe offscreen"),
            size:                wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count:     1,
            sample_count:        1,
            dimension:           wgpu::TextureDimension::D2,
            format:              self.target_format,
            usage:               wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats:        &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("globe blit bg"),
            layout:  &self.blit_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.blit_sampler) },
            ],
        });
        self.offscreen = Some(Offscreen { _tex: tex, view, blit_bind_group, width, height });
        self.render_dirty = true;
    }

    pub fn upload_points(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, points: &[GlobePoint]) {
        const MAX: usize = (256 * 1024 * 1024) / std::mem::size_of::<GlobePoint>();
        let points = if points.len() > MAX { &points[..MAX] } else { points };
        self.point_count = points.len() as u32;
        self.render_dirty = true;
        if points.is_empty() { return; }
        let data: &[u8] = bytemuck::cast_slice(points);
        if self.point_buf.size() < data.len() as u64 {
            self.point_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("globe point instances"),
                size:               data.len() as u64,
                usage:              wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        queue.write_buffer(&self.point_buf, 0, data);
    }

    /// Bake `data`'s grid into the raster overlay texture via a blue→red ramp
    /// over [display_min, display_max]; `None` clears the overlay.
    pub fn update_raster(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: Option<&RasterData>) {
        let (w, h, rgba) = match data {
            Some(d) => (d.width as u32, d.height as u32, bake_raster_rgba(d)),
            None => (1, 1, vec![0u8; 4]),
        };

        if self.raster_dims != (w, h) {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label:           Some("raster texture"),
                size:            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count:    1,
                dimension:       wgpu::TextureDimension::D2,
                format:          wgpu::TextureFormat::Rgba8Unorm,
                usage:           wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats:    &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.sphere_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   Some("globe sphere bg"),
                layout:  &self.sphere_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.camera_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.earth_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.earth_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.raster_sampler) },
                ],
            });
            self.raster_tex = tex;
            self.raster_view = view;
            self.raster_dims = (w, h);
        }

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture:   &self.raster_tex,
                mip_level: 0,
                origin:    wgpu::Origin3d::ZERO,
                aspect:    wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * w), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.render_dirty = true;
    }
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty:                 wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size:   None,
        },
        count: None,
    }
}

// ── Callback ──────────────────────────────────────────────────────────────────

pub struct GlobeCallback {
    pub camera:       GlobeCamera,
    pub screen_size:  [f32; 2],
    pub render_dirty: bool,
}

impl CallbackTrait for GlobeCallback {
    fn prepare(
        &self,
        device:    &wgpu::Device,
        queue:     &wgpu::Queue,
        screen:    &ScreenDescriptor,
        encoder:   &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let ppp = screen.pixels_per_point;
        let Some(pipeline) = resources.get_mut::<GlobePipeline>() else { return vec![]; };

        let w = (self.screen_size[0] * ppp).round() as u32;
        let h = (self.screen_size[1] * ppp).round() as u32;
        if w == 0 || h == 0 { return vec![]; }
        pipeline.ensure_offscreen(device, w, h);

        let aspect = w as f32 / h as f32;
        let cam_uniform = GlobeCameraUniform {
            view_proj: self.camera.view_proj(aspect),
            eye_dir:   self.camera.eye_dir(),
            _pad:      0.0,
        };
        queue.write_buffer(&pipeline.camera_buf, 0, bytemuck::bytes_of(&cam_uniform));
        queue.write_buffer(&pipeline.screen_buf, 0, bytemuck::bytes_of(&ScreenUniform {
            size: [w as f32, h as f32],
            _pad: [0.0; 2],
        }));

        if self.render_dirty || pipeline.render_dirty {
            if let Some(off) = &pipeline.offscreen {
                let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label:                    Some("globe pass"),
                    color_attachments:        &[Some(wgpu::RenderPassColorAttachment {
                        view:           &off.view,
                        resolve_target: None,
                        depth_slice:    None,
                        ops:            wgpu::Operations {
                            load:  wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes:         None,
                    occlusion_query_set:      None,
                    multiview_mask:           None,
                });

                // Sphere
                rp.set_pipeline(&pipeline.sphere_pipeline);
                rp.set_bind_group(0, &pipeline.sphere_bind_group, &[]);
                rp.set_vertex_buffer(0, pipeline.sphere_vbuf.slice(..));
                rp.set_index_buffer(pipeline.sphere_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..pipeline.sphere_idx_count, 0, 0..1);

                // Points
                if pipeline.point_count > 0 {
                    rp.set_pipeline(&pipeline.point_pipeline);
                    rp.set_bind_group(0, &pipeline.point_bind_group, &[]);
                    rp.set_vertex_buffer(0, pipeline.quad_vbuf.slice(..));
                    rp.set_vertex_buffer(1, pipeline.point_buf.slice(..));
                    rp.draw(0..6, 0..pipeline.point_count);
                }
            }
            pipeline.render_dirty = false;
        }
        vec![]
    }

    fn paint(
        &self,
        _info:       egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources:   &CallbackResources,
    ) {
        let Some(pipeline) = resources.get::<GlobePipeline>() else { return };
        let Some(off)      = &pipeline.offscreen          else { return };
        render_pass.set_pipeline(&pipeline.blit_pipeline);
        render_pass.set_bind_group(0, &off.blit_bind_group, &[]);
        render_pass.draw(0..6, 0..1);
    }
}

// ── CPU point collection ──────────────────────────────────────────────────────

fn pack_color(c: Color32) -> u32 {
    let [r, g, b, a] = c.to_array();
    r as u32 | ((g as u32) << 8) | ((b as u32) << 16) | ((a as u32) << 24)
}

/// Convert EPSG:4326 lon/lat → 3D point on unit sphere, collect for globe render.
/// Points use each layer's flat color only — `gis_editor` has no per-attribute
/// color ramp yet, so the globe matches the flat map's coloring exactly.
pub fn collect_globe_points(layers: &[LayerEntry], point_size: f32, out: &mut Vec<GlobePoint>) {
    out.clear();
    for entry in layers {
        if !entry.visible || !entry.show_points { continue; }
        let LayerKind::Points(pc) = &entry.data else { continue };

        let packed = pack_color(Color32::from_rgb(entry.color[0], entry.color[1], entry.color[2]));

        for (i, (_, p)) in pc.points.iter().enumerate() {
            if !pc.filter_mask[i] { continue; }
            let lon_r = (p[0] as f32).to_radians();
            let lat_r = (p[1] as f32).to_radians();
            let cp = lat_r.cos();
            let position = [cp * lon_r.sin(), lat_r.sin(), cp * lon_r.cos()];
            out.push(GlobePoint { position, color: packed, size: point_size });
        }
    }
}

// ── Sphere mesh generation ────────────────────────────────────────────────────

fn generate_sphere(slices: u32, stacks: u32) -> (Vec<SphereVert>, Vec<u32>, u32) {
    let mut verts:   Vec<SphereVert> = Vec::new();
    let mut indices: Vec<u32>        = Vec::new();

    for s in 0..=stacks {
        // phi: -π/2 (south) → +π/2 (north)
        let phi = -std::f32::consts::FRAC_PI_2
            + s as f32 * std::f32::consts::PI / stacks as f32;
        let cp = phi.cos();
        let sp = phi.sin();

        // v=0 at north pole, v=1 at south pole (matches equirectangular textures)
        let v = 1.0 - s as f32 / stacks as f32;

        for sl in 0..=slices {
            // theta offset by -π so the texture seam falls at the anti-meridian
            // (lon=±180°) rather than the prime meridian.
            let theta = sl as f32 * 2.0 * std::f32::consts::PI / slices as f32
                - std::f32::consts::PI;

            // matches collect_globe_points: x=cp*sin(lon), y=sin(lat), z=cp*cos(lon)
            let x = cp * theta.sin();
            let y = sp;
            let z = cp * theta.cos();

            // u=0 at lon=-180°, u=0.5 at lon=0° (prime meridian), u=1 at lon=+180°
            let u = sl as f32 / slices as f32;

            verts.push(SphereVert { pos: [x, y, z], nrm: [x, y, z], uv: [u, v] });
        }
    }

    for s in 0..stacks {
        for sl in 0..slices {
            let a = s * (slices + 1) + sl;
            let b = (s + 1) * (slices + 1) + sl;
            // CCW from outside
            indices.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
        }
    }

    let idx_count = indices.len() as u32;
    (verts, indices, idx_count)
}
