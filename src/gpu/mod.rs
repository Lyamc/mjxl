//! GPU compute for the parts of Motion JPEG XL that are not a bitstream.
//!
//! Vulkan Video (`VK_KHR_video_decode_queue` / `VK_KHR_video_encode_queue`)
//! standardizes H.264, H.265, AV1, and VP9. It has no JPEG XL profile, so a
//! Vulkan Video session cannot decode or encode an `.mjxl` sample. The heavy
//! pixel work — dirty-tile search, 8×8 DCT/IDCT, and rectangle paste — runs as
//! compute through wgpu. The instance asks for a Vulkan adapter first and
//! falls back to DX12 or Metal. ANS, the MA tree, and the frame header stay
//! on the CPU in `o3-jxl`: they are serial and they are the actual codestream.

mod dct;
mod dirty;
pub(crate) mod paste;
pub use paste::RgbRect;

use std::sync::{Arc, OnceLock};

use crate::error::{Error, Result};

const DIRTY_WGSL: &str = include_str!("../shaders/dirty_tiles.wgsl");
const DCT_WGSL: &str = include_str!("../shaders/dct8.wgsl");
const PASTE_WGSL: &str = include_str!("../shaders/paste.wgsl");

/// Where tile search, transforms, and paste run.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum Backend {
    Cpu,
    Gpu,
    /// Vulkan compute when an adapter exists, otherwise CPU.
    #[default]
    Auto,
}

pub struct Gpu {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub info: String,
    pub(crate) dirty_layout: wgpu::BindGroupLayout,
    pub(crate) dirty: wgpu::ComputePipeline,
    pub(crate) dct_layout: wgpu::BindGroupLayout,
    pub(crate) dct: wgpu::ComputePipeline,
    pub(crate) paste_layout: wgpu::BindGroupLayout,
    pub(crate) paste: wgpu::ComputePipeline,
}

static SHARED: OnceLock<Option<Arc<Gpu>>> = OnceLock::new();

impl Gpu {
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self> {
        let (adapter, info) = pick_adapter().await?;
        let mut limits = wgpu::Limits::default();
        let cap = adapter.limits();
        if cap.max_buffer_size > limits.max_buffer_size {
            limits.max_buffer_size = cap.max_buffer_size.min(1 << 30);
        }
        if cap.max_storage_buffer_binding_size > limits.max_storage_buffer_binding_size {
            let cap_buf = limits.max_buffer_size.min(u32::MAX as u64) as u32;
            limits.max_storage_buffer_binding_size =
                cap.max_storage_buffer_binding_size.min(cap_buf);
        }
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("mjxl"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| Error::Gpu(e.to_string()))?;

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let dirty_layout = bind_layout(
            &device,
            "dirty",
            &[storage(true), storage(true), storage(false), uniform()],
        );
        let dct_layout = bind_layout(&device, "dct", &[storage(true), storage(false), uniform()]);
        let paste_layout = bind_layout(
            &device,
            "paste",
            &[storage(false), storage(true), uniform()],
        );
        let gpu = Self {
            dirty: pipeline(&device, "dirty", DIRTY_WGSL, &dirty_layout),
            dct: pipeline(&device, "dct8", DCT_WGSL, &dct_layout),
            paste: pipeline(&device, "paste", PASTE_WGSL, &paste_layout),
            dirty_layout,
            dct_layout,
            paste_layout,
            device,
            queue,
            info,
        };
        gpu.device.poll(wgpu::Maintain::Wait);
        if let Some(err) = gpu.device.pop_error_scope().await {
            return Err(Error::Gpu(err.to_string()));
        }
        Ok(gpu)
    }
}

/// `Gpu` if this backend should use compute, or `None` for a CPU fallback.
/// [`Backend::Gpu`] is an error when no adapter can be opened.
pub fn select(backend: Backend) -> Result<Option<Arc<Gpu>>> {
    match backend {
        Backend::Cpu => Ok(None),
        Backend::Auto => Ok(shared()),
        Backend::Gpu => shared()
            .ok_or_else(|| Error::Gpu("no Vulkan, DX12, or Metal adapter".into()))
            .map(Some),
    }
}

pub fn shared() -> Option<Arc<Gpu>> {
    SHARED
        .get_or_init(|| match Gpu::new() {
            Ok(gpu) => Some(Arc::new(gpu)),
            Err(e) => {
                eprintln!("mjxl: compute adapter unavailable ({e})");
                None
            }
        })
        .clone()
}

async fn pick_adapter() -> Result<(wgpu::Adapter, String)> {
    let opts = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    };
    let vulkan = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    if let Some(adapter) = vulkan.request_adapter(&opts).await {
        return Ok(describe(adapter));
    }
    let any = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN
            | wgpu::Backends::DX12
            | wgpu::Backends::METAL
            | wgpu::Backends::GL,
        ..Default::default()
    });
    let adapter = any
        .request_adapter(&opts)
        .await
        .ok_or_else(|| Error::Gpu("no compatible GPU adapter".into()))?;
    Ok(describe(adapter))
}

fn describe(adapter: wgpu::Adapter) -> (wgpu::Adapter, String) {
    let info = adapter.get_info();
    let label = format!("{:?} {}", info.backend, info.name);
    (adapter, label)
}

fn storage(read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform() -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bind_layout(
    device: &wgpu::Device,
    label: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> wgpu::BindGroupLayout {
    let entries: Vec<_> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let mut entry = e.clone();
            entry.binding = i as u32;
            entry
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

fn pipeline(
    device: &wgpu::Device,
    label: &str,
    wgsl: &str,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

pub(crate) fn uniform_buf(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bytes: &[u8],
) -> wgpu::Buffer {
    let mut padded = vec![0u8; 256];
    padded[..bytes.len().min(256)].copy_from_slice(&bytes[..bytes.len().min(256)]);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mjxl-uniform"),
        size: 256,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, &padded);
    buffer
}

pub(crate) fn storage_buf(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    data: &[u8],
    read_only: bool,
) -> wgpu::Buffer {
    let size = (data.len().max(4) as u64).div_ceil(4) * 4;
    let usage = if read_only {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
    } else {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
    };
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    });
    if !data.is_empty() {
        queue.write_buffer(&buffer, 0, data);
    }
    buffer
}

pub(crate) fn readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    size: u64,
) -> Result<Vec<u8>> {
    let size = size.div_ceil(4) * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mjxl-readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("mjxl-readback"),
    });
    enc.copy_buffer_to_buffer(src, 0, &staging, 0, size);
    queue.submit(Some(enc.finish()));
    let slice = staging.slice(..size);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|_| Error::Gpu("readback closed".into()))?
        .map_err(|e| Error::Gpu(e.to_string()))?;
    let mapped = slice.get_mapped_range();
    let out = mapped.to_vec();
    drop(mapped);
    staging.unmap();
    Ok(out)
}

pub(crate) fn dispatch(
    gpu: &Gpu,
    pipeline: &wgpu::ComputePipeline,
    layout: &wgpu::BindGroupLayout,
    buffers: &[&wgpu::Buffer],
    groups_x: u32,
    groups_y: u32,
) {
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, buf)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: buf.as_entire_binding(),
        })
        .collect();
    let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mjxl-bind"),
        layout,
        entries: &entries,
    });
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mjxl-compute"),
        });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mjxl"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups_x.max(1), groups_y.max(1), 1);
    }
    gpu.queue.submit(Some(enc.finish()));
}
