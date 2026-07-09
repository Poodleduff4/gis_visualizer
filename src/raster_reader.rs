/// GeoTIFF raster reader. Pure Rust (no FFI), works on desktop and wasm32.
/// Expects single- or multi-band 32-bit float TIFFs on a canonical full
/// -180..180 / -90..90 canvas, row 0 = north, col 0 = lon -180, NaN = no data.
use anyhow::{anyhow, Result};
use flatgeobuf::GeometryType;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::tags::Tag;

use crate::filter::FilterLogic;
use crate::gis_layer::{LayerEntry, LayerKind, RasterBand, RasterData, RasterDisplayMode};
use crate::gis_reader::{GisFilePath, LayerDescriptor};

/// Sentinel geometry-type value for raster layers — unused by any raster
/// code path (only vector/point layers dispatch on `geometry_type.0`).
const RASTER_GEOMETRY_TYPE: GeometryType = GeometryType(255);

/// Reads a tag that TIFF stores as one value per sample (e.g. `BitsPerSample`,
/// `SampleFormat`). For single-band files it's a scalar; for interleaved
/// multi-band files (`SamplesPerPixel` > 1) it's a list with one entry per
/// sample — all bands share the same format, so the first entry suffices.
fn first_per_sample_tag<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
    tag: Tag,
) -> Option<u32> {
    decoder
        .get_tag_u32(tag)
        .ok()
        .or_else(|| decoder.get_tag_u32_vec(tag).ok().and_then(|v| v.first().copied()))
}

fn parse_name(stem: &str) -> (String, String) {
    match stem.split_once('_') {
        Some((var, date)) => (var.to_string(), date.to_string()),
        None => (stem.to_string(), String::new()),
    }
}

// ── Raster descriptor (metadata-only, no pixel decode) ─────────────────────────

pub struct RasterDescriptor {
    pub name: String,
    pub variable: String,
    pub date: String,
    pub width: u32,
    pub height: u32,
    pub units: String,
    pub bits_per_sample: u16,
    pub is_f32: bool,
    pub file_size: u64,
    /// Desktop: source path, re-opened on Load.
    pub path: Option<std::path::PathBuf>,
    /// Wasm: source bytes + filename, re-decoded on Load.
    pub bytes: Option<(Vec<u8>, String)>,
}

/// Read dimensions + tags from the TIFF header only — no pixel data decoded.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_raster_descriptor_sync(path: &std::path::Path) -> Result<RasterDescriptor> {
    let file = std::fs::File::open(path)?;
    let file_size = file.metadata()?.len();
    let mut decoder = Decoder::new(std::io::BufReader::new(file))?;
    let (width, height) = decoder.dimensions()?;
    let units = decoder.get_tag_ascii_string(Tag::ImageDescription).unwrap_or_default();
    let bits_per_sample = first_per_sample_tag(&mut decoder, Tag::BitsPerSample).unwrap_or(0) as u16;
    let sample_format = first_per_sample_tag(&mut decoder, Tag::SampleFormat).unwrap_or(3);
    let is_f32 = sample_format == 3 && bits_per_sample == 32;

    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("raster");
    let (variable, date) = parse_name(stem);
    let name = if date.is_empty() { variable.clone() } else { format!("{variable} {date}") };

    Ok(RasterDescriptor {
        name, variable, date, width, height, units,
        bits_per_sample, is_f32, file_size,
        path: Some(path.to_owned()), bytes: None,
    })
}

/// Read dimensions + tags from in-memory TIFF bytes — no pixel data decoded.
pub fn read_raster_descriptor_bytes(bytes: Vec<u8>, filename: &str) -> Result<RasterDescriptor> {
    let file_size = bytes.len() as u64;
    let mut decoder = Decoder::new(std::io::Cursor::new(&bytes))?;
    let (width, height) = decoder.dimensions()?;
    let units = decoder.get_tag_ascii_string(Tag::ImageDescription).unwrap_or_default();
    let bits_per_sample = first_per_sample_tag(&mut decoder, Tag::BitsPerSample).unwrap_or(0) as u16;
    let sample_format = first_per_sample_tag(&mut decoder, Tag::SampleFormat).unwrap_or(3);
    let is_f32 = sample_format == 3 && bits_per_sample == 32;

    let stem = filename.strip_suffix(".tif").or_else(|| filename.strip_suffix(".tiff")).unwrap_or(filename);
    let (variable, date) = parse_name(stem);
    let name = if date.is_empty() { variable.clone() } else { format!("{variable} {date}") };
    let filename = filename.to_string();

    Ok(RasterDescriptor {
        name, variable, date, width, height, units,
        bits_per_sample, is_f32, file_size,
        path: None, bytes: Some((bytes, filename)),
    })
}

/// Split one `read_image()` result into per-band planes, honoring
/// `PlanarConfiguration` (1 = chunky/interleaved, 2 = planar — TIFF default is
/// chunky when the tag is absent).
fn deinterleave_bands<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
    width: usize,
    height: usize,
    samples: usize,
    raw: Vec<f32>,
) -> Vec<Vec<f32>> {
    let px = width * height;
    let planar = decoder.get_tag_u32(Tag::PlanarConfiguration).unwrap_or(1) == 2;
    (0..samples)
        .map(|b| {
            if planar {
                raw[b * px..(b + 1) * px].to_vec()
            } else {
                (0..px).map(|i| raw[i * samples + b]).collect()
            }
        })
        .collect()
}

/// Decode every band of a (possibly multi-band) TIFF: either interleaved
/// samples-per-pixel within one IFD, or one band per page (multi-page stack).
fn decode_bands<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
) -> Result<(usize, usize, Vec<Vec<f32>>)> {
    let (width, height) = decoder.dimensions()?;
    let (width, height) = (width as usize, height as usize);
    let samples = decoder.get_tag_u32(Tag::SamplesPerPixel).unwrap_or(1) as usize;

    let DecodingResult::F32(raw) = decoder.read_image()? else {
        return Err(anyhow!("expected a 32-bit float TIFF"));
    };

    if samples > 1 {
        return Ok((width, height, deinterleave_bands(decoder, width, height, samples, raw)));
    }

    let mut bands = vec![raw];
    while decoder.more_images() {
        decoder.next_image()?;
        let DecodingResult::F32(page) = decoder.read_image()? else {
            return Err(anyhow!("expected a 32-bit float TIFF"));
        };
        bands.push(page);
    }
    Ok((width, height, bands))
}

fn band_stats(values: &[f32]) -> (f64, f64) {
    let mut lo = f64::MAX;
    let mut hi = f64::MIN;
    for &v in values {
        if v.is_finite() {
            lo = lo.min(v as f64);
            hi = hi.max(v as f64);
        }
    }
    if lo.is_finite() && hi.is_finite() { (lo, hi) } else { (0.0, 0.0) }
}

fn build_layer_entry(
    name: String,
    width: usize,
    height: usize,
    bands_raw: Vec<Vec<f32>>,
    units: String,
    location: GisFilePath,
) -> LayerEntry {
    let single = bands_raw.len() == 1;
    let bands: Vec<RasterBand> = bands_raw.into_iter().enumerate().map(|(i, values)| {
        let (data_min, data_max) = band_stats(&values);
        RasterBand {
            name: if single { name.clone() } else { format!("Band {}", i + 1) },
            values, data_min, data_max,
            display_min: data_min,
            display_max: data_max,
        }
    }).collect();

    let num_features = (width * height) as u64;
    let raster = RasterData {
        width, height,
        display_mode: RasterDisplayMode::Single(0),
        bands,
        units,
    };

    LayerEntry {
        data: LayerKind::Raster(raster),
        visible: true,
        name: name.clone(),
        color: [255, 255, 255],
        opacity: 255,
        descriptor: LayerDescriptor {
            name,
            num_features,
            field_names: Vec::new(),
            geometry_type: RASTER_GEOMETRY_TYPE,
            location,
        },
        filters: Vec::new(),
        filter_logic: FilterLogic::default(),
        roi_bboxes: Vec::new(),
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_raster_sync(path: &std::path::Path) -> Result<LayerEntry> {
    let file = std::fs::File::open(path)?;
    let mut decoder = Decoder::new(std::io::BufReader::new(file))?;
    let units = decoder.get_tag_ascii_string(Tag::ImageDescription).unwrap_or_default();
    let (width, height, bands) = decode_bands(&mut decoder)?;

    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("raster");
    let (variable, date) = parse_name(stem);
    let name = if date.is_empty() { variable } else { format!("{variable} {date}") };

    Ok(build_layer_entry(
        name, width, height, bands, units,
        GisFilePath::LocalFile(path.to_string_lossy().into_owned()),
    ))
}

pub fn load_raster_bytes(bytes: Vec<u8>, filename: &str) -> Result<LayerEntry> {
    let raw: std::sync::Arc<[u8]> = std::sync::Arc::from(bytes.as_slice());
    let mut decoder = Decoder::new(std::io::Cursor::new(bytes))?;
    let units = decoder.get_tag_ascii_string(Tag::ImageDescription).unwrap_or_default();
    let (width, height, bands) = decode_bands(&mut decoder)?;

    let stem = filename.strip_suffix(".tif").or_else(|| filename.strip_suffix(".tiff")).unwrap_or(filename);
    let (variable, date) = parse_name(stem);
    let name = if date.is_empty() { variable } else { format!("{variable} {date}") };

    Ok(build_layer_entry(
        name, width, height, bands, units,
        GisFilePath::Bytes(raw, filename.to_string()),
    ))
}
