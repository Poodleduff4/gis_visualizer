use proj4rs::Proj;

/// Reprojects coordinates from a source EPSG CRS to WGS84 (EPSG:4326) — the
/// only CRS this app's viewport/shaders understand.
#[derive(Clone)]
pub struct CrsTransform {
    from: Proj,
    to: Proj,
    /// True when `from` was built from an ellipsoid-only fallback because the
    /// CRS's real datum (e.g. NAD27) needs an external grid-shift file that
    /// proj4rs doesn't bundle. Positions are then off by up to ~100m instead
    /// of being exact — surfaced to the user rather than failing silently.
    pub approximate_datum: bool,
}

/// Legacy datums that require an external NADCON/NTv2 grid file for an exact
/// shift to WGS84, mapped to their reference ellipsoid for an approximate,
/// grid-free fallback transform.
const GRID_DATUM_FALLBACKS: &[(&str, &str)] = &[("NAD27", "clrk66"), ("NAD27CGQ77", "clrk66")];

impl CrsTransform {
    pub fn from_epsg(epsg: u16) -> anyhow::Result<Self> {
        let to = Self::wgs84()?;
        if let Ok(from) = Proj::from_epsg_code(epsg) {
            return Ok(Self { from, to, approximate_datum: false });
        }
        // Retry without the grid-dependent datum, using its reference ellipsoid
        // instead — no datum-shift correction, but still lands in the right
        // place rather than failing outright.
        let def = crs_definitions::from_code(epsg)
            .ok_or_else(|| anyhow::anyhow!("Unknown EPSG code: {epsg}"))?;
        let patched = Self::patch_grid_datum(def.proj4).ok_or_else(|| {
            anyhow::anyhow!("EPSG:{epsg} needs a datum-shift grid file that isn't available")
        })?;
        let from = Proj::from_proj_string(&patched).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Self { from, to, approximate_datum: true })
    }

    fn wgs84() -> anyhow::Result<Proj> {
        Proj::from_proj_string("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    fn patch_grid_datum(proj4: &str) -> Option<String> {
        GRID_DATUM_FALLBACKS.iter().find_map(|(datum, ellps)| {
            let needle = format!("+datum={datum}");
            proj4
                .contains(&needle)
                .then(|| proj4.replace(&needle, &format!("+ellps={ellps}")))
        })
    }

    /// Mutates `xy` in place. Leaves it unchanged and returns `false` if the
    /// point falls outside the source projection's valid domain.
    pub fn convert(&self, xy: &mut [f64; 2]) -> bool {
        let mut p = if self.from.is_latlong() {
            (xy[0].to_radians(), xy[1].to_radians(), 0.0)
        } else {
            (xy[0], xy[1], 0.0)
        };
        if proj4rs::transform::transform(&self.from, &self.to, &mut p).is_err() {
            return false;
        }
        xy[0] = if self.to.is_latlong() { p.0.to_degrees() } else { p.0 };
        xy[1] = if self.to.is_latlong() { p.1.to_degrees() } else { p.1 };
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn web_mercator_origin_maps_to_zero_zero() {
        let t = CrsTransform::from_epsg(3857).unwrap();
        let mut xy = [0.0, 0.0];
        assert!(t.convert(&mut xy));
        assert!(xy[0].abs() < 1e-9);
        assert!(xy[1].abs() < 1e-9);
        assert!(!t.approximate_datum);
    }

    #[test]
    fn web_mercator_known_offset() {
        // ~Toronto in EPSG:3857 (metres) -> approx (-79.38, 43.65) in WGS84.
        let t = CrsTransform::from_epsg(3857).unwrap();
        let mut xy = [-8_837_000.0, 5_411_000.0];
        assert!(t.convert(&mut xy));
        assert!((xy[0] - (-79.38)).abs() < 0.1, "lon={}", xy[0]);
        assert!((xy[1] - 43.65).abs() < 0.1, "lat={}", xy[1]);
    }

    #[test]
    fn nad83_mtm10_toronto_no_fallback_needed() {
        // EPSG:2952, NAD83(CSRS)/MTM zone 10 -- no grid-dependent datum.
        let t = CrsTransform::from_epsg(2952).unwrap();
        assert!(!t.approximate_datum);
        let mut xy = [304000.0, 4837000.0];
        assert!(t.convert(&mut xy));
        assert!((xy[0] - (-79.51)).abs() < 0.1, "lon={}", xy[0]);
        assert!((xy[1] - 43.67).abs() < 0.1, "lat={}", xy[1]);
    }

    #[test]
    fn nad27_mtm10_falls_back_to_ellipsoid_only() {
        // EPSG:7991, NAD27/MTM zone 10 -- proj4rs can't do the NAD27 grid
        // shift, so this should still succeed via the ellipsoid fallback and
        // land within ~1km of the same real-world point (NAD27->NAD83 shift
        // in southern Ontario is on the order of tens of metres).
        let t = CrsTransform::from_epsg(7991).unwrap();
        assert!(t.approximate_datum);
        let mut xy = [304000.0, 4837000.0];
        assert!(t.convert(&mut xy));
        assert!((xy[0] - (-79.51)).abs() < 0.02, "lon={}", xy[0]);
        assert!((xy[1] - 43.67).abs() < 0.02, "lat={}", xy[1]);
    }
}
