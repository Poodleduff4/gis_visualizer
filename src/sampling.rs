//! Sub-population sampling for the "Sampling" top-bar action. Pure id
//! selection logic — building the resulting layer is `LayerEntry::subset_by_ids`
//! (`gis_layer.rs`), same helper "create layer from selection" uses.

use std::collections::HashMap;

use rand::rng;
use rand::seq::SliceRandom;
use rand::RngExt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SamplingMethod {
    #[default]
    Random,
    Systematic,
    Stratified,
}

impl SamplingMethod {
    pub const ALL: [SamplingMethod; 3] =
        [SamplingMethod::Random, SamplingMethod::Systematic, SamplingMethod::Stratified];

    pub fn label(&self) -> &'static str {
        match self {
            SamplingMethod::Random => "Random",
            SamplingMethod::Systematic => "Systematic",
            SamplingMethod::Stratified => "Stratified (by attribute)",
        }
    }
}

/// Picks a sub-population of `ids` per `method`, keeping roughly `fraction`
/// (0.0-1.0) of them. `ids` is expected to already be whatever the caller
/// considers "the layer's current features" (e.g. filter-mask applied).
/// `group_of` assigns each id a stratum key — required for `Stratified`;
/// ignored otherwise. Returned ids are sorted ascending.
pub fn sample_ids(
    ids: &[usize],
    method: SamplingMethod,
    fraction: f64,
    group_of: Option<&dyn Fn(usize) -> String>,
) -> Vec<usize> {
    let fraction = fraction.clamp(0.0, 1.0);
    if ids.is_empty() || fraction <= 0.0 {
        return Vec::new();
    }
    if fraction >= 1.0 {
        let mut all = ids.to_vec();
        all.sort_unstable();
        return all;
    }

    let mut result = match method {
        SamplingMethod::Random => random_subset(ids, fraction, &mut rng()),
        SamplingMethod::Systematic => {
            let step = (1.0 / fraction).round().max(1.0) as usize;
            let offset = rng().random_range(0..step);
            ids.iter().copied().skip(offset).step_by(step).collect()
        }
        SamplingMethod::Stratified => {
            let Some(group_of) = group_of else {
                return sample_ids(ids, SamplingMethod::Random, fraction, None);
            };
            let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
            for &id in ids {
                groups.entry(group_of(id)).or_default().push(id);
            }
            let mut rng_inst = rng();
            groups
                .into_values()
                .flat_map(|members| random_subset(&members, fraction, &mut rng_inst))
                .collect()
        }
    };
    result.sort_unstable();
    result
}

fn random_subset(ids: &[usize], fraction: f64, rng: &mut impl rand::Rng) -> Vec<usize> {
    let mut shuffled = ids.to_vec();
    shuffled.shuffle(rng);
    let k = ((ids.len() as f64) * fraction).round().max(1.0) as usize;
    shuffled.truncate(k.min(ids.len()));
    shuffled
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_keeps_roughly_the_requested_fraction() {
        let ids: Vec<usize> = (0..1000).collect();
        let sampled = sample_ids(&ids, SamplingMethod::Random, 0.1, None);
        assert_eq!(sampled.len(), 100);
        assert!(sampled.windows(2).all(|w| w[0] < w[1]), "should be sorted+deduped");
    }

    #[test]
    fn systematic_keeps_every_nth() {
        let ids: Vec<usize> = (0..100).collect();
        let sampled = sample_ids(&ids, SamplingMethod::Systematic, 0.25, None);
        assert_eq!(sampled.len(), 25);
        let step = sampled[1] - sampled[0];
        assert_eq!(step, 4);
    }

    #[test]
    fn stratified_samples_each_group_independently() {
        let ids: Vec<usize> = (0..100).collect();
        let group_of = |id: usize| if id < 50 { "a".to_string() } else { "b".to_string() };
        let sampled = sample_ids(&ids, SamplingMethod::Stratified, 0.5, Some(&group_of));
        let a_count = sampled.iter().filter(|&&id| id < 50).count();
        let b_count = sampled.iter().filter(|&&id| id >= 50).count();
        assert_eq!(a_count, 25);
        assert_eq!(b_count, 25);
    }

    #[test]
    fn fraction_at_or_above_one_keeps_everything() {
        let ids: Vec<usize> = (0..10).collect();
        assert_eq!(sample_ids(&ids, SamplingMethod::Random, 1.0, None), ids);
    }
}
