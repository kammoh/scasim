//! Internal helpers that both activity paths share: where the changes of a time point go, and the
//! buffers that collect them.
//!
//! A *slot* is a place for statistics. Slot 0 is "before the first bin". Slots `1..=n` are the
//! `n` bins. Slot `n + 1` is "at or after the end". A run keeps one slot array per channel.

use super::{ActivityTrace, Bins, ChannelTrace, FullStats, PowerPlan, RunInfo, Totals};

/// Marks a time point whose changes are ignored.
const IGNORED: u32 = u32::MAX;

/// The slot of every time point in a list of times, and the range of slots that they use.
///
/// The fast path decodes one section at a time. It allocates buffers only for the slots that the
/// section uses. A section of a long run can have millions of time points, but only a few bins.
pub(crate) struct Placement {
    /// The global slot that local slot 0 stands for.
    pub first_slot: usize,
    /// The number of local slots. It is 0 if every time point is ignored.
    pub slot_count: usize,
    /// The local slot of each time point, or `IGNORED`.
    local: Vec<u32>,
}

impl Placement {
    /// Places each time of `times` (not decreasing). Times after `last_time` are ignored.
    pub fn new(times: &[u64], bins: &Bins, last_time: Option<u64>) -> Placement {
        let global: Vec<Option<usize>> = times
            .iter()
            .map(|&t| {
                let cut = last_time.is_some_and(|last| t > last);
                (!cut).then(|| bins.slot_of(t))
            })
            .collect();
        let used = global.iter().flatten();
        let (first_slot, end_slot) = match (used.clone().min(), used.max()) {
            (Some(&min), Some(&max)) => (min, max + 1),
            _ => (0, 0),
        };
        let local = global
            .iter()
            .map(|slot| {
                slot.map_or(IGNORED, |s| {
                    u32::try_from(s - first_slot).expect("Bins::new limits the number of bins")
                })
            })
            .collect();
        Placement {
            first_slot,
            slot_count: end_slot - first_slot,
            local,
        }
    }

    /// The local slot of time point `index`, or `None` if its changes are ignored.
    #[inline]
    pub fn local_slot(&self, index: usize) -> Option<usize> {
        let slot = self.local[index];
        (slot != IGNORED).then_some(slot as usize)
    }

    /// The global slot of time point `index`, or `None` if its changes are ignored.
    #[inline]
    pub fn global_slot(&self, index: usize) -> Option<usize> {
        self.local_slot(index).map(|s| s + self.first_slot)
    }
}

/// The statistics of one channel for a range of slots, as one array per statistic.
///
/// Without full statistics, only `toggles` has entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SlotStats {
    toggles: Vec<u64>,
    rise: Vec<u64>,
    fall: Vec<u64>,
    hw_delta: Vec<i64>,
}

impl SlotStats {
    pub fn new(slots: usize, full: bool) -> SlotStats {
        let full_len = if full { slots } else { 0 };
        SlotStats {
            toggles: vec![0; slots],
            rise: vec![0; full_len],
            fall: vec![0; full_len],
            hw_delta: vec![0; full_len],
        }
    }

    /// Adds a change to `slot`. With `FULL == false`, only the toggles count.
    #[inline]
    pub fn add<const FULL: bool>(&mut self, slot: usize, d: &Totals) {
        self.toggles[slot] += d.toggles;
        if FULL {
            self.rise[slot] += d.rise;
            self.fall[slot] += d.fall;
            self.hw_delta[slot] += d.hw_delta;
        }
    }

    /// Adds slot `i` of `other` to slot `offset + i` of `self`, for every slot of `other`.
    pub fn add_window(&mut self, offset: usize, other: &SlotStats) {
        fn add_all<T: Copy + std::ops::AddAssign>(dst: &mut [T], src: &[T]) {
            for (d, s) in dst.iter_mut().zip(src) {
                *d += *s;
            }
        }
        add_all(&mut self.toggles[offset..], &other.toggles);
        if !other.rise.is_empty() {
            add_all(&mut self.rise[offset..], &other.rise);
            add_all(&mut self.fall[offset..], &other.fall);
            add_all(&mut self.hw_delta[offset..], &other.hw_delta);
        }
    }

    fn totals(&self, slot: usize) -> Totals {
        if self.rise.is_empty() {
            Totals {
                toggles: self.toggles[slot],
                ..Totals::default()
            }
        } else {
            Totals {
                toggles: self.toggles[slot],
                rise: self.rise[slot],
                fall: self.fall[slot],
                hw_delta: self.hw_delta[slot],
            }
        }
    }

    /// Converts an array with the layout "before, bins, after" to a channel trace.
    pub fn into_channel_trace(mut self, name: String) -> ChannelTrace {
        let last = self.toggles.len() - 1;
        let before = self.totals(0);
        let after = self.totals(last);
        fn remove_ends<T>(v: &mut Vec<T>) {
            if !v.is_empty() {
                v.pop();
                v.remove(0);
            }
        }
        let full = !self.rise.is_empty();
        remove_ends(&mut self.toggles);
        remove_ends(&mut self.rise);
        remove_ends(&mut self.fall);
        remove_ends(&mut self.hw_delta);
        ChannelTrace {
            name,
            toggles: self.toggles,
            full: full.then_some(FullStats {
                rise: self.rise,
                fall: self.fall,
                hw_delta: self.hw_delta,
            }),
            before,
            after,
        }
    }
}

/// Adds the per-channel statistics `b` to `a`. Both cover the same slots, or one of them is
/// empty (no data yet).
pub(crate) fn merge_channel_stats(mut a: Vec<SlotStats>, b: Vec<SlotStats>) -> Vec<SlotStats> {
    if a.is_empty() {
        return b;
    }
    for (x, y) in a.iter_mut().zip(&b) {
        x.add_window(0, y);
    }
    a
}

/// Builds the result of a run from the statistics of its channels (one slot array per channel).
pub(crate) fn assemble_trace(
    plan: &PowerPlan,
    bins: &Bins,
    stats: Vec<SlotStats>,
    timescale_exponent: Option<i8>,
    info: RunInfo,
) -> ActivityTrace {
    ActivityTrace {
        times: bins.starts().to_vec(),
        timescale_exponent,
        channels: plan
            .channels
            .iter()
            .zip(stats)
            .map(|(spec, stats)| stats.into_channel_trace(spec.name.clone()))
            .collect(),
        info,
    }
}

/// Empty statistics for every channel of `plan`, with room for every bin of `bins`.
pub(crate) fn new_channel_stats(plan: &PowerPlan, bins: &Bins) -> Vec<SlotStats> {
    plan.channels
        .iter()
        .map(|_| SlotStats::new(bins.len() + 2, plan.full_stats))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_uses_only_the_slots_of_the_section() {
        // Bins [10, 20), [20, 30), [30, 40). Slots: 0 before, 1..=3 bins, 4 after.
        let bins = Bins::new(vec![10, 20, 30], Some(40)).unwrap();
        let p = Placement::new(&[20, 21, 29, 30], &bins, None);
        assert_eq!((p.first_slot, p.slot_count), (2, 2));
        let local: Vec<_> = (0..4).map(|i| p.local_slot(i)).collect();
        assert_eq!(local, vec![Some(0), Some(0), Some(0), Some(1)]);
        let global: Vec<_> = (0..4).map(|i| p.global_slot(i)).collect();
        assert_eq!(global, vec![Some(2), Some(2), Some(2), Some(3)]);
    }

    #[test]
    fn placement_covers_before_and_after() {
        let bins = Bins::new(vec![10, 20], Some(30)).unwrap();
        let p = Placement::new(&[5, 10, 25, 30, 99], &bins, None);
        assert_eq!((p.first_slot, p.slot_count), (0, 4));
        let global: Vec<_> = (0..5).map(|i| p.global_slot(i)).collect();
        assert_eq!(global, vec![Some(0), Some(1), Some(2), Some(3), Some(3)]);
    }

    #[test]
    fn placement_ignores_times_after_the_last_time() {
        let bins = Bins::new(vec![10, 20, 30], None).unwrap();
        let p = Placement::new(&[10, 20, 30], &bins, Some(20));
        assert_eq!((p.first_slot, p.slot_count), (1, 2));
        assert_eq!(p.global_slot(2), None);
        let none = Placement::new(&[30, 40], &bins, Some(20));
        assert_eq!(none.slot_count, 0);
        assert_eq!(none.local_slot(0), None);
    }

    #[test]
    fn stats_add_windows_and_convert() {
        let d = |t, r, f, h| Totals {
            toggles: t,
            rise: r,
            fall: f,
            hw_delta: h,
        };
        // Layout: before, two bins, after.
        let mut global = SlotStats::new(4, true);
        let mut local = SlotStats::new(2, true);
        local.add::<true>(0, &d(3, 2, 1, 2));
        local.add::<true>(1, &d(1, 0, 1, -2));
        global.add::<true>(0, &d(1, 1, 0, 2));
        global.add_window(1, &local);
        global.add::<true>(3, &d(5, 3, 2, 2));
        let trace = global.into_channel_trace("c".into());
        assert_eq!(trace.toggles, vec![3, 1]);
        let full = trace.full.unwrap();
        assert_eq!(full.rise, vec![2, 0]);
        assert_eq!(full.fall, vec![1, 1]);
        assert_eq!(full.hw_delta, vec![2, -2]);
        assert_eq!(trace.before, d(1, 1, 0, 2));
        assert_eq!(trace.after, d(5, 3, 2, 2));
    }

    #[test]
    fn toggles_only_stats_have_no_full_arrays() {
        let mut s = SlotStats::new(3, false);
        s.add::<false>(
            1,
            &Totals {
                toggles: 4,
                rise: 9,
                fall: 9,
                hw_delta: 9,
            },
        );
        s.add::<false>(
            2,
            &Totals {
                toggles: 7,
                ..Totals::default()
            },
        );
        let trace = s.into_channel_trace("c".into());
        assert_eq!(trace.toggles, vec![4]);
        assert!(trace.full.is_none());
        assert_eq!(
            trace.after,
            Totals {
                toggles: 7,
                ..Totals::default()
            }
        );
    }

    #[test]
    fn merge_treats_an_empty_list_as_no_data() {
        let mut one = SlotStats::new(2, false);
        one.add::<false>(
            0,
            &Totals {
                toggles: 1,
                ..Totals::default()
            },
        );
        let merged = merge_channel_stats(Vec::new(), vec![one.clone()]);
        assert_eq!(merged, vec![one.clone()]);
        let merged = merge_channel_stats(merged, vec![one.clone()]);
        assert_eq!(merged[0].toggles, vec![2, 0]);
        let merged = merge_channel_stats(merged, Vec::new());
        assert_eq!(merged[0].toggles, vec![2, 0]);
    }
}
