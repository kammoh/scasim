//! Channels per scope: one channel for each scope below a given scope, for a ranking of the
//! leakage by the parts of a design.

use crate::hierarchy::{HierarchyIndex, Selection, scope_prefix_len};
use crate::power::{ChannelSpec, PowerPlan};
use std::collections::BTreeMap;

/// The signals of one generated channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopeChannel {
    /// The scope path that names the channel.
    pub name: String,
    /// One path for each signal in the channel: the path with the deepest scope of the signal
    /// (see [`group_by_scope`]).
    pub paths: Vec<String>,
}

/// The result of [`group_by_scope`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ScopeGroups {
    /// The channels, sorted by name. Every channel has at least one signal.
    pub channels: Vec<ScopeChannel>,
    /// The number of selected signals with more than one path in the scope (aliases).
    pub aliased: usize,
    /// The number of selected signals that have no path in the scope. They are in the channel
    /// `(outside SCOPE)`.
    pub outside: usize,
}

/// Groups the selected signals by scope.
///
/// `scope` is a scope path. `depth` (at least 1) is the number of scope levels below it. There is
/// one channel for each scope exactly `depth` levels below `scope` that holds selected signals,
/// and one channel named `scope` for the signals directly in `scope`. A signal in a deeper scope
/// belongs to its ancestor `depth` levels below `scope`.
///
/// A signal can have several paths (aliases). It belongs to exactly one channel: the channel of
/// its path with the deepest scope that lies in `scope`. If several paths have equally deep
/// scopes, the smallest path (by text) decides. A scope without selected signals gives no channel.
///
/// A selected signal with no path in `scope` goes to one more channel, named
/// `(outside SCOPE)`, with its smallest path. So the channels always partition the selection.
///
/// `selected` has one flag for each handle of `index`.
pub fn group_by_scope(
    index: &HierarchyIndex,
    selected: &[bool],
    scope: &str,
    depth: usize,
) -> ScopeGroups {
    let mut by_name: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut groups = ScopeGroups::default();
    for (paths, _) in index.paths.iter().zip(selected).filter(|(_, s)| **s) {
        // The paths in `scope`, each with the number of scope names that `scope` takes.
        let inside: Vec<_> = paths
            .iter()
            .filter_map(|p| scope_prefix_len(&p.scope_names, scope).map(|k| (k, p)))
            .collect();
        // The deepest scope first. Then the smallest path.
        let Some(&(k, chosen)) = inside
            .iter()
            .min_by_key(|(_, p)| (std::cmp::Reverse(p.scope_names.len()), &p.path))
        else {
            groups.outside += 1;
            let smallest = paths
                .iter()
                .map(|p| &p.path)
                .min()
                .expect("a selected signal has a path");
            by_name
                .entry(format!("(outside {scope})"))
                .or_default()
                .push(smallest.clone());
            continue;
        };
        if inside.len() > 1 {
            groups.aliased += 1;
        }
        let level = chosen.scope_names.len().min(k.saturating_add(depth));
        by_name
            .entry(chosen.scope_names[..level].join("."))
            .or_default()
            .push(chosen.path.clone());
    }
    groups.channels = by_name
        .into_iter()
        .map(|(name, paths)| ScopeChannel { name, paths })
        .collect();
    groups
}

/// A plan with the channel `total` (the whole selection `base`) and one channel for each group,
/// in the order of the groups.
pub fn scope_plan(base: Selection, groups: &ScopeGroups) -> PowerPlan {
    let mut plan = PowerPlan::toggles(base);
    plan.channels[0].name = "total".into();
    plan.channels
        .extend(groups.channels.iter().map(|c| ChannelSpec {
            name: c.name.clone(),
            selection: Selection::from_paths(c.paths.iter().cloned()),
        }));
    plan
}
