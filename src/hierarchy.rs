//! Signal paths and ordered selection rules that choose which signals contribute to a channel.

use regex::Regex;

/// One variable path that refers to a signal handle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignalPath {
    /// Scope names from the root, then the variable name, joined by `.`. A trailing bit range
    /// such as ` [3:0]` is removed from the variable name. No name is added or assumed.
    pub path: String,
    /// The scope part of `path` (everything before the variable name), without a trailing `.`.
    pub scope: String,
    /// The names of the enclosing scopes, outermost first. A name can contain `.` (a Verilog
    /// escaped identifier), so `scope` alone cannot tell it from two nested scopes. `scope:`
    /// rules use these names. The list is empty for a variable outside every scope.
    pub scope_names: Vec<String>,
    /// Module (component) names of the enclosing scopes, outermost first. An empty string means
    /// that the file has no module name for that scope.
    pub modules: Vec<String>,
    /// True if the file declares this variable as an alias of a variable declared earlier.
    /// `wellen` does not report this, so it is always false for paths that come from `wellen`.
    pub is_alias: bool,
}

/// The variable paths of every signal handle in one waveform file.
#[derive(Debug, Clone, Default)]
pub struct HierarchyIndex {
    /// `paths[h]` lists every variable path of handle index `h`. Aliases share one handle.
    /// A handle without a path is never selectable.
    pub paths: Vec<Vec<SignalPath>>,
    /// True if at least one scope in the file has a module (component) name.
    pub has_module_names: bool,
}

/// Removes a trailing bit range such as ` [3:0]`, `[31:0]`, or ` [7]` from a variable name.
///
/// A range without a space before it must contain a colon: `mem[3]` is an array element and stays.
///
/// A name that consists only of a bit range, such as `[3:0]`, stays whole, so the result is never
/// empty. The same holds if only white space comes before the range. For example, `wellen` can
/// turn `tdata[1714295607408:1714295607407]` into the scope `tdata` and the variable name
/// `[1714295607408:1714295607407]`.
pub fn strip_bit_range(name: &str) -> &str {
    let Some(open) = name.rfind('[') else {
        return name;
    };
    let Some(inner) = name[open + 1..].strip_suffix(']') else {
        return name;
    };
    let is_range = !inner.is_empty()
        && inner
            .chars()
            .all(|c| c.is_ascii_digit() || c == ':' || c == '-');
    if !is_range {
        return name;
    }
    let base = match name[..open].strip_suffix(' ') {
        // Verilog dumpers: "data [3:0]" and "bit [7]"
        Some(base) => base,
        // VHDL dumpers: "data[3:0]". Without a colon, "mem[3]" is an array element.
        None if inner.contains(':') => &name[..open],
        None => name,
    };
    // A name that consists only of a bit range stays whole. The name must not become empty or
    // consist only of white space.
    if base.trim().is_empty() { name } else { base }
}

/// The scopes that are open while a hierarchy is walked.
#[derive(Default)]
struct ScopeStack {
    entries: Vec<OpenScope>,
}

struct OpenScope {
    name: String,
    module: String,
    /// Names of all open scopes up to and including this one, joined by `.`.
    path: String,
}

impl ScopeStack {
    fn push(&mut self, name: &str, module: &str) {
        let path = match self.entries.last() {
            Some(parent) => format!("{}.{name}", parent.path),
            None => name.to_string(),
        };
        self.entries.push(OpenScope {
            name: name.to_string(),
            module: module.to_string(),
            path,
        });
    }

    fn pop(&mut self) {
        self.entries.pop();
    }

    /// Path of the innermost open scope, or an empty string at the top level.
    fn path(&self) -> &str {
        self.entries.last().map_or("", |e| e.path.as_str())
    }

    fn names(&self) -> Vec<String> {
        self.entries.iter().map(|e| e.name.clone()).collect()
    }

    fn modules(&self) -> Vec<String> {
        self.entries.iter().map(|e| e.module.clone()).collect()
    }
}

impl HierarchyIndex {
    /// Adds the path of variable `name` in the open scopes to handle `handle`.
    fn add(&mut self, handle: usize, scopes: &ScopeStack, name: &str, is_alias: bool) {
        if self.paths.len() <= handle {
            self.paths.resize(handle + 1, Vec::new());
        }
        let name = strip_bit_range(name);
        let scope = scopes.path().to_string();
        let path = if scope.is_empty() {
            name.to_string()
        } else {
            format!("{scope}.{name}")
        };
        self.paths[handle].push(SignalPath {
            path,
            scope,
            scope_names: scopes.names(),
            modules: scopes.modules(),
            is_alias,
        });
    }

    /// Builds the index from the hierarchy of an FST file. Handle indices are `fst-reader`
    /// handle indices. Variables of type event, string, or real get no path, and neither do
    /// variables of width 0 (for example VHDL arrays with a null range), because they have no
    /// bit-vector value to count.
    pub fn from_fst<R: std::io::BufRead + std::io::Seek>(
        reader: &mut fst_reader::FstReader<R>,
    ) -> Result<Self, fst_reader::ReaderError> {
        use fst_reader::{FstHierarchyEntry, FstVarType};
        let mut index = HierarchyIndex {
            paths: vec![Vec::new(); reader.get_header().max_handle as usize],
            has_module_names: false,
        };
        let mut scopes = ScopeStack::default();
        reader.read_hierarchy(|entry| match entry {
            FstHierarchyEntry::Scope {
                name, component, ..
            } => {
                index.has_module_names |= !component.is_empty();
                scopes.push(&name, &component);
            }
            FstHierarchyEntry::UpScope => scopes.pop(),
            FstHierarchyEntry::Var {
                tpe,
                name,
                length,
                handle,
                is_alias,
                ..
            } => {
                let has_bit_vector = length > 0
                    && !(tpe == FstVarType::Event
                        || tpe == FstVarType::GenericString
                        || tpe.is_real());
                if has_bit_vector {
                    index.add(handle.get_index(), &scopes, &name, is_alias);
                }
            }
            _ => {}
        })?;
        Ok(index)
    }

    /// Builds the index from a `wellen` hierarchy (any format that `wellen` reads). Handle indices
    /// are `wellen` signal-ref indices.
    ///
    /// The scope tree is walked, so an escaped name that contains `.` stays whole. Variables
    /// that are not bit vectors (strings, reals) and events (width 0) get no path.
    pub fn from_wellen(hierarchy: &wellen::Hierarchy) -> Self {
        let mut index = HierarchyIndex::default();
        let mut scopes = ScopeStack::default();
        for var in hierarchy.vars() {
            index.add_wellen_var(hierarchy, &scopes, var);
        }
        for scope in hierarchy.scopes() {
            index.add_wellen_scope(hierarchy, &mut scopes, scope);
        }
        index
    }

    fn add_wellen_scope(
        &mut self,
        hierarchy: &wellen::Hierarchy,
        scopes: &mut ScopeStack,
        scope_ref: wellen::ScopeRef,
    ) {
        let scope = &hierarchy[scope_ref];
        let module = scope.component(hierarchy).unwrap_or("");
        self.has_module_names |= !module.is_empty();
        scopes.push(scope.name(hierarchy), module);
        for var in scope.vars(hierarchy) {
            self.add_wellen_var(hierarchy, scopes, var);
        }
        for child in scope.scopes(hierarchy) {
            self.add_wellen_scope(hierarchy, scopes, child);
        }
        scopes.pop();
    }

    fn add_wellen_var(
        &mut self,
        hierarchy: &wellen::Hierarchy,
        scopes: &ScopeStack,
        var_ref: wellen::VarRef,
    ) {
        let var = &hierarchy[var_ref];
        if !matches!(var.signal_encoding(hierarchy), wellen::SignalEncoding::BitVector(w) if w > 0)
        {
            return;
        }
        let name = var.name(hierarchy);
        // `wellen` merges bit-blasted vectors into derived signals. Give the path to every
        // underlying signal, so that handle indices match the file's own handles.
        match hierarchy.get_derived_signal(var.signal_ref()) {
            Some(derived) => {
                for input in derived.inputs() {
                    self.add(input.index(), scopes, name, false);
                }
            }
            None => self.add(var.signal_ref().index(), scopes, name, false),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SelectionError {
    #[error("a module rule needs module names, but this waveform file has none")]
    NoModuleNames,
    #[error("invalid regular expression")]
    Regex(#[from] regex::Error),
    #[error(
        "invalid rule `{0}`; a rule is +KIND:VALUE (include) or -KIND:VALUE (exclude), where KIND \
         is scope, signal, regex, or module"
    )]
    BadRule(String),
}

/// What a rule matches. All string comparisons are exact; nothing is case-folded.
#[derive(Debug, Clone)]
pub enum Rule {
    /// A regular expression that must match the whole path.
    Regex(Regex),
    /// A scope and everything below it. The text matches a signal if it equals the names of
    /// some leading scopes of the signal, joined by `.`.
    ///
    /// A scope name can contain a dot (a Verilog escaped identifier), and the text cannot tell
    /// that name from two nested scopes. So `scope:a.b` selects the signals in a scope named
    /// `a.b` and the signals in the scope `b` inside `a`. `scope:a` selects the scope `a` and
    /// everything below it, but not a scope named `a.b`.
    Scope(String),
    /// One exact signal path.
    Signal(String),
    /// Every signal inside an instance of this module (definition name).
    Module(String),
}

/// True if the `.`-join of some non-empty prefix of `names` equals `path`.
fn scope_names_start_with(names: &[String], path: &str) -> bool {
    let mut rest = path;
    for (i, name) in names.iter().enumerate() {
        if i > 0 {
            match rest.strip_prefix('.') {
                Some(after_dot) => rest = after_dot,
                None => return false,
            }
        }
        match rest.strip_prefix(name.as_str()) {
            Some(after_name) => rest = after_name,
            None => return false,
        }
        if rest.is_empty() {
            return true;
        }
    }
    false
}

impl Rule {
    /// Parses `kind:value`, where kind is `scope`, `signal`, `regex`, or `module`.
    pub fn parse(spec: &str) -> Result<Rule, SelectionError> {
        let (kind, value) = spec
            .split_once(':')
            .ok_or_else(|| SelectionError::BadRule(spec.into()))?;
        match kind {
            "scope" => Ok(Rule::Scope(value.into())),
            "signal" => Ok(Rule::Signal(value.into())),
            "module" => Ok(Rule::Module(value.into())),
            "regex" => Ok(Rule::Regex(Regex::new(&format!("^(?:{value})$"))?)),
            _ => Err(SelectionError::BadRule(spec.into())),
        }
    }

    fn matches(&self, p: &SignalPath) -> bool {
        match self {
            Rule::Regex(r) => r.is_match(&p.path),
            Rule::Scope(s) => scope_names_start_with(&p.scope_names, s),
            Rule::Signal(s) => p.path == *s,
            Rule::Module(m) => p.modules.iter().any(|c| c == m),
        }
    }
}

/// Whether a rule adds signals to the selection or removes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Include,
    Exclude,
}

/// One rule of a [`Selection`], with the text it was parsed from.
#[derive(Debug, Clone)]
pub struct SelectionRule {
    pub action: Action,
    pub rule: Rule,
    /// The rule as the user wrote it, for example `+scope:tb.dut`. Used in messages.
    pub text: String,
}

/// An ordered list of include and exclude rules.
///
/// An empty list selects every handle that has a path. Otherwise the starting state is the
/// opposite of the first rule's action. Each rule in order then sets the state of every handle
/// that has at least one path that matches the rule. The last matching rule wins.
///
/// Examples: `+scope:tb.dut -signal:tb.dut.clk` selects the DUT without its clock. The clock
/// stays out when it has an alias outside `tb.dut`. `-scope:tb.dut.u_rng
/// +signal:tb.dut.u_rng.state` selects everything except the RNG, but keeps its state register.
#[derive(Debug, Clone, Default)]
pub struct Selection {
    pub rules: Vec<SelectionRule>,
}

/// The result of applying a [`Selection`] to a [`HierarchyIndex`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolution {
    /// One flag per handle index. Handles without a path are never selected.
    pub selected: Vec<bool>,
    /// The text of every rule that matches no handle. Such a rule is probably a typo.
    pub unmatched_rules: Vec<String>,
}

impl Selection {
    /// Selects every signal.
    pub fn all() -> Self {
        Self::default()
    }

    /// Parses rules of the form `+kind:value` (include) or `-kind:value` (exclude). The kinds
    /// are `scope`, `signal`, `regex`, and `module`.
    pub fn parse<S: AsRef<str>>(specs: &[S]) -> Result<Self, SelectionError> {
        let rules = specs
            .iter()
            .map(|spec| {
                let text = spec.as_ref();
                let (action, rest) = if let Some(rest) = text.strip_prefix('+') {
                    (Action::Include, rest)
                } else if let Some(rest) = text.strip_prefix('-') {
                    (Action::Exclude, rest)
                } else {
                    return Err(SelectionError::BadRule(text.into()));
                };
                let rule = Rule::parse(rest).map_err(|e| match e {
                    SelectionError::BadRule(_) => SelectionError::BadRule(text.into()),
                    other => other,
                })?;
                Ok(SelectionRule {
                    action,
                    rule,
                    text: text.into(),
                })
            })
            .collect::<Result<_, SelectionError>>()?;
        Ok(Selection { rules })
    }

    /// Applies the rules to `index`. Fails if a `module:` rule is used on a file without module
    /// names.
    pub fn resolve(&self, index: &HierarchyIndex) -> Result<Resolution, SelectionError> {
        let uses_modules = self.rules.iter().any(|r| matches!(r.rule, Rule::Module(_)));
        if uses_modules && !index.has_module_names {
            return Err(SelectionError::NoModuleNames);
        }
        let start = self
            .rules
            .first()
            .is_none_or(|first| first.action == Action::Exclude);
        let mut selected: Vec<bool> = index
            .paths
            .iter()
            .map(|paths| start && !paths.is_empty())
            .collect();
        let mut unmatched_rules = Vec::new();
        for r in &self.rules {
            let state = r.action == Action::Include;
            let mut matched = false;
            for (flag, paths) in selected.iter_mut().zip(&index.paths) {
                if paths.iter().any(|p| r.rule.matches(p)) {
                    *flag = state;
                    matched = true;
                }
            }
            if !matched {
                unmatched_rules.push(r.text.clone());
            }
        }
        Ok(Resolution {
            selected,
            unmatched_rules,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sp(path: &str, scope: &str, modules: &[&str]) -> SignalPath {
        SignalPath {
            path: path.into(),
            scope: scope.into(),
            scope_names: if scope.is_empty() {
                vec![]
            } else {
                scope.split('.').map(String::from).collect()
            },
            modules: modules.iter().map(|m| m.to_string()).collect(),
            is_alias: false,
        }
    }

    /// A small design. Handle 0 is a clock net with three aliases, as simulators write it.
    /// Handle 1 is a data signal. Handles 2 and 3 are in a random number generator.
    fn design() -> HierarchyIndex {
        HierarchyIndex {
            paths: vec![
                vec![
                    sp("tb.clk", "tb", &["tb_top"]),
                    sp("tb.dut.clk", "tb.dut", &["tb_top", "core"]),
                    sp("tb.dut.u_a.ck", "tb.dut.u_a", &["tb_top", "core", "adder"]),
                ],
                vec![sp("tb.dut.data", "tb.dut", &["tb_top", "core"])],
                vec![sp(
                    "tb.dut.u_rng.state",
                    "tb.dut.u_rng",
                    &["tb_top", "core", "rng"],
                )],
                vec![sp(
                    "tb.dut.u_rng.mask",
                    "tb.dut.u_rng",
                    &["tb_top", "core", "rng"],
                )],
                vec![],
            ],
            has_module_names: true,
        }
    }

    fn resolve(rules: &[&str]) -> Resolution {
        Selection::parse(rules).unwrap().resolve(&design()).unwrap()
    }

    #[test]
    fn strip_bit_range_removes_only_a_trailing_range() {
        assert_eq!(strip_bit_range("data [31:0]"), "data");
        assert_eq!(strip_bit_range("bit [7]"), "bit");
        assert_eq!(strip_bit_range("op1[31:0]"), "op1");
        assert_eq!(strip_bit_range("runner[0:20]"), "runner");
        assert_eq!(strip_bit_range("mem[3]"), "mem[3]");
        assert_eq!(strip_bit_range("mem[3] [7:0]"), "mem[3]");
        assert_eq!(strip_bit_range("weird [a:b]"), "weird [a:b]");
        assert_eq!(strip_bit_range("plain"), "plain");
        assert_eq!(strip_bit_range("open[3"), "open[3");
    }

    #[test]
    fn strip_bit_range_keeps_a_name_that_is_only_a_range() {
        // `wellen` splits `tdata[1714295607408:1714295607407]` into the scope `tdata` and
        // the variable name `[1714295607408:1714295607407]`.
        assert_eq!(strip_bit_range("[3:0]"), "[3:0]");
        assert_eq!(
            strip_bit_range("[1714295607408:1714295607407]"),
            "[1714295607408:1714295607407]"
        );
        assert_eq!(strip_bit_range(" [3:0]"), " [3:0]");
        // Only white space before the range counts as nothing.
        assert_eq!(strip_bit_range("  [3:0]"), "  [3:0]");
        assert_eq!(strip_bit_range("\t[3:0]"), "\t[3:0]");
    }

    #[test]
    fn a_name_that_is_only_a_range_does_not_leave_a_trailing_dot_in_the_path() {
        let mut scopes = ScopeStack::default();
        scopes.push("tdata", "");
        let mut index = HierarchyIndex::default();
        index.add(0, &scopes, "[1714295607408:1714295607407]", false);
        assert_eq!(
            index.paths[0][0].path,
            "tdata.[1714295607408:1714295607407]"
        );
        assert_eq!(index.paths[0][0].scope, "tdata");
    }

    #[test]
    fn empty_selection_selects_every_handle_that_has_a_path() {
        let r = resolve(&[]);
        assert_eq!(r.selected, vec![true, true, true, true, false]);
        assert!(r.unmatched_rules.is_empty());
    }

    #[test]
    fn an_alias_inside_an_included_scope_selects_the_handle() {
        assert_eq!(
            resolve(&["+scope:tb.dut.u_a"]).selected,
            vec![true, false, false, false, false]
        );
        assert_eq!(resolve(&["+scope:tb.other"]).selected, vec![false; 5],);
    }

    #[test]
    fn a_later_exclude_wins_even_if_an_alias_is_inside_the_included_scope() {
        // The clock has an alias in tb.dut, but `signal:tb.dut.clk` excludes it by name.
        let r = resolve(&["+scope:tb.dut", "-signal:tb.dut.clk"]);
        assert_eq!(r.selected, vec![false, true, true, true, false]);
        assert!(r.unmatched_rules.is_empty());
    }

    #[test]
    fn an_exclude_by_the_alias_outside_the_scope_also_removes_the_handle() {
        // `tb.clk` is another path of the clock net that `tb.dut.clk` names.
        let r = resolve(&["+scope:tb.dut", "-signal:tb.clk"]);
        assert_eq!(r.selected, vec![false, true, true, true, false]);
    }

    #[test]
    fn an_exclude_first_starts_from_everything() {
        let r = resolve(&["-scope:tb.dut.u_rng", "+signal:tb.dut.u_rng.state"]);
        // Handle 4 has no path and stays unselected.
        assert_eq!(r.selected, vec![true, true, true, false, false]);
    }

    #[test]
    fn the_last_matching_rule_wins() {
        let r = resolve(&["+scope:tb", "-scope:tb.dut", "+regex:tb\\.dut\\.data"]);
        // The clock net still has the path tb.clk, which only the first rule matches, but it
        // also has paths in tb.dut, which the second rule matches.
        assert_eq!(r.selected, vec![false, true, false, false, false]);
    }

    #[test]
    fn unmatched_rules_are_reported_in_order() {
        let r = resolve(&["+scope:tb.dut", "-signal:tb.nothing", "+regex:zzz.*"]);
        assert_eq!(
            r.unmatched_rules,
            vec!["-signal:tb.nothing", "+regex:zzz.*"]
        );
        assert_eq!(r.selected, vec![true, true, true, true, false]);
    }

    /// Three handles: `x` in the literal scope `a.b` (an escaped identifier), `y` in the scope
    /// `b` inside `a`, and `v` in `a`.
    fn dotted_design() -> HierarchyIndex {
        let named = |path: &str, names: &[&str]| SignalPath {
            path: path.into(),
            scope: names.join("."),
            scope_names: names.iter().map(|n| n.to_string()).collect(),
            modules: vec![String::new(); names.len()],
            is_alias: false,
        };
        HierarchyIndex {
            paths: vec![
                vec![named("a.b.x", &["a.b"])],
                vec![named("a.b.y", &["a", "b"])],
                vec![named("a.v", &["a"])],
            ],
            has_module_names: false,
        }
    }

    fn select_dotted(rules: &[&str]) -> Vec<bool> {
        Selection::parse(rules)
            .unwrap()
            .resolve(&dotted_design())
            .unwrap()
            .selected
    }

    #[test]
    fn a_scope_rule_does_not_split_a_scope_name_that_contains_a_dot() {
        // `a` is the outer scope of `y` and `v`. It is not a prefix of the scope `a.b`.
        assert_eq!(select_dotted(&["+scope:a"]), [false, true, true]);
        // The text `a.b` names both the literal scope and the scope `b` inside `a`.
        assert_eq!(select_dotted(&["+scope:a.b"]), [true, true, false]);
        assert_eq!(select_dotted(&["+scope:a.b.x"]), [false; 3]);
        assert_eq!(select_dotted(&["+scope:a."]), [false; 3]);
        assert_eq!(select_dotted(&["+scope:"]), [false; 3]);
    }

    #[test]
    fn signal_and_regex_rules_compare_the_joined_path() {
        assert_eq!(select_dotted(&["+signal:a.b.x"]), [true, false, false]);
        assert_eq!(select_dotted(&["+regex:a\\.b\\..*"]), [true, true, false]);
    }

    #[test]
    fn scope_rule_respects_name_boundaries() {
        assert_eq!(resolve(&["+scope:tb.du"]).selected, vec![false; 5]);
        assert_eq!(resolve(&["+scope:tb.dut.u_"]).selected, vec![false; 5]);
        assert_eq!(
            resolve(&["+scope:tb.dut.u_rng"]).selected,
            vec![false, false, true, true, false]
        );
        // The whole subtree, including the net with a path directly in `tb`.
        assert_eq!(
            resolve(&["+scope:tb"]).selected,
            vec![true, true, true, true, false]
        );
    }

    #[test]
    fn regex_rule_must_match_the_whole_path() {
        assert_eq!(resolve(&["+regex:dut"]).selected, vec![false; 5]);
        assert_eq!(
            resolve(&["+regex:tb\\.dut\\.u_rng\\..*"]).selected,
            vec![false, false, true, true, false]
        );
        // The alternation is anchored as a whole, not per branch.
        assert!(resolve(&["+regex:tb\\.clk|nothing"]).selected[0]);
        assert_eq!(resolve(&["+regex:tb|nothing"]).selected, vec![false; 5]);
    }

    #[test]
    fn module_rule_matches_any_enclosing_instance() {
        assert_eq!(
            resolve(&["+module:adder"]).selected,
            vec![true, false, false, false, false]
        );
        assert_eq!(
            resolve(&["+module:rng"]).selected,
            vec![false, false, true, true, false]
        );
    }

    #[test]
    fn module_rule_fails_without_module_names() {
        let mut index = design();
        index.has_module_names = false;
        let s = Selection::parse(&["+module:adder"]).unwrap();
        assert!(matches!(
            s.resolve(&index),
            Err(SelectionError::NoModuleNames)
        ));
        // Other kinds are fine.
        let s = Selection::parse(&["+scope:tb"]).unwrap();
        assert!(s.resolve(&index).is_ok());
    }

    #[test]
    fn parse_accepts_all_kinds_and_keeps_the_order_and_text() {
        let s =
            Selection::parse(&["+scope:a.b", "-signal:a.b.c", "+regex:x|y", "-module:m"]).unwrap();
        let actions: Vec<_> = s.rules.iter().map(|r| r.action).collect();
        assert_eq!(
            actions,
            vec![
                Action::Include,
                Action::Exclude,
                Action::Include,
                Action::Exclude
            ]
        );
        assert!(matches!(&s.rules[0].rule, Rule::Scope(v) if v == "a.b"));
        assert!(matches!(&s.rules[1].rule, Rule::Signal(v) if v == "a.b.c"));
        assert!(matches!(&s.rules[2].rule, Rule::Regex(_)));
        assert!(matches!(&s.rules[3].rule, Rule::Module(v) if v == "m"));
        assert_eq!(s.rules[1].text, "-signal:a.b.c");
    }

    #[test]
    fn parse_rejects_bad_rules() {
        // No sign.
        assert!(matches!(
            Selection::parse(&["scope:tb"]),
            Err(SelectionError::BadRule(t)) if t == "scope:tb"
        ));
        // Unknown kind; the message shows the whole rule.
        assert!(matches!(
            Selection::parse(&["+glob:a*"]),
            Err(SelectionError::BadRule(t)) if t == "+glob:a*"
        ));
        // No colon.
        assert!(matches!(
            Selection::parse(&["+scope"]),
            Err(SelectionError::BadRule(_))
        ));
        assert!(matches!(
            Selection::parse(&["+regex:("]),
            Err(SelectionError::Regex(_))
        ));
        assert!(Selection::parse::<&str>(&[]).unwrap().rules.is_empty());
    }
}
