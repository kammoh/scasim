//! Tests that the memory guard fails BEFORE a large allocation, not after it.
//!
//! This binary counts every allocation of the process with a global allocator and records the
//! peak. A test runs a call that must fail early, and compares its peak with the peak of a
//! call that has to do the work. The tests share the counters, so each test takes a lock first.
//!
//! The time table of the file is read before any check can run. The tests do not count it:
//! they compare peaks, never absolute sizes.

use fst_writer::{
    FstFileType, FstInfo, FstScopeType, FstSignalType, FstVarDirection, FstVarType, open_fst,
};
use scasim::hierarchy::Selection;
use scasim::power::reference::{activity_reference, activity_reference_binned};
use scasim::power::{Bins, PowerError, PowerPlan, activity, activity_binned, hierarchy_index};
use std::alloc::{GlobalAlloc, Layout, System};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering::Relaxed};

/// Wraps the system allocator. It tracks the number of live bytes, their peak, and the number of
/// large requests.
struct CountingAllocator;

/// A request for at least this many bytes is large. The file of the tests has a time table of
/// 8 MB, and every copy of it is a large request.
const LARGE: usize = 1_000_000;

static LIVE: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicIsize = AtomicIsize::new(0);
static LARGE_REQUESTS: AtomicUsize = AtomicUsize::new(0);

fn grow(bytes: usize) {
    let now = LIVE.fetch_add(bytes as isize, Relaxed) + bytes as isize;
    PEAK.fetch_max(now, Relaxed);
}

fn count_large(size: usize) {
    if size >= LARGE {
        LARGE_REQUESTS.fetch_add(1, Relaxed);
    }
}

fn shrink(bytes: usize) {
    LIVE.fetch_sub(bytes as isize, Relaxed);
}

// SAFETY: every method forwards to `System` with the same arguments. The counters have no
// influence on the memory that the methods return.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            grow(layout.size());
            count_large(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            grow(layout.size());
            count_large(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        shrink(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            if new_size >= layout.size() {
                grow(new_size - layout.size());
                count_large(new_size);
            } else {
                shrink(layout.size() - new_size);
            }
        }
        new
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Only one test at a time may measure, because the counters are global.
static LOCK: Mutex<()> = Mutex::new(());

fn serialized() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// What a call allocated.
#[derive(Debug, Clone, Copy)]
struct Usage {
    /// The most bytes that were live at one time during the call, above the bytes that were live
    /// before it.
    peak: usize,
    /// The number of large requests (`LARGE` bytes or more).
    large: usize,
}

/// Runs `f` and returns its result and what it allocated.
fn measure<T>(f: impl FnOnce() -> T) -> (T, Usage) {
    let start = LIVE.load(Relaxed);
    PEAK.store(start, Relaxed);
    let large_before = LARGE_REQUESTS.load(Relaxed);
    let result = f();
    let usage = Usage {
        peak: (PEAK.load(Relaxed) - start).max(0) as usize,
        large: LARGE_REQUESTS.load(Relaxed) - large_before,
    };
    (result, usage)
}

/// Runs `f` in a rayon pool with one thread, so that the thread count does not change the
/// estimates.
fn single_threaded<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(f)
}

const STEPS: u64 = 1_000_000;

/// An FST file with one 1-bit signal that toggles at each of the times 1 to `STEPS`. The time
/// table has `STEPS + 1` distinct times (time 0 comes first). There is one section.
fn toggling_fst(dir: &Path) -> PathBuf {
    let path = dir.join("toggling.fst");
    let info = FstInfo {
        start_time: 0,
        timescale_exponent: -12,
        version: "scasim test".into(),
        date: "2026-10-08".into(),
        file_type: FstFileType::Verilog,
    };
    let mut header = open_fst(&path, &info).unwrap();
    header.scope("tb", "", FstScopeType::Module).unwrap();
    let id = header
        .var(
            "s0",
            FstSignalType::bit_vec(1),
            FstVarType::Wire,
            FstVarDirection::Implicit,
            None,
        )
        .unwrap();
    header.up_scope().unwrap();
    let mut body = header.finish().unwrap();
    body.signal_change(id, b"0").unwrap();
    for time in 1..=STEPS {
        body.time_change(time).unwrap();
        body.signal_change(id, if time % 2 == 1 { b"1" } else { b"0" })
            .unwrap();
    }
    body.finish().unwrap();
    path
}

fn toggle_plan(memory_limit: u64) -> PowerPlan {
    let mut plan = PowerPlan::toggles(Selection::all());
    plan.memory_limit = memory_limit;
    plan
}

#[test]
fn the_fixture_has_one_section_and_distinct_times() {
    let _lock = serialized();
    let dir = tempfile::tempdir().unwrap();
    let path = toggling_fst(dir.path());
    let reader = fst_reader::FstReader::open_and_read_time_table(std::io::BufReader::new(
        std::fs::File::open(&path).unwrap(),
    ))
    .unwrap();
    let table = reader.get_time_table().unwrap();
    assert_eq!(table.len() as u64, STEPS + 1);
    assert!(table.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(reader.sections().len(), 1);
}

/// The identity bins are a copy of the time table: 8 MB for this file. The guard must refuse the
/// run before it makes the copy.
#[test]
fn the_identity_bins_are_not_allocated_when_the_result_is_too_large() {
    let _lock = serialized();
    let dir = tempfile::tempdir().unwrap();
    let path = toggling_fst(dir.path());
    // The baseline reads the time table and the hierarchy, and nothing else.
    let (index, baseline) = measure(|| hierarchy_index(&path).unwrap());
    drop(index);
    let (result, used) = measure(|| single_threaded(|| activity(&path, &toggle_plan(1024))));
    assert!(
        matches!(&result, Err(PowerError::Memory { limit: 1024, .. })),
        "{result:?}"
    );
    // A copy of the time table would be one more large request. (The peak does not show it:
    // while the reader decodes the time table, it holds about as much as the copy would add.)
    assert!(
        used.large <= baseline.large,
        "used {used:?}, baseline {baseline:?}"
    );
    assert!(
        used.peak <= baseline.peak + 1_000_000,
        "used {used:?}, baseline {baseline:?}"
    );
}

/// The reference path makes the same copy of the time table for its identity bins.
#[test]
fn the_reference_path_does_not_allocate_identity_bins_when_the_result_is_too_large() {
    let _lock = serialized();
    let dir = tempfile::tempdir().unwrap();
    let path = toggling_fst(dir.path());
    // The baseline reads the header and the body (with the time table) with `wellen`.
    let options = wellen::LoadOptions {
        multi_thread: true,
        remove_scopes_with_empty_name: false,
    };
    let ((), baseline) = measure(|| {
        let header = wellen::viewers::read_header_from_file(&path, &options).unwrap();
        let body = wellen::viewers::read_body(header.body, &header.hierarchy, None).unwrap();
        drop(body);
    });
    let (result, used) = measure(|| activity_reference(&path, &toggle_plan(1024)));
    assert!(
        matches!(&result, Err(PowerError::Memory { limit: 1024, .. })),
        "{result:?}"
    );
    assert!(
        used.large <= baseline.large,
        "used {used:?}, baseline {baseline:?}"
    );
    assert!(
        used.peak <= baseline.peak + 1_000_000,
        "used {used:?}, baseline {baseline:?}"
    );
}

/// With user bins, the time table of the section is the only large input. The placement of its
/// time points needs memory, too. The guard must refuse the run before it allocates it.
#[test]
fn the_placement_is_not_allocated_when_it_is_too_large() {
    let _lock = serialized();
    let dir = tempfile::tempdir().unwrap();
    let path = toggling_fst(dir.path());
    let bins = Bins::new(vec![0], None).unwrap();

    let (result, refused) =
        measure(|| single_threaded(|| activity_binned(&path, &toggle_plan(1024), &bins)));
    let Err(PowerError::Memory { what, .. }) = &result else {
        panic!("expected a memory error, got {result:?}");
    };
    assert!(what.starts_with("the placement"), "{what}");

    let (result, done) = measure(|| {
        single_threaded(|| {
            activity_binned(&path, &toggle_plan(PowerPlan::DEFAULT_MEMORY_LIMIT), &bins)
        })
    });
    assert_eq!(result.unwrap().channels[0].toggles, vec![STEPS]);
    // The placement of 1 million time points needs at least 4 MB. The run that does the work has
    // it, and the refused run does not.
    assert!(
        refused.large < done.large,
        "refused {refused:?}, done {done:?}"
    );
    assert!(
        refused.peak + 3_000_000 <= done.peak,
        "refused {refused:?}, done {done:?}"
    );
}

/// The reference path (`wellen`) loads every selected signal completely. With user bins, the
/// result is tiny, so only the estimate for the loaded signals can refuse the run.
#[test]
fn the_reference_path_does_not_load_signals_when_they_are_too_large() {
    let _lock = serialized();
    let dir = tempfile::tempdir().unwrap();
    let path = toggling_fst(dir.path());
    let bins = Bins::new(vec![0], None).unwrap();

    let (result, refused) = measure(|| activity_reference_binned(&path, &toggle_plan(1024), &bins));
    let Err(PowerError::Memory { what, .. }) = &result else {
        panic!("expected a memory error, got {result:?}");
    };
    assert!(
        what.starts_with("the largest signal that wellen loads"),
        "{what}"
    );

    let (result, done) = measure(|| {
        activity_reference_binned(&path, &toggle_plan(PowerPlan::DEFAULT_MEMORY_LIMIT), &bins)
    });
    assert_eq!(result.unwrap().channels[0].toggles, vec![STEPS]);
    // Loading the signal needs at least 5 bytes for each of the 1 million changes.
    assert!(
        refused.peak + 4_000_000 <= done.peak,
        "refused {refused:?}, done {done:?}"
    );
}
