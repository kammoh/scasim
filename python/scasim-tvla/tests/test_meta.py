import gzip
import json
import os

import pytest

from scasim_tvla.meta import MetaError, MetaWriter, read_meta


def make(tmp_path, name="meta.json", **kw):
    return MetaWriter(tmp_path / name, **kw)


def test_round_trip_schema(tmp_path):
    w = make(
        tmp_path,
        time=(1, -12),
        seeds={"base": 1234567, "schedule": 99},
        batch_id="b0007",
        design={"toplevel": "Top", "config": "sha256:abc"},
        waveform="tvla.fst",
        design_random={"requested": "off", "applied": "off", "how": "hook: h"},
        extensions={"x": 1},
    )
    w.segment(100, 200, 0)
    w.segment(200, 300, 1, group=0)
    w.commit()
    d = json.loads((tmp_path / "meta.json").read_text())
    assert d == {
        "scasim_meta": 1,
        "batch": {
            "id": "b0007",
            "seeds": {"base": 1234567, "schedule": 99},
            "design_random": {"requested": "off", "applied": "off", "how": "hook: h"},
            "status": "committed",
        },
        "design": {"toplevel": "Top", "config": "sha256:abc"},
        "waveform": "tvla.fst",
        "time": {"mantissa": 1, "exponent": -12},
        "segments": [
            {"id": 0, "start": 100, "end": 200, "label": 0, "group": 0},
            {"id": 1, "start": 200, "end": 300, "label": 1, "group": 0},
        ],
        "labels": {"0": "fixed", "1": "random"},
        "groups": {"0": "default"},
        "extensions": {"x": 1},
    }
    assert read_meta(tmp_path / "meta.json") == d


def test_optional_keys_absent(tmp_path):
    w = make(tmp_path)
    w.segment(0, 1, 0)
    w.commit()
    d = read_meta(tmp_path / "meta.json")
    assert "waveform" not in d
    assert "design" not in d
    assert d["extensions"] == {}


def test_gzip(tmp_path):
    w = make(tmp_path, "meta.json.gz")
    w.segment(0, 5, 1)
    w.commit()
    raw = (tmp_path / "meta.json.gz").read_bytes()
    assert raw[:2] == b"\x1f\x8b"
    assert json.loads(gzip.decompress(raw))["segments"][0]["end"] == 5
    assert read_meta(tmp_path / "meta.json.gz")["scasim_meta"] == 1


def test_diagnostic_status(tmp_path):
    w = make(tmp_path)
    w.segment(0, 5, 0)
    w.write_diagnostic()
    assert read_meta(tmp_path / "meta.json")["batch"]["status"] == "diagnostic"


def test_diagnostic_may_be_empty(tmp_path):
    w = make(tmp_path)
    w.write_diagnostic()
    assert read_meta(tmp_path / "meta.json")["segments"] == []


def test_commit_without_segments_fails(tmp_path):
    w = make(tmp_path)
    with pytest.raises(MetaError):
        w.commit()
    assert not (tmp_path / "meta.json").exists()


@pytest.mark.parametrize("start,end", [(1.0, 2), (1, 2.5), ("1", 2), (True, 2), (None, 2)])
def test_times_must_be_int(tmp_path, start, end):
    with pytest.raises(MetaError):
        make(tmp_path).segment(start, end, 0)


@pytest.mark.parametrize("start,end", [(5, 5), (6, 5), (-1, 3)])
def test_end_after_start_and_nonnegative(tmp_path, start, end):
    with pytest.raises(MetaError):
        make(tmp_path).segment(start, end, 0)


def test_segments_in_increasing_order_without_overlap(tmp_path):
    w = make(tmp_path)
    w.segment(10, 20, 0)
    with pytest.raises(MetaError):
        w.segment(15, 30, 0)  # overlaps
    with pytest.raises(MetaError):
        w.segment(5, 8, 0)  # before
    w.segment(20, 30, 0)  # adjacent is fine


def test_numpy_style_ints_accepted(tmp_path):
    class I:
        def __index__(self):
            return 7

    w = make(tmp_path)
    w.segment(I(), 9, 0)
    w.commit()
    assert read_meta(tmp_path / "meta.json")["segments"][0]["start"] == 7


@pytest.mark.parametrize("label", [-1, 65536, 2, "0", 0.0])
def test_label_must_be_declared_and_in_range(tmp_path, label):
    with pytest.raises(MetaError):
        make(tmp_path).segment(0, 1, label)


def test_label_range_limits(tmp_path):
    w = make(tmp_path, labels={0: "a", 65535: "b"})
    w.segment(0, 1, 65535)
    with pytest.raises(MetaError):
        make(tmp_path, labels={65536: "x"})
    with pytest.raises(MetaError):
        make(tmp_path, labels={})


def test_group_must_be_declared(tmp_path):
    w = make(tmp_path, groups={0: "default", 3: "other"})
    w.segment(0, 1, 0, group=3)
    with pytest.raises(MetaError):
        w.segment(1, 2, 0, group=1)


def test_time_validation(tmp_path):
    with pytest.raises(MetaError):
        make(tmp_path, time=(0, -12))
    with pytest.raises(MetaError):
        make(tmp_path, time=(1, -12.0))
    w = make(tmp_path, time=(10, -9))
    w.segment(0, 1, 0)
    w.commit()
    assert read_meta(tmp_path / "meta.json")["time"] == {"mantissa": 10, "exponent": -9}


def test_seeds_must_be_u64_ints(tmp_path):
    with pytest.raises(MetaError):
        make(tmp_path, seeds={"base": -1})
    with pytest.raises(MetaError):
        make(tmp_path, seeds={"base": 2**64})
    with pytest.raises(MetaError):
        make(tmp_path, seeds={"base": 1.5})
    make(tmp_path, seeds={"base": 2**64 - 1})


def test_stable_ids_with_skipped_warmup(tmp_path):
    w = make(tmp_path)
    w.skip_ids(2)
    assert w.segment(0, 1, 0) == 2
    assert w.segment(1, 2, 1) == 3
    w.commit()
    ids = [s["id"] for s in read_meta(tmp_path / "meta.json")["segments"]]
    assert ids == [2, 3]


def test_skip_after_segment_fails(tmp_path):
    w = make(tmp_path)
    w.segment(0, 1, 0)
    with pytest.raises(MetaError):
        w.skip_ids(1)


def test_finished_writer_is_closed(tmp_path):
    w = make(tmp_path)
    w.segment(0, 1, 0)
    w.commit()
    with pytest.raises(MetaError):
        w.segment(1, 2, 0)
    with pytest.raises(MetaError):
        w.commit()
    with pytest.raises(MetaError):
        w.write_diagnostic()


def test_atomic_write_leaves_old_file_on_failure(tmp_path, monkeypatch):
    p = tmp_path / "meta.json"
    p.write_text("OLD")
    w = make(tmp_path)
    w.segment(0, 1, 0)

    def boom(src, dst):
        raise OSError("rename failed")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        w.commit()
    assert p.read_text() == "OLD"
    assert sorted(x.name for x in tmp_path.iterdir()) == ["meta.json"]  # temp file removed


def test_no_partial_file_visible_before_commit(tmp_path):
    w = make(tmp_path)
    w.segment(0, 1, 0)
    assert list(tmp_path.iterdir()) == []


def test_creates_parent_directory(tmp_path):
    w = MetaWriter(tmp_path / "a" / "b" / "meta.json")
    w.segment(0, 1, 0)
    w.commit()
    assert (tmp_path / "a" / "b" / "meta.json").exists()


def test_read_meta_rejects_other_versions(tmp_path):
    (tmp_path / "m.json").write_text('{"scasim_meta": 2}')
    with pytest.raises(MetaError):
        read_meta(tmp_path / "m.json")


def test_set_design_random(tmp_path):
    w = make(tmp_path)
    w.set_design_random({"requested": "on", "applied": "on", "how": "hook: h"})
    w.segment(0, 1, 0)
    w.commit()
    assert read_meta(tmp_path / "meta.json")["batch"]["design_random"]["how"] == "hook: h"


def test_file_permissions_follow_umask(tmp_path):
    old = os.umask(0o022)
    try:
        w = make(tmp_path)
        w.segment(0, 1, 0)
        w.commit()
    finally:
        os.umask(old)
    assert (tmp_path / "meta.json").stat().st_mode & 0o777 == 0o644
