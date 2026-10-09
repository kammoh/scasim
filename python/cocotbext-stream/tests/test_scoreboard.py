"""Scoreboard matching and reports, without a simulator."""

import pytest

from cocotbext.stream import Scoreboard


def make(**kwargs):
    board = Scoreboard(**kwargs)
    return board, board.add_channel("out")


def test_in_order_match():
    board, ch = make()
    ch.expect_many([{"data": 1}, {"data": 2}])
    ch.observe({"data": 1})
    ch.observe({"data": 2})
    assert ch.matched == 2
    assert ch.pending == 0
    board.check()


def test_mismatch_report_is_field_level():
    board, ch = make(fail_fast=False)
    ch.expect({"data": 0x1235, "last": 1, "vec": [1, 3]})
    ch.observe({"data": 0x1234, "last": 1, "vec": [1, 2]})
    assert len(board.mismatches) == 1
    diffs = {d.field: (d.expected, d.observed) for d in board.mismatches[0].diffs}
    assert diffs == {"data": (0x1235, 0x1234), "vec[1]": (3, 2)}
    with pytest.raises(AssertionError) as info:
        board.check()
    text = str(info.value)
    assert "out" in text
    assert "data: expected 0x1235, got 0x1234" in text
    assert "vec[1]: expected 3, got 2" in text
    assert "last" not in text


def test_fail_fast_raises_at_the_mismatch():
    board, ch = make()
    ch.expect({"data": 1})
    with pytest.raises(AssertionError, match="data: expected 1, got 2"):
        ch.observe({"data": 2})
    assert len(board.mismatches) == 1


def test_none_in_expected_is_dont_care():
    board, ch = make()
    ch.expect({"data": 1, "last": None})
    ch.observe({"data": 1, "last": 0})
    board.check()


def test_expected_may_use_a_subset_of_fields():
    board, ch = make()
    ch.expect({"data": 1})
    ch.observe({"data": 1, "last": 0, "vec": [4, 5]})
    board.check()


def test_missing_observed_field():
    board, ch = make(fail_fast=False)
    ch.expect({"data": 1, "last": 0})
    ch.observe({"data": 1})
    assert [d.field for d in board.mismatches[0].diffs] == ["last"]
    assert "not in the observed item" in str(board.mismatches[0])


def test_array_length_mismatch():
    board, ch = make(fail_fast=False)
    ch.expect({"vec": [1, 2, 3]})
    ch.observe({"vec": [1, 2]})
    assert [d.field for d in board.mismatches[0].diffs] == ["vec"]


def test_expected_logic_like_values_compare_as_ints():
    class Intish:
        def __int__(self):
            return 7

    board, ch = make()
    ch.expect({"data": Intish()})
    ch.observe({"data": 7})
    board.check()


def test_unresolved_observed_value_is_a_mismatch():
    board, ch = make(fail_fast=False)
    ch.expect({"data": 3})
    ch.observe({"data": "00x1"})
    assert "00x1" in str(board.mismatches[0])


def test_unexpected_item_in_order():
    board, ch = make(fail_fast=False)
    ch.observe({"data": 9})
    assert len(board.unexpected) == 1
    with pytest.raises(AssertionError, match="unexpected"):
        board.check()


def test_unexpected_item_fail_fast():
    board, ch = make()
    with pytest.raises(AssertionError, match="unexpected"):
        ch.observe({"data": 9})


def test_missing_items_at_completion():
    board, ch = make()
    ch.expect_many([{"data": 1}, {"data": 2}, {"data": 3}])
    ch.observe({"data": 1})
    assert board.pending == 2
    assert [tx for _, tx in board.missing] == [{"data": 2}, {"data": 3}]
    with pytest.raises(AssertionError, match="missing") as info:
        board.check()
    assert "data: 2" in str(info.value)


def test_out_of_order_needs_a_key():
    with pytest.raises(ValueError):
        Scoreboard(in_order=False)


def test_out_of_order_match():
    board, ch = make(in_order=False, key=lambda t: t["id"])
    ch.expect_many([{"id": 1, "data": 10}, {"id": 2, "data": 20}, {"id": 3, "data": 30}])
    ch.observe({"id": 3, "data": 30})
    ch.observe({"id": 1, "data": 10})
    ch.observe({"id": 2, "data": 20})
    assert ch.matched == 3
    board.check()


def test_out_of_order_field_mismatch_for_a_known_key():
    board, ch = make(in_order=False, key=lambda t: t["id"], fail_fast=False)
    ch.expect({"id": 1, "data": 10})
    ch.observe({"id": 1, "data": 11})
    assert [d.field for d in board.mismatches[0].diffs] == ["data"]


def test_out_of_order_duplicate_is_unexpected():
    board, ch = make(in_order=False, key=lambda t: t["id"], fail_fast=False)
    ch.expect({"id": 1})
    ch.observe({"id": 1})
    ch.observe({"id": 1})
    assert len(board.unexpected) == 1
    with pytest.raises(AssertionError, match="unexpected"):
        board.check()


def test_out_of_order_unknown_key_is_unexpected():
    board, ch = make(in_order=False, key=lambda t: t["id"], fail_fast=False)
    ch.expect({"id": 1})
    ch.observe({"id": 5})
    assert len(board.unexpected) == 1
    assert board.pending == 1


def test_out_of_order_same_key_expected_twice_matches_twice():
    board, ch = make(in_order=False, key=lambda t: t["id"])
    ch.expect_many([{"id": 1, "data": 10}, {"id": 1, "data": 11}])
    ch.observe({"id": 1, "data": 10})
    ch.observe({"id": 1, "data": 11})
    board.check()


def test_out_of_order_missing_items():
    board, ch = make(in_order=False, key=lambda t: t["id"])
    ch.expect_many([{"id": 1}, {"id": 2}])
    ch.observe({"id": 2})
    assert [tx for _, tx in board.missing] == [{"id": 1}]
    with pytest.raises(AssertionError, match="missing"):
        board.check()


def test_channels_are_independent():
    board = Scoreboard()
    a = board.add_channel("a")
    b = board.add_channel("b")
    a.expect({"x": 1})
    b.expect({"x": 2})
    b.observe({"x": 2})
    a.observe({"x": 1})
    board.check()


def test_none_element_in_an_expected_array_is_dont_care():
    board, ch = make()
    ch.expect({"vec": [1, None, 3]})
    ch.observe({"vec": [1, 99, 3]})
    board.check()
