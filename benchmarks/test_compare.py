"""Tests for the benchmark comparator.

``compare.py`` decides whether a pull request passes, so its decision logic is
worth pinning down. Everything here runs on synthetic numbers -- no benchmark
is executed -- which is why these are ordinary tests rather than benchmarks and
run in a plain ``pytest benchmarks/``.

The cases that matter are the boundaries: a ratio just under the threshold, a
ratio just over it, and a ratio over it on a quantity too small to care about.
"""
import json
from types import SimpleNamespace

import pytest

from compare import compare, compare_one, main

# The production defaults, so the tests fail if those move without thought.
ARGS = SimpleNamespace(time_threshold=2.0, mem_threshold=1.15,
                       time_floor_ms=1.0, mem_floor_mb=1.0)


def bench(name, time_ms, mem_mb=1000.0):
    """One entry as pytest-benchmark writes it, keyed by full name."""
    entry = {'name': name, 'fullname': f'benchmarks/test_predict.py::{name}',
             'stats': {'min': time_ms / 1e3}, 'extra_info': {}}
    if mem_mb is not None:
        entry['extra_info']['peak_mem_mb'] = mem_mb
    return {entry['fullname']: entry}


def run(baseline, contender, args=ARGS):
    return compare(baseline, contender, args)


def test_time_just_under_threshold_passes():
    """1.9x is a lot, but the threshold is 2x and noise reaches 30%."""
    _, _, _, failed = run(bench('a', 10.0), bench('a', 19.0))
    assert not failed


def test_time_exactly_at_threshold_passes():
    """The comparison is ``>`` threshold, so landing exactly on it is fine."""
    rows, _, _, failed = run(bench('a', 10.0), bench('a', 20.0))
    assert not failed
    assert rows[0]['t_status'] == 'ok'


def test_time_over_threshold_and_over_floor_fails():
    rows, _, _, failed = run(bench('a', 10.0), bench('a', 30.0))
    assert failed
    assert rows[0]['t_status'] == 'regressed'


def test_time_over_threshold_but_under_floor_is_reported_not_failed():
    """0.2 ms -> 0.8 ms is 4x, and still nobody's problem."""
    rows, _, _, failed = run(bench('a', 0.2), bench('a', 0.8))
    assert not failed
    assert rows[0]['t_status'] == 'under-floor'
    assert rows[0]['t_ratio'] == pytest.approx(4.0)


def test_memory_over_threshold_and_over_floor_fails():
    rows, _, _, failed = run(bench('a', 10.0, mem_mb=100.0),
                             bench('a', 10.0, mem_mb=200.0))
    assert failed
    assert rows[0]['m_status'] == 'regressed'


def test_memory_under_threshold_passes():
    """Memory is repeatable, so the gate is tight -- but 10% still clears."""
    _, _, _, failed = run(bench('a', 10.0, mem_mb=100.0),
                          bench('a', 10.0, mem_mb=110.0))
    assert not failed


def test_memory_over_threshold_but_under_floor_is_reported_not_failed():
    rows, _, _, failed = run(bench('a', 10.0, mem_mb=0.064),
                             bench('a', 10.0, mem_mb=0.512))
    assert not failed
    assert rows[0]['m_status'] == 'under-floor'


def test_missing_memory_is_skipped_cleanly():
    """A baseline predating peak_mem_mb still gets compared on time."""
    rows, _, _, failed = run(bench('a', 10.0, mem_mb=None),
                             bench('a', 10.0, mem_mb=500.0))
    assert not failed
    assert rows[0]['m_ratio'] is None
    assert rows[0]['base_m'] is None


def test_benchmark_missing_extra_info_entirely():
    """Not just an absent key -- an absent ``extra_info`` dict."""
    baseline = bench('a', 10.0)
    del next(iter(baseline.values()))['extra_info']
    rows, _, _, failed = run(baseline, bench('a', 10.0))
    assert not failed
    assert rows[0]['m_ratio'] is None


def test_added_and_removed_are_reported_but_do_not_fail():
    """Adding or removing a scenario is a legitimate thing for a PR to do."""
    baseline = {**bench('shared', 10.0), **bench('gone', 10.0)}
    contender = {**bench('shared', 10.0), **bench('new', 10.0)}
    rows, added, removed, failed = run(baseline, contender)
    assert not failed
    assert [r['name'] for r in rows] == ['shared']
    assert added == ['new']
    assert removed == ['gone']


def test_no_common_benchmarks_fails():
    """A comparison that checked nothing must not report success."""
    rows, added, removed, failed = run(bench('a', 10.0), bench('b', 10.0))
    assert rows == []
    assert failed, 'zero overlap has to be red, or the gate can silently ' \
                   'switch itself off'
    assert added == ['b'] and removed == ['a']


def test_both_sides_empty_fails():
    """Two runs that collected nothing at all is the same failure."""
    _, _, _, failed = run({}, {})
    assert failed


def test_improvement_does_not_fail():
    rows, _, _, failed = run(bench('a', 100.0, mem_mb=100.0),
                             bench('a', 10.0, mem_mb=10.0))
    assert not failed
    assert rows[0]['t_ratio'] == pytest.approx(0.1)


def test_zero_baseline_is_skipped_rather_than_infinite():
    """A zero baseline would otherwise be an infinite ratio."""
    assert compare_one(0.0, 1.0, 2.0, 0.0) == (None, 'ok')


def test_failures_sort_above_larger_under_floor_ratios():
    """The row the reader is here for goes first.

    ``tiny`` moves by the larger ratio (5x against 3x) but only by 0.4 ms, so
    it is reported and not failed; ``real`` is the actual regression and has to
    sort above it.
    """
    baseline = {**bench('real', 10.0), **bench('tiny', 0.1)}
    contender = {**bench('real', 30.0), **bench('tiny', 0.5)}
    rows, _, _, failed = run(baseline, contender)
    assert failed
    assert rows[0]['t_status'] == 'regressed'
    assert rows[1]['t_status'] == 'under-floor'
    assert rows[0]['name'] == 'real', 'a 5x ratio on a 0.1 ms benchmark ' \
                                      'must not outrank a real regression'


def write(tmp_path, name, entries):
    """Write ``entries`` out as a pytest-benchmark JSON file."""
    path = tmp_path / name
    path.write_text(json.dumps({'benchmarks': list(entries.values())}))
    return str(path)


def test_main_exit_codes(tmp_path):
    """End to end, including the summary file the workflow feeds to Actions."""
    base = write(tmp_path, 'base.json', bench('a', 10.0))
    good = write(tmp_path, 'good.json', bench('a', 12.0))
    bad = write(tmp_path, 'bad.json', bench('a', 30.0))
    summary = tmp_path / 'summary.md'

    assert main([base, good]) == 0
    assert main([base, bad]) == 1
    assert main([base, bad, '--no-fail']) == 0

    assert main([base, bad, '--summary', str(summary)]) == 1
    assert 'regressed past its threshold' in summary.read_text()


def test_main_respects_custom_thresholds(tmp_path):
    base = write(tmp_path, 'base.json', bench('a', 10.0))
    contender = write(tmp_path, 'head.json', bench('a', 30.0))
    assert main([base, contender]) == 1
    assert main([base, contender, '--time-threshold', '5']) == 0


def test_main_reports_empty_comparison_without_claiming_a_regression(
        tmp_path, capsys):
    base = write(tmp_path, 'base.json', bench('a', 10.0))
    contender = write(tmp_path, 'head.json', bench('b', 10.0))
    assert main([base, contender]) == 1
    out = capsys.readouterr().out
    assert 'no benchmarks in common' in out
    assert 'regressed past its threshold' not in out
