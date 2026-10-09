"""Compare two benchmark runs and fail on a regression.

Reads two JSON files written by ``pytest --benchmark-json`` and reports the
contender's run time and peak memory relative to the baseline for every shared
benchmark. Prints a Markdown table and exits non-zero if anything regressed
past its threshold.

    python benchmarks/compare.py baseline.json contender.json

``memory_metric`` says what ``peak_mem_mb`` measured: ``rss_delta`` (sampled
process RSS) or ``cuda_allocated_delta`` (Torch's CUDA allocator). Memory is
compared only when both runs report the same metric; untagged entries predate
the tag and are compared on time only.

Run time varies with runner load, so the time threshold is a generous 2x and
catches only major regressions.

Both checks also require an absolute change, because some benchmarks are tiny
(a 0.17 ms build, a 0.08 MB prediction). Ratio-only breaches are shown as
``(under floor)`` and do not fail the run.
"""
import argparse
import json
import sys

# Well outside the 15-30% run-to-run drift measured for `min` on one machine:
TIME_THRESHOLD = 2.0
MEM_THRESHOLD = 1.15

# Absolute changes below these never fail the run:
TIME_FLOOR_MS = 1.0
MEM_FLOOR_MB = 1.0


def load(path):
    """Return ``{fullname: benchmark}`` from a pytest-benchmark JSON file."""
    with open(path) as f:
        return {b['fullname']: b for b in json.load(f)['benchmarks']}


def memory(bench):
    """Return ``(peak_mem_mb, memory_metric)``; either may be None."""
    info = bench.get('extra_info', {})
    return info.get('peak_mem_mb'), info.get('memory_metric')


def compare_one(base, head, threshold, floor):
    """Compare a single metric.

    Returns ``(ratio, status)``, where status is ``'ok'``, ``'under-floor'``
    (over the threshold but below the absolute floor) or ``'regressed'``.
    A missing value or a baseline <= 0 returns ``(None, 'ok')``.
    """
    if base is None or head is None or base <= 0:
        return None, 'ok'
    ratio = head / base
    if ratio <= threshold:
        return ratio, 'ok'
    return ratio, 'regressed' if head - base > floor else 'under-floor'


def fmt_delta(ratio, status):
    """Return a ratio as a signed percentage, marked if over a limit."""
    if ratio is None:
        return '--'
    cell = f'{ratio - 1:+.0%}'
    if status == 'regressed':
        return f'**{cell}** :warning:'
    if status == 'under-floor':
        return f'{cell} (under floor)'
    return cell


def compare(baseline, contender, args):
    """Return ``(rows, added, removed, failed)``.

    Benchmarks present on only one side are reported but do not fail the run.
    An empty intersection fails (e.g., renamed benchmarks, changed
    parametrization, or a collection error).
    """
    rows, failed = [], False
    for name in baseline.keys() & contender.keys():
        base, head = baseline[name], contender[name]
        base_m, base_metric = memory(base)
        head_m, head_metric = memory(head)
        t_ratio, t_status = compare_one(base['stats']['min'],
                                        head['stats']['min'],
                                        args.time_threshold,
                                        args.time_floor_ms / 1e3)
        if base_metric and base_metric == head_metric:
            m_ratio, m_status = compare_one(base_m, head_m,
                                            args.mem_threshold,
                                            args.mem_floor_mb)
        else:
            m_ratio, m_status = None, 'ok'
        failed |= 'regressed' in (t_status, m_status)
        rows.append({
            'name': head['name'],
            'base_t': base['stats']['min'] * 1e3,
            'head_t': head['stats']['min'] * 1e3,
            't_ratio': t_ratio, 't_status': t_status,
            'base_m': base_m, 'head_m': head_m,
            'base_metric': base_metric, 'head_metric': head_metric,
            'm_ratio': m_ratio, 'm_status': m_status,
        })
    # Regressions first, then by largest ratio:
    rows.sort(key=lambda r: ('regressed' in (r['t_status'], r['m_status']),
                             max(r['t_ratio'] or 0, r['m_ratio'] or 0)),
              reverse=True)
    added = sorted(contender[n]['name'] for n in contender.keys() - baseline)
    removed = sorted(baseline[n]['name'] for n in baseline.keys() - contender)
    return rows, added, removed, failed or not rows


def render(rows, added, removed, failed, args):
    """Return the report as Markdown."""
    def mb(value):
        return '--' if value is None else f'{value:.3f} MB'

    def metric(r):
        base, head = r['base_metric'] or '--', r['head_metric'] or '--'
        return base if base == head else f'{base} &rarr; {head}'

    out = ['## Benchmark comparison', '']
    if not rows:
        out += [
            ':warning: **The two runs have no benchmarks in common, so '
            'nothing was compared.**',
            '',
            'This is reported as a failure rather than a pass. Renamed '
            'benchmarks, a changed parametrization or a collection error can '
            'all empty the comparison, and a gate that checked nothing must '
            'not report success.',
            '',
        ]
    else:
        out += [
            '| Benchmark | Time base | Time head | &Delta; time '
            '| Mem metric | Mem base | Mem head | &Delta; mem |',
            '|---|--:|--:|--:|---|--:|--:|--:|',
        ]
        for r in rows:
            out.append(
                f"| `{r['name']}` "
                f"| {r['base_t']:.3f} ms | {r['head_t']:.3f} ms "
                f"| {fmt_delta(r['t_ratio'], r['t_status'])} "
                f"| {metric(r)} "
                f"| {mb(r['base_m'])} | {mb(r['head_m'])} "
                f"| {fmt_delta(r['m_ratio'], r['m_status'])} |"
            )
        out.append('')
        if any(r['base_metric'] != r['head_metric'] for r in rows):
            out += ['Memory is not compared where the metrics differ.', '']

    for label, names in [('this branch', added), ('the base branch', removed)]:
        if names:
            listed = ', '.join(f'`{n}`' for n in names)
            out += [f'Only in {label} (not compared): {listed}', '']

    out += [
        f'Thresholds: time &times;{args.time_threshold:g} '
        f'(min {args.time_floor_ms:g} ms), '
        f'memory &times;{args.mem_threshold:g} '
        f'(min {args.mem_floor_mb:g} MB).',
        '',
    ]
    if not rows:
        pass  # empty comparison is reported above
    elif failed:
        out += [
            ':warning: **A benchmark regressed past its threshold.**',
            '',
            'Time and sampled RSS on a shared runner are not repeatable: '
            'confirm a time or `rss_delta` regression with `make bench` on a '
            'quiet machine before treating it as one.',
            '',
            'If the regression is real and you intend to accept it, say so in '
            'the pull request and merge over the failure. Do not raise the '
            'thresholds to make it green: once merged, the new cost is the '
            'baseline every later comparison runs against, so the check '
            'protects the next pull request exactly as before.',
        ]
    else:
        out.append('No regression past the thresholds above.')
    return '\n'.join(out) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('baseline', help='pytest-benchmark JSON to compare against')
    p.add_argument('contender', help='pytest-benchmark JSON under scrutiny')
    p.add_argument('--time-threshold', type=float, default=TIME_THRESHOLD,
                   help=f'fail above this time ratio '
                        f'(default: {TIME_THRESHOLD:g})')
    p.add_argument('--mem-threshold', type=float, default=MEM_THRESHOLD,
                   help=f'fail above this memory ratio '
                        f'(default: {MEM_THRESHOLD:g})')
    p.add_argument('--time-floor-ms', type=float, default=TIME_FLOOR_MS,
                   help=f'ignore time regressions smaller than this, in ms '
                        f'(default: {TIME_FLOOR_MS:g})')
    p.add_argument('--mem-floor-mb', type=float, default=MEM_FLOOR_MB,
                   help=f'ignore memory regressions smaller than this, in MB '
                        f'(default: {MEM_FLOOR_MB:g})')
    p.add_argument('--summary', metavar='PATH',
                   help='also append the report here, e.g. '
                        '$GITHUB_STEP_SUMMARY')
    p.add_argument('--no-fail', action='store_true',
                   help='report regressions but always exit 0')
    args = p.parse_args(argv)

    rows, added, removed, failed = compare(load(args.baseline),
                                           load(args.contender), args)
    report = render(rows, added, removed, failed, args)
    sys.stdout.write(report)
    if args.summary:
        with open(args.summary, 'a') as f:
            f.write(report)
    return 1 if failed and not args.no_fail else 0


if __name__ == '__main__':
    sys.exit(main())
