"""Compare two benchmark runs and fail on a regression.

Reads two JSON files written by ``pytest --benchmark-json`` and reports, for
every benchmark they have in common, how the contender's run time and peak
memory moved relative to the baseline. Prints a Markdown table and exits
non-zero if anything regressed past its threshold.

    python benchmarks/compare.py baseline.json contender.json

Time and memory are held to deliberately different standards, because they are
not equally trustworthy measurements.

``tracemalloc`` counts allocations rather than sampling the process, so
repeated runs of unchanged code report the same peak to the byte. 
It does not see raw ``malloc`` inside the Cython/OpenMP kernels,
so the number is a floor on total memory rather than the whole of it.

Processing time may vary depending on runner load, so we set a genuine 20%
regression threshold, with the goal of catching the disasters instead of
every tiny change.

Both checks also require an absolute change, not just a ratio. Several
benchmarks are small enough (a 0.17 ms build, a 0.08 MB prediction) that a
large ratio there is noise on a number too small to matter. Ratio-only
breaches are still shown, marked ``(under floor)``, so nothing is hidden.
"""
import argparse
import json
import sys

# Well outside the 15-30% run-to-run drift measured for `min` on one machine;
# see the module docstring for why this is loose and memory is not.
TIME_THRESHOLD = 2.0
MEM_THRESHOLD = 1.15

# Below these, a ratio is a large change to a number nobody notices.
TIME_FLOOR_MS = 1.0
MEM_FLOOR_MB = 1.0


def load(path):
    """Return ``{fullname: benchmark}`` from a pytest-benchmark JSON file."""
    with open(path) as f:
        return {b['fullname']: b for b in json.load(f)['benchmarks']}


def compare_one(base, head, threshold, floor):
    """Compare a single metric.

    Returns ``(ratio, status)``, where status is ``'ok'``, ``'under-floor'``
    (over the threshold but too small in absolute terms to count) or
    ``'regressed'``. Either value being missing or zero yields
    ``(None, 'ok')``: a benchmark that reports no memory, or a baseline of
    exactly zero, is skipped rather than treated as an infinite regression.
    """
    if base is None or head is None or base <= 0:
        return None, 'ok'
    ratio = head / base
    if ratio <= threshold:
        return ratio, 'ok'
    return ratio, 'regressed' if head - base > floor else 'under-floor'


def fmt_delta(ratio, status):
    """Render a ratio as a signed percentage, marked if it broke a limit."""
    if ratio is None:
        return '--'
    cell = f'{ratio - 1:+.0%}'
    if status == 'regressed':
        return f'**{cell}** :warning:'
    if status == 'under-floor':
        return f'{cell} (under floor)'
    return cell


def compare(baseline, contender, args):
    """Build the report rows and decide whether the run failed.

    Returns ``(rows, added, removed, failed)``. Benchmarks present on only one
    side are reported but never fail the run: adding or removing a scenario is
    a legitimate thing for a pull request to do.

    Having *nothing* in common does fail, though. An empty intersection means
    the comparison checked no code at all, which a renamed benchmark, a changed
    parametrization or a collection error can all cause silently.
    """
    rows, failed = [], False
    for name in baseline.keys() & contender.keys():
        base, head = baseline[name], contender[name]
        # The baseline comes from an older commit, which may predate a
        # benchmark recording memory at all; such a row is reported on time
        # alone rather than crashing the comparison.
        base_m = base.get('extra_info', {}).get('peak_mem_mb')
        head_m = head.get('extra_info', {}).get('peak_mem_mb')
        t_ratio, t_status = compare_one(base['stats']['min'],
                                        head['stats']['min'],
                                        args.time_threshold,
                                        args.time_floor_ms / 1e3)
        m_ratio, m_status = compare_one(base_m, head_m, args.mem_threshold,
                                        args.mem_floor_mb)
        failed |= 'regressed' in (t_status, m_status)
        rows.append({
            'name': head['name'],
            'base_t': base['stats']['min'] * 1e3,
            'head_t': head['stats']['min'] * 1e3,
            't_ratio': t_ratio, 't_status': t_status,
            'base_m': base_m, 'head_m': head_m,
            'm_ratio': m_ratio, 'm_status': m_status,
        })
    # Anything that actually failed goes first, then everything else by how
    # much it moved.
    rows.sort(key=lambda r: ('regressed' in (r['t_status'], r['m_status']),
                             max(r['t_ratio'] or 0, r['m_ratio'] or 0)),
              reverse=True)
    added = sorted(contender[n]['name'] for n in contender.keys() - baseline)
    removed = sorted(baseline[n]['name'] for n in baseline.keys() - contender)
    return rows, added, removed, failed or not rows


def render(rows, added, removed, failed, args):
    """Render the whole report as Markdown."""
    def mb(value):
        return '--' if value is None else f'{value:.3f} MB'

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
            '| Mem base | Mem head | &Delta; mem |',
            '|---|--:|--:|--:|--:|--:|--:|',
        ]
        for r in rows:
            out.append(
                f"| `{r['name']}` "
                f"| {r['base_t']:.3f} ms | {r['head_t']:.3f} ms "
                f"| {fmt_delta(r['t_ratio'], r['t_status'])} "
                f"| {mb(r['base_m'])} | {mb(r['head_m'])} "
                f"| {fmt_delta(r['m_ratio'], r['m_status'])} |"
            )
        out.append('')

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
        pass  # already explained above; do not also claim a regression
    elif failed:
        out += [
            ':warning: **A benchmark regressed past its threshold.**',
            '',
            'The allocations tracemalloc sees are highly repeatable, so a '
            'memory regression is worth explaining rather than re-running. '
            'Time on a shared runner is not repeatable: confirm a time '
            'regression with `make bench` on a quiet machine before treating '
            'it as one.',
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
