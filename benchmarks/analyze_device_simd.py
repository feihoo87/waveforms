"""Aggregate repeated device benchmarks and check cross-build sample errors.

Run in the isolated benchmark directory containing PREFIX-final-*.json and
PREFIX-samples/*.npy, produced with benchmark_common_pipeline.py --samples-dir.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--before', default='baseline')
    parser.add_argument('--after', default='candidate3')
    args = parser.parse_args()
    directory = args.directory
    report = {'platform': platform.platform(), 'python': sys.version,
              'numpy': np.__version__, 'builds': {}, 'comparison': {}}
    for label, prefix in (('before', args.before), ('after', args.after)):
        build = {'source_sha256': hashlib.sha256(
            (directory / prefix / 'waveforms' / '_cwaveform.c').read_bytes()).hexdigest(),
            'runs': {}, 'median_ms': {}}
        for suite in ('pipeline', 'simd', 'nonlinear'):
            runs = [json.loads(path.read_text()) for path in sorted(
                directory.glob(f'{prefix}-final-{suite}-*.json'))]
            assert len(runs) == 3, (prefix, suite, len(runs))
            build['runs'][suite] = runs
            if suite == 'pipeline':
                build['median_ms'][suite] = {
                    case: {metric: statistics.median(
                        run['cases'][case]['ms'][metric] for run in runs)
                           for metric in runs[0]['cases'][case]['ms']}
                    for case in runs[0]['cases']}
            else:
                build['median_ms'][suite] = {
                    metric: statistics.median(run['milliseconds'][metric] for run in runs)
                    for metric in runs[0]['milliseconds']}
        report['builds'][label] = build
    before, after = report['builds']['before'], report['builds']['after']
    for case in before['median_ms']['pipeline']:
        a = before['runs']['pipeline'][0]['cases'][case]
        b = after['runs']['pipeline'][0]['cases'][case]
        entry = {'samples': a['samples'], 'pickle_bytes': a['pickle_bytes'],
                 'core_bytes': a['core_bytes'], 'samples_comparison': {},
                 'speedup': {key: value / after['median_ms']['pipeline'][case][key]
                             for key, value in before['median_ms']['pipeline'][case].items()}}
        assert a['core_sha256'] == b['core_sha256']
        assert a['pickle_sha256'] == b['pickle_sha256']
        assert a['pickle_bytes'] == b['pickle_bytes']
        assert a['core_bytes'] == b['core_bytes']
        for kind in a['sha256']:
            left = np.load(directory / f'{args.before}-samples' / f'{case}-{kind}.npy')
            right = np.load(directory / f'{args.after}-samples' / f'{case}-{kind}.npy')
            assert left.dtype == right.dtype and left.shape == right.shape
            equal = np.array_equal(left, right)
            error = float(np.max(np.abs(left.astype(float) - right.astype(float))))
            if kind != 'float64':
                assert equal, (case, kind, error)
            else:
                np.testing.assert_allclose(left, right, rtol=2e-14, atol=2e-14)
            entry['samples_comparison'][kind] = {'equal': equal, 'max_absolute_error': error}
        report['comparison'][case] = entry
    for name in dict.fromkeys((args.before, 'candidate1', 'candidate2', args.after)):
        path = directory / f'kernels-{name}.json'
        if path.exists():
            report.setdefault('kernel_probes', {})[name] = json.loads(path.read_text())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
