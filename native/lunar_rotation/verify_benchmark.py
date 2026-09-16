#!/usr/bin/env python3
"""Compile/run the opt-in native port against the retained Python propagation archive.

No network access or package installation. Requires NumPy, SpiceyPy, CMake and a
C++20 compiler. --parent points to extracted jx_lunar_propagation_2026-09-16.
All outputs go to a NEW --out directory; parent inputs are read-only.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
import numpy as np
import spiceypy


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distance(q, p) -> float:
    q, p = np.asarray(q, dtype=np.longdouble), np.asarray(p, dtype=np.longdouble)
    q, p = q / np.linalg.norm(q), p / np.linalg.norm(p)
    s = q[0] * p[0] + np.dot(q[1:], p[1:])
    v = q[0] * p[1:] - p[0] * q[1:] - np.cross(q[1:], p[1:])
    return float(2 * np.arctan2(np.linalg.norm(v), abs(s)))


def rotate(q, w):
    q = np.asarray(q, dtype=float)
    q /= np.linalg.norm(q)
    s, v = q[0], q[1:]
    return w + 2 * s * np.cross(v, w) + 2 * np.cross(v, np.cross(v, w))


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parent', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    parent, out = args.parent.resolve(), args.out.resolve()
    here = Path(__file__).resolve().parent
    if out.exists():
        raise ValueError('Output directory already exists; choose a fresh path.')
    if out.is_relative_to(parent) or parent.is_relative_to(out):
        raise ValueError('Keep test outputs separate from parent inputs.')
    expected = json.loads((here / 'kernel_manifest.json').read_text())
    found = list(parent.rglob('de440s.bsp'))
    if len(found) != 1:
        raise ValueError('Expected exactly one retained DE440s kernel.')
    kernels = found[0].parent
    for name, sha in expected.items():
        if digest(kernels / name) != sha:
            raise ValueError('Kernel hash mismatch: ' + name)
    before = {}
    for line in (parent / 'SHA256SUMS.txt').read_text().splitlines():
        wanted, name = line.split(maxsplit=1)
        name = name.lstrip('*')
        file = (parent / name).resolve()
        if not file.is_relative_to(parent) or digest(file) != wanted:
            raise ValueError('Invalid parent manifest entry: ' + name)
        before[name] = wanted
    lib = next(Path(spiceypy.__file__).resolve().parent.rglob('libcspice.so'))
    out.mkdir(parents=True)
    log = []
    def run(command):
        result = subprocess.run([str(x) for x in command], capture_output=True, text=True)
        log.append({'command': [str(x) for x in command], 'exit': result.returncode,
                    'stdout': result.stdout, 'stderr': result.stderr})
        (out / 'commands.json').write_text(json.dumps(log, indent=2) + '\n')
        result.check_returncode()
    run(['cmake', '-S', here, '-B', out / 'build', '-DCMAKE_BUILD_TYPE=Release',
         '-DCSPICE_LIBRARY=' + str(lib)])
    run(['cmake', '--build', out / 'build', '--parallel', '2'])
    run(['ctest', '--test-dir', out / 'build', '--output-on-failure'])
    executable = out / 'build/jx_lunar_native'
    for steps in (32, 64, 128):
        run([executable, kernels, out / ('h' + str(steps)), steps])
    run([executable, kernels, out / 'replay', 128])
    checks, values = [], []
    for arc in (0, 32, 64, 96):
        ref = rows(parent / f'results/arc{arc:03d}_dop_fine_trace.csv')
        lookup = {(float(r['elapsed_day']), int(r['lane'] == 'candidate')): r for r in ref}
        for steps in (32, 64, 128):
            max_angle, max_rate = 0.0, 0.0
            native = rows(out / f'h{steps}/arc{arc}.csv')
            if len(native) != 258 or len(lookup) != 258:
                raise ValueError('Unexpected trajectory row count.')
            for r in native:
                p = lookup[(float(r['elapsed_day']), int(r['lane']))]
                q = np.array([float(r[f'q{i}']) for i in range(4)])
                w = np.array([float(r[f'w{i}_day']) for i in range(3)]) / 86400
                pq = [float(p[f'q_{i}']) for i in range(4)]
                pw = np.array([float(p[f'omega_inertial_rad_s_{i}']) for i in range(3)])
                max_angle = max(max_angle, distance(q, pq))
                max_rate = max(max_rate, float(np.linalg.norm(rotate(q, w) - pw)))
            values.append({'arc': arc, 'steps_per_day': steps,
                           'max_angle_rad': max_angle, 'max_rate_rad_s': max_rate})
            if steps == 128:
                checks.extend([max_angle < 2e-10, max_rate < 2e-17])
        checks.append((out / f'h128/arc{arc}.csv').read_bytes() ==
                      (out / f'replay/arc{arc}.csv').read_bytes())
    for folder in ['h32', 'h64', 'h128', 'replay']:
        e = json.loads((out / folder / 'execution.json').read_text())
        checks.append(e['reference_queries'] == 4 and e['reference_queries_during_integration'] == 0)
    for name, sha in before.items():
        if digest(parent / name) != sha:
            raise ValueError('Parent input changed: ' + name)
    result = {'decision': 'NATIVE_REPRODUCTION_PASSED' if all(checks) else 'NOT_PASSED',
              'comparisons': values, 'checks_passed': sum(checks), 'checks_total': len(checks),
              'parent_files_unchanged': len(before), 'cspice_library_sha256': digest(lib),
              'scope': 'Native versus retained Python trajectories, not new physical validation or a production merge.'}
    (out / 'REPRODUCTION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if all(checks) else 1


if __name__ == '__main__':
    raise SystemExit(main())
