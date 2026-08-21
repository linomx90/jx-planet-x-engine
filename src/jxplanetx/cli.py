"""Command-line entry point for reproducible JX validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .claims import assess_claim
from .gates import run_core_gates
from .provenance import source_manifest, write_run_record
from .production_benchmark import reproduce
from .de441_anchor import audit_de441_independent_reference, import_de441_anchor, run_de441_precision_pair
from .decimal_bs import run_block_reference, run_reference
from .ias15_gate import compare_ias15_members, compare_ias15_population, compare_source_control_effect, finalize_ias15_equation_gate, run_ias15_member


def validate(args: argparse.Namespace) -> int:
    gates = run_core_gates(args.decimal_digits)
    claim = assess_claim(gates, observational=False)
    payload = {
        "engine_version": __version__,
        "command": "validate",
        "configuration": {
            "decimal_digits": args.decimal_digits,
            "arithmetic": "decimal.Decimal; no binary-float conversion",
            "benchmarks": {
                "oscillator": {"duration": "1", "steps_h": "0.05", "steps_h_over_2": "0.025"},
                "two_body": {
                    "frame": "idealized inertial Cartesian",
                    "origin": "fixed central mass",
                    "epoch": "dimensionless t=0",
                    "units": "canonical GM=1, radius=1",
                    "dt": "0.01",
                    "steps": 1000,
                },
            },
        },
        "gates": [g.as_dict() for g in gates],
        "claim_control": asdict(claim),
        "software": source_manifest(Path(__file__).resolve().parents[2]),
    }
    record = write_run_record(args.output, payload)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if all(g.passed for g in gates) else 2


def reproduce_yoshida6(args: argparse.Namespace) -> int:
    result = reproduce(args.bundle_dir, args.run_dir)
    payload = {
        "engine_version": __version__,
        "command": "reproduce-yoshida6",
        "result": result,
        "software": source_manifest(Path(__file__).resolve().parents[2]),
    }
    record = write_run_record(args.output, payload)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if result["all_gates_passed"] else 2


def import_de441(args: argparse.Namespace) -> int:
    result = import_de441_anchor(args.source_csv, args.output_csv, args.metadata)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def run_de441(args: argparse.Namespace) -> int:
    result = run_de441_precision_pair(args.binary, args.state_csv, args.run_dir)
    payload = {
        "engine_version": __version__,
        "command": "run-de441-anchor-gate",
        "result": result,
        "software": source_manifest(Path(__file__).resolve().parents[2]),
    }
    record = write_run_record(args.output, payload)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if result["precision_pair_passed"] else 2


def run_bs(args: argparse.Namespace) -> int:
    summary = run_reference(args.state_csv, args.trajectory, args.summary, args.years, args.decimal_digits)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def run_bs_blocks(args: argparse.Namespace) -> int:
    summary = run_block_reference(
        args.state_csv, args.run_dir, args.trajectory, args.summary,
        args.years, args.decimal_digits, args.workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_block_invariants_passed"] else 2


def audit_de441(args: argparse.Namespace) -> int:
    result = audit_de441_independent_reference(
        args.pair_record, args.y6_trajectory, args.bs_trajectory, args.bs_summary
    )
    payload = {
        "engine_version": __version__,
        "command": "audit-de441-independent-reference",
        "result": result,
        "software": source_manifest(Path(__file__).resolve().parents[2]),
    }
    record = write_run_record(args.output, payload)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if result["all_numerical_gates_passed"] else 2


def run_ias15(args: argparse.Namespace) -> int:
    result = run_ias15_member(
        args.state_csv, args.gm_file, args.trajectory, args.summary,
        args.family, args.inclination, args.phase, args.epsilon,
        args.years, args.output_interval,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def compare_ias15(args: argparse.Namespace) -> int:
    result = compare_ias15_members(args.loose, args.tight, args.label)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


def compare_ias15_population_cli(args: argparse.Namespace) -> int:
    result = compare_ias15_population(args.first, args.second, args.label, args.classification)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["population_gate_passed"] else 2


def compare_source_effect_cli(args: argparse.Namespace) -> int:
    result = compare_source_control_effect(
        args.control_first, args.source_first, args.control_second, args.source_second,
        args.label, args.classification,
    )
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["source_effect_gate_passed"] else 2


def finalize_ias15_cli(args: argparse.Namespace) -> int:
    result = finalize_ias15_equation_gate(args.run_root)
    payload = {
        "engine_version": __version__,
        "command": "finalize-ias15-equation-gate",
        "result": result,
        "software": source_manifest(Path(__file__).resolve().parents[2]),
    }
    record = write_run_record(args.output, payload)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if result["all_required_gates_passed"] else 2


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jxplanetx", description="JX Planet X scientific engine")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(required=True)
    v = sub.add_parser("validate", help="run core numerical validity gates")
    v.add_argument("--decimal-digits", type=int, default=80)
    v.add_argument("--output", default="runs/validation.json")
    v.set_defaults(func=validate)
    r = sub.add_parser("reproduce-yoshida6", help="reproduce the locked 160/224-bit synthetic-state benchmark")
    r.add_argument("--bundle-dir", required=True, help="verified extracted JX benchmark bundle")
    r.add_argument("--run-dir", default="runs/yoshida6_reproduction")
    r.add_argument("--output", default="runs/yoshida6_reproduction.json")
    r.set_defaults(func=reproduce_yoshida6)
    d = sub.add_parser("import-de441-anchor", help="import the locked DE441 giant-planet state and synthetic tracer subset")
    d.add_argument("--source-csv", required=True)
    d.add_argument("--output-csv", default="runs/de441_anchor/initial_state.csv")
    d.add_argument("--metadata", default="runs/de441_anchor/metadata.json")
    d.set_defaults(func=import_de441)
    g = sub.add_parser("run-de441-anchor-gate", help="run the locked 160/224-bit Yoshida pair on the DE441 anchor")
    g.add_argument("--binary", required=True)
    g.add_argument("--state-csv", default="runs/de441_anchor/initial_state.csv")
    g.add_argument("--run-dir", default="runs/de441_anchor/gate")
    g.add_argument("--output", default="runs/de441_anchor/gate.json")
    g.set_defaults(func=run_de441)
    b = sub.add_parser("run-bs-reference", help="run independent Decimal Bulirsch-Stoer reference")
    b.add_argument("--state-csv", required=True)
    b.add_argument("--trajectory", required=True)
    b.add_argument("--summary", required=True)
    b.add_argument("--years", type=int, default=100)
    b.add_argument("--decimal-digits", type=int, default=78)
    b.set_defaults(func=run_bs)
    k = sub.add_parser("run-bs-block-reference", help="run parallel independent BS massless-tracer blocks")
    k.add_argument("--state-csv", required=True)
    k.add_argument("--run-dir", required=True)
    k.add_argument("--trajectory", required=True)
    k.add_argument("--summary", required=True)
    k.add_argument("--years", type=int, default=100)
    k.add_argument("--decimal-digits", type=int, default=78)
    k.add_argument("--workers", type=int, default=4)
    k.set_defaults(func=run_bs_blocks)
    a = sub.add_parser("audit-de441-independent-reference", help="close the DE441 numerical gate against independent BS")
    a.add_argument("--pair-record", default="runs/de441_anchor/gate.json")
    a.add_argument("--y6-trajectory", default="runs/de441_anchor/gate/de441_anchor_y6_224.csv")
    a.add_argument("--bs-trajectory", default="runs/de441_anchor/bs_reference_224_blocked_corrected.csv")
    a.add_argument("--bs-summary", default="runs/de441_anchor/bs_reference_224_blocked_corrected_summary.json")
    a.add_argument("--output", default="runs/de441_anchor/independent_reference_gate.json")
    a.set_defaults(func=audit_de441)
    i = sub.add_parser("run-ias15-member", help="run one preserved-state adaptive IAS15 member")
    i.add_argument("--state-csv", required=True)
    i.add_argument("--gm-file", required=True)
    i.add_argument("--family", required=True)
    i.add_argument("--inclination", type=float, default=0.0)
    i.add_argument("--phase", type=int, default=0)
    i.add_argument("--epsilon", type=float, required=True)
    i.add_argument("--years", type=int, default=100000)
    i.add_argument("--output-interval", type=int, default=1000)
    i.add_argument("--trajectory", required=True)
    i.add_argument("--summary", required=True)
    i.set_defaults(func=run_ias15)
    c = sub.add_parser("compare-ias15-members", help="compare loose and tight IAS15 trajectories")
    c.add_argument("--loose", required=True)
    c.add_argument("--tight", required=True)
    c.add_argument("--label", required=True)
    c.add_argument("--output", required=True)
    c.set_defaults(func=compare_ias15)
    u = sub.add_parser("compare-ias15-population", help="compare chaotic-run population observables")
    u.add_argument("--first", required=True)
    u.add_argument("--second", required=True)
    u.add_argument("--label", required=True)
    u.add_argument("--classification", required=True)
    u.add_argument("--output", required=True)
    u.set_defaults(func=compare_ias15_population_cli)
    e = sub.add_parser("compare-source-control-effect", help="test source-minus-control population-effect reproducibility")
    e.add_argument("--control-first", required=True)
    e.add_argument("--source-first", required=True)
    e.add_argument("--control-second", required=True)
    e.add_argument("--source-second", required=True)
    e.add_argument("--label", required=True)
    e.add_argument("--classification", required=True)
    e.add_argument("--output", required=True)
    e.set_defaults(func=compare_source_effect_cli)
    f = sub.add_parser("finalize-ias15-equation-gate", help="write the governing IAS15 equation-gate record")
    f.add_argument("--run-root", default="runs/ias15_equation_gate")
    f.add_argument("--output", default="runs/ias15_equation_gate/governing_result.json")
    f.set_defaults(func=finalize_ias15_cli)
    return p


def main() -> int:
    args = parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
