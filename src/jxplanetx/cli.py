"""Command-line entry point for reproducible JX validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .claims import assess_claim
from .gates import run_core_gates
from .provenance import runtime_source_manifest, write_run_record
from .production_benchmark import reproduce
from .de441_anchor import audit_de441_independent_reference, import_de441_anchor, run_de441_precision_pair
from .decimal_bs import run_block_reference, run_reference
from .ensemble_validation import (
    EnsembleVerdict,
    finalize_ensemble_validation,
    prepare_ensemble_plan,
    register_ensemble_member,
    write_example_contract,
)
from .ias15_gate import compare_ias15_members, compare_ias15_population, compare_source_control_effect, finalize_ias15_equation_gate, run_ias15_member
from .population_scale import run_population_scale_gate
from .encounter_tail import run_encounter_tail_pilot
from .survey_selection_v2 import (
    SurveySelectionVerdict,
    finalize_survey_selection as finalize_survey_selection_v2,
    load_survey_contract,
    parse_ossos_tracked_file,
    run_analytic_survey_pilot,
    write_detection_csv,
    write_ossos_model_file,
)
from .survey_selection_v3 import finalize_survey_selection as finalize_survey_selection_v3


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
        "software": runtime_source_manifest(),
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
        "software": runtime_source_manifest(),
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
        "software": runtime_source_manifest(),
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
        "software": runtime_source_manifest(),
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
        "software": runtime_source_manifest(),
    }
    record = write_run_record(args.output, payload)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if result["all_required_gates_passed"] else 2


def write_ensemble_contract_cli(args: argparse.Namespace) -> int:
    result = write_example_contract(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def prepare_ensemble_cli(args: argparse.Namespace) -> int:
    result = prepare_ensemble_plan(args.contract, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def register_ensemble_member_cli(args: argparse.Namespace) -> int:
    result = register_ensemble_member(
        args.plan,
        args.member_id,
        args.arm,
        args.method_id,
        args.trajectory,
        args.validity,
        args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["payload"]["validity_passed"] else 2


def finalize_ensemble_cli(args: argparse.Namespace) -> int:
    result = finalize_ensemble_validation(args.plan, args.run_root)
    payload = {
        "engine_version": __version__,
        "command": "finalize-ensemble-validation",
        "result": result,
        "software": runtime_source_manifest(),
    }
    record = write_run_record(args.output, payload)
    print(json.dumps(record, indent=2, sort_keys=True))
    if result["verdict"] == EnsembleVerdict.PASSED.value:
        return 0
    return 2 if result["verdict"] == EnsembleVerdict.INVALID.value else 3


def run_population_scale_cli(args: argparse.Namespace) -> int:
    result = run_population_scale_gate(args.contract, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] == "SCALE_GATE_PASSED" else 3


def run_encounter_tail_cli(args: argparse.Namespace) -> int:
    result = run_encounter_tail_pilot(args.contract, args.run_dir, args.output, args.workers)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] == "ENCOUNTER_TAIL_PILOT_PASSED" else 3


def write_survey_population_cli(args: argparse.Namespace) -> int:
    contract = load_survey_contract(args.contract)
    result = write_ossos_model_file(
        args.output,
        contract,
        args.model_id,
        args.seed_block,
        args.count,
        namespace=args.namespace,
        start_index=args.start_index,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def normalize_ossos_tracked_cli(args: argparse.Namespace) -> int:
    rows = parse_ossos_tracked_file(args.input, args.model_id, args.seed_block)
    result = write_detection_csv(args.output, rows)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def finalize_survey_selection_cli(args: argparse.Namespace) -> int:
    contract = load_survey_contract(args.contract)
    finalizer = (
        finalize_survey_selection_v3
        if contract["experiment_id"].endswith("v3-exact-zeta-corrective-replay")
        else finalize_survey_selection_v2
    )
    result = finalizer(
        args.contract,
        args.correct_manifest,
        args.wrong_manifest,
        args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["verdict"] == SurveySelectionVerdict.PASSED.value:
        return 0
    return 2 if result["verdict"] == SurveySelectionVerdict.INVALID.value else 3


def run_survey_pilot_cli(args: argparse.Namespace) -> int:
    result = run_analytic_survey_pilot(
        args.contract,
        args.run_dir,
        args.output,
        draws_per_block=args.draws_per_block,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["verdict"] == SurveySelectionVerdict.INVALID.value:
        return 2
    return 3


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
    wt = sub.add_parser("write-ensemble-contract", help="write a preregistration contract template")
    wt.add_argument("--output", required=True)
    wt.set_defaults(func=write_ensemble_contract_cli)
    pe = sub.add_parser("prepare-ensemble", help="lock a contract and generate deterministic phase/uncertainty draws")
    pe.add_argument("--contract", required=True)
    pe.add_argument("--output", required=True, help="immutable plan-lock JSON")
    pe.set_defaults(func=prepare_ensemble_cli)
    re = sub.add_parser("register-ensemble-member", help="strictly validate and register one planned member trajectory")
    re.add_argument("--plan", required=True)
    re.add_argument("--member-id", required=True)
    re.add_argument("--arm", choices=("control", "source"), required=True)
    re.add_argument("--method-id", required=True)
    re.add_argument("--trajectory", required=True)
    re.add_argument("--validity", required=True, help="jx-integrator-validity/v1 JSON")
    re.add_argument("--output", required=True)
    re.set_defaults(func=register_ensemble_member_cli)
    fe = sub.add_parser("finalize-ensemble-validation", help="finalize a complete locked chaotic-population ensemble")
    fe.add_argument("--plan", required=True)
    fe.add_argument("--run-root", required=True)
    fe.add_argument("--output", required=True)
    fe.set_defaults(func=finalize_ensemble_cli)
    ps = sub.add_parser(
        "run-population-scale-gate",
        help="run a locked paired massless-tracer scale gate without string-hash identities",
    )
    ps.add_argument("--contract", required=True)
    ps.add_argument("--output", required=True)
    ps.set_defaults(func=run_population_scale_cli)
    et = sub.add_parser(
        "run-encounter-tail-pilot",
        help="run the checkpointed controlled-synthetic 10-kyr encounter-tail pilot and dt/2 audit",
    )
    et.add_argument("--contract", required=True)
    et.add_argument("--run-dir", required=True)
    et.add_argument("--output", required=True)
    et.add_argument("--workers", type=int)
    et.set_defaults(func=run_encounter_tail_cli)
    so = sub.add_parser(
        "write-survey-population",
        help="write one deterministic JX-O1 intrinsic-population block for OSSOS",
    )
    so.add_argument("--contract", required=True)
    so.add_argument("--model-id", choices=("correct", "wrong"), required=True)
    so.add_argument("--seed-block", type=int, required=True)
    so.add_argument("--count", type=int, required=True)
    so.add_argument("--namespace", default="final")
    so.add_argument("--start-index", type=int, default=0)
    so.add_argument("--output", required=True)
    so.set_defaults(func=write_survey_population_cli)
    no = sub.add_parser(
        "normalize-ossos-tracked",
        help="strictly normalize one official OSSOS tracked-output file",
    )
    no.add_argument("--input", required=True)
    no.add_argument("--model-id", choices=("correct", "wrong"), required=True)
    no.add_argument("--seed-block", type=int, required=True)
    no.add_argument("--output", required=True)
    no.set_defaults(func=normalize_ossos_tracked_cli)
    fs = sub.add_parser(
        "finalize-survey-selection",
        help="evaluate the frozen JX-O1 calibration, power, provenance, and replay gates",
    )
    fs.add_argument("--contract", required=True)
    fs.add_argument("--correct-manifest", required=True)
    fs.add_argument("--wrong-manifest", required=True)
    fs.add_argument("--output", required=True)
    fs.set_defaults(func=finalize_survey_selection_cli)
    sp = sub.add_parser(
        "run-survey-selection-pilot",
        help="run the non-final analytic JX-O1 software pilot",
    )
    sp.add_argument("--contract", required=True)
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--draws-per-block", type=int, default=10_000)
    sp.set_defaults(func=run_survey_pilot_cli)
    return p


def main() -> int:
    args = parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
