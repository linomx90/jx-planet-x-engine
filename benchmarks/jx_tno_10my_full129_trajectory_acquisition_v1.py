#!/usr/bin/env python3
"""Freeze the next full-129 TNO trajectory-acquisition boundary.

This is intentionally a prepare-only preregistration artifact.  It binds the
passed writer, 1 kyr GPU smoke, 1 kyr restart, lifecycle, bridge, and grid
artifacts to exact files and local annotated Git objects.  It does not expose
an execution or resume command because the existing evidence does not qualify
routed/removed restart or an adapter from real dynamics into the qualified
held-descriptor writer.  Those gaps are recorded as executable blockers rather
than silently promoted into 10 Myr authority.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "benchmarks/jx_tno_10my_full129_trajectory_acquisition_v1.py"
TEST_PATH = PROJECT_ROOT / "tests/test_tno_10my_full129_trajectory_acquisition_v1.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs/tno_10my_full129_trajectory_acquisition_v1"

EXPECTED_BRANCH = "research/tno-10my-full129-trajectory-acquisition-v1-preregistered-20261011"
PREREGISTRATION_TAG = "tno-10my-full129-trajectory-acquisition-v1-preregistered-20261011"
FUTURE_RESULT_TAG = "tno-10my-full129-trajectory-acquisition-v1-result-20261011"

PROTOCOL_SCHEMA = "jx.tno.10my-full129-trajectory-acquisition.protocol.v1"
PROTOCOL_DOMAIN = b"jx.tno.10my-full129-trajectory-acquisition.protocol.v1\0"
FILESYSTEM_TRUST_BOUNDARY = "NO_CONCURRENT_SAME_UID_FILESYSTEM_MUTATOR"
PREPARED_ROSTER = frozenset({"PROTOCOL.json"})


class TrajectoryAcquisitionError(RuntimeError):
    """The prepare-only acquisition boundary failed closed."""


@dataclass(frozen=True)
class FilePin:
    gate: str
    role: str
    relative_path: str
    byte_length: int
    sha256: str
    commit: str


@dataclass(frozen=True)
class GitPin:
    role: str
    tag_name: str
    tag_object: str
    tag_payload_bytes: int
    tag_payload_sha256: str
    peeled_commit: str
    commit_tree: str


WRITER_PREREG_COMMIT = "1df6cec879831be2dae785ee6ca1f86103a3c26c"
WRITER_RESULT_COMMIT = "b4bd47ccee94be51aa8606c73dfc8d447f6a6d65"
SMOKE_PREREG_COMMIT = "cc5394e0b538a3427871678aa05cb7938c65f530"
SMOKE_RESULT_COMMIT = "001d02e4997162dcd6b3ab7e7ea9cb58ad2fe1a8"
RESTART_PREREG_COMMIT = "77ba5a2d95b2384656ac2d4824cf21c6e8162b2b"
RESTART_RESULT_COMMIT = "cc1f4498ae84c1a98ac89aa5a4b05d87dd211e95"
LIFECYCLE_PREREG_COMMIT = "9ec5151e5a979e46a516ec7efdf7c6e4b90da620"
LIFECYCLE_RESULT_COMMIT = "d63e1cc9ddba00ab883776b0bce10fd00051a586"
BRIDGE_GRID_COMMIT = "13c17fab588ec6b073a036d245be4f67c48ec002"

GIT_PINS = (
    GitPin(
        "writer_v3_preregistration",
        "tno-10my-full128-writer-qualification-v3-preregistered-20261011",
        "05d216d7c93a234e378c2aebd450d7e3db2a99b9", 255,
        "7f17bd1b10af4930521e57dd408da39c18e23f90bb0b5ebac561dc6df34aef15",
        WRITER_PREREG_COMMIT, "1d1c262138611e2c3fd9444f5bf9e3254cdff931",
    ),
    GitPin(
        "writer_v3_result",
        "tno-10my-full128-writer-qualification-v3-result-20261011",
        "b8a441cf8e126174e9901ad256d03b816ce0762d", 243,
        "cb662df42e8e80015cb3cc60b71491f117ff228bad2f0a0b0fe4dadb7ef41ae1",
        WRITER_RESULT_COMMIT, "bcbb1b57b6a7f992220489c9504bfc71eb050adf",
    ),
    GitPin(
        "smoke_v3_preregistration",
        "tno-10my-reduced5-execution-v3-preregistered-20261011",
        "bfe5e4b2551ab013e879c64e274db449afa3e135", 246,
        "bf10dbbdf933334e680e7561b7f0ac2c02569c98dfb2651143544a45198b680e",
        SMOKE_PREREG_COMMIT, "53115991c45e02e593881ebde3c8538e06719a74",
    ),
    GitPin(
        "smoke_v3_result",
        "tno-10my-full128-1kyr-gpu-smoke-v3-result-20261011",
        "07065b9acebe9eb3ab34daef71a0ef8b6e4c6c78", 251,
        "6e6808fe01eade6bf271d13e136b654310a09a3dfddfd9aa3c3dc175fa05d685",
        SMOKE_RESULT_COMMIT, "065fc2e967167163471314dda36dbe90d940aeca",
    ),
    GitPin(
        "restart_v1_preregistration",
        "tno-1kyr-full129-restart-v1-preregistered-20261011",
        "13c6ab3510daafb01dbdfbf69fa8d2100a6f3e5c", 269,
        "5470501f230d3f24041e2ca1cb441c06a8fcd50a6da2b700308a89d9b594dbba",
        RESTART_PREREG_COMMIT, "142533c6c0ac26f99db629f7cc4873eb983ea342",
    ),
    GitPin(
        "restart_v1_result",
        "tno-1kyr-full129-restart-v1-result-20261011",
        "600e0555b90cfc71729d464e844c7bd01512f1aa", 244,
        "d1e07f620af1865d19d0f53b5636ee3a9dae4747afe751955cc0aafdadbfa419",
        RESTART_RESULT_COMMIT, "35f0eb2e685d3d78ea8f173635d28fb8d52ac527",
    ),
    GitPin(
        "lifecycle_v3_preregistration",
        "tno-1kyr-reduced5-lifecycle-gate-v3-preregistered-20261011",
        "94297db889f5b5ba2f4d9eaef2adfa792a3740cd", 255,
        "a551458c822d437a4a0453a0ebf7607e4fa6e020fa43c1f44e7d3e0995520306",
        LIFECYCLE_PREREG_COMMIT, "2a4ad00c1ec1c0d3053f0c7063e68e7cebbcd21d",
    ),
    GitPin(
        "lifecycle_v3_result",
        "tno-1kyr-reduced5-lifecycle-gate-v3-result-20261011",
        "1734a1f59168f5cca11a969b02a3eed809ff2675", 243,
        "a2538fcaf84ef27348194d68491ae99461c523864388327b9c3a919240ac6316",
        LIFECYCLE_RESULT_COMMIT, "b854680de3067423a4f35850a4b175a939c0cf91",
    ),
    GitPin(
        "bridge_grid_preregistration",
        "tno-10my-reduced5-execution-v1-preregistered-20261011",
        "9ea2950aadd358e2276e774cf39780b3dcfd2a25", 255,
        "a3f1c29a24c884c464909623c3e085d33daed3361d8bb614b79ff50fcd0740e1",
        BRIDGE_GRID_COMMIT, "f7e296d78e8fa43ae759add2167cfb9bbbeaf6df",
    ),
)

RESULT_LINEAGES = (
    ("writer_v3", WRITER_PREREG_COMMIT, WRITER_RESULT_COMMIT),
    ("smoke_v3", SMOKE_PREREG_COMMIT, SMOKE_RESULT_COMMIT),
    ("restart_v1", RESTART_PREREG_COMMIT, RESTART_RESULT_COMMIT),
    ("lifecycle_v3", LIFECYCLE_PREREG_COMMIT, LIFECYCLE_RESULT_COMMIT),
)

FILE_PINS = (
    # Production-scale synthetic writer qualification and bounded archive.
    FilePin("writer_v3", "source", "benchmarks/jx_tno_10my_full128_writer_qualification_v3.py", 226_961, "0cd73517d271699263322eaf6e884b3705046d434dbd2f0ebe899915b9a144a3", WRITER_PREREG_COMMIT),
    FilePin("writer_v3", "test", "tests/test_tno_10my_full128_writer_qualification_v3.py", 76_407, "3ac266d6bca5cf79d9743e4f900ad5bec76bb3f810ccf0a8f4c1b25f1378bf04", WRITER_PREREG_COMMIT),
    FilePin("writer_v3", "protocol", "runs/tno_10my_full128_writer_qualification_v3/PROTOCOL.json", 10_853, "18f043a98899ee426ffb4003daa034cae6c1619464d42506e83b10dcfe97cdd9", WRITER_PREREG_COMMIT),
    FilePin("writer_v3", "digest_roster", "runs/tno_10my_full128_writer_qualification_v3/DIGEST_ROSTER.jsonl", 23_689_197, "0c07031eeef5488b700486dc3007105143e3c4c5cc5514e7365bcfbdbb4cc0e3", WRITER_RESULT_COMMIT),
    FilePin("writer_v3", "result", "runs/tno_10my_full128_writer_qualification_v3/QUALIFICATION_RESULT.json", 21_827, "c63d880e14c172130de57d6469027809cb1d50dc5900f103b8757cad08f25a07", WRITER_RESULT_COMMIT),

    # Relocation-safe batch-0 full129 GPU smoke.
    FilePin("smoke_v3", "source", "benchmarks/jx_tno_10my_full128_gpu_smoke_v3.py", 39_277, "d560c4e66082c7d1e3bb5b69dcdd1671e8cf4f22ea22e145e7aaff2dc3880df9", SMOKE_PREREG_COMMIT),
    FilePin("smoke_v3", "test", "tests/test_tno_10my_full128_gpu_smoke_v3.py", 18_460, "8a464269569d12624ad262faac882c7b367604865bb5da323aec72c89a2e14a0", SMOKE_PREREG_COMMIT),
    FilePin("smoke_v3", "protocol", "runs/tno_10my_full128_gpu_smoke_v3/PROTOCOL.json", 23_245, "92b2ad9069de25795163154ba70a67c80a1a62f8feb07c4bde925b3e437edac3", SMOKE_PREREG_COMMIT),
    FilePin("smoke_v3", "checkpoints", "runs/tno_10my_full128_gpu_smoke_v3/CHECKPOINTS.bin", 135_072, "07bcfb91100fa9a5f9ae23fb225586a0b37502d8b12ec6658e6412ba0166781d", SMOKE_RESULT_COMMIT),
    FilePin("smoke_v3", "continuation", "runs/tno_10my_full128_gpu_smoke_v3/CONTINUATION.bin", 130_032, "0c13da9af19c55952424da7acecc82c982ff8c3ef381a7822b1f91b1abe96980", SMOKE_RESULT_COMMIT),
    FilePin("smoke_v3", "lifecycle", "runs/tno_10my_full128_gpu_smoke_v3/LIFECYCLE.bin", 2_709, "d5fe0dd289a74ae355cbc3e9d7fb7c86ce42fe5a8492683df82cac7331b5ebce", SMOKE_RESULT_COMMIT),
    FilePin("smoke_v3", "runtime_pre", "runs/tno_10my_full128_gpu_smoke_v3/RUNTIME_PRE.json", 28_292, "c7004fff502b4e1a30e6c137e9f055d3c01b7c5cff165da3961752369eadfaf2", SMOKE_RESULT_COMMIT),
    FilePin("smoke_v3", "runtime_post", "runs/tno_10my_full128_gpu_smoke_v3/RUNTIME_POST.json", 15_310, "ab29fbf42b175f8f80b54e0ffd70cba67d17c36919e09b5267ea746fcd8c71ea", SMOKE_RESULT_COMMIT),
    FilePin("smoke_v3", "segments", "runs/tno_10my_full128_gpu_smoke_v3/SEGMENTS.jsonl", 30_267, "c552dc4e360ddf93b155310f49ca1bcca73da1a893fa8e9213ddc6475dd64f28", SMOKE_RESULT_COMMIT),
    FilePin("smoke_v3", "result", "runs/tno_10my_full128_gpu_smoke_v3/RESULT.json", 24_919, "93ee3db258ac21a9a17c964da96688d561f16affaee830297b638912041ed66f", SMOKE_RESULT_COMMIT),

    # Fresh-process midpoint restart qualification.
    FilePin("restart_v1", "source", "benchmarks/jx_tno_1kyr_full129_restart_qualification_v1.py", 117_439, "c3efc220d49ce08ab08e0ba617626b065764650e9e7bef7934a79b1866b2006c", RESTART_PREREG_COMMIT),
    FilePin("restart_v1", "test", "tests/test_tno_1kyr_full129_restart_qualification_v1.py", 25_238, "a7ac98390f18b1eef5603799ac76213166272f36713957f6648da94764b445e9", RESTART_PREREG_COMMIT),
    FilePin("restart_v1", "protocol", "runs/tno_1kyr_full129_restart_qualification_v1/PROTOCOL.json", 17_613, "93d09da17dffbc98b8bdccf423b906dec14628a63ed22680ba5d84c4db33c522", RESTART_PREREG_COMMIT),
    FilePin("restart_v1", "checkpoint_suffix", "runs/tno_1kyr_full129_restart_qualification_v1/CHECKPOINT_SUFFIX.bin", 70_752, "be2b9335069b3cfe673fcbaebee63c380337201deb7f76073a8a7d578a30bebf", RESTART_RESULT_COMMIT),
    FilePin("restart_v1", "lifecycle_suffix", "runs/tno_1kyr_full129_restart_qualification_v1/LIFECYCLE_SUFFIX.bin", 1_419, "7a5a2b7d1ce40ed7c5612cd9c7823b0e3998fa2d0dfe602e017c65b409b3c9f1", RESTART_RESULT_COMMIT),
    FilePin("restart_v1", "removals", "runs/tno_1kyr_full129_restart_qualification_v1/REMOVALS.jsonl", 0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", RESTART_RESULT_COMMIT),
    FilePin("restart_v1", "restart_request", "runs/tno_1kyr_full129_restart_qualification_v1/RESTART_REQUEST.json", 1_228, "2a10b9bf01834c867f0029af8cbfb6e21bd19e74db476fbb5621fb76f3c2fb57", RESTART_RESULT_COMMIT),
    FilePin("restart_v1", "restart_suffix", "runs/tno_1kyr_full129_restart_qualification_v1/RESTART_SUFFIX.bin", 68_112, "7dd417e4ad52278dabdeadc84eb5c59974553fe80979e9ad49c56c2b4869673d", RESTART_RESULT_COMMIT),
    FilePin("restart_v1", "routes", "runs/tno_1kyr_full129_restart_qualification_v1/ROUTES.jsonl", 0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", RESTART_RESULT_COMMIT),
    FilePin("restart_v1", "runtime_pre", "runs/tno_1kyr_full129_restart_qualification_v1/RUNTIME_PRE.json", 28_589, "586cf12a92692330263372871ef0523718bc4a4a4743095239d772daebce3406", RESTART_RESULT_COMMIT),
    FilePin("restart_v1", "runtime_post", "runs/tno_1kyr_full129_restart_qualification_v1/RUNTIME_POST.json", 15_623, "ca418aca63f2dafad5bc76073cf663fc46cd88ada57e5057ebf893b1903c44a7", RESTART_RESULT_COMMIT),
    FilePin("restart_v1", "segments", "runs/tno_1kyr_full129_restart_qualification_v1/SEGMENTS.jsonl", 16_741, "cd9dd145c835b98d1b5dd9d536edd169d90ef1a5215ecef4ce6c6c4c89eac7ee", RESTART_RESULT_COMMIT),
    FilePin("restart_v1", "result", "runs/tno_1kyr_full129_restart_qualification_v1/RESULT.json", 16_634, "43a838d2faab571355c4d278702063c0e794bae0a3ff7c684fb2969deb26b98c", RESTART_RESULT_COMMIT),

    # Full129 bridge and production trajectory grid, preregistered together.
    FilePin("bridge_v1", "source", "benchmarks/jx_tno_full128_reduced5_bridge_v1.py", 68_486, "7b6e0ee72618410da95639628c74feb5113e62a202d6d51c4a103795fcd8a4df", BRIDGE_GRID_COMMIT),
    FilePin("bridge_v1", "test", "tests/test_tno_full128_reduced5_bridge_v1.py", 18_430, "df375289cce1a8f66dae0c060a1b952ad38d55c3fe5bf83f5cf78ab372a9494d", BRIDGE_GRID_COMMIT),
    FilePin("bridge_v1", "protocol", "runs/tno_full128_reduced5_bridge_v1/PROTOCOL.json", 10_469, "a97ce6daefd2417606219744cb168ab3dcb1cb2de676a1f1671bef10493b5ccb", BRIDGE_GRID_COMMIT),
    FilePin("bridge_v1", "bridge", "runs/tno_full128_reduced5_bridge_v1/BRIDGE.json", 1_208_834, "0570e7d81917caad871de2b2a143d4fc083851894d3c72aab1b77f8e9591fcec", BRIDGE_GRID_COMMIT),
    FilePin("grid_v1", "source", "benchmarks/jx_tno_10my_full128_trajectory_grid_v1.py", 77_247, "3dbc7ad7e5825b863c47b17ad0311e4986538df98d26ea7545b9fb53bb4562d0", BRIDGE_GRID_COMMIT),
    FilePin("grid_v1", "test", "tests/test_tno_10my_full128_trajectory_grid_v1.py", 22_002, "083f68468639183697b1b2ae0df2b3c6e300817d0e8079aab7ff5a84234f9c22", BRIDGE_GRID_COMMIT),
    FilePin("grid_v1", "protocol", "runs/tno_10my_full128_trajectory_grid_v1/PROTOCOL.json", 8_234, "fd8106e150bdf174f631ee25bc30f38bf09e5f006c4fd2e8bf26866f8b4e7e93", BRIDGE_GRID_COMMIT),

    # Reduced-five predecessor freezes deterministic massive-source chunk replay.
    FilePin("grid_adapter_v1", "source", "benchmarks/jx_tno_10my_reduced5_grid_adapter_v1.py", 111_493, "324498cd36726e461ab6910fdf0fe13a1b024346293558f535bea557cdb61b3a", BRIDGE_GRID_COMMIT),
    FilePin("grid_adapter_v1", "test", "tests/test_tno_10my_reduced5_grid_adapter_v1.py", 37_454, "909e6f63ff229e8909245001728acd58a8d4bb7df80993a14b578659c6dd96b8", BRIDGE_GRID_COMMIT),
    FilePin("grid_adapter_v1", "protocol", "runs/tno_10my_reduced5_grid_adapter_v1/PROTOCOL.json", 11_844, "6b10412f06f3c398646179fd163347ac269c03b08486c867e4da821e082be3f2", BRIDGE_GRID_COMMIT),

    # Twelve-batch reduced-five 1 kyr lifecycle evidence used by the bridge/grid lineage.
    FilePin("lifecycle_v3", "source", "benchmarks/jx_tno_1kyr_reduced5_lifecycle_gate_v3.py", 151_995, "02e72fb6a635b74dd31dc364e18cbaf03c5b9e1833caf5aba4a063d7198e47c7", LIFECYCLE_PREREG_COMMIT),
    FilePin("lifecycle_v3", "test", "tests/test_tno_1kyr_reduced5_lifecycle_gate_v3.py", 42_017, "b4eb43b00e1fab0055a43cc4f26db2bef6bf29d51656bcfab1d792596b7845a8", LIFECYCLE_PREREG_COMMIT),
    FilePin("lifecycle_v3", "protocol", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/PROTOCOL.json", 9_178, "9c1e7f77b3a2fd9f6af5542aa2ee3fa7dc00d669d496c809da28b675c8519199", LIFECYCLE_PREREG_COMMIT),
    FilePin("lifecycle_v3", "prefix", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/PREFIX_STATES.bin", 169_344, "a27a929f6f440fb7484ef1bc3c9d55c94a05b24cdb28be0f57904e319b1a9a9b", LIFECYCLE_RESULT_COMMIT),
    FilePin("lifecycle_v3", "restart_streams", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/RESTART_STREAMS.bin", 9_504, "3330f6d83b71f32c2a0fb14bc9f2e26ab13def77f1ea17527d69552473c80173", LIFECYCLE_RESULT_COMMIT),
    FilePin("lifecycle_v3", "routes", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/ROUTES.jsonl", 0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", LIFECYCLE_RESULT_COMMIT),
    FilePin("lifecycle_v3", "runtime", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/RUNTIME.json", 12_980, "43bc1edba432bed75f8a9eb1db9784c13116e1a93e20bb4b43cb634d6abd1dcd", LIFECYCLE_RESULT_COMMIT),
    FilePin("lifecycle_v3", "runtime_preflight", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/RUNTIME_PREFLIGHT.json", 26_179, "7df113a70bcb1012ea05b2145f141bd8b788c504fc61c0f2b075671d0810ea80", LIFECYCLE_RESULT_COMMIT),
    FilePin("lifecycle_v3", "segments", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/SEGMENTS.jsonl", 239_895, "e9452debe803a7213d0878c6c2bd35199bfe966e7e6891f12d072113c825f64f", LIFECYCLE_RESULT_COMMIT),
    FilePin("lifecycle_v3", "result", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/RESULT.json", 66_669, "3d85ae01fa826d0132fc0cbe6188159f2810bc9cf240a9daec7e2b3525054aa4", LIFECYCLE_RESULT_COMMIT),
)

ROOT_POLICIES = {
    "writer_v3": {
        "path": "runs/tno_10my_full128_writer_qualification_v3",
        "allowed_rosters": (
            frozenset({"PROTOCOL.json", "DIGEST_ROSTER.jsonl", "QUALIFICATION_RESULT.json"}),
            frozenset({"PROTOCOL.json", "DIGEST_ROSTER.jsonl", "QUALIFICATION_RESULT.json", "EVIDENCE", "FINAL"}),
        ),
    },
    "smoke_v3": {
        "path": "runs/tno_10my_full128_gpu_smoke_v3",
        "allowed_rosters": (frozenset({"PROTOCOL.json", "CHECKPOINTS.bin", "CONTINUATION.bin", "LIFECYCLE.bin", "RUNTIME_PRE.json", "RUNTIME_POST.json", "SEGMENTS.jsonl", "RESULT.json"}),),
    },
    "restart_v1": {
        "path": "runs/tno_1kyr_full129_restart_qualification_v1",
        "allowed_rosters": (frozenset({"PROTOCOL.json", "CHECKPOINT_SUFFIX.bin", "LIFECYCLE_SUFFIX.bin", "REMOVALS.jsonl", "RESTART_REQUEST.json", "RESTART_SUFFIX.bin", "ROUTES.jsonl", "RUNTIME_PRE.json", "RUNTIME_POST.json", "SEGMENTS.jsonl", "RESULT.json"}),),
    },
    "bridge_v1": {
        "path": "runs/tno_full128_reduced5_bridge_v1",
        "allowed_rosters": (frozenset({"PROTOCOL.json", "BRIDGE.json"}),),
    },
    "grid_v1": {
        "path": "runs/tno_10my_full128_trajectory_grid_v1",
        "allowed_rosters": (frozenset({"PROTOCOL.json"}),),
    },
    "grid_adapter_v1": {
        "path": "runs/tno_10my_reduced5_grid_adapter_v1",
        "allowed_rosters": (frozenset({"PROTOCOL.json"}),),
    },
    "lifecycle_v3": {
        "path": "runs/tno_1kyr_reduced5_lifecycle_gate_v3",
        "allowed_rosters": (frozenset({"PROTOCOL.json", "PREFIX_STATES.bin", "RESTART_STREAMS.bin", "ROUTES.jsonl", "RUNTIME.json", "RUNTIME_PREFLIGHT.json", "SEGMENTS.jsonl", "RESULT.json"}),),
    },
}

WRITER_CLAIMS = {
    "classification_generated": False,
    "dynamics_executed": False,
    "production_scale_writer_qualified": True,
    "resonance_classification_generated": False,
    "science_result": False,
    "synthetic_io_only": True,
    "ten_myr_execution_authorized": False,
    "ten_myr_trajectory_acquired": False,
}
SMOKE_CLAIMS = {
    "bounded_gpu_smoke_executed": True,
    "bounded_gpu_smoke_passed": True,
    "classification_generated": False,
    "gpu_executed": True,
    "planet_nine_evidence": False,
    "population_inference": False,
    "robust_128_clone_classification_generated": False,
    "science_result": False,
    "ten_myr_execution_authorized": False,
    "ten_myr_execution_completed": False,
}
RESTART_CLAIMS = {
    "batch0_full129_all_active_restart_equivalence_qualified": True,
    "classification_generated": False,
    "general_disk_checkpoint_restart_qualified": False,
    "gpu_executed": True,
    "restart_execution_completed": True,
    "routed_or_removed_restart_qualified": False,
    "science_result": False,
    "ten_myr_execution_authorized": False,
}
LIFECYCLE_CLAIMS = {
    "classification_generated": False,
    "dynamics_executed": True,
    "encounter_route_qualified": False,
    "one_thousand_year_lifecycle_gate_passed": True,
    "planet_nine_evidence": False,
    "population_inference": False,
    "resonance_classification_generated": False,
    "science_result": False,
    "ten_myr_execution_authorized": False,
    "ten_myr_execution_completed": False,
}
BRIDGE_CLAIMS = {
    "artifact_prepared": True,
    "classification_generated": False,
    "dynamics_executed": False,
    "observational_likelihood_evaluated": False,
    "planet_nine_evidence": False,
    "population_inference": False,
    "resonance_classification_generated": False,
    "science_result": False,
    "ten_myr_execution_authorized": False,
    "ten_myr_execution_completed": False,
}
GRID_CLAIMS = {
    "classification_generated": False,
    "dynamics_executed": False,
    "planet_nine_evidence": False,
    "population_inference": False,
    "resonance_classification_generated": False,
    "science_result": False,
    "ten_myr_execution_authorized": False,
    "ten_myr_trajectory_acquired": False,
    "trajectory_acquisition_execution_authorized": False,
    "trajectory_acquisition_mechanics_defined": True,
}
GRID_ADAPTER_CLAIMS = {
    "classification_generated": False,
    "first_1000_year_prefix_crossbound": False,
    "first_1000_year_prefix_gate_implemented": True,
    "planet_nine_evidence": False,
    "production_scale_writer_qualified": False,
    "reduced5_encounter_adapter_composition_frozen": True,
    "reduced5_encounter_adapter_executed": False,
    "reduced5_encounter_adapter_qualified": False,
    "science_result": False,
    "synthetic_mechanical_qualification_possible": True,
    "ten_myr_execution_authorized": False,
    "ten_myr_execution_completed": False,
    "writer_and_validator_implemented": True,
}


def _sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise TrajectoryAcquisitionError("SHA-256 input must be exact bytes")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii") + b"\n"
    except (TypeError, ValueError) as exc:
        raise TrajectoryAcquisitionError("value is not canonical-JSON encodable") from exc


def _domain_hash(domain: bytes, value: Mapping[str, Any]) -> str:
    if type(domain) is not bytes or type(value) is not dict:
        raise TrajectoryAcquisitionError("domain-hash input differs")
    record = dict(value)
    record.pop("content_sha256", None)
    return _sha256(domain + _canonical_json(record))


def _strict_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise TrajectoryAcquisitionError("JSON has a duplicate or invalid key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise TrajectoryAcquisitionError(f"JSON has non-finite constant {value}")


def _strict_json(payload: bytes) -> dict[str, Any]:
    if type(payload) is not bytes:
        raise TrajectoryAcquisitionError("JSON payload must be exact bytes")
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrajectoryAcquisitionError("input is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise TrajectoryAcquisitionError("JSON root must be an exact object")
    return value


def _exact_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(_exact_equal(left[key], right[key]) for key in left)
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(_exact_equal(a, b) for a, b in zip(left, right, strict=True))
    return bool(left == right)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
        value.st_size, value.st_mtime_ns, value.st_ctime_ns,
    )


def _require_no_symlink_ancestors(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:-1]:
        current = current / component
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise TrajectoryAcquisitionError(f"{label} has a symlink or non-directory ancestor")


@contextmanager
def _held_directory(path: Path, label: str) -> Iterator[int]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    _require_no_symlink_ancestors(absolute, label)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(absolute.parent, flags)
    descriptor = -1
    try:
        lexical = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(lexical.st_mode):
            raise TrajectoryAcquisitionError(f"{label} is not a directory")
        descriptor = os.open(absolute.name, flags, dir_fd=parent_fd)
        before = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_mode) != (lexical.st_dev, lexical.st_ino, lexical.st_mode):
            raise TrajectoryAcquisitionError(f"{label} changed before open")
        yield descriptor
        after = os.fstat(descriptor)
        lexical_after = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            (after.st_dev, after.st_ino, after.st_mode) != (before.st_dev, before.st_ino, before.st_mode)
            or (lexical_after.st_dev, lexical_after.st_ino, lexical_after.st_mode)
            != (before.st_dev, before.st_ino, before.st_mode)
        ):
            raise TrajectoryAcquisitionError(f"{label} changed during use")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _read_exact_file(path: Path, byte_length: int, digest: str, label: str) -> bytes:
    if type(byte_length) is not int or byte_length < 0 or not _is_sha256(digest):
        raise TrajectoryAcquisitionError(f"{label} pin differs")
    absolute = Path(os.path.abspath(os.fspath(path)))
    _require_no_symlink_ancestors(absolute, label)
    lexical = os.stat(absolute, follow_symlinks=False)
    if not stat.S_ISREG(lexical.st_mode) or lexical.st_nlink != 1 or lexical.st_size != byte_length:
        raise TrajectoryAcquisitionError(f"{label} is not the exact single-link regular file")
    descriptor = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(lexical):
            raise TrajectoryAcquisitionError(f"{label} changed before open")
        chunks: list[bytes] = []
        remaining = byte_length
        offset = 0
        hasher = hashlib.sha256()
        while remaining:
            chunk = os.pread(descriptor, min(1 << 20, remaining), offset)
            if not chunk:
                raise TrajectoryAcquisitionError(f"{label} ended early")
            chunks.append(chunk)
            hasher.update(chunk)
            offset += len(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        lexical_after = os.stat(absolute, follow_symlinks=False)
        if _stat_identity(after) != _stat_identity(before) or _stat_identity(lexical_after) != _stat_identity(before):
            raise TrajectoryAcquisitionError(f"{label} changed during read")
        if hasher.hexdigest() != digest:
            raise TrajectoryAcquisitionError(f"{label} SHA-256 differs")
        return payload
    finally:
        os.close(descriptor)


def _read_single_link_regular_file(path: Path, label: str, maximum_bytes: int) -> bytes:
    if type(maximum_bytes) is not int or maximum_bytes < 1:
        raise TrajectoryAcquisitionError(f"{label} maximum byte count differs")
    absolute = Path(os.path.abspath(os.fspath(path)))
    _require_no_symlink_ancestors(absolute, label)
    lexical = os.stat(absolute, follow_symlinks=False)
    if (
        not stat.S_ISREG(lexical.st_mode)
        or lexical.st_nlink != 1
        or lexical.st_size < 1
        or lexical.st_size > maximum_bytes
    ):
        raise TrajectoryAcquisitionError(f"{label} is not a bounded single-link regular file")
    descriptor = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(lexical):
            raise TrajectoryAcquisitionError(f"{label} changed before open")
        payload = bytearray()
        while len(payload) < before.st_size:
            chunk = os.pread(
                descriptor, min(1 << 20, before.st_size - len(payload)), len(payload)
            )
            if not chunk:
                raise TrajectoryAcquisitionError(f"{label} ended early")
            payload.extend(chunk)
        after = os.fstat(descriptor)
        lexical_after = os.stat(absolute, follow_symlinks=False)
        if (
            _stat_identity(after) != _stat_identity(before)
            or _stat_identity(lexical_after) != _stat_identity(before)
        ):
            raise TrajectoryAcquisitionError(f"{label} changed during read")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _git_bytes(arguments: Sequence[str], label: str) -> bytes:
    if type(arguments) not in (list, tuple) or any(type(item) is not str for item in arguments):
        raise TrajectoryAcquisitionError(f"{label} Git arguments differ")
    completed = subprocess.run(
        ["git", *arguments], cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-2_000:]
        raise TrajectoryAcquisitionError(f"{label} Git query failed: {detail}")
    return completed.stdout


def _git_text(arguments: Sequence[str], label: str) -> str:
    try:
        return _git_bytes(arguments, label).decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise TrajectoryAcquisitionError(f"{label} Git output is not ASCII") from exc


def _verify_git_pin(pin: GitPin) -> dict[str, Any]:
    if type(pin) is not GitPin:
        raise TrajectoryAcquisitionError("Git pin type differs")
    observed_oid = _git_text(["rev-parse", f"refs/tags/{pin.tag_name}"], pin.role)
    object_type = _git_text(["cat-file", "-t", pin.tag_object], pin.role)
    payload = _git_bytes(["cat-file", "tag", pin.tag_object], pin.role)
    peeled = _git_text(["rev-parse", f"{pin.tag_name}^{{}}"], pin.role)
    commit_type = _git_text(["cat-file", "-t", pin.peeled_commit], pin.role)
    tree = _git_text(["rev-parse", f"{pin.peeled_commit}^{{tree}}"], pin.role)
    if (
        observed_oid != pin.tag_object or object_type != "tag"
        or len(payload) != pin.tag_payload_bytes
        or _sha256(payload) != pin.tag_payload_sha256
        or not payload.startswith(f"object {pin.peeled_commit}\ntype commit\n".encode("ascii"))
        or peeled != pin.peeled_commit or commit_type != "commit"
        or tree != pin.commit_tree
    ):
        raise TrajectoryAcquisitionError(f"{pin.role} annotated Git custody differs")
    return {
        "role": pin.role,
        "tag_name": pin.tag_name,
        "tag_object": pin.tag_object,
        "tag_object_type": "tag",
        "tag_payload_byte_length": pin.tag_payload_bytes,
        "tag_payload_sha256": pin.tag_payload_sha256,
        "peeled_commit": pin.peeled_commit,
        "commit_tree": pin.commit_tree,
    }


def _verify_result_lineage(label: str, prereg_commit: str, result_commit: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", prereg_commit, result_commit],
        cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0 or completed.stdout != b"":
        raise TrajectoryAcquisitionError(f"{label} result commit does not descend from preregistration")
    return {
        "gate": label,
        "preregistration_commit": prereg_commit,
        "result_commit": result_commit,
        "result_descends_from_preregistration": True,
    }


def _validate_root_rosters() -> dict[str, Any]:
    records: dict[str, Any] = {}
    for gate in sorted(ROOT_POLICIES):
        policy = ROOT_POLICIES[gate]
        root = PROJECT_ROOT / policy["path"]
        with _held_directory(root, f"{gate} retained root") as root_fd:
            names = frozenset(os.listdir(root_fd))
            if names not in policy["allowed_rosters"]:
                raise TrajectoryAcquisitionError(f"{gate} retained root roster differs")
            for name in names:
                metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                if name in ("EVIDENCE", "FINAL") and gate == "writer_v3":
                    if not stat.S_ISDIR(metadata.st_mode):
                        raise TrajectoryAcquisitionError("writer local payload entry is not a directory")
                elif not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise TrajectoryAcquisitionError(f"{gate} retained root file type differs")
        records[gate] = {
            "relative_path": policy["path"],
            "allowed_rosters": [sorted(roster) for roster in policy["allowed_rosters"]],
            "local_large_payload_directories_optional_for_archive_binding": gate == "writer_v3",
        }
    return records


def _validate_canonical_record(
    payload: bytes, *, schema: str, status: str, content_sha256: str,
    domain: bytes, claims: Mapping[str, bool], label: str,
) -> dict[str, Any]:
    value = _strict_json(payload)
    if (
        payload != _canonical_json(value)
        or value.get("schema") != schema
        or value.get("status") != status
        or value.get("content_sha256") != content_sha256
        or value.get("content_sha256") != _domain_hash(domain, value)
        or not _exact_equal(value.get("claims"), claims)
    ):
        raise TrajectoryAcquisitionError(f"{label} canonical identity or claim ceiling differs")
    return value


def _pin_by(gate: str, role: str) -> FilePin:
    values = [pin for pin in FILE_PINS if pin.gate == gate and pin.role == role]
    if len(values) != 1:
        raise TrajectoryAcquisitionError(f"{gate}/{role} file pin roster differs")
    return values[0]


def _result_file_receipts(value: Mapping[str, Any], gate: str, roles: Mapping[str, str]) -> None:
    expected = {
        output_name: {
            "bytes": _pin_by(gate, role).byte_length,
            "sha256": _pin_by(gate, role).sha256,
        }
        for output_name, role in roles.items()
    }
    if not _exact_equal(value.get("files"), expected):
        raise TrajectoryAcquisitionError(f"{gate} retained result file locators differ")


def _validate_semantic_gates(payloads: Mapping[tuple[str, str], bytes]) -> dict[str, Any]:
    writer_protocol = _validate_canonical_record(
        payloads[("writer_v3", "protocol")],
        schema="jx.tno.10my-full128-writer-qualification.protocol.v3",
        status="PREREGISTERED_SYNTHETIC_IO_WRITER_QUALIFICATION_ONLY",
        content_sha256="79aae04b5c3c7986e37785136f69901879f7db883fa50488191952d92990b1bd",
        domain=b"jx.tno.10my-full128-writer-qualification.protocol.v3\0",
        claims={**WRITER_CLAIMS, "production_scale_writer_qualified": False},
        label="writer V3 protocol",
    )
    writer_result = _validate_canonical_record(
        payloads[("writer_v3", "result")],
        schema="jx.tno.10my-full128-writer-qualification.result.v3",
        status="SYNTHETIC_IO_WRITER_QUALIFICATION_COMPLETE_NO_DYNAMICS_OR_SCIENCE_CLAIM",
        content_sha256="dd6d9fe6528e9353a38029ec3e7a06260857c4ab3dfd861423b20a54d4a84a49",
        domain=b"jx.tno.10my-full128-writer-qualification.result.v3\0",
        claims=WRITER_CLAIMS, label="writer V3 result",
    )
    digest = writer_result.get("digest_roster")
    final_files = writer_result.get("final_files")
    producer = writer_result.get("producer_runtime_identity")
    if (
        type(digest) is not dict
        or digest.get("name") != "DIGEST_ROSTER.jsonl"
        or type(digest.get("logical_byte_length")) is not int
        or digest["logical_byte_length"] != _pin_by("writer_v3", "digest_roster").byte_length
        or digest.get("sha256") != _pin_by("writer_v3", "digest_roster").sha256
        or type(digest.get("record_count")) is not int or digest["record_count"] != 12_001
        or type(digest.get("shard_record_count")) is not int or digest["shard_record_count"] != 12_000
        or type(final_files) is not dict
        or sum(item.get("logical_byte_length", -1) for item in final_files.values()) != 4_743_277_848
        or type(producer) is not dict
        or producer.get("python_optimize") != 0 or type(producer.get("python_optimize")) is not int
        or producer.get("writer_source_sha256") != _pin_by("writer_v3", "source").sha256
        or producer.get("grid_source_sha256") != _pin_by("grid_v1", "source").sha256
    ):
        raise TrajectoryAcquisitionError("writer V3 retained production geometry/custody differs")

    smoke_protocol = _validate_canonical_record(
        payloads[("smoke_v3", "protocol")],
        schema="jx.tno.10my-full128-gpu-smoke.protocol.v3",
        status="PREREGISTERED_EXACT_BATCH0_FULL129_1KYR_GPU_SMOKE_V3_NOT_EXECUTED",
        content_sha256="964b04b6ecbbd37c3f59882275c672488c2aaf37c457bbdb5fb59b682182cb53",
        domain=b"jx.tno.10my-full128-gpu-smoke.protocol.v3\0",
        claims={key: False for key in SMOKE_CLAIMS}, label="smoke V3 protocol",
    )
    smoke_result = _validate_canonical_record(
        payloads[("smoke_v3", "result")],
        schema="jx.tno.10my-full128-gpu-smoke.result.v3",
        status="FULL129_BATCH0_1KYR_GPU_SMOKE_PASS_TEN_MYR_NOT_AUTHORIZED",
        content_sha256="e51ac1578a867e7f2f9e707ef983114992ee9a04cbddecae21d09dd27627a2c8",
        domain=b"jx.tno.10my-full128-gpu-smoke.result.v3\0",
        claims=SMOKE_CLAIMS, label="smoke V3 result",
    )
    _result_file_receipts(smoke_result, "smoke_v3", {
        "CHECKPOINTS.bin": "checkpoints", "CONTINUATION.bin": "continuation",
        "LIFECYCLE.bin": "lifecycle", "RUNTIME_PRE.json": "runtime_pre",
        "RUNTIME_POST.json": "runtime_post", "SEGMENTS.jsonl": "segments",
    })
    if (
        smoke_result.get("batch_order") != 0 or type(smoke_result.get("batch_order")) is not int
        or smoke_result.get("tracer_count") != 129 or type(smoke_result.get("tracer_count")) is not int
        or smoke_result.get("checkpoint_years") != list(range(0, 1_001, 50))
        or smoke_result.get("segment_count") != 20 or type(smoke_result.get("segment_count")) is not int
        or smoke_result.get("route_record_count") != 0 or type(smoke_result.get("route_record_count")) is not int
        or smoke_result.get("removal_record_count") != 0 or type(smoke_result.get("removal_record_count")) is not int
        or smoke_result.get("all_predeclared_gates_passed") is not True
    ):
        raise TrajectoryAcquisitionError("smoke V3 bounded scope differs")

    restart_protocol = _validate_canonical_record(
        payloads[("restart_v1", "protocol")],
        schema="jx.tno.1kyr-full129-restart-qualification.protocol.v1",
        status="PREPARED_BATCH0_FULL129_YEAR500_TO1000_RESTART_NOT_EXECUTED",
        content_sha256="8081c851c5977fc3fdc042d1970edf428f1fbb5823da64aa5b00e6af1f3e09dd",
        domain=b"jx.tno.1kyr-full129-restart-qualification.protocol.v1\0",
        claims={key: False for key in RESTART_CLAIMS}, label="restart V1 protocol",
    )
    restart_result = _validate_canonical_record(
        payloads[("restart_v1", "result")],
        schema="jx.tno.1kyr-full129-restart-qualification.result.v1",
        status="BATCH0_FULL129_ALL_ACTIVE_YEAR500_TO1000_FRESH_PROCESS_RESTART_PASS_TEN_MYR_NOT_AUTHORIZED",
        content_sha256="a1c480144bf3a96ed0b21bdfb311d7063cff2362f80d483c8b42d60d5f5ffa94",
        domain=b"jx.tno.1kyr-full129-restart-qualification.result.v1\0",
        claims=RESTART_CLAIMS, label="restart V1 result",
    )
    _result_file_receipts(restart_result, "restart_v1", {
        "CHECKPOINT_SUFFIX.bin": "checkpoint_suffix", "LIFECYCLE_SUFFIX.bin": "lifecycle_suffix",
        "REMOVALS.jsonl": "removals", "RESTART_REQUEST.json": "restart_request",
        "RESTART_SUFFIX.bin": "restart_suffix", "ROUTES.jsonl": "routes",
        "RUNTIME_PRE.json": "runtime_pre", "RUNTIME_POST.json": "runtime_post",
        "SEGMENTS.jsonl": "segments",
    })
    if (
        restart_result.get("batch_order") != 0 or type(restart_result.get("batch_order")) is not int
        or restart_result.get("tracer_count") != 129 or type(restart_result.get("tracer_count")) is not int
        or restart_result.get("start_year") != 500 or type(restart_result.get("start_year")) is not int
        or restart_result.get("end_year") != 1_000 or type(restart_result.get("end_year")) is not int
        or restart_result.get("segment_count") != 10 or type(restart_result.get("segment_count")) is not int
        or restart_result.get("route_record_count") != 0 or type(restart_result.get("route_record_count")) is not int
        or restart_result.get("removal_record_count") != 0 or type(restart_result.get("removal_record_count")) is not int
        or restart_result.get("all_predeclared_gates_passed") is not True
    ):
        raise TrajectoryAcquisitionError("restart V1 bounded scope differs")

    lifecycle_protocol = _validate_canonical_record(
        payloads[("lifecycle_v3", "protocol")],
        schema="jx.tno.1kyr-reduced5-lifecycle-gate.protocol.v3",
        status="PREREGISTERED_BOUNDED_1KYR_GATE_NOT_YET_EXECUTED",
        content_sha256="538e076274ab5e1c850cce069405ba67c3cab41226fa684a686aaaa81cdc32d9",
        domain=b"jx.tno.1kyr-reduced5-lifecycle-gate.protocol.v3\0",
        claims={**LIFECYCLE_CLAIMS, "dynamics_executed": False, "one_thousand_year_lifecycle_gate_passed": False},
        label="lifecycle V3 protocol",
    )
    lifecycle_result = _validate_canonical_record(
        payloads[("lifecycle_v3", "result")],
        schema="jx.tno.1kyr-reduced5-lifecycle-gate.result.v3",
        status="ONE_THOUSAND_YEAR_LIFECYCLE_PREFIX_PASS_TEN_MYR_NOT_AUTHORIZED",
        content_sha256="d1dffed406f02774f4af874489db4a8209570ebeadffdc6a2e5cc2ea43b39264",
        domain=b"jx.tno.1kyr-reduced5-lifecycle-gate.result.v3\0",
        claims=LIFECYCLE_CLAIMS, label="lifecycle V3 result",
    )

    bridge_protocol = _validate_canonical_record(
        payloads[("bridge_v1", "protocol")],
        schema="jx.tno.full128-reduced5-bridge.protocol.v1",
        status="PREPARE_ONLY_FULL128_NATIVE_EPOCH_BRIDGE_NO_DYNAMICS",
        content_sha256="0458f700c64e798ecbfa927e6d722cd6d3b2b4ae8d6f70765f3fa2ea5684a0cd",
        domain=b"jx.tno.full128-reduced5-bridge.protocol.v1\0",
        claims={**BRIDGE_CLAIMS, "artifact_prepared": False}, label="bridge V1 protocol",
    )
    bridge = _validate_canonical_record(
        payloads[("bridge_v1", "bridge")],
        schema="jx.tno.full128-reduced5-bridge.states.v1",
        status="TWELVE_FULL128_NATIVE_EPOCH_REDUCED5_BATCHES_PREPARED_NO_DYNAMICS",
        content_sha256="2c095be924f10481ced30af205836ebd18b4113398a7688b77f06259a2ac780b",
        domain=b"jx.tno.full128-reduced5-bridge.states.v1\0",
        claims=BRIDGE_CLAIMS, label="bridge V1 artifact",
    )
    if (
        bridge.get("batch_count") != 12 or type(bridge.get("batch_count")) is not int
        or bridge.get("tracer_count_per_batch") != 129 or type(bridge.get("tracer_count_per_batch")) is not int
        or bridge.get("all_twelve_selected9_projections_bitwise_equal") is not True
        or bridge.get("execution_authorized") is not False
    ):
        raise TrajectoryAcquisitionError("bridge V1 full129 identity differs")

    grid_protocol = _validate_canonical_record(
        payloads[("grid_v1", "protocol")],
        schema="jx.tno.10my-full128-trajectory-grid.protocol.v1",
        status="PREPARE_ONLY_TRAJECTORY_ACQUISITION_MECHANICS_NO_DYNAMICS",
        content_sha256="da6dce31b784f50f96e40b33ddb39d78a4421c47329095dc915a507a1996ad49",
        domain=b"jx.tno.10my-full128-trajectory-grid.protocol.v1\0",
        claims=GRID_CLAIMS, label="grid V1 protocol",
    )
    geometry = grid_protocol.get("geometry")
    if (
        type(geometry) is not dict
        or geometry.get("batch_count") != 12 or type(geometry.get("batch_count")) is not int
        or geometry.get("tracer_count") != 129 or type(geometry.get("tracer_count")) is not int
        or geometry.get("shards_total") != 12_000 or type(geometry.get("shards_total")) is not int
        or geometry.get("total_steps") != 640_000_000 or type(geometry.get("total_steps")) is not int
        or geometry.get("fixed_binary_bytes") != 4_743_277_848 or type(geometry.get("fixed_binary_bytes")) is not int
        or geometry.get("no_element_file_in_this_version") is not True
    ):
        raise TrajectoryAcquisitionError("grid V1 production geometry differs")

    grid_adapter_protocol = _validate_canonical_record(
        payloads[("grid_adapter_v1", "protocol")],
        schema="jx.tno.10my-reduced5-grid-adapter.protocol.v1",
        status="PREPARE_ONLY_WRITER_IMPLEMENTED_ADAPTER_COMPOSITION_FROZEN_EXECUTION_BLOCKED",
        content_sha256="f13b551ea074ce6233c9f3b8aa5e0b75421aa26639141564312087af331812a9",
        domain=b"jx.tno.10my-reduced5-grid-adapter.protocol.v1\0",
        claims=GRID_ADAPTER_CLAIMS, label="reduced-five grid adapter V1 protocol",
    )
    source_retention = grid_adapter_protocol.get("full_source_node_retention")
    if (
        type(source_retention) is not dict
        or source_retention.get("all_batches_bytes") != 1_843_200_002_880
        or type(source_retention.get("all_batches_bytes")) is not int
        or source_retention.get("nodes_per_batch") != 640_000_001
        or type(source_retention.get("nodes_per_batch")) is not int
        or source_retention.get("per_source_node_bytes") != 240
        or type(source_retention.get("per_source_node_bytes")) is not int
        or source_retention.get("retained") is not False
        or source_retention.get("replacement")
        != "FIXED_WIDTH_SOURCE_CHUNK_RECEIPTS_PLUS_DETERMINISTIC_CHUNK_REPLAY"
    ):
        raise TrajectoryAcquisitionError("grid adapter V1 source retention geometry differs")

    return {
        "writer_v3": {
            "protocol_content_sha256": writer_protocol["content_sha256"],
            "result_content_sha256": writer_result["content_sha256"],
            "production_scale_synthetic_writer_qualified": True,
            "real_dynamics_adapter_qualified": False,
        },
        "smoke_v3": {
            "protocol_content_sha256": smoke_protocol["content_sha256"],
            "result_content_sha256": smoke_result["content_sha256"],
            "scope": "BATCH0_FULL129_1KYR_ALL_ACTIVE",
            "passed": True,
            "ten_myr_authorized": False,
        },
        "restart_v1": {
            "protocol_content_sha256": restart_protocol["content_sha256"],
            "result_content_sha256": restart_result["content_sha256"],
            "scope": "BATCH0_FULL129_ALL_ACTIVE_YEAR500_TO1000",
            "passed": True,
            "general_disk_checkpoint_restart_qualified": False,
            "routed_or_removed_restart_qualified": False,
        },
        "lifecycle_v3": {
            "protocol_content_sha256": lifecycle_protocol["content_sha256"],
            "result_content_sha256": lifecycle_result["content_sha256"],
            "scope": "TWELVE_REDUCED5_SELECTED9_1KYR",
            "passed": True,
            "encounter_route_qualified": False,
        },
        "bridge_v1": {
            "protocol_content_sha256": bridge_protocol["content_sha256"],
            "artifact_content_sha256": bridge["content_sha256"],
            "twelve_full129_batches_prepared": True,
            "dynamics_executed": False,
        },
        "grid_v1": {
            "protocol_content_sha256": grid_protocol["content_sha256"],
            "production_geometry_frozen": True,
            "execution_authorized": False,
        },
        "grid_adapter_v1": {
            "protocol_content_sha256": grid_adapter_protocol["content_sha256"],
            "source_corpus_if_fully_retained_bytes": 1_843_200_002_880,
            "source_corpus_retained": False,
            "deterministic_source_chunk_replay_specified": True,
            "execution_authorized": False,
        },
    }


def validate_bound_prerequisites() -> dict[str, Any]:
    roots = _validate_root_rosters()
    payloads: dict[tuple[str, str], bytes] = {}
    files: dict[str, list[dict[str, Any]]] = {}
    for pin in FILE_PINS:
        payload = _read_exact_file(
            PROJECT_ROOT / pin.relative_path, pin.byte_length, pin.sha256,
            f"{pin.gate}/{pin.role}",
        )
        committed = _git_bytes(
            ["show", "--no-ext-diff", "--no-textconv", f"{pin.commit}:{pin.relative_path}"],
            f"{pin.gate}/{pin.role} committed bytes",
        )
        if committed != payload:
            raise TrajectoryAcquisitionError(f"{pin.gate}/{pin.role} live bytes differ from bound commit")
        payloads[(pin.gate, pin.role)] = payload
        files.setdefault(pin.gate, []).append({
            "role": pin.role,
            "relative_path": pin.relative_path,
            "byte_length": pin.byte_length,
            "sha256": pin.sha256,
            "committed_at": pin.commit,
            "live_equals_committed": True,
        })
    git_objects = [_verify_git_pin(pin) for pin in GIT_PINS]
    lineages = [_verify_result_lineage(*item) for item in RESULT_LINEAGES]
    semantics = _validate_semantic_gates(payloads)
    return {
        "root_roster_policies": roots,
        "exact_files_by_gate": {
            gate: sorted(records, key=lambda item: item["role"])
            for gate, records in sorted(files.items())
        },
        "annotated_git_objects": git_objects,
        "result_commit_lineages": lineages,
        "semantic_gate_summary": semantics,
    }


def _implementation_record(path: Path, role: str) -> dict[str, Any]:
    payload = _read_single_link_regular_file(path, role, 2_000_000)
    return {
        "role": role,
        "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
        "byte_length": len(payload),
        "sha256": _sha256(payload),
    }


def _claim_ceiling() -> dict[str, bool]:
    return {
        "prepare_only_protocol_frozen": True,
        "production_execution_implemented": False,
        "production_execution_authorized": False,
        "dynamics_executed": False,
        "gpu_executed": False,
        "ten_myr_trajectory_acquired": False,
        "ten_myr_execution_completed": False,
        "classification_generated": False,
        "resonance_classification_generated": False,
        "population_inference": False,
        "planet_nine_evidence": False,
        "science_result": False,
    }


def build_protocol() -> dict[str, Any]:
    prerequisites = validate_bound_prerequisites()
    protocol: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "status": "PREREGISTRATION_CANDIDATE_PREPARE_ONLY_EXECUTION_NOT_IMPLEMENTED",
        "objective": "Bind every passed full129 prerequisite and freeze the remaining executable boundary without running dynamics.",
        "implementation": {
            "source": _implementation_record(SCRIPT_PATH, "trajectory acquisition source"),
            "test": _implementation_record(TEST_PATH, "trajectory acquisition test"),
        },
        "prerequisite_custody": prerequisites,
        "production_geometry": {
            "batch_count": 12,
            "tracer_count_per_batch": 129,
            "massive_count": 5,
            "body_count": 134,
            "steps_per_julian_year": 64,
            "total_steps_per_batch": 640_000_000,
            "shard_steps": 640_000,
            "shards_per_batch": 1_000,
            "total_shards": 12_000,
            "b22_interval_steps": 12_800,
            "b22_node_count_per_batch": 50_001,
            "ml_short_interval_steps": 3_200,
            "ml_short_final_step": 32_000_000,
            "ml_short_node_count_per_batch": 10_001,
            "state_record_bytes": 6_432,
            "lifecycle_record_bytes": 129,
            "final_binary_bytes": 4_743_277_848,
            "trajectory_only_no_elements_during_dynamics": True,
            "removed_states_retained": True,
        },
        "execution_readiness": {
            "decision": "PREPARE_ONLY_FAIL_CLOSED_NO_EXECUTE_OR_RESUME_SURFACE",
            "determinate_passed_surfaces": [
                "TWELVE_BATCH_FULL129_INITIAL_STATE_BRIDGE",
                "PRODUCTION_12000_SHARD_OUTPUT_GEOMETRY",
                "PRODUCTION_SCALE_SYNTHETIC_HELD_FD_WRITER",
                "BATCH0_FULL129_1KYR_GPU_SMOKE",
                "BATCH0_FULL129_ALL_ACTIVE_MIDPOINT_RESTART",
                "TWELVE_BATCH_SELECTED9_1KYR_LIFECYCLE_PREFIX",
            ],
            "blocking_contracts": [
                "REAL_DYNAMICS_TO_HELD_FD_SHARD_WRITER_ADAPTER_NOT_IMPLEMENTED_OR_QUALIFIED",
                "FULL129_PER_SHARD_CONTINUATION_AND_LIFECYCLE_RESUME_SCHEMA_NOT_FROZEN",
                "ROUTED_OR_REMOVED_TRACER_RESTART_NOT_QUALIFIED",
                "GENERAL_DISK_CHECKPOINT_RESTART_NOT_QUALIFIED",
                "ALL_12_FULL129_GLOBAL_1KYR_STOP_GATE_NOT_EXECUTED",
                "REAL_ROUTE_TRIGGER_INPUT_EVIDENCE_NOT_CAPTURED_OR_QUALIFIED",
                "OUTCOME_BLIND_CAPPED_VARIABLE_EVIDENCE_WRITER_NOT_QUALIFIED",
                "ROUTED_AND_REMOVED_5MYR_RESTART_NOT_QUALIFIED",
                "FOUR_PREREGISTERED_5MYR_BATCH_PILOTS_NOT_COMPLETED",
                "DETERMINISTIC_SOURCE_CHUNK_REGENERATION_AND_RECEIPTS_NOT_QUALIFIED",
                "REAL_SOURCE_ROUTE_RUNTIME_AND_PRODUCER_IDENTITY_RECEIPTS_NOT_CROSSBOUND_TO_WRITER_V3",
                "FRESH_CHILD_12_BATCH_12000_SHARD_ORCHESTRATOR_AND_CRASH_RESUME_VALIDATOR_NOT_IMPLEMENTED",
            ],
            "mandatory_phase_order": [
                "IMPLEMENT_AND_QUALIFY_REAL_DYNAMICS_WRITER_ADAPTER_WITH_REAL_ROUTE_TRIGGER_INPUT_EVIDENCE",
                "QUALIFY_OUTCOME_BLIND_CAPPED_VARIABLE_EVIDENCE_WRITER",
                "EXECUTE_ALL_12_FULL129_GLOBAL_1KYR_STOP_GATE",
                "STOP_AND_REVIEW_ALL_12_1KYR_RESULTS_BEFORE_ANY_BUILD_BEYOND_1KYR",
                "QUALIFY_ROUTED_AND_REMOVED_5MYR_RESTART",
                "EXECUTE_FOUR_SEPARATELY_PREREGISTERED_5MYR_BATCH_PILOTS",
                "STOP_AND_REVIEW_FOUR_5MYR_RESULTS_BEFORE_ANY_10MYR_EXECUTION",
            ],
            "next_executable_successor_requirements": {
                "fresh_child_only": True,
                "production_python_optimize": 0,
                "exact_default_root_only": True,
                "externally_preregistered_commit_and_annotated_tag_required": True,
                "writer_custody": "PORT_V3_HELD_PARENT_ATTEMPT_LOCAL_PARTIAL_MANIFEST_LAST_RESULT_LAST_AND_STORAGE_RESERVATION",
                "runtime_custody": "PORT_SMOKE_V3_PATH_NEUTRAL_NATIVE_RUNTIME_AND_FRESH_CHILD_IMPORT_GUARDS",
                "restart_custody": "RETAIN_FULL129_CONTINUATION_LIFECYCLE_ROUTE_REMOVAL_AND_SOURCE_ENDPOINTS_FOR_EVERY_SHARD",
                "resume_rule": "REUSE_ONLY_FULLY_REPLAYED_SHARDS_AND_REQUIRE_LIVE_RUNTIME_MATCH_BEFORE_ANY_MUTATION",
                "source_corpus_if_fully_retained_bytes": 1_843_200_002_880,
                "source_corpus_decimal_terabytes": "1.843200002880",
                "source_custody": "DETERMINISTICALLY_REGENERATE_AND_CROSSBIND_FIXED_WIDTH_SOURCE_CHUNKS_INSTEAD_OF_RETAINING_THE_FULL_SOURCE_CORPUS",
                "claim_ceiling_until_completed_validation": "NO_10MY_CLASSIFICATION_OR_SCIENCE_AUTHORITY",
            },
        },
        "preregistration": {
            "expected_branch": EXPECTED_BRANCH,
            "annotated_tag": PREREGISTRATION_TAG,
            "future_result_tag": FUTURE_RESULT_TAG,
            "prepared_default_root": DEFAULT_OUTPUT_ROOT.relative_to(PROJECT_ROOT).as_posix(),
            "prepared_roster": ["PROTOCOL.json"],
            "source_and_test_only_before_prepare": True,
            "no_automatic_git_or_execution_action": True,
            "git_custody_interpretation": "CURRENT_LOCAL_ANNOTATED_RESULT_TAG_OBJECTS_AND_COMMITTED_ARCHIVE_BYTES_ARE_EXPLICITLY_VERIFIED;THE_PREDECESSOR_RESULTS_DID_NOT_SELF_AUTHENTICATE_THEIR_FUTURE_RESULT_TAG_OBJECTS",
        },
        "filesystem_trust_boundary": FILESYSTEM_TRUST_BOUNDARY,
        "public_cli": ["build", "audit-prerequisites", "prepare", "validate-prepared"],
        "explicitly_absent_public_cli": ["execute", "execute-worker", "resume", "validate-result"],
        "claims": _claim_ceiling(),
        "content_sha256": "",
    }
    protocol["content_sha256"] = _domain_hash(PROTOCOL_DOMAIN, protocol)
    return protocol


def validate_protocol(value: Mapping[str, Any]) -> str:
    if type(value) is not dict:
        raise TrajectoryAcquisitionError("protocol must be an exact object")
    expected = build_protocol()
    if (
        not _exact_equal(value, expected)
        or value.get("content_sha256") != _domain_hash(PROTOCOL_DOMAIN, value)
        or not _exact_equal(value.get("claims"), _claim_ceiling())
    ):
        raise TrajectoryAcquisitionError("protocol bytes or claim ceiling differ")
    return value["content_sha256"]


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise TrajectoryAcquisitionError("protocol write made no progress")
        view = view[written:]


def _rename_noreplace(directory_fd: int, source: str, destination: str) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise TrajectoryAcquisitionError("renameat2 is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(directory_fd, source.encode("ascii"), directory_fd, destination.encode("ascii"), 1) != 0:
        observed_errno = ctypes.get_errno()
        if observed_errno == errno.EEXIST:
            raise TrajectoryAcquisitionError("protocol destination already exists")
        raise TrajectoryAcquisitionError(f"renameat2 failed: {os.strerror(observed_errno)}")


def prepare(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    protocol = build_protocol()
    payload = _canonical_json(protocol)
    root = Path(os.path.abspath(os.fspath(output_root)))
    if not root.name or root.name in (".", ".."):
        raise TrajectoryAcquisitionError("prepared root name differs")
    _require_no_symlink_ancestors(root, "prepared root")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(root.parent, flags)
    created = False
    try:
        try:
            os.mkdir(root.name, 0o700, dir_fd=parent_fd)
            created = True
        except FileExistsError as exc:
            raise TrajectoryAcquisitionError("prepared root already exists; create-only publication refused") from exc
        root_fd = os.open(root.name, flags, dir_fd=parent_fd)
        try:
            if os.listdir(root_fd):
                raise TrajectoryAcquisitionError("new prepared root is not empty")
            partial = ".PROTOCOL.json.partial"
            descriptor = os.open(
                partial,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=root_fd,
            )
            try:
                _write_all(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            _rename_noreplace(root_fd, partial, "PROTOCOL.json")
            os.fsync(root_fd)
        except Exception:
            for name in tuple(os.listdir(root_fd)):
                metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                if stat.S_ISREG(metadata.st_mode):
                    os.unlink(name, dir_fd=root_fd)
            os.fsync(root_fd)
            raise
        finally:
            os.close(root_fd)
        os.fsync(parent_fd)
    except Exception:
        if created:
            try:
                os.rmdir(root.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                pass
        raise
    finally:
        os.close(parent_fd)
    return validate_prepared(root)


def validate_prepared(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    expected_payload = _canonical_json(build_protocol())
    root = Path(os.path.abspath(os.fspath(output_root)))
    with _held_directory(root, "prepared acquisition root") as root_fd:
        if frozenset(os.listdir(root_fd)) != PREPARED_ROSTER:
            raise TrajectoryAcquisitionError("prepared root roster differs")
        metadata = os.stat("PROTOCOL.json", dir_fd=root_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != len(expected_payload)
        ):
            raise TrajectoryAcquisitionError("prepared protocol is not a single-link regular file")
        descriptor = os.open("PROTOCOL.json", os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=root_fd)
        try:
            before = os.fstat(descriptor)
            payload_buffer = bytearray()
            while len(payload_buffer) < before.st_size:
                chunk = os.pread(
                    descriptor,
                    min(1 << 20, before.st_size - len(payload_buffer)),
                    len(payload_buffer),
                )
                if not chunk:
                    raise TrajectoryAcquisitionError("prepared protocol ended early")
                payload_buffer.extend(chunk)
            after = os.fstat(descriptor)
            if _stat_identity(after) != _stat_identity(before):
                raise TrajectoryAcquisitionError("prepared protocol changed during read")
        finally:
            os.close(descriptor)
        if frozenset(os.listdir(root_fd)) != PREPARED_ROSTER:
            raise TrajectoryAcquisitionError("prepared root roster changed during validation")
    payload = bytes(payload_buffer)
    if payload != expected_payload:
        raise TrajectoryAcquisitionError("prepared protocol bytes differ")
    value = _strict_json(payload)
    if payload != _canonical_json(value):
        raise TrajectoryAcquisitionError("prepared protocol is not canonical")
    validate_protocol(value)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("build", "audit-prerequisites", "prepare", "validate-prepared"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build":
        value = build_protocol()
    elif arguments.command == "audit-prerequisites":
        value = validate_bound_prerequisites()
    elif arguments.command == "prepare":
        value = prepare(arguments.output_root)
    else:
        value = validate_prepared(arguments.output_root)
    sys.stdout.buffer.write(_canonical_json(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
