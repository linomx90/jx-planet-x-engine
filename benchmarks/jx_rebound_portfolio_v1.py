from __future__ import annotations

"""Prepare-only charter for the JX versus REBOUND 5.2.2 portfolio.

This module deliberately has no benchmark execution path.  It freezes the
evidence audit, comparator identity, finite lane roster, and fairness/claim
rules that successor lane protocols must inherit before any new timing run.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
from typing import Any, Iterable, Mapping


SCHEMA = "jx.rebound-portfolio-charter.v1"
STATUS = "PORTFOLIO_CHARTER_PREPARED_NO_LANE_EXECUTION_AUTHORIZED"
BASE_COMMIT = "9d4b6e4f5b5ca65e338effb16bfa1e8acd9ab5cd"
BASE_TREE = "bef89e14cec21d949419962c732ab87855cb7b0b"
REBOUND_VERSION = "5.2.2"
REBOUND_GITHASH = "c34134716cd484ed9cdb9f7d54ef8d859bdf547d"
REBOUND_WHEEL = {
    "filename": (
        "rebound-5.2.2-cp314-cp314-manylinux_2_24_x86_64."
        "manylinux_2_28_x86_64.whl"
    ),
    "bytes": 906885,
    "sha256": "6f971600864e7c2df3cc00df7df8c1d5a399e08d522789a6a9724d12358945b8",
    "pypi_upload_time_utc": "2026-10-05T13:10:26.880767Z",
}
REBOUND_SDIST = {
    "filename": "rebound-5.2.2.tar.gz",
    "bytes": 386244,
    "sha256": "cefb2bce433a134e0f6fa0cb68df01079b0a5c160fadd2a2c85b06f286735a5f",
    "pypi_upload_time_utc": "2026-10-05T13:10:50.100700Z",
}
REBOUND_RUNTIME = {
    "python_module_sha256": (
        "924e19bfddb22bca9cbff0c641eaf3b2f791287f33df2d8d852f80aac78f8cdd"
    ),
    "librebound_sha256": (
        "574c062c4dacf8c912c5821d92c0b1b361d31801998f3248cc55e5e742670f04"
    ),
    "wheel_metadata_member": "rebound-5.2.2.dist-info/METADATA",
    "wheel_metadata_bytes": 9389,
    "wheel_metadata_sha256": (
        "ce67696222e092693a9b0d93c7b57f7cc3c138a466ca02f48811b0bf2ef78661"
    ),
}
REBOUNDX_SDIST = {
    "version": "5.1.0",
    "filename": "reboundx-5.1.0.tar.gz",
    "bytes": 99005,
    "sha256": "25fcccc8194a83b548a8e11d81f295fb12a21ece2538e61be93d8256d9cbc1b2",
    "use": "P6_EIH_BATCH_ONLY",
    "loaded_runtime_identity": "BLOCKED_UNTIL_SUCCESSOR_BUILD_IS_FROZEN",
}

RECOVERED_REPORT_INVENTORY: Mapping[str, Any] = {
    "path": "benchmarks/jx_rebound_portfolio_v1_recovered_report_inventory.json",
    "bytes": 159351,
    "sha256": "d42edaaedd06ad0e10ad3429c0d328566aedb89e824b98c1de06b8c9371b80ea",
    "schema": "jx-recovered-report-inventory-v3",
    "source_audit_inventory_bytes": 222680,
    "source_audit_inventory_sha256": (
        "c763877da8f993219048ed9ccd5e6e641d411a9f99c4e6e5681f8531a9cb3b73"
    ),
    "portable_transform": (
        "REMOVE_113_KEYS_CONTAINING_NONPORTABLE_AND_ADD_"
        "NONPORTABLE_LOCAL_PATHS_OMITTED_TRUE_ONLY"
    ),
    "counts": {
        "referenced_basenames": 84,
        "basename_present": 75,
        "basename_absent": 9,
        "retained_artifact_present_and_not_known_wrong_bytes": 74,
        "prior_hash_bound_expected_artifact_present": 69,
        "prior_hash_bound_expected_artifact_absent_or_wrong_bytes": 6,
        "no_prior_cryptographic_report_binder": 9,
        "present_without_prior_cryptographic_report_binder": 5,
        "absent_without_prior_cryptographic_report_binder": 4,
        "present_with_raw_timing_sample_arrays": 24,
        "present_without_raw_timing_sample_arrays": 51,
        "prior_hash_bound_expected_artifacts_with_raw_timing_sample_arrays": 21,
    },
}

RECOVERED_RC13_SDIST: Mapping[str, Any] = {
    "release": "0.6.0rc13",
    "artifact": {
        "filename": "jxplanetx-0.6.0rc13.tar.gz",
        "bytes": 766889,
        "sha256": (
            "9285a3ab7136ba76ff41d0e0b001462e626cb6e6499fac7237845a4996a1b555"
        ),
        "custody": "LOCAL_RETAINED_CHECKSUM_BOUND_RELEASE_SOURCE_ARCHIVE",
        "archive_bytes_in_git": False,
    },
    "binding_manifest": {
        "path": "RELEASE_MANIFEST_v0.6.0rc13.json",
        "bytes": 6422,
        "sha256": (
            "8547485e410b880910700c5b7e1563fbca06e32e57eb06c623e35e8b2c55dd95"
        ),
        "git_blob": "06fe376aa7c280a34c48f4b2edadf111e5ea7e9a",
        "base_repository_tracked": True,
    },
    "members": (
        {
            "path": "jxplanetx-0.6.0rc13/src/jxplanetx/gr15_eih_1pn.py",
            "bytes": 16469,
            "sha256": (
                "aa694614e55feb827db5075c46d1489bcdca580dad7d0843957de948d2dddc85"
            ),
            "role": "HISTORICAL_PROTOCOL_BOUND_CPU_EIH_WRAPPER",
        },
        {
            "path": "jxplanetx-0.6.0rc13/src/jxplanetx/gr15_eih_1pn_cuda.py",
            "bytes": 60448,
            "sha256": (
                "7765a3ed6e628b63031841feab91e05e21edb190f57290eb8089d1ff1b1d3f53"
            ),
            "role": "HISTORICAL_PROTOCOL_BOUND_CUDA_EIH_WRAPPER",
        },
    ),
    "claim_limit": (
        "RECOVERS_EXACT_HISTORICAL_WRAPPER_SOURCE_BYTES; DOES_NOT_RECOVER_THE_"
        "RAW_TIMING_REPORT_INSIDE_THIS_SOURCE_ARCHIVE_OR_ESTABLISH_MODEL_"
        "CONCORDANCE; SEPARATELY_BOUND_EXTERNAL_REPORTS MAY SURVIVE"
    ),
}

RECOVERED_P8_ACCURACY_BUNDLE: Mapping[str, Any] = {
    "artifact": {
        "path": "runs/jx_gpu_rebound_global_symplectic_bridge_100yr_v1.tar.gz",
        "bytes": 1345188,
        "sha256": (
            "45123f7b2ec860d6b50d00c84e09b89e9a4c04618b1194d3148dab93342a9155"
        ),
        "custody": "TWO_HASH_IDENTICAL_RECOVERED_RESEARCH_SNAPSHOT_COPIES",
        "embedded_root": "runs/jx_gpu_rebound_global_symplectic_bridge_100yr_v1",
        "member_count": 42,
        "regular_file_count": 29,
        "directory_count": 13,
    },
    "members": (
        {
            "path": (
                "runs/jx_gpu_rebound_global_symplectic_bridge_100yr_v1/"
                "PROTOCOL.json"
            ),
            "bytes": 3069,
            "sha256": (
                "db16798e0e0d06378ee1e22be787ba8e74f2afa4bef8e810ed81579e458ab693"
            ),
            "role": "FROZEN_HISTORICAL_ACCURACY_PROTOCOL",
        },
        {
            "path": (
                "runs/jx_gpu_rebound_global_symplectic_bridge_100yr_v1/"
                "evidence/normal/report.json"
            ),
            "bytes": 11715,
            "sha256": (
                "2b85c6d8c6e0ddb4ba47bc4825a1b8ef3642521d311ac5dc387d96a61eff6dfd"
            ),
            "role": "NORMAL_MODE_ACCURACY_REPORT",
        },
        {
            "path": (
                "runs/jx_gpu_rebound_global_symplectic_bridge_100yr_v1/"
                "evidence/optimized/report.json"
            ),
            "bytes": 11715,
            "sha256": (
                "2b85c6d8c6e0ddb4ba47bc4825a1b8ef3642521d311ac5dc387d96a61eff6dfd"
            ),
            "role": "OPTIMIZED_MODE_BYTE_IDENTICAL_ACCURACY_REPORT",
        },
        {
            "path": (
                "runs/jx_gpu_rebound_global_symplectic_bridge_100yr_v1/"
                "SUMMARY.json"
            ),
            "bytes": 1597,
            "sha256": (
                "a29c36fd9e8dc6c25baa934f84672000a237c173fb7c209aa23d2351208d4f02"
            ),
            "role": "PORTABLE_ACCURACY_SUMMARY",
        },
        {
            "path": (
                "runs/jx_gpu_rebound_global_symplectic_bridge_100yr_v1/"
                "SHA256SUMS"
            ),
            "bytes": 2996,
            "sha256": (
                "7ebdaabbeb990d79fa5965947e3c1e3ed38a824ef5f762ea86b916a23f063e96"
            ),
            "role": "INTERNAL_FILE_ROSTER_RECEIPT",
        },
        {
            "path": (
                "runs/jx_gpu_rebound_global_symplectic_bridge_100yr_v1/"
                "inputs/ELEVEN_BODY_INPUT.json"
            ),
            "bytes": 4647,
            "sha256": (
                "0718ba6c05444c720d012bc7906e3eee226664b58d7929648917f9f19003d4c5"
            ),
            "role": "FROZEN_ELEVEN_BODY_INPUT",
        },
        {
            "path": (
                "runs/jx_gpu_rebound_global_symplectic_bridge_100yr_v1/"
                "EXECUTION_RECORD.json"
            ),
            "bytes": 1045,
            "sha256": (
                "2249b221a02938b3bf745212437e725d5c2e586a94d6d7e53b74a858bb183edf"
            ),
            "role": "HISTORICAL_EXECUTION_RECORD",
        },
        {
            "path": (
                "runs/jx_gpu_rebound_global_symplectic_bridge_100yr_v1/verify.py"
            ),
            "bytes": 3428,
            "sha256": (
                "932d86aca5f2f19f1a2d09d426ad27a145196a903a683f942f04a7ba70aa49aa"
            ),
            "role": "HISTORICAL_OFFLINE_VERIFIER",
        },
    ),
    "sha256sums": {
        "path": (
            "runs/jx_gpu_rebound_global_symplectic_bridge_100yr_v1/SHA256SUMS"
        ),
        "bytes": 2996,
        "sha256": (
            "7ebdaabbeb990d79fa5965947e3c1e3ed38a824ef5f762ea86b916a23f063e96"
        ),
        "covered_regular_files": 28,
    },
    "result_status": "PASS_INDEPENDENT_GLOBAL_SYMPLECTIC_100_YEAR_REPRODUCTION",
    "calibration_disclosure": (
        "CONFIGURATION_SELECTED_AFTER_EXPLORATORY_SAME_STATE_RUN; EXPLORATORY_"
        "TRANSCRIPT_NOT_RETAINED; CONFIRMATION_THRESHOLDS_ARE_NOT_BLIND_"
        "PREDICTIONS_OR_INDEPENDENT_PHYSICS_VALIDATION"
    ),
    "claim_limit": (
        "HISTORICAL_REBOUND_5_1_1_SAME_MODEL_CONFIRMATION_AND_REPRODUCTION_"
        "SCREEN_ONLY; NO_TIMING_OR_SPEED_CLAIM; NOT_PREREGISTERED_INDEPENDENT_"
        "PHYSICS_VALIDATION; NOT_CURRENT_REBOUND_5_2_2_PROOF"
    ),
}

# Relative to the retained backup's ``tmp-experiments`` directory.  These are
# selected direct-performance, comparator-accuracy, and outcome-blind scientific
# artifacts for this portfolio, not every JSON file in the broader backup.
RECOVERED_HISTORICAL_REPORTS: tuple[
    tuple[str, int, str, str, str], ...
] = (
    (
        "jx_fused_rebound_matrix_rc4_v8_rtx5060ti.json",
        25232,
        "5bb1fc6566d9d7dd5b76279da414f4331e321b3a4e0f5d7b6d988ca20f32fbc4",
        "RELEASE_MANIFEST_v0.6.0rc4.json",
        "DIRECT_REBOUND_5_1_1_SUMMARY_STATISTICS_NO_RAW_TIMING_SAMPLE_ARRAYS",
    ),
    (
        "jx_fused_rebound_matrix_rc4_v8_rtx4050.json",
        25241,
        "34b3b7d98bc3c5ef564d70ad1618618ff8e86b4bca373cdea7cd165d89444c40",
        "RELEASE_MANIFEST_v0.6.0rc4.json",
        "DIRECT_REBOUND_5_1_1_SUMMARY_STATISTICS_NO_RAW_TIMING_SAMPLE_ARRAYS_"
        "COMMON_OWNERSHIP_SECOND_HOST",
    ),
    (
        "jx-fused-rebound-multimachine-rc4.json",
        4620,
        "14e0e1bd3df5f8b1c33716e0296f133c81e569cdf41ddc5b69217c353375883f",
        "RELEASE_MANIFEST_v0.6.0rc4.json",
        "AGGREGATE_COMMON_OWNERSHIP_NOT_INDEPENDENT_REPLICATION",
    ),
    (
        "jx_smalln_hybrid_matrix_rc5_final_v2.json",
        31557,
        "88ddce1554e8f7fbc0f8828cd457a01a6688ab1327e1775fdd0850681c1fa17c",
        "RELEASE_MANIFEST_v0.6.0rc5.json",
        "DIRECT_REBOUND_5_1_1_SUMMARY_STATISTICS_NO_RAW_TIMING_SAMPLE_ARRAYS",
    ),
    (
        "jx_fused_rebound_matrix_rc5_v8_rtx5060ti.json",
        25248,
        "e18ea8fec04410ab9584cec7000a5f77068fcd3958c96b6dda5d6a3d1a34c0dc",
        "RELEASE_MANIFEST_v0.6.0rc5.json",
        "DIRECT_REBOUND_5_1_1_SUMMARY_STATISTICS_NO_RAW_TIMING_SAMPLE_ARRAYS",
    ),
    (
        "jx_rebound_long_timing_race_rc5_v2.json",
        43660,
        "dbd7f46b0cecc95b5f81a69b57e201855fbe8165b3adb3e83b072d2ae2531eba",
        "RELEASE_MANIFEST_v0.6.0rc5.json",
        "DIRECT_REBOUND_5_1_1_RAW_SAMPLES_MANDATORY_LOSS_SENTINEL",
    ),
    (
        "jx_challenger_v2_rc5_full/challenger_v2.json",
        8232,
        "97bf6b1691d8ff1e7c6e2e17e4e1f797f2f00c60ea895a11368b1d38ec40291a",
        "RELEASE_MANIFEST_v0.6.0rc5.json",
        "VALIDATION_BUNDLE_TIMINGS_NOT_CROSS_STUDY_COMPARABLE",
    ),
    (
        "jx_challenger_v2_rc5_full/leapfrog.json",
        152773,
        "c59681e17b443dd011b46ce8d6ef2c527f92671187ecc410382ebb461fb27ea5",
        "RC5_RELEASE_MANIFEST_TO_CHALLENGER_BUNDLE_TO_CHILD_RAW_REPORT",
        "VALIDATION_CHILD_TIMINGS_NOT_CROSS_STUDY_COMPARABLE",
    ),
    (
        "jx_challenger_v2_rc5_full/whfast.json",
        223692,
        "a2de3be854c45b4c42e87bd5362aeed16041c953c0d0ce3175b171fbb7d74bdf",
        "RC5_RELEASE_MANIFEST_TO_CHALLENGER_BUNDLE_TO_CHILD_RAW_REPORT",
        "VALIDATION_CHILD_TIMINGS_NOT_CROSS_STUDY_COMPARABLE",
    ),
    (
        "jx_challenger_v2_rc5_full/hybrid.json",
        760875,
        "4f902480fc758018c0d1cdceb8dd68b4efa7b3ef9ca6a54104c3ace1540fa5d8",
        "RC5_RELEASE_MANIFEST_TO_CHALLENGER_BUNDLE_TO_CHILD_RAW_REPORT",
        "VALIDATION_CHILD_TIMINGS_NOT_CROSS_STUDY_COMPARABLE",
    ),
    (
        "jx_fast_wh_rc6_rebound_race_v3.json",
        32423,
        "65503ce28e43582b064b1779b1aa2246fd691a9cea9c5a2b649c56b2c1354342",
        "RELEASE_MANIFEST_v0.6.0rc6.json",
        "DIRECT_REBOUND_5_1_1_RAW_SAMPLES",
    ),
    (
        "jx_fast_wh_rc7_rebound_race_v1.json",
        32178,
        "7dec5cd0d247abed56522d0c9e98f8b3d2068496ba7b88f10ab33f1dc2c16fad",
        "RELEASE_MANIFEST_v0.6.0rc7.json",
        "DIRECT_REBOUND_5_1_1_RAW_SAMPLES",
    ),
    (
        "jx_native_kepler_v4_rebound_race_v2.json",
        39047,
        "a6c64b6d1d7ed45791a30f0d016818bd89050731f840d12864dc44198a032a3a",
        "RELEASE_MANIFEST_v0.6.0rc7.json",
        "PRIVATE_DEVELOPMENT_REBOUND_5_1_1_RAW_SAMPLES",
    ),
    (
        "jx_fast_wh_rc8_rebound_race_v3.json",
        32544,
        "5829202be78fb17f819010f31cafec44bebfcec8b5a465b90594325e906bfc28",
        "RELEASE_MANIFEST_v0.6.0rc8.json",
        "DIRECT_REBOUND_5_1_1_RAW_SAMPLES",
    ),
    (
        "jx_gr15_rc9_acceptance_v1.json",
        35881,
        "005ab24e9e0189c233c48a13874c4064f8087a08c6a20ae6490d81d50d63076a",
        "RELEASE_MANIFEST_v0.6.0rc9.json",
        "DIRECT_REBOUND_5_1_1_RAW_SAMPLES",
    ),
    (
        "jx_gr15_rc9_acceptance_v1_repeat.json",
        35811,
        "2e94f3549d61fab325a4eee2cc4924a01912db95df2af1cb801507938deee510",
        "RELEASE_MANIFEST_v0.6.0rc9.json",
        "DIRECT_REBOUND_5_1_1_RAW_SAMPLES_REPEAT",
    ),
    (
        "jx_gr15_rc10_acceptance_v1.json",
        35989,
        "e61b8f970f7fe1b0f676096895cde00a0e38ee3d8568b7fbc5ab2a89294404a3",
        "RELEASE_MANIFEST_v0.6.0rc10.json",
        "DIRECT_REBOUND_5_1_1_RAW_SAMPLES",
    ),
    (
        "jx_gr15_rc10_acceptance_repeat.json",
        35873,
        "531b7b10d3f43240fd7a9252637ca14c531ad2e13f699a2b1cda4e5c0c707566",
        "RELEASE_MANIFEST_v0.6.0rc10.json",
        "DIRECT_REBOUND_5_1_1_RAW_SAMPLES_REPEAT",
    ),
    (
        "jx_gr15_rc10_acceptance_rtx4050.json",
        35910,
        "6b66a14d8b6d8c8fafc662ae1801b1048e94836b4a89992cf6b2a4196dd051a2",
        "RELEASE_MANIFEST_v0.6.0rc10.json",
        "DIRECT_REBOUND_5_1_1_RAW_SAMPLES_SECOND_HOST_COMMON_OWNERSHIP",
    ),
    (
        "jx_gr15_eih_1pn_de440_attribution_rc11.json",
        13581,
        "e2d7160eb3fecd73f34947d6430d6e33ebbd951f8b69080b5fa4fa852abc378d",
        "RELEASE_MANIFEST_v0.6.0rc11.json",
        "SCIENTIFIC_EIH_DE440_ATTRIBUTION_NOT_REBOUND_SPEED_EVIDENCE",
    ),
    (
        "jx_eih_1pn_cuda_scale_rc12.json",
        4895,
        "ebe5016631367bdd1dfa9922c7bf0e0013b1dff879c8b08052f5d86220c0bfdd",
        "RELEASE_MANIFEST_v0.6.0rc12.json",
        "JX_CPU_CUDA_INTERNAL_SCALE_NOT_REBOUND_SPEED_EVIDENCE",
    ),
    (
        "jx_gr15_eih_1pn_cuda_scale_rc13.json",
        3763,
        "92f359924ff8aa054a8393edb6bcd2b04beeabbb52f1f87fc5bd4626b9e70449",
        "RELEASE_MANIFEST_v0.6.0rc13.json",
        "JX_CPU_CUDA_INTERNAL_SCALE_NOT_REBOUND_SPEED_EVIDENCE",
    ),
    (
        "jx_gr15_dispatch_rc14_acceptance.json",
        8434,
        "3a06d2c330ece0f2a61df126e423f859defb66dba83eaae382975248f6f4e125",
        "RELEASE_MANIFEST_v0.6.0rc14.json",
        "JX_CPU_CUDA_INTERNAL_DISPATCH_NOT_REBOUND_SPEED_EVIDENCE",
    ),
    (
        "jx_gr15_ten_year_dispatch_crossover_rc13.json",
        68342,
        "9024fc27ce9c5a72fc6c689706b823d1d6861bd4cda7d9584f1448326753376d",
        "NEW_PORTFOLIO_CHARTER_BINDING_TO_PROTOCOL_F6DBAF9D",
        "DIRECT_REBOUND_5_1_1_REBOUNDX_5_1_0_RAW_SAMPLES_MODEL_CONCORDANCE_"
        "UNESTABLISHED",
    ),
    (
        "jx_gr15_cuda_rebound_ias15_rc13.json",
        53584,
        "030642a811b314fd058583d5b0fe4f5f5c3905c8ae6434fc0443ee0ed15f6212",
        "NEW_PORTFOLIO_CHARTER_BINDING_TO_PROTOCOL_9470617D",
        "DIRECT_REBOUND_5_1_1_REBOUNDX_5_1_0_RAW_SAMPLES_MODEL_CONCORDANCE_"
        "UNESTABLISHED",
    ),
    (
        "jx_native_reversible_hybrid_loop_timing_v1.json",
        22333,
        "f1557ca4778ebd8aeec5477cd9d21f5d2b608089324bdb4c070e9d60293d3a6c",
        "BOUND_DIRTY_RECOVERED_RESEARCH_LEDGER",
        "PRIVATE_REBOUND_5_1_1_RAW_SAMPLES_SCOPE_DIFFERS",
    ),
    (
        "jx_lunar_translational_physics_prereg_v3.json",
        4826,
        "00e0fcae37009424a824b602e63c74f00e766ca972e03e36f2b92a73f059dca6",
        "BOUND_DIRTY_TRANSLATIONAL_EIH_LEDGER",
        "OUTCOME_BLIND_LUNAR_EIH_ATTRIBUTION_PREREGISTRATION_NOT_SPEED_EVIDENCE",
    ),
    (
        "jx_lunar_translational_physics_prefix_v3.json",
        13062,
        "2a60cc5729e61b10a5e2cadc06ba26a75b2687b2fdf021d8d4d4b41a8d93b4f5",
        "BOUND_DIRTY_TRANSLATIONAL_EIH_LEDGER",
        "CPU_CUDA_LUNAR_EIH_ATTRIBUTION_PREFIX_GATE_NOT_REBOUND_SPEED_EVIDENCE",
    ),
    (
        "jx_lunar_translational_physics_holdout_v1.json",
        20432,
        "57bcdd78d0fc582c21ad9831896f629e50f8084b81d9bd69b94e7eb9d0893f3b",
        "BOUND_DIRTY_TRANSLATIONAL_EIH_LEDGER",
        "SCIENCE_SCREENING_EIH_ATTRIBUTION_HOLDOUT_NOT_REBOUND_SPEED_EVIDENCE",
    ),
    (
        "jx_lunar_eih_confirmation_prereg_v1.json",
        4659,
        "b0328a8d9397cad82a1deef92429b091bac19532663653828971ac16ba269fb4",
        "BOUND_DIRTY_TRANSLATIONAL_EIH_LEDGER",
        "OUTCOME_BLIND_EIH_CONFIRMATION_PREREGISTRATION_NOT_SPEED_EVIDENCE",
    ),
    (
        "jx_lunar_eih_confirmation_prefix_v1.json",
        25410,
        "ebbb8080712f978f12d1240f45771637ed0e25768e2f45246f9ff8720ab06297",
        "BOUND_DIRTY_TRANSLATIONAL_EIH_LEDGER",
        "CPU_CUDA_EIH_CONFIRMATION_PREFIX_GATE_NOT_REBOUND_SPEED_EVIDENCE",
    ),
    (
        "jx_lunar_eih_confirmation_v1.json",
        34167,
        "aeef5ee710e0874a90aa4330cb121c8b7b2b0898af35fd8c10638eb1d6f711f0",
        "BOUND_DIRTY_TRANSLATIONAL_EIH_LEDGER",
        "SCIENCE_SCREENING_EIH_FRESH_START_CONFIRMATION_NOT_REBOUND_SPEED_EVIDENCE",
    ),
    (
        "jx_gr15_de440_long_validation_report.json",
        13849,
        "cfe7c99bb6c3683651325cc70623de76244051ff6c361ec61e51763677d2729c",
        "NEW_CHARTER_BINDING_LEDGER_NAME_AND_SEMANTIC_DIGEST_ONLY_BEFORE_CHARTER",
        "GR15_REBOUND_5_1_1_100Y_ACCURACY_AND_ACCOUNTING_NOT_SPEED_EVIDENCE",
    ),
    (
        "jx_current_100y_same_model_report.json",
        2186,
        "6da87aaaf232b13a09158bfb82c0cf30493da5ea7b1637afcab8d8ae86dfad53",
        "NEW_CHARTER_BINDING_WITH_COMPANION_GENERATOR_SOURCE_NO_PRIOR_REPORT_RECEIPT",
        "REBOUND_5_1_1_REBOUNDX_5_1_0_SINGLE_TIMING_SCREENING_ONLY_NOT_SOLVER_"
        "SPEED_EVIDENCE_ASYMMETRIC_PYTHON_FORCE_CALLBACK_VS_FUSED_JX",
    ),
)


PUBLIC_EVIDENCE: tuple[tuple[str, str, str], ...] = (
    (
        "BASELINE_RECOVERY_20261013.json",
        "f55a4f894770e8e20facaa8a4434e38ab5d4c4974c369b347aa23fa034751d0a",
        "RECOVERY_SCOPE_AND_RETIRED_TEST_RECEIPT",
    ),
    (
        "RELEASE_MANIFEST_v0.6.0rc4.json",
        "72aa2caeca14ed44115ce9b0658e2503b1251806ee9db878eda585a2753a4476",
        "HISTORICAL_REBOUND_5_1_1_BATCH_PRIOR",
    ),
    (
        "RELEASE_MANIFEST_v0.6.0rc5.json",
        "80914c479dc5538672c6645023003050de9c4e96813a445b453002941dbb173c",
        "HISTORICAL_REBOUND_5_1_1_CPU_BATCH_AND_LONG_LOSS_PRIOR",
    ),
    (
        "RELEASE_MANIFEST_v0.6.0rc6.json",
        "f2044ff915119715c54a5fa36a74d1819b5d6911f0cc5e271981eb18441b5ad6",
        "HISTORICAL_REBOUND_5_1_1_EQUAL_OUTPUT_WH_PRIOR",
    ),
    (
        "RELEASE_MANIFEST_v0.6.0rc7.json",
        "caf49921748f5c246970226c7ab89d19c3f8877f3801add482819fbe85d56f94",
        "HISTORICAL_REBOUND_5_1_1_WH_AND_PRIVATE_V4_PRIOR",
    ),
    (
        "RELEASE_MANIFEST_v0.6.0rc8.json",
        "7dff70c4701e856f84fb6048552881061ecca2b373880297abb2cc463c8abdc5",
        "HISTORICAL_REBOUND_5_1_1_EQUAL_OUTPUT_WH_PRIOR",
    ),
    (
        "RELEASE_MANIFEST_v0.6.0rc9.json",
        "deffc5aa7c74624314b4d48d8209ea047e2dd8dbe487830ef8d78f3a0ce916f8",
        "HISTORICAL_REBOUND_5_1_1_GR15_PRIOR",
    ),
    (
        "RELEASE_MANIFEST_v0.6.0rc10.json",
        "2048b4d59332d903678444397ae0de78743500e91bcf701e7441c9a022fd15e9",
        "HISTORICAL_REBOUND_5_1_1_GR15_PRIOR",
    ),
    (
        "RELEASE_MANIFEST_v0.6.0rc11.json",
        "bd6a2a871be899d6d76687eff2af3403145f5388cd1c3325181ea700b084aebe",
        "HISTORICAL_EIH_DE440_ATTRIBUTION_NOT_SPEED_EVIDENCE",
    ),
    (
        "RELEASE_MANIFEST_v0.6.0rc12.json",
        "c12e05e94702a7f29b14c0ebc1e585c24fb389c7e7895273fd3c5b1319641e79",
        "HISTORICAL_JX_EIH_CPU_CUDA_SCALE_NOT_REBOUND_SPEED_EVIDENCE",
    ),
    (
        "RELEASE_MANIFEST_v0.6.0rc13.json",
        "8547485e410b880910700c5b7e1563fbca06e32e57eb06c623e35e8b2c55dd95",
        "HISTORICAL_RELEASE_MANIFEST_BINDS_RECOVERED_EIH_SOURCE_ARCHIVE",
    ),
    (
        "RELEASE_MANIFEST_v0.6.0rc14.json",
        "9f78b4b529fbc6f5837cae05ddcd65ffd3333f08bba2df082ed4e165033a5f59",
        "HISTORICAL_RELEASE_MANIFEST_BINDS_INTERNAL_JX_DISPATCH_TIMING_ONLY",
    ),
    (
        "docs/JX_REBOUND_COMPETITION_RESEARCH_2026-09-22.md",
        "bad96fec6ccdd0748471c6de9fa1cd6c275332d22cb1826455da7311e52b8b9f",
        "RESEARCH_LEDGER_NOT_CURRENT_PROOF",
    ),
    (
        "docs/JX_REBOUND_COMPETITIVE_RELEASE_TRACK.md",
        "74bd2f5a647dafaa1f80ac87d2a48d31bc5e46ddc984e22e9f597ec1d725cdd1",
        "RELEASE_TRACK_NOT_PORTABLE_PROOF",
    ),
    (
        "results/performance_20260922/jx_raw_c_trace_baseline_v1.json",
        "1aa18c1f7faf110be2624ee2cf6e1220816839d9174e0cc38c2dd2e0ca0a5b27",
        "FULL_SAMPLES_PRESENT_NOT_EQUAL_ACCURACY",
    ),
    (
        "results/performance_20260922/jx_raw_c_trace_stateful_near_v2.json",
        "0cab0fb03693f737b8f73b1e7772e1759e0f5aae03defccfd3ff13aefea3e538",
        "FULL_SAMPLES_PRESENT_NOT_EQUAL_ACCURACY",
    ),
    (
        "runs/de441_population_100k/final_result_v1.json",
        "24b7572cf130c683acd66a4677dac62d0d30b15b24f9bf5997b612a8045d7efd",
        "REBOUND_4_4_11_SCIENCE_SCREEN_NOT_SPEED_EVIDENCE",
    ),
)


RECOVERED_RESEARCH: tuple[tuple[str, str, str], ...] = (
    (
        "docs/JX_REBOUND_COMPETITION_RESEARCH_2026-09-22.md",
        "a64fbbf18a83df837848b06e1dce54c7f5fe2d90c3e1f3725cebd3f379525cec",
        "DIRTY_RECOVERED_LEDGER_HISTORICAL_DISCLOSURE_NOT_CURRENT_PROOF",
    ),
    (
        "benchmarks/jx_fused_accuracy_matched_rebound.py",
        "1ea4272d239c6f473f9e3fbd7a567c7b7bd9d6453997a1a0e0b0c68bca9128a6",
        "RECONSTRUCTION_INPUT_ONLY",
    ),
    (
        "benchmarks/jx_rebound_long_timing_race.py",
        "f614906bf2f98ac595b2422f240d1c62604a6be88f8f4caacf7eb19cc44cbc7d",
        "RECONSTRUCTION_INPUT_ONLY",
    ),
    (
        "benchmarks/jx_native_kepler_v8_rebound_race.py",
        "bca6610cbce5d4c4e0b67acd22cb2f2263db286ed58c8189f1f1808364e49103",
        "PRIVATE_CANDIDATE_REQUIRES_PROMOTION",
    ),
    (
        "benchmarks/jx_native_kepler_v4_rebound_race.py",
        "f135be0c73dd9cbe1f73537d71f434f2b4260b235ebeb17718ac838f6d649bfc",
        "WH_V8_DEPENDENCY_CLOSURE",
    ),
    (
        "benchmarks/rebound_whfast_comparison.py",
        "2a3b233eeda1ec3b99e4bee6e999ead6753c9d51e996f2282919a0057ff4123d",
        "WH_V8_DEPENDENCY_CLOSURE",
    ),
    (
        "benchmarks/jx_native_kepler_v8_preregistration.json",
        "c0073cb9d7bcf8c1f3a3b1c0ab42ed99e361dbcc42453888ebf45d2424aba451",
        "PRIVATE_PREREGISTRATION_HISTORY",
    ),
    (
        "benchmarks/jx_native_kepler_v8_rebound_race_protocol.json",
        "5535cec62b665269645542221c0599a73a8e37668316c2712903381928a572ec",
        "PRIVATE_PROTOCOL_HISTORY",
    ),
    (
        "src/jxplanetx/_wisdom_holman_loop_v8.c",
        "6fe8b4fcda8947bc21aacce56d9e3383a1d5d579176c742a994628967927b004",
        "PRIVATE_CANDIDATE_REQUIRES_PROMOTION",
    ),
    (
        "src/jxplanetx/fast_wisdom_holman.py",
        "99da08305f5c0a191dd53e64d02a9a396b471aaf9f7b3a062545b1de220f7a72",
        "WH_V8_DEPENDENCY_CLOSURE",
    ),
    (
        "src/jxplanetx/_wisdom_holman_loop_v2.c",
        "38b8898bc6c1358522a0f653fc55849719e955889f556fd9f5850ad199346750",
        "WH_V8_DEPENDENCY_CLOSURE",
    ),
    (
        "src/jxplanetx/_wisdom_holman_loop_v3.c",
        "7143fe32304b04901d107b48bf8fe5ae9043118b81c25c635e5d4b412c93b629",
        "WH_V8_DEPENDENCY_CLOSURE",
    ),
    (
        "src/jxplanetx/_wisdom_holman_loop_v4.c",
        "10981b416e4bb90a89babfc05cdd9c97bf1e5b49a0956490300bc5cbc591a16c",
        "WH_V8_DEPENDENCY_CLOSURE",
    ),
    (
        "src/jxplanetx/_wisdom_holman_loop_v6.c",
        "c1da285ec602185f093c7f702097650f009827285332e2aeb1c5f06aa848590d",
        "WH_V8_DEPENDENCY_CLOSURE",
    ),
    (
        "src/jxplanetx/_wisdom_holman_loop_v7.c",
        "337c92c1ea4aba54f64e8e336889a02dccb23d2bd6c275dedfac39e73fd9c0bb",
        "WH_V8_DEPENDENCY_CLOSURE",
    ),
    (
        "src/jxplanetx/_wisdom_holman_cpu_v7_module.c",
        "dba08b8c899f4039e5e24fbb897f23c57c99da00deb9e084960fb910cd96b94c",
        "WH_V8_DEPENDENCY_CLOSURE",
    ),
    (
        "src/jxplanetx/_wisdom_holman_cpu_v8_module.c",
        "fcf77481d241902f2c05eabaaac9bfa89f208e645d9ae760baf67dcd03e9d288",
        "PRIVATE_CANDIDATE_REQUIRES_PROMOTION",
    ),
    (
        "benchmarks/jx_gr15_v6_result.json",
        "5a5d3d245cf179d941d9185023b9f81ff74dd44017b52a6b3f976561129f1ef6",
        "PRIVATE_COMPACT_RESULT_NOT_CURRENT_PROOF",
    ),
    (
        "benchmarks/jx_gr15_v6_preregistration.json",
        "c3ccfb8490b42c09439d1f086341b434455b0a4e72d7e5ef571547dae363615d",
        "PRIVATE_PREREGISTRATION_HISTORY",
    ),
    (
        "benchmarks/jx_gr15_v6_ias15_race_protocol.json",
        "af4b43b2298dc65636f552303c18f4176462250aeb9516df7f8ce3371fb9712b",
        "PRIVATE_PROTOCOL_HISTORY",
    ),
    (
        "benchmarks/jx_gr15_v6_ias15_race.py",
        "b147b8d8e87dc3fdfc409193b98f806831d22a48c44a2e71902cde1ce6104806",
        "PRIVATE_CANDIDATE_REQUIRES_PROMOTION",
    ),
    (
        "benchmarks/jx_gr15_ias15.py",
        "fc945fac5ca10b6d89c45f85f9201cdf4518019c3f63b3af3997a0885c899590",
        "GR15_V6_DEPENDENCY_CLOSURE",
    ),
    (
        "src/jxplanetx/gr15.py",
        "80a8e34ceeffd5c714ec5a51564285bd255ac35868b8700faa9830fe2cb7738d",
        "GR15_V6_DEPENDENCY_CLOSURE",
    ),
    (
        "src/jxplanetx/_native_force_abi.py",
        "fe716d6eb4dfddd0d2f9cd5b0c5a605073173c9053ca93aa945baa97ef2e3fbb",
        "GR15_V6_DEPENDENCY_CLOSURE",
    ),
    (
        "src/jxplanetx/_native_force_abi.h",
        "93d3b8d93b500048cf7edc217be9905ea98b363bb0c98efffbb3638757339c30",
        "GR15_V6_DEPENDENCY_CLOSURE",
    ),
    (
        "src/jxplanetx/_gr15_v3_core.c",
        "036d1dc7e4e2e26c908c6dde9ff27bdf6a5804b88aa3f3e374f1f512642c1e41",
        "GR15_V6_DEPENDENCY_CLOSURE",
    ),
    (
        "src/jxplanetx/_gr15_v3_core.h",
        "65efaa1b16ff5e7fe2b9e8afe26f90e11f6e6ad8b75711e4fc2b1e3e3b3967f5",
        "GR15_V6_DEPENDENCY_CLOSURE",
    ),
    (
        "src/jxplanetx/_gr15_v3_cpu_module.c",
        "fb7b1fee96bd10c5ea837ebaac22c1ae32aae05af1cfca257da68441741efbe3",
        "GR15_V6_DEPENDENCY_CLOSURE",
    ),
    (
        "src/jxplanetx/_gr15_v5_candidate.py",
        "f4760e801f53a83aeefbf680044bac309c4f432703b12bb0e84cc907667121b5",
        "GR15_V6_DEPENDENCY_CLOSURE",
    ),
    (
        "src/jxplanetx/_gr15_v5_core.c",
        "1b269b86771d262790e41a925eb15ae27a40cf27a184dccdbddfc00d16a6344c",
        "GR15_V6_DEPENDENCY_CLOSURE",
    ),
    (
        "src/jxplanetx/_gr15_v5_core.h",
        "fa9b8c36ad9ec7647e884b2cc565a9116f9b2c156a285d0aa44e54fced4e2ebb",
        "GR15_V6_DEPENDENCY_CLOSURE",
    ),
    (
        "src/jxplanetx/_gr15_v5_cpu_module.c",
        "5b702ebf08d91c669f7721a6461bd37de834db21c329921dc2a46f6e070322b1",
        "GR15_V6_DEPENDENCY_CLOSURE",
    ),
    (
        "src/jxplanetx/_gr15_v6_core.c",
        "f7bb50da78c76a3fb3d3f41d2245e4af8a06cdf6a62460867f4a15c27ae16bbf",
        "PRIVATE_TWO_BODY_SPECIALIZATION_REQUIRES_PROMOTION",
    ),
    (
        "src/jxplanetx/_gr15_v6_core.h",
        "f7aaf55f6d6ba4f8d4913d7a8d9b21bc9889f8656664ba5a29c5ff2692079183",
        "GR15_V6_DEPENDENCY_CLOSURE",
    ),
    (
        "src/jxplanetx/_gr15_v6_candidate.py",
        "44f97637903545ed53e8c392616eeabb12e787c9be6b3bb58cc922a4e9edf09f",
        "PRIVATE_TWO_BODY_SPECIALIZATION_REQUIRES_PROMOTION",
    ),
    (
        "src/jxplanetx/_gr15_v6_cpu_module.c",
        "608a869bc8365c40334568d285e2f55b2068ccb7c1a4acd717dd03a919a4be83",
        "PRIVATE_TWO_BODY_SPECIALIZATION_REQUIRES_PROMOTION",
    ),
    (
        "benchmarks/jx_gr15_ten_year_dispatch_crossover.py",
        "49b106a9b09f74e5794bcb5b0a197540f9861e1b4b3fa6fdc0dcd8f135436726",
        "EIH_BATCH_RECONSTRUCTION_INPUT_ONLY",
    ),
    (
        "benchmarks/jx_gr15_ten_year_dispatch_crossover_protocol.json",
        "f6dbaf9dd43cc82b405119e5b7d0a383508ef33708712e0e9048b0ec8d966196",
        "EIH_BATCH_HISTORICAL_PROTOCOL_WITH_MODEL_CAVEAT",
    ),
    (
        "benchmarks/jx_gr15_cuda_rebound_ias15.py",
        "7841c22868b5eca70a9e70e803d94acbc8184d4ae76b36b4af869e5da3dceae5",
        "EIH_BATCH_HISTORICAL_DEPENDENCY_EXACT",
    ),
    (
        "benchmarks/jx_gr15_cuda_rebound_ias15_protocol.json",
        "9470617ddb747cf9df2c4c29a63cf431158c6195724b3ee392aeb30f46feb317",
        "EIH_BATCH_HISTORICAL_DEPENDENCY_EXACT",
    ),
    (
        "src/jxplanetx/gr15_eih_1pn_cuda.py",
        "c214cd17045667e06f16533df3b5e95c664a348d587432daf1fac6887181d9d3",
        "LATER_EIH_WRAPPER_RECONSTRUCTION_NOT_HISTORICAL_PROTOCOL_BYTES",
    ),
    (
        "src/jxplanetx/gr15_eih_1pn.py",
        "2bc2bdb8676ec95700fbb719d934368d0156d402846c58dd5c252b9d5fbcb53d",
        "LATER_EIH_WRAPPER_RECONSTRUCTION_NOT_HISTORICAL_PROTOCOL_BYTES",
    ),
    (
        "src/jxplanetx/_gr15_eih_1pn_core.c",
        "bef3a607c4c3674d342745570c68b462a6178e789089c84419d6f23dd810fad1",
        "EIH_BATCH_HISTORICAL_DEPENDENCY_EXACT",
    ),
    (
        "src/jxplanetx/_eih_1pn_core.c",
        "de4b173ff26aab1a5d012aca79b2687c81ce8ef97db290e48b00ca3d21f40746",
        "EIH_BATCH_HISTORICAL_DEPENDENCY_EXACT",
    ),
    (
        "runs/jx_gpu_de440_external_epoch_validation_v1/inputs/"
        "SOURCE_ELEVEN_BODY_INPUT.json",
        "0718ba6c05444c720d012bc7906e3eee226664b58d7929648917f9f19003d4c5",
        "EIH_BATCH_HISTORICAL_INPUT_EXACT",
    ),
    (
        "benchmarks/jx_native_stateful_near_hybrid_prototype.py",
        "a10b2af3acd4f41463a10e046f613e98c2f88332a2d49266f86812be5c967658",
        "RECOVERED_GENERAL_HYBRID_PROTOTYPE",
    ),
    (
        "benchmarks/jx_native_endpoint_sync_hybrid_prototype.py",
        "af0b3dd57884a7f0ba2e218acc40524889f1e4eb4233239be722b900b2c62321",
        "RECOVERED_GENERAL_HYBRID_PROTOTYPE",
    ),
    (
        "benchmarks/jx_raw_c_trace_harness.c",
        "8f35e97d0216bbc5b064672bb872c1e068b6378b1ebda39175f1c20dd38fc4d8",
        "RECOVERED_TRACE_HARNESS",
    ),
    (
        "benchmarks/jx_raw_c_trace_stateful_near_harness.py",
        "2443f16c245ab44477c631d26386e314e970aa5309045bf8b3c260b8a5792f47",
        "RECOVERED_TRACE_HARNESS",
    ),
    (
        "benchmarks/jx_native_circular_binary_far_result.json",
        "d124adb29f87585115b7ebec74e1e236de87ee3940fc767ade1c0c439f252985",
        "PRIVATE_NARROW_RESULT_NOT_GENERAL_HYBRID_PROOF",
    ),
    (
        "benchmarks/jx_native_circular_binary_far_preregistration.json",
        "36d5c80a52e48e4b6f05e769aa2d5c58b5abdd1b2dd9bae3367f7fa1a3cfdd16",
        "PRIVATE_NARROW_PREREGISTRATION_HISTORY",
    ),
    (
        "benchmarks/jx_raw_c_trace_circular_binary_far_protocol.json",
        "fe1c3636f829659af33c9aff4d48404f085ee7e07229c113c8fd47c99f579fa4",
        "PRIVATE_NARROW_TRACE_PROTOCOL_HISTORY",
    ),
    (
        "benchmarks/jx_raw_c_mercurius_circular_binary_far_protocol.json",
        "f1f9ee020fda59fb2b5d4e75222e753716ddc2d032245a7950d8c9202d45c847",
        "PRIVATE_NARROW_MERCURIUS_PROTOCOL_HISTORY",
    ),
    (
        "benchmarks/jx_native_circular_binary_far.c",
        "3a0f30c7636afcad460b081be054f2e2a36cb483bb572bc900a569d5a1b71320",
        "PRIVATE_PROOF_GATED_SPECIALIZATION",
    ),
    (
        "docs/JX_LUNAR_TRANSLATIONAL_PHYSICS_NEXT_GATE.md",
        "dd2577aae16639ba5198e7efa3b003b2e6ca5fadb16b9e5e273de019f52a105e",
        "DIRTY_RECOVERED_OUTCOME_BLIND_TRANSLATIONAL_EIH_LEDGER",
    ),
    (
        "docs/JX_GR15_DE440_LONG_VALIDATION.md",
        "4ccbd55c5a103d150b25516326c507691bcdc275efeecf0ecdcb5a8eca84e736",
        "DIRTY_RECOVERED_GR15_IAS15_LONG_ACCURACY_LEDGER",
    ),
    (
        "runs/jx_gpu_rebound_global_symplectic_bridge_100yr_v1.tar.gz",
        "45123f7b2ec860d6b50d00c84e09b89e9a4c04618b1194d3148dab93342a9155",
        "P8_HISTORICAL_ACCURACY_BUNDLE_NO_SPEED_CLAIM",
    ),
)


TRACER_PRIORS: tuple[Mapping[str, Any], ...] = (
    {
        "id": "jx_cuda_massless_tracer_acceptance_v3",
        "result_sha256": (
            "88d1337048ceceeddee2d9280c4e51a5dab57714694a1771de91e77709bae9a9"
        ),
        "protocol_sha256": (
            "0beda4fdfb98394dd7d0fc505b46da94b01ea8adcbaac9769251294ca1b3e75b"
        ),
        "initial_state_sha256": (
            "ebb4d6b8952e78a6eb91e9591c2ddff36831f3f92b98952a9f50da802e82f048"
        ),
        "endpoint_state_sha256": (
            "1159bb9a15c15a87fa7345b1bffef716c9545e116e6491ea479031de29694920"
        ),
        "routes_sha256": (
            "6b35f6703a78c85b05dc030cd17990f8fc0ef74be0e5ee1467a70f467c85c2da"
        ),
        "tracers": 16800,
        "routed_steps": 522,
        "quiet_and_routed_subsets_not_yet_frozen": True,
        "steps": 10000,
        "years": 5000,
        "historical_tracer_steps_per_second": 32163291.197485007,
        "use": "JX_ENGINEERING_PRIOR_NO_REBOUND_COMPARISON",
    },
    {
        "id": "de441_population_100k",
        "result_sha256": (
            "24b7572cf130c683acd66a4677dac62d0d30b15b24f9bf5997b612a8045d7efd"
        ),
        "population_payload_sha256": (
            "2e4def4aa9fbbfaf451dc563480d7010c3afd8e39b7ffeceb91a4f6a055442d1"
        ),
        "population_manifest_sha256": (
            "c3dd5528f382c4fd0ff71921d6d5be635faf71ca82e1d73ddd268fcbe62e8381"
        ),
        "state_manifest_sha256": (
            "ec6e2ba8877d206838172dbb573318a2122977e263c054994d4c82a409a43d36"
        ),
        "control_state_sha256": (
            "edfbfd50ca29b28b46fe43cae8cb99bd73a9303a5f2e5babf7bf9ad2a820ff14"
        ),
        "source_state_sha256": (
            "81a8cf50f2ce6d17e90369efcaa82cf82a0955665851fae0207e4c6cfae4b6cf"
        ),
        "contract_sha256": (
            "1f3ccf6f29d4ec9890b9d66d19858c148760b00167281ce386afdfd0c8801725"
        ),
        "tracers_per_arm": 100000,
        "years": 10000,
        "comparator_version": "4.4.11",
        "use": "WORKLOAD_DESIGN_ONLY_NOT_CURRENT_SPEED_PROOF",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_stream(handle: Any) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def audit_recovered_rc13_sdist(archive_path: Path) -> dict[str, Any]:
    """Verify the retained rc13 source archive and its two EIH members.

    The caller supplies a local locator, but the returned portable receipt never
    records it.  One no-follow file descriptor is held throughout both archive
    passes so pathname replacement cannot change the audited inode.
    """

    archive_path = Path(archive_path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise ValueError("platform does not provide O_NOFOLLOW")
    try:
        descriptor = os.open(archive_path, flags | nofollow)
    except OSError as exc:
        raise ValueError("recovered rc13 sdist cannot be opened safely") from exc
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            before = os.fstat(handle.fileno())
            expected_artifact = RECOVERED_RC13_SDIST["artifact"]
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("recovered rc13 sdist is not a regular file")
            if before.st_nlink != 1:
                raise ValueError("recovered rc13 sdist link count differs")
            if before.st_size != expected_artifact["bytes"]:
                raise ValueError("recovered rc13 sdist byte length differs")
            first_digest = _sha256_stream(handle)
            if first_digest != expected_artifact["sha256"]:
                raise ValueError("recovered rc13 sdist hash differs")

            handle.seek(0)
            with tarfile.open(fileobj=handle, mode="r:gz") as archive:
                all_members = archive.getmembers()
                all_member_names = [member.name for member in all_members]
                if len(all_member_names) != len(set(all_member_names)):
                    raise ValueError("recovered rc13 sdist has duplicate member names")
                expected_members = RECOVERED_RC13_SDIST["members"]
                observed: list[dict[str, Any]] = []
                for expected in expected_members:
                    matches = [
                        member
                        for member in all_members
                        if member.name == expected["path"]
                    ]
                    if len(matches) != 1:
                        raise ValueError("recovered rc13 sdist member roster differs")
                    member = matches[0]
                    if not member.isfile() or member.size != expected["bytes"]:
                        raise ValueError("recovered rc13 sdist member shape differs")
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ValueError("recovered rc13 sdist member cannot be read")
                    with extracted:
                        digest = _sha256_stream(extracted)
                    if digest != expected["sha256"]:
                        raise ValueError("recovered rc13 sdist member hash differs")
                    observed.append(
                        {
                            "path": expected["path"],
                            "bytes": expected["bytes"],
                            "sha256": digest,
                            "role": expected["role"],
                        }
                    )

            handle.seek(0)
            second_digest = _sha256_stream(handle)
            after = os.fstat(handle.fileno())
            stable_identity = (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
            ) == (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
            )
            if not stable_identity or second_digest != first_digest:
                raise ValueError("recovered rc13 sdist changed during audit")
    except tarfile.TarError as exc:
        raise ValueError("recovered rc13 sdist cannot be parsed") from exc

    return {
        "status": "RECOVERED_RC13_EIH_SOURCE_ARCHIVE_MATCHES",
        "artifact": dict(RECOVERED_RC13_SDIST["artifact"]),
        "members": observed,
        "absolute_path_omitted": True,
        "raw_timing_report_embedded_in_this_source_archive": False,
        "model_concordance_established": False,
    }


def audit_recovered_p8_accuracy_bundle(archive_path: Path) -> dict[str, Any]:
    """Verify the retained P8 archive through one held, no-follow descriptor."""

    expected = RECOVERED_P8_ACCURACY_BUNDLE
    expected_artifact = expected["artifact"]
    archive_path = Path(archive_path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise ValueError("platform does not provide O_NOFOLLOW")
    try:
        descriptor = os.open(archive_path, flags | nofollow)
    except OSError as exc:
        raise ValueError("recovered P8 bundle cannot be opened safely") from exc

    observed_key_members: list[dict[str, Any]] = []
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("recovered P8 bundle is not a regular file")
            if before.st_size != expected_artifact["bytes"]:
                raise ValueError("recovered P8 bundle byte length differs")
            first_digest = _sha256_stream(handle)
            if first_digest != expected_artifact["sha256"]:
                raise ValueError("recovered P8 bundle hash differs")

            handle.seek(0)
            with tarfile.open(fileobj=handle, mode="r:gz") as archive:
                members = archive.getmembers()
                names = [member.name for member in members]
                if len(names) != len(set(names)):
                    raise ValueError("recovered P8 bundle has duplicate member names")
                root_name = expected_artifact["embedded_root"]
                for member in members:
                    portable = PurePosixPath(member.name)
                    if (
                        portable.is_absolute()
                        or str(portable) != member.name
                        or any(part in {"", ".", ".."} for part in portable.parts)
                        or not (
                            member.name == root_name
                            or member.name.startswith(root_name + "/")
                        )
                    ):
                        raise ValueError("recovered P8 bundle has unsafe member path")
                    if not (member.isfile() or member.isdir()):
                        raise ValueError("recovered P8 bundle has unsupported member type")
                regular = [member for member in members if member.isfile()]
                directories = [member for member in members if member.isdir()]
                if (
                    len(members) != expected_artifact["member_count"]
                    or len(regular) != expected_artifact["regular_file_count"]
                    or len(directories) != expected_artifact["directory_count"]
                ):
                    raise ValueError("recovered P8 bundle member counts differ")
                member_by_name = {member.name: member for member in members}

                key_payloads: dict[str, bytes] = {}
                for item in expected["members"]:
                    member = member_by_name.get(item["path"])
                    if (
                        member is None
                        or not member.isfile()
                        or member.size != item["bytes"]
                    ):
                        raise ValueError("recovered P8 bundle key member shape differs")
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ValueError("recovered P8 bundle key member cannot be read")
                    with extracted:
                        payload = extracted.read(item["bytes"] + 1)
                    if len(payload) != item["bytes"]:
                        raise ValueError("recovered P8 bundle key member length differs")
                    digest = hashlib.sha256(payload).hexdigest()
                    if digest != item["sha256"]:
                        raise ValueError("recovered P8 bundle key member hash differs")
                    key_payloads[item["path"]] = payload
                    observed_key_members.append(
                        {
                            "path": item["path"],
                            "bytes": item["bytes"],
                            "sha256": digest,
                            "role": item["role"],
                        }
                    )

                json_suffixes = (
                    "PROTOCOL.json",
                    "SUMMARY.json",
                    "EXECUTION_RECORD.json",
                    "ELEVEN_BODY_INPUT.json",
                    "evidence/normal/report.json",
                    "evidence/optimized/report.json",
                )
                parsed_key_json: dict[str, dict[str, Any]] = {}
                for name, payload in key_payloads.items():
                    if name.endswith(json_suffixes):
                        try:
                            parsed = json.loads(
                                payload,
                                parse_constant=_reject_nonfinite_json_constant,
                                object_pairs_hook=_reject_duplicate_json_pairs,
                            )
                        except (UnicodeDecodeError, ValueError) as exc:
                            raise ValueError(
                                "recovered P8 bundle key JSON is not strict"
                            ) from exc
                        if not isinstance(parsed, dict):
                            raise ValueError(
                                "recovered P8 bundle key JSON is not an object"
                            )
                        parsed_key_json[name] = parsed

                checksum_spec = expected["sha256sums"]
                checksum_payload = key_payloads.get(checksum_spec["path"])
                if checksum_payload is None:
                    raise ValueError("recovered P8 checksum ledger is not a key member")
                try:
                    checksum_text = checksum_payload.decode("ascii")
                except UnicodeDecodeError as exc:
                    raise ValueError("recovered P8 checksum ledger is not ASCII") from exc
                if not checksum_text.endswith("\n"):
                    raise ValueError("recovered P8 checksum ledger newline differs")
                checksum_rows: list[tuple[str, str]] = []
                for line in checksum_text.splitlines():
                    if len(line) < 67 or line[64:66] != "  ":
                        raise ValueError("recovered P8 checksum row shape differs")
                    digest, relative = line[:64], line[66:]
                    if (
                        len(digest) != 64
                        or digest.lower() != digest
                        or any(character not in "0123456789abcdef" for character in digest)
                    ):
                        raise ValueError("recovered P8 checksum digest differs")
                    portable = PurePosixPath(relative)
                    if (
                        portable.is_absolute()
                        or str(portable) != relative
                        or any(part in {"", ".", ".."} for part in portable.parts)
                    ):
                        raise ValueError("recovered P8 checksum path is unsafe")
                    checksum_rows.append((relative, digest))
                relative_names = [row[0] for row in checksum_rows]
                if (
                    len(checksum_rows) != checksum_spec["covered_regular_files"]
                    or relative_names != sorted(relative_names)
                    or len(relative_names) != len(set(relative_names))
                ):
                    raise ValueError("recovered P8 checksum roster differs")
                checksum_member_name = checksum_spec["path"]
                expected_covered = {
                    member.name.removeprefix(root_name + "/")
                    for member in regular
                    if member.name != checksum_member_name
                }
                if set(relative_names) != expected_covered:
                    raise ValueError("recovered P8 checksum coverage differs")
                for relative, expected_digest in checksum_rows:
                    member = member_by_name.get(f"{root_name}/{relative}")
                    if member is None or not member.isfile():
                        raise ValueError("recovered P8 checksummed member is missing")
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ValueError("recovered P8 checksummed member cannot be read")
                    with extracted:
                        digest = _sha256_stream(extracted)
                    if digest != expected_digest:
                        raise ValueError("recovered P8 checksummed member hash differs")

                normal_name = f"{root_name}/evidence/normal/report.json"
                optimized_name = f"{root_name}/evidence/optimized/report.json"
                if key_payloads[normal_name] != key_payloads[optimized_name]:
                    raise ValueError("recovered P8 normal and optimized reports differ")
                if parsed_key_json[normal_name].get("status") != expected["result_status"]:
                    raise ValueError("recovered P8 report status differs")
                protocol_name = f"{root_name}/PROTOCOL.json"
                calibration = parsed_key_json[protocol_name].get(
                    "calibration_disclosure"
                )
                claims = parsed_key_json[protocol_name].get("claim_controls")
                if not isinstance(calibration, dict) or not isinstance(claims, dict):
                    raise ValueError("recovered P8 claim controls differ")
                if (
                    calibration.get(
                        "configuration_selected_after_exploratory_same_state_run"
                    )
                    is not True
                    or calibration.get("exploratory_raw_transcript_retained") is not False
                    or calibration.get(
                        "gates_are_confirmation_thresholds_not_blind_predictions"
                    )
                    is not True
                    or claims.get("gpu_speed_superiority_claimed") is not False
                    or claims.get(
                        "same_state_calibration_is_independent_validation_claimed"
                    )
                    is not False
                ):
                    raise ValueError("recovered P8 calibration disclosure differs")

            handle.seek(0)
            second_digest = _sha256_stream(handle)
            after = os.fstat(handle.fileno())
            stable_identity = (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) == (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if not stable_identity or second_digest != first_digest:
                raise ValueError("recovered P8 bundle changed during audit")
    except tarfile.TarError as exc:
        raise ValueError("recovered P8 bundle cannot be parsed") from exc

    return {
        "status": "RECOVERED_P8_ACCURACY_BUNDLE_MATCHES",
        "artifact": dict(expected_artifact),
        "key_members": observed_key_members,
        "sha256sums": dict(expected["sha256sums"]),
        "result_status": expected["result_status"],
        "calibration_disclosure": expected["calibration_disclosure"],
        "claim_limit": expected["claim_limit"],
        "absolute_path_omitted": True,
        "speed_claim_authorized": False,
    }


def _receipt(path: str, sha256: str, role: str) -> dict[str, Any]:
    return {"path": path, "sha256": sha256, "role": role}


def _rc13_source_archive_document() -> dict[str, Any]:
    return {
        "release": RECOVERED_RC13_SDIST["release"],
        "artifact": dict(RECOVERED_RC13_SDIST["artifact"]),
        "binding_manifest": dict(RECOVERED_RC13_SDIST["binding_manifest"]),
        "members": [dict(member) for member in RECOVERED_RC13_SDIST["members"]],
        "claim_limit": RECOVERED_RC13_SDIST["claim_limit"],
    }


def _p8_accuracy_bundle_document() -> dict[str, Any]:
    return {
        "artifact": dict(RECOVERED_P8_ACCURACY_BUNDLE["artifact"]),
        "members": [
            dict(member) for member in RECOVERED_P8_ACCURACY_BUNDLE["members"]
        ],
        "sha256sums": dict(RECOVERED_P8_ACCURACY_BUNDLE["sha256sums"]),
        "result_status": RECOVERED_P8_ACCURACY_BUNDLE["result_status"],
        "calibration_disclosure": RECOVERED_P8_ACCURACY_BUNDLE[
            "calibration_disclosure"
        ],
        "claim_limit": RECOVERED_P8_ACCURACY_BUNDLE["claim_limit"],
    }


def _historical_report_receipt(
    path: str,
    byte_length: int,
    sha256: str,
    binding: str,
    role: str,
) -> dict[str, Any]:
    return {
        "path": path,
        "bytes": byte_length,
        "sha256": sha256,
        "binding": binding,
        "role": role,
    }


def _historical_report_roster_sha256() -> str:
    return hashlib.sha256(
        b"\n".join(
            f"{path}\0{byte_length}\0{digest}\0{binding}\0{role}".encode("utf-8")
            for path, byte_length, digest, binding, role in RECOVERED_HISTORICAL_REPORTS
        )
    ).hexdigest()


def _open_relative_regular_nofollow(root_fd: int, relative_path: str) -> int:
    portable = PurePosixPath(relative_path)
    parts = portable.parts
    if (
        portable.is_absolute()
        or not parts
        or str(portable) != relative_path
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError("historical report locator is not a canonical relative path")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory_flag:
        raise ValueError("platform lacks required no-follow directory support")
    common = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow
    parent_fd = os.dup(root_fd)
    try:
        for component in parts[:-1]:
            next_fd = os.open(
                component,
                common | directory_flag,
                dir_fd=parent_fd,
            )
            os.close(parent_fd)
            parent_fd = next_fd
        return os.open(parts[-1], common, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError("historical report cannot be opened without symlinks") from exc
    finally:
        os.close(parent_fd)


def audit_recovered_report_root(root: Path) -> dict[str, Any]:
    """Verify the selected external historical-report roster without path custody.

    Every path component is opened relative to held no-follow directory file
    descriptors.  Each report is strict-JSON parsed and hashed twice on one held
    inode.  The returned portable receipt intentionally omits the caller's local
    backup path.
    """

    expected_paths = [entry[0] for entry in RECOVERED_HISTORICAL_REPORTS]
    if len(expected_paths) != len(set(expected_paths)):
        raise ValueError("historical report roster contains duplicate paths")
    root_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    if not getattr(os, "O_NOFOLLOW", 0) or not getattr(os, "O_DIRECTORY", 0):
        raise ValueError("platform lacks required no-follow directory support")
    try:
        root_fd = os.open(Path(root), root_flags)
    except OSError as exc:
        raise ValueError("historical report root cannot be opened safely") from exc
    observed: list[dict[str, Any]] = []
    try:
        root_before = os.fstat(root_fd)
        if not stat.S_ISDIR(root_before.st_mode):
            raise ValueError("historical report root is not a directory")
        for path, byte_length, expected_hash, binding, role in RECOVERED_HISTORICAL_REPORTS:
            descriptor = _open_relative_regular_nofollow(root_fd, path)
            with os.fdopen(descriptor, "rb", closefd=True) as handle:
                before = os.fstat(handle.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError(f"historical report is not regular: {path}")
                if before.st_nlink != 1:
                    raise ValueError(f"historical report link count differs: {path}")
                if before.st_size != byte_length:
                    raise ValueError(f"historical report byte length differs: {path}")
                raw = handle.read(byte_length + 1)
                if len(raw) != byte_length:
                    raise ValueError(f"historical report read length differs: {path}")
                first_digest = hashlib.sha256(raw).hexdigest()
                if first_digest != expected_hash:
                    raise ValueError(f"historical report hash differs: {path}")
                try:
                    parsed = json.loads(
                        raw,
                        parse_constant=_reject_nonfinite_json_constant,
                    )
                except (UnicodeDecodeError, ValueError) as exc:
                    raise ValueError(f"historical report is not strict JSON: {path}") from exc
                if not isinstance(parsed, dict):
                    raise ValueError(f"historical report top level is not an object: {path}")
                handle.seek(0)
                second_digest = _sha256_stream(handle)
                after = os.fstat(handle.fileno())
                stable_identity = (
                    before.st_dev,
                    before.st_ino,
                    before.st_mode,
                    before.st_nlink,
                    before.st_size,
                    before.st_mtime_ns,
                ) == (
                    after.st_dev,
                    after.st_ino,
                    after.st_mode,
                    after.st_nlink,
                    after.st_size,
                    after.st_mtime_ns,
                )
                if not stable_identity or second_digest != first_digest:
                    raise ValueError(f"historical report changed during audit: {path}")
                observed.append(
                    {
                        "path": path,
                        "bytes": byte_length,
                        "sha256": first_digest,
                        "binding": binding,
                        "role": role,
                        "schema": parsed.get("schema"),
                        "status": parsed.get("status"),
                    }
                )
        root_after = os.fstat(root_fd)
        if (root_before.st_dev, root_before.st_ino) != (
            root_after.st_dev,
            root_after.st_ino,
        ):
            raise ValueError("historical report root identity changed")
    finally:
        os.close(root_fd)
    return {
        "status": "SELECTED_RECOVERED_HISTORICAL_REPORTS_MATCH",
        "files": len(observed),
        "ordered_roster_sha256": _historical_report_roster_sha256(),
        "reports": observed,
        "absolute_paths_omitted": True,
        "current_speed_proof_count": 0,
    }


def _load_recovered_report_inventory(root: Path) -> dict[str, Any]:
    expected = RECOVERED_REPORT_INVENTORY
    path = root / expected["path"]
    if not path.is_file() or path.is_symlink():
        raise ValueError("portable recovered-report inventory is missing or not regular")
    raw = path.read_bytes()
    if len(raw) != expected["bytes"]:
        raise ValueError("portable recovered-report inventory byte length differs")
    if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
        raise ValueError("portable recovered-report inventory hash differs")
    try:
        parsed = json.loads(raw, parse_constant=_reject_nonfinite_json_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("portable recovered-report inventory is not strict JSON") from exc
    if not isinstance(parsed, dict) or parsed.get("schema") != expected["schema"]:
        raise ValueError("portable recovered-report inventory schema differs")
    if raw != canonical_bytes(parsed):
        raise ValueError("portable recovered-report inventory is not canonical JSON")
    if not parsed.get("nonportable_local_paths_omitted"):
        raise ValueError("portable recovered-report inventory path policy differs")
    if b"/home/" in raw or b"/tmp/" in raw:
        raise ValueError("portable recovered-report inventory leaks a local absolute path")
    counts = parsed.get("counts")
    if counts != expected["counts"]:
        raise ValueError("portable recovered-report inventory counts differ")
    reports = parsed.get("reports")
    if not isinstance(reports, list) or len(reports) != counts["referenced_basenames"]:
        raise ValueError("portable recovered-report inventory report roster differs")
    basenames = [report.get("basename") for report in reports if isinstance(report, dict)]
    if len(basenames) != len(reports) or len(set(basenames)) != len(basenames):
        raise ValueError("portable recovered-report inventory basenames are not unique")
    if basenames != sorted(basenames):
        raise ValueError("portable recovered-report inventory order differs")

    present = [report for report in reports if report.get("present") is True]
    absent = [report for report in reports if report.get("present") is False]
    exact_present = [
        report
        for report in reports
        if report.get("exact_expected_artifact_present") is True
    ]
    exact_absent_or_wrong = [
        report
        for report in reports
        if report.get("exact_expected_artifact_present") is False
    ]
    no_prior_binder = [
        report
        for report in reports
        if report.get("exact_expected_artifact_present") is None
    ]
    observed_raw = [
        report
        for report in present
        if report.get("raw_timing_sample_arrays_present") is True
    ]
    derived_counts = {
        "referenced_basenames": len(reports),
        "basename_present": len(present),
        "basename_absent": len(absent),
        "retained_artifact_present_and_not_known_wrong_bytes": sum(
            report.get("custody_status") != "SAME_BASENAME_WRONG_BYTES"
            for report in present
        ),
        "prior_hash_bound_expected_artifact_present": len(exact_present),
        "prior_hash_bound_expected_artifact_absent_or_wrong_bytes": len(
            exact_absent_or_wrong
        ),
        "no_prior_cryptographic_report_binder": len(no_prior_binder),
        "present_without_prior_cryptographic_report_binder": sum(
            report.get("present") is True for report in no_prior_binder
        ),
        "absent_without_prior_cryptographic_report_binder": sum(
            report.get("present") is False for report in no_prior_binder
        ),
        "present_with_raw_timing_sample_arrays": len(observed_raw),
        "present_without_raw_timing_sample_arrays": len(present) - len(observed_raw),
        "prior_hash_bound_expected_artifacts_with_raw_timing_sample_arrays": sum(
            report.get("raw_timing_sample_arrays_present") is True
            for report in exact_present
        ),
    }
    if derived_counts != counts:
        raise ValueError("portable recovered-report inventory derived counts differ")
    if any(
        record.get("verification") not in {
            "VERIFIED_BINDER_IDENTITY",
            "RECORDED_REFERENCE_NAME",
            "RETAINED_SOURCE_RELATIONSHIP_ONLY",
        }
        for report in reports
        for record in report.get("binding_or_reference_records", [])
    ):
        raise ValueError("portable recovered-report binder classification differs")
    return {
        "path": expected["path"],
        "bytes": expected["bytes"],
        "sha256": expected["sha256"],
        "schema": expected["schema"],
        "counts": dict(counts),
        "source_audit_inventory_bytes": expected["source_audit_inventory_bytes"],
        "source_audit_inventory_sha256": expected["source_audit_inventory_sha256"],
        "portable_transform": expected["portable_transform"],
        "nonportable_local_paths_omitted": True,
        "scope": parsed.get("scope"),
        "definitions": dict(parsed.get("definitions", {})),
    }


def _verify_receipts(root: Path, entries: Iterable[tuple[str, str, str]]) -> None:
    for relative_path, expected, _role in entries:
        path = root / relative_path
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"required regular evidence file missing: {relative_path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"evidence hash differs: {relative_path}")


def audit_recovered_root(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    _verify_receipts(root, RECOVERED_RESEARCH)
    return {
        "status": "RECOVERED_RESEARCH_BYTES_MATCH",
        "root_not_part_of_portable_receipt": True,
        "files": len(RECOVERED_RESEARCH),
        "ordered_roster_sha256": hashlib.sha256(
            b"\n".join(
                f"{path}\0{digest}\0{role}".encode("utf-8")
                for path, digest, role in RECOVERED_RESEARCH
            )
        ).hexdigest(),
    }


def audit_recovered_roots(roots: Iterable[Path]) -> dict[str, Any]:
    resolved_roots = [root.resolve(strict=True) for root in roots]
    if not resolved_roots:
        raise ValueError("at least one recovered root is required")
    if len(set(resolved_roots)) != len(resolved_roots):
        raise ValueError("recovered roots must be distinct")
    receipts = [audit_recovered_root(root) for root in resolved_roots]
    roster_hashes = {receipt["ordered_roster_sha256"] for receipt in receipts}
    if len(roster_hashes) != 1:
        raise ValueError("recovered roots do not share one selected-byte roster")
    return {
        "status": "RECOVERED_RESEARCH_COPIES_MATCH",
        "copies": len(receipts),
        "files_per_copy": len(RECOVERED_RESEARCH),
        "ordered_roster_sha256": next(iter(roster_hashes)),
        "absolute_paths_omitted": True,
    }


def _lanes() -> list[dict[str, Any]]:
    return [
        {
            "id": "P1_CPU_SMALLN",
            "class": "CURRENT_OVERLAP",
            "jx": "SUPPORTED_COMPILED_CPU_RKF78",
            "rebound": "IAS15",
            "model": "FULLY_MUTUAL_NEWTONIAN",
            "body_counts": [2, 11, 32],
            "systems": [1],
            "timing_scopes": ["NUMERICAL_CORE", "END_TO_END"],
            "accuracy_gate": "MAX_COMPONENTWISE_CHECKPOINT_ERROR_LE_1E_MINUS_10",
            "historical_prior": "JX_WON_ALL_THREE_ROWS_AGAINST_5_1_1",
        },
        {
            "id": "P2_BATCH_SMALLN",
            "class": "CURRENT_OVERLAP",
            "jx": ["SUPPORTED_CPU_BATCH", "SUPPORTED_FUSED_CUDA_RKF78"],
            "rebound": "IAS15_PRESTARTED_PROCESS_POOL",
            "model": "FULLY_MUTUAL_NEWTONIAN",
            "body_counts": [2, 11, 32],
            "systems": [1, 8, 64, 256],
            "cpu_worker_counts": [1, 6],
            "timing_scopes": ["INTEGRATION_ONLY", "END_TO_END"],
            "accuracy_gate": "MAX_COMPONENTWISE_CHECKPOINT_ERROR_LE_1E_MINUS_10",
            "historical_prior": "JX_GPU_WON_256_SYSTEM_ROWS_AGAINST_5_1_1",
        },
        {
            "id": "P3_WH_EQUAL_AND_ENDPOINT",
            "class": "CURRENT_OVERLAP_PLUS_RECOVERED_CANDIDATE",
            "jx": ["SUPPORTED_WH", "RECOVERED_PRIVATE_WH_V8"],
            "rebound": "WHFAST",
            "model": "WEAK_HIERARCHY_MUTUAL_NEWTONIAN_THREE_BODY",
            "periods": 100,
            "fixed_steps": 6400,
            "retained_outputs": 401,
            "timing_scopes": [
                "EQUAL_OUTPUT_SINGLE_PASS",
                "JX_MANDATORY_REPLAY",
                "MATCHED_ENDPOINT_ONLY_BOTH_ENGINES",
            ],
            "accuracy_gate": "INHERIT_RC8_IAS15_ENVELOPE_WITHOUT_WEAKENING",
            "historical_prior": "SINGLE_PASS_WON_REPLAY_AND_ENDPOINT_LOST",
        },
        {
            "id": "P4_GR15_TWO_BODY",
            "class": "CURRENT_OVERLAP_PLUS_RECOVERED_SPECIALIZATION",
            "jx": ["SUPPORTED_GR15_V3", "RECOVERED_PRIVATE_GR15_V6_TWO_BODY"],
            "rebound": "IAS15",
            "workloads": [
                "CIRCULAR_100_PERIOD_ENDPOINT",
                "CIRCULAR_100_PERIOD_101_OUTPUTS",
                "ECCENTRIC_E_0_9_ONE_PERIOD_17_OUTPUTS",
            ],
            "accuracy_gates": {
                "circular_position_velocity_l2_max": "0x1.12e0be826d695p-29",
                "eccentric_position_velocity_l2_max": "0x1.12e0be826d695p-30",
                "circular_relative_total_energy_max": "0x1.5fd7fe1796495p-36",
                "circular_relative_angular_momentum_max": "0x1.5fd7fe1796495p-36",
                "eccentric_relative_total_energy_max": "0x1.5fd7fe1796495p-37",
                "eccentric_relative_angular_momentum_max": "0x1.5fd7fe1796495p-37",
            },
            "qualification_only_gates": {
                "reverse_position_componentwise_max": "0x1.12e0be826d695p-29",
                "reverse_velocity_componentwise_max": "0x1.12e0be826d695p-29",
                "reverse_probe_is_not_a_timed_row": True,
            },
            "historical_prior": "SUPPORTED_TIED_OR_LOST_PRIVATE_V6_WON_NARROW_ROWS",
        },
        {
            "id": "P5_ENCOUNTER",
            "class": "CURRENT_OVERLAP_PLUS_RECOVERED_PROTOTYPES",
            "jx": [
                "SUPPORTED_HYBRID",
                "RECOVERED_STATEFUL_NATIVE_LOOP",
                "PROOF_GATED_CIRCULAR_FAR_SPECIALIZATION",
            ],
            "rebound": ["TRACE", "MERCURIUS"],
            "workloads": ["ALL_FAR", "CLOSE_SCATTER", "CENTRAL_PERICENTER"],
            "timing_scopes": ["RAW_ARRAY_ALL_OUTPUTS", "PUBLIC_API_END_TO_END"],
            "accuracy_gate": "EXACT_LANE_GATE_MUST_BE_FROZEN_BEFORE_EXECUTION",
            "historical_prior": "GENERAL_JX_LOST_CIRCULAR_FAR_SPECIALIZATION_WON",
            "specialization_cannot_support_general_claim": True,
            "circular_far_history": {
                "preregistration_sha256": (
                    "36d5c80a52e48e4b6f05e769aa2d5c58b5abdd1b2dd9bae3367f7fa1a3cfdd16"
                ),
                "trace_protocol_sha256": (
                    "fe1c3636f829659af33c9aff4d48404f085ee7e07229c113c8fd47c99f579fa4"
                ),
                "mercurius_protocol_sha256": (
                    "f1f9ee020fda59fb2b5d4e75222e753716ddc2d032245a7950d8c9202d45c847"
                ),
            },
        },
        {
            "id": "P6_EIH_BATCH",
            "class": "CURRENT_OVERLAP_RECONSTRUCTION_REQUIRED",
            "jx": ["NATIVE_CPU_EIH_BATCH", "CUDA_EIH_BATCH"],
            "rebound": "IAS15_PLUS_REBOUNDX_GR_FULL",
            "model": "ELEVEN_BODY_NEWTONIAN_PLUS_MUTUAL_EIH_1PN",
            "years": 10,
            "systems": [1, 2, 4, 8, 16, 32, 64, 128],
            "cpu_worker_counts": [1, 6],
            "accuracy_gate": "EXACT_RC14_SUCCESSOR_GATE_REQUIRED",
            "historical_prior": (
                "RECOVERED_FULL_RAW_REBOUND_5_1_1_REBOUNDX_5_1_0_REPORTS; "
                "NOT_CURRENT_AND_MODEL_CONCORDANCE_UNESTABLISHED"
            ),
            "equation_relationship": (
                "MODEL_CONCORDANCE_UNESTABLISHED_UNSCORED_UNTIL_ANALYTIC_AND_NUMERIC_"
                "EQUIVALENCE_PROTOCOL_PASSES"
            ),
            "recovered_runner_sha256": (
                "49b106a9b09f74e5794bcb5b0a197540f9861e1b4b3fa6fdc0dcd8f135436726"
            ),
            "recovered_protocol_sha256": (
                "f6dbaf9dd43cc82b405119e5b7d0a383508ef33708712e0e9048b0ec8d966196"
            ),
            "historical_source_closure": {
                "exact_loose_historical_dependencies_retained": 5,
                "historical_protocol_bound_wrappers_recovered_from_rc13_sdist": 2,
                "recovered_source_archive": _rc13_source_archive_document(),
                "expected_cuda_wrapper_sha256": (
                    "7765a3ed6e628b63031841feab91e05e21edb190f57290eb8089d1ff1b1d3f53"
                ),
                "observed_later_cuda_wrapper_sha256": (
                    "c214cd17045667e06f16533df3b5e95c664a348d587432daf1fac6887181d9d3"
                ),
                "expected_cpu_wrapper_sha256": (
                    "aa694614e55feb827db5075c46d1489bcdca580dad7d0843957de948d2dddc85"
                ),
                "observed_later_cpu_wrapper_sha256": (
                    "2bc2bdb8676ec95700fbb719d934368d0156d402846c58dd5c252b9d5fbcb53d"
                ),
            },
        },
        {
            "id": "P7_LONG_ANTI_CHERRY_PICK",
            "class": "MANDATORY_LOSS_SENTINEL",
            "workloads": [
                "ELEVEN_BODY_NEWTONIAN_100_YEARS_VS_IAS15",
                "SMOOTH_HIERARCHY_100_PERIODS_VS_WHFAST",
                "CLOSE_SCATTER_VS_TRACE_AND_MERCURIUS",
            ],
            "accuracy_gate": "INHERIT_LOCKED_LONG_RACE_GATES_WITHOUT_WEAKENING",
            "historical_prior": "REBOUND_WON_ALL_ROWS_BY_LARGE_MARGINS",
            "may_not_be_removed_after_any_negative_result": True,
        },
        {
            "id": "P8_REPRO_ACCURACY",
            "class": "NO_SPEED_CLAIM",
            "model": "ELEVEN_BODY_MUTUAL_NEWTONIAN_100_YEARS",
            "input_sha256": (
                "0718ba6c05444c720d012bc7906e3eee226664b58d7929648917f9f19003d4c5"
            ),
            "comparators": ["JX_GPU_RKF78", "IAS15", "GLOBAL_LEAPFROG"],
            "purpose": "REPRODUCIBILITY_ACCURACY_AND_CONSERVATION_ONLY",
            "historical_bundle_sha256": (
                "45123f7b2ec860d6b50d00c84e09b89e9a4c04618b1194d3148dab93342a9155"
            ),
            "historical_bundle_is_confirmation_after_same_state_calibration": True,
            "historical_exploratory_transcript_retained": False,
            "independent_physics_validation_claimed": False,
            "speed_claim_authorized": False,
        },
        {
            "id": "P9_MASSLESS_POPULATION",
            "class": "CAPABILITY_GAP_NEW_IMPLEMENTATION_REQUIRED",
            "jx": "CUDA_SHARED_ACTIVE_PASSIVE_RESTRICTED_SOLVER",
            "rebound": ["WHFAST_TEST_PARTICLES", "MERCURIUS_TEST_PARTICLES"],
            "source_count": 5,
            "tracer_counts": [16800, 100000],
            "steps": 10000,
            "timing_scopes": ["DEVICE_RESIDENT_CORE", "END_TO_END"],
            "workloads": ["QUIET", "ROUTED_ENCOUNTER"],
            "accuracy_gate": "SEPARATE_PREREGISTRATION_REQUIRED_BEFORE_IMPLEMENTATION",
            "historical_prior": "JX_ENGINEERING_PASS_AND_REBOUND_SCIENCE_RUN_NOT_COMPARABLE",
            "claim_scope": "HARDWARE_SYSTEM_THROUGHPUT_NOT_SOLVER_SPEED",
            "active_source_contract": (
                "IDENTICAL_INITIAL_ACTIVE_AND_MASSLESS_BYTES_MATCHED_EQUATIONS_STEPS_"
                "OUTPUT_EPOCHS_AND_COMMON_ORACLE_GATES; EACH_ENGINE_EVOLVES_ITS_OWN_"
                "ACTIVE_HISTORY"
            ),
            "frozen_source_replay_is_not_a_native_rebound_comparison": True,
        },
        {
            "id": "P10_ARCHIVE_CHECKPOINT_SERVICE",
            "class": "ENGINEERING_SERVICE_COMPARISON_UNSCORED",
            "jx": "LOCKED_SESSION_CHECKPOINT_AND_DURABLE_OUTPUT",
            "rebound": "SIMULATIONARCHIVE",
            "workloads": [
                "CHECKPOINT_WRITE",
                "RESTART_REPLAY",
                "SPARSE_OUTPUT_ARCHIVE",
            ],
            "timing_scopes": ["WRITE_SERVICE", "READ_RESTART_SERVICE"],
            "semantic_concordance": "BLOCKED_UNTIL_EXACT_SERVICE_CONTRACT_IS_FROZEN",
            "speed_claim_authorized": False,
            "use": "REPORT_ENGINEERING_COSTS_WITHOUT_CALLING_THEM_SOLVER_SPEED",
        },
    ]


def _row(
    row_id: str,
    lane: str,
    jx: str,
    rebound: str,
    scope: str,
    resource: str,
    *,
    workload: str = "INHERITED_FROM_LANE",
    gate: str = "EXACT_LANE_PREREGISTRATION_REQUIRED",
    preparation_status: str = "BLOCKED_EXACT_LANE_PROTOCOL",
    required: bool = True,
    claim_kind: str = "PRIMARY_SCORED_CAPABILITY",
) -> dict[str, Any]:
    return {
        "id": row_id,
        "lane": lane,
        "jx_variant": jx,
        "rebound_variant": rebound,
        "timing_scope": scope,
        "workload": workload,
        "resource_class": resource,
        "gate_binding": gate,
        "preparation_status": preparation_status,
        "outcome": "NOT_RUN",
        "required_for_portfolio": required,
        "claim_kind": claim_kind,
    }


def _row_roster() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bodies in (2, 11, 32):
        for scope in ("CORE", "END_TO_END"):
            rows.append(
                _row(
                    f"P1_B{bodies}_{scope}",
                    "P1_CPU_SMALLN",
                    "SUPPORTED_CPU_RKF78",
                    "IAS15",
                    scope,
                    "CPU_1_WORKER",
                )
            )
    for bodies in (2, 11, 32):
        for systems in (1, 8, 64, 256):
            for jx, rebound, resource in (
                ("CPU_BATCH_1", "IAS15_POOL_1", "CPU_1_WORKER"),
                ("CPU_BATCH_6", "IAS15_POOL_6", "CPU_6_WORKERS"),
                ("FUSED_CUDA", "IAS15_POOL_6", "ONE_GPU_VS_CPU_6_WORKERS"),
            ):
                for scope in ("CORE", "END_TO_END"):
                    primary = jx != "FUSED_CUDA" or systems >= 64
                    rows.append(
                        _row(
                            f"P2_B{bodies}_N{systems}_{jx}_{scope}",
                            "P2_BATCH_SMALLN",
                            jx,
                            rebound,
                            scope,
                            resource,
                            workload=f"B{bodies}_N{systems}",
                            required=primary,
                            preparation_status="BLOCKED_EXACT_LANE_PROTOCOL",
                            claim_kind=(
                                "PRIMARY_HARDWARE_SYSTEM_THROUGHPUT"
                                if jx == "FUSED_CUDA" and primary
                                else (
                                    "GPU_CROSSOVER_DIAGNOSTIC_UNSCORED"
                                    if jx == "FUSED_CUDA"
                                    else "PRIMARY_SCORED_CAPABILITY"
                                )
                            ),
                        )
                    )
    for jx in ("SUPPORTED_WH", "RECOVERED_WH_V8"):
        for scope in (
            "EQUAL_OUTPUT_SINGLE_PASS",
            "MANDATORY_REPLAY",
            "MATCHED_ENDPOINT_ONLY_BOTH_ENGINES",
        ):
            rows.append(
                _row(
                    f"P3_{jx}_{scope}",
                    "P3_WH_EQUAL_AND_ENDPOINT",
                    jx,
                    "WHFAST",
                    scope,
                    "CPU_1_WORKER",
                    required=jx == "RECOVERED_WH_V8",
                    preparation_status="BLOCKED_EXACT_LANE_PROTOCOL",
                    claim_kind=(
                        "PRIMARY_SCORED_CAPABILITY"
                        if jx == "RECOVERED_WH_V8"
                        else "LEGACY_DIAGNOSTIC_UNSCORED"
                    ),
                )
            )
    for jx in ("SUPPORTED_GR15_V3", "RECOVERED_GR15_V6_TWO_BODY"):
        for workload in ("CIRCULAR_ENDPOINT", "CIRCULAR_101_OUTPUTS", "ECCENTRIC_17_OUTPUTS"):
            for scope in ("CORE", "END_TO_END"):
                rows.append(
                    _row(
                        f"P4_{jx}_{workload}_{scope}",
                        "P4_GR15_TWO_BODY",
                        jx,
                        "IAS15",
                        scope,
                        "CPU_1_WORKER",
                        workload=workload,
                        required=jx == "RECOVERED_GR15_V6_TWO_BODY",
                        preparation_status="BLOCKED_EXACT_LANE_PROTOCOL",
                        claim_kind=(
                            "PRIMARY_SCORED_CAPABILITY"
                            if jx == "RECOVERED_GR15_V6_TWO_BODY"
                            else "LEGACY_DIAGNOSTIC_UNSCORED"
                        ),
                    )
                )
    for jx in ("SUPPORTED_HYBRID", "RECOVERED_STATEFUL_NATIVE_LOOP"):
        for workload in ("ALL_FAR", "CLOSE_SCATTER", "CENTRAL_PERICENTER"):
            for rebound in ("TRACE", "MERCURIUS"):
                for scope in ("RAW_ARRAY_ALL_OUTPUTS", "PUBLIC_API_END_TO_END"):
                    rows.append(
                        _row(
                            f"P5_{jx}_{workload}_{rebound}_{scope}",
                            "P5_ENCOUNTER",
                            jx,
                            rebound,
                            scope,
                            "CPU_1_WORKER",
                            workload=workload,
                            required=jx == "RECOVERED_STATEFUL_NATIVE_LOOP",
                            preparation_status="BLOCKED_EXACT_LANE_PROTOCOL",
                            claim_kind=(
                                "PRIMARY_SCORED_CAPABILITY"
                                if jx == "RECOVERED_STATEFUL_NATIVE_LOOP"
                                else "LEGACY_DIAGNOSTIC_UNSCORED"
                            ),
                        )
                    )
    for rebound in ("TRACE", "MERCURIUS"):
        for scope in ("RAW_ARRAY_ALL_OUTPUTS", "PUBLIC_API_END_TO_END"):
            rows.append(
                _row(
                    f"P5_CIRCULAR_FAR_SPECIALIZATION_{rebound}_{scope}",
                    "P5_ENCOUNTER",
                    "PROOF_GATED_CIRCULAR_FAR",
                    rebound,
                    scope,
                    "CPU_1_WORKER",
                    workload="ALL_FAR_CIRCULAR_TWO_BODY_PROVED_DOMAIN",
                    gate="CIRCULAR_FAR_PROOF_AND_EXACT_LANE_PREREGISTRATION_REQUIRED",
                    claim_kind="SPECIALIZED_DOMAIN_SCORED",
                )
            )
    for systems in (1, 2, 4, 8, 16, 32, 64, 128):
        for jx, rebound, resource in (
            ("EIH_CPU_1", "IAS15_REBOUNDX_1", "CPU_1_WORKER"),
            ("EIH_CPU_6", "IAS15_REBOUNDX_6", "CPU_6_WORKERS"),
            ("EIH_CUDA", "IAS15_REBOUNDX_6", "ONE_GPU_VS_CPU_6_WORKERS"),
        ):
            for scope in ("CORE", "END_TO_END"):
                rows.append(
                    _row(
                        f"P6_N{systems}_{jx}_{scope}",
                        "P6_EIH_BATCH",
                        jx,
                        rebound,
                        scope,
                        resource,
                        workload=f"ELEVEN_BODY_EIH_N{systems}",
                        gate="MODEL_CONCORDANCE_AND_EXACT_LANE_PREREGISTRATION_REQUIRED",
                        preparation_status="BLOCKED_MODEL_CONCORDANCE",
                    )
                )
    for row_id, jx, rebound in (
        ("P7_ELEVEN_BODY_100Y", "JX_RKF78", "IAS15"),
        ("P7_HIERARCHY_100P", "JX_WH", "WHFAST"),
        ("P7_CLOSE_SCATTER_TRACE", "JX_HYBRID", "TRACE"),
        ("P7_CLOSE_SCATTER_MERCURIUS", "JX_HYBRID", "MERCURIUS"),
    ):
        rows.append(
            _row(
                row_id,
                "P7_LONG_ANTI_CHERRY_PICK",
                jx,
                rebound,
                "MATCHED_END_TO_END",
                "CPU_1_WORKER",
                workload=row_id.removeprefix("P7_"),
            )
        )
    rows.append(
        _row(
            "P8_100Y_REPRO_ACCURACY",
            "P8_REPRO_ACCURACY",
            "JX_GPU_RKF78",
            "IAS15_AND_GLOBAL_LEAPFROG",
            "NO_TIMING",
            "UNSCORED_ACCURACY",
            workload="ELEVEN_BODY_100Y_REPRO_ACCURACY",
            gate="FROZEN_100Y_ACCURACY_PROTOCOL_REQUIRED",
            preparation_status="BLOCKED_CURRENT_RUNTIME_REPRO_PROTOCOL",
            required=False,
            claim_kind="ACCURACY_ONLY_UNSCORED",
        )
    )
    for tracers in (16800, 100000):
        for workload in ("QUIET", "ROUTED_ENCOUNTER"):
            for rebound in ("WHFAST_TEST_PARTICLES", "MERCURIUS_TEST_PARTICLES"):
                for scope in ("DEVICE_RESIDENT_CORE", "END_TO_END"):
                    primary = (
                        (workload == "QUIET" and rebound == "WHFAST_TEST_PARTICLES")
                        or (
                            workload == "ROUTED_ENCOUNTER"
                            and rebound == "MERCURIUS_TEST_PARTICLES"
                        )
                    )
                    rows.append(
                        _row(
                            f"P9_T{tracers}_{workload}_{rebound}_{scope}",
                            "P9_MASSLESS_POPULATION",
                            "NEW_CUDA_SHARED_ACTIVE_PASSIVE",
                            rebound,
                            scope,
                            "ONE_GPU_VS_CPU_6_WORKERS",
                            workload=workload,
                            gate=(
                                "IDENTICAL_INITIAL_ACTIVE_AND_MASSLESS_BYTES_MATCHED_"
                                "EQUATIONS_STEPS_OUTPUT_EPOCHS_COMMON_ORACLE_AND_EXACT_"
                                "LANE_PREREGISTRATION_REQUIRED"
                            ),
                            preparation_status="UNAVAILABLE_CAPABILITY",
                            required=primary,
                            claim_kind=(
                                "PRIMARY_HARDWARE_SYSTEM_THROUGHPUT"
                                if primary
                                else "CROSS_INTEGRATOR_DIAGNOSTIC_UNSCORED"
                            ),
                        )
                    )
    for workload, scope in (
        ("CHECKPOINT_WRITE", "WRITE_SERVICE"),
        ("RESTART_REPLAY", "READ_RESTART_SERVICE"),
        ("SPARSE_OUTPUT_ARCHIVE", "WRITE_SERVICE"),
    ):
        rows.append(
            _row(
                f"P10_{workload}",
                "P10_ARCHIVE_CHECKPOINT_SERVICE",
                "JX_LOCKED_SESSION_SERVICE",
                "REBOUND_SIMULATIONARCHIVE",
                scope,
                "CPU_1_WORKER_UNSCORED_SERVICE",
                workload=workload,
                gate="EXACT_SERVICE_SEMANTIC_CONCORDANCE_PROTOCOL_REQUIRED",
                preparation_status="BLOCKED_SERVICE_CONCORDANCE",
                required=False,
                claim_kind="ENGINEERING_SERVICE_REPORT_ONLY_UNSCORED",
            )
        )
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("row identifiers are not unique")
    allowed_preparation = {
        "BLOCKED_EXACT_LANE_PROTOCOL",
        "BLOCKED_MODEL_CONCORDANCE",
        "BLOCKED_CURRENT_RUNTIME_REPRO_PROTOCOL",
        "BLOCKED_SERVICE_CONCORDANCE",
        "UNAVAILABLE_CAPABILITY",
    }
    if any(row["preparation_status"] not in allowed_preparation for row in rows):
        raise ValueError("row preparation status is not recognized")
    if any(row["outcome"] != "NOT_RUN" for row in rows):
        raise ValueError("prepare-only row unexpectedly has an outcome")
    if len(rows) != 196:
        raise ValueError("finite row roster count differs")
    if sum(bool(row["required_for_portfolio"]) for row in rows) != 151:
        raise ValueError("required row roster count differs")
    return rows


def build_protocol(root: Path | None = None) -> dict[str, Any]:
    project_root = (root or Path(__file__).resolve().parents[1]).resolve()
    _verify_receipts(project_root, PUBLIC_EVIDENCE)
    report_inventory = _load_recovered_report_inventory(project_root)
    source = Path(__file__).resolve()
    rows = _row_roster()
    required_rows = sum(bool(row["required_for_portfolio"]) for row in rows)
    if required_rows == 0:
        raise ValueError("portfolio cannot have an empty required-row set")
    protocol = {
        "schema": SCHEMA,
        "status": STATUS,
        "prepared_date": "2026-10-15",
        "objective": (
            "Measure a finite, accuracy-gated JX-versus-current-REBOUND portfolio; "
            "preserve every loss and capability gap; optimize without weakening physics, "
            "outputs, tolerances, or resource accounting."
        ),
        "base_authority": {
            "commit": BASE_COMMIT,
            "tree": BASE_TREE,
            "version": "0.6.0rc16",
        },
        "builder": {
            "path": "benchmarks/jx_rebound_portfolio_v1.py",
            "sha256": _sha256(source),
            "stdlib_only": True,
            "dynamics_imports": False,
        },
        "comparator": {
            "name": "REBOUND",
            "version": REBOUND_VERSION,
            "githash": REBOUND_GITHASH,
            "wheel": dict(REBOUND_WHEEL),
            "sdist": dict(REBOUND_SDIST),
            "loaded_wheel_runtime": dict(REBOUND_RUNTIME),
            "reboundx": dict(REBOUNDX_SDIST),
            "license": "GPL-3.0-only",
            "clean_room_rule": "DO_NOT_COPY_REBOUND_OR_REBOUNDX_IMPLEMENTATION_INTO_JX",
        },
        "prepare_host_observation": {
            "cpu": "AMD Ryzen 5 5500",
            "physical_cores": 6,
            "logical_cpus": 12,
            "isa": ["AVX", "AVX2", "FMA"],
            "avx512_available": False,
            "whfast512_lane_status": "CAPABILITY_NOT_RUN_ON_THIS_HOST",
            "gpu": "NVIDIA GeForce RTX 5060 Ti",
            "gpu_uuid": "GPU-66041ce7-445e-a391-aadd-703ae6ebc672",
            "gpu_compute_capability": "12.0",
            "gpu_memory_mib": 16311,
            "driver_version": "595.99.02",
            "python": "3.14.4",
            "observation_is_not_a_future_runtime_receipt": True,
        },
        "evidence_audit": {
            "public": [_receipt(*entry) for entry in PUBLIC_EVIDENCE],
            "recovered_research": [_receipt(*entry) for entry in RECOVERED_RESEARCH],
            "recovered_release_source_archive": _rc13_source_archive_document(),
            "recovered_p8_historical_accuracy_bundle": (
                _p8_accuracy_bundle_document()
            ),
            "selected_recovered_historical_reports": [
                _historical_report_receipt(*entry)
                for entry in RECOVERED_HISTORICAL_REPORTS
            ],
            "selected_recovered_historical_report_count": len(
                RECOVERED_HISTORICAL_REPORTS
            ),
            "selected_recovered_roster_includes_reports_preregistrations_and_"
            "science_gates": True,
            "selected_recovered_historical_report_ordered_roster_sha256": (
                _historical_report_roster_sha256()
            ),
            "selected_external_direct_comparative_raw_sample_report_count": sum(
                "RAW_SAMPLES" in entry[4]
                and "NOT_REBOUND_SPEED_EVIDENCE" not in entry[4]
                and "VALIDATION_" not in entry[4]
                and "AGGREGATE_" not in entry[4]
                for entry in RECOVERED_HISTORICAL_REPORTS
            ),
            "recovered_copies": 2,
            "selected_recovered_research_roster_bytes_match_across_copies": True,
            "selected_recovered_research_roster_file_count": len(
                RECOVERED_RESEARCH
            ),
            "recovered_snapshot_git_observation": {
                "head": "31890c335ec85cc7ed4b2858e169f98ea829fb64",
                "tree": "933f46374918b1a5cfb067c66d03601c9ef24f32",
                "selected_roster_git_status": {
                    "untracked_files": 55,
                    "tracked_modified_files": [
                        "benchmarks/jx_fused_accuracy_matched_rebound.py"
                    ],
                    "tracked_clean_files": [
                        "benchmarks/rebound_whfast_comparison.py"
                    ],
                },
                "selected_roster_is_not_a_clean_committed_tree": True,
                "portable_authority_is_content_hash_not_local_absolute_path": True,
                "git_status_counts_are_recorded_observations_not_revalidated_by_"
                "audit_recovered": True,
            },
            "tracer_priors": [dict(item) for item in TRACER_PRIORS],
            "historical_direct_speed_comparator_versions": ["5.1.1"],
            "historical_non_speed_comparator_versions": ["4.4.11"],
            "current_speed_proof_count": 0,
            "original_temporary_locators_are_not_custody": True,
            "recovered_report_inventory": report_inventory,
            "recovered_report_inventory_scope_policy": {
                "included": [
                    "DIRECT_JX_REBOUND_OR_IAS15_TIMING_OR_ACCURACY_RECORDS",
                    "CANDIDATE_AND_RUNTIME_GATES_ALREADY_IN_THE_V2_INVENTORY",
                    "LATEST_OUTCOME_BLIND_TRANSLATIONAL_EIH_CHAIN",
                    "CANONICAL_GR15_100_YEAR_SAME_MODEL_VALIDATION",
                    "CURRENT_100_YEAR_SAME_EQUATION_INTEGRATOR_SCREEN",
                ],
                "excluded_reviewed_categories": [
                    "NONCOMPARATOR_LUNAR_LLR_AND_COUPLED_MODEL_SCIENCE",
                    "EFFECTIVE_FORCE_FITS_AND_REPLICATIONS",
                    "DEFORMABLE_MANTLE_GEODETIC_AND_PUBLIC_LUNAR_CHAINS",
                    "GPU_COUPLED_LUNAR_SCIENCE_WITHOUT_REBOUND_OR_IAS15_COMPARISON",
                    "SUPERSEDED_OR_INTERMEDIATE_DIRECT_DEVELOPMENT_VARIANTS_AND_"
                    "REPEATS_WHOSE_CONCLUSION_IS_PRESERVED_BY_A_SELECTED_"
                    "CANONICAL_REPORT",
                ],
                "explicit_canonical_duplicate_exclusions": [
                    "jx_current_newtonian_100y_report.json",
                    "jx_gr15_de440_long_validation_report_rerun.json",
                ],
                "p8_complete_archive_is_bound_separately_not_a_basename_row": True,
                "not_an_inventory_of_every_json_in_retained_storage": True,
            },
            "tracked_raw_c_comparative_reports": [
                "jx_raw_c_trace_baseline_v1",
                "jx_raw_c_trace_stateful_near_v2",
            ],
            "bound_absent_report_receipts": {
                "private_wh_v8": {
                    "bytes": 33865,
                    "sha256": (
                        "d37eda3c83d8bc5e3360ee74c2fb1616318ee5c385ddd8854bf14306fb03d3aa"
                    ),
                    "receipt_source": "BOUND_DIRTY_RECOVERED_RESEARCH_LEDGER",
                },
                "private_gr15_v6": {
                    "bytes": 30136,
                    "sha256": (
                        "30b5251c7a4fb575e5c5164b1b947805f8a33d00fa5d8aa6b7efe8291f741b53"
                    ),
                    "receipt_source": (
                        "BOUND_GR15_V6_COMPACT_RESULT_AND_DIRTY_RESEARCH_LEDGER"
                    ),
                },
                "private_circular_far_trace": {
                    "bytes": 24986,
                    "sha256": (
                        "4a44bc5a962417ee7f19a0d332697bde6dc1e30497eff49e73b5612fd5916a8f"
                    ),
                    "receipt_source": "BOUND_CIRCULAR_FAR_COMPACT_RESULT",
                },
                "private_circular_far_mercurius": {
                    "bytes": 25432,
                    "sha256": (
                        "733139e1d7757fd5bfaf072a0f4b1ec0133e20f9d1b6f247d95726ea4b4f27a2"
                    ),
                    "receipt_source": "BOUND_CIRCULAR_FAR_COMPACT_RESULT",
                },
                "private_native_stateful_near": {
                    "bytes": 23005,
                    "sha256": (
                        "96b3d0b5021195cb83d04d2c62e96dd7f2872b075696de777dc326d31562aca3"
                    ),
                    "receipt_source": "BOUND_DIRTY_RECOVERED_RESEARCH_LEDGER",
                },
                "private_raw_c_trace_stateful_near": {
                    "bytes": 41267,
                    "sha256": (
                        "3e2d881066ac6ea0c3c4851d279726a24b08816dde49545317984cbed644832e"
                    ),
                    "receipt_source": "BOUND_DIRTY_RECOVERED_RESEARCH_LEDGER",
                },
            },
            "unreceipted_missing_historical_output_name_count": 4,
            "private_wh_v8_confirmatory_report_present": False,
            "private_wh_v8_confirmatory_receipt_source": (
                "BOUND_DIRTY_RECOVERED_RESEARCH_LEDGER"
            ),
            "private_wh_v8_bound_pilot_throughput_relative_to_rebound": (
                "1.198506411"
            ),
            "private_gr15_v6_timing_scope_disclosure": (
                "JX_DEFENSIVE_OUTPUT_COPY_EXCLUDED_WHILE_REBOUND_PYTHON_PARTICLE_"
                "READBACK_INCLUDED; NO_RAW_C_EQUIVALENCE_CLAIM"
            ),
            "historical_scientific_model_evidence": {
                "latest_translational_eih_chain_artifact_count": 6,
                "fresh_start_confirmation_report_sha256": (
                    "aeef5ee710e0874a90aa4330cb121c8b7b2b0898af35fd8c10638eb1d6f711f0"
                ),
                "fresh_start_confirmation_primary_aggregate_improvement_fraction": (
                    "0.9921351988919058"
                ),
                "fresh_start_confirmation_claim_state": "SCREENING_ONLY",
                "de440_is_a_fitted_reference": True,
                "independent_physical_validation": False,
                "reboundx_gr_full_equation_concordance_established": False,
                "gr15_100y_same_model_report_sha256": (
                    "cfe7c99bb6c3683651325cc70623de76244051ff6c361ec61e51763677d2729c"
                ),
                "gr15_100y_is_accuracy_and_accounting_not_speed_evidence": True,
                "same_equation_100y_timing_report_sha256": (
                    "6da87aaaf232b13a09158bfb82c0cf30493da5ea7b1637afcab8d8ae86dfad53"
                ),
                "same_equation_100y_timing_not_solver_speed_evidence": True,
                "same_equation_100y_timing_asymmetry": (
                    "FUSED_NATIVE_JX_VS_REBOUND_IAS15_WITH_JX_EIH_EQUATIONS_"
                    "THROUGH_PYTHON_ADDITIONAL_FORCES_CALLBACK; ONE_SAMPLE_PER_SIDE"
                ),
            },
            "candidate_source_rosters_recovered": [
                "WH_V8",
                "GR15_V6",
                "EIH_BATCH_HISTORICAL_WRAPPERS_RECOVERED_FROM_RC13_SDIST",
            ],
            "rc13_eih_release_source_archive_recovered": True,
            "all_candidate_build_configurations_exact_bytes_recovered": False,
            "reason_dependency_closure_is_not_complete": (
                "THE_PROTOCOL_BOUND_PYPROJECT_BYTES_NO_LONGER_EXIST_AT_THE_DIRTY_SNAPSHOT_"
                "HEAD_FOR_OTHER_PRIVATE_CANDIDATES; GENERAL_HYBRID_DEPENDENCY_CLOSURE_"
                "IS_NOT_YET_FROZEN; AND_SIX_SPECIFIED_PRIVATE_RAW_REPORT_RECEIPTS_"
                "REMAIN_ABSENT"
            ),
        },
        "historical_priors": {
            "narrow_jx_wins": [
                "SHORT_SMALLN_CPU",
                "CUDA_BATCH_THROUGHPUT",
                "EQUAL_OUTPUT_WH_SINGLE_PASS",
                "PRIVATE_WH_V8_EQUAL_OUTPUT_PILOT",
                "PRIVATE_GR15_V6_TWO_BODY",
                "PRIVATE_PROOF_GATED_CIRCULAR_FAR",
            ],
            "jx_losses": [
                "MANDATORY_REPLAY_WH",
                "ENDPOINT_ONLY_WH",
                "ONE_LONG_ELEVEN_BODY_TRAJECTORY",
                "GENERAL_CLOSE_ENCOUNTER_TRACE",
                "GENERAL_CLOSE_ENCOUNTER_MERCURIUS",
            ],
            "not_current_claims": True,
        },
        "lanes": _lanes(),
        "row_roster": rows,
        "row_outcome_enum": {
            "NOT_RUN": "CURRENT_PREPARED_STATE; NEVER_COUNTS_AS_A_WIN",
            "JX_WIN": (
                "ALL_GATES_PASS_AND_MEDIAN_OF_PAIRED_R_I_GE_1_05_AND_"
                "LOWER_95_PERCENT_PAIRED_BOUND_GT_1"
            ),
            "REBOUND_WIN": (
                "ALL_GATES_PASS_AND_MEDIAN_OF_PAIRED_R_I_LE_0_9523809523809523_AND_"
                "UPPER_95_PERCENT_PAIRED_BOUND_LT_1"
            ),
            "TIE": "ALL_GATES_PASS_AND_NEITHER_RESOLVED_WIN_RULE_PASSES",
            "INVALID": "ANY_PHYSICS_ACCURACY_FAIRNESS_OR_RUNTIME_GATE_FAILS",
            "UNAVAILABLE": "REQUIRED_ENGINE_CAPABILITY_OR_RUNTIME_IS_ABSENT",
            "UNSCORED": "ROW_IS_EXPLICITLY_ACCURACY_ONLY_OR_MODEL_CONCORDANCE_ONLY",
            "SKIPPED": "REQUIRED_ROW_WAS_NOT_EXECUTED; NEVER_COUNTS_AS_A_WIN",
        },
        "row_classification_thresholds_decimal": {
            "jx_win_median_minimum": "1.05",
            "rebound_win_median_maximum": "0.9523809523809523",
            "jx_win_lower_bound_strictly_greater_than": "1.0",
            "rebound_win_upper_bound_strictly_less_than": "1.0",
        },
        "aggregate_rule": {
            "total_row_count": len(rows),
            "required_row_count": required_rows,
            "required_row_count_must_be_nonzero": True,
            "terminal_result_row_ids_must_equal_charter_row_ids_exactly": True,
            "terminal_result_requires_each_row_exactly_once": True,
            "missing_duplicate_or_unexpected_row_invalidates_result": True,
            "required_flags_must_match_charter_exactly": True,
            "finite_portfolio_beaten": (
                "TRUE_IFF_REQUIRED_ROW_COUNT_GT_ZERO_AND_EVERY_REQUIRED_ROW_OUTCOME_IS_"
                "EXACTLY_JX_WIN"
            ),
            "ties_block_success": True,
            "losses_block_success": True,
            "invalid_rows_block_success": True,
            "unavailable_rows_block_success": True,
            "not_run_rows_block_success": True,
            "skipped_rows_block_success": True,
            "unscored_rows_never_count_as_wins": True,
            "capability_gaps_never_count_as_wins": True,
            "general_framework_superiority_authorized": False,
        },
        "global_fairness_contract": {
            "same_initial_binary64_bytes": True,
            "same_physical_model_and_body_order": True,
            "same_units_frame_origin_epoch_and_G": True,
            "same_horizon_and_exact_output_epochs": True,
            "same_failure_domain": True,
            "same_accuracy_gate_before_timing_interpretation": True,
            "setup_allocation_transfer_sync_readback_and_replay_reported_separately": True,
            "numerical_core_and_end_to_end_never_merged": True,
            "latency_and_ensemble_throughput_never_merged": True,
            "cpu_worker_budget": [1, 6],
            "blas_and_openmp_threads_per_worker": 1,
            "gpu_count": 1,
            "warmups_each_implementation": 5,
            "timed_repetitions_by_lane": {
                "P1_CPU_SMALLN": 31,
                "P2_BATCH_SMALLN": 31,
                "P3_WH_EQUAL_AND_ENDPOINT": 31,
                "P4_GR15_TWO_BODY": 31,
                "P5_ENCOUNTER": 31,
                "P6_EIH_BATCH": 15,
                "P7_LONG_ANTI_CHERRY_PICK": 7,
                "P8_REPRO_ACCURACY": 0,
                "P9_MASSLESS_POPULATION": 15,
                "P10_ARCHIVE_CHECKPOINT_SERVICE": 15,
            },
            "execution_order": (
                "PAIR_INDEX_EVEN_JX_THEN_REBOUND_ODD_REBOUND_THEN_JX_WITHIN_EACH_ROW"
            ),
            "execution_order_seed": 2026101501,
            "balanced_interleaved_order_required": True,
            "raw_samples_required": True,
            "median_is_primary": True,
            "paired_ratio_definition": "R_I_EQUALS_REBOUND_SECONDS_I_DIVIDED_BY_JX_SECONDS_I",
            "paired_speed_ratio_ci": {
                "method": "PAIRED_BLOCK_PERCENTILE_BOOTSTRAP_OF_MEDIAN_R_I",
                "resamples": 100000,
                "rng": "NUMPY_GENERATOR_PCG64",
                "seed": 2026101502,
                "quantiles": [0.025, 0.975],
                "quantile_method": "linear",
            },
            "outlier_deletion_forbidden": True,
            "early_stopping_forbidden": True,
            "minimum_resolved_speedup": 1.05,
            "win_requires": [
                "ALL_ACCURACY_CONSERVATION_FAILURE_AND_DETERMINISM_GATES_PASS",
                "MEDIAN_OF_PAIRED_R_I_GE_1_05",
                "LOWER_95_PERCENT_PAIRED_SPEED_RATIO_BOUND_GT_1_0",
            ],
            "negative_results_retained": True,
            "post_result_row_or_lane_removal_forbidden": True,
            "successor_runtime_lock_required_fields": [
                "OS_KERNEL_GLIBC_AND_LIBC_IDENTITY",
                "PYTHON_NUMPY_CUPY_AND_CUDA_TOOLKIT_RUNTIME_IDENTITY",
                "COMPILER_LINKER_FLAGS_AND_BUILD_BACKEND_IDENTITY",
                "JX_WHEEL_SDIST_MODULE_AND_LOADED_SHARED_LIBRARY_HASHES",
                "CPU_AFFINITY_AND_SMT_POLICY",
                "CPU_GOVERNOR_AND_CLOCK_OBSERVATION",
                "OMP_OPENBLAS_MKL_NUMEXPR_AND_BLIS_THREAD_ENVIRONMENT",
                "GPU_UUID_DRIVER_RUNTIME_POWER_CLOCK_AND_PERSISTENCE_STATE",
                "GPU_SYNCHRONIZATION_POINTS",
                "CORE_SETUP_ALLOCATION_TRANSFER_READBACK_REPLAY_INCLUSIONS",
                "PROCESS_POOL_STARTUP_AND_LIFETIME",
            ],
        },
        "accuracy_first_tuning_contract": {
            "timed_rows_may_not_select_integrator_options": True,
            "calibration_inputs_must_be_disjoint_from_timed_inputs": True,
            "both_engines_get_symmetric_accuracy_first_tuning_opportunity": True,
            "settings_locked_before_any_timed_evaluation": True,
            "required_settings_by_comparator": {
                "IAS15": ["epsilon", "minimum_dt", "exact_finish_time"],
                "WHFAST": [
                    "coordinates",
                    "safe_mode",
                    "corrector",
                    "corrector2",
                    "kernel",
                    "dt",
                ],
                "TRACE": ["dt", "peri_mode", "encounter_configuration"],
                "MERCURIUS": ["dt", "hillfac", "safe_mode"],
            },
            "current_settings_frozen": False,
            "execution_blocked_until_successor_lock": True,
        },
        "execution_blockers": [
            "COMMIT_AND_ANNOTATED_TAG_THIS_PREPARE_ONLY_PORTFOLIO",
            "FREEZE_EXACT_JX_BUILT_ARTIFACT_AND_RUNTIME",
            "FREEZE_PER_LANE_PROTOCOL_INPUTS_GATES_AND_RUNNER_BEFORE_TIMING",
            "PROMOTE_OR_REIMPLEMENT_EACH_PRIVATE_CANDIDATE_IN_A_SEALED_SUCCESSOR",
            "BUILD_AND_BIND_REBOUNDX_RUNTIME_BEFORE_P6",
            "IMPLEMENT_AND_PREREGISTER_NATIVE_MASSLESS_SOLVER_BEFORE_P9",
        ],
        "known_nonexhaustive_capability_gaps_not_convertible_to_speed_wins": {
            "list_is_nonexhaustive": True,
            "items": [
                "GENERAL_COLLISION_MERGE_BOUNCE_FRAGMENTATION",
                "TREE_OR_FMM_LARGE_ACTIVE_N",
                "VARIATIONAL_EQUATIONS",
                "ARBITRARY_ODE_AND_CUSTOM_FORCE_ECOSYSTEM",
                "GENERAL_REGULARIZATION",
                "BOUNDARY_CONDITION_BREADTH",
                "MPI_DISTRIBUTED_EXECUTION",
                "MEGNO_AND_CHAOS_INDICATORS",
                "INTEGRATOR_PORTFOLIO_BREADTH",
                "WHFAST512_ON_NON_AVX512_HOST",
            ],
        },
        "claims": {
            "portfolio_charter_prepared": True,
            "portfolio_preregistered": False,
            "finite_row_roster_defined": True,
            "all_row_physics_inputs_and_gates_frozen": False,
            "selected_evidence_inventory_hash_bound": True,
            "dependency_closure_inventory_complete": False,
            "rebound_5_2_2_runtime_identified": True,
            "benchmark_executed": False,
            "jx_beats_rebound_generally": False,
            "jx_beats_rebound_on_any_5_2_2_lane": False,
            "portable_performance": False,
            "scientific_qualification": False,
            "production_authorization": False,
        },
    }
    return protocol


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_nonfinite_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {token}")


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member is forbidden: {key}")
        result[key] = value
    return result


def validate_protocol(path: Path, root: Path | None = None) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        parsed = json.loads(raw, parse_constant=_reject_nonfinite_json_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("protocol is not strict JSON") from exc
    expected = build_protocol(root)
    if raw != canonical_bytes(parsed):
        raise ValueError("protocol is not canonical JSON")
    if parsed != expected:
        raise ValueError("protocol differs from deterministic build")
    return {
        "status": "VALID_PORTFOLIO_CHARTER",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "lanes": len(parsed["lanes"]),
        "rows": len(parsed["row_roster"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--output", type=Path)
    validate = subparsers.add_parser("validate")
    validate.add_argument("protocol", type=Path)
    recovered = subparsers.add_parser("audit-recovered")
    recovered.add_argument("root", type=Path, nargs="+")
    recovered_sdist = subparsers.add_parser("audit-rc13-sdist")
    recovered_sdist.add_argument("archive", type=Path)
    recovered_p8 = subparsers.add_parser("audit-p8-bundle")
    recovered_p8.add_argument("archive", type=Path)
    recovered_reports = subparsers.add_parser("audit-reports")
    recovered_reports.add_argument("root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        payload = canonical_bytes(build_protocol())
        if args.output is None:
            print(payload.decode("utf-8"), end="")
        else:
            args.output.write_bytes(payload)
        return 0
    if args.command == "validate":
        print(json.dumps(validate_protocol(args.protocol), sort_keys=True))
        return 0
    if args.command == "audit-recovered":
        print(json.dumps(audit_recovered_roots(args.root), sort_keys=True))
        return 0
    if args.command == "audit-rc13-sdist":
        print(json.dumps(audit_recovered_rc13_sdist(args.archive), sort_keys=True))
        return 0
    if args.command == "audit-p8-bundle":
        print(json.dumps(audit_recovered_p8_accuracy_bundle(args.archive), sort_keys=True))
        return 0
    if args.command == "audit-reports":
        print(json.dumps(audit_recovered_report_root(args.root), sort_keys=True))
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
