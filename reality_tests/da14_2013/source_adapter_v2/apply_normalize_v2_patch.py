#!/usr/bin/env python3
"""Apply the prospectively registered MPC envelope parser correction."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

OLD_SHA256 = "32e7d20ab7ef8e488bba9299c78a1eabcb47146c28d88a580a99b4cd0d35e2fd"
NEW_SHA256 = "b9b9d49e576786f845779fff8ca33a56ddc424f1a17c6fac0a912819abf0d236"

OLD_BLOCK = '''def normalize_mpc_observations(payload: Any, *, designation: str,\n                               maximum_records: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:\n    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):\n        raise SourceSchemaError("MPC response does not match the single-object envelope")\n    envelope = payload[0]\n'''

NEW_BLOCK = '''def _mpc_single_object_envelope(payload: Any) -> tuple[dict[str, Any], dict[str, Any]]:\n    """Extract the one requested-object envelope from the MPC API response.\n\n    The documented consumer path is ``response.json()[0]``.  The live API\n    observed on 2026-09-02 returned one object envelope plus one opaque integer\n    metadata item.  Accept only those two narrowly defined layouts and never\n    interpret or persist the ancillary integer value.\n    """\n    if not isinstance(payload, list):\n        raise SourceSchemaError("MPC response is not an array envelope")\n    object_items = [item for item in payload if isinstance(item, dict)]\n    ancillary_items = [\n        item for item in payload\n        if not isinstance(item, dict)\n    ]\n    if len(object_items) != 1:\n        raise SourceSchemaError("MPC response does not contain exactly one object envelope")\n    if len(ancillary_items) > 1:\n        raise SourceSchemaError("MPC response contains too many ancillary envelope items")\n    if ancillary_items and (\n        isinstance(ancillary_items[0], bool)\n        or not isinstance(ancillary_items[0], int)\n    ):\n        raise SourceSchemaError("MPC ancillary envelope item is not an integer")\n    envelope = object_items[0]\n    metadata = {\n        "envelope_item_count": len(payload),\n        "object_envelope_count": 1,\n        "opaque_integer_metadata_present": bool(ancillary_items),\n        "opaque_integer_metadata_retained": False,\n    }\n    return envelope, metadata\n\n\ndef normalize_mpc_observations(payload: Any, *, designation: str,\n                               maximum_records: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:\n    envelope, envelope_metadata = _mpc_single_object_envelope(payload)\n'''

OLD_METADATA_BLOCK = '''        "identical_content_groups": sum(1 for count in seen.values() if count > 1),\n    }\n'''
NEW_METADATA_BLOCK = '''        "identical_content_groups": sum(1 for count in seen.values() if count > 1),\n        "envelope": envelope_metadata,\n    }\n'''


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    original = args.target.read_bytes()
    observed = digest(original)
    if observed != OLD_SHA256:
        raise SystemExit(f"BLOCKED: expected original normalize.py {OLD_SHA256}, observed {observed}")
    text = original.decode("utf-8")
    if text.count(OLD_BLOCK) != 1:
        raise SystemExit("BLOCKED: MPC envelope target block count is not exactly one")
    if text.count(OLD_METADATA_BLOCK) < 1:
        raise SystemExit("BLOCKED: MPC metadata target block is absent")
    text = text.replace(OLD_BLOCK, NEW_BLOCK, 1)
    text = text.replace(OLD_METADATA_BLOCK, NEW_METADATA_BLOCK, 1)
    updated = text.encode("utf-8")
    observed_new = digest(updated)
    if observed_new != NEW_SHA256:
        raise SystemExit(f"BLOCKED: patched normalize.py hash mismatch: {observed_new}")
    args.target.write_bytes(updated)
    print(f"NORMALIZE_V2_PATCH_APPLIED sha256={observed_new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
