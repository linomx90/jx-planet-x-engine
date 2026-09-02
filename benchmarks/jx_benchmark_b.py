#!/usr/bin/env python3
"""Loader for the deterministically split JX Benchmark B source."""
from __future__ import annotations

import os
from pathlib import Path

_PARTS = Path(__file__).with_name("jx_benchmark_b_parts")
_SOURCE = "".join(path.read_text(encoding="utf-8") for path in sorted(_PARTS.glob("part_*.pyfrag")))
_CODE = compile(_SOURCE, str(Path(__file__).resolve()), "exec")
if os.environ.get("JXB_COMPILE_ONLY") != "1":
    exec(_CODE, {"__name__": "__main__", "__file__": str(Path(__file__).resolve())})
