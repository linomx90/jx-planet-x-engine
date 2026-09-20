#!/usr/bin/env python3
"""Fail closed when a public source snapshot crosses the JX release boundary."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


MAXIMUM_FILE_BYTES = 5 * 1024 * 1024

ALLOWED_DIRECTORIES = {
    ".github",
    "audits",
    "benchmarks",
    "docs",
    "runs",
    "src",
    "tests",
    "tools",
}

ALLOWED_TOP_LEVEL_FILES = {
    ".env.example",
    ".gitignore",
    "CHANGELOG.md",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "PUBLICATION_POLICY.md",
    "README.md",
    "RELEASE_MANIFEST_v0.3.0.json",
    "RELEASE_MANIFEST_v0.6.0rc1.json",
    "RELEASE_NOTES.md",
    "RELEASE_SHA256SUMS.txt",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
}

ALLOWED_SUFFIXES = {
    "",
    ".cff",
    ".csv",
    ".json",
    ".log",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

FORBIDDEN_DIRECTORY_NAMES = {
    ".venv",
    "archives",
    "build",
    "dist",
    "evidence",
    "library",
    "private",
    "venv",
}

FORBIDDEN_EXACT_NAMES = {
    ".env",
    ".netrc",
    "id_ed25519",
    "id_rsa",
}

FORBIDDEN_SUFFIXES = {
    ".7z",
    ".bin",
    ".core",
    ".dll",
    ".dylib",
    ".exe",
    ".jks",
    ".kdbx",
    ".key",
    ".p12",
    ".pem",
    ".pfx",
    ".rar",
    ".so",
    ".tar",
    ".tgz",
    ".whl",
    ".zip",
}

SECRET_PATTERNS = (
    (
        "private-key material",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
    ("AWS access key", re.compile(r"AKIA[0-9A-Z]{16}")),
    (
        "GitHub credential",
        re.compile(r"(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    ),
    ("Slack credential", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    (
        "credential assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|client[_-]?secret|password|passwd|authorization|bearer)"
            r"\s*[:=]\s*['\"][^'\"\r\n]{8,}['\"]"
        ),
    ),
    (
        "URL with embedded credentials",
        re.compile(r"https?://[^/@\s:]+:[^/@\s]+@"),
    ),
)

LOCAL_PATH_PATTERNS = (
    re.compile("/" + "home" + r"/[^/\s]+/"),
    re.compile("/" + "Users" + r"/[^/\s]+/"),
    re.compile(r"[A-Za-z]:\\" + "Users" + r"\\[^\\\s]+\\"),
    re.compile("file" + r"://"),
)


def _candidate_paths(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in root.rglob("*")
                if ".git" not in path.relative_to(root).parts
                and "__pycache__" not in path.relative_to(root).parts
                and (path.is_file() or path.is_symlink())
            ),
            key=lambda path: path.as_posix(),
        )
    )


def check(root: Path) -> list[str]:
    root = root.resolve(strict=True)
    failures: list[str] = []
    paths = _candidate_paths(root)
    if not paths:
        return ["public snapshot contains no files"]

    for path in paths:
        relative = path.relative_to(root)
        display = relative.as_posix()
        parts = relative.parts

        if path.is_symlink():
            failures.append(f"{display}: symbolic links are forbidden")
            continue

        if len(parts) == 1:
            if display not in ALLOWED_TOP_LEVEL_FILES:
                failures.append(f"{display}: unapproved top-level file")
        elif parts[0] not in ALLOWED_DIRECTORIES:
            failures.append(f"{display}: unapproved top-level directory")

        if any(part in FORBIDDEN_DIRECTORY_NAMES for part in parts[:-1]):
            failures.append(f"{display}: forbidden private/generated directory")

        lowered_name = path.name.lower()
        if lowered_name in FORBIDDEN_EXACT_NAMES:
            failures.append(f"{display}: forbidden credential filename")
        if lowered_name.startswith(".env.") and lowered_name != ".env.example":
            failures.append(f"{display}: only .env.example may be public")
        if lowered_name.startswith(("credentials.", "secrets.")):
            failures.append(f"{display}: forbidden credential filename")
        if any(lowered_name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            failures.append(f"{display}: forbidden archive/binary/key suffix")

        suffix = path.suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            failures.append(f"{display}: unapproved public file suffix {suffix!r}")

        size = path.stat().st_size
        if size > MAXIMUM_FILE_BYTES:
            failures.append(
                f"{display}: {size} bytes exceeds {MAXIMUM_FILE_BYTES}-byte cap"
            )
            continue

        payload = path.read_bytes()
        if b"\x00" in payload:
            failures.append(f"{display}: binary NUL byte detected")
            continue
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            failures.append(f"{display}: public source files must be UTF-8 text")
            continue

        for label, pattern in SECRET_PATTERNS:
            match = pattern.search(text)
            if match is not None:
                line = text.count("\n", 0, match.start()) + 1
                failures.append(f"{display}:{line}: possible {label}")
        for pattern in LOCAL_PATH_PATTERNS:
            match = pattern.search(text)
            if match is not None:
                line = text.count("\n", 0, match.start()) + 1
                failures.append(f"{display}:{line}: local filesystem path detected")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    failures = check(arguments.root)
    if failures:
        for failure in failures:
            print(f"PUBLICATION_BOUNDARY_FAILURE: {failure}", file=sys.stderr)
        return 1
    print("PUBLICATION_BOUNDARY_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
