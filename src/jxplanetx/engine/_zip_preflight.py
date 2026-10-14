"""Allocation-bounded preflight for the engine's fixed-roster ZIP formats.

``zipfile.ZipFile`` builds one ``ZipInfo`` object per central-directory entry
before callers can inspect the roster.  The public engine loaders therefore
check the tiny, fixed central directory directly before handing bytes to
``zipfile``.  This is not a general ZIP parser; it accepts only the classic,
single-disk, no-comment/no-extra-field layout emitted by our deterministic
writers.  Full canonical-byte validation remains the responsibility of each
format-specific loader.
"""

from __future__ import annotations

import struct


_EOCD = struct.Struct("<4s4H2IH")
_CENTRAL_HEADER = struct.Struct("<4s6H3I5H2I")
_EOCD_SIGNATURE = b"PK\x05\x06"
_CENTRAL_SIGNATURE = b"PK\x01\x02"
_ZIP64_U16 = 0xFFFF
_ZIP64_U32 = 0xFFFFFFFF


def preflight_exact_zip_roster(raw: bytes, expected_names: tuple[str, ...]) -> None:
    """Reject non-canonical/large rosters before ``zipfile`` allocates entries.

    ``ValueError`` is intentionally used so each public codec can translate the
    failure into its own stable contract-error type.
    """

    if type(raw) is not bytes or type(expected_names) is not tuple:
        raise ValueError("ZIP preflight requires exact bytes and a tuple roster")
    if not expected_names or len(expected_names) >= _ZIP64_U16:
        raise ValueError("ZIP preflight roster is outside the classic ZIP bound")
    try:
        encoded_names = tuple(name.encode("ascii") for name in expected_names)
    except (AttributeError, UnicodeEncodeError) as exc:
        raise ValueError("ZIP preflight roster must contain ASCII names") from exc
    if any(not name or b"\x00" in name for name in encoded_names):
        raise ValueError("ZIP preflight roster contains an invalid name")

    if len(raw) < _EOCD.size:
        raise ValueError("ZIP bytes are too short for an end record")
    eocd_offset = len(raw) - _EOCD.size
    (
        signature,
        disk_number,
        central_disk,
        entries_on_disk,
        entry_count,
        central_size,
        central_offset,
        comment_size,
    ) = _EOCD.unpack_from(raw, eocd_offset)
    expected_count = len(encoded_names)
    expected_central_size = sum(
        _CENTRAL_HEADER.size + len(name) for name in encoded_names
    )
    if (
        signature != _EOCD_SIGNATURE
        or disk_number != 0
        or central_disk != 0
        or entries_on_disk != expected_count
        or entry_count != expected_count
        or entry_count == _ZIP64_U16
        or central_size == _ZIP64_U32
        or central_offset == _ZIP64_U32
        or comment_size != 0
        or central_size != expected_central_size
        or central_offset + central_size != eocd_offset
    ):
        raise ValueError("ZIP end record does not describe the exact fixed roster")

    cursor = central_offset
    for expected_name in encoded_names:
        if cursor + _CENTRAL_HEADER.size > eocd_offset:
            raise ValueError("ZIP central directory is truncated")
        fields = _CENTRAL_HEADER.unpack_from(raw, cursor)
        (
            member_signature,
            _version_made_by,
            _version_needed,
            _flags,
            compression,
            _modified_time,
            _modified_date,
            _crc32,
            compressed_size,
            uncompressed_size,
            name_size,
            extra_size,
            member_comment_size,
            member_disk,
            _internal_attributes,
            _external_attributes,
            local_header_offset,
        ) = fields
        cursor += _CENTRAL_HEADER.size
        name_end = cursor + name_size
        if (
            member_signature != _CENTRAL_SIGNATURE
            or compression != 0
            or compressed_size == _ZIP64_U32
            or uncompressed_size == _ZIP64_U32
            or local_header_offset == _ZIP64_U32
            or extra_size != 0
            or member_comment_size != 0
            or member_disk != 0
            or name_size != len(expected_name)
            or name_end > eocd_offset
            or raw[cursor:name_end] != expected_name
        ):
            raise ValueError("ZIP central directory is not the exact fixed roster")
        cursor = name_end
    if cursor != eocd_offset:
        raise ValueError("ZIP central directory has trailing data")


__all__ = ["preflight_exact_zip_roster"]
