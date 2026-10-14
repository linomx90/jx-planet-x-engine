from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "tools"
    / "jx_tno_common_epoch_horizons_oracle_acquisition_v1.py"
)


def _load_source_module() -> types.ModuleType:
    name = "jx_tno_common_epoch_horizons_oracle_acquisition_v1_tested_source"
    payload = SCRIPT_PATH.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(SCRIPT_PATH)
    sys.modules[name] = module
    exec(compile(payload, str(SCRIPT_PATH), "exec"), module.__dict__)
    return module


ACQUISITION = _load_source_module()
PREREG = ACQUISITION._prereg_module()


def _response(target: object, vector_seed: int | None = None) -> bytes:
    seed = target.order + 1 if vector_seed is None else vector_seed
    elements = target.element_tokens
    vector = (
        f"{seed}.125",
        f"-{seed}.25",
        f"{seed}.5",
        f"0.00{seed}1",
        f"-0.00{seed}2",
        f"0.00{seed}3",
    )
    result = "\n".join(
        (
            f"Target body name: {target.full_name} {{source: JPL#{target.solution_id}}}",
            "Center body name: Sun (10) {source: DE441}",
            "Small perturbers: Yes {source: SB441-N16}",
            " EPOCH= 2461200.5 ! synthetic epoch (TDB)",
            f" EC= {elements[0]} QR= {elements[1]} TP= {elements[2]}",
            f" OM= {elements[3]} W= {elements[4]} IN= {elements[5]}",
            "Output units    : AU-D",
            "Output type     : GEOMETRIC cartesian states",
            "Reference frame : Ecliptic of J2000.0",
            "Initial IAU76/J2000 heliocentric ecliptic osculating elements",
            "Geometric state vectors have NO corrections or aberrations applied.",
            'Barycentric Dynamical Time ("TDB" or T_eph) output was requested.',
            "JDTDB, Calendar Date (TDB), X, Y, Z, VX, VY, VZ,",
            "$$SOE",
            "2461200.5, A.D. 2026-Jun-09 00:00:00.0000, "
            + ", ".join(vector)
            + ",",
            "$$EOE",
        )
    )
    return json.dumps(
        {
            "signature": {
                "source": "NASA/JPL Horizons API",
                "version": "1.2",
            },
            "result": result,
        },
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _payloads() -> dict[int, bytes]:
    return {target.spk_id: _response(target) for target in PREREG.TARGETS}


class _FakeSocket:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)


class _FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        status: int = 200,
        headers: list[tuple[str, str]] | None = None,
        oversized_read: bool = False,
    ) -> None:
        self.payload = payload
        self.status = status
        self.headers = [] if headers is None else headers
        self.offset = 0
        self.read_sizes: list[int] = []
        self.closed = False
        self.oversized_read = oversized_read

    def getheaders(self) -> list[tuple[str, str]]:
        return list(self.headers)

    def read(self, amount: int) -> bytes:
        self.read_sizes.append(amount)
        if self.oversized_read and self.offset == 0:
            self.offset = len(self.payload)
            return self.payload
        block = self.payload[self.offset : self.offset + amount]
        self.offset += len(block)
        return block

    def read1(self, amount: int) -> bytes:
        return self.read(amount)

    def close(self) -> None:
        self.closed = True


class _FakeConnection:
    def __init__(self, response: _FakeResponse, *args: object, **kwargs: object) -> None:
        self.response = response
        self.constructor_args = args
        self.constructor_kwargs = kwargs
        self.sock = _FakeSocket()
        self.connect_count = 0
        self.request_calls: list[tuple[str, str, bool]] = []
        self.headers: list[tuple[str, str]] = []
        self.endheaders_count = 0
        self.getresponse_count = 0
        self.close_count = 0

    def connect(self) -> None:
        self.connect_count += 1

    def putrequest(
        self, method: str, target: str, *, skip_accept_encoding: bool = False
    ) -> None:
        self.request_calls.append((method, target, skip_accept_encoding))

    def putheader(self, key: str, value: str) -> None:
        self.headers.append((key, value))

    def endheaders(self) -> None:
        self.endheaders_count += 1

    def getresponse(self) -> _FakeResponse:
        self.getresponse_count += 1
        return self.response

    def close(self) -> None:
        self.close_count += 1


def _connection_factory(
    response: _FakeResponse,
) -> tuple[object, list[_FakeConnection]]:
    instances: list[_FakeConnection] = []

    def factory(*args: object, **kwargs: object) -> _FakeConnection:
        connection = _FakeConnection(response, *args, **kwargs)
        instances.append(connection)
        return connection

    return factory, instances


def _timestamps(root: Path) -> dict[str, tuple[int, int, int]]:
    paths = [root, root / "raw", root / "CUSTODY.json"] + [
        root / "raw" / name for name in PREREG.RESULT_RAW_ROSTER
    ]
    result: dict[str, tuple[int, int, int]] = {}
    for path in paths:
        metadata = path.stat()
        result[str(path.relative_to(root.parent))] = (
            metadata.st_atime_ns,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
    return result


class TnoCommonEpochHorizonsOracleAcquisitionV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payloads = _payloads()

    def _authorized(self) -> mock._patch:
        return mock.patch.object(
            ACQUISITION,
            "_verify_execution_authority",
            return_value=(
                ACQUISITION._preregistration_identity(),
                PREREG,
                {"commit": "test-only-committed-successor"},
            ),
        )

    def _acquire(self, output: Path) -> dict[str, object]:
        with self._authorized(), mock.patch.object(
            ACQUISITION,
            "_fetch",
            side_effect=lambda url: self.payloads[_spk_from_url(url)],
        ):
            return ACQUISITION._acquire_to_root(output)

    def test_preregistration_identity_is_exact_before_request_zero(self) -> None:
        identity, module = ACQUISITION._verify_preregistration()
        self.assertEqual(len(module.TARGETS), 12)
        self.assertEqual(
            identity,
            {
                "commit": "3a2df6013e60efc54e4ab8ef566fbd4d80370fa9",
                "protocol_content_sha256": (
                    "06aee564cdc8c41be671129da96ed9a531423f68238de972c78533c32d1036cc"
                ),
                "protocol_file_bytes": 52597,
                "protocol_file_sha256": (
                    "69b445578e38ae5d4605b78f5d4a83d14bdf6ba28a9bab1709b63ecf44c1f5a8"
                ),
                "source_bytes": 74892,
                "source_sha256": (
                    "0fce70c8f8eeaad60e35db53c64aa446503b48c6e4b39648df369655d11147d0"
                ),
                "tag": "tno-common-epoch-horizons-oracle-v1-preregistered-20261014",
                "tag_object": "cb5bc0da8bd761d9ca400d9021937dd7b795bcac",
                "tag_payload_bytes": 254,
                "tag_payload_sha256": (
                    "afb257367709d9dafc18a09630d001c8a7ec3a08d4d66f2341e634dc7eaf3ec4"
                ),
                "test_bytes": 28052,
                "test_sha256": (
                    "cff7e39e2833e1bec388d3d1d6d791f10063b7bc348618e5f12f66dc2cd86f1c"
                ),
                "tree": "a1dd4b80d38668f58a5c2bb0c346eb75cdacfd5e",
            },
        )
        self.assertEqual(ACQUISITION.PREREG_MANIFEST_BYTES, 13067)
        self.assertEqual(ACQUISITION.PREREG_MANIFEST_SHA256, (
            "eccd3a1173b4304978cb9eee261eba4d182e792b4f284cff9249795dc2983187"
        ))

        fetcher = mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            with mock.patch.object(
                ACQUISITION, "_fetch", fetcher
            ), mock.patch.object(ACQUISITION, "_git", return_value=b"not-a-tag\n"):
                with self.assertRaisesRegex(
                    ACQUISITION.AcquisitionError, "annotated tag"
                ):
                    ACQUISITION._acquire_to_root(output)
            fetcher.assert_not_called()
            self.assertFalse(output.exists())

    def test_dirty_or_untracked_successor_stops_before_request_zero(self) -> None:
        identity = ACQUISITION._preregistration_identity()
        for status in (
            b" M tools/jx_tno_common_epoch_horizons_oracle_acquisition_v1.py\n",
            b"?? hostile-untracked\n",
        ):
            with self.subTest(status=status):
                fetcher = mock.Mock()
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "result"
                    with mock.patch.object(
                        ACQUISITION,
                        "_verify_preregistration",
                        return_value=(identity, PREREG),
                    ), mock.patch.object(
                        ACQUISITION, "_git", return_value=status
                    ), mock.patch.object(
                        ACQUISITION, "_fetch", fetcher
                    ):
                        with self.assertRaisesRegex(
                            ACQUISITION.AcquisitionError, "worktree is not clean"
                        ):
                            ACQUISITION._acquire_to_root(output)
                    fetcher.assert_not_called()
                    self.assertFalse(output.exists())

    def test_successor_parent_changeset_and_committed_bytes_are_authority_gates(
        self,
    ) -> None:
        tree = "3" * 40

        def raw_commit(parent: str) -> bytes:
            return (
                f"tree {tree}\n"
                f"parent {parent}\n"
                "author Test <test@example.com> 0 +0000\n"
                "committer Test <test@example.com> 0 +0000\n"
                "\n"
                "successor fixture\n"
            ).encode("ascii")

        commit = raw_commit(ACQUISITION.PREREG_COMMIT)
        head = hashlib.sha1(
            b"commit " + str(len(commit)).encode("ascii") + b"\0" + commit
        ).hexdigest()
        real_git = ACQUISITION._git
        exact_changes = (
            b"A\ttests/test_tno_common_epoch_horizons_oracle_acquisition_v1.py\n"
            b"A\ttools/jx_tno_common_epoch_horizons_oracle_acquisition_v1.py\n"
            b"M\ttools/test_matrix_manifest.json\n"
        )

        def fake_git(*arguments: str) -> bytes:
            if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
                return b""
            if arguments == ("symbolic-ref", "--quiet", "--short", "HEAD"):
                return (ACQUISITION.SUCCESSOR_BRANCH + "\n").encode("ascii")
            if arguments == ("rev-parse", "--verify", "HEAD^{commit}"):
                return (head + "\n").encode("ascii")
            if arguments == ("cat-file", "commit", head):
                return commit
            if arguments == (
                "diff-tree",
                "--no-commit-id",
                "--name-status",
                "--no-renames",
                "--no-ext-diff",
                "--no-textconv",
                "-r",
                ACQUISITION.PREREG_COMMIT,
                head,
            ):
                return exact_changes
            if arguments == (
                "cat-file",
                "blob",
                f"{ACQUISITION.PREREG_COMMIT}:{ACQUISITION.PREREG_MANIFEST_RELATIVE_PATH}",
            ):
                return real_git(*arguments)
            if arguments == (
                "cat-file",
                "blob",
                f"{head}:{ACQUISITION.SUCCESSOR_SOURCE_RELATIVE_PATH}",
            ):
                return b"committed-bytes-differ\n"
            raise AssertionError(f"unexpected git call: {arguments!r}")

        wrong_commit = raw_commit("2" * 40)
        wrong_head = hashlib.sha1(
            b"commit "
            + str(len(wrong_commit)).encode("ascii")
            + b"\0"
            + wrong_commit
        ).hexdigest()

        def wrong_parent_git(*arguments: str) -> bytes:
            if arguments == ("rev-parse", "--verify", "HEAD^{commit}"):
                return (wrong_head + "\n").encode("ascii")
            if arguments == ("cat-file", "commit", wrong_head):
                return wrong_commit
            return fake_git(*arguments)

        with mock.patch.object(ACQUISITION, "_git", side_effect=wrong_parent_git):
            with self.assertRaisesRegex(
                ACQUISITION.AcquisitionError, "direct single-parent child"
            ):
                ACQUISITION._verify_successor_commit()

        def extra_change_git(*arguments: str) -> bytes:
            if arguments[:1] == ("diff-tree",):
                return exact_changes + b"M\tREADME.md\n"
            return fake_git(*arguments)

        with mock.patch.object(ACQUISITION, "_git", side_effect=extra_change_git):
            with self.assertRaisesRegex(
                ACQUISITION.AcquisitionError, "changeset differs"
            ):
                ACQUISITION._verify_successor_commit()

        with mock.patch.object(ACQUISITION, "_git", side_effect=fake_git):
            with self.assertRaisesRegex(
                ACQUISITION.AcquisitionError, "held worktree bytes differ"
            ):
                ACQUISITION._verify_successor_commit()

    def test_exact_query_roster_and_response_custody_contract(self) -> None:
        protocol = json.loads(ACQUISITION.PREREG_PROTOCOL_PATH.read_bytes())
        frozen_urls = protocol["future_transport_contract"]["ordered_exact_urls"]
        self.assertEqual([target.query_url for target in PREREG.TARGETS], frozen_urls)
        self.assertEqual(
            [target.solution_id for target in PREREG.TARGETS],
            [51, 12, 3, 7, 11, 5, 14, 29, 25, 28, 38, 34],
        )
        custody = ACQUISITION.build_custody_from_payloads(dict(self.payloads))
        self.assertEqual(tuple(custody), PREREG.CUSTODY_TOP_LEVEL_KEYS)
        self.assertEqual(custody["schema"], ACQUISITION.CUSTODY_SCHEMA)
        self.assertEqual(custody["status"], ACQUISITION.CUSTODY_STATUS)
        self.assertEqual(custody["record_count"], 12)
        self.assertEqual(custody["claims"], PREREG.CUSTODY_CLAIMS)
        self.assertEqual(custody["nonclaims"], PREREG.CUSTODY_NONCLAIMS)
        for target, record in zip(PREREG.TARGETS, custody["records"], strict=True):
            self.assertEqual(tuple(record), PREREG.CUSTODY_RECORD_KEYS)
            self.assertEqual(record["full_name"], target.full_name)
            self.assertEqual(record["solution_id"], target.solution_id)
            self.assertEqual(record["planetary_ephemeris"], "DE441")
            self.assertEqual(record["small_body_perturbers"], "SB441-N16")
            self.assertLessEqual(record["tp_binary64_ulp_distance"], 1)
        without_content = dict(custody)
        without_content.pop("content_sha256")
        self.assertEqual(
            custody["content_sha256"],
            hashlib.sha256(
                ACQUISITION.CUSTODY_DOMAIN
                + ACQUISITION._canonical_json(without_content)
            ).hexdigest(),
        )
        self.assertEqual(
            custody["ordered_vector_sha256"],
            PREREG.ordered_vector_sha256(custody["records"]),
        )

    def test_response_gate_rejects_extra_key_oversize_and_bad_identity(self) -> None:
        target = PREREG.TARGETS[0]
        payload = self.payloads[target.spk_id]
        boundary = payload + b" " * (ACQUISITION.MAX_RESPONSE_BYTES - len(payload))
        boundary_record = ACQUISITION._response_record(boundary, target, PREREG)
        self.assertEqual(boundary_record["raw_bytes"], ACQUISITION.MAX_RESPONSE_BYTES)
        value = json.loads(payload)
        value["hostile"] = True
        with self.assertRaisesRegex(ACQUISITION.AcquisitionError, "top-level"):
            ACQUISITION._response_record(
                json.dumps(value).encode("utf-8"), target, PREREG
            )
        with self.assertRaisesRegex(ACQUISITION.AcquisitionError, "1048576-byte"):
            ACQUISITION._response_record(
                b" " * (ACQUISITION.MAX_RESPONSE_BYTES + 1), target, PREREG
            )
        value = json.loads(payload)
        value["result"] = value["result"].replace(target.full_name, "wrong", 1)
        with self.assertRaisesRegex(ACQUISITION.AcquisitionError, "full name"):
            ACQUISITION._response_record(
                json.dumps(value).encode("utf-8"), target, PREREG
            )

    def test_direct_transport_is_one_exact_get_with_frozen_headers(self) -> None:
        target = PREREG.TARGETS[0]
        payload = self.payloads[target.spk_id]
        response = _FakeResponse(
            payload,
            headers=[
                ("Content-Encoding", "identity"),
                ("Content-Length", str(len(payload))),
            ],
        )
        factory, instances = _connection_factory(response)
        observed = ACQUISITION._fetch(
            target.query_url, connection_factory=factory, clock=lambda: 0.0
        )
        self.assertEqual(observed, payload)
        self.assertEqual(len(instances), 1)
        connection = instances[0]
        parsed = urlsplit(target.query_url)
        self.assertEqual(connection.connect_count, 1)
        self.assertIs(connection.auto_open, False)
        self.assertEqual(
            connection.request_calls,
            [("GET", parsed.path + "?" + parsed.query, True)],
        )
        self.assertEqual(connection.headers, list(ACQUISITION.REQUEST_HEADERS))
        self.assertEqual(connection.endheaders_count, 1)
        self.assertEqual(connection.getresponse_count, 1)
        self.assertEqual(connection.close_count, 1)
        self.assertTrue(response.closed)
        self.assertGreaterEqual(len(response.read_sizes), 2)
        self.assertTrue(all(size <= 65_536 for size in response.read_sizes))

    def test_transport_rejects_redirect_encoding_and_bad_lengths_without_retry(self) -> None:
        target = PREREG.TARGETS[0]
        cases = (
            (
                _FakeResponse(b"", status=302, headers=[("Location", target.query_url)]),
                "HTTP status 302",
            ),
            (
                _FakeResponse(b"{}", headers=[("Content-Encoding", "gzip")]),
                "Content-Encoding",
            ),
            (
                _FakeResponse(b"{}", headers=[("Transfer-Encoding", "gzip")]),
                "Transfer-Encoding",
            ),
            (
                _FakeResponse(
                    b"{}",
                    headers=[("Content-Length", "2"), ("Content-Length", "2")],
                ),
                "duplicate or malformed",
            ),
            (
                _FakeResponse(b"{}", headers=[("Content-Length", "+2")]),
                "duplicate or malformed",
            ),
            (
                _FakeResponse(b"{}", headers=[("Content-Length", "3")]),
                "differs from entity",
            ),
        )
        for response, message in cases:
            with self.subTest(message=message):
                factory, instances = _connection_factory(response)
                with self.assertRaisesRegex(ACQUISITION.AcquisitionError, message):
                    ACQUISITION._fetch(
                        target.query_url,
                        connection_factory=factory,
                        clock=lambda: 0.0,
                    )
                self.assertEqual(len(instances), 1)
                self.assertEqual(instances[0].connect_count, 1)
                self.assertEqual(instances[0].getresponse_count, 1)

    def test_remote_huge_content_length_is_bounded_and_cleans_staging(self) -> None:
        response = _FakeResponse(
            b"",
            headers=[("Content-Length", "9" * 5_000)],
        )
        factory, instances = _connection_factory(response)
        original_fetch = ACQUISITION._fetch

        def fetch_with_hostile_header(url: str) -> bytes:
            return original_fetch(url, connection_factory=factory)

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "result"
            with self._authorized(), mock.patch.object(
                ACQUISITION,
                "_fetch",
                side_effect=fetch_with_hostile_header,
            ):
                with self.assertRaisesRegex(
                    ACQUISITION.AcquisitionError,
                    "Content-Length exceeds the byte ceiling",
                ):
                    ACQUISITION._acquire_to_root(output)
            self.assertEqual(len(instances), 1)
            self.assertEqual(instances[0].connect_count, 1)
            self.assertEqual(instances[0].getresponse_count, 1)
            self.assertTrue(response.closed)
            self.assertEqual(list(parent.iterdir()), [])

    def test_transport_streaming_cap_boundary_and_cap_plus_one(self) -> None:
        target = PREREG.TARGETS[0]
        boundary = b"x" * ACQUISITION.MAX_RESPONSE_BYTES
        response = _FakeResponse(
            boundary, headers=[("Content-Length", str(len(boundary)))]
        )
        factory, instances = _connection_factory(response)
        self.assertEqual(
            ACQUISITION._fetch(
                target.query_url, connection_factory=factory, clock=lambda: 0.0
            ),
            boundary,
        )
        self.assertEqual(len(instances), 1)
        self.assertEqual(max(response.read_sizes), ACQUISITION.STREAM_CHUNK_BYTES)
        self.assertEqual(response.read_sizes[-1], 1)

        oversized = b"x" * (ACQUISITION.MAX_RESPONSE_BYTES + 1)
        response = _FakeResponse(oversized)
        factory, instances = _connection_factory(response)
        with self.assertRaisesRegex(ACQUISITION.AcquisitionError, "1048576-byte"):
            ACQUISITION._fetch(
                target.query_url, connection_factory=factory, clock=lambda: 0.0
            )
        self.assertEqual(len(instances), 1)
        self.assertLessEqual(sum(response.read_sizes), len(oversized))
        self.assertEqual(response.offset, ACQUISITION.MAX_RESPONSE_BYTES + 1)

    def test_transport_rejects_oversized_single_read_and_timeout(self) -> None:
        target = PREREG.TARGETS[0]
        response = _FakeResponse(
            b"x" * (ACQUISITION.STREAM_CHUNK_BYTES + 1), oversized_read=True
        )
        factory, _instances = _connection_factory(response)
        with self.assertRaisesRegex(ACQUISITION.AcquisitionError, "requested bound"):
            ACQUISITION._fetch(
                target.query_url, connection_factory=factory, clock=lambda: 0.0
            )

        ticks = iter((0.0, 0.0, 0.0, 121.0))
        response = _FakeResponse(b"{}")
        factory, instances = _connection_factory(response)
        with self.assertRaisesRegex(ACQUISITION.AcquisitionError, "total timeout"):
            ACQUISITION._fetch(
                target.query_url,
                connection_factory=factory,
                clock=lambda: next(ticks),
            )
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].connect_count, 1)

    def test_tls_context_setup_is_inside_the_total_deadline(self) -> None:
        release = threading.Event()
        context_started = threading.Event()
        factory = mock.Mock(side_effect=AssertionError("connection factory ran late"))

        def stalled_context() -> object:
            context_started.set()
            release.wait(1.0)
            return object()

        started = time.monotonic()
        try:
            with mock.patch.object(
                ACQUISITION.ssl,
                "create_default_context",
                side_effect=stalled_context,
            ), mock.patch.object(ACQUISITION, "TOTAL_TIMEOUT_SECONDS", 0.01):
                with self.assertRaisesRegex(
                    ACQUISITION.AcquisitionError,
                    "TLS setup exceeded the total timeout",
                ):
                    ACQUISITION._fetch(
                        PREREG.TARGETS[0].query_url,
                        connection_factory=factory,
                    )
        finally:
            release.set()
        self.assertTrue(context_started.is_set())
        self.assertLess(time.monotonic() - started, 0.5)
        time.sleep(0.01)
        factory.assert_not_called()

    def test_dns_resolution_is_bounded_before_any_connection_attempt(self) -> None:
        release = threading.Event()
        calls: list[tuple[object, ...]] = []

        def stalled_resolver(*arguments: object) -> list[tuple[object, ...]]:
            calls.append(arguments)
            release.wait(1.0)
            return []

        started = time.monotonic()
        try:
            with self.assertRaisesRegex(
                ACQUISITION.AcquisitionError,
                "DNS resolution exceeded the connect timeout",
            ):
                ACQUISITION._bounded_resolve(
                    ACQUISITION.EXPECTED_HOST,
                    443,
                    0.01,
                    resolver=stalled_resolver,
                )
        finally:
            release.set()
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(
            calls,
            [
                (
                    ACQUISITION.EXPECTED_HOST,
                    443,
                    ACQUISITION.socket.AF_UNSPEC,
                    ACQUISITION.socket.SOCK_STREAM,
                    ACQUISITION.socket.IPPROTO_TCP,
                )
            ],
        )

        class StalledConnection:
            def __init__(self) -> None:
                self.closed = 0

            def connect(self) -> None:
                release.clear()
                release.wait(1.0)

            def close(self) -> None:
                self.closed += 1

        connection = StalledConnection()
        started = time.monotonic()
        try:
            with self.assertRaisesRegex(
                ACQUISITION.AcquisitionError,
                "connection exceeded the connect timeout",
            ):
                ACQUISITION._bounded_connection_setup(connection, 0.01)
        finally:
            release.set()
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertGreaterEqual(connection.closed, 1)

    def test_acquire_is_single_pass_custody_last_and_atomic_noreplace(self) -> None:
        calls: list[str] = []
        rename_observations: list[tuple[str, set[str]]] = []
        original_rename = ACQUISITION._rename_noreplace

        def fetcher(url: str) -> bytes:
            calls.append(url)
            return self.payloads[_spk_from_url(url)]

        def observing_rename(
            source_fd: int, source: str, destination_fd: int, destination: str
        ) -> None:
            if destination == "CUSTODY.json":
                roster = set(os.listdir(source_fd))
            else:
                staged_fd = os.open(
                    source,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                    dir_fd=source_fd,
                )
                try:
                    roster = set(os.listdir(staged_fd))
                finally:
                    os.close(staged_fd)
            rename_observations.append((destination, roster))
            original_rename(source_fd, source, destination_fd, destination)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            with mock.patch.object(
                ACQUISITION, "_rename_noreplace", side_effect=observing_rename
            ), self._authorized(), mock.patch.object(
                ACQUISITION, "_fetch", side_effect=fetcher
            ):
                custody = ACQUISITION._acquire_to_root(output)
            self.assertEqual(calls, [target.query_url for target in PREREG.TARGETS])
            self.assertEqual(
                rename_observations[0],
                ("CUSTODY.json", {"raw", ".CUSTODY.json.partial"}),
            )
            self.assertEqual(
                rename_observations[1],
                ("result", {"raw", "CUSTODY.json"}),
            )
            self.assertEqual(custody, ACQUISITION.validate_result(output))
            self.assertEqual(set(os.listdir(output)), {"raw", "CUSTODY.json"})
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE((output / "raw").stat().st_mode), 0o700)
            for path in [output / "CUSTODY.json", *(output / "raw").iterdir()]:
                metadata = path.stat()
                self.assertTrue(stat.S_ISREG(metadata.st_mode))
                self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o600)
                self.assertEqual(metadata.st_nlink, 1)
                self.assertEqual(metadata.st_uid, os.geteuid())

    def test_failed_response_stops_sequence_and_leaves_no_result_or_stage(self) -> None:
        calls: list[str] = []
        fail_at = 4

        def fetcher(url: str) -> bytes:
            calls.append(url)
            if len(calls) == fail_at:
                return b'{"signature":{},"result":"bad"}'
            return self.payloads[_spk_from_url(url)]

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "result"
            with self._authorized(), mock.patch.object(
                ACQUISITION, "_fetch", side_effect=fetcher
            ):
                with self.assertRaises(ACQUISITION.AcquisitionError):
                    ACQUISITION._acquire_to_root(output)
            self.assertEqual(
                calls,
                [target.query_url for target in PREREG.TARGETS[:fail_at]],
            )
            self.assertFalse(output.exists())
            self.assertEqual(list(parent.iterdir()), [])

    def test_unwrapped_sealed_parser_error_fails_closed_and_cleans_stage(self) -> None:
        calls: list[str] = []
        hostile_target = PREREG.TARGETS[1]
        hostile_value = json.loads(self.payloads[hostile_target.spk_id])
        hostile_value["result"] = (
            hostile_value["result"]
            .replace("$$SOE", "$$TEMP")
            .replace("$$EOE", "$$SOE")
            .replace("$$TEMP", "$$EOE")
        )
        hostile_payload = json.dumps(
            hostile_value,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")

        def fetcher(url: str) -> bytes:
            calls.append(url)
            if len(calls) == 2:
                return hostile_payload
            return self.payloads[_spk_from_url(url)]

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "result"
            with self._authorized(), mock.patch.object(
                ACQUISITION, "_fetch", side_effect=fetcher
            ):
                with self.assertRaisesRegex(
                    ACQUISITION.AcquisitionError,
                    "response validator failed closed",
                ):
                    ACQUISITION._acquire_to_root(output)
            self.assertEqual(
                calls,
                [target.query_url for target in PREREG.TARGETS[:2]],
            )
            self.assertFalse(output.exists())
            self.assertEqual(list(parent.iterdir()), [])

    def test_publication_fault_removes_stage_and_existing_root_is_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "result"
            with mock.patch.object(
                ACQUISITION,
                "_rename_noreplace",
                side_effect=ACQUISITION.AcquisitionError("injected rename fault"),
            ):
                with self.assertRaisesRegex(
                    ACQUISITION.AcquisitionError, "injected rename fault"
                ):
                    self._acquire(output)
            self.assertFalse(output.exists())
            self.assertEqual(list(parent.iterdir()), [])

            output.mkdir(mode=0o700)
            sentinel = output / "sentinel"
            sentinel.write_bytes(b"owned-by-racer")
            fetcher = mock.Mock()
            with self._authorized(), mock.patch.object(
                ACQUISITION, "_fetch", fetcher
            ):
                with self.assertRaisesRegex(
                    ACQUISITION.AcquisitionError, "already exists"
                ):
                    ACQUISITION._acquire_to_root(output)
            fetcher.assert_not_called()
            self.assertEqual(sentinel.read_bytes(), b"owned-by-racer")

    def test_final_rename_race_preserves_competing_root_and_cleans_stage(self) -> None:
        original_rename = ACQUISITION._rename_noreplace
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "result"

            def racing_rename(
                source_fd: int, source: str, destination_fd: int, destination: str
            ) -> None:
                if destination == "CUSTODY.json":
                    original_rename(source_fd, source, destination_fd, destination)
                    return
                output.mkdir(mode=0o700)
                (output / "sentinel").write_bytes(b"racer")
                raise ACQUISITION.AcquisitionError(
                    "create-only publication destination already exists"
                )

            with mock.patch.object(
                ACQUISITION, "_rename_noreplace", side_effect=racing_rename
            ):
                with self.assertRaisesRegex(
                    ACQUISITION.AcquisitionError, "already exists"
                ):
                    self._acquire(output)
            self.assertEqual((output / "sentinel").read_bytes(), b"racer")
            self.assertEqual(
                [path.name for path in parent.iterdir()],
                ["result"],
            )

    def test_held_leaf_close_error_cannot_skip_prepublication_cleanup(self) -> None:
        original_rename = ACQUISITION._rename_noreplace
        original_close = ACQUISITION.os.close
        final_rename_failed = False
        held_close_failed = False

        def failing_final_rename(
            source_fd: int, source: str, destination_fd: int, destination: str
        ) -> None:
            nonlocal final_rename_failed
            if destination == "CUSTODY.json":
                original_rename(source_fd, source, destination_fd, destination)
                return
            final_rename_failed = True
            raise ACQUISITION.AcquisitionError("injected final rename fault")

        def failing_held_close(descriptor: int) -> None:
            nonlocal held_close_failed
            try:
                is_regular = stat.S_ISREG(os.fstat(descriptor).st_mode)
            except OSError:
                is_regular = False
            if final_rename_failed and not held_close_failed and is_regular:
                held_close_failed = True
                original_close(descriptor)
                raise OSError("injected held-leaf close fault")
            original_close(descriptor)

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "result"
            with self._authorized(), mock.patch.object(
                ACQUISITION,
                "_fetch",
                side_effect=lambda url: self.payloads[_spk_from_url(url)],
            ), mock.patch.object(
                ACQUISITION,
                "_rename_noreplace",
                side_effect=failing_final_rename,
            ), mock.patch.object(
                ACQUISITION.os, "close", side_effect=failing_held_close
            ):
                with self.assertRaisesRegex(
                    ACQUISITION.AcquisitionError, "final rename fault"
                ):
                    ACQUISITION._acquire_to_root(output)
            self.assertTrue(final_rename_failed)
            self.assertTrue(held_close_failed)
            self.assertEqual(list(parent.iterdir()), [])

    def test_staging_dirent_swap_is_rejected_and_replacement_is_never_removed(
        self,
    ) -> None:
        original_validate = ACQUISITION._validate_result_fds
        observed: dict[str, Path] = {}

        def swap_after_validation(
            root_fd: int,
            root_identity: os.stat_result,
            raw_fd: int,
            raw_identity: os.stat_result,
            module: types.ModuleType,
        ) -> tuple[dict[str, object], list[tuple[object, ...]]]:
            custody, held = original_validate(
                root_fd, root_identity, raw_fd, raw_identity, module
            )
            path = next(observed["parent"].glob(".result.staging-*"))
            moved = path.with_name(path.name + ".held-original")
            path.rename(moved)
            path.mkdir(mode=0o700)
            path.chmod(0o700)
            (path / "replacement-sentinel").write_bytes(b"do-not-remove")
            observed.update(replacement=path, original=moved)
            return custody, held

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            observed["parent"] = output.parent
            with mock.patch.object(
                ACQUISITION,
                "_validate_result_fds",
                side_effect=swap_after_validation,
            ):
                with self.assertRaisesRegex(
                    ACQUISITION.AcquisitionError,
                    "dirent no longer identifies the held directory",
                ):
                    self._acquire(output)
            self.assertFalse(output.exists())
            self.assertEqual(
                (observed["replacement"] / "replacement-sentinel").read_bytes(),
                b"do-not-remove",
            )
            self.assertEqual(
                set(os.listdir(observed["original"])), {"raw", "CUSTODY.json"}
            )

    def test_raw_dirent_swap_is_rejected_and_replacement_is_never_removed(
        self,
    ) -> None:
        original_validate = ACQUISITION._validate_result_fds
        observed: dict[str, Path] = {}

        def swap_raw_after_validation(
            root_fd: int,
            root_identity: os.stat_result,
            raw_fd: int,
            raw_identity: os.stat_result,
            module: types.ModuleType,
        ) -> tuple[dict[str, object], list[tuple[object, ...]]]:
            custody, held = original_validate(
                root_fd, root_identity, raw_fd, raw_identity, module
            )
            path = next(observed["parent"].glob(".result.staging-*"))
            raw = path / "raw"
            moved = path / "raw-held-original"
            raw.rename(moved)
            raw.mkdir(mode=0o700)
            raw.chmod(0o700)
            (raw / "replacement-sentinel").write_bytes(b"do-not-remove")
            observed.update(replacement=raw, original=moved, stage=path)
            return custody, held

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            observed["parent"] = output.parent
            with mock.patch.object(
                ACQUISITION,
                "_validate_result_fds",
                side_effect=swap_raw_after_validation,
            ):
                with self.assertRaisesRegex(
                    ACQUISITION.AcquisitionError,
                    "dirent no longer identifies the held directory",
                ):
                    self._acquire(output)
            self.assertFalse(output.exists())
            self.assertEqual(
                (observed["replacement"] / "replacement-sentinel").read_bytes(),
                b"do-not-remove",
            )
            self.assertEqual(
                set(os.listdir(observed["original"])), set(PREREG.RESULT_RAW_ROSTER)
            )
            self.assertTrue(observed["stage"].is_dir())

    def test_leaf_swap_after_replay_is_rejected_without_deleting_replacement(
        self,
    ) -> None:
        original_validate = ACQUISITION._validate_result_fds
        observed: dict[str, object] = {}

        def swap_leaf_after_validation(
            root_fd: int,
            root_identity: os.stat_result,
            raw_fd: int,
            raw_identity: os.stat_result,
            module: types.ModuleType,
        ) -> tuple[dict[str, object], list[tuple[object, ...]]]:
            custody, held = original_validate(
                root_fd, root_identity, raw_fd, raw_identity, module
            )
            parent = observed["parent"]
            self.assertIsInstance(parent, Path)
            stage = next(parent.glob(".result.staging-*"))
            target = stage / "raw" / PREREG.RESULT_RAW_ROSTER[0]
            displaced = parent / "displaced-original.json"
            target.rename(displaced)
            valuable = parent / "valuable-sentinel.json"
            valuable.write_bytes(b"valuable-replacement-must-survive")
            valuable.chmod(0o600)
            valuable_inode = valuable.stat().st_ino
            valuable.rename(target)
            observed.update(
                stage=stage,
                replacement=target,
                replacement_inode=valuable_inode,
            )
            return custody, held

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            observed["parent"] = parent
            output = parent / "result"
            with mock.patch.object(
                ACQUISITION,
                "_validate_result_fds",
                side_effect=swap_leaf_after_validation,
            ):
                with self.assertRaisesRegex(
                    ACQUISITION.AcquisitionError,
                    "refused replaced or unsafe raw cleanup entry",
                ):
                    self._acquire(output)
            self.assertFalse(output.exists())
            replacement = observed["replacement"]
            self.assertIsInstance(replacement, Path)
            self.assertEqual(
                replacement.read_bytes(), b"valuable-replacement-must-survive"
            )
            self.assertEqual(replacement.stat().st_ino, observed["replacement_inode"])
            self.assertTrue(observed["stage"].is_dir())

    def test_result_parent_chain_swap_fails_acquisition_and_offline_replay(
        self,
    ) -> None:
        original_validate = ACQUISITION._validate_result_fds
        with tempfile.TemporaryDirectory() as directory:
            container = Path(directory)
            parent = container / "parent"
            parent.mkdir(mode=0o700)
            output = parent / "result"
            moved_parent = container / "parent-held-original"

            def swap_parent_after_staged_validation(
                root_fd: int,
                root_identity: os.stat_result,
                raw_fd: int,
                raw_identity: os.stat_result,
                module: types.ModuleType,
            ) -> tuple[dict[str, object], list[tuple[object, ...]]]:
                custody, held = original_validate(
                    root_fd, root_identity, raw_fd, raw_identity, module
                )
                parent.rename(moved_parent)
                parent.mkdir(mode=0o700)
                (parent / "replacement-sentinel").write_bytes(b"preserve")
                return custody, held

            with mock.patch.object(
                ACQUISITION,
                "_validate_result_fds",
                side_effect=swap_parent_after_staged_validation,
            ):
                with self.assertRaisesRegex(
                    ACQUISITION.AcquisitionError,
                    "output parent path no longer identifies",
                ):
                    self._acquire(output)
            self.assertFalse(output.exists())
            self.assertEqual(
                (parent / "replacement-sentinel").read_bytes(), b"preserve"
            )
            self.assertEqual(list(moved_parent.iterdir()), [])

        with tempfile.TemporaryDirectory() as directory:
            container = Path(directory)
            parent = container / "parent"
            parent.mkdir(mode=0o700)
            output = parent / "result"
            self._acquire(output)
            moved_parent = container / "parent-held-original"
            original_response_record = ACQUISITION._response_record
            calls = 0

            def swap_parent_during_replay(
                payload: bytes, target: object, module: types.ModuleType
            ) -> dict[str, object]:
                nonlocal calls
                record = original_response_record(payload, target, module)
                calls += 1
                if calls == len(PREREG.TARGETS):
                    parent.rename(moved_parent)
                    parent.mkdir(mode=0o700)
                return record

            with mock.patch.object(
                ACQUISITION,
                "_response_record",
                side_effect=swap_parent_during_replay,
            ):
                with self.assertRaisesRegex(
                    ACQUISITION.AcquisitionError,
                    "result parent path no longer identifies",
                ):
                    ACQUISITION.validate_result(output)
            self.assertEqual(calls, len(PREREG.TARGETS))
            self.assertFalse(output.exists())
            self.assertTrue((moved_parent / "result" / "CUSTODY.json").is_file())

    def test_early_directory_faults_roll_back_private_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "result"
            with self._authorized(), mock.patch.object(
                ACQUISITION.os,
                "fchmod",
                side_effect=OSError("injected staging chmod fault"),
            ):
                with self.assertRaisesRegex(OSError, "staging chmod fault"):
                    ACQUISITION._acquire_to_root(output)
            self.assertEqual(list(parent.iterdir()), [])

            with self._authorized(), mock.patch.object(
                ACQUISITION,
                "_create_private_directory_at",
                side_effect=ACQUISITION.AcquisitionError(
                    "injected raw directory fault"
                ),
            ):
                with self.assertRaisesRegex(
                    ACQUISITION.AcquisitionError, "raw directory fault"
                ):
                    ACQUISITION._acquire_to_root(output)
            self.assertEqual(list(parent.iterdir()), [])

    def test_create_helpers_transactionally_remove_partial_artifacts(self) -> None:
        original_fsync = ACQUISITION.os.fsync

        def injected_write(_descriptor: int, _payload: object) -> int:
            raise OSError("injected artifact write fault")

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "result"
            with self._authorized(), mock.patch.object(
                ACQUISITION,
                "_fetch",
                side_effect=lambda url: self.payloads[_spk_from_url(url)],
            ), mock.patch.object(
                ACQUISITION.os, "write", side_effect=injected_write
            ):
                with self.assertRaisesRegex(OSError, "artifact write fault"):
                    ACQUISITION._acquire_to_root(output)
            self.assertEqual(list(parent.iterdir()), [])

        failed_file_fsync = False

        def injected_file_fsync(descriptor: int) -> None:
            nonlocal failed_file_fsync
            if not failed_file_fsync and stat.S_ISREG(os.fstat(descriptor).st_mode):
                failed_file_fsync = True
                raise OSError("injected artifact fsync fault")
            original_fsync(descriptor)

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "result"
            with self._authorized(), mock.patch.object(
                ACQUISITION,
                "_fetch",
                side_effect=lambda url: self.payloads[_spk_from_url(url)],
            ), mock.patch.object(
                ACQUISITION.os, "fsync", side_effect=injected_file_fsync
            ):
                with self.assertRaisesRegex(OSError, "artifact fsync fault"):
                    ACQUISITION._acquire_to_root(output)
            self.assertTrue(failed_file_fsync)
            self.assertEqual(list(parent.iterdir()), [])

    def test_identity_registration_is_inside_create_transaction(self) -> None:
        class RejectingIdentities(dict[str, os.stat_result]):
            def __setitem__(self, key: str, value: os.stat_result) -> None:
                raise MemoryError("injected identity registration fault")

        with tempfile.TemporaryDirectory() as directory:
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                with self.assertRaisesRegex(MemoryError, "registration fault"):
                    ACQUISITION._write_create_only_at(
                        descriptor,
                        "artifact.json",
                        b"{}\n",
                        RejectingIdentities(),
                    )
                self.assertEqual(os.listdir(directory), [])
            finally:
                os.close(descriptor)

        original_write = ACQUISITION._write_create_only_at
        for failure_name in ("raw", "custody"):
            with self.subTest(failure_name=failure_name), tempfile.TemporaryDirectory() as directory:
                parent = Path(directory)
                output = parent / "result"
                raised = False

                def fail_after_registered_create(
                    directory_fd: int,
                    name: str,
                    payload: bytes,
                    identities: dict[str, os.stat_result],
                ) -> os.stat_result:
                    nonlocal raised
                    result = original_write(
                        directory_fd, name, payload, identities
                    )
                    should_raise = (
                        failure_name == "raw" and name.endswith(".json")
                    ) or (
                        failure_name == "custody"
                        and name == ".CUSTODY.json.partial"
                    )
                    if should_raise and not raised:
                        raised = True
                        raise ACQUISITION.AcquisitionError(
                            "injected post-registration failure"
                        )
                    return result

                with self._authorized(), mock.patch.object(
                    ACQUISITION,
                    "_fetch",
                    side_effect=lambda url: self.payloads[_spk_from_url(url)],
                ), mock.patch.object(
                    ACQUISITION,
                    "_write_create_only_at",
                    side_effect=fail_after_registered_create,
                ):
                    with self.assertRaisesRegex(
                        ACQUISITION.AcquisitionError,
                        "post-registration failure",
                    ):
                        ACQUISITION._acquire_to_root(output)
                self.assertTrue(raised)
                self.assertEqual(list(parent.iterdir()), [])

    def test_process_interrupt_uses_frozen_crash_staging_semantics(self) -> None:
        original_write = ACQUISITION._write_create_only_at
        interrupted = False

        def interrupt_after_registered_create(
            directory_fd: int,
            name: str,
            payload: bytes,
            identities: dict[str, os.stat_result],
        ) -> os.stat_result:
            nonlocal interrupted
            result = original_write(directory_fd, name, payload, identities)
            if not interrupted:
                interrupted = True
                raise KeyboardInterrupt("injected process-control interrupt")
            return result

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "result"
            with self._authorized(), mock.patch.object(
                ACQUISITION,
                "_fetch",
                side_effect=lambda url: self.payloads[_spk_from_url(url)],
            ), mock.patch.object(
                ACQUISITION,
                "_write_create_only_at",
                side_effect=interrupt_after_registered_create,
            ):
                with self.assertRaisesRegex(
                    KeyboardInterrupt, "process-control interrupt"
                ):
                    ACQUISITION._acquire_to_root(output)
            self.assertTrue(interrupted)
            self.assertFalse(output.exists())
            residue = list(parent.iterdir())
            self.assertEqual(len(residue), 1)
            self.assertTrue(residue[0].name.startswith(".result.staging-"))
            self.assertEqual(stat.S_IMODE(residue[0].stat().st_mode), 0o700)

    def test_custody_bound_covers_twelve_valid_large_responses(self) -> None:
        near_cap: dict[int, bytes] = {}
        ordinary_calendar = "A.D. 2026-Jun-09 00:00:00.0000"
        near_cap_calendar = "A.D. " + ("\x7f" * 935_000) + "00:00:00.0000"
        for target in PREREG.TARGETS:
            value = json.loads(self.payloads[target.spk_id])
            value["result"] = value["result"].replace(
                ordinary_calendar, near_cap_calendar
            )
            payload = json.dumps(
                value,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            self.assertLessEqual(len(payload), ACQUISITION.MAX_RESPONSE_BYTES)
            ACQUISITION._response_record(payload, target, PREREG)
            near_cap[target.spk_id] = payload
        near_cap_custody = ACQUISITION._canonical_json(
            ACQUISITION.build_custody_from_payloads(near_cap, PREREG)
        )
        self.assertGreater(near_cap_custody, b"")
        self.assertGreater(len(near_cap_custody), 64 * 1_048_576)
        self.assertLessEqual(len(near_cap_custody), ACQUISITION.MAX_CUSTODY_BYTES)

        inflated: dict[int, bytes] = {}
        large_calendar = "A.D. " + ("X" * 100_000) + "00:00:00.0000"
        for target in PREREG.TARGETS:
            value = json.loads(self.payloads[target.spk_id])
            value["result"] = value["result"].replace(
                ordinary_calendar, large_calendar
            )
            payload = json.dumps(
                value,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            self.assertLessEqual(len(payload), ACQUISITION.MAX_RESPONSE_BYTES)
            ACQUISITION._response_record(payload, target, PREREG)
            inflated[target.spk_id] = payload

        self.assertEqual(ACQUISITION.MAX_CUSTODY_BYTES, 83_886_080)
        expected = ACQUISITION.build_custody_from_payloads(inflated, PREREG)
        self.assertGreater(
            len(ACQUISITION._canonical_json(expected)),
            1_048_576,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            with self._authorized(), mock.patch.object(
                ACQUISITION,
                "_fetch",
                side_effect=lambda url: inflated[_spk_from_url(url)],
            ):
                observed = ACQUISITION._acquire_to_root(output)
            self.assertEqual(observed, expected)
            self.assertGreater((output / "CUSTODY.json").stat().st_size, 1_048_576)
            self.assertEqual(ACQUISITION.validate_result(output), expected)

    def test_post_mkdir_stat_and_open_faults_remove_private_staging(self) -> None:
        original_stat = ACQUISITION.os.stat
        failed_staging_stat = False

        def injected_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
            nonlocal failed_staging_stat
            if (
                not failed_staging_stat
                and isinstance(path, str)
                and path.startswith(".result.staging-")
                and kwargs.get("dir_fd") is not None
            ):
                failed_staging_stat = True
                raise OSError("injected post-mkdir staging stat fault")
            return original_stat(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "result"
            with self._authorized(), mock.patch.object(
                ACQUISITION.os, "stat", side_effect=injected_stat
            ):
                with self.assertRaisesRegex(OSError, "staging stat fault"):
                    ACQUISITION._acquire_to_root(output)
            self.assertTrue(failed_staging_stat)
            self.assertEqual(list(parent.iterdir()), [])

        original_open = ACQUISITION.os.open
        failed_raw_open = False

        def injected_open(path: object, *args: object, **kwargs: object) -> int:
            nonlocal failed_raw_open
            if (
                not failed_raw_open
                and path == "raw"
                and kwargs.get("dir_fd") is not None
            ):
                failed_raw_open = True
                raise OSError("injected post-mkdir raw open fault")
            return original_open(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "result"
            with self._authorized(), mock.patch.object(
                ACQUISITION.os, "open", side_effect=injected_open
            ):
                with self.assertRaisesRegex(OSError, "raw open fault"):
                    ACQUISITION._acquire_to_root(output)
            self.assertTrue(failed_raw_open)
            self.assertEqual(list(parent.iterdir()), [])

    def test_post_custody_rename_stat_fault_removes_private_staging(self) -> None:
        original_stat = ACQUISITION.os.stat
        failed_destination_stat = False

        def injected_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
            nonlocal failed_destination_stat
            directory_fd = kwargs.get("dir_fd")
            if (
                not failed_destination_stat
                and path == "CUSTODY.json"
                and isinstance(directory_fd, int)
                and "CUSTODY.json" in os.listdir(directory_fd)
            ):
                failed_destination_stat = True
                raise OSError("injected post-rename custody stat fault")
            return original_stat(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "result"
            with self._authorized(), mock.patch.object(
                ACQUISITION,
                "_fetch",
                side_effect=lambda url: self.payloads[_spk_from_url(url)],
            ), mock.patch.object(
                ACQUISITION.os, "stat", side_effect=injected_stat
            ):
                with self.assertRaisesRegex(OSError, "custody stat fault"):
                    ACQUISITION._acquire_to_root(output)
            self.assertTrue(failed_destination_stat)
            self.assertEqual(list(parent.iterdir()), [])

    def test_successful_artifact_identity_survives_close_error(self) -> None:
        original_close = ACQUISITION.os.close
        failed_regular_close = False

        def injected_close(descriptor: int) -> None:
            nonlocal failed_regular_close
            try:
                is_regular = stat.S_ISREG(os.fstat(descriptor).st_mode)
            except OSError:
                is_regular = False
            if not failed_regular_close and is_regular:
                failed_regular_close = True
                original_close(descriptor)
                raise OSError("injected post-create close fault")
            original_close(descriptor)

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "result"
            with self._authorized(), mock.patch.object(
                ACQUISITION,
                "_fetch",
                side_effect=lambda url: self.payloads[_spk_from_url(url)],
            ), mock.patch.object(
                ACQUISITION.os, "close", side_effect=injected_close
            ):
                observed = ACQUISITION._acquire_to_root(output)
            self.assertTrue(failed_regular_close)
            self.assertEqual(observed, ACQUISITION.validate_result(output))
            self.assertEqual([path.name for path in parent.iterdir()], ["result"])

    def test_result_root_reanchor_fault_closes_every_descriptor(self) -> None:
        proc_fds = Path("/proc/self/fd")
        if not proc_fds.is_dir():
            self.skipTest("descriptor accounting requires /proc/self/fd")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            self._acquire(output)
            original_reanchor = ACQUISITION._reanchor_directory_path
            calls = 0

            def fault_second_reanchor(
                path: Path, descriptors: object, label: str
            ) -> None:
                nonlocal calls
                calls += 1
                if calls % 2 == 0:
                    raise ACQUISITION.AcquisitionError(
                        "injected result-root reanchor fault"
                    )
                original_reanchor(path, descriptors, label)

            before = len(os.listdir(proc_fds))
            with mock.patch.object(
                ACQUISITION,
                "_reanchor_directory_path",
                side_effect=fault_second_reanchor,
            ):
                for _attempt in range(10):
                    with self.assertRaisesRegex(
                        ACQUISITION.AcquisitionError, "result-root reanchor fault"
                    ):
                        ACQUISITION.validate_result(output)
            self.assertEqual(calls, 20)
            self.assertEqual(len(os.listdir(proc_fds)), before)

    def test_post_rename_parent_fsync_failure_keeps_complete_result(self) -> None:
        original_fsync = ACQUISITION.os.fsync
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"

            def fault_after_publish(descriptor: int) -> None:
                if output.exists():
                    raise OSError("injected parent fsync fault")
                original_fsync(descriptor)

            with mock.patch.object(
                ACQUISITION.os, "fsync", side_effect=fault_after_publish
            ):
                with self.assertRaisesRegex(
                    ACQUISITION.AcquisitionError, "durability is indeterminate"
                ):
                    self._acquire(output)
            self.assertTrue(output.is_dir())
            self.assertEqual(set(os.listdir(output)), {"raw", "CUSTODY.json"})
            ACQUISITION.validate_result(output)

    def test_offline_replay_is_read_only_and_never_calls_transport(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            expected = self._acquire(output)
            before = _timestamps(output)
            with mock.patch.object(
                ACQUISITION,
                "_fetch",
                side_effect=AssertionError("network path used by offline replay"),
            ) as fetch:
                observed = ACQUISITION.validate_result(output)
            fetch.assert_not_called()
            self.assertEqual(observed, expected)
            self.assertEqual(_timestamps(output), before)

    def test_offline_replay_rejects_a_raw_directory_dirent_swap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            self._acquire(output)
            original_response_record = ACQUISITION._response_record
            calls = 0

            def swap_after_last_response(
                payload: bytes, target: object, module: types.ModuleType
            ) -> dict[str, object]:
                nonlocal calls
                record = original_response_record(payload, target, module)
                calls += 1
                if calls == len(PREREG.TARGETS):
                    raw = output / "raw"
                    raw.rename(output.parent / "raw-held-original")
                    raw.mkdir(mode=0o700)
                    raw.chmod(0o700)
                return record

            with mock.patch.object(
                ACQUISITION, "_response_record", side_effect=swap_after_last_response
            ):
                with self.assertRaisesRegex(
                    ACQUISITION.AcquisitionError,
                    "dirent no longer identifies the held directory",
                ):
                    ACQUISITION.validate_result(output)
            self.assertEqual(calls, 12)
            self.assertTrue((output / "raw").is_dir())
            self.assertEqual(set(os.listdir(output / "raw")), set())
            self.assertEqual(
                set(os.listdir(output.parent / "raw-held-original")),
                set(PREREG.RESULT_RAW_ROSTER),
            )

    def test_offline_replay_rejects_raw_custody_link_mode_and_roster_attacks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            base = parent / "base"
            self._acquire(base)

            def copied(name: str) -> Path:
                destination = parent / name
                shutil.copytree(base, destination)
                return destination

            attacked = copied("raw-tamper")
            raw = attacked / "raw" / PREREG.RESULT_RAW_ROSTER[0]
            with raw.open("r+b") as stream:
                stream.seek(0)
                stream.write(b"[")
            with self.assertRaises(ACQUISITION.AcquisitionError):
                ACQUISITION.validate_result(attacked)

            attacked = copied("custody-extra")
            custody_path = attacked / "CUSTODY.json"
            custody = json.loads(custody_path.read_bytes())
            custody["extra"] = False
            custody_path.write_bytes(ACQUISITION._canonical_json(custody))
            with self.assertRaisesRegex(
                ACQUISITION.AcquisitionError, "top-level key roster"
            ):
                ACQUISITION.validate_result(attacked)

            attacked = copied("custody-noncanonical")
            custody_path = attacked / "CUSTODY.json"
            custody = json.loads(custody_path.read_bytes())
            custody_path.write_text(json.dumps(custody, indent=2), encoding="ascii")
            with self.assertRaisesRegex(
                ACQUISITION.AcquisitionError, "not exact canonical"
            ):
                ACQUISITION.validate_result(attacked)

            attacked = copied("fifo")
            raw = attacked / "raw" / PREREG.RESULT_RAW_ROSTER[0]
            raw.unlink()
            os.mkfifo(raw, mode=0o600)
            raw.chmod(0o600)
            with self.assertRaisesRegex(
                ACQUISITION.AcquisitionError, "owned regular nlink-one"
            ):
                ACQUISITION.validate_result(attacked)

            prereg_fifo = parent / "prereg-source-fifo"
            os.mkfifo(prereg_fifo, mode=0o600)
            with self.assertRaisesRegex(
                ACQUISITION.AcquisitionError, "nonregular"
            ):
                ACQUISITION._read_regular_path(prereg_fifo, "test FIFO", 1024)

            attacked = copied("symlink")
            raw = attacked / "raw" / PREREG.RESULT_RAW_ROSTER[0]
            raw.unlink()
            raw.symlink_to(PREREG.RESULT_RAW_ROSTER[1])
            with self.assertRaisesRegex(
                ACQUISITION.AcquisitionError, "owned regular nlink-one"
            ):
                ACQUISITION.validate_result(attacked)

            attacked = copied("hardlink")
            first = attacked / "raw" / PREREG.RESULT_RAW_ROSTER[0]
            second = attacked / "raw" / PREREG.RESULT_RAW_ROSTER[1]
            second.unlink()
            os.link(first, second)
            with self.assertRaisesRegex(
                ACQUISITION.AcquisitionError, "nlink-one"
            ):
                ACQUISITION.validate_result(attacked)

            attacked = copied("mode")
            raw = attacked / "raw" / PREREG.RESULT_RAW_ROSTER[0]
            raw.chmod(0o644)
            with self.assertRaisesRegex(
                ACQUISITION.AcquisitionError, "0600"
            ):
                ACQUISITION.validate_result(attacked)

            attacked = copied("extra-roster")
            extra = attacked / "raw" / "extra.json"
            extra.write_bytes(b"{}")
            extra.chmod(0o600)
            with self.assertRaisesRegex(
                ACQUISITION.AcquisitionError, "roster differs"
            ):
                ACQUISITION.validate_result(attacked)

            attacked = copied("missing-custody")
            (attacked / "CUSTODY.json").unlink()
            with self.assertRaisesRegex(
                ACQUISITION.AcquisitionError, "no committed result"
            ):
                ACQUISITION.validate_result(attacked)

    def test_optimized_offline_cli_replays_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            expected = self._acquire(output)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-OO",
                    str(SCRIPT_PATH),
                    "validate-result",
                    "--output-root",
                    str(output),
                ],
                cwd=PROJECT_ROOT,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout), expected)

    def test_source_scope_and_cli_exclude_dynamics_and_resume(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertTrue(
            {"rebound", "reboundx", "numpy", "scipy", "jax", "spiceypy"}.isdisjoint(
                imported
            )
        )
        parser = ACQUISITION._parser()
        choices = next(
            action.choices
            for action in parser._actions
            if getattr(action, "dest", None) == "command"
        )
        self.assertEqual(
            tuple(choices),
            ("audit-prerequisites", "acquire", "validate-result"),
        )
        self.assertNotIn("resume", source)
        self.assertNotIn("urllib.request", source)
        self.assertEqual(
            tuple(inspect.signature(ACQUISITION.acquire).parameters), ()
        )
        self.assertEqual(
            tuple(inspect.signature(ACQUISITION._acquire_to_root).parameters),
            ("root",),
        )
        self.assertNotIn("--output-root", choices["acquire"]._option_string_actions)
        self.assertIn("--output-root", choices["validate-result"]._option_string_actions)


def _spk_from_url(url: str) -> int:
    for target in PREREG.TARGETS:
        if target.query_url == url:
            return target.spk_id
    raise AssertionError(f"unexpected URL: {url}")


if __name__ == "__main__":
    unittest.main()
