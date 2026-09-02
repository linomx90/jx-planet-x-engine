from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Iterable

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .errors import IntegrityError, ValidationError
from .util import atomic_write_bytes, atomic_write_text, canonical_json_bytes, refuse_symlink, sha256_bytes

SCHEMA = "jx-sealed-records/v1"
ALGORITHM = "RSA-3072-OAEP-SHA256+AES-256-GCM-per-record"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(value: str) -> bytes:
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except Exception as error:
        raise IntegrityError("invalid base64 in encrypted container") from error


def public_key_fingerprint(public_key: rsa.RSAPublicKey) -> str:
    der = public_key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return sha256_bytes(der)


def load_public_key(path: Path) -> rsa.RSAPublicKey:
    refuse_symlink(path)
    key = serialization.load_pem_public_key(path.read_bytes())
    if not isinstance(key, rsa.RSAPublicKey) or key.key_size < 3072:
        raise ValidationError("curator public key must be RSA with at least 3072 bits")
    return key


def load_private_key(path: Path, password: bytes | None) -> rsa.RSAPrivateKey:
    refuse_symlink(path)
    key = serialization.load_pem_private_key(path.read_bytes(), password=password)
    if not isinstance(key, rsa.RSAPrivateKey) or key.key_size < 3072:
        raise ValidationError("custodian private key must be RSA with at least 3072 bits")
    return key


def generate_keypair(private_path: Path, public_path: Path, password: bytes, receipt_path: Path) -> dict[str, Any]:
    if len(password) < 16:
        raise ValidationError("private-key password must contain at least 16 bytes")
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    public_key = private_key.public_key()
    private_bytes = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.BestAvailableEncryption(password),
    )
    public_bytes = public_key.public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    fingerprint = public_key_fingerprint(public_key)
    atomic_write_bytes(private_path, private_bytes, 0o600)
    atomic_write_bytes(public_path, public_bytes, 0o444)
    receipt = {
        "schema": "jx-holdout-custodian-key-receipt/v1",
        "algorithm": ALGORITHM,
        "public_key_fingerprint_sha256": fingerprint,
        "private_key_path": private_path.name,
        "public_key_path": public_path.name,
        "instruction": "Move the encrypted private key and its password offline. The curator and fitting environment must retain only the public key.",
    }
    atomic_write_text(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n", 0o444)
    return receipt


def _aad(header_core: dict[str, Any], index: int) -> bytes:
    return canonical_json_bytes({"header": header_core, "record_index": index})


def encrypt_records(
    records: list[dict[str, Any]],
    output_path: Path,
    public_key: rsa.RSAPublicKey,
    *,
    dataset_name: str,
    source_sha256: str,
    cutoff_utc: str,
    holdout_end_utc: str,
) -> dict[str, Any]:
    data_key = bytearray(os.urandom(32))
    nonce_prefix = os.urandom(4)
    try:
        wrapped_key = public_key.encrypt(
            bytes(data_key),
            padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=b"JX-HOLDOUT-V1"),
        )
        header_core = {
            "schema": SCHEMA,
            "algorithm": ALGORITHM,
            "dataset_name": dataset_name,
            "record_count": len(records),
            "source_sha256": source_sha256,
            "cutoff_utc": cutoff_utc,
            "holdout_end_utc": holdout_end_utc,
            "public_key_fingerprint_sha256": public_key_fingerprint(public_key),
        }
        header = dict(header_core)
        header.update({"wrapped_key_b64": _b64(wrapped_key), "nonce_prefix_b64": _b64(nonce_prefix)})
        lines = [canonical_json_bytes(header)]
        aes = AESGCM(bytes(data_key))
        for index, record in enumerate(records):
            nonce = nonce_prefix + index.to_bytes(8, "big")
            plaintext = canonical_json_bytes(record)
            ciphertext = aes.encrypt(nonce, plaintext, _aad(header_core, index))
            lines.append(canonical_json_bytes({"i": index, "nonce_b64": _b64(nonce), "ciphertext_b64": _b64(ciphertext)}))
        payload = b"\n".join(lines) + b"\n"
        atomic_write_bytes(output_path, payload, 0o400)
        return {
            "schema": SCHEMA,
            "algorithm": ALGORITHM,
            "dataset_name": dataset_name,
            "record_count": len(records),
            "public_key_fingerprint_sha256": header_core["public_key_fingerprint_sha256"],
        }
    finally:
        for index in range(len(data_key)):
            data_key[index] = 0


def decrypt_records(path: Path, private_key: rsa.RSAPrivateKey) -> list[dict[str, Any]]:
    refuse_symlink(path)
    with path.open("rb") as stream:
        raw_lines = [line.rstrip(b"\n") for line in stream if line.strip()]
    if not raw_lines:
        raise IntegrityError("encrypted container is empty")
    try:
        header = json.loads(raw_lines[0].decode("utf-8"))
    except Exception as error:
        raise IntegrityError("encrypted container header is invalid") from error
    if header.get("schema") != SCHEMA or header.get("algorithm") != ALGORITHM:
        raise IntegrityError("encrypted container schema or algorithm mismatch")
    expected_fingerprint = header.get("public_key_fingerprint_sha256")
    if public_key_fingerprint(private_key.public_key()) != expected_fingerprint:
        raise IntegrityError("custodian private key does not match the curation public key")
    try:
        data_key = bytearray(
            private_key.decrypt(
                _unb64(header["wrapped_key_b64"]),
                padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=b"JX-HOLDOUT-V1"),
            )
        )
    except Exception as error:
        raise IntegrityError("could not unwrap holdout data key") from error
    header_core = {key: header[key] for key in (
        "schema", "algorithm", "dataset_name", "record_count", "source_sha256", "cutoff_utc", "holdout_end_utc", "public_key_fingerprint_sha256"
    )}
    records: list[dict[str, Any]] = []
    try:
        aes = AESGCM(bytes(data_key))
        expected_count = int(header["record_count"])
        if len(raw_lines) - 1 != expected_count:
            raise IntegrityError("encrypted record count mismatch")
        for expected_index, raw in enumerate(raw_lines[1:]):
            try:
                envelope = json.loads(raw.decode("utf-8"))
                index = int(envelope["i"])
                if index != expected_index:
                    raise IntegrityError("encrypted record order mismatch")
                nonce = _unb64(envelope["nonce_b64"])
                if len(nonce) != 12 or nonce[-8:] != index.to_bytes(8, "big"):
                    raise IntegrityError("encrypted record nonce mismatch")
                plaintext = aes.decrypt(nonce, _unb64(envelope["ciphertext_b64"]), _aad(header_core, index))
                record = json.loads(plaintext.decode("utf-8"))
            except IntegrityError:
                raise
            except Exception as error:
                raise IntegrityError(f"encrypted record {expected_index} failed authentication") from error
            if not isinstance(record, dict):
                raise IntegrityError("decrypted record is not an object")
            records.append(record)
        return records
    finally:
        for index in range(len(data_key)):
            data_key[index] = 0
