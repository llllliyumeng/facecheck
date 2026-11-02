"""Per-video key wrapping interface and an in-memory SM2 prototype provider.

The provider owns key selection and private-key access. This local prototype
does not persist keys and is not a KMS; production implementations should
implement :class:`VideoKeyProvider` against a trusted key registry/HSM.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import importlib
import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

from .crypto import sm2_decrypt, sm2_encrypt

WRAPPED_KEY_ALGORITHM = "SM2-encryption-C1C3C2"
WRAPPED_KEY_FORMAT = b"FaceCheck-video-key-envelope-v1\x00"
VIDEO_KEY_MATERIAL_SIZE = 48
KEY_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class KeyManagementError(ValueError):
    """Raised when a key envelope cannot be safely created or opened."""


@dataclass(frozen=True)
class PublicVideoKeyRecord:
    """Non-secret key-directory record; trust comes from the pinned digest."""

    key_id: str
    algorithm: str
    public_key: str
    status: str = "active"

    def as_dict(self) -> dict[str, str]:
        return {
            "key_id": self.key_id,
            "algorithm": self.algorithm,
            "public_key": self.public_key,
            "status": self.status,
        }


class PinnedVideoKeyDirectory:
    """Validate a public-key directory against a digest supplied by a trust anchor.

    The digest must arrive through a separately protected configuration channel.
    This class does not make a writable local JSON file a production trust root.
    """

    def __init__(self, records: Sequence[PublicVideoKeyRecord], pinned_sha256: str):
        if not isinstance(pinned_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", pinned_sha256):
            raise KeyManagementError("public key directory pin is invalid")
        if (any(not isinstance(record, PublicVideoKeyRecord) for record in records)
                or any(not isinstance(record.key_id, str) for record in records)):
            raise KeyManagementError("public key directory is invalid")
        normalized = sorted(records, key=lambda record: record.key_id)
        if not normalized or len({record.key_id for record in normalized}) != len(normalized):
            raise KeyManagementError("public key directory is invalid")
        active = 0
        for record in normalized:
            if (not isinstance(record.key_id, str)
                    or not KEY_ID_PATTERN.fullmatch(record.key_id)
                    or not isinstance(record.algorithm, str)
                    or record.algorithm != WRAPPED_KEY_ALGORITHM
                    or not isinstance(record.status, str)
                    or record.status not in {"active", "decrypt_only", "revoked"}
                    or not isinstance(record.public_key, str)):
                raise KeyManagementError("public key directory is invalid")
            public_hex = record.public_key[2:] if record.public_key.startswith("04") else record.public_key
            try:
                public_bytes = bytes.fromhex(public_hex)
            except ValueError:
                raise KeyManagementError("public key directory is invalid") from None
            if len(public_bytes) != 64:
                raise KeyManagementError("public key directory is invalid")
            active += record.status == "active"
        if active != 1:
            raise KeyManagementError("public key directory must have exactly one active key")
        payload = json.dumps(
            {"version": 1, "records": [record.as_dict() for record in normalized]},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        if not hmac.compare_digest(digest, pinned_sha256):
            raise KeyManagementError("public key directory pin mismatch")
        self._records = {record.key_id: record for record in normalized}
        self.sha256 = digest

    @classmethod
    def from_file(cls, path, *, pinned_sha256: str) -> "PinnedVideoKeyDirectory":
        """Load public records from a file whose digest is pinned separately."""
        from pathlib import Path

        directory_path = Path(path)
        if (directory_path.is_symlink() or not directory_path.is_file()
                or directory_path.stat().st_size > 64 * 1024):
            raise KeyManagementError("public key directory file is unavailable or unsafe")
        try:
            root = json.loads(directory_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise KeyManagementError("public key directory file is invalid") from None
        if (not isinstance(root, dict) or set(root) != {"version", "records"}
                or isinstance(root.get("version"), bool) or root.get("version") != 1
                or not isinstance(root.get("records"), list)):
            raise KeyManagementError("public key directory file is invalid")
        records = []
        for value in root["records"]:
            if (not isinstance(value, dict)
                    or set(value) != {"key_id", "algorithm", "public_key", "status"}):
                raise KeyManagementError("public key directory file is invalid")
            try:
                records.append(PublicVideoKeyRecord(**value))
            except TypeError:
                raise KeyManagementError("public key directory file is invalid") from None
        return cls(records, pinned_sha256)

    @classmethod
    def digest_records(cls, records: Sequence[PublicVideoKeyRecord]) -> str:
        normalized = sorted(records, key=lambda record: record.key_id)
        payload = json.dumps(
            {"version": 1, "records": [record.as_dict() for record in normalized]},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def active_for_wrap(self) -> PublicVideoKeyRecord:
        return next(record for record in self._records.values() if record.status == "active")

    def resolve(self, key_id: str, *, for_decryption: bool = True) -> PublicVideoKeyRecord:
        record = self._records.get(key_id)
        if record is None or record.status == "revoked":
            raise KeyManagementError("video key is unavailable")
        if not for_decryption and record.status != "active":
            raise KeyManagementError("video key cannot be used for new envelopes")
        return record


class VideoKeyProviderRegistry:
    """In-process provider router used by tests and local adapters.

    Production implementations should keep lifecycle state in the KMS/HSM and
    use this interface only as a request router, not as the revocation authority.
    """

    def __init__(self, providers: Sequence[VideoKeyProvider], active_key_id: str):
        by_id = {provider.key_id: provider for provider in providers}
        if (not by_id or len(by_id) != len(providers)
                or active_key_id not in by_id):
            raise KeyManagementError("video key provider registry is invalid")
        self._providers = by_id
        self._active_key_id = active_key_id
        self._states = {key_id: "active" if key_id == active_key_id else "decrypt_only"
                        for key_id in by_id}

    def active_for_wrap(self) -> VideoKeyProvider:
        if self._states.get(self._active_key_id) != "active":
            raise KeyManagementError("active video key is unavailable")
        return self._providers[self._active_key_id]

    def resolve(self, key_id: str) -> VideoKeyProvider:
        if self._states.get(key_id) not in {"active", "decrypt_only"}:
            raise KeyManagementError("video key is unavailable")
        return self._providers[key_id]

    def activate(self, key_id: str) -> None:
        if key_id not in self._providers or self._states.get(key_id) == "revoked":
            raise KeyManagementError("video key is unavailable")
        for existing in self._states:
            if self._states[existing] == "active":
                self._states[existing] = "decrypt_only"
        self._states[key_id] = "active"
        self._active_key_id = key_id

    def revoke(self, key_id: str, *, readiness_check: Callable[[str], bool]) -> None:
        if key_id not in self._providers or key_id == self._active_key_id:
            raise KeyManagementError("active or unknown video key cannot be revoked")
        if not readiness_check(key_id):
            raise KeyManagementError("video key still has registered references")
        self._states[key_id] = "revoked"
        self._providers.pop(key_id, None)


def load_video_key_provider_bundle(factory_spec: str | None = None) -> dict[str, object] | None:
    """Load a deployment supplied provider adapter from trusted process config."""
    spec = factory_spec if factory_spec is not None else os.getenv(
        "FACECHECK_VIDEO_KEY_PROVIDER_FACTORY", "",
    )
    if not spec:
        return None
    module_name, separator, factory_name = spec.partition(":")
    if not separator or not module_name or not factory_name:
        raise KeyManagementError("video key provider factory configuration is invalid")
    try:
        factory = getattr(importlib.import_module(module_name), factory_name)
        bundle = factory()
    except Exception:
        raise KeyManagementError("video key provider adapter is unavailable") from None
    if not isinstance(bundle, Mapping):
        raise KeyManagementError("video key provider adapter returned invalid metadata")
    target = bundle.get("target_provider")
    resolver = bundle.get("source_provider_resolver")
    revoker = bundle.get("revoke_key")
    if (not isinstance(getattr(target, "key_id", None), str)
            or not callable(getattr(target, "wrap_video_key", None))
            or not callable(getattr(target, "unwrap_video_key", None))
            or not callable(resolver)
            or (revoker is not None and not callable(revoker))):
        raise KeyManagementError("video key provider adapter returned invalid interfaces")
    return {"target_provider": target, "source_provider_resolver": resolver,
            "revoke_key": revoker}


@dataclass(frozen=True)
class WrappedVideoKey:
    algorithm: str
    key_id: str
    ciphertext: bytes

    def to_manifest(self) -> dict[str, str]:
        return {
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "wrapped_key_b64": base64.b64encode(self.ciphertext).decode("ascii"),
        }

    @classmethod
    def from_manifest(cls, value: object) -> "WrappedVideoKey":
        if not isinstance(value, Mapping) or set(value) != {
            "algorithm", "key_id", "wrapped_key_b64",
        }:
            raise KeyManagementError("video key envelope is invalid")
        algorithm = value.get("algorithm")
        key_id = value.get("key_id")
        encoded = value.get("wrapped_key_b64")
        if (algorithm != WRAPPED_KEY_ALGORITHM or not isinstance(key_id, str)
                or not KEY_ID_PATTERN.fullmatch(key_id) or not isinstance(encoded, str)
                or len(encoded) > 8192):
            raise KeyManagementError("video key envelope is invalid")
        try:
            ciphertext = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (UnicodeEncodeError, binascii.Error, ValueError):
            raise KeyManagementError("video key envelope is invalid") from None
        if not ciphertext:
            raise KeyManagementError("video key envelope is invalid")
        return cls(algorithm=algorithm, key_id=key_id, ciphertext=ciphertext)


class VideoKeyProvider(Protocol):
    """Interface expected from a trusted local keystore or remote KMS."""

    @property
    def key_id(self) -> str: ...

    def wrap_video_key(
        self, key_material: bytes, *, video_id: str, key_version: str,
    ) -> WrappedVideoKey: ...

    def unwrap_video_key(
        self, envelope: WrappedVideoKey, *, video_id: str, key_version: str,
    ) -> bytes: ...


class Sm2MemoryKeyProvider:
    """Small-test SM2 provider; caller injects keys and owns their protection."""

    def __init__(
        self,
        *,
        key_id: str,
        public_key: str,
        private_key: str | None = None,
    ) -> None:
        if not isinstance(key_id, str) or not KEY_ID_PATTERN.fullmatch(key_id):
            raise KeyManagementError("SM2 key ID is invalid")
        if not isinstance(public_key, str):
            raise KeyManagementError("SM2 public key is invalid")
        public_hex = public_key[2:] if public_key.startswith("04") else public_key
        try:
            public_bytes = bytes.fromhex(public_hex)
        except ValueError:
            raise KeyManagementError("SM2 public key is invalid") from None
        if len(public_bytes) != 64:
            raise KeyManagementError("SM2 public key is invalid")
        if private_key is not None and (not isinstance(private_key, str) or not private_key):
            raise KeyManagementError("SM2 private key is invalid")
        self._key_id = key_id
        self._public_key = public_hex
        self._private_key = private_key

    @property
    def key_id(self) -> str:
        return self._key_id

    def wrap_video_key(
        self, key_material: bytes, *, video_id: str, key_version: str,
    ) -> WrappedVideoKey:
        if not isinstance(key_material, bytes) or len(key_material) != VIDEO_KEY_MATERIAL_SIZE:
            raise KeyManagementError("video key material has an invalid length")
        context = self._context(video_id, key_version)
        payload = (
            WRAPPED_KEY_FORMAT
            + len(context).to_bytes(2, "big")
            + context
            + len(key_material).to_bytes(2, "big")
            + key_material
        )
        try:
            ciphertext = sm2_encrypt(self._public_key, payload)
        except Exception:
            raise KeyManagementError("video key wrapping failed") from None
        if not isinstance(ciphertext, bytes) or not ciphertext:
            raise KeyManagementError("video key wrapping failed")
        return WrappedVideoKey(WRAPPED_KEY_ALGORITHM, self._key_id, ciphertext)

    def unwrap_video_key(
        self, envelope: WrappedVideoKey, *, video_id: str, key_version: str,
    ) -> bytes:
        if (self._private_key is None or not isinstance(envelope, WrappedVideoKey)
                or envelope.algorithm != WRAPPED_KEY_ALGORITHM
                or envelope.key_id != self._key_id):
            raise KeyManagementError("video key envelope could not be opened")
        try:
            payload = sm2_decrypt(self._private_key, envelope.ciphertext)
            prefix_length = len(WRAPPED_KEY_FORMAT)
            if payload[:prefix_length] != WRAPPED_KEY_FORMAT:
                raise ValueError
            context_length_offset = prefix_length
            context_length = int.from_bytes(payload[context_length_offset:context_length_offset + 2], "big")
            context_start = context_length_offset + 2
            context_end = context_start + context_length
            context = payload[context_start:context_end]
            key_length_end = context_end + 2
            key_length = int.from_bytes(payload[context_end:key_length_end], "big")
            key_material = payload[key_length_end:]
            expected_context = self._context(video_id, key_version)
            if (len(context) != context_length or len(key_material) != key_length
                    or key_length != VIDEO_KEY_MATERIAL_SIZE
                    or not hmac.compare_digest(context, expected_context)):
                raise ValueError
            return key_material
        except Exception:
            raise KeyManagementError("video key envelope could not be opened") from None

    def _context(self, video_id: str, key_version: str) -> bytes:
        if (not isinstance(video_id, str) or not video_id
                or not isinstance(key_version, str) or not key_version):
            raise KeyManagementError("video key context is invalid")
        try:
            if len(video_id.encode("utf-8")) > 64 or len(key_version.encode("utf-8")) > 128:
                raise KeyManagementError("video key context is invalid")
        except UnicodeEncodeError:
            raise KeyManagementError("video key context is invalid")
        return json.dumps(
            {
                "algorithm": WRAPPED_KEY_ALGORITHM,
                "key_id": self._key_id,
                "key_version": key_version,
                "video_id": video_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
