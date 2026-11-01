"""Per-video key wrapping interface and an in-memory SM2 prototype provider.

The provider owns key selection and private-key access. This local prototype
does not persist keys and is not a KMS; production implementations should
implement :class:`VideoKeyProvider` against a trusted key registry/HSM.
"""
from __future__ import annotations

import base64
import binascii
import hmac
import json
import re
from dataclasses import dataclass
from typing import Mapping, Protocol

from .crypto import sm2_decrypt, sm2_encrypt

WRAPPED_KEY_ALGORITHM = "SM2-encryption-C1C3C2"
WRAPPED_KEY_FORMAT = b"FaceCheck-video-key-envelope-v1\x00"
VIDEO_KEY_MATERIAL_SIZE = 48
KEY_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class KeyManagementError(ValueError):
    """Raised when a key envelope cannot be safely created or opened."""


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
