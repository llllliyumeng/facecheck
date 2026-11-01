"""Small, testable building blocks for the stage-two crypto experiments.

SM4-CBC is authenticated with encrypt-then-MAC using SM3 as the hash
primitive. This is not an AEAD mode and the encryption and MAC keys are
intentionally separate.
"""

from __future__ import annotations

import hmac
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional

from .crypto_backends import get_crypto_backend

SM4_BLOCK_SIZE = 16
SM4_KEY_SIZE = 16
SM3_DIGEST_SIZE = 32
SM4_KEY_ID_SIZE = 8
SM4_ENVELOPE_HEADER = b"FC-SM4-CBC-SM3-ETM\x01"
SM2_PRIVATE_KEY = (
    "00B9AB0B828FF68872F21A837FC303668428DEA11DCD1B24429D0C99E24EED83D5"
)
SM2_PUBLIC_KEY = (
    "B9C9A6E04E9C91F7BA880429273747D7EF5DDEB0BB2FF6317EB00BEF331A83081A6994B8993F3F5D6EADDDB81872266C87C018FB4162F5AF347B483E24620207"
)
SM3_KDF_DOMAIN = b"FaceGuard-SM3-KDF-v1"


def sm3_digest(data: bytes) -> bytes:
    """Return the SM3 digest of bytes."""
    return get_crypto_backend().sm3_digest(data)


def sm3_hmac(key: bytes, data: bytes) -> bytes:
    """Return HMAC-SM3 using the SM3 block size defined by the standard."""
    return get_crypto_backend().sm3_hmac(key, data)


@lru_cache(maxsize=32)
def _key_id(key: bytes) -> bytes:
    """Cache the stable key identifier across video chunks."""
    return sm3_digest(key)[:SM4_KEY_ID_SIZE]


def sm3_kdf(master_key: bytes, context: bytes, length: int) -> bytes:
    """Derive key material with the SM3 counter KDF and domain separation."""
    if not isinstance(master_key, bytes) or not master_key:
        raise ValueError("master_key must not be empty")
    if not isinstance(context, bytes) or not context:
        raise ValueError("context must not be empty")
    if not isinstance(length, int) or length <= 0:
        raise ValueError("length must be a positive integer")
    output = bytearray()
    context_header = (
        SM3_KDF_DOMAIN + len(context).to_bytes(4, "big") + context
    )
    counter = 1
    while len(output) < length:
        output.extend(
            sm3_digest(
                context_header
                + len(master_key).to_bytes(4, "big")
                + master_key
                + counter.to_bytes(4, "big")
            )
        )
        counter += 1
    return bytes(output[:length])


def derive_storage_keys(
    encryption_master_key: bytes,
    mac_master_key: bytes,
    key_version: str,
) -> tuple[bytes, bytes]:
    """Derive version-bound SM4 and SM3-MAC keys for video chunks."""
    if not isinstance(key_version, str) or not key_version:
        raise ValueError("key_version must not be empty")
    _validate_key(encryption_master_key, SM4_KEY_SIZE, "encryption_master_key")
    if not isinstance(mac_master_key, bytes) or not mac_master_key:
        raise ValueError("mac_master_key must not be empty")
    version = key_version.encode("utf-8")
    return (
        sm3_kdf(encryption_master_key, b"sm4-encryption:" + version, SM4_KEY_SIZE),
        sm3_kdf(mac_master_key, b"sm3-mac:" + version, SM3_DIGEST_SIZE),
    )


def _validate_key(key: bytes, expected_size: int, name: str) -> None:
    if len(key) != expected_size:
        raise ValueError(f"{name} must be {expected_size} bytes")


@dataclass(frozen=True)
class Sm4Envelope:
    """Authenticated SM4-CBC payload with an explicit versioned encoding."""

    key_id: bytes
    iv: bytes
    tag: bytes
    ciphertext: bytes

    def encode(self) -> bytes:
        return b"".join((SM4_ENVELOPE_HEADER, self.key_id, self.iv, self.tag, self.ciphertext))

    @classmethod
    def decode(cls, encoded: bytes) -> "Sm4Envelope":
        minimum = (
            len(SM4_ENVELOPE_HEADER)
            + SM4_KEY_ID_SIZE
            + SM4_BLOCK_SIZE
            + SM3_DIGEST_SIZE
            + SM4_BLOCK_SIZE
        )
        if len(encoded) < minimum or not encoded.startswith(SM4_ENVELOPE_HEADER):
            raise ValueError("invalid SM4 envelope")
        offset = len(SM4_ENVELOPE_HEADER)
        key_id = encoded[offset : offset + SM4_KEY_ID_SIZE]
        offset += SM4_KEY_ID_SIZE
        iv = encoded[offset : offset + SM4_BLOCK_SIZE]
        offset += SM4_BLOCK_SIZE
        tag = encoded[offset : offset + SM3_DIGEST_SIZE]
        ciphertext = encoded[offset + SM3_DIGEST_SIZE :]
        if not ciphertext or len(ciphertext) % SM4_BLOCK_SIZE:
            raise ValueError("invalid SM4 ciphertext length")
        return cls(key_id=key_id, iv=iv, tag=tag, ciphertext=ciphertext)


def sm4_encrypt(
    plaintext: bytes,
    encryption_key: bytes,
    mac_key: bytes,
    associated_data: bytes = b"",
    stats: Optional[Dict[str, float]] = None,
) -> bytes:
    """Encrypt and authenticate bytes with SM4-CBC and encrypt-then-MAC."""
    _validate_key(encryption_key, SM4_KEY_SIZE, "encryption_key")
    if not mac_key:
        raise ValueError("mac_key must not be empty")
    iv = os.urandom(SM4_BLOCK_SIZE)
    key_id_started = time.perf_counter()
    key_id = _key_id(encryption_key)
    if stats is not None:
        stats["key_id_seconds"] = stats.get("key_id_seconds", 0.0) + (
            time.perf_counter() - key_id_started
        )
    cipher_started = time.perf_counter()
    ciphertext = get_crypto_backend().sm4_cbc_encrypt(encryption_key, iv, plaintext)
    if stats is not None:
        stats["sm4_seconds"] = stats.get("sm4_seconds", 0.0) + (
            time.perf_counter() - cipher_started
        )
    authenticated_parts = (
        SM4_ENVELOPE_HEADER,
        len(associated_data).to_bytes(4, "big"),
        associated_data,
        key_id,
        iv,
        ciphertext,
    )
    mac_started = time.perf_counter()
    tag = get_crypto_backend().sm3_hmac_parts(mac_key, authenticated_parts)
    if stats is not None:
        stats["hmac_sm3_seconds"] = stats.get("hmac_sm3_seconds", 0.0) + (
            time.perf_counter() - mac_started
        )
    return Sm4Envelope(key_id=key_id, iv=iv, tag=tag, ciphertext=ciphertext).encode()


def sm4_decrypt(
    envelope: bytes,
    encryption_key: bytes,
    mac_key: bytes,
    associated_data: bytes = b"",
    stats: Optional[Dict[str, float]] = None,
) -> bytes:
    """Authenticate and decrypt an SM4-CBC envelope."""
    _validate_key(encryption_key, SM4_KEY_SIZE, "encryption_key")
    if not mac_key:
        raise ValueError("mac_key must not be empty")
    parsed = Sm4Envelope.decode(envelope)
    key_id_started = time.perf_counter()
    key_id = _key_id(encryption_key)
    if stats is not None:
        stats["key_id_seconds"] = stats.get("key_id_seconds", 0.0) + (
            time.perf_counter() - key_id_started
        )
    if not hmac.compare_digest(parsed.key_id, key_id):
        raise ValueError("SM4 envelope authentication failed")
    authenticated_parts = (
        SM4_ENVELOPE_HEADER,
        len(associated_data).to_bytes(4, "big"),
        associated_data,
        parsed.key_id,
        parsed.iv,
        parsed.ciphertext,
    )
    mac_started = time.perf_counter()
    expected_tag = get_crypto_backend().sm3_hmac_parts(mac_key, authenticated_parts)
    if stats is not None:
        stats["hmac_sm3_seconds"] = stats.get("hmac_sm3_seconds", 0.0) + (
            time.perf_counter() - mac_started
        )
    if not hmac.compare_digest(parsed.tag, expected_tag):
        raise ValueError("SM4 envelope authentication failed")
    try:
        cipher_started = time.perf_counter()
        plaintext = get_crypto_backend().sm4_cbc_decrypt(
            encryption_key, parsed.iv, parsed.ciphertext
        )
        if stats is not None:
            stats["sm4_seconds"] = stats.get("sm4_seconds", 0.0) + (
                time.perf_counter() - cipher_started
            )
        return plaintext
    except (ValueError, IndexError) as exc:
        raise ValueError("invalid SM4 ciphertext") from exc


def sm2_encrypt(public_key: str, plaintext: bytes) -> bytes:
    """Encrypt a small key-encryption payload with an SM2 public key."""
    from gmssl import sm2

    return sm2.CryptSM2(public_key=public_key, private_key="").encrypt(plaintext)


def sm2_decrypt(private_key: str, ciphertext: bytes) -> bytes:
    """Decrypt an SM2 key-encryption payload with an SM2 private key."""
    from gmssl import sm2

    return sm2.CryptSM2(public_key="", private_key=private_key).decrypt(ciphertext)


def sm2_sign(private_key: str, public_key: str, data: bytes, random_hex: str) -> str:
    """Create an SM2 signature in the existing fixed-width r||s encoding."""
    return get_crypto_backend().sm2_sign(private_key, public_key, data, random_hex)


def sm2_verify(public_key: str, data: bytes, signature: str) -> bool:
    """Verify an SM2 signature."""
    return get_crypto_backend().sm2_verify(public_key, data, signature)
