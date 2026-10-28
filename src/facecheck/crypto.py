"""Small, testable building blocks for the stage-two crypto experiments.

SM4-CBC is authenticated with encrypt-then-MAC using SM3 as the hash
primitive. This is not an AEAD mode and the encryption and MAC keys are
intentionally separate.
"""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass

from gmssl import sm2, sm3
from gmssl.sm4 import CryptSM4, SM4_DECRYPT, SM4_ENCRYPT

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


def sm3_digest(data: bytes) -> bytes:
    """Return the SM3 digest of bytes."""
    return bytes.fromhex(sm3.sm3_hash(list(data)))


def sm3_hmac(key: bytes, data: bytes) -> bytes:
    """Return HMAC-SM3 using the SM3 block size defined by the standard."""
    block_size = 64
    if len(key) > block_size:
        key = sm3_digest(key)
    key = key.ljust(block_size, b"\x00")
    inner = bytes(byte ^ 0x36 for byte in key)
    outer = bytes(byte ^ 0x5C for byte in key)
    return sm3_digest(outer + sm3_digest(inner + data))


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
        return SM4_ENVELOPE_HEADER + self.key_id + self.iv + self.tag + self.ciphertext

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
) -> bytes:
    """Encrypt and authenticate bytes with SM4-CBC and encrypt-then-MAC."""
    _validate_key(encryption_key, SM4_KEY_SIZE, "encryption_key")
    if not mac_key:
        raise ValueError("mac_key must not be empty")
    iv = os.urandom(SM4_BLOCK_SIZE)
    key_id = sm3_digest(encryption_key)[:SM4_KEY_ID_SIZE]
    cipher = CryptSM4(padding_mode=3)
    cipher.set_key(encryption_key, SM4_ENCRYPT)
    ciphertext = cipher.crypt_cbc(iv, plaintext)
    authenticated_data = (
        SM4_ENVELOPE_HEADER + len(associated_data).to_bytes(4, "big")
        + associated_data + key_id + iv + ciphertext
    )
    tag = sm3_hmac(mac_key, authenticated_data)
    return Sm4Envelope(key_id=key_id, iv=iv, tag=tag, ciphertext=ciphertext).encode()


def sm4_decrypt(
    envelope: bytes,
    encryption_key: bytes,
    mac_key: bytes,
    associated_data: bytes = b"",
) -> bytes:
    """Authenticate and decrypt an SM4-CBC envelope."""
    _validate_key(encryption_key, SM4_KEY_SIZE, "encryption_key")
    if not mac_key:
        raise ValueError("mac_key must not be empty")
    parsed = Sm4Envelope.decode(envelope)
    key_id = sm3_digest(encryption_key)[:SM4_KEY_ID_SIZE]
    if not hmac.compare_digest(parsed.key_id, key_id):
        raise ValueError("SM4 envelope authentication failed")
    authenticated_data = (
        SM4_ENVELOPE_HEADER + len(associated_data).to_bytes(4, "big")
        + associated_data + parsed.key_id + parsed.iv + parsed.ciphertext
    )
    expected_tag = sm3_hmac(mac_key, authenticated_data)
    if not hmac.compare_digest(parsed.tag, expected_tag):
        raise ValueError("SM4 envelope authentication failed")
    cipher = CryptSM4(padding_mode=3)
    cipher.set_key(encryption_key, SM4_DECRYPT)
    try:
        return cipher.crypt_cbc(parsed.iv, parsed.ciphertext)
    except (ValueError, IndexError) as exc:
        raise ValueError("invalid SM4 ciphertext") from exc


def sm2_encrypt(public_key: str, plaintext: bytes) -> bytes:
    """Encrypt a small key-encryption payload with an SM2 public key."""
    return sm2.CryptSM2(public_key=public_key, private_key="").encrypt(plaintext)


def sm2_decrypt(private_key: str, ciphertext: bytes) -> bytes:
    """Decrypt an SM2 key-encryption payload with an SM2 private key."""
    return sm2.CryptSM2(public_key="", private_key=private_key).decrypt(ciphertext)


def sm2_sign(private_key: str, public_key: str, data: bytes, random_hex: str) -> str:
    """Create an SM2 signature; this is separate from SM2 key encapsulation."""
    signer = sm2.CryptSM2(public_key=public_key, private_key=private_key)
    return signer.sign_with_sm3(data, random_hex)


def sm2_verify(public_key: str, data: bytes, signature: str) -> bool:
    """Verify an SM2 signature."""
    verifier = sm2.CryptSM2(public_key=public_key, private_key="")
    return verifier.verify_with_sm3(signature, data)
