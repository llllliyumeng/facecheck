from __future__ import annotations

import secrets

import pytest

from facecheck.crypto import (
    SM2_PRIVATE_KEY,
    SM2_PUBLIC_KEY,
    Sm4Envelope,
    sm2_sign,
    sm2_verify,
    sm3_digest,
    sm3_hmac,
    sm4_decrypt,
    sm4_encrypt,
)
from facecheck.crypto_backends import GmSSLNativeBackend, PythonBackend


@pytest.fixture(scope="module")
def native() -> GmSSLNativeBackend:
    try:
        return GmSSLNativeBackend()
    except RuntimeError as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize("payload", [b"", b"x", b"a" * 16, b"a" * 17, b"large" * 20000])
def test_sm4_cbc_cross_backend_and_envelope_compatibility(
    native: GmSSLNativeBackend, payload: bytes
) -> None:
    python = PythonBackend()
    key, iv = secrets.token_bytes(16), secrets.token_bytes(16)
    native_ciphertext = native.sm4_cbc_encrypt(key, iv, payload)
    python_ciphertext = python.sm4_cbc_encrypt(key, iv, payload)
    assert native_ciphertext == python_ciphertext
    assert python.sm4_cbc_decrypt(key, iv, native_ciphertext) == payload
    assert native.sm4_cbc_decrypt(key, iv, python_ciphertext) == payload

    mac_key = secrets.token_bytes(32)
    native_envelope = sm4_encrypt(payload, key, mac_key, associated_data=b"chunk-0")
    parsed = Sm4Envelope.decode(native_envelope)
    assert native.sm3_hmac(
        mac_key,
        b"FC-SM4-CBC-SM3-ETM\x01" + (7).to_bytes(4, "big") + b"chunk-0"
        + parsed.key_id + parsed.iv + parsed.ciphertext,
    ) == parsed.tag
    assert sm4_decrypt(native_envelope, key, mac_key, associated_data=b"chunk-0") == payload


def test_native_sm2_signatures_interoperate_in_both_directions(
    native: GmSSLNativeBackend,
) -> None:
    from gmssl import sm2

    message = b"SM2 backend interoperability"
    random_hex = secrets.token_hex(32)
    native_signature = native.sm2_sign(
        SM2_PRIVATE_KEY, SM2_PUBLIC_KEY, message, random_hex
    )
    assert len(native_signature) == 128
    assert sm2_verify(SM2_PUBLIC_KEY, message, native_signature)
    assert not native.sm2_verify(SM2_PUBLIC_KEY, message + b"!", native_signature)

    python_signer = sm2.CryptSM2(
        public_key=SM2_PUBLIC_KEY, private_key=SM2_PRIVATE_KEY
    )
    python_signature = python_signer.sign_with_sm3(message, random_hex)
    assert native.sm2_verify(SM2_PUBLIC_KEY, message, python_signature)
    assert not native.sm2_verify(SM2_PUBLIC_KEY, message + b"!", python_signature)

    wrong_public_key = "".join(reversed([SM2_PUBLIC_KEY[:64], SM2_PUBLIC_KEY[64:]]))
    assert not native.sm2_verify(wrong_public_key, message, python_signature)


def test_sm4_envelope_rejects_tampering_and_wrong_keys(
    native: GmSSLNativeBackend,
) -> None:
    key, mac_key = secrets.token_bytes(16), secrets.token_bytes(32)
    envelope = sm4_encrypt(b"authenticated payload", key, mac_key)
    with pytest.raises(ValueError, match="authentication failed"):
        sm4_decrypt(envelope, key, secrets.token_bytes(32))
    altered = bytearray(envelope)
    altered[-1] ^= 1
    with pytest.raises(ValueError, match="authentication failed"):
        sm4_decrypt(bytes(altered), key, mac_key)
    assert native.sm3_digest(b"abc").hex() == (
        "66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0"
    )
