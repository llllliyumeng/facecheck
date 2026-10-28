from __future__ import annotations

import pytest

from facecheck.crypto_backend import (
    sm3_digest as backend_sm3_digest,
    sm3_hmac as backend_sm3_hmac,
    sm4_cbc_decrypt,
    sm4_cbc_encrypt,
)
from facecheck.crypto import (
    Sm4Envelope,
    sm3_digest,
    sm3_hmac,
    sm4_decrypt,
    sm4_encrypt,
)


def _gmssl_hmac_sm3(key: bytes, data: bytes) -> bytes:
    from gmssl import sm3

    block_size = 64
    if len(key) > block_size:
        key = bytes.fromhex(sm3.sm3_hash(list(key)))
    padded_key = key.ljust(block_size, b"\x00")
    inner = bytes(value ^ 0x36 for value in padded_key)
    outer = bytes(value ^ 0x5C for value in padded_key)
    inner_digest = bytes.fromhex(sm3.sm3_hash(list(inner + data)))
    return bytes.fromhex(sm3.sm3_hash(list(outer + inner_digest)))


def test_native_backend_matches_sm4_and_sm3_reference_outputs() -> None:
    from gmssl import sm3
    from gmssl.sm4 import CryptSM4, SM4_ENCRYPT

    key = bytes.fromhex("0123456789abcdeffedcba9876543210")
    iv = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
    plaintext = b"authorized backend interoperability test" * 17

    reference_cipher = CryptSM4(padding_mode=3)
    reference_cipher.set_key(key, SM4_ENCRYPT)
    expected_ciphertext = reference_cipher.crypt_cbc(iv, plaintext)

    assert sm4_cbc_encrypt(key, iv, plaintext) == expected_ciphertext
    assert sm4_cbc_decrypt(key, iv, expected_ciphertext) == plaintext
    assert backend_sm3_digest(b"abc").hex() == sm3.sm3_hash(list(b"abc"))


def test_native_hmac_matches_existing_sm3_hmac_and_rejects_modified_data() -> None:
    key = b"independent-mac-key"
    data = b"authenticated data"
    tag = backend_sm3_hmac(key, data)

    padded_key = key.ljust(64, b"\x00")
    inner = bytes(value ^ 0x36 for value in padded_key)
    outer = bytes(value ^ 0x5C for value in padded_key)
    reference_tag = backend_sm3_digest(outer + backend_sm3_digest(inner + data))
    assert tag == reference_tag
    assert tag != backend_sm3_hmac(key, data + b"!")


@pytest.mark.parametrize(
    ("key", "data"),
    [
        (b"", b""),
        (b"ordinary-key", b""),
        (b"ordinary-key", b"a"),
        (b"ordinary-key", b"a" * 63),
        (b"ordinary-key", b"a" * 64),
        (b"ordinary-key", b"a" * 65),
        (b"k" * 64, b"message at the block boundary"),
        (b"k" * 65, b"message with a key requiring SM3 reduction"),
        (bytes(range(256)), bytes(range(256))),
    ],
)
def test_native_hmac_matches_gmssl_at_message_and_key_boundaries(
    key: bytes, data: bytes
) -> None:
    assert backend_sm3_hmac(key, data) == _gmssl_hmac_sm3(key, data)


def test_native_sm4_rejects_invalid_key_and_padding() -> None:
    with pytest.raises(ValueError):
        sm4_cbc_encrypt(b"short", b"\x00" * 16, b"payload")
    ciphertext = sm4_cbc_encrypt(
        b"0123456789abcdef", b"\x00" * 16, b"payload"
    )
    tampered = bytearray(ciphertext)
    tampered[-1] ^= 1
    with pytest.raises(ValueError):
        sm4_cbc_decrypt(b"0123456789abcdef", b"\x00" * 16, bytes(tampered))


def test_native_backend_reads_legacy_gmssl_envelopes() -> None:
    from gmssl.sm4 import CryptSM4, SM4_DECRYPT, SM4_ENCRYPT

    key = b"0123456789abcdef"
    mac_key = b"independent-mac-key"
    associated_data = b"legacy-context"
    iv = bytes(range(16))
    plaintext = b"legacy encrypted payload"

    cipher = CryptSM4(padding_mode=3)
    cipher.set_key(key, SM4_ENCRYPT)
    ciphertext = cipher.crypt_cbc(iv, plaintext)
    key_id = sm3_digest(key)[:8]
    authenticated_data = (
        b"FC-SM4-CBC-SM3-ETM\x01"
        + len(associated_data).to_bytes(4, "big")
        + associated_data
        + key_id
        + iv
        + ciphertext
    )
    legacy = Sm4Envelope(
        key_id=key_id,
        iv=iv,
        tag=sm3_hmac(mac_key, authenticated_data),
        ciphertext=ciphertext,
    ).encode()

    assert sm4_decrypt(legacy, key, mac_key, associated_data) == plaintext

    new_envelope = sm4_encrypt(plaintext, key, mac_key, associated_data)
    parsed = Sm4Envelope.decode(new_envelope)
    decryptor = CryptSM4(padding_mode=3)
    decryptor.set_key(key, SM4_DECRYPT)
    assert decryptor.crypt_cbc(parsed.iv, parsed.ciphertext) == plaintext
