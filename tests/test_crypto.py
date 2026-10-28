import secrets

import pytest

from facecheck.crypto import (
    SM2_PRIVATE_KEY,
    SM2_PUBLIC_KEY,
    sm2_decrypt,
    sm2_encrypt,
    sm2_sign,
    sm2_verify,
    sm3_digest,
    sm4_decrypt,
    sm4_encrypt,
)


def test_sm3_standard_vector_abc() -> None:
    assert sm3_digest(b"abc").hex() == (
        "66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0"
    )


def test_sm4_standard_vector() -> None:
    from gmssl.sm4 import CryptSM4, SM4_ENCRYPT

    key = bytes.fromhex("0123456789abcdeffedcba9876543210")
    plaintext = key
    cipher = CryptSM4(padding_mode=0)
    cipher.set_key(key, SM4_ENCRYPT)
    # gmssl 3.2.2's ECB helper emits an extra block for a full input even
    # with padding_mode=0; the first block is the standard single-block vector.
    assert cipher.crypt_ecb(plaintext)[:16].hex() == "681edf34d206965e86b3e94f536e4246"


def test_sm4_authenticated_encryption_round_trip_and_random_iv() -> None:
    encryption_key = b"0123456789abcdef"
    mac_key = b"separate-sm3-mac-key"
    first = sm4_encrypt(b"authorized test payload", encryption_key, mac_key)
    second = sm4_encrypt(b"authorized test payload", encryption_key, mac_key)

    assert first != second
    assert sm4_decrypt(first, encryption_key, mac_key) == b"authorized test payload"


def test_sm4_rejects_wrong_keys_and_modified_ciphertext() -> None:
    envelope = sm4_encrypt(b"payload", b"0123456789abcdef", b"mac-key")

    with pytest.raises(ValueError, match="authentication failed"):
        sm4_decrypt(envelope, b"fedcba9876543210", b"mac-key")
    with pytest.raises(ValueError, match="authentication failed"):
        sm4_decrypt(envelope, b"0123456789abcdef", b"wrong-mac-key")
    modified = bytearray(envelope)
    modified[-1] ^= 1
    with pytest.raises(ValueError, match="authentication failed"):
        sm4_decrypt(bytes(modified), b"0123456789abcdef", b"mac-key")


def test_sm2_key_encryption_and_decryption() -> None:
    key_material = b"0123456789abcdef"
    ciphertext = sm2_encrypt(SM2_PUBLIC_KEY, key_material)

    assert sm2_decrypt(SM2_PRIVATE_KEY, ciphertext) == key_material


def test_sm2_signature_verification_and_tamper_detection() -> None:
    data = b"authorized test message"
    random_hex = secrets.token_hex(32)
    signature = sm2_sign(SM2_PRIVATE_KEY, SM2_PUBLIC_KEY, data, random_hex)

    assert sm2_verify(SM2_PUBLIC_KEY, data, signature)
    assert not sm2_verify(SM2_PUBLIC_KEY, b"tampered message", signature)


def test_sm4_rejects_invalid_envelopes_and_empty_mac_key() -> None:
    with pytest.raises(ValueError, match="mac_key must not be empty"):
        sm4_encrypt(b"payload", b"0123456789abcdef", b"")
    with pytest.raises(ValueError, match="invalid SM4 envelope"):
        sm4_decrypt(b"not-an-envelope", b"0123456789abcdef", b"mac-key")


def test_sm4_context_is_authenticated() -> None:
    envelope = sm4_encrypt(
        b"payload",
        b"0123456789abcdef",
        b"mac-key",
        associated_data=b"video-1:chunk-0:key-v1",
    )

    with pytest.raises(ValueError, match="authentication failed"):
        sm4_decrypt(
            envelope,
            b"0123456789abcdef",
            b"mac-key",
            associated_data=b"video-2:chunk-0:key-v1",
        )
