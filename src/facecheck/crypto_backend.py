"""Native SM4/SM3 primitives used by the authenticated envelope layer."""

from __future__ import annotations

from cryptography.hazmat.primitives import hashes, hmac, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def sm3_digest(data: bytes) -> bytes:
    digest = hashes.Hash(hashes.SM3())
    digest.update(data)
    return digest.finalize()


def sm3_hmac(key: bytes, data: bytes) -> bytes:
    return sm3_hmac_parts(key, (data,))


def sm3_hmac_parts(key: bytes, parts: tuple[bytes, ...]) -> bytes:
    authenticator = hmac.HMAC(key, hashes.SM3())
    for part in parts:
        authenticator.update(part)
    return authenticator.finalize()


def sm4_cbc_encrypt(key: bytes, iv: bytes, plaintext: bytes) -> bytes:
    padder = padding.PKCS7(algorithms.SM4.block_size).padder()
    padded = padder.update(plaintext) + padder.finalize()
    encryptor = Cipher(algorithms.SM4(key), modes.CBC(iv)).encryptor()
    return encryptor.update(padded) + encryptor.finalize()


def sm4_cbc_decrypt(key: bytes, iv: bytes, ciphertext: bytes) -> bytes:
    decryptor = Cipher(algorithms.SM4(key), modes.CBC(iv)).decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder = padding.PKCS7(algorithms.SM4.block_size).unpadder()
    return unpadder.update(padded) + unpadder.finalize()
