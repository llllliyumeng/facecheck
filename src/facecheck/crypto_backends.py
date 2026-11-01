"""Selectable Python and GmSSL C backends for the existing crypto formats."""
from __future__ import annotations

import ctypes
import ctypes.util
import functools
import os
import threading
from pathlib import Path
from typing import Protocol

from gmssl import sm2

from . import crypto_backend as python_primitives


class CryptoBackend(Protocol):
    name: str

    def sm3_digest(self, data: bytes) -> bytes: ...
    def sm3_hmac(self, key: bytes, data: bytes) -> bytes: ...
    def sm3_hmac_parts(self, key: bytes, parts: tuple[bytes, ...]) -> bytes: ...
    def sm4_cbc_encrypt(self, key: bytes, iv: bytes, data: bytes) -> bytes: ...
    def sm4_cbc_decrypt(self, key: bytes, iv: bytes, data: bytes) -> bytes: ...
    def sm2_sign(self, private_key: str, public_key: str, data: bytes, random_hex: str) -> str: ...
    def sm2_verify(self, public_key: str, data: bytes, signature: str) -> bool: ...


class PythonBackend:
    """Compatibility backend using gmssl Python and cryptography primitives."""

    name = "python"

    def sm3_digest(self, data: bytes) -> bytes:
        return python_primitives.sm3_digest(data)

    def sm3_hmac(self, key: bytes, data: bytes) -> bytes:
        return python_primitives.sm3_hmac(key, data)

    def sm3_hmac_parts(self, key: bytes, parts: tuple[bytes, ...]) -> bytes:
        return python_primitives.sm3_hmac_parts(key, parts)

    def sm4_cbc_encrypt(self, key: bytes, iv: bytes, data: bytes) -> bytes:
        return python_primitives.sm4_cbc_encrypt(key, iv, data)

    def sm4_cbc_decrypt(self, key: bytes, iv: bytes, data: bytes) -> bytes:
        return python_primitives.sm4_cbc_decrypt(key, iv, data)

    def sm2_sign(self, private_key: str, public_key: str, data: bytes, random_hex: str) -> str:
        signer = sm2.CryptSM2(public_key=public_key, private_key=private_key)
        return signer.sign_with_sm3(data, random_hex)

    def sm2_verify(self, public_key: str, data: bytes, signature: str) -> bool:
        verifier = sm2.CryptSM2(public_key=public_key, private_key="")
        try:
            return bool(verifier.verify_with_sm3(signature, data))
        except (ValueError, TypeError):
            return False


_U8 = ctypes.c_uint8
_U8P = ctypes.POINTER(_U8)
_U32 = ctypes.c_uint32
_U64 = ctypes.c_uint64
_Size = ctypes.c_size_t


class _Sm2Scalar(ctypes.Array):
    _type_ = _U64
    _length_ = 4


class _Sm2Point(ctypes.Structure):
    _fields_ = [(name, _Sm2Scalar) for name in ("X", "Y", "Z")]


class _Sm2Key(ctypes.Structure):
    _fields_ = [("public_key", _Sm2Point), ("private_key", _Sm2Scalar)]


class _Sm2Signature(ctypes.Structure):
    _fields_ = [("r", _U8 * 32), ("s", _U8 * 32)]


class _Sm3Context(ctypes.Structure):
    _fields_ = [("digest", _U32 * 8), ("nblocks", _U64), ("block", _U8 * 64), ("num", _Size)]


class _Sm3HmacContext(ctypes.Structure):
    _fields_ = [("sm3_ctx", _Sm3Context), ("key", _U8 * 64)]


class _Sm4Key(ctypes.Structure):
    _fields_ = [("round_keys", _U32 * 32)]


def _buffer(data: bytes):
    return (_U8 * len(data)).from_buffer_copy(data) if data else (_U8 * 1)()


def _pointer(data: bytes):
    """Point at immutable Python bytes without copying the payload."""
    if not data:
        return (_U8 * 1)()
    return ctypes.cast(ctypes.c_char_p(data), _U8P)


def _public_bytes(public_key: str) -> bytes:
    value = public_key[2:] if public_key.startswith("04") else public_key
    raw = bytes.fromhex(value)
    if len(raw) != 64:
        raise ValueError("SM2 public key must contain 64 coordinate bytes")
    return raw


def _library_candidates() -> list[str]:
    configured = os.environ.get("FACECHECK_GMSSL_LIBRARY")
    candidates = [configured] if configured else []
    located = ctypes.util.find_library("gmssl")
    if located:
        candidates.append(located)
    candidates.extend([
        "/opt/homebrew/opt/gmssl/lib/libgmssl.dylib",
        "/usr/local/opt/gmssl/lib/libgmssl.dylib",
        "libgmssl.so.3",
        "libgmssl.so",
    ])
    return list(dict.fromkeys(item for item in candidates if item))


def _openssl_library_candidates() -> list[str]:
    configured = os.environ.get("FACECHECK_OPENSSL_LIBRARY")
    candidates = [configured] if configured else []
    candidates.extend([
        "/opt/homebrew/opt/openssl@4/lib/libcrypto.4.dylib",
        "/opt/homebrew/opt/openssl@3/lib/libcrypto.3.dylib",
        "/usr/local/opt/openssl@4/lib/libcrypto.4.dylib",
        "libcrypto.so.3",
        "libcrypto.so",
    ])
    return list(dict.fromkeys(item for item in candidates if item))


class GmSSLNativeBackend:
    """GmSSL 3 C API binding; C calls through CDLL release the Python GIL."""

    name = "gmssl-native"

    def __init__(self, library_path: str | None = None):
        candidates = [library_path] if library_path else _library_candidates()
        errors = []
        self.library_path = ""
        for candidate in candidates:
            try:
                self.lib = ctypes.CDLL(candidate)
                self.library_path = candidate
                break
            except OSError as exc:
                errors.append(f"{candidate}: {exc}")
        else:
            detail = "; ".join(errors) or "no library candidates found"
            raise RuntimeError(f"GmSSL shared library unavailable ({detail})")
        self._bind()
        version = self.lib.gmssl_version_str().decode("ascii", errors="replace")
        if not version.startswith("GmSSL 3."):
            raise RuntimeError(f"GmSSL 3.x is required; found {version}")
        self.version = version
        sm4_impl = os.environ.get("FACECHECK_SM4_NATIVE_IMPL", "gmssl").lower()
        if sm4_impl not in {"gmssl", "openssl"}:
            raise RuntimeError("FACECHECK_SM4_NATIVE_IMPL must be gmssl or openssl")
        self._openssl = self._load_openssl_sm4() if sm4_impl == "openssl" else None
        self._openssl_contexts = threading.local()

    def _load_openssl_sm4(self):
        for candidate in _openssl_library_candidates():
            try:
                lib = ctypes.CDLL(candidate)
                lib.EVP_CIPHER_CTX_new.restype = ctypes.c_void_p
                lib.EVP_CIPHER_CTX_free.argtypes = [ctypes.c_void_p]
                lib.EVP_CIPHER_CTX_reset.argtypes = [ctypes.c_void_p]
                lib.EVP_CIPHER_CTX_reset.restype = ctypes.c_int
                lib.EVP_sm4_cbc.restype = ctypes.c_void_p
                lib.EVP_CIPHER_fetch.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
                lib.EVP_CIPHER_fetch.restype = ctypes.c_void_p
                lib.EVP_CIPHER_free.argtypes = [ctypes.c_void_p]
                lib.EVP_CipherInit_ex.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    _U8P, _U8P, ctypes.c_int,
                ]
                lib.EVP_CipherInit_ex.restype = ctypes.c_int
                lib.EVP_CipherUpdate.argtypes = [
                    ctypes.c_void_p, _U8P, ctypes.POINTER(ctypes.c_int),
                    _U8P, ctypes.c_int,
                ]
                lib.EVP_CipherUpdate.restype = ctypes.c_int
                lib.EVP_CipherFinal_ex.argtypes = [
                    ctypes.c_void_p, _U8P, ctypes.POINTER(ctypes.c_int),
                ]
                lib.EVP_CipherFinal_ex.restype = ctypes.c_int
                lib.OpenSSL_version.argtypes = [ctypes.c_int]
                lib.OpenSSL_version.restype = ctypes.c_char_p
                cipher = lib.EVP_CIPHER_fetch(None, b"SM4-CBC", None)
                if not cipher:
                    continue
                ctx = lib.EVP_CIPHER_CTX_new()
                if not ctx:
                    lib.EVP_CIPHER_free(cipher)
                    continue
                try:
                    if lib.EVP_CipherInit_ex(ctx, cipher, None, None, None, 1) != 1:
                        continue
                finally:
                    lib.EVP_CIPHER_CTX_free(ctx)
                self.openssl_library_path = candidate
                self.openssl_version = lib.OpenSSL_version(0).decode("ascii", errors="replace")
                self._openssl_cipher = cipher
                return lib
            except (AttributeError, OSError):
                continue
        self.openssl_library_path = None
        self.openssl_version = None
        self._openssl_cipher = None
        return None

    def _bind(self) -> None:
        lib = self.lib
        lib.gmssl_version_str.restype = ctypes.c_char_p
        lib.sm3_init.argtypes = [ctypes.POINTER(_Sm3Context)]
        lib.sm3_update.argtypes = [ctypes.POINTER(_Sm3Context), _U8P, _Size]
        lib.sm3_finish.argtypes = [ctypes.POINTER(_Sm3Context), _U8P]
        lib.sm3_hmac_init.argtypes = [ctypes.POINTER(_Sm3HmacContext), _U8P, _Size]
        lib.sm3_hmac_update.argtypes = [ctypes.POINTER(_Sm3HmacContext), _U8P, _Size]
        lib.sm3_hmac_finish.argtypes = [ctypes.POINTER(_Sm3HmacContext), _U8P]
        lib.sm4_set_encrypt_key.argtypes = [ctypes.POINTER(_Sm4Key), _U8P]
        lib.sm4_set_decrypt_key.argtypes = [ctypes.POINTER(_Sm4Key), _U8P]
        lib.sm4_cbc_padding_encrypt.argtypes = [
            ctypes.POINTER(_Sm4Key), _U8P, _U8P, _Size, _U8P, ctypes.POINTER(_Size),
        ]
        lib.sm4_cbc_padding_encrypt.restype = ctypes.c_int
        lib.sm4_cbc_padding_decrypt.argtypes = [
            ctypes.POINTER(_Sm4Key), _U8P, _U8P, _Size, _U8P, ctypes.POINTER(_Size),
        ]
        lib.sm4_cbc_padding_decrypt.restype = ctypes.c_int
        lib.sm2_z256_from_bytes.argtypes = [ctypes.POINTER(_Sm2Scalar), _U8P]
        lib.sm2_z256_point_from_bytes.argtypes = [ctypes.POINTER(_Sm2Point), _U8P]
        lib.sm2_key_set_private_key.argtypes = [ctypes.POINTER(_Sm2Key), ctypes.POINTER(_Sm2Scalar)]
        lib.sm2_key_set_private_key.restype = ctypes.c_int
        lib.sm2_key_set_public_key.argtypes = [ctypes.POINTER(_Sm2Key), ctypes.POINTER(_Sm2Point)]
        lib.sm2_key_set_public_key.restype = ctypes.c_int
        lib.sm2_compute_z.argtypes = [
            _U8P, ctypes.POINTER(_Sm2Point), ctypes.c_char_p, _Size,
        ]
        lib.sm2_compute_z.restype = ctypes.c_int
        lib.sm2_do_sign.argtypes = [ctypes.POINTER(_Sm2Key), _U8P, ctypes.POINTER(_Sm2Signature)]
        lib.sm2_do_sign.restype = ctypes.c_int
        lib.sm2_do_verify.argtypes = [ctypes.POINTER(_Sm2Key), _U8P, ctypes.POINTER(_Sm2Signature)]
        lib.sm2_do_verify.restype = ctypes.c_int

    def sm3_digest(self, data: bytes) -> bytes:
        ctx, output, source = _Sm3Context(), (_U8 * 32)(), _pointer(data)
        self.lib.sm3_init(ctypes.byref(ctx))
        self.lib.sm3_update(ctypes.byref(ctx), source, len(data))
        self.lib.sm3_finish(ctypes.byref(ctx), output)
        return bytes(output)

    def sm3_hmac(self, key: bytes, data: bytes) -> bytes:
        if not key:
            raise ValueError("HMAC key must not be empty")
        ctx, output, kbuf, source = _Sm3HmacContext(), (_U8 * 32)(), _pointer(key), _pointer(data)
        self.lib.sm3_hmac_init(ctypes.byref(ctx), kbuf, len(key))
        self.lib.sm3_hmac_update(ctypes.byref(ctx), source, len(data))
        self.lib.sm3_hmac_finish(ctypes.byref(ctx), output)
        return bytes(output)

    def sm3_hmac_parts(self, key: bytes, parts: tuple[bytes, ...]) -> bytes:
        if not key:
            raise ValueError("HMAC key must not be empty")
        ctx, output = _Sm3HmacContext(), (_U8 * 32)()
        self.lib.sm3_hmac_init(ctypes.byref(ctx), _pointer(key), len(key))
        for part in parts:
            self.lib.sm3_hmac_update(ctypes.byref(ctx), _pointer(part), len(part))
        self.lib.sm3_hmac_finish(ctypes.byref(ctx), output)
        return bytes(output)

    @staticmethod
    def _check_sm4_inputs(key: bytes, iv: bytes) -> None:
        if len(key) != 16:
            raise ValueError("SM4 key must be 16 bytes")
        if len(iv) != 16:
            raise ValueError("SM4 IV must be 16 bytes")

    @functools.lru_cache(maxsize=32)
    def _sm4_encrypt_key(self, key: bytes) -> _Sm4Key:
        key_ctx = _Sm4Key()
        self.lib.sm4_set_encrypt_key(ctypes.byref(key_ctx), _pointer(key))
        return key_ctx

    @functools.lru_cache(maxsize=32)
    def _sm4_decrypt_key(self, key: bytes) -> _Sm4Key:
        key_ctx = _Sm4Key()
        self.lib.sm4_set_decrypt_key(ctypes.byref(key_ctx), _pointer(key))
        return key_ctx

    def sm4_cbc_encrypt(self, key: bytes, iv: bytes, data: bytes) -> bytes:
        self._check_sm4_inputs(key, iv)
        if self._openssl is not None:
            return self._openssl_sm4_cbc(key, iv, data, encrypt=True)
        key_ctx, outlen = self._sm4_encrypt_key(key), _Size()
        ivbuf, source = _pointer(iv), _pointer(data)
        output = (_U8 * (len(data) + 16))()
        result = self.lib.sm4_cbc_padding_encrypt(
            ctypes.byref(key_ctx), ivbuf, source, len(data), output, ctypes.byref(outlen),
        )
        if result != 1:
            raise ValueError("GmSSL SM4-CBC encryption failed")
        return ctypes.string_at(ctypes.addressof(output), outlen.value)

    def sm4_cbc_decrypt(self, key: bytes, iv: bytes, data: bytes) -> bytes:
        self._check_sm4_inputs(key, iv)
        if not data or len(data) % 16:
            raise ValueError("SM4-CBC ciphertext must be a non-empty block multiple")
        if self._openssl is not None:
            return self._openssl_sm4_cbc(key, iv, data, encrypt=False)
        key_ctx, outlen = self._sm4_decrypt_key(key), _Size()
        ivbuf, source = _pointer(iv), _pointer(data)
        output = (_U8 * len(data))()
        result = self.lib.sm4_cbc_padding_decrypt(
            ctypes.byref(key_ctx), ivbuf, source, len(data), output, ctypes.byref(outlen),
        )
        if result != 1:
            raise ValueError("GmSSL SM4-CBC decryption failed")
        return ctypes.string_at(ctypes.addressof(output), outlen.value)

    def _openssl_sm4_cbc(self, key: bytes, iv: bytes, data: bytes, *, encrypt: bool) -> bytes:
        lib = self._openssl
        direction = "encrypt" if encrypt else "decrypt"
        contexts = getattr(self._openssl_contexts, direction, None)
        if contexts is None:
            ctx = lib.EVP_CIPHER_CTX_new()
            contexts = [ctx]
            setattr(self._openssl_contexts, direction, contexts)
        ctx = contexts[0]
        if not ctx:
            raise RuntimeError("OpenSSL cipher context allocation failed")
        output = (_U8 * (len(data) + 16))()
        total = ctypes.c_int()
        final = ctypes.c_int()
        if lib.EVP_CIPHER_CTX_reset(ctx) != 1:
            raise RuntimeError("OpenSSL cipher context reset failed")
        if lib.EVP_CipherInit_ex(
            ctx, self._openssl_cipher, None, _pointer(key), _pointer(iv), int(encrypt)
        ) != 1:
            raise ValueError("OpenSSL SM4-CBC initialization failed")
        if len(data) > 0x7FFFFFFF:
            raise ValueError("SM4 input exceeds the OpenSSL EVP single-update limit")
        if lib.EVP_CipherUpdate(
            ctx, output, ctypes.byref(total), _pointer(data), len(data)
        ) != 1:
            raise ValueError("OpenSSL SM4-CBC update failed")
        outptr = ctypes.cast(ctypes.byref(output, total.value), _U8P)
        if lib.EVP_CipherFinal_ex(ctx, outptr, ctypes.byref(final)) != 1:
            raise ValueError("OpenSSL SM4-CBC padding validation failed")
        return ctypes.string_at(ctypes.addressof(output), total.value + final.value)

    @functools.lru_cache(maxsize=16)
    def _private_key(self, private_key: str, public_key: str) -> _Sm2Key:
        private = bytes.fromhex(private_key)
        # The project's historical key is a 33-byte, zero-padded hex scalar.
        # GmSSL's fixed-width C API accepts the canonical 32-byte scalar.
        if len(private) == 33 and private[0] == 0:
            private = private[1:]
        if len(private) != 32:
            raise ValueError("SM2 private key must be 32 bytes")
        scalar, point, key = _Sm2Scalar(), _Sm2Point(), _Sm2Key()
        self.lib.sm2_z256_from_bytes(ctypes.byref(scalar), _buffer(private))
        public = _public_bytes(public_key)
        if self.lib.sm2_z256_point_from_bytes(ctypes.byref(point), _buffer(public)) != 1:
            raise ValueError("invalid SM2 public key")
        if self.lib.sm2_key_set_public_key(ctypes.byref(key), ctypes.byref(point)) != 1:
            raise ValueError("invalid SM2 public key")
        # set_public_key initializes the complete key structure, so install
        # the private scalar last (otherwise it is cleared by that call).
        if self.lib.sm2_key_set_private_key(ctypes.byref(key), ctypes.byref(scalar)) != 1:
            raise ValueError("invalid SM2 private key")
        return key

    @functools.lru_cache(maxsize=16)
    def _public_key(self, public_key: str) -> _Sm2Key:
        point, key = _Sm2Point(), _Sm2Key()
        if self.lib.sm2_z256_point_from_bytes(ctypes.byref(point), _buffer(_public_bytes(public_key))) != 1:
            raise ValueError("invalid SM2 public key")
        if self.lib.sm2_key_set_public_key(ctypes.byref(key), ctypes.byref(point)) != 1:
            raise ValueError("invalid SM2 public key")
        return key

    def _message_digest(self, public_key: str, data: bytes) -> bytes:
        key = self._public_key(public_key)
        za = (_U8 * 32)()
        identity = b"1234567812345678"
        if self.lib.sm2_compute_z(za, ctypes.byref(key.public_key), identity, len(identity)) != 1:
            raise ValueError("failed to compute SM2 ZA digest")
        ctx, output = _Sm3Context(), (_U8 * 32)()
        self.lib.sm3_init(ctypes.byref(ctx))
        self.lib.sm3_update(ctypes.byref(ctx), za, 32)
        source = _buffer(data)
        self.lib.sm3_update(ctypes.byref(ctx), _pointer(data), len(data))
        self.lib.sm3_finish(ctypes.byref(ctx), output)
        return bytes(output)

    def sm2_sign(self, private_key: str, public_key: str, data: bytes, random_hex: str) -> str:
        del random_hex  # Native GmSSL obtains per-signature randomness from its CSPRNG.
        key = self._private_key(private_key, public_key)
        digest, signature = self._message_digest(public_key, data), _Sm2Signature()
        if self.lib.sm2_do_sign(ctypes.byref(key), _buffer(digest), ctypes.byref(signature)) != 1:
            raise RuntimeError("GmSSL SM2 signing failed")
        return bytes(signature.r).hex() + bytes(signature.s).hex()

    def sm2_verify(self, public_key: str, data: bytes, signature: str) -> bool:
        try:
            raw = bytes.fromhex(signature)
            if len(raw) != 64:
                return False
            key = self._public_key(public_key)
            parsed = _Sm2Signature.from_buffer_copy(raw)
            digest = self._message_digest(public_key, data)
            return self.lib.sm2_do_verify(ctypes.byref(key), _buffer(digest), ctypes.byref(parsed)) == 1
        except (ValueError, TypeError):
            return False


NativeBackend = GmSSLNativeBackend


@functools.lru_cache(maxsize=1)
def get_crypto_backend() -> CryptoBackend:
    requested = os.environ.get("FACECHECK_CRYPTO_BACKEND", "auto").lower()
    if requested not in {"auto", "python", "native"}:
        raise RuntimeError("FACECHECK_CRYPTO_BACKEND must be auto, python, or native")
    if requested == "python":
        return PythonBackend()
    try:
        return GmSSLNativeBackend()
    except (OSError, RuntimeError):
        if requested == "native":
            raise
        return PythonBackend()
