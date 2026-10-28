"""Minimal dependency/API smoke test for stage two.

Run from the project root with:
    uv run python scripts/verify_crypto_dependencies.py
"""

from gmssl import sm2, sm3
from gmssl.sm4 import CryptSM4, SM4_DECRYPT, SM4_ENCRYPT


def main() -> None:
    assert hasattr(sm2, "CryptSM2")
    assert hasattr(sm3, "sm3_hash")
    assert CryptSM4 is not None
    assert SM4_ENCRYPT != SM4_DECRYPT
    print("gmssl SM2/SM3/SM4 API smoke test: OK")


if __name__ == "__main__":
    main()
