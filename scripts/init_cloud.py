#!/usr/bin/env python3
"""Create private, persistent runtime credentials without printing any secret."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
from urllib.parse import urlsplit


def initialize(directory: Path, public_origin: str) -> Path:
    parsed = urlsplit(public_origin)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
            or parsed.path not in {'', '/'} or parsed.query or parsed.fragment):
        raise ValueError('public origin must be https://host[:port] without a path')
    directory = directory.expanduser().resolve()
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    def private(name, data):
        with os.fdopen(os.open(directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as fp:
            fp.write(data)
    private('sm4.key', secrets.token_bytes(16))
    private('mac.key', secrets.token_bytes(32))
    principals = []
    for role in ['admin', 'operator', 'auditor']:
        token = secrets.token_urlsafe(32)
        private(role + '.token', token.encode('ascii'))
        principals.append(dict(id=role, role=role, token_sha256=hashlib.sha256(token.encode()).hexdigest()))
    private('credentials.json', json.dumps(dict(version=1, principals=principals), indent=2).encode())
    root = Path(__file__).resolve().parents[1]
    env = dict(FACECHECK_DEPLOYMENT='cloud', FACECHECK_PUBLIC_ORIGIN=public_origin.rstrip('/'),
               FACECHECK_DATA_ROOT=str(directory/'runtime'),
               FACECHECK_DATABASE=str(directory/'runtime/facecheck.sqlite3'),
               FACECHECK_STORAGE_ROOT=str(directory/'runtime/storage'),
               FACECHECK_AUTH_CREDENTIALS_FILE=str(directory/'credentials.json'),
               FACECHECK_SM4_KEY_FILE=str(directory/'sm4.key'), FACECHECK_MAC_KEY_FILE=str(directory/'mac.key'),
               FACECHECK_CRYPTO_BACKEND='auto', FACECHECK_SM4_NATIVE_IMPL='gmssl',
               FACECHECK_FEATURE_CACHE='1',
               FACECHECK_RECOGNITION_MODEL='arcface', FACECHECK_YUNET_MODEL=str(root/'models/face_detection_yunet_2023mar.onnx'),
               FACECHECK_ARCFACE_MODEL=str(root/'models/arcface_w600k_r50.onnx'),
               FACECHECK_SFACE_MODEL=str(root/'models/face_recognition_sface_2021dec.onnx'))
    # JSON-quoted values are parsed by uvicorn's dotenv reader; source this file only via that reader.
    private('cloud.env', ''.join(f'{k}={json.dumps(v)}\n'for k,v in env.items()).encode())
    return directory/'cloud.env'


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--public-origin', required=True)
    args=parser.parse_args()
    try: config=initialize(args.directory,args.public_origin)
    except (ValueError,OSError)as exc:parser.exit(1,f'Initialization failed ({type(exc).__name__}); existing files were not overwritten.\n')
    print(f'Private runtime configuration created: {config}\nNo secret values were printed. Keep the directory and its backups private.')
