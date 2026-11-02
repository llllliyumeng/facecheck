"""Authenticated, encrypted online feature cache; never used by formal YTF runs.

A hit is accepted only for byte-identical encrypted storage plus identical model,
preprocessing and extraction code. Cache payloads reuse existing SM4/HMAC APIs
with separate purpose-derived keys. This is trusted-server exact search, not
ciphertext ANN or a cancellable biometric template.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import threading
import zipfile

import numpy as np

from .crypto import sm3_kdf, sm4_encrypt, sm4_decrypt
from .face_search import FaceCandidate
from .storage import StorageError

MAX_CACHE_BYTES = 64 * 1024 * 1024
MAX_SOURCE_BYTES = 256 * 1024 * 1024
SCHEMA = 'facecheck-online-frame-features-v1'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode('utf-8')


def _hash_file(path, limit):
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        with os.fdopen(fd, 'rb') as fp:
            info = os.fstat(fp.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise StorageError('feature source exceeds its safe file limit')
            h = hashlib.sha256(); size = 0
            for block in iter(lambda: fp.read(1024*1024), b''):
                size += len(block)
                if size > limit:raise StorageError('feature source exceeds its safe file limit')
                h.update(block)
            return h.hexdigest(), size
    except OSError:
        raise StorageError('feature source is unavailable or unsafe') from None


def storage_fingerprint(directory, expected_manifest_sha256=None):
    directory=Path(directory);chunks=directory/'chunks';manifest=directory/'manifest.json'
    if directory.is_symlink() or not directory.is_dir() or chunks.is_symlink() or not chunks.is_dir():
        raise StorageError('feature source directory is unsafe')
    mh,size=_hash_file(manifest,1024*1024)
    if expected_manifest_sha256 is not None and mh != expected_manifest_sha256:
        raise StorageError('feature source manifest does not match its trusted anchor')
    files=sorted(chunks.iterdir())
    if len(files)>512:raise StorageError('too many feature source chunks')
    records=[]
    for p in files:
        if not re.fullmatch(r'[0-9]{8}\.bin',p.name):raise StorageError('unexpected feature source chunk')
        h,n=_hash_file(p,MAX_SOURCE_BYTES-size);size+=n;records.append((p.name,h,n))
    return hashlib.sha256(_canonical(dict(manifest=mh,chunks=records))).hexdigest()


def representation_fingerprint(config):
    models=[config.detection.model_path,config.embedding.model_path]
    hashes=[_hash_file(Path(p),512*1024*1024)[0]for p in models]
    code={name:_hash_file(Path(__file__).with_name(name),1024*1024)[0]for name in
          ['face_embedding.py','face_detection.py','face_search.py','recognition.py','video.py']}
    return hashlib.sha256(_canonical(dict(schema=SCHEMA,models=hashes,code=code,
                         detection=asdict(config.detection),embedding=asdict(config.embedding),
                         clip_detections_to_image=config.clip_detections_to_image))).hexdigest()


class FeatureStore:
    def __init__(self, root, encryption_master, mac_master):
        self.root=Path(root)
        if self.root.is_symlink():raise StorageError('feature cache root is unsafe')
        self.encryption_key=sm3_kdf(encryption_master,b'facecheck-feature-cache-encryption-v1',16)
        self.mac_key=sm3_kdf(mac_master,b'facecheck-feature-cache-authentication-v1',32)
        self._guard=threading.Lock();self._locks={}

    def lock(self, video_id):
        with self._guard:return self._locks.setdefault(video_id,threading.Lock())

    def inventory_video(self, video_id):
        """Return a digest inventory for one cache directory without decrypting it."""
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', video_id):
            raise StorageError('invalid feature cache video identifier')
        if self.root.is_symlink():
            raise StorageError('protected feature cache root is unsafe')
        directory = self.root / video_id
        if not directory.exists():
            return []
        if directory.is_symlink() or not directory.is_dir():
            raise StorageError('protected feature cache directory is unsafe')
        items = []
        for path in sorted(directory.iterdir()):
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CACHE_BYTES:
                raise StorageError('protected feature cache contains an unsafe entry')
            digest, size = _hash_file(path, MAX_CACHE_BYTES)
            items.append({'reference': str(path), 'sha256': digest, 'size': size})
        return items

    def inventory_all(self):
        """Inventory cache directories; callers still need to attest the cache root."""
        if self.root.is_symlink():
            raise StorageError('protected feature cache root is unsafe')
        if not self.root.exists():
            return {}
        if not self.root.is_dir():
            raise StorageError('protected feature cache root is unsafe')
        result = {}
        for directory in sorted(self.root.iterdir()):
            if directory.is_symlink() or not directory.is_dir():
                raise StorageError('protected feature cache contains an unsafe entry')
            result[directory.name] = self.inventory_video(directory.name)
        return result

    def invalidate(self, video_id):
        """Delete one video's derived cache so it cannot outlive key migration."""
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', video_id):
            raise StorageError('invalid feature cache video identifier')
        with self.lock(video_id):
            items = self.inventory_video(video_id)
            directory = self.root / video_id
            if directory.exists():
                shutil.rmtree(directory)
            return {'video_id': video_id, 'deleted_files': len(items),
                    'deleted_sha256': [item['sha256'] for item in items]}

    def context(self, video_id, source, representation):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',video_id):raise StorageError('invalid feature video identifier')
        aad=_canonical(dict(schema=SCHEMA,video_id=video_id,source=source,representation=representation))
        identity=hashlib.sha256(aad).hexdigest()
        return self.root/video_id/(identity+'.bin'),aad

    def get(self, path, aad, dimension):
        if not path.exists():return None
        if path.is_symlink() or path.parent.is_symlink() or path.stat().st_size>MAX_CACHE_BYTES:
            raise StorageError('protected feature cache is unsafe')
        try:
            # Authenticate before decrypting, and decrypt before parsing any archive.
            raw=sm4_decrypt(path.read_bytes(),self.encryption_key,self.mac_key,associated_data=aad)
            if len(raw)>MAX_CACHE_BYTES:raise ValueError
            with zipfile.ZipFile(io.BytesIO(raw))as archive:
                if sum(i.file_size for i in archive.infolist())>MAX_CACHE_BYTES:raise ValueError
                if set(archive.namelist())!={'vectors.npy','ids.npy','frames.npy','times.npy'}:raise ValueError
            with np.load(io.BytesIO(raw),allow_pickle=False)as data:
                vectors=data['vectors'];ids=data['ids'];frames=data['frames'];times=data['times']
                self._validate(vectors,ids,frames,times,dimension)
                return [FaceCandidate(str(ids[i]),int(frames[i]),float(times[i]),vectors[i].copy())for i in range(len(ids))]
        except (ValueError,OSError,KeyError,zipfile.BadZipFile,EOFError,TypeError):
            # A corrupt authenticated cache is rejected, never silently used or downgraded.
            raise StorageError('protected feature cache authentication or validation failed') from None

    @staticmethod
    def _validate(vectors,ids,frames,times,dimension):
        n=len(ids)
        if (vectors.dtype!=np.float32 or vectors.shape!=(n,dimension) or ids.shape!=(n,)
                or ids.dtype.kind!='U' or frames.shape!=(n,) or frames.dtype.kind not in 'iu'
                or times.shape!=(n,) or times.dtype.kind!='f'
                or not np.all(np.isfinite(vectors)) or not np.all(np.isfinite(times))
                or np.any(frames<0) or np.any(times<0)
                or len(set(ids.tolist()))!=n
                or any(not re.fullmatch(r'frame-[0-9]+-face-[0-9]+',x)for x in ids.tolist())
                or (n and not np.allclose(np.linalg.norm(vectors,axis=1),1.0,atol=1e-5))):
            raise ValueError('invalid feature representation')

    def put(self, path, aad, candidates, dimension):
        vectors=np.asarray([c.vector for c in candidates],dtype=np.float32).reshape((-1,dimension))
        ids=np.asarray([c.candidate_id for c in candidates],dtype='U80')
        frames=np.asarray([c.frame_index for c in candidates],dtype=np.int64)
        times=np.asarray([c.timestamp_seconds for c in candidates],dtype=np.float64)
        self._validate(vectors,ids,frames,times,dimension)
        buf=io.BytesIO();np.savez(buf,vectors=vectors,ids=ids,frames=frames,times=times)
        raw=buf.getvalue()
        if len(raw)>MAX_CACHE_BYTES-256:raise StorageError('feature cache exceeds memory limit')
        sealed=sm4_encrypt(raw,self.encryption_key,self.mac_key,associated_data=aad)
        self.root.mkdir(mode=0o700,parents=True,exist_ok=True)
        if self.root.is_symlink():raise StorageError('feature cache root is unsafe')
        path.parent.mkdir(mode=0o700,exist_ok=True)
        if path.parent.is_symlink():raise StorageError('feature cache directory is unsafe')
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.feature-',suffix='.partial',delete=False)as fp:
                temporary=Path(fp.name);fp.write(sealed);fp.flush();os.fsync(fp.fileno())
            os.replace(temporary,path)
        finally:
            if temporary is not None:temporary.unlink(missing_ok=True)
