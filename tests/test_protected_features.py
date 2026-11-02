import os
from dataclasses import replace
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import pytest

from facecheck.database import TaskRepository
from facecheck.crypto import SM4_ENVELOPE_HEADER
from facecheck.face_detection import FaceDetectionConfig
from facecheck.face_embedding import ArcFaceConfig
from facecheck.face_search import FaceSearchConfig, FaceCandidate, rank_candidates
from facecheck.feature_store import FeatureStore, representation_fingerprint, storage_fingerprint
from facecheck.service import VideoTaskService
from facecheck.storage import StorageError, store_encrypted_video, restore_encrypted_video


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    monkeypatch.setenv('FACECHECK_FEATURE_CACHE','1')
    repo=TaskRepository(tmp_path/'db.sqlite');key=os.urandom(16);mac=os.urandom(32)
    svc=VideoTaskService(repo,tmp_path/'storage',encryption_key=key,mac_key=mac)
    source=tmp_path/'source';source.write_bytes(os.urandom(2048))
    stored=store_encrypted_video(source,svc.storage_root,key,mac,video_id='sample',chunk_size=1024,key_version='cache-test-v1')
    detector=tmp_path/'yunet.onnx';model=tmp_path/'arcface.onnx'
    detector.write_bytes(b'detector-test');model.write_bytes(b'embedding-test')
    cfg=FaceSearchConfig(FaceDetectionConfig(detector),ArcFaceConfig(model),top_k=5,min_similarity=None)
    q=np.eye(1,512,dtype=np.float32)[0]
    candidates=[FaceCandidate('frame-0-face-0',0,0.0,q.copy()),
                FaceCandidate('frame-9-face-1',9,1.5,np.eye(1,512,1,dtype=np.float32)[0])]
    calls=[]
    def extract(directory,query,config,encryption_key,mac_key,*,temp_root=None,stats=None,candidate_sink=None,**kwargs):
        calls.append(1)
        output=tmp_path/('restored-'+str(len(calls)))
        restore_encrypted_video(directory,output,encryption_key,mac_key)
        try:
            assert output.read_bytes()==source.read_bytes()
            if candidate_sink:candidate_sink(candidates)
            return rank_candidates(query,candidates,config,stats=stats)
        finally:output.unlink(missing_ok=True)
    monkeypatch.setattr('facecheck.service.search_encrypted_video',extract)
    return svc,stored,cfg,q,candidates,calls


def run(p):
    svc,stored,cfg,q,_,_=p;stats={}
    return svc._search_stored(stored.directory,q,cfg,None,stats),stats


def test_hit_preserves_exact_ranking_and_skips_repeated_restore(prepared):
    first,cold=run(prepared);second,warm=run(prepared)
    assert first==second and [r.similarity for r in first]==[1.0,0.0]
    assert cold['feature_cache_hit']is False and warm['feature_cache_hit']is True
    assert len(prepared[-1])==1
    cache=next(prepared[0].feature_store.root.rglob('*.bin')).read_bytes()
    assert not cache.startswith(b'PK') and prepared[3].tobytes()not in cache


def test_changed_query_reuses_same_gallery_features(prepared):
    run(prepared);svc,stored,cfg,_,_,calls=prepared
    q=np.eye(1,512,1,dtype=np.float32)[0];stats={}
    result=svc._search_stored(stored.directory,q,cfg,None,stats)
    assert stats['feature_cache_hit']is True and len(calls)==1
    assert result[0].candidate_id=='frame-9-face-1'


def test_cache_is_persistent_across_service_restart(prepared,monkeypatch):
    first,_=run(prepared);svc,stored,cfg,q,_,calls=prepared
    other=VideoTaskService(svc.repository,svc.storage_root,encryption_key=svc.encryption_key,mac_key=svc.mac_key)
    stats={};assert other._search_stored(stored.directory,q,cfg,None,stats)==first
    assert stats['feature_cache_hit']is True and len(calls)==1


def test_model_and_preprocessing_changes_invalidate(prepared):
    run(prepared);svc,stored,cfg,q,_,calls=prepared
    cfg.embedding.model_path.write_bytes(b'new-model-test')
    stats={};svc._search_stored(stored.directory,q,cfg,None,stats)
    assert stats['feature_cache_hit']is False and len(calls)==2
    assert representation_fingerprint(cfg)!=representation_fingerprint(replace(cfg,clip_detections_to_image=False))
    # Query-side thresholds and output limits do not alter the Gallery representation.
    assert representation_fingerprint(cfg)==representation_fingerprint(replace(cfg,top_k=1,min_similarity=.5))


def test_ciphertext_tamper_does_not_hide_behind_a_cache_hit(prepared):
    run(prepared);p=next((prepared[1].directory/'chunks').glob('*.bin'))
    value=bytearray(p.read_bytes());value[-1]^=1;p.write_bytes(value)
    with pytest.raises(ValueError,match='authentication'):run(prepared)


@pytest.mark.parametrize('offset',[-1,len(SM4_ENVELOPE_HEADER)+8+16])
def test_tampered_cache_ciphertext_or_mac_is_rejected(prepared,offset):
    run(prepared);p=next(prepared[0].feature_store.root.rglob('*.bin'))
    value=bytearray(p.read_bytes());value[offset]^=1;p.write_bytes(value)
    with pytest.raises(StorageError,match='authentication'):run(prepared)
    assert len(prepared[-1])==1 # no silent rebuild masking a corrupted cache


def test_wrong_cache_key_and_rebound_context_are_rejected(prepared):
    run(prepared);svc,stored,cfg,_,_,_=prepared;store=svc.feature_store
    p,aad=store.context(stored.video_id,storage_fingerprint(stored.directory),representation_fingerprint(cfg))
    other=FeatureStore(store.root,os.urandom(16),os.urandom(32))
    with pytest.raises(StorageError):other.get(p,aad,512)
    with pytest.raises(StorageError):store.get(p,aad+b'tampered',512)


def test_source_changed_during_extraction_never_publishes(prepared,monkeypatch):
    svc,stored,cfg,q,candidates,_=prepared
    def changing(*args,candidate_sink=None,**kwargs):
        p=stored.directory/'manifest.json';p.write_bytes(p.read_bytes()+b' ')
        candidate_sink(candidates)
    monkeypatch.setattr('facecheck.service.search_encrypted_video',changing)
    with pytest.raises(StorageError,match='changed'):run(prepared)
    assert not list(svc.feature_store.root.rglob('*.bin'))


def test_symlink_and_invalid_vector_rejected(prepared,tmp_path):
    svc,stored,cfg,q,candidates,_=prepared
    p=stored.directory/'chunks'/'00000000.bin';data=p.read_bytes();p.unlink()
    target=tmp_path/'elsewhere';target.write_bytes(data);p.symlink_to(target)
    with pytest.raises(StorageError):storage_fingerprint(stored.directory)
    path,aad=svc.feature_store.context('other','source','representation')
    invalid=[FaceCandidate('frame-0-face-0',0,0.,np.full(512,np.nan,dtype=np.float32))]
    with pytest.raises(ValueError):svc.feature_store.put(path,aad,invalid,512)


def test_single_flight_concurrent_queries_only_extract_once(prepared):
    with ThreadPoolExecutor(max_workers=4)as pool:
        results=list(pool.map(lambda _:run(prepared),range(4)))
    assert len(prepared[-1])==1
    assert sum(not stats['feature_cache_hit']for _,stats in results)==1
    assert all(r==results[0][0]for r,_ in results)


def test_empty_gallery_cache_preserves_no_results(prepared):
    svc,_,cfg,_,_,_=prepared;path,aad=svc.feature_store.context('empty','x','y')
    svc.feature_store.put(path,aad,[],512)
    assert svc.feature_store.get(path,aad,512)==[]


def test_anchored_restore_checks_the_parsed_snapshot_before_decrypt(prepared,tmp_path,monkeypatch):
    import hashlib
    svc,stored,cfg,q,_,_=prepared;path=stored.directory/'manifest.json'
    expected=hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(path.read_bytes()+b' ')
    def forbidden(*args,**kwargs):raise AssertionError('CBC must not run for an untrusted manifest')
    monkeypatch.setattr('facecheck.storage.sm4_decrypt',forbidden)
    with pytest.raises(StorageError,match='anchor'):
        restore_encrypted_video(stored.directory,tmp_path/'output',svc.encryption_key,svc.mac_key,expected_manifest_sha256=expected)
    with pytest.raises(StorageError,match='anchor'):storage_fingerprint(stored.directory,expected)
    assert not (tmp_path/'output').exists()
