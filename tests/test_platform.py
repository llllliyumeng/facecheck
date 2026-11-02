"""Bounded platform integration; fake embeddings, real storage/authentication and Runner."""
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from facecheck.main import create_app
from facecheck.database import TaskRepository
from facecheck.platform_catalog import PlatformCatalog, ROOT, safe_file
from facecheck.platform_jobs import JobManager, JobStore, JobConflict
from facecheck.platform_worker import run_job
from facecheck.ytf_runner import FaceFeature, FeatureExtractionResult


class TinyCatalog(PlatformCatalog):
    """Registered test fixture, injected in tests only, never an HTTP path."""
    def inputs(self, dataset_id):
        if dataset_id != 'ytf_smoke20':
            raise KeyError(dataset_id)
        return self.root / 'manifest.json', self.root / 'split.json', 'full'

    def dataset(self, dataset_id):
        self.inputs(dataset_id)
        return dict(id=dataset_id, name='fixture', available=True, scope='full', estimated_minutes='tiny')


@pytest.fixture
def tiny(tmp_path):
    root = tmp_path / 'fixture'
    root.mkdir()
    (root / 'models').mkdir()
    for name in ['face_detection_yunet_2023mar.onnx', 'face_recognition_sface_2021dec.onnx', 'arcface_w600k_r50.onnx']:
        (root / 'models' / name).write_bytes(b'fake model only used with fake embeddings')
    rows, assignments = [], []
    for i in range(4):
        identity = f'id_{i}'
        assignments.append(dict(identity_id=identity, split='threshold_selection' if i < 2 else 'test'))
        for role in ('gallery', 'query'):
            media_id = f'{role}_{i}'
            path = root / (media_id + ('.mp4' if role == 'gallery' else '.png'))
            path.write_bytes((media_id * 20).encode())
            rows.append(dict(identity_id=identity, video_id=media_id, video_path=path.name,
                             split=role, source_group_id=media_id, label=identity,
                             media_type='video' if role == 'gallery' else 'image'))
    (root / 'manifest.json').write_text(json.dumps(dict(records=rows)))
    (root / 'split.json').write_text(json.dumps(dict(seed='fixture', assignments=assignments)))
    return TinyCatalog(root)


class FakeFeatures:
    calls = []
    def __init__(self, detection, embedding, **kwargs):
        self.dimension = embedding.expected_dimension

    def extract(self, record):
        self.calls.append((record['video_id'], record.get('_experiment_mode')))
        vec = np.zeros(self.dimension, np.float32)
        vec[int(record['identity_id'].split('_')[-1])] = 1
        return FeatureExtractionResult((FaceFeature(vec, 3, 3.0),), 1, 1, 0, 0,
                                       dict(face_detection_seconds=.001, feature_extraction_seconds=.001, video_read_decode_extract_seconds=.001, frame_extraction_seconds=None, query_image_decode_seconds=None))


def enqueue(manager, monkeypatch, **kwargs):
    class NoChild:
        pid = os.getpid()
        def wait(self):
            return 0
    # Keep queued until test explicitly calls the real worker function.
    monkeypatch.setattr(manager, 'spawn_worker', lambda *a, **k: NoChild())
    monkeypatch.setattr(manager, 'watch_process', lambda *args: None)
    manager.store.reconcile = lambda: None
    config = manager.prepare('ytf_smoke20', 'sface', **kwargs)
    return manager.launch(config)


def test_real_evidence_and_requirements_are_file_backed():
    catalog = PlatformCatalog()
    actual = json.loads((ROOT / 'data/ytf/experiments/ytf_arcface_test_20261101/metrics.json').read_text())
    history = catalog.history('arcface_test')
    assert history['metrics'] == actual
    assert history['readonly'] and history['complete']
    assert catalog.history('models_sample100')['complete']
    assert catalog.history('models_sample100')['kind'] == 'exploratory'
    requirement = catalog.requirements()
    assert len(requirement['research']) == 8
    assert len(requirement['indicators']) == 18
    rows = {r['requirement']: r for r in requirement['indicators']}
    assert rows['明文 Top-1']['actual'] == actual['plaintext']['retrieval']['top1_accuracy']['value']
    assert rows['SM4 完整加/解密吞吐']['status'] == '未达标'
    assert rows['密文域结果一致率']['status'] == '未实现'
    assert rows['日均处理能力']['actual'] is None
    assert rows['上传与检索端到端平均响应']['status'] == '部分验证'
    assert rows['上传与检索端到端平均响应']['actual']['cold_seconds'] > 2
    assert rows['终端/分辨率/并发']['status'] == '部分验证'
    assert rows['系统接口调用成功率']['status'] == '部分验证'
    assert rows['审计日志完整率']['status'] == '未测试'
    assert '尾部截断' in rows['审计日志完整率']['note']
    assert rows['视频与特征篡改检出率']['actual']['video_trials'] == 100
    assert catalog.dataset('ytf_selection')['identity_count'] == 502


def test_safe_files_refuse_traversal_symlinks_and_unknown(tmp_path):
    directory = tmp_path / 'result'; directory.mkdir()
    (directory / 'metrics.json').write_text('{}')
    assert safe_file(directory, 'metrics.json').is_file()
    for name in ['../metrics.json', '.env', '/etc/passwd', 'model.onnx']:
        with pytest.raises(ValueError): safe_file(directory, name)
    secret = tmp_path / 'secret'; secret.write_text('not exposed')
    (directory / 'report.md').symlink_to(secret)
    with pytest.raises(FileNotFoundError): safe_file(directory, 'report.md')


def test_job_single_slot_cancel_persistence_and_no_overwrite(tmp_path, tiny, monkeypatch):
    manager = JobManager(tmp_path / 'db.sqlite', tmp_path / 'runs', tiny)
    job = enqueue(manager, monkeypatch)
    original = Path(job['output_dir']) / 'request.json'
    saved = original.read_bytes()
    assert job['status'] == 'queued'
    with pytest.raises(JobConflict): enqueue(manager, monkeypatch)
    assert original.read_bytes() == saved
    assert len(list((tmp_path / 'runs').iterdir())) == 1
    assert manager.store.cancel(job['id'])['cancel_requested']
    with pytest.raises(JobConflict): enqueue(manager, monkeypatch)
    run_job(manager.store, job['id'], catalog=tiny, backend_type=FakeFeatures)
    assert JobStore(tmp_path / 'db.sqlite').get(job['id'])['status'] == 'cancelled'
    assert not (Path(job['output_dir']) / 'result').exists()
    next_job = enqueue(manager, monkeypatch)
    assert next_job['id'] != job['id']
    assert len(manager.store.logs(job['id'])) >= 3


def test_batch_real_runner_authenticated_storage_and_cached_media(tmp_path, tiny, monkeypatch):
    manager = JobManager(tmp_path / 'db.sqlite', tmp_path / 'runs', tiny)
    job = enqueue(manager, monkeypatch)
    FakeFeatures.calls = []
    run_job(manager.store, job['id'], catalog=tiny, backend_type=FakeFeatures)
    result = manager.store.get(job['id'])
    assert result['status'] == 'completed', result
    assert result['processed'] == result['total'] == 16
    assert len(FakeFeatures.calls) == len(set(FakeFeatures.calls)) == 16
    directory = manager.result_dir(job['id'])
    metrics = json.loads((directory / 'metrics.json').read_text())
    assert metrics['mode_consistency']['consistent']
    assert metrics['encryption_integrity']['all_source_restore_sha256_equal']
    assert 'report.md' in [r['name'] for r in tiny.result(directory)['artifacts']]
    assert any(e['event_type'] == 'experiment_completed' for e in TaskRepository(manager.store.database).list_audit(100, 0))


@pytest.mark.parametrize('mode', ['plaintext', 'encrypted_restore'])
def test_live_retrieval_and_identity_disjoint_candidates(tmp_path, tiny, monkeypatch, mode):
    manager = JobManager(tmp_path / 'db.sqlite', tmp_path / 'runs', tiny)
    job = enqueue(manager, monkeypatch, kind='retrieval', query_id='query_0', mode=mode)
    FakeFeatures.calls = []
    run_job(manager.store, job['id'], catalog=tiny, backend_type=FakeFeatures)
    done = manager.store.get(job['id'])
    assert done['status'] == 'completed', done
    assert done['processed'] == done['total'] == 3
    result = json.loads((manager.result_dir(job['id']) / 'result.json').read_text())
    assert result['gallery_count'] == 2
    assert {r['identity_id'] for r in result['results']} == {'id_0', 'id_1'}
    assert result['results'][0]['identity_id'] == 'id_0'
    assert result['results'][0]['candidate_frame_index'] == 3
    assert len(FakeFeatures.calls) == 3
    assert len(result['restore_integrity']) == (2 if mode == 'encrypted_restore' else 0)


def test_modified_model_and_worker_exception_are_incomplete(tmp_path, tiny, monkeypatch):
    manager = JobManager(tmp_path / 'db.sqlite', tmp_path / 'runs', tiny)
    job = enqueue(manager, monkeypatch)
    (tiny.root / 'models/face_recognition_sface_2021dec.onnx').write_bytes(b'changed')
    run_job(manager.store, job['id'], catalog=tiny, backend_type=FakeFeatures)
    result = manager.store.get(job['id'])
    assert result['status'] == 'failed' and not result['complete']
    assert '发生变化' in result['error']
    assert not manager.result_dir(job['id']).exists()


def test_dead_worker_reconciliation_and_completed_survives_restart(tmp_path, tiny, monkeypatch):
    manager = JobManager(tmp_path / 'db.sqlite', tmp_path / 'runs', tiny)
    job = enqueue(manager, monkeypatch)
    with manager.store.connect() as db:
        db.execute("UPDATE experiment_jobs SET pid=99999999,status='running' WHERE id=?", (job['id'],))
    # Restore method shadowed by fixture helper; ps finds no process.
    JobStore.reconcile(manager.store)
    assert manager.store.get(job['id'])['status'] == 'failed'
    second = enqueue(manager, monkeypatch)
    run_job(manager.store, second['id'], catalog=tiny, backend_type=FakeFeatures)
    restarted = JobManager(manager.store.database, manager.output_root, tiny)
    assert restarted.store.get(second['id'])['status'] == 'completed'


def test_api_validation_history_download_and_audit_export(tmp_path):
    app = create_app(database_path=tmp_path / 'db.sqlite', storage_root=tmp_path / 'storage')
    with TestClient(app) as client:
        for endpoint in ['overview', 'datasets', 'histories', 'requirements', 'benchmarks', 'jobs']:
            assert client.get('/api/v1/platform/' + endpoint).status_code == 200
        for config in [dict(dataset_id='/etc/passwd'), dict(command='rm'), dict(model='other'), dict(kind='test'), dict(top_k=0)]:
            assert client.post('/api/v1/platform/jobs', json=config).status_code == 422
        assert client.post('/api/v1/platform/jobs', json=dict(kind='retrieval', dataset_id='ytf_selection', query_id='q')).status_code == 400
        assert client.get('/api/v1/platform/histories/arcface_test/files/.env').status_code == 404
        assert client.get('/api/v1/platform/histories/nope').status_code == 404
        assert client.get('/api/v1/platform/histories/arcface_test/files/metrics.json').status_code == 200
        assert client.get('/api/v1/platform/jobs/nope').status_code == 404
        preview = client.post('/api/v1/platform/jobs/preview', json={})
        assert preview.status_code == 200 and preview.json()['config']['formal'] is False
        assert client.get('/api/v1/platform/jobs').json()['total'] == 0
        repo = TaskRepository(tmp_path / 'db.sqlite')
        for i in range(205): repo.record_event(None, 'example', str(i))
        exported = client.get('/api/v1/platform/audit/export?format=json').json()
        assert exported['count'] == 205
        assert client.get('/api/v1/platform/audit/export?format=md').status_code == 200
        assert client.get('/api/v1/platform/audit/export?format=csv').headers['content-disposition'].endswith('.csv"')


def test_single_video_verify_download_and_tamper_rejection(tmp_path):
    source = tmp_path / 'fixture.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 4, (16, 16))
    assert writer.isOpened()
    writer.write(np.zeros((16, 16, 3), np.uint8)); writer.release()
    app = create_app(database_path=tmp_path / 'db.sqlite', storage_root=tmp_path / 'storage', encryption_key=os.urandom(16), mac_key=os.urandom(32))
    with TestClient(app) as client:
        task = client.post('/api/v1/videos', files={'file': ('fixture.mp4', source.read_bytes(), 'video/mp4')}).json()
        assert task['status'] == 'completed'
        path = f"/api/v1/platform/videos/{task['task_id']}"
        verified = client.post(path + '/verify')
        assert verified.status_code == 200
        assert verified.json()['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
        restored = client.post(path + '/restore')
        assert restored.content == source.read_bytes()
        directory = tmp_path / 'storage' / task['storage_id']
        chunk = next((directory / 'chunks').iterdir())
        value = bytearray(chunk.read_bytes());value[len(value)//2] ^= 1;chunk.write_bytes(value)
        assert client.post(path + '/verify').status_code == 400
        assert client.post(path + '/restore').status_code == 400


def test_actual_subprocess_safe_cancellation(tmp_path, tiny, monkeypatch):
    manager = JobManager(tmp_path / 'db.sqlite', tmp_path / 'runs', tiny)
    original = manager.spawn_worker
    def cancel_before_spawn(args, **kwargs):
        # Verify real subprocess startup without running any inference/dataset experiment.
        manager.store.cancel(args[-1])
        return original(args, **kwargs)
    monkeypatch.setattr(manager, 'spawn_worker', cancel_before_spawn)
    job = manager.launch(manager.prepare('ytf_smoke20', 'sface'))
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        done = manager.store.get(job['id'])
        if done['status'] not in ('queued', 'running'): break
        time.sleep(.05)
    assert done['status'] == 'cancelled', done
    assert not manager.result_dir(job['id']).exists()


def test_history_directory_symlink_and_result_symlink_escape(tmp_path, tiny, monkeypatch):
    history_parent = tiny.root / 'data/ytf/experiments'
    history_parent.mkdir(parents=True)
    external = tmp_path / 'external'; external.mkdir()
    (history_parent / 'ytf_arcface_test_20261101').symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError): tiny.history('arcface_test')
    manager = JobManager(tmp_path / 'db.sqlite', tmp_path / 'runs', tiny)
    job = enqueue(manager, monkeypatch)
    (Path(job['output_dir']) / 'result').symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError): manager.result_dir(job['id'])


def test_running_cancellation_does_not_publish_partial_bundle(tmp_path, tiny, monkeypatch):
    manager = JobManager(tmp_path / 'db.sqlite', tmp_path / 'runs', tiny)
    job = enqueue(manager, monkeypatch)
    class CancelOnFirst(FakeFeatures):
        def extract(self, record):
            result = super().extract(record)
            manager.store.cancel(job['id'])
            return result
    run_job(manager.store, job['id'], catalog=tiny, backend_type=CancelOnFirst)
    result = manager.store.get(job['id'])
    assert result['status'] == 'cancelled' and result['processed'] == 1
    assert not manager.result_dir(job['id']).exists()
    assert not list(Path(job['output_dir']).glob('.*staging*'))


def test_diagnostic_preview_limits_and_formal_test_cannot_be_submitted(tmp_path):
    app = create_app(database_path=tmp_path / 'db.sqlite', storage_root=tmp_path / 'storage')
    with TestClient(app) as client:
        prefix = '/api/v1/platform'
        preview = client.get(prefix + '/histories/arcface_test/preview/query_results.csv?limit=2')
        assert preview.status_code == 200
        assert len(preview.json()['items']) == 2 and preview.json()['truncated']
        assert client.get(prefix + '/histories/arcface_test/preview/metrics.json').status_code == 404
        assert client.get(prefix + '/histories/arcface_test/preview/query_results.csv?limit=1000').status_code == 422
        request = dict(dataset_id='ytf_selection', model='arcface')
        config = client.post(prefix + '/jobs/preview', json=request).json()['config']
        assert config['evaluation_scope'] == 'threshold_selection'
        assert config['total'] == 2008
        assert config['mode'] == 'paired' and not config['formal']
        for field in ['evaluation_scope', 'output_dir', 'manifest_path', 'frozen_selection_threshold']:
            assert client.post(prefix + '/jobs', json={**request, field: 'test'}).status_code == 422
        assert client.get(prefix + '/jobs').json()['total'] == 0
