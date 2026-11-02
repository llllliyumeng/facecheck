"""Targeted cloud RBAC / cookie / restart / concurrency regressions."""
import asyncio
import hashlib
import importlib.util
import json
import os
import threading
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from facecheck.main import create_app
from facecheck.security import COOKIE, secret_file


@pytest.fixture
def cloud(tmp_path, monkeypatch):
    spec=importlib.util.spec_from_file_location('cloud_init', Path(__file__).resolve().parents[1]/'scripts/init_cloud.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    folder=tmp_path/'private'; module.initialize(folder,'https://facecheck.test')
    for name in ['FACECHECK_SM4_KEY','FACECHECK_MAC_KEY']:
        monkeypatch.delenv(name,raising=False)
    values=dict(FACECHECK_DEPLOYMENT='cloud',FACECHECK_PUBLIC_ORIGIN='https://facecheck.test',
                FACECHECK_AUTH_CREDENTIALS_FILE=str(folder/'credentials.json'),
                FACECHECK_SM4_KEY_FILE=str(folder/'sm4.key'),FACECHECK_MAC_KEY_FILE=str(folder/'mac.key'))
    for k,v in values.items():monkeypatch.setenv(k,v)
    app=create_app(tmp_path/'db.sqlite',tmp_path/'storage')
    return app,folder


def client(app):return TestClient(app,base_url='https://facecheck.test')
def bearer(folder,role):return {'Authorization':'Bearer '+(folder/(role+'.token')).read_text()}
def login(c,folder,role='operator'):
    return c.post('/api/v1/auth/login',json={'token':(folder/(role+'.token')).read_text()},headers={'Origin':'https://facecheck.test'})


@pytest.mark.parametrize('path,method',[
    ('/api/v1/tasks','GET'),('/api/v1/audit','GET'),('/api/v1/videos','POST'),
    ('/api/v1/platform/histories','GET'),('/api/v1/platform/jobs','POST'),
    ('/api/v1/platform/videos/x/restore','POST'),('/api/v1/videos/x/search','POST'),
])
def test_anonymous_rejected_before_body_processing(cloud,path,method):
    app,_=cloud;c=client(app);r=c.request(method,path)
    assert r.status_code==401
    assert app.state.video_service.repository.count()==0
    assert r.headers['Cache-Control']=='no-store'


def test_auditor_is_not_read_only_plaintext_access(cloud):
    app,folder=cloud;c=client(app);h=bearer(folder,'auditor')
    assert c.get('/api/v1/audit',headers=h).status_code==200
    assert c.get('/api/v1/security/summary',headers=h).json()['authentication_required'] is True
    assert c.get('/api/v1/tasks',headers=h).status_code==403
    assert c.get('/api/v1/platform/key-rotations',headers=h).status_code==403
    assert c.get('/api/v1/platform/key-rotations',headers=bearer(folder,'admin')).status_code==200
    assert c.post('/api/v1/platform/key-rotations/missing/revoke',headers=bearer(folder,'admin')).status_code==503
    assert c.post('/api/v1/platform/key-rotations/plan',
                  json={'source_key_id':'legacy-api-v1','target_key_id':'recipient-next'},
                  headers=bearer(folder,'admin')).status_code==400
    assert c.post('/api/v1/platform/videos/x/restore',headers=h).status_code==403
    assert c.post('/api/v1/videos/x/search',headers=h).status_code==403
    assert c.post('/api/v1/platform/jobs/preview',headers=h).status_code==403


def test_cookie_flags_csrf_and_logout_revocation(cloud):
    app,folder=cloud;c=client(app);r=login(c,folder)
    assert r.status_code==200
    flags=r.headers['set-cookie'].lower()
    assert all(x in flags for x in ['httponly','secure','samesite=strict'])
    saved=c.cookies.get(COOKIE)
    assert c.get('/api/v1/tasks').status_code==200
    assert c.post('/api/v1/videos').status_code==403
    assert c.post('/api/v1/videos',headers={'Origin':'https://evil.test'}).status_code==403
    assert c.post('/api/v1/auth/logout',headers={'Origin':'https://facecheck.test'}).status_code==200
    c.cookies.set(COOKIE,saved,domain='facecheck.test',path='/')
    assert c.get('/api/v1/tasks').status_code==401


def test_credential_file_change_revokes_bearer_and_session(cloud):
    app,folder=cloud;c=client(app);assert login(c,folder).status_code==200
    data=json.loads((folder/'credentials.json').read_text())
    data['principals']=[r for r in data['principals']if r['role']!='operator']
    (folder/'credentials.json').write_text(json.dumps(data))
    assert c.get('/api/v1/tasks').status_code==401
    assert c.get('/api/v1/tasks',headers=bearer(folder,'operator')).status_code==401


def test_sessions_survive_restart_and_never_store_plaintext_token(cloud):
    app,folder=cloud;c=client(app);assert login(c,folder).status_code==200
    cookie=c.cookies.get(COOKIE)
    other=create_app(app.state.video_service.repository.path,app.state.video_service.storage_root)
    restarted=client(other);restarted.cookies.set(COOKIE,cookie,domain='facecheck.test',path='/')
    assert restarted.get('/api/v1/tasks').status_code==200
    raw=app.state.video_service.repository.path.read_bytes()
    assert cookie.encode()not in raw
    for f in folder.glob('*.token'):assert f.read_bytes()not in raw


def test_fail_closed_on_bad_credential_file_host_https_and_schema(cloud):
    app,folder=cloud;c=client(app)
    assert c.get('/docs').status_code==404
    assert c.get('/openapi.json').status_code==404
    assert c.get('/api/v1/tasks',headers={'Host':'evil.test'}).status_code==400
    assert TestClient(app,base_url='http://facecheck.test').get('/api/v1/auth/status').status_code==400
    assert TestClient(app,base_url='http://localhost').get('/health').status_code==200
    (folder/'credentials.json').write_text('{}')
    assert c.get('/api/v1/tasks',headers=bearer(folder,'admin')).status_code==503


def test_login_abuse_and_no_token_echo(cloud):
    app,folder=cloud;c=client(app);bad='secret-invalid-token-'+'x'*32
    for _ in range(10):
        r=c.post('/api/v1/auth/login',json={'token':bad},headers={'Origin':'https://facecheck.test'})
        assert r.status_code==401 and bad not in r.text
    assert login(c,folder).status_code==429


def test_login_body_limit_and_cross_origin(cloud):
    app,folder=cloud;c=client(app)
    assert c.post('/api/v1/auth/login',json={'token':'x'*5000},headers={'Origin':'https://facecheck.test'}).status_code==413
    assert c.post('/api/v1/auth/login',json={'token':(folder/'admin.token').read_text()},headers={'Origin':'https://evil.test'}).status_code==403


def test_cloud_configuration_rejects_missing_auth_and_demo_keys(tmp_path,monkeypatch):
    monkeypatch.setenv('FACECHECK_DEPLOYMENT','cloud')
    monkeypatch.delenv('FACECHECK_AUTH_CREDENTIALS_FILE',raising=False)
    with pytest.raises(ValueError,match='credentials'):create_app(tmp_path/'a.db',tmp_path/'a')


def test_cloud_demo_key_rejection(cloud):
    app,folder=cloud
    with pytest.raises(ValueError,match='high-entropy'):
        create_app(app.state.video_service.repository.path,app.state.video_service.storage_root,encryption_key=b'0123456789abcdef',mac_key=os.urandom(32))


def test_secret_file_rejects_ambiguous_environment(tmp_path,monkeypatch):
    p=tmp_path/'secret';p.write_bytes(os.urandom(16))
    monkeypatch.setenv('EXAMPLE_SECRET_FILE',str(p));monkeypatch.setenv('EXAMPLE_SECRET','conflict')
    with pytest.raises(ValueError,match='configure only'):secret_file('EXAMPLE_SECRET')
    monkeypatch.delenv('EXAMPLE_SECRET')
    assert secret_file('EXAMPLE_SECRET')==p.read_bytes()


def test_initializer_does_not_overwrite_existing_secrets(cloud):
    _,folder=cloud;before=(folder/'sm4.key').read_bytes()
    spec=importlib.util.spec_from_file_location('cloud_init_check',Path(__file__).resolve().parents[1]/'scripts/init_cloud.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    with pytest.raises(FileExistsError):mod.initialize(folder,'https://facecheck.test')
    assert (folder/'sm4.key').read_bytes()==before
    assert folder.stat().st_mode&0o777==0o700
    assert all(p.stat().st_mode&0o077==0 for p in folder.iterdir())


def test_upload_does_not_block_event_loop(tmp_path,monkeypatch):
    app=create_app(tmp_path/'app.db',tmp_path/'storage',encryption_key=os.urandom(16),mac_key=os.urandom(32))
    started=threading.Event();release=threading.Event()
    def slow_process(*args,**kwargs):
        started.set();release.wait(3)
        repo=app.state.video_service.repository;repo.create_task('x','x.mp4',1);return repo.get('x')
    monkeypatch.setattr(app.state.video_service,'create_and_process',slow_process)
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://localhost')as c:
            upload=asyncio.create_task(c.post('/api/v1/videos',files={'file':('x.mp4',b'x')}))
            try:
                for _ in range(200):
                    if started.is_set():break
                    await asyncio.sleep(.01)
                assert started.is_set()
                begin=time.perf_counter();health=await asyncio.wait_for(c.get('/health'),1)
                assert health.status_code==200 and time.perf_counter()-begin<1
            finally:release.set()
            assert (await upload).status_code==201
    asyncio.run(exercise())


def test_search_metrics_are_isolated_between_workers(tmp_path):
    app=create_app(tmp_path/'metrics.db',tmp_path/'storage',encryption_key=os.urandom(16),mac_key=os.urandom(32))
    service=app.state.video_service;barrier=threading.Barrier(2);results=[]
    def work(value):
        service.last_search_metrics={'request':value};barrier.wait(timeout=3);results.append(service.last_search_metrics['request'])
    threads=[threading.Thread(target=work,args=(i,))for i in range(2)]
    for thread in threads:thread.start()
    for thread in threads:thread.join()
    assert sorted(results)==[0,1] and service.last_search_metrics=={}


def test_admin_revocation_denies_bearer_and_cookie_and_operator_cannot_revoke(cloud):
    app,folder=cloud;op=client(app);assert login(op,folder).status_code==200
    assert op.post('/api/v1/auth/admin/principals/auditor/revoke',headers={'Origin':'https://facecheck.test'}).status_code==403
    admin=client(app);h=bearer(folder,'admin')
    assert admin.get('/api/v1/auth/admin/principals',headers=h).status_code==200
    assert admin.post('/api/v1/auth/admin/principals/operator/revoke',headers=h).status_code==200
    assert op.get('/api/v1/tasks').status_code==401
    assert op.get('/api/v1/tasks',headers=bearer(folder,'operator')).status_code==401
    assert login(op,folder).status_code==401
    repo=app.state.video_service.repository
    assert any(r['event_type']=='credential_revoked'for r in repo.list_audit(100,0))


def test_manifest_anchor_rejects_modified_metadata_before_restore(cloud,tmp_path):
    import cv2
    import numpy as np
    app,folder=cloud;c=client(app);h=bearer(folder,'operator')
    source=tmp_path/'source.mp4';writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),4,(16,16))
    assert writer.isOpened();writer.write(np.zeros((16,16,3),dtype=np.uint8));writer.release()
    r=c.post('/api/v1/videos',headers=h,files={'file':('source.mp4',source.read_bytes(),'video/mp4')})
    assert r.status_code==201 and r.json()['status']=='completed'
    task=r.json();assert len(task['processing_metrics']['manifest_sha256'])==64
    endpoint=f"/api/v1/platform/videos/{task['task_id']}/verify"
    verified=c.post(endpoint,headers=h)
    assert verified.status_code==200 and verified.json()['trusted_manifest_anchor_verified']is True
    manifest=app.state.video_service.storage_root/task['storage_id']/'manifest.json'
    original=manifest.read_bytes();manifest.write_bytes(original+b' ')
    assert c.post(endpoint,headers=h).status_code==400
    manifest.write_bytes(original)
    assert c.post(endpoint,headers=h).status_code==200


def test_cloud_rejects_legacy_unanchored_records(cloud):
    app,folder=cloud;svc=app.state.video_service;repo=svc.repository
    repo.create_task('legacy','legacy.mp4',1);repo.transition('legacy','processing');repo.transition('legacy','completed',storage_id='legacy')
    with pytest.raises(ValueError,match='anchor'):svc.check_storage_manifest(repo.get('legacy'),svc.storage_root/'legacy')
