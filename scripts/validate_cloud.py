#!/usr/bin/env python3
"""Real local HTTPS / authenticated storage validation, not a YTF accuracy experiment.

Uses a fresh private directory and isolated database. Optional one-video/one-query
input checks cold/warm online cache equivalence; original media remain read-only.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import ssl
import statistics
import subprocess
import sys
import tempfile
import time

import cv2
import httpx
import numpy as np

from init_cloud import initialize

ROOT=Path(__file__).resolve().parents[1]


def validate(sample_video=None,query_image=None):
    if bool(sample_video)!=bool(query_image):raise ValueError('provide both sample video and query image')
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [sample_video,query_image]if p}
    result={'kind':'isolated_cloud_functional_and_online_cache_validation','formal_ytf_experiment':False,
            'environment':dict(os=platform.platform(),python=platform.python_version(),machine=platform.machine())}
    started=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='facecheck-cloud-validation-')as scratch:
        root=Path(scratch)
        with socket.socket()as listener:
            listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
        origin=f'https://localhost:{port}';private=root/'private';env_file=initialize(private,origin)
        cfg=root/'openssl.cnf'
        cfg.write_text('[req]\ndistinguished_name=dn\nx509_extensions=ext\nprompt=no\n[dn]\nCN=localhost\n[ext]\nsubjectAltName=DNS:localhost,IP:127.0.0.1\nbasicConstraints=critical,CA:TRUE\nkeyUsage=digitalSignature,keyEncipherment,keyCertSign\n')
        cert=root/'tls.crt';tls_key=root/'tls.key'
        subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1','-config',str(cfg),'-keyout',str(tls_key),'-out',str(cert)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        tls_key.chmod(0o600);context=ssl.create_default_context(cafile=str(cert))
        env={k:v for k,v in os.environ.items()if not k.startswith('FACECHECK_')}
        env.update(PORT=str(port),FACECHECK_BIND_HOST='127.0.0.1')
        log=(root/'server.log').open('ab');server=None
        def start():
            proc=subprocess.Popen(['bash',str(ROOT/'scripts/start_cloud.sh'),str(env_file),str(cert),str(tls_key)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            for _ in range(200):
                if proc.poll()is not None:raise RuntimeError('isolated HTTPS server failed to start')
                try:
                    with httpx.Client(verify=context,trust_env=False,timeout=1)as c:
                        if c.get(origin+'/health').status_code==200:return proc
                except httpx.HTTPError:pass
                time.sleep(.05)
            proc.terminate();proc.wait(timeout=10);raise RuntimeError('HTTPS readiness timed out')
        def stop(proc):
            if proc and proc.poll()is None:
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
        def make_client():return httpx.Client(base_url=origin,verify=context,trust_env=False,timeout=120)
        def auth(role):return {'Authorization':'Bearer '+(private/(role+'.token')).read_text()}
        try:
            server=start()
            with make_client()as c:
                health=c.get('/health');assert health.status_code==200
                stream=health.extensions.get('network_stream');ssl_object=stream.get_extra_info('ssl_object')if stream else None
                result['https']={'certificate_verification_enabled':True,'hostname_verified':'localhost',
                                 'protocol':ssl_object.version()if ssl_object else None,
                                 'test_certificate_only':True}
                assert c.get('/api/v1/tasks').status_code==401
                assert c.get('/api/v1/tasks',headers=auth('auditor')).status_code==403
                assert c.get('/api/v1/audit',headers=auth('auditor')).status_code==200
                assert c.post('/api/v1/auth/admin/principals/auditor/revoke',headers=auth('operator')).status_code==403
                result['access_control']={'anonymous_denied':True,'auditor_cannot_read_videos':True,
                                          'auditor_can_read_audit':True,'operator_cannot_revoke_credentials':True}
                source=root/'synthetic.mp4';writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),4,(32,24))
                if not writer.isOpened():raise RuntimeError('test video writer unavailable')
                for i in range(4):writer.write(np.full((24,32,3),i*30,dtype=np.uint8))
                writer.release();video_bytes=source.read_bytes()
                def upload(_):
                    begin=time.perf_counter()
                    with make_client()as api:
                        r=api.post('/api/v1/videos',headers=auth('operator'),files={'file':('synthetic.mp4',video_bytes,'video/mp4')})
                        assert r.status_code==201 and r.json()['status']=='completed'
                        return r.json()['task_id'],time.perf_counter()-begin
                with ThreadPoolExecutor(max_workers=10)as pool:uploads=list(pool.map(upload,range(10)))
                times=[r[1]for r in uploads]
                result['ten_concurrent_https_uploads']={'requests':10,'completed':len(uploads),'success_rate':1.0,
                    'seconds':dict(median=statistics.median(times),min=min(times),max=max(times)),
                    'scope':'four-frame synthetic videos; includes TLS and actual encryption/storage; not sustained availability or 10 video inference streams'}
                task=uploads[0][0]
                restored=c.post(f'/api/v1/platform/videos/{task}/restore',headers=auth('operator'))
                assert restored.status_code==200 and hashlib.sha256(restored.content).digest()==hashlib.sha256(video_bytes).digest()
                result['restored_sha256_matches']=True
                verified=c.post(f'/api/v1/platform/videos/{task}/verify',headers=auth('operator'))
                assert verified.status_code==200 and verified.json()['trusted_manifest_anchor_verified']is True
                manifest=private/'runtime/storage'/task/'manifest.json';original_manifest=manifest.read_bytes()
                manifest.write_bytes(original_manifest+b' ')
                assert c.post(f'/api/v1/platform/videos/{task}/verify',headers=auth('operator')).status_code==400
                manifest.write_bytes(original_manifest)
                result['manifest_metadata_tamper_rejected']=True
                # Actual persisted opaque session, then a full HTTPS service restart on the same private DB.
                login=c.post('/api/v1/auth/login',json={'token':(private/'operator.token').read_text()},headers={'Origin':origin})
                assert login.status_code==200
                result['session_cookie']={k:k in login.headers['set-cookie'].lower()for k in ['secure','httponly','samesite=strict']}
                assert c.post('/api/v1/platform/videos/'+task+'/verify').status_code==403
                stop(server);server=start()
                assert c.get('/api/v1/tasks').status_code==200
                result['session_and_keys_survive_restart']=True
                again=c.post(f'/api/v1/platform/videos/{task}/restore',headers=auth('operator'))
                assert again.status_code==200 and again.content==video_bytes
                if sample_video:
                    uploaded=c.post('/api/v1/videos',headers=auth('operator'),files={'file':('authorized-sample.mp4',sample_video.read_bytes(),'video/mp4')})
                    assert uploaded.status_code==201 and uploaded.json()['status']=='completed'
                    sample_task=uploaded.json()['task_id'];query=query_image.read_bytes();runs=[]
                    for _ in range(2):
                        begin=time.perf_counter()
                        r=c.post(f'/api/v1/videos/{sample_task}/search?top_k=10',headers=auth('operator'),files={'query_image':('query.img',query,'application/octet-stream')})
                        if r.status_code!=200:raise RuntimeError(f'online cache validation failed with HTTP {r.status_code}')
                        runs.append((r.json(),time.perf_counter()-begin))
                    assert runs[0][0]['results']==runs[1][0]['results']
                    assert runs[0][0]['timings']['feature_cache_hit']is False and runs[1][0]['timings']['feature_cache_hit']is True
                    assert runs[0][0]['timings'].get('feature_count',0)>0
                    result['online_feature_cache']={'sample_video_sha256':source_hashes[str(sample_video)],'query_image_sha256':source_hashes[str(query_image)],
                         'query_count':1,'gallery_video_count':1,'accuracy_evaluation':False,'results_exactly_equal':True,
                         'result_count':len(runs[0][0]['results']),
                         'cold':dict(http_seconds=runs[0][1],timings=runs[0][0]['timings']),
                         'warm':dict(http_seconds=runs[1][1],timings=runs[1][0]['timings']),
                         'speedup_for_this_one_request':runs[0][1]/runs[1][1]}
                    cache=next((private/'runtime/protected-features'/sample_task).glob('*.bin'))
                    payload=bytearray(cache.read_bytes());payload[-1]^=1;cache.write_bytes(payload)
                    bad=c.post(f'/api/v1/videos/{sample_task}/search',headers=auth('operator'),files={'query_image':('query.img',query)})
                    assert bad.status_code==400
                    result['online_feature_cache']['tampered_cache_rejected']=True
                block=next((private/'runtime/storage'/task/'chunks').glob('*.bin'))
                payload=bytearray(block.read_bytes());payload[-1]^=1;block.write_bytes(payload)
                assert c.post(f'/api/v1/platform/videos/{task}/verify',headers=auth('operator')).status_code==400
                result['tampered_video_rejected']=True
                revoke=c.post('/api/v1/auth/admin/principals/operator/revoke',headers=auth('admin'))
                assert revoke.status_code==200
                assert c.get('/api/v1/tasks',headers=auth('operator')).status_code==401
                assert c.get('/api/v1/tasks').status_code==401
                result['revoked_access_token_and_session_rejected']=True
                overview=c.get('/api/v1/platform/overview',headers=auth('admin')).json()
                result['backend']={k:overview[k]for k in ['crypto_backend','crypto_library_version','sm4_implementation','default_model']}
            result['source_media_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in source_hashes.items())
            result['status']='passed';result['total_seconds']=time.perf_counter()-started
            return result
        finally:stop(server);log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-video',type=Path)
    parser.add_argument('--query-image',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.exit(2,'Refusing to overwrite an existing validation result\n')
    result=validate(args.sample_video,args.query_image)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8')as fp:json.dump(result,fp,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))
