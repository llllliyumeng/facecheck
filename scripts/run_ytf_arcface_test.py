"""Run the one frozen, paired-mode ArcFace YTF test; never selects a threshold."""
from __future__ import annotations

import argparse
import json
import logging
import platform
import resource
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
from facecheck.crypto_backends import get_crypto_backend
from facecheck.face_detection import FaceDetectionConfig
from facecheck.recognition import DEFAULT_ARCFACE_THRESHOLD, recognition_config
from facecheck.storage import store_encrypted_video, restore_encrypted_video
from facecheck.ytf_qg_protocol import TEST, THRESHOLD_SELECTION, split_identities, validate_layout
from facecheck.ytf_runner import OpenCVFeatureBackend, RunnerConfig, _load_inputs, _sha256, run_ytf_runner
from run_ytf_full import read_keys

SELECTION = ROOT / "data/ytf/experiments/ytf_arcface_w600k_r50_20261101/selection_full_plaintext"
DEFAULT_OUTPUT = ROOT / "data/ytf/experiments/ytf_arcface_test_20261101"


def write_json(path, value, *, replace=False):
    if replace:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str)+'\n')
        temporary.replace(path)
    else:
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, default=str)
            stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    output = args.output.absolute()
    execution = output.with_name(output.name+'.execution')
    if any(p.exists() or p.is_symlink() for p in (output, execution)):
        raise ValueError('result/execution directory exists; refusing another run or overwrite')
    manifest = ROOT/'data/ytf/experiment_full.json'
    split_path = ROOT/'data/ytf/query_gallery_identity_split.json'
    records, assignment, _, split_json = _load_inputs(manifest, split_path)
    if Counter(assignment.values()) != {TEST:502, THRESHOLD_SELECTION:502}:
        raise ValueError('original identity-disjoint split must contain 502 test and 502 selection identities')
    if assignment != split_identities(assignment, seed=split_json['seed']):
        raise ValueError('split differs from deterministic identity assignment')
    active = [r for r in records if assignment[r['identity_id']]==TEST]
    test_ids = sorted(identity for identity, side in assignment.items() if side==TEST)
    layout = validate_layout(active, {identity:TEST for identity in test_ids})
    if layout['query_count'] != 502 or layout['gallery_count'] != 502:
        raise ValueError('test must contain exactly 502 Query/Gallery records')
    selection_config = json.loads((SELECTION/'config.json').read_text())
    selection_metrics = json.loads((SELECTION/'metrics.json').read_text())
    selection_ids = json.loads((SELECTION/'identity_ids.json').read_text())
    if (json.loads((SELECTION/'run_status.json').read_text())['state'] != 'complete' or
            selection_metrics['identity_count'] != 502 or
            not json.loads((SELECTION/'comparison.json').read_text())['test_gate_met']):
        raise ValueError('complete selection and entry gate are required')
    if set(selection_ids['identity_ids']).intersection(test_ids):
        raise ValueError('test/selection identity leakage')
    if (_sha256(manifest) != selection_config['manifest_sha256'] or
            _sha256(split_path) != selection_ids['identity_split_sha256']):
        raise ValueError('manifest/split changed since selection')
    embedding = recognition_config('arcface')
    yunet = ROOT/'models/face_detection_yunet_2023mar.onnx'
    if (_sha256(embedding.model_path) != selection_config['model']['file_sha256'] or
            _sha256(yunet) != selection_config['yunet']['sha256']):
        raise ValueError('model weights changed since selection')
    threshold = selection_metrics['tar_far_selection']['threshold']
    if threshold != DEFAULT_ARCFACE_THRESHOLD:
        raise ValueError('default threshold must equal the frozen ArcFace selection threshold')
    if (selection_config['yunet']['frame_interval_seconds'] != 1.0 or
            selection_config['yunet']['score_threshold'] != 0.9 or
            selection_config['gallery']['quality_filter'] or
            selection_config['gallery']['feature_aggregation'] != 'none'):
        raise ValueError('selection configuration does not match the frozen baseline protocol')
    if any(not Path(r['video_path']).is_file() or Path(r['video_path']).stat().st_size==0 for r in active):
        raise ValueError('missing/empty test media')
    galleries = [r for r in active if r['split']=='gallery']
    gallery_bytes = sum(Path(r['video_path']).stat().st_size for r in galleries)
    disk_free = shutil.disk_usage(output.parent).free
    required = max(10*1024**3, 2*gallery_bytes+2*1024**3)
    if disk_free < required:
        raise ValueError('insufficient disk for encrypted/restored temporary Gallery files')
    # Reads the existing private key file; never writes it or includes key material in outputs.
    encryption_key, mac_key = read_keys(ROOT/'.env.ytf-experiment')
    crypto = get_crypto_backend()
    config = RunnerConfig(
        manifest, split_path, output, FaceDetectionConfig(yunet), embedding,
        evaluation_scope=TEST, frozen_selection_threshold=threshold,
        score_aggregation='max', query_use_annotation_box=False,
        gallery_quality_filter=False, gallery_feature_aggregation='none', encryption_workers=4,
    )
    config.validate()
    preflight = {
        'test_identity_count':502, 'test_gallery_count':502, 'test_query_count':502,
        'output':str(output), 'evaluation_scope':TEST, 'modes':['plaintext','encrypted_restore'],
        'frozen_threshold':threshold, 'threshold_source':str(SELECTION/'metrics.json'),
        'gallery_bytes':gallery_bytes, 'disk_free_bytes':disk_free, 'required_disk_bytes':required,
        'estimated_seconds_from_selection':2*selection_metrics['timing_seconds']['total_wall'],
        'model_sha256':_sha256(embedding.model_path), 'yunet_sha256':_sha256(yunet),
        'manifest_sha256':_sha256(manifest), 'split_sha256':_sha256(split_path),
        'platform':platform.platform(), 'machine':platform.machine(), 'python':sys.version,
        'opencv':cv2.__version__, 'numpy':np.__version__,
        'crypto_backend':crypto.name, 'gmssl_version':getattr(crypto,'version',None),
        'crypto_library':getattr(crypto,'library_path',None),
        'model_spec':'w600k_r50; OpenCV DNN CPU; 112x112 five-landmark alignment; RGB (x-127.5)/127.5; 512 float32 L2; max cosine',
        'protocol_unchanged':True, 'selection_rerun':False, 'threshold_tuned_on_test':False,
    }
    print(json.dumps(preflight, indent=2), flush=True)
    if args.preflight_only:
        return
    execution.mkdir(mode=0o700)
    code_paths = sorted((ROOT/'src/facecheck').glob('*.py')) + [Path(__file__).resolve()]
    code_hashes = {str(p.relative_to(ROOT)):_sha256(p) for p in code_paths}
    # Snapshot all existing experiment/benchmark/report outputs, including failed experiments.
    protected_paths = [p for parent in (ROOT/'data/ytf/experiments', ROOT/'results', ROOT/'docs')
                       if parent.exists() for p in parent.rglob('*')
                       if p.is_file() and execution not in p.parents]
    protected_hashes = {str(p.relative_to(ROOT)):_sha256(p) for p in protected_paths}
    write_json(execution/'preflight.json', preflight)
    write_json(execution/'source_sha256.json', code_hashes)
    write_json(execution/'protected_sha256.json', protected_hashes)
    write_json(execution/'identity_ids.json', {'split':TEST, 'count':502, 'identities':test_ids})
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(execution/'runner.log')])
    counts = Counter()
    started = time.perf_counter()
    state = {'state':'running', 'complete':False, 'started_at_utc':datetime.now(timezone.utc).isoformat()}
    def save_status():
        write_json(execution/'run_status.json', {**state, 'counts':dict(counts),
                   'elapsed_seconds':time.perf_counter()-started}, replace=True)
    save_status()

    class ProgressBackend(OpenCVFeatureBackend):
        def extract(self, record):
            result = super().extract(record)
            mode = record['_experiment_mode']
            counts[mode+'_media'] += 1
            counts[mode+'_features'] += len(result.features)
            counts[mode+'_failures'] += result.feature_extraction_failure_count
            if counts[mode+'_media'] % 10 == 0 or counts[mode+'_media']==1004:
                logging.info('mode=%s media=%s/1004 features=%s failures=%s elapsed=%.1fs', mode,
                             counts[mode+'_media'],counts[mode+'_features'],counts[mode+'_failures'],time.perf_counter()-started)
                save_status()
            return result

    def encrypt(*args, **kwargs):
        stored = store_encrypted_video(*args, **kwargs)
        counts['encrypted_videos'] += 1
        counts['encrypted_chunks'] += stored.manifest['chunk_count']
        return stored

    def restore(directory, destination, *keys, **kwargs):
        result = restore_encrypted_video(directory, destination, *keys, **kwargs)
        counts['authenticated_restored_videos'] += 1
        counts['authenticated_restored_chunks'] += json.loads((directory/'manifest.json').read_text())['chunk_count']
        return result

    try:
        metrics = run_ytf_runner(config, encryption_key=encryption_key, mac_key=mac_key,
                                 backend_factory=ProgressBackend, encrypt=encrypt, restore=restore)
        if (set(metrics['matrix_audit']) != {TEST} or
                metrics['matrix_audit'][TEST]['expected_pairs'] != 502*502 or
                any(counts[mode+'_media'] != 1004 for mode in ('plaintext','encrypted_restore')) or
                counts['encrypted_videos'] != 502 or counts['authenticated_restored_videos'] != 502):
            raise ValueError('published test bundle failed completeness check')
        if not all(_sha256(ROOT/name)==digest for name,digest in protected_hashes.items()):
            raise ValueError('a protected historical output changed')
        if not all(_sha256(ROOT/name)==digest for name,digest in code_hashes.items()):
            raise ValueError('source changed during the formal run')
        state.update(state='complete',complete=True,historical_outputs_unchanged=True,source_unchanged=True)
        logging.info('COMPLETE output=%s consistency=%s',output,metrics['mode_consistency'])
    except BaseException as exc:
        state.update(state='incomplete',complete=False,reason=type(exc).__name__)
        logging.exception('INCOMPLETE: do not treat partial artifacts as formal results')
        raise
    finally:
        state.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save_status()


if __name__ == '__main__':
    main()
