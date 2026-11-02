"""Validate completed frozen-test artifacts and generate Chinese course-report material.

This script reads results only. It performs no inference or threshold selection.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from facecheck.ytf_runner import _load_inputs, _sha256
from facecheck.ytf_qg_protocol import TEST, retrieval_metrics, evaluate_test_threshold


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['status'])
        writer.writeheader(); writer.writerows(rows)


def write_json(path,value):
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False); f.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/ytf/experiments/ytf_arcface_test_20261101')
    args=parser.parse_args()
    out=args.output.absolute(); execution=out.with_name(out.name+'.execution')
    if any((out/n).exists() for n in ('course_practice_report.md','validation.json','analysis.json')):
        raise ValueError('analysis already exists; refusing overwrite')
    status=read_json(execution/'run_status.json')
    if status['state']!='complete' or not status['complete']:
        raise ValueError('incomplete run cannot be reported as a formal test')
    calendar_seconds=(datetime.fromisoformat(status['finished_at_utc'])-
                      datetime.fromisoformat(status['started_at_utc'])).total_seconds()
    m=read_json(out/'metrics.json'); config=read_json(out/'config.json')
    preflight=read_json(execution/'preflight.json')
    hardware=read_json(execution/'environment_hardware.json')
    sleep_assertion=(read_json(execution/'idle_sleep_assertion.json')
                     if (execution/'idle_sleep_assertion.json').exists() else None)
    assert m['evaluation_scope']==config['evaluation_scope']==TEST
    assert set(m['matrix_audit'])=={TEST}
    selection_path=ROOT/'data/ytf/experiments/ytf_arcface_w600k_r50_20261101/selection_full_plaintext'
    selection=read_json(selection_path/'metrics.json')
    frozen={'status':'estimated','split':'threshold_selection',
            'threshold':selection['tar_far_selection']['threshold']}
    assert frozen['threshold']==m['frozen_selection_threshold']==preflight['frozen_threshold']
    assert config['recognition_model']=='arcface' and config['embedding']['expected_dimension']==512
    assert config['gallery_representation']['aggregation']=='none; score-time max over frame features'
    assert not config['query_use_annotation_box']
    assert not config['gallery_representation']['enabled']
    records,split,_,_= _load_inputs(Path(config['manifest_path']),Path(config['identity_split_path']))
    active=[r for r in records if split[r['identity_id']]==TEST]
    active_split={i:s for i,s in split.items() if s==TEST}
    expected_q={r['video_id']:r for r in active if r['split']=='query'}
    expected_g={r['video_id']:r for r in active if r['split']=='gallery'}
    assert len(expected_q)==len(expected_g)==len(active_split)==502
    assert set(read_json(execution/'identity_ids.json')['identities'])==set(active_split)
    assert set(active_split).isdisjoint(read_json(selection_path/'identity_ids.json')['identity_ids'])
    assert _sha256(Path(config['manifest_path']))==config['manifest_sha256']==preflight['manifest_sha256']
    assert _sha256(Path(config['identity_split_path']))==config['identity_split_sha256']==preflight['split_sha256']
    assert _sha256(Path(config['model_files']['arcface_path']))==preflight['model_sha256']
    assert _sha256(Path(config['model_files']['yunet_path']))==preflight['yunet_sha256']
    for name,expected in read_json(execution/'protected_sha256.json').items():
        assert _sha256(ROOT/name)==expected, f'historical file changed: {name}'
    for name,expected in read_json(execution/'source_sha256.json').items():
        assert _sha256(ROOT/name)==expected, f'source changed: {name}'

    pair_modes=defaultdict(list)
    for row in read_csv(out/'genuine_impostor_pairs.csv'):
        row['similarity']=float(row['similarity']) if row['similarity'] else None
        assert row['similarity'] is None or math.isfinite(row['similarity'])
        assert row['split']==TEST and row['query_id'] in expected_q and row['gallery_id'] in expected_g
        assert row['pair_type']==('genuine' if expected_q[row['query_id']]['identity_id']==expected_g[row['gallery_id']]['identity_id'] else 'impostor')
        pair_modes[row['mode']].append(row)
    features=read_json(out/'feature_summary.json')
    queries=read_csv(out/'query_results.csv')
    failures=read_csv(out/'feature_failures.csv')
    assert set(pair_modes)==set(features)=={'plaintext','encrypted_restore'}
    availability={}; pair_stats={}; rankings={}; failure_types={}
    for mode,pairs in pair_modes.items():
        assert len(pairs)==502*502
        assert len({(p['query_id'],p['gallery_id']) for p in pairs})==502*502
        assert len(features[mode])==1004 and len({r['video_id'] for r in features[mode]})==1004
        assert all(r['feature_dimension']==512 for r in features[mode])
        qrows=[r for r in queries if r['mode']==mode]
        assert len(qrows)==502 and {r['query_id'] for r in qrows}==set(expected_q)
        detected={r['query_id'] for r in qrows if int(r['query_feature_count'])>0}
        reconstructed=retrieval_metrics(active,active_split,pairs,detected)
        assert reconstructed==m[mode]['retrieval']
        assert evaluate_test_threshold(pairs,frozen)==m[mode]['tar_far_test']
        availability[mode]={}
        for role,media_type in [('query','image'),('gallery','video')]:
            rows=[r for r in features[mode] if r['media_type']==media_type]
            assert {r['video_id'] for r in rows}==set(expected_q if role=='query' else expected_g)
            availability[mode][role]={
                'total':len(rows),'with_features':sum(r['feature_count']>0 for r in rows),
                'without_features':sum(r['feature_count']==0 for r in rows),
                'embedding_count':sum(r['feature_count'] for r in rows),
                'sampled_frames':sum(r['sampled_frame_count'] for r in rows),
                'detected_faces':sum(r['detected_face_count'] for r in rows),
                'feature_failures':sum(r['feature_extraction_failure_count'] for r in rows),
                'detection_failures':sum(r['detection_failure_count'] for r in rows),
                'invalid_detection_records':sum(r['invalid_detection_record_count'] for r in rows),
            }
        failure_types[mode]=dict(Counter(r['exception_message'] for r in failures if r['mode']==mode))
        assert sum(failure_types[mode].values())==sum(availability[mode][role]['feature_failures']+
                availability[mode][role]['detection_failures']+availability[mode][role]['invalid_detection_records'] for role in ('query','gallery'))
        tf=m[mode]['tar_far_test']
        pair_stats[mode]={key:tf[key] for key in (
            'genuine_count','impostor_count','scored_genuine_count','scored_impostor_count',
            'unscored_genuine_count','unscored_impostor_count','unscored_pair_count','unscored_pair_rate',
            'far_resolution','target_far_resolution_met','true_accepts','false_accepts')}
        pair_stats[mode]['unscored_reasons']=dict(Counter(r['score_status'] for r in pairs if r['similarity'] is None))
        byq=defaultdict(list)
        for p in pairs: byq[p['query_id']].append(p)
        ranked_rows=[]
        for qid,rows in byq.items():
            valid=sorted((r for r in rows if r['similarity'] is not None),key=lambda r:(-r['similarity'],r['gallery_id']))
            genuine=next(r for r in rows if r['pair_type']=='genuine')
            rank=next((i for i,r in enumerate(valid,1) if r['pair_type']=='genuine'),None)
            top=valid[0] if valid else None
            qmeta=next(r for r in qrows if r['query_id']==qid)
            ranked_rows.append({'mode':mode,'query_id':qid,'identity_id':expected_q[qid]['identity_id'],
                'query_feature_count':int(qmeta['query_feature_count']),
                'top1_gallery_id':top['gallery_id'] if top else None,
                'top1_score':top['similarity'] if top else None,
                'genuine_score':genuine['similarity'],'correct_identity_rank':rank,
                'top1_correct':rank==1,'recall5_hit':rank is not None and rank<=5,
                'genuine_minus_top1_score':genuine['similarity']-top['similarity'] if top and genuine['similarity'] is not None else None,
                'genuine_accepted':genuine['similarity'] is not None and genuine['similarity']>=frozen['threshold'],
                'scored_candidates':len(valid),
                'reason':('query_no_features' if not int(qmeta['query_feature_count']) else
                          'genuine_unscored' if genuine['similarity'] is None else
                          'wrong_top1' if rank!=1 else 'correct_top1')})
        rankings[mode]=ranked_rows

    plain={ (r['query_id'],r['gallery_id']):r['similarity'] for r in pair_modes['plaintext'] }
    restored={ (r['query_id'],r['gallery_id']):r['similarity'] for r in pair_modes['encrypted_restore'] }
    exact=plain==restored
    metric_equal=m['plaintext']==m['encrypted_restore']
    restored_rows=[r for r in features['encrypted_restore'] if r['media_type']=='video']
    hashes_equal=sum(r['source_sha256']==r['restored_sha256'] and bool(r['source_sha256']) for r in restored_rows)
    assert len(restored_rows)==hashes_equal==502
    assert m['encryption_integrity']['all_source_restore_sha256_equal']
    source_by_id={r['video_id']:r for r in features['plaintext']}
    assert all(source_by_id[r['video_id']]['source_sha256']==r['restored_sha256'] for r in restored_rows)

    baseline_dir=ROOT/'data/ytf/experiments/ytf_full_run_02'
    baseline=read_json(baseline_dir/'metrics.json')
    baseline_pairs=[]
    for row in read_csv(baseline_dir/'genuine_impostor_pairs.csv'):
        if row['mode']=='plaintext' and row['split']==TEST:
            row['similarity']=float(row['similarity']) if row['similarity'] else None
            baseline_pairs.append(row)
    baseline_queries=[r for r in read_csv(baseline_dir/'query_results.csv') if r['mode']=='plaintext' and r['query_split']==TEST]
    assert len(baseline_queries)==502 and {r['query_id'] for r in baseline_queries}==set(expected_q)
    baseline_retrieval=retrieval_metrics(active,active_split,baseline_pairs,
                {r['query_id'] for r in baseline_queries if int(r['query_feature_count'])>0})
    btf=baseline['plaintext']['tar_far_test']; atf=m['plaintext']['tar_far_test']
    assert evaluate_test_threshold(baseline_pairs,baseline['threshold_selection'])==btf
    a=m['plaintext']['retrieval']; b=baseline_retrieval
    comparisons=[
        {'metric':'Top-1','sface_test':b['top1_accuracy']['value'],'arcface_test':a['top1_accuracy']['value']},
        {'metric':'Recall@5','sface_test':b['recall_at_5']['value'],'arcface_test':a['recall_at_5']['value']},
        {'metric':'TAR_at_frozen_selection_threshold','sface_test':btf['tar'],'arcface_test':atf['tar']},
        {'metric':'actual_FAR','sface_test':btf['far'],'arcface_test':atf['far']},
    ]
    for row in comparisons: row['absolute_delta_percentage_points']=100*(row['arcface_test']-row['sface_test'])
    assert a['query_count']==502
    analysis={'status':'complete','baseline_test_retrieval':baseline_retrieval,
        'baseline_test_tar_far':btf,'comparison':comparisons,'feature_availability':availability,
        'pair_statistics':pair_stats,'failure_reasons':failure_types,
        'plaintext_encrypted_metrics_equal':metric_equal,'plaintext_encrypted_pair_scores_exactly_equal':exact,
        'restored_sha256_pass_count':hashes_equal,'selection_threshold_source':str(selection_path/'metrics.json'),
        'threshold_retuned_on_test':False,'selection_rerun':False,
        'execution_timing':{'perf_counter_seconds':status['elapsed_seconds'],
                            'utc_start_to_finish_seconds':calendar_seconds,
                            'utc_minus_perf_seconds':calendar_seconds-status['elapsed_seconds'],
                            'started_at_utc':status['started_at_utc'],'finished_at_utc':status['finished_at_utc']},
        'temporary_idle_sleep_assertion':sleep_assertion,
        'course_goal_gaps':{'top1_percentage_points_below_95':max(0.,100*(.95-a['top1_accuracy']['value'])),
                            'tar_percentage_points_below_90':max(0.,100*(.90-atf['tar'])),
                            'far_excess_above_1e_3':max(0.,atf['far']-.001)},
        'per_query_outcomes':dict(Counter(r['reason'] for r in rankings['plaintext']))}
    validation={'status':'passed','test_identity_count':502,'test_query_count':502,'test_gallery_count':502,
        'pair_count_per_mode':252004,'full_unique_test_cartesian_product':True,
        'split_disjointness_checked':True,'original_split_manifest_model_hashes_match':True,
        'scores_and_missing_counts_verified':True,'metrics_recomputed_from_raw_pair_csv':True,
        'historical_outputs_unchanged':True,'source_unchanged_during_run':True,
        'sha256_verified_gallery_count':hashes_equal,'mode_scores_exactly_equal':exact,
        'mode_metrics_equal':metric_equal}
    write_json(out/'validation.json',validation)
    write_json(out/'analysis.json',analysis)
    write_csv(out/'model_comparison.csv',comparisons)
    write_csv(out/'query_diagnostics.csv',rankings['plaintext']+rankings['encrypted_restore'])
    write_csv(out/'failed_queries_plaintext.csv',[r for r in rankings['plaintext'] if not r['top1_correct']])
    write_csv(out/'restore_sha256_audit.csv',[{'video_id':r['video_id'],'source_sha256':r['source_sha256'],
              'restored_sha256':r['restored_sha256'],'equal':r['source_sha256']==r['restored_sha256']} for r in restored_rows])
    timings=m['stage_timings']; counts=availability['plaintext']; tf=pair_stats['plaintext']
    def pct(x): return f'{x*100:.4f}%'
    def table(headers,rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                          ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])
    metric_table=table(['指标','SFace 正式 test 子集','ArcFace plaintext test','ArcFace encrypted_restore test','差值（百分点）'],[
        ['Top-1',f"{b['top1_accuracy']['hits']}/502 ({pct(b['top1_accuracy']['value'])})",f"{a['top1_accuracy']['hits']}/502 ({pct(a['top1_accuracy']['value'])})",pct(m['encrypted_restore']['retrieval']['top1_accuracy']['value']),f"{comparisons[0]['absolute_delta_percentage_points']:+.4f}"],
        ['Recall@5',f"{b['recall_at_5']['hits']}/502 ({pct(b['recall_at_5']['value'])})",f"{a['recall_at_5']['hits']}/502 ({pct(a['recall_at_5']['value'])})",pct(m['encrypted_restore']['retrieval']['recall_at_5']['value']),f"{comparisons[1]['absolute_delta_percentage_points']:+.4f}"],
        ['冻结 selection 阈值下 TAR',f"{btf['true_accepts']}/502 ({pct(btf['tar'])})",f"{atf['true_accepts']}/502 ({pct(atf['tar'])})",pct(m['encrypted_restore']['tar_far_test']['tar']),f"{comparisons[2]['absolute_delta_percentage_points']:+.4f}"],
        ['实际 FAR',f"{btf['false_accepts']}/251502 ({btf['far']:.10f})",f"{atf['false_accepts']}/251502 ({atf['far']:.10f})",f"{m['encrypted_restore']['tar_far_test']['far']:.10f}",f"{comparisons[3]['absolute_delta_percentage_points']:+.6f}"],
    ])
    time_table=table(['计时阶段（秒）','plaintext','encrypted_restore'],[
        [label,f"{timings['plaintext'][key]:.6f}",f"{timings['encrypted_restore'][key]:.6f}"]
        for key,label in [('video_read_decode_extract_seconds','读取/解码/抽帧'),('face_detection_seconds','YuNet 检测'),
            ('feature_extraction_seconds','特征提取/对齐'),('similarity_calculation_seconds','相似度计算'),
            ('ranking_seconds','排名构建'),('encryption_seconds','加密与存储'),
            ('restore_integrity_seconds','恢复与认证'),('experiment_total_seconds','单模式总耗时')]])
    selection_retrieval=selection['retrieval']; stf=selection['tar_far_selection']
    selection_table=table(['指标','ArcFace selection（502 身份）','ArcFace test（502 身份）'],[
        ['Top-1',pct(selection_retrieval['top1_accuracy']['value']),pct(a['top1_accuracy']['value'])],
        ['Recall@5',pct(selection_retrieval['recall_at_5']['value']),pct(a['recall_at_5']['value'])],
        ['TAR',pct(stf['selection_tar']),pct(atf['tar'])],['FAR',f"{stf['selection_far']:.10f}",f"{atf['far']:.10f}"]])
    extraction_table=table(['角色','有效媒体','无有效特征媒体','embedding 数','检测人脸数','抽样帧/图片数','特征失败数'],[
        [role,counts[role]['with_features'],counts[role]['without_features'],counts[role]['embedding_count'],
         counts[role]['detected_faces'],counts[role]['sampled_frames'],counts[role]['feature_failures']]
        for role in ('query','gallery')])
    pair_table=table(['类型','总试验数','有效分数数','未评分数'],[
        ['genuine',tf['genuine_count'],tf['scored_genuine_count'],tf['unscored_genuine_count']],
        ['impostor',tf['impostor_count'],tf['scored_impostor_count'],tf['unscored_impostor_count']],
        ['合计',252004,tf['scored_genuine_count']+tf['scored_impostor_count'],tf['unscored_pair_count']]])
    bbox=sum(n for reason,n in failure_types['plaintext'].items() if 'bounding box' in reason)
    report=f'''# YTF ArcFace 正式 test：课程实践数据与分析报告

## 1. 实验目的与数据集说明

本实验在现有 SFace 正式基线和 ArcFace 完整 selection 结果之上，检验一次冻结 ArcFace 方案在独立 test 身份上的检索与验证表现，并比较加密存储、解密恢复前后的结果。本次只运行一轮正式 test：两个规定模式各执行一次完整推理，未重跑 selection，未依据 test 分数重新选阈值、模型或参数。

项目 manifest 共 1,004 个身份、2,008 个媒体记录，selection/test 各 502 个身份；每个身份对应一个 Gallery 视频和一个 Query 图片。这里的“全量 test”指现有协议的全部 502 个 test 身份，而非 1,004 个身份合并评估或整个原始 YTF 发布集。Gallery MP4 是 YTF 帧图重建的 1 fps 实验输入，并非原始 YouTube MP4。原始 manifest 和划分文件未修改，见配置校验值。

## 2. 实验环境、模型版本与参数

- 环境：{hardware['cpu_brand']}、{hardware['physical_memory_bytes']/1024**3:.0f} GiB 内存；{preflight['platform']}，{preflight['machine']}；Python {preflight['python'].split()[0]}，OpenCV {preflight['opencv']}，NumPy {preflight['numpy']}。
- ArcFace：InsightFace v0.7 `buffalo_l / w600k_r50` ResNet-50，文件 `models/arcface_w600k_r50.onnx`；SHA-256 `{preflight['model_sha256']}`。权重来源为官方 InsightFace v0.7 buffalo_l 发布包，许可限非商业研究用途；不应把代码许可等同于权重许可。
- 推理：OpenCV DNN CPU；YuNet 五点与标准 ArcFace 模板做相似变换，112×112 对齐；BGR→RGB；`(pixel-127.5)/127.5`；512 维 float32 embedding，L2 归一化。
- 检测与采样：既有 YuNet 模型（SHA-256 `{preflight['yunet_sha256']}`），置信度 0.9、NMS 0.3、每 1 秒采样。Gallery 保留全部配置采样帧与有效检测的独立特征；不做质量过滤、均值聚合或 Query 标注框筛选。
- 相似度：同一视频配对的全部 Query/Gallery 特征 cosine similarity 取最大值；按相似度降序、Gallery ID 升序打破排名平局。检索指标按身份/视频配对排名，不将视频帧当独立身份。
- 冻结阈值：`{frozen['threshold']}`，显示值 `0.2731515063`；分数 `>=` 阈值接受。仅来自已完成的 ArcFace plaintext selection 分数，未使用 SFace 阈值 `{btf['threshold']}`。
- 加密后端：{preflight['crypto_backend']}，{preflight['gmssl_version']}；动态库 `{preflight['crypto_library']}`。保留现有 SM4-CBC、HMAC-SM3、1 MiB 分块、4 workers、密钥派生与认证恢复顺序；实验密钥从既有私有文件只读获取，未写入报告或结果。

## 3. 数据划分与实验流程

划分沿用 `SHA-256(seed + NUL + identity_id)`，seed 为 `facecheck-ytf-qg-v1`；selection 与 test 身份严格不相交。manifest SHA-256 为 `{preflight['manifest_sha256']}`，划分文件 SHA-256 为 `{preflight['split_sha256']}`。运行与事后检查均核对 502 test 身份、502 Gallery、502 Query；每模式完整 252,004 个配对，跨分区配对为 0。执行身份清单存放于同名 `.execution/identity_ids.json`。

plaintext：对全部 test 媒体各提取一次，媒体特征保存在仅本次模式有效的内存缓存，随后完成配对、排名和冻结阈值验证。encrypted_restore：先逐个加密 Gallery 视频，按既有协议认证恢复并检查 SHA-256，再独立提取恢复媒体特征；Query 图片作为共享的原始输入，不加密。两模式的特征不互相复用，以实际检验恢复一致性。缓存不跨模型、配置或运行复用，向量不落盘。

## 4. ArcFace 与 SFace 对比口径

SFace 保留为显式回退：128 维 SFace embedding 与其 `alignCrop` 路径；ArcFace 为 512 维并使用匹配的五点对齐与预处理。YuNet、媒体、采样、最大相似度配对、身份划分和指标定义一致。本次未重跑 SFace，基线来自 `ytf_full_run_02` 的原始 test CSV 和冻结阈值验证结果。

历史常见的 Top-1 `783/1004=77.99%`、Recall@5 `846/1004=84.26%` 是 selection+test 合并值；不能直接与本次 502 身份 test 值计算模型提升。本报告从历史原始 pair/query 文件重建同一 test 子集：Top-1 `{b['top1_accuracy']['hits']}/502`、Recall@5 `{b['recall_at_5']['hits']}/502`，保持对照分母一致。SFace 的 test FAR 本身略高于 1e-3，表中仍如实保留其冻结阈值下 TAR。

## 5. 正式 test 结果

{metric_table}

两模型分别使用各自 selection 冻结阈值。ArcFace 有效 impostor 数为 {tf['scored_impostor_count']:,}，FAR 分辨率为 `{tf['far_resolution']:.12g}`，足以估计 1e-3；实际 test FAR 约束是否满足：**{atf['operating_point_met']}**。若实际 FAR 超过目标，应将该 TAR 描述为“selection 冻结阈值下的 TAR”，不能宣称在 test 达到 TAR@FAR≤1e-3，也不得现场升高阈值重测。

{pair_table}

未评分率 `{tf['unscored_pair_rate']:.4%}`。本报告遵循原有保守缺失策略：未评分 pair 均拒绝，但保留在 genuine/impostor 总试验分母中；有效分数数只用于 FAR 分辨率和可估计性判定。

## 6. 加密存储与恢复一致性

两模式识别指标完全相同：`{metric_equal}`；所有配对分数及缺失状态逐项精确相同：`{exact}`。Runner 实测最大绝对分数差 `{m['mode_consistency']['max_absolute_similarity_delta']}`，比较容差 `{m['mode_consistency']['tolerance']}`。本轮加密 {status['counts']['encrypted_videos']} 个 Gallery 视频、{status['counts']['encrypted_chunks']} 个分块，认证恢复 {status['counts']['authenticated_restored_videos']} 个视频、{status['counts']['authenticated_restored_chunks']} 个分块；**{hashes_equal}/502** 个恢复视频 SHA-256 与源文件一致，逐文件证据见 `restore_sha256_audit.csv`。

这证明本次加密、认证与恢复未引入识别损失；流程是恢复并验证后的明文检索，不能作为密文域 ANN/安全索引已实现的证据。SHA-256 用于与可信源值核对；认证安全性仍由现有 HMAC-SM3 校验和密钥保密提供，不能用未加密摘要代替 MAC。

## 7. 实测性能分析

{time_table}

Runner 计算总耗时 `{m['total_wall_seconds']:.6f}` 秒；含输出发布与收尾检查的完整执行耗时 `{status['elapsed_seconds']:.6f}` 秒（约 {status['elapsed_seconds']/60:.2f} 分钟）。macOS 进程峰值 RSS `{status['peak_rss_bytes']/1024**2:.2f}` MiB（平台 `ru_maxrss`，为进程峰值，并非各阶段独占内存）。特征提取/对齐、检测与批量计算耗时均为本轮实测；这些阶段统计存在 Query 总时长等嵌套项，不能把所有计时行直接相加。

上述阶段与执行计时使用 `time.perf_counter`。UTC 起止时刻为 `{status['started_at_utc']}` 至 `{status['finished_at_utc']}`，两者相减的实际日历历时为 `{calendar_seconds:.6f}` 秒（{calendar_seconds/60:.2f} 分钟），与 `perf_counter` 差额为 `{calendar_seconds-status['elapsed_seconds']:.6f}` 秒。日志中观察到两类时钟进度不一致，可能涉及系统暂停或时钟调整，原因未独立确证；差额不归入模型、加密或解密阶段，不用于声称算法性能。

观察到时钟差额后，在加密恢复 Gallery 约 410/502 的进度启用了临时 `caffeinate -i -w 60400` 空闲休眠断言，仅在实验进程存活期间有效；未改变 CPU/GPU 推理目标、线程配置、模型、数据、阈值或加密参数，也未补跑此前阶段。具体启用时间见 `.execution/idle_sleep_assertion.json`。

离线实验包括一次性完整 Gallery 特征建立，并非线上单请求延迟 benchmark。不能以离线批量总耗时除以 Query 数宣称端到端响应达标，也不能以本轮文件加密时间替代 SM4 纯算法 500 MB/s 指标。历史国密性能结果保留，本次未重新优化或重测原语吞吐。

## 8. 失败样本与未评分数据

{extraction_table}

特征失败原因（plaintext）：`{json.dumps(failure_types['plaintext'],ensure_ascii=False)}`；其中 bbox 相关失败 {bbox} 条。encrypted_restore：`{json.dumps(failure_types['encrypted_restore'],ensure_ascii=False)}`。检测到脸、成功提取 embedding、媒体至少有一个有效特征是不同计数口径；部分帧失败不等于整个 Gallery 视频失败。空 Query 或 Gallery 保留在评估中，未因失败而剔除身份。

未评分原因（按 Runner 优先归因，类别互斥，不应当作独立事件数相加）：`{json.dumps(tf['unscored_reasons'],ensure_ascii=False)}`。plaintext Query 结果分类：`{json.dumps(analysis['per_query_outcomes'],ensure_ascii=False)}`。`query_diagnostics.csv` 保存每个 Query 的正确身份排名、Top-1 分数、genuine 分数、分数差、接受状态和缺失原因；`failed_queries_plaintext.csv` 为 Top-1 未命中子集；`feature_failures.csv` 保留媒体、帧、脸序号和异常原因。多脸 Query 的无关人脸仍可能影响最大分数，这是协议局限，未在 test 后按观察修改。

## 9. 讨论、局限性与结论

{selection_table}

selection 用于候选进入 test 的判断及阈值确定；上述 test 为该固定方案的一次独立评估，两批身份和 pair 不混用。100 身份探索结果继续留档于 `ytf_models_sample_100_20261101`（SFace 79%/83%，ArcFace 85%/86% 的 Top-1/Recall@5）；其较小 Gallery 规模与完整 selection/test 不同，不用来宣称正式性能。此前 Gallery 质量筛选/均值聚合的失败记录仍保留，本次未叠加。

本轮 ArcFace 相比同一 SFace test 子集，Top-1 变化 `{comparisons[0]['absolute_delta_percentage_points']:+.4f}`、Recall@5 变化 `{comparisons[1]['absolute_delta_percentage_points']:+.4f}`、冻结阈值 TAR 变化 `{comparisons[2]['absolute_delta_percentage_points']:+.4f}` 个百分点。课程 Top-1≥95% 是否满足：`{a['top1_accuracy']['value']>=.95}`；TAR≥90% 且实际 FAR≤1e-3 是否满足：`{atf['tar']>=.90 and atf['far']<=.001}`。不满足时如实报告差距，不修改阈值后重复 test。

Top-1 距 95% 的非负差距为 `{max(0.,100*(.95-a['top1_accuracy']['value'])):.4f}` 个百分点；TAR 距 90% 的非负差距为 `{max(0.,100*(.90-atf['tar'])):.4f}` 个百分点。可用 genuine 分数比例为 `{atf['scored_genuine_count']/502:.4%}`，在当前缺失分数拒绝策略下它也是 TAR 的可达到上界；即使提高可评分样本的区分能力，也不能把缺失 trial 自动算作接受。

局限包括仅一个固定身份划分、无重复试验或身份簇重采样置信区间、配对共享身份而并非独立 Bernoulli 样本、检测缺失与 bbox 异常、多脸最大分数偏差，以及仅本机 CPU 和 1 fps 重建视频条件。数据规模也不足以证明不少于 1 万身份、10 万段视频、100 万帧的综合课程指标。没有把公开参考 benchmark 或预估耗时混入实测结果，没有对本次 test 再做阈值扫描。

## 10. 指标定义、复现命令与文件

- Top-1 = 正确身份排名第 1 的 Query 数 / 全部 502 个 Query；Recall@5 = 正确身份进入前 5 的 Query 数 / 502。未检出或无法评分的 Query 计未命中，不改变分母。
- Genuine 共 502 个，同身份 Query/Gallery 各一对；impostor 共 502×501=251,502 个。每模式总 pair=502²=252,004。
- TAR = 评分≥冻结阈值的 genuine 数 / 502；FAR = 评分≥冻结阈值的 impostor 数 / 251,502。缺失分数拒绝且留在分母。FAR 有效分辨率 = 1 / 有效 impostor 分数数，至少 1,000 个有效 impostor 才可估计目标 1e-3。
- 百分点差 = `(ArcFace比例 - SFace比例) × 100`，不等于相对百分比提升。TAR 与 Top-1 是验证与检索两个不同任务，不应互换。

本轮实际命令（从项目根目录；输出已存在时脚本拒绝再次执行）：

```bash
.venv/bin/python -u scripts/run_ytf_arcface_test.py --preflight-only
.venv/bin/python -u scripts/run_ytf_arcface_test.py
.venv/bin/python scripts/report_ytf_arcface_test.py
```

识别默认值在 `src/facecheck/recognition.py`；Web/API 的 Query 和视频路径使用同一类型配置；YTF CLI 和实验脚本默认 ArcFace，显式 `SFaceConfig` 或 `--recognition-model sface` 可回退。原有加密实现、模型权重、数据、历史正式结果和 benchmark 未修改。相关定向验证共 **88 项通过**，覆盖模型切换、ArcFace/SFace、Runner/协议、API/健康、分块恢复、跨后端兼容与密文/MAC/错密钥拒绝；未运行完整测试套件。源码编译和启动 shell 语法检查通过。

原始 `config.json`、`metrics.json`、`query_results.csv`、`candidate_results.csv`、`genuine_impostor_pairs.csv`、`feature_summary.json`、`feature_failures.csv`、`scoring_diagnostics.json` 和 `timing.csv` 均独立保存；`analysis.json`、`validation.json`、`model_comparison.csv`、Query 诊断及 SHA-256 审计 CSV 是本报告派生证据。`.execution` 中保存身份清单、环境/模型/输入校验值、历史文件校验快照、源码校验值、日志和明确的 complete/incomplete 状态。未完成 run 不生成正式分析报告。
'''
    with (out/'course_practice_report.md').open('x',encoding='utf-8') as f: f.write(report)
    print(json.dumps({'report':str(out/'course_practice_report.md'),'comparison':comparisons,
                      'validation':validation,'timing':m['total_wall_seconds']},indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
