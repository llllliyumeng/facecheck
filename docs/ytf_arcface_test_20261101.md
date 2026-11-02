# YTF ArcFace 正式 test：课程实践数据与分析报告

## 1. 实验目的与数据集说明

本实验在现有 SFace 正式基线和 ArcFace 完整 selection 结果之上，检验一次冻结 ArcFace 方案在独立 test 身份上的检索与验证表现，并比较加密存储、解密恢复前后的结果。本次只运行一轮正式 test：两个规定模式各执行一次完整推理，未重跑 selection，未依据 test 分数重新选阈值、模型或参数。

项目 manifest 共 1,004 个身份、2,008 个媒体记录，selection/test 各 502 个身份；每个身份对应一个 Gallery 视频和一个 Query 图片。这里的“全量 test”指现有协议的全部 502 个 test 身份，而非 1,004 个身份合并评估或整个原始 YTF 发布集。Gallery MP4 是 YTF 帧图重建的 1 fps 实验输入，并非原始 YouTube MP4。原始 manifest 和划分文件未修改，见配置校验值。

## 2. 实验环境、模型版本与参数

- 环境：Apple M4、16 GiB 内存；macOS-26.6.2-arm64-arm-64bit，arm64；Python 3.9.6，OpenCV 4.14.0，NumPy 2.0.2。
- ArcFace：InsightFace v0.7 `buffalo_l / w600k_r50` ResNet-50，文件 `models/arcface_w600k_r50.onnx`；SHA-256 `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43`。权重来源为官方 InsightFace v0.7 buffalo_l 发布包，许可限非商业研究用途；不应把代码许可等同于权重许可。
- 推理：OpenCV DNN CPU；YuNet 五点与标准 ArcFace 模板做相似变换，112×112 对齐；BGR→RGB；`(pixel-127.5)/127.5`；512 维 float32 embedding，L2 归一化。
- 检测与采样：既有 YuNet 模型（SHA-256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`），置信度 0.9、NMS 0.3、每 1 秒采样。Gallery 保留全部配置采样帧与有效检测的独立特征；不做质量过滤、均值聚合或 Query 标注框筛选。
- 相似度：同一视频配对的全部 Query/Gallery 特征 cosine similarity 取最大值；按相似度降序、Gallery ID 升序打破排名平局。检索指标按身份/视频配对排名，不将视频帧当独立身份。
- 冻结阈值：`0.2731515063106565`，显示值 `0.2731515063`；分数 `>=` 阈值接受。仅来自已完成的 ArcFace plaintext selection 分数，未使用 SFace 阈值 `0.43351951106296693`。
- 加密后端：gmssl-native，GmSSL 3.2.0；动态库 `/opt/homebrew/opt/gmssl/lib/libgmssl.dylib`。保留现有 SM4-CBC、HMAC-SM3、1 MiB 分块、4 workers、密钥派生与认证恢复顺序；实验密钥从既有私有文件只读获取，未写入报告或结果。

## 3. 数据划分与实验流程

划分沿用 `SHA-256(seed + NUL + identity_id)`，seed 为 `facecheck-ytf-qg-v1`；selection 与 test 身份严格不相交。manifest SHA-256 为 `dbe5b240c7a3560f6fdeb0bf74f9e3d086bdf23b18179d53968b7ef600e4c9f8`，划分文件 SHA-256 为 `b77f87126b57ca0c938c5a7ffcc3b881bc1f2b531e074e9565f0f030cdd0f197`。运行与事后检查均核对 502 test 身份、502 Gallery、502 Query；每模式完整 252,004 个配对，跨分区配对为 0。执行身份清单存放于同名 `.execution/identity_ids.json`。

plaintext：对全部 test 媒体各提取一次，媒体特征保存在仅本次模式有效的内存缓存，随后完成配对、排名和冻结阈值验证。encrypted_restore：先逐个加密 Gallery 视频，按既有协议认证恢复并检查 SHA-256，再独立提取恢复媒体特征；Query 图片作为共享的原始输入，不加密。两模式的特征不互相复用，以实际检验恢复一致性。缓存不跨模型、配置或运行复用，向量不落盘。

## 4. ArcFace 与 SFace 对比口径

SFace 保留为显式回退：128 维 SFace embedding 与其 `alignCrop` 路径；ArcFace 为 512 维并使用匹配的五点对齐与预处理。YuNet、媒体、采样、最大相似度配对、身份划分和指标定义一致。本次未重跑 SFace，基线来自 `ytf_full_run_02` 的原始 test CSV 和冻结阈值验证结果。

历史常见的 Top-1 `783/1004=77.99%`、Recall@5 `846/1004=84.26%` 是 selection+test 合并值；不能直接与本次 502 身份 test 值计算模型提升。本报告从历史原始 pair/query 文件重建同一 test 子集：Top-1 `399/502`、Recall@5 `431/502`，保持对照分母一致。SFace 的 test FAR 本身略高于 1e-3，表中仍如实保留其冻结阈值下 TAR。

## 5. 正式 test 结果

| 指标 | SFace 正式 test 子集 | ArcFace plaintext test | ArcFace encrypted_restore test | 差值（百分点） |
|---|---|---|---|---|
| Top-1 | 399/502 (79.4821%) | 432/502 (86.0558%) | 86.0558% | +6.5737 |
| Recall@5 | 431/502 (85.8566%) | 448/502 (89.2430%) | 89.2430% | +3.3865 |
| 冻结 selection 阈值下 TAR | 395/502 (78.6853%) | 435/502 (86.6534%) | 86.6534% | +7.9681 |
| 实际 FAR | 253/251502 (0.0010059562) | 251/251502 (0.0009980040) | 0.0009980040 | -0.000795 |

两模型分别使用各自 selection 冻结阈值。ArcFace 有效 impostor 数为 239,000，FAR 分辨率为 `4.18410041841e-06`，足以估计 1e-3；实际 test FAR 约束是否满足：**True**。若实际 FAR 超过目标，应将该 TAR 描述为“selection 冻结阈值下的 TAR”，不能宣称在 test 达到 TAR@FAR≤1e-3，也不得现场升高阈值重测。

| 类型 | 总试验数 | 有效分数数 | 未评分数 |
|---|---|---|---|
| genuine | 502 | 478 | 24 |
| impostor | 251502 | 239000 | 12502 |
| 合计 | 252004 | 239478 | 12526 |

未评分率 `4.9706%`。本报告遵循原有保守缺失策略：未评分 pair 均拒绝，但保留在 genuine/impostor 总试验分母中；有效分数数只用于 FAR 分辨率和可估计性判定。

## 6. 加密存储与恢复一致性

两模式识别指标完全相同：`True`；所有配对分数及缺失状态逐项精确相同：`True`。Runner 实测最大绝对分数差 `0.0`，比较容差 `1e-06`。本轮加密 502 个 Gallery 视频、653 个分块，认证恢复 502 个视频、653 个分块；**502/502** 个恢复视频 SHA-256 与源文件一致，逐文件证据见 `restore_sha256_audit.csv`。

这证明本次加密、认证与恢复未引入识别损失；流程是恢复并验证后的明文检索，不能作为密文域 ANN/安全索引已实现的证据。SHA-256 用于与可信源值核对；认证安全性仍由现有 HMAC-SM3 校验和密钥保密提供，不能用未加密摘要代替 MAC。

## 7. 实测性能分析

| 计时阶段（秒） | plaintext | encrypted_restore |
|---|---|---|
| 读取/解码/抽帧 | 11.151113 | 10.697023 |
| YuNet 检测 | 336.176853 | 349.257937 |
| 特征提取/对齐 | 2432.425010 | 2589.925413 |
| 相似度计算 | 13.809249 | 13.336816 |
| 排名构建 | 2.588468 | 2.575707 |
| 加密与存储 | 0.000000 | 4.605686 |
| 恢复与认证 | 0.000000 | 3.338330 |
| 单模式总耗时 | 2800.575830 | 2978.686473 |

Runner 计算总耗时 `5782.639777` 秒；含输出发布与收尾检查的完整执行耗时 `5788.167146` 秒（约 96.47 分钟）。macOS 进程峰值 RSS `2575.05` MiB（平台 `ru_maxrss`，为进程峰值，并非各阶段独占内存）。特征提取/对齐、检测与批量计算耗时均为本轮实测；这些阶段统计存在 Query 总时长等嵌套项，不能把所有计时行直接相加。

上述阶段与执行计时使用 `time.perf_counter`。UTC 起止时刻为 `2026-11-01T09:31:57.940154+00:00` 至 `2026-11-01T11:48:15.756428+00:00`，两者相减的实际日历历时为 `8177.816274` 秒（136.30 分钟），与 `perf_counter` 差额为 `2389.649128` 秒。日志中观察到两类时钟进度不一致，可能涉及系统暂停或时钟调整，原因未独立确证；差额不归入模型、加密或解密阶段，不用于声称算法性能。

观察到时钟差额后，在加密恢复 Gallery 约 410/502 的进度启用了临时 `caffeinate -i -w 60400` 空闲休眠断言，仅在实验进程存活期间有效；未改变 CPU/GPU 推理目标、线程配置、模型、数据、阈值或加密参数，也未补跑此前阶段。具体启用时间见 `.execution/idle_sleep_assertion.json`。

离线实验包括一次性完整 Gallery 特征建立，并非线上单请求延迟 benchmark。不能以离线批量总耗时除以 Query 数宣称端到端响应达标，也不能以本轮文件加密时间替代 SM4 纯算法 500 MB/s 指标。历史国密性能结果保留，本次未重新优化或重测原语吞吐。

## 8. 失败样本与未评分数据

| 角色 | 有效媒体 | 无有效特征媒体 | embedding 数 | 检测人脸数 | 抽样帧/图片数 | 特征失败数 |
|---|---|---|---|---|---|---|
| query | 478 | 24 | 502 | 502 | 502 | 0 |
| gallery | 501 | 1 | 91030 | 91030 | 91212 | 0 |

特征失败原因（plaintext）：`{}`；其中 bbox 相关失败 0 条。encrypted_restore：`{}`。检测到脸、成功提取 embedding、媒体至少有一个有效特征是不同计数口径；部分帧失败不等于整个 Gallery 视频失败。空 Query 或 Gallery 保留在评估中，未因失败而剔除身份。

未评分原因（按 Runner 优先归因，类别互斥，不应当作独立事件数相加）：`{"gallery_face_not_detected": 478, "query_face_not_detected": 12048}`。plaintext Query 结果分类：`{"correct_top1": 432, "wrong_top1": 46, "query_no_features": 24}`。`query_diagnostics.csv` 保存每个 Query 的正确身份排名、Top-1 分数、genuine 分数、分数差、接受状态和缺失原因；`failed_queries_plaintext.csv` 为 Top-1 未命中子集；`feature_failures.csv` 保留媒体、帧、脸序号和异常原因。多脸 Query 的无关人脸仍可能影响最大分数，这是协议局限，未在 test 后按观察修改。

## 9. 讨论、局限性与结论

| 指标 | ArcFace selection（502 身份） | ArcFace test（502 身份） |
|---|---|---|
| Top-1 | 86.2550% | 86.0558% |
| Recall@5 | 88.2470% | 89.2430% |
| TAR | 86.0558% | 86.6534% |
| FAR | 0.0008628162 | 0.0009980040 |

selection 用于候选进入 test 的判断及阈值确定；上述 test 为该固定方案的一次独立评估，两批身份和 pair 不混用。100 身份探索结果继续留档于 `ytf_models_sample_100_20261101`（SFace 79%/83%，ArcFace 85%/86% 的 Top-1/Recall@5）；其较小 Gallery 规模与完整 selection/test 不同，不用来宣称正式性能。此前 Gallery 质量筛选/均值聚合的失败记录仍保留，本次未叠加。

本轮 ArcFace 相比同一 SFace test 子集，Top-1 变化 `+6.5737`、Recall@5 变化 `+3.3865`、冻结阈值 TAR 变化 `+7.9681` 个百分点。课程 Top-1≥95% 是否满足：`False`；TAR≥90% 且实际 FAR≤1e-3 是否满足：`False`。不满足时如实报告差距，不修改阈值后重复 test。

Top-1 距 95% 的非负差距为 `8.9442` 个百分点；TAR 距 90% 的非负差距为 `3.3466` 个百分点。可用 genuine 分数比例为 `95.2191%`，在当前缺失分数拒绝策略下它也是 TAR 的可达到上界；即使提高可评分样本的区分能力，也不能把缺失 trial 自动算作接受。

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

## 11. 补充核验与本轮文件变更

`baseline_media_audit.json` 对两模式各 1,004 个媒体记录逐项核对，确认与 SFace 正式 test 子集的源文件 SHA-256、采样帧数、检测人脸数、特征数和特征失败数完全一致。提升来源于 ArcFace 及其匹配的对齐/预处理，没有叠加采样、检测或 Gallery 聚合改动。当前 test 的 bbox/提取异常为 0，不代表历史 selection 的越界记录已消失；旧记录继续原样保留。

正确身份排名分组：`{'rank_1': 432, 'rank_above_5': 30, 'unscored_genuine': 24, 'rank_2_to_5': 16}`。SFace→ArcFace 的 Top-1 迁移：`{'correct_to_correct': 397, 'miss_to_miss': 68, 'miss_to_correct': 35, 'correct_to_miss': 2}`；其中 miss 包含错误排名和无特征情况。有效 genuine 中低于冻结阈值的有 43 对，另有 24 对未评分，二者都计为拒绝。完整数据见 `failure_summary.json`，这些事后统计未用于调参或重跑。

本轮新增/修改文件（原始模型、密码实现、旧测试和历史结果未修改）：

- `src/facecheck/recognition.py`
- `src/facecheck/service.py`
- `src/facecheck/face_search.py`
- `src/facecheck/experiment.py`
- `src/facecheck/ytf_runner.py`
- `scripts/run_ytf_full.py`
- `scripts/run_ytf_arcface_test.py`
- `scripts/report_ytf_arcface_test.py`
- `scripts/start_local.sh`
- `.env.example`
- `frontend/index.html`
- `README.md`
- `models/README.md`
- `docs/ytf_arcface_w600k_r50.md`
- `tests/test_recognition_defaults.py`
- `docs/ytf_arcface_test_20261101.md`

本轮定向测试实际命令：

```bash
.venv/bin/pytest -q tests/test_recognition_defaults.py tests/test_face_embedding.py tests/test_face_search.py tests/test_ytf_runner.py tests/test_ytf_qg_protocol.py tests/test_experiment.py tests/test_api.py tests/test_health.py tests/test_video_pipeline.py::test_parallel_chunk_workers_preserve_authenticated_round_trip tests/test_video_pipeline.py::test_storage_rejects_missing_swapped_tampered_and_metadata_chunks tests/test_crypto_backends.py::test_sm4_cbc_cross_backend_and_envelope_compatibility tests/test_crypto_backends.py::test_sm4_envelope_rejects_tampering_and_wrong_keys
```

结果：88 项通过。本轮未删除或修改已有测试，仅增加模型默认值与 Web/API 双端配置一致性的定向验证。
