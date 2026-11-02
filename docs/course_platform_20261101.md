# FaceCheck 课程展示与批量实验平台

本次改造保留 FastAPI、SQLite、原生 HTML/JavaScript，不新增运行依赖，不修改识别、检测或密码算法。所有历史实验和 benchmark 保留。新任务从白名单配置启动，不自动启动全量 selection/test。

## 现有架构与本次集成

原前端包含总览、视频管理、单视频检索、安全中心、静态实验分析与审计查询。视频任务使用 `tasks` 表，事件使用 `audit_events` 表，单视频上传/检索由 `api.py` 与 `VideoTaskService` 同步执行。原接口继续兼容。

原批量实验须从终端运行 `facecheck.experiment`、`facecheck.ytf_runner` 或 `scripts/run_ytf_*.py`，没有 Web 批量进度表。YTF Runner 本身已支持每媒体/每模式内存缓存、identity-disjoint 协议、selection/test scope、评分诊断、报告和原子发布。因此保留原 Runner，不重构缓存，不修改采样或指标定义。

模块对应关系：

| 能力 | 实际模块与路径 |
|---|---|
| 密码与后端选择 | `crypto.py`、`crypto_backends.py`、`crypto_backend.py`，本次未修改 |
| 分块、派生、完整性与恢复 | `storage.py`、`chunks.py`，本次未修改 |
| YuNet / ArcFace / SFace | `face_detection.py`、`face_embedding.py`、`recognition.py`，本次未修改 |
| 单视频检索 | `face_search.py`、`service.py`，本次未修改 |
| identity-disjoint 评价 | `ytf_qg_protocol.py`、`ytf_runner.py`，本次未修改 |
| 登记输入与只读证据 | 新增 `platform_catalog.py` |
| 持久化任务与工作进程 | 新增 `platform_jobs.py`、`platform_worker.py` |
| 平台 API | 新增 `platform_api.py`，由 `main.py` 挂载 |
| 展示与轮询 | `frontend/index.html`、新增 `platform.js` 与 `platform.css` |

## 页面与后端的真实对应关系

| 页面 | 数据 / 操作 |
|---|---|
| 总览 | 运行时默认模型、实际密码后端和库版本、数据集登记、视频计数、近期批量任务、正式 test 概览 |
| 视频管理 | 原有上传与加密；新详情按钮调用原有恢复函数，逐块认证并校验大小/SHA-256；校验通过后可下载临时恢复文件 |
| 人脸检索 | 原单视频 API，返回候选编号、帧号、时间戳、相似度；运行时模型信息由后端提供 |
| 批量数据集实验 | 固定配置、资源预检、显式启动、状态/进度/日志、取消、历史任务与结果文件 |
| 数据集 1:N 检索 | 固定20身份集 Query，真实后台提取/评分；Gallery 限制在同一分区，单媒体只提取一次，返回 Top-K 与帧位置 |
| 历史实验 | 只读正式 test、selection、SFace 历史、100身份探索、负收益 Gallery 方案；指标表、失败CSV预览、原始报告下载 |
| 国密与性能记录 | 每份 benchmark 分别显示环境、多轮中位数/最小值/最大值/标准差、原语与完整封装、1080p/分辨率/API结果；下载原始JSON |
| 安全中心 | 当前 SM4-CBC/HMAC-SM3、实际库版本、密钥版本/派生/配置状态、授权和撤销限制；不显示密钥 |
| 课题指标对照 | 完整8项研究内容与18行指标（SM2签名/验签分行）；实际数值、来源、状态与口径 |
| 审计 | 原查询接口；新增 SQLite 事件 SHA-256 链、全链校验接口及带校验摘要的 CSV/JSON/Markdown 导出；实验、恢复、校验、导出事件记录到 SQLite |

实时数据集检索不是历史候选回放。`encrypted_restore` 均表示“认证与解密恢复后进行明文检索”，未实现密文域安全索引/ANN。探索性检索报告不产生正式 Top-1/TAR 结论。

## 新增 API

统一前缀 `/api/v1/platform`。原 `/api/v1/videos`、`/tasks`、`/audit`、`/security/summary` 不变。

| 方法 / 相对路径 | 功能 |
|---|---|
| GET `/overview` | 运行时配置、规模、近期任务、正式结果 |
| GET `/datasets`、`/datasets/{id}` | 已登记 manifest、身份/Gallery/Query/帧数、分区与运行范围 |
| GET `/datasets/ytf_smoke20/queries` | 已登记 Query ID、身份与分区 |
| POST `/jobs/preview` | 校验媒体/模型、输入摘要、输出空间、耗时规划；不创建任务 |
| POST `/jobs` | 校验同上，返回202；独立进程启动 |
| GET `/jobs?limit=50&offset=0` | 持久化任务历史；检测异常退出 |
| GET `/jobs/{id}` | 状态、媒体数、阶段、计时、错误；完整任务带结果 |
| GET `/jobs/{id}/logs?after=0` | 按序号增量日志，每页最多200条 |
| POST `/jobs/{id}/cancel` | 请求合作式安全取消，退出前不释放任务槽 |
| GET `/jobs/{id}/files/{filename}` | 仅完整任务的白名单结果下载 |
| GET `/jobs/{id}/preview/{filename}?limit=20` | 完整任务的诊断CSV前若干行 |
| GET `/histories`、`/histories/{id}` | 只读已登记历史结果 |
| GET `/histories/{id}/files/{filename}` | 历史白名单附件下载 |
| GET `/histories/{id}/preview/{filename}?limit=20` | 诊断/Query/对照CSV，最多100行 |
| GET `/benchmarks`、`/benchmarks/{id}` | 指标 benchmark 读取与下载 |
| GET `/requirements` | 研究内容与指标对照 |
| GET `/audit/integrity` | 在同一数据库快照内校验全链并返回检查范围/失败位置 |
| GET `/audit/export?format=csv\|json\|md&task_id=...` | 单事务审计快照及全链摘要；不截断为UI前50/200条 |
| POST `/videos/{task_id}/verify` | 临时恢复、认证/SHA-256校验、审计，随后删除临时明文 |
| POST `/videos/{task_id}/restore` | 同上并下载；响应结束删除临时明文 |

请求 schema 只接受登记数据集、模型、任务类型、已登记 Query、实际检索模式与有界 Top-K，拒绝额外字段（包括路径、命令、test scope、外部阈值）。批量模式固定 paired，网页明确显示两种实际模式。诊断预览与下载均检查白名单和目录边界，包括符号链接越界。

## 数据集与固定参数

- `ytf_smoke20`：`data/ytf/smoke_20/experiment_smoke_20.json`，原有20身份、20 Gallery视频、20 Query图像；原分区10 selection + 10 test。批量为该固定小集的探索性 full；不是正式 test 重跑。实时检索每个 Query 只检索其分区的10个 Gallery。
- `ytf_selection`：`data/ytf/experiment_full.json` 与 `query_gallery_identity_split.json`；只筛选原502个 threshold-selection 身份。不会创建或改变划分文件，不开放正式全量test入口。
- 帧数为 manifest 中 Gallery 的 `ytf_frame_count` 之和；Gallery 为1 fps重建 MP4，不是原始 YouTube视频。Query图像不能计作额外视频。
- YuNet置信度0.9、视频1秒采样、Query/Gallery保留有效检测、逐帧特征、cosine/max、默认Top-K=5；不启用质量筛选、均值聚合或标注Query框。
- 模型仅ArcFace/SFace二选一，各自使用已有对齐/预处理；运行输入记录模型/检测器/manifest/split SHA-256。Web默认仍为已有ArcFace配置。
- 阈值只能由该次 plaintext selection 分数经原有协议选取。正式test历史阈值不被改写；不提供重新调参或重跑正式test按钮。
- 时间规划是资源提示，不作为实测指标。完整selection paired可能1–3小时甚至更久；不要为演示随意启动。

## 任务持久化、取消与边界

`experiment_jobs` 保存 queued/running/completed/failed/cancelled、冻结配置、独立输出路径、PID、进度、起止时间、工作计时与错误；`experiment_logs` 保存递增日志。原视频表不复用。

SQLite partial UNIQUE index 在 queued/running 状态只允许一个任务槽；worker 再以 POSIX 文件锁保护运行。HTTP用 `Popen` 固定参数、`shell=False` 的默认行为启动模块，不在请求线程执行推理。父进程负责回收子进程；服务重启后通过 PID + worker模块 + job ID 核对存活，异常退出记为失败/incomplete。仅处理完全部输出才完成；不支持断点续跑，不自动重跑。

取消通过数据库标志，在媒体提取前后、阶段边界检查；长视频或一次评分期间取消不会立即生效。保持运行槽，直到worker实际退出。Runner异常时清理staging及临时加密/恢复工作目录。进程被强杀时可能残留未发布staging/系统临时文件；它们不会作为完整结果开放下载，需要本机管理员核对后清理。

输出为 `<storage父目录>/platform_runs/<随机job ID>/`，批量结果在其 `result/` 子目录。任务启动配置在 `request.json`；结果目录由Runner原子发布，历史目录从不作为目标。UUID目录以 `exist_ok=False` 创建。完成状态、日志跨浏览器/服务重启保留；worker失败不把部分指标当正式结果。

批量与数据集检索使用进程内随机实验密钥，退出即销毁，不写入配置、日志、特征或结果。Web上传仍使用现有配置的运行时密钥和版本。保留既有GmSSL/Python/OpenSSL路径、KDF、CBC/填充、MAC绑定、先认证再解密和恢复SHA-256语义。

独立命令行脚本不参与平台任务锁；不要同时手动启动离线全量脚本。单视频上传/检索仍为原同步API，可能与批量worker竞争CPU，课堂应分步演示。本机演示没有RBAC/KMS/多租户隔离，不提供云端规模化承诺。

## 指标的数据来源和解释

页面不硬编码实测百分比。`PlatformCatalog`每次从登记文件解析数据：

- ArcFace正式test：`data/ytf/experiments/ytf_arcface_test_20261101/{metrics,config,analysis,validation}.json`；报告、失败样本、恢复校验CSV可下载。
- ArcFace完整selection：`data/ytf/experiments/ytf_arcface_w600k_r50_20261101/selection_full_plaintext/`。
- SFace正式历史：`ytf_full_run_02`；公平同test子集对照读取ArcFace的`analysis.json.baseline_test_*`，不拿selection+test汇总直接与单test比较。
- 100身份探索：`ytf_models_sample_100_20261101`；失败质量方案：`ytf_gallery_quality_20261101/selection`，均保留单独类别。
- 性能列表读取`results/indicator_benchmark_*.json`，逐份展示，不从其中挑选单次峰值。指标表的SM2/SM4基准明确引用`indicator_benchmark_20261101_sm4_final.json`，不是当前设备重新测量。较快optimized记录同样保留，可查其环境元数据不足的限制。
- 错误密钥/分块篡改试验读取`indicator_benchmark_20261031.json`；这些不能证明RBAC或特征篡改总体检出率。

实际ArcFace正式test为Top-1 86.06%、Recall@5 89.24%、TAR86.65%，仍低于Top-1≥95%、TAR≥90%目标。502 test身份不满足万人规模。恢复后两模式指标相同、502视频SHA-256一致，不代表密文域ANN达标。

SM4完整封装吞吐未达500 MB/s；SM2本机多轮benchmark达标。两帧1080p合成样本、5分辨率、10次本机请求仅作对应样本证据，不扩大解释为持续视频、3类终端、日吞吐或长期成功率。未测指标返回null；无密文域检索、可撤销模板、边缘云协同、完整KMS/RBAC、审计防删证明等能力，如实标注。

性能页面本次提供历史数据与下载，不新增网页benchmark执行器，也没有重新运行历史benchmark。原脚本仍为`scripts/benchmark_indicators.py`，需人工指定全新输出文件；不要覆盖历史JSON。

## 定向验证

执行：

```bash
.venv/bin/python -m pytest -o addopts='' tests/test_platform.py tests/test_api.py tests/test_health.py tests/test_ytf_runner.py tests/test_crypto.py -q
node --check frontend/platform.js
```

结果：**57 passed in 1.56s**。仅相关测试，未运行完整套件。新增测试覆盖登记/输入限制、真实文件指标来源、路径与符号链接越界、单任务槽、不覆盖旧输出、queued/running取消、实际子进程启动后安全取消、异常退出、重启后记录保留、模型摘要变化拒绝、媒体缓存复用、实时检索分区、诊断预览和审计导出。

后续审计链原型单独新增7项定向测试；与视频 API、云端访问控制和平台导出组合运行时共 **39 passed**。覆盖并发写入无分叉、旧数据库一次性补链、内容改写/中间删除/重排检测及三种格式的成功/失败校验状态。此结果只验证代码路径，不表示审计完整率达标；不受独立锚点保护的尾部截断和整库回滚仍不能检测。

微型夹具使用模拟embedding与真实Runner/SM4-CBC/HMAC-SM3存储恢复，验证两模式一致；不把模拟embedding结果作为识别性能。单视频测试用16×16合成视频，验证SHA-256、下载与篡改拒绝。没有启动真实YTF数据集实验、正式selection/test或新benchmark。

Chrome headless真实浏览器检查：总览与其余9个导航分区、正式结果详情、资源预检、诊断CSV预览、合成视频上传与恢复SHA-256校验通过；桌面1440×1000和手机390×844无脚本异常，手机无页面横向溢出。隔离临时数据库中资源预检后的实验任务数为0；未在正式数据库创建演示实验。浏览器检查不是长期可用性或识别准确率证据。

## 启动与课堂演示

```bash
cd /Users/mac/Desktop/nuist/facecheck
./scripts/start_local.sh
```

访问`http://127.0.0.1:8000/`。如旧服务仍在运行，先停止并重新启动以加载新增API；启动脚本在端口已有健康服务时会直接复用它。

建议按以下顺序演示：

1. 总览展示实际默认模型、真实正式test指标与研究边界。
2. 课题指标对照展示8项研究内容、目标/证据/状态，说明未达与未测项。
3. 历史实验打开ArcFace正式test与selection，查看同test SFace对照、失败CSV和课程报告下载；说明test未参与选型/阈值选择。
4. 国密性能记录比较多轮原语/完整封装及不同历史环境，说明SM4未达500 MB/s。
5. 上传已授权小视频，在任务详情执行完整性恢复校验，下载恢复文件；到人脸检索上传已授权Query，查看真实帧号/时间戳/相似度。
6. 批量页选择20身份集和模型，资源预检；只有决定实际运行时再点击启动。查看queued/running、媒体进度、持久化日志和取消。完成后报告自动随Runner发布。
7. 数据集检索选择固定Query、模型、模式执行1:N，与批量任务共用资源槽。该步骤可能需要数分钟；勿为演示启动完整selection。
8. 审计页查询事件，导出CSV或生成Markdown快照报告。新导出的审计快照不证明审计完整率/生成成功率目标。

本次改造未重测识别准确率；历史正式结果仅展示，未修改。新增代码不承诺尚未验证的云端部署、持续负载或国密性能。
