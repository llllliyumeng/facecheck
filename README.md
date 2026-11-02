# FaceCheck

云端人像视频存储、传输与检索隐私保护系统课程实践项目。

FaceCheck 使用 Python、FastAPI、SQLite 和 OpenCV，提供本机演示与 HTTPS 云端单实例部署两种模式。系统面向**获得授权的测试视频和查询图片**，提供：

- 视频上传和任务状态管理；
- 视频分片加密存储；
- SM4-CBC 数据加密；
- HMAC-SM3 分片完整性认证；
- 恢复后的 SHA-256 校验；
- SQLite 审计事件 SHA-256 前向链、校验接口和带链状态的审计导出；
- YuNet 人脸检测；
- ArcFace（默认）/ SFace（可选）特征提取；
- 已加密视频恢复并进行内存相似度检索；
- 安全摘要、审计记录和性能实验展示；
- 中文 FaceCheck 课程展示平台：批量数据集任务、历史实验、国密性能记录、课题指标对照、审计导出；
- 云端 HTTPS、组织级 admin/operator/auditor 权限、可撤销令牌与持久化会话；
- 二进制密钥文件注入、可信任务库 manifest 摘要锚点；
- SM4-CBC + HMAC-SM3 保护的持久化特征缓存，复用同一逐帧特征和评分规则。

默认仍为仅绑定 `127.0.0.1` 的本机演示。云端入口强制 HTTPS、有效访问凭据、独立强密钥和 GmSSL native；支持同一组织内的角色授权。新增云端链路与受保护特征缓存已有本机真实 HTTPS 验证，远程服务器和持续规模化负载尚未部署测量。

云端启动与实际验证结果见 [云端服务与在线检索改进](docs/cloud_service_20261101.md)。

## 云端服务入口与在线检索优化

先在私有目录创建一次运行配置（不会打印密钥或覆盖旧配置）：

```bash
.venv/bin/python scripts/init_cloud.py \
  --directory data/cloud-instance \
  --public-origin https://your-domain.example:8443
```

配置使用独立新数据库和存储，不混入原演示数据。配置目录中的密钥、凭据和数据需要受控保留；重启复用同一配置即可继续恢复视频和登录会话。浏览器令牌交换为 Secure、HttpOnly、SameSite 会话，不写 localStorage。

已有匹配域名的 TLS 证书时启动：

```bash
./scripts/start_cloud.sh data/cloud-instance/cloud.env /path/to/fullchain.pem /path/to/privkey.pem
```

Linux 单实例容器部署文件位于 `deploy/`，包含固定 GmSSL 3.2.0 源码版本、多阶段镜像和 Caddy HTTPS 反向代理。镜像首次构建涉及基础镜像与编译依赖下载；本机 Docker 引擎当前未启动，本轮仅验证 Compose 配置，未将容器构建或远程部署写作已完成。

在线特征缓存在 cloud 模式默认开启，本机可显式设置 `FACECHECK_FEATURE_CACHE=1`。首次请求认证恢复、提取既有逐帧特征后加密保存；后续请求验证缓存 MAC、完整源密文/manifest 摘要、模型权重和预处理/提取代码指纹，匹配才复用。模型、采样、Query 策略、维度、阈值和评分不因缓存改变。缓存损坏直接拒绝，不静默忽略。正式 YTF Runner 不使用该缓存。

## 目录结构

```text
facecheck/
├── frontend/                 # 中文控制台页面
├── models/                   # YuNet/SFace 随仓库分发；ArcFace 权重单独获取、不纳入 Git
├── scripts/                  # 本地启动脚本
├── src/facecheck/
│   ├── api.py                # FastAPI 路由和响应模型
│   ├── crypto.py             # SM4、HMAC-SM3 和加密封装
│   ├── crypto_backend.py     # Python/cryptography 兼容原语
│   ├── crypto_backends.py    # GmSSL native、Python 和可选 OpenSSL 后端
│   ├── database.py           # SQLite 视频任务和审计记录
│   ├── platform_api.py       # 数据集/批量任务/历史结果/指标/导出接口
│   ├── platform_catalog.py   # 只读白名单数据与真实指标来源
│   ├── platform_jobs.py      # SQLite 持久化任务和进程管理
│   ├── platform_worker.py    # 独立进程复用 YTF Runner
│   ├── face_search.py        # 视频恢复、人脸检测和检索
│   ├── service.py            # 业务服务层
│   └── storage.py            # 分片加密存储和恢复
├── tests/                    # 自动化测试
├── data/                     # 本地数据库和密文存储，不纳入 Git
├── .env.example              # 非敏感配置示例
├── pyproject.toml
└── uv.lock
```

## 环境要求

- Python 3.9 或更高版本；
- [uv](https://docs.astral.sh/uv/)（推荐，但不是必须）；
- macOS Apple Silicon、macOS Intel 或 Windows 10/11；
- 项目 `models/` 目录中已提供 YuNet 和 SFace；默认 ArcFace 权重须按[模型说明](models/README.md)单独获取；
- 仅使用明确授权的本地测试素材。

### 方式一：使用 uv（推荐）

安装 uv 后检查：

```bash
uv --version
python --version
```

如果 macOS/Linux 提示 `uv: command not found`，可以先执行：

```bash
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

Windows PowerShell 如果提示找不到 `uv`，请重新打开终端，或将 uv 的安装目录加入用户 `PATH`。

### 方式二：不使用 uv

Windows 和 macOS 都可以直接使用 Python 自带的虚拟环境和 `pip`。项目依赖声明在 `pyproject.toml` 中，首次安装时执行：

Windows PowerShell：

```powershell
cd C:\path\to\facecheck
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest httpx
```

如果 PowerShell 禁止激活脚本，可以不激活环境，直接使用虚拟环境中的 Python：

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip install pytest httpx
```

macOS：

```bash
cd /Users/mac/Desktop/nuist/facecheck
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest httpx
```

不使用 uv 时，后续命令将 `uv run` 替换为已激活虚拟环境中的 `python -m`：

```text
uv run uvicorn ...       → python -m uvicorn ...
uv run pytest -vv        → python -m pytest -vv
uv run python ...        → python ...
```

不使用 uv 的安装方式不会自动按照 `uv.lock` 锁定所有传递依赖版本。课程答辩和团队协作建议统一使用 uv；如果队友无法使用 uv，应记录 Python 版本和依赖安装时间，并优先保持 `pyproject.toml` 版本范围不变。

## 模型准备

仓库包含 YuNet 和 SFace；ArcFace 权重因文件较大且仅限非商业研究用途，不纳入普通 Git。默认使用 ArcFace 的成员需从官方 `buffalo_l` 模型包中单独提取权重，并核验 SHA-256。仓库自带文件与本机 ArcFace 文件为：

```text
models/face_detection_yunet_2023mar.onnx
models/face_recognition_sface_2021dec.onnx
```

另行获取后将 `w600k_r50.onnx` 保存为 `models/arcface_w600k_r50.onnx`。校验方法和许可边界见 [模型说明](models/README.md)。

模型来源、许可证说明和 SHA-256 校验值见 [models/README.md](models/README.md)。课程报告中应保留模型来源和许可证信息。

如果模型在其他目录，启动前设置：

```text
FACECHECK_YUNET_MODEL=/path/to/face_detection_yunet_2023mar.onnx
FACECHECK_RECOGNITION_MODEL=arcface
FACECHECK_ARCFACE_MODEL=/path/to/arcface_w600k_r50.onnx
FACECHECK_SFACE_MODEL=/path/to/face_recognition_sface_2021dec.onnx
```

默认识别模型为 InsightFace `buffalo_l / w600k_r50` ArcFace：OpenCV DNN CPU、112×112 五点相似变换对齐、RGB `(pixel-127.5)/127.5`、512 维 L2 归一化特征和 cosine similarity。Web/API 和 YTF CLI 共用默认模型配置；Web/API 未指定 `min_similarity` 时使用完整 selection 冻结阈值 `0.2731515063106565`（显示为 `0.2731515063`）。该阈值来自 502 个 selection 身份，未使用 test 数据选择。

设置 `FACECHECK_RECOGNITION_MODEL=sface` 可回退 Web/API；实验 CLI 使用 `--recognition-model sface --sface-model ...`。显式传入 `SFaceConfig` 的 Python 调用继续使用 SFace，不会被默认值覆盖，也不会自动使用 ArcFace 阈值。两种模型特征维度不同，不能混用。ArcFace 权重限非商业研究用途，来源、许可与校验值见 [模型说明](docs/ytf_arcface_w600k_r50.md)。

## macOS 启动

### 方式一：自动启动脚本（推荐）

在项目根目录执行：

```bash
cd /Users/mac/Desktop/nuist/facecheck
uv sync --python python3
chmod +x scripts/start_local.sh
./scripts/start_local.sh
```

脚本会检查虚拟环境和模型文件，默认使用本地演示密钥，并启动：

```text
http://127.0.0.1:8000/
```

端口被占用时：

```bash
PORT=8001 ./scripts/start_local.sh
```

脚本只适用于 macOS/Linux 的 Bash。生产环境不要使用脚本内置的演示密钥。

### 方式二：手动启动

```bash
cd /Users/mac/Desktop/nuist/facecheck
uv sync --python python3

export FACECHECK_SM4_KEY='0123456789abcdef'
export FACECHECK_MAC_KEY='local-demo-mac-key'
export FACECHECK_YUNET_MODEL="$PWD/models/face_detection_yunet_2023mar.onnx"
export FACECHECK_RECOGNITION_MODEL=arcface
export FACECHECK_ARCFACE_MODEL="$PWD/models/arcface_w600k_r50.onnx"
export FACECHECK_SFACE_MODEL="$PWD/models/face_recognition_sface_2021dec.onnx"

uv run uvicorn facecheck.main:app --host 127.0.0.1 --port 8000
```

如果使用的是 `venv + pip`，先激活 `.venv`，再执行：

```bash
python -m uvicorn facecheck.main:app --host 127.0.0.1 --port 8000
```

密钥配置和启动命令必须在同一个终端会话执行。`FACECHECK_SM4_KEY` 编码后必须正好是 16 字节，MAC 密钥不能为空。不要把真实密钥写入代码、README、`.env` 或 Git。

## Windows 启动

Windows 不使用 `scripts/start_local.sh`。可以使用上面的 uv 或 `venv + pip` 安装方式，然后用 PowerShell 手动启动。

### PowerShell

在项目根目录执行：

```powershell
cd C:\path\to\facecheck
uv sync --python python

$env:FACECHECK_SM4_KEY = "0123456789abcdef"
$env:FACECHECK_MAC_KEY = "local-demo-mac-key"
$env:FACECHECK_YUNET_MODEL = "$PWD\models\face_detection_yunet_2023mar.onnx"
$env:FACECHECK_RECOGNITION_MODEL = "arcface"
$env:FACECHECK_ARCFACE_MODEL = "$PWD\models\arcface_w600k_r50.onnx"
$env:FACECHECK_SFACE_MODEL = "$PWD\models\face_recognition_sface_2021dec.onnx"

uv run uvicorn facecheck.main:app --host 127.0.0.1 --port 8000
```

如果没有使用 uv，激活 `.venv` 后执行：

```powershell
python -m uvicorn facecheck.main:app --host 127.0.0.1 --port 8000
```

如果使用其他端口：

```powershell
uv run uvicorn facecheck.main:app --host 127.0.0.1 --port 8001
```

访问：

```text
http://127.0.0.1:8000/
```

关闭服务：在运行 Uvicorn 的窗口按 `Ctrl+C`。关闭 PowerShell 后，通过 `$env:` 设置的配置不会保留。

### Windows CMD

```bat
cd /d C:\path\to\facecheck
uv sync --python python

set FACECHECK_SM4_KEY=0123456789abcdef
set FACECHECK_MAC_KEY=local-demo-mac-key
set FACECHECK_YUNET_MODEL=%CD%\models\face_detection_yunet_2023mar.onnx
set FACECHECK_RECOGNITION_MODEL=arcface
set FACECHECK_ARCFACE_MODEL=%CD%\models\arcface_w600k_r50.onnx
set FACECHECK_SFACE_MODEL=%CD%\models\face_recognition_sface_2021dec.onnx

uv run uvicorn facecheck.main:app --host 127.0.0.1 --port 8000
```

## 访问地址

服务启动后：

- 控制台：http://127.0.0.1:8000/
- 健康检查：http://127.0.0.1:8000/health
- Swagger API：http://127.0.0.1:8000/docs

`/health` 只能证明 HTTP 服务正在运行，不能证明加密密钥已经配置。如果上传或检索提示 `encryption configuration is unavailable`，请停止旧的 Uvicorn 进程，在同一个终端重新设置密钥后启动。

## 本地演示流程

1. 准备明确授权的测试视频和查询图片；
2. 启动服务并打开控制台；
3. 在“视频管理”上传视频；
4. 等待任务状态变为 `completed`；
5. 进入“人脸检索”，在任务列表中选择已完成任务；
6. 上传授权查询图片并开始检索；
7. 查看候选编号、帧号、时间戳和相似度；
8. 在“安全中心”查看加密和完整性说明；
9. 在“历史实验”查看 ArcFace/SFace 实际结果，在“国密与性能记录”查看各份 benchmark；
10. 在“审计日志”查看操作记录。

相似度是数学相似度，不是身份确认概率。不要使用未获授权的人脸图片、视频或身份信息。

## 课程展示与批量任务

新增“批量数据集实验”“数据集 1:N 检索”“历史实验”“国密与性能记录”“课题指标对照”，保留视频管理、单视频检索与安全中心。页面从后端读取既有 JSON/CSV；正式 test、selection 与探索性结果分别标记。详细接口、任务机制、定向验证与课堂流程见 [课程平台说明](docs/course_platform_20261101.md)。

- 批量任务默认选择已登记的 YTF 20 身份集；手动资源预检后再点击启动。现有 Runner 同时运行 `plaintext` 与 `encrypted_restore`。
- 完整 502 身份 selection 也可显式选择，可能耗时数小时；不开放正式 test 重跑、任意路径或任意命令。
- 批量实验与数据集实时检索共用一个 SQLite 任务槽，独立进程运行。结果写入 `data/platform_runs/<随机任务ID>/`（自定义 storage 时在其同级目录），不会覆盖历史输出。
- 取消在媒体边界/评分完成后生效；退出前保持资源槽。进程异常退出标记 `failed / incomplete`，不自动重跑或续跑。
- 视频详情支持“恢复并校验完整性”和“校验并下载恢复视频”；审计接口提供全链校验，CSV/JSON/Markdown导出附带校验摘要。
- 本地审计链不能证明尾部未截断或数据库整体未回滚；日志完整率和报告生成成功率仍未测。
- 当前为恢复后明文检索，没有密文域 ANN。正式 test Top-1/TAR、SM4 吞吐未达目标；未测试项明确列示。批量 worker 当前使用 macOS/Linux 的 POSIX 运行锁。

## API 快速示例

上传视频：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/videos \
  -F "file=@authorized-sample.mp4"
```

查询任务：

```bash
curl http://127.0.0.1:8000/api/v1/tasks/<task_id>
curl "http://127.0.0.1:8000/api/v1/tasks?limit=20&offset=0"
```

主要接口：

```text
GET  /health
POST /api/v1/videos
GET  /api/v1/videos
GET  /api/v1/tasks
GET  /api/v1/tasks/{task_id}
POST /api/v1/videos/{task_id}/search
GET  /api/v1/security/summary
GET  /api/v1/audit
```

## 测试与检查

使用 uv：

```bash
uv run pytest -vv
uv run python -m compileall -q src tests
git diff --check
uv lock --check
```

使用 `venv + pip`：

Windows PowerShell：

```powershell
python -m pytest -vv
python -m compileall -q src tests
```

macOS：

```bash
python -m pytest -vv
python -m compileall -q src tests
```

`git diff --check` 和 `git status --short` 不依赖 uv。没有 uv 时无法执行 `uv lock --check`，但仍应提交并同步 `pyproject.toml` 和 `uv.lock`，避免不同成员悄悄修改依赖范围。

## 离线真实视频检索实验

`facecheck.experiment` 是独立离线实验模块，不改变现有 API/SQLite 检索语义。查询记录可使用 `media_type: "image"` 或 `"video"`；gallery 记录必须是视频。每个 query 会检索清单中的全部 gallery 视频。身份 ID 应使用去标识编号，`label` 应是与身份一致的匿名标签。gallery/query 不得共享 `source_group_id`，因此同一视频或相邻采集片段应作为一个来源组整体划分。

清单示例（路径相对于 JSON 文件）：

```json
{
  "records": [
    {"identity_id":"id_001", "video_id":"gallery_001", "video_path":"media/gallery_001.mp4", "split":"gallery", "source_group_id":"session_a", "label":"subject_001", "media_type":"video"},
    {"identity_id":"id_001", "video_id":"query_001", "video_path":"media/query_001.png", "split":"query", "source_group_id":"session_b", "label":"subject_001", "media_type":"image"},
    {"identity_id":"id_002", "video_id":"gallery_002", "video_path":"media/gallery_002.mp4", "split":"gallery", "source_group_id":"session_c", "label":"subject_002", "media_type":"video"}
  ]
}
```

所有记录都必须指向真实存在的文件；每个身份的 label 必须一致。为防止来源泄漏，query 与 gallery 的 `source_group_id` 不可相同。查询视频按同一抽帧间隔提取多个人脸作为查询特征；查询图片只使用 YuNet 返回的首个人脸，和当前业务接口保持一致。

明文实验：

```bash
.venv/bin/python -m facecheck.experiment experiment.json results/plaintext --mode plaintext \
  --yunet-model models/face_detection_yunet_2023mar.onnx \
  --sface-model models/face_recognition_sface_2021dec.onnx \
  --frame-interval 1.0 --top-k 10
```

加密存储、解密并验证后检索：

```bash
FACECHECK_SM4_KEY='16-byte-key-here' FACECHECK_MAC_KEY='separate-mac-key' \
.venv/bin/python -m facecheck.experiment experiment.json results/encrypted --mode encrypted_restore \
  --yunet-model models/face_detection_yunet_2023mar.onnx \
  --sface-model models/face_recognition_sface_2021dec.onnx \
  --frame-interval 1.0 --top-k 10
```

同批次成对比较可在 Python 中调用 `run_comparison(manifest, output_dir, detection=..., embedding=..., encryption_key=..., mac_key=...)`；两种模式共用 manifest 和所有检索参数。输出目录必须为空，避免覆盖旧实验结果。每种模式导出 `config.json`、`query_results.csv`、`candidate_results.csv`、`identity_results.csv`、`genuine_impostor_pairs.csv`、`threshold_selection_pairs.csv`、`metrics.json`、`timing.csv` 和 `report.txt`。明文模式的恢复 SHA-256 留空；加密恢复模式会比对并记录原视频和恢复文件 SHA-256。

指标按 query 计分母：Top-1 命中率和 Recall@K 是至少一个正确身份出现在相应身份排名范围内的 query 数 / query 总数；Precision@K 是每个 query 的前 K 个身份槽位中正确身份数 / K，再对 query 平均，不足 K 个候选的槽位按未命中计；未检出率包含查询无脸或没有任何候选；错误匹配率是 Top-1 存在但身份错误的 query 数 / query 总数。帧级候选保留原始结果，身份级排名先按身份/视频取最高相似帧，再跨结果排序，避免同视频连续帧占满身份排名。

TAR@FAR=1e-3 使用 genuine/impostor 配对分数；阈值应从独立的 threshold-selection 配对集合选定，再在 test 配对集合评估。可把阈值选择配对作为 `threshold_selection={"pairs": [...]}` 传给 `run_experiment`，每条配对含 `pair_type`（`genuine`/`impostor`）及 `similarity`。样本不足 1000 个 impostor 配对时指标明确返回 `not_estimable`。离线入口暂不提供配对清单 CLI 参数，需从 Python API 提供。加密模式明确是“恢复并验证明文后检索”，不是密文域检索。

### 指标改进进展

历史 SFace 正式基线使用人脸边界几何的安全裁剪，相关实现继续保留。ArcFace YTF 评估沿用已验证的 selection 实现与严格 bbox 校验；不因 test 结果改动检测、采样、Query/Gallery 规则或相似度。Web/API 保持其原有检测与几何处理规则，仅切换为匹配的 ArcFace 对齐和 embedding。

ArcFace 正式 test 的独立入口为 `python scripts/run_ytf_arcface_test.py`，先用 `--preflight-only` 检查资源和冻结配置。入口只读取既有 502 身份 test split，运行 plaintext 与 encrypted_restore，不重选阈值，也不重跑 selection；输出和 `.execution` 目录已存在时拒绝再次执行。正式完成状态必须以 `.execution/run_status.json` 为准。课程数据报告与原始结果保存在 `data/ytf/experiments/ytf_arcface_test_20261101/`，不得覆盖历史结果。

在固定 20 身份诊断集、YuNet 阈值 0.85、1 fps 抽帧条件下，裁剪后 Query 20/20 可提取特征，Gallery/Query 特征失败记录为 0，明文与 `encrypted_restore` 的 Top-1 均为 90%（18/20），Recall@5 均为 100%（20/20），两种模式相似度最大差异为 0。该结果用于验证改进方向，不能替代 10,000 人综合测试，也不能宣称已经达到 95% Top-1 和 TAR@FAR=1e-3 不低于 90% 的项目目标。

最新完整规模复测为 1004 身份（`data/ytf/experiments/ytf_full_run_02`）：边界裁剪后明文/恢复检索 Top-1 为 77.988%（783/1004），较 `run_01` 的 77.888% 提升 0.100 个百分点；TAR 为 78.6853%（395/502），提升 0.199 个百分点。测试集实测 FAR 为 0.001005956，仍略高于 1e-3，因此该工作点仍未达标；Top-1 和 TAR 也仍低于 95% 和 90% 目标。加密恢复与明文的 504008 条配对相似度最大差异为 0，证明恢复链路一致，但仍不是密文域检索。后续应继续使用独立测试集推进特征质量、批量加密、密文域索引和密钥轮换，并分别记录吞吐、时延、准确率和审计完整率；不得用诊断集结果替换正式指标。

完整的指标逐项核对、当前证据、状态和差距见 [docs/indicator_gap_analysis.md](docs/indicator_gap_analysis.md)。优化阶段、外部数据/设备依赖见 [优化路线图](docs/optimization_roadmap_20261101.md)；密文检索威胁模型和架构决策见 [ADR 0001](docs/adr/0001-private-search-architecture.md)。可用 `scripts/evaluate_indicators.py` 对正式指标、benchmark、manifest 和云端验证 JSON 生成带源文件 SHA-256 的新核对结果；脚本不运行实验并拒绝覆盖输出文件。

阶段二新增的 v2 密钥信封原型及边界见 [ADR 0002](docs/adr/0002-video-key-envelope-and-rotation.md)。另有单机 SQLite 轮换队列、数据库内对象引用/manifest 锚点事务切换、缓存清理与撤销门槛原型；最新小样本记录见 `results/phase2_rotation_control_smoke_20261102_v5.json`，指标核对见 `results/facecheck_review_20261102_phase2_v8.json`。v2 仍是显式调用，不改变当前服务写入 v1 格式的默认流程。没有生产 KMS/HSM、分布式队列、完整备份盘点或真实全局撤销证明，旧密钥失效率 100% 尚未证明。

国密后端默认配置为 `FACECHECK_CRYPTO_BACKEND=auto`：优先加载 GmSSL 3 C 原生后端，动态库不可用时回退到 `PythonBackend`。设置为 `native` 时要求 GmSSL C 库可加载；设置为 `python` 可强制使用兼容实现。原生后端中的 SM4 默认由 GmSSL 执行；`FACECHECK_SM4_NATIVE_IMPL=openssl` 可选用 OpenSSL EVP SM4-CBC，`gmssl` 或未设置时使用 GmSSL。Python 兼容后端使用 gmssl Python 实现 SM2，并使用 cryptography 的 SM3/SM4。

本机实测为 macOS 26.6.2 arm64、Python 3.9.6、gmssl Python 包 3.2.2、GmSSL C 动态库 3.2.0。最新 `final` benchmark 的 SM2 签名/验签中位数为 2958/933 次/s，达到本机吞吐目标；SM4 完整 SM4-CBC+HMAC-SM3 为69.34/84.00 MB/s，未达500 MB/s。历史 `optimized` 记录为146.30/175.63 MB/s，构建/子实现元数据不足且仍低于目标。两帧1080p样本、4线程路径为18.02 ms，不能代替持续流测试。所有历史记录均保留；性能口径、交叉兼容性和记录差异见 [docs/crypto_backend_performance_20261031.md](docs/crypto_backend_performance_20261031.md)。

默认测试使用合成数据和临时目录，不依赖外网、模型或真实人脸素材。真实模型集成测试需要显式提供本地模型和授权素材。

## 当前安全实现

- SM4-CBC 分块加密；
- 每个分片使用随机 IV；
- 按 `key_version` 使用 SM3 KDF 派生独立 SM4 和 HMAC-SM3 子密钥；
- HMAC-SM3 认证数据绑定视频、分片序号和密钥版本；
- 认证通过后才进行分片解密；
- 恢复临时文件通过大小和 SHA-256 校验后才发布；
- 错误密钥、篡改密文、缺失分片和异常 manifest 会被拒绝；
- embedding 默认在可信进程内处理，不写入 API 响应或日志；cloud 启用受保护缓存时以 SM4-CBC + HMAC-SM3 认证密文落盘；
- 人脸检测框和关键点默认进行图像边界裁剪，非相交或非法几何仍拒绝；
- 视频分片加解密支持有界并行，服务默认使用 4 个加密工作线程，可通过 `FACECHECK_ENCRYPT_WORKERS` 调整；
- 提供 `rotate_encrypted_video_key` 单视频迁移函数：新密文验证成功前不修改旧存储；
- 新增 opt-in `store_enveloped_video` / `restore_enveloped_video` v2 原型：逐视频随机生成 SM4/MAC 主密钥材料，以调用方注入的 SM2 公钥封装，manifest 用独立派生的 HMAC-SM3 认证；恢复必须提供 KeyProvider 和可信的预期视频 ID；
- 新增 `rewrap_enveloped_video` 和 file-backed `rotate_enveloped_batch` 原型：生成独立目标副本、完整恢复校验、保留源对象，并通过权限受限的原子 JSON journal 支持续跑；它不切换数据库引用、不清理缓存/备份，也不撤销旧私钥；
- 本机演示默认绑定 `127.0.0.1`；独立 cloud 模式要求 HTTPS、私有二进制密钥和访问凭据；
- cloud 模式支持 admin/operator/auditor 组织级角色、可撤销会话/令牌、同源写请求检查；
- cloud 特征缓存使用专用派生上下文加密和认证，命中前校验源密文/manifest、模型及处理代码指纹；
- 上传时将 manifest 摘要锚定到任务数据库，cloud 恢复校验该锚点；这依赖应用和数据库可信，不是独立 manifest 签名；
- cloud HTTPS、本机并发上传及特征缓存已有固定样本验证，具体范围和限制见 [docs/cloud_service_20261101.md](docs/cloud_service_20261101.md)。

## 待实现功能和已知限制

以下能力仍未实现或缺少目标规模证据，不应描述为已达标：

- 多租户隔离和逐视频所有者权限；
- 生产级 KMS/可信 SM2 公钥目录、v2 与 Web/cloud 上传流程集成、服务级批量轮换任务、数据库引用原子切换、缓存/备份迁移及全局旧密钥撤销；
- v1 manifest 的独立认证及两种格式的外部防回滚（cloud v1 当前使用数据库摘要锚点；v2 原型有逐视频 HMAC-SM3，但没有独立新鲜度锚点）；
- 密文域人脸检索、可撤销且跨版本不可关联的模板；
- 边缘设备推理与边缘—云协同协议；
- 多节点可恢复任务队列、对象存储和规模化部署；
- 审计日志外部不可变锚定、完整率统计和长期报告成功率测量；
- ≥1万人/10万段/100万帧数据集、三类终端、持续10路推理、日吞吐及目标云端性能证据。

当前检索是“恢复并验证明文视频后，在内存中执行人脸检测和相似度计算”，不是密文域检索。平台批量实验使用持久化后台任务；单视频 HTTP 上传/检索仍在请求生命周期内完成计算，并非可断点恢复的离线队列任务。未在缺口清单标为 PASS 的小样本均不能代替目标环境和持续负载验收。

## 数据和密钥注意事项

本地运行可能产生以下文件：

```text
data/facecheck.sqlite3
data/storage/
```

这些文件包含任务记录和密文数据，已被 `.gitignore` 忽略。不要将数据库、密钥、原始视频、查询图片、模型文件、embedding 或恢复出的临时明文提交到 Git。

## Git 协作建议

建议提交到 Git 的内容：

- `src/`、`tests/`、`frontend/`、`scripts/`；
- `README.md`、`DEVELOPMENT_HISTORY.md`、`pyproject.toml`、`uv.lock`；
- `.env.example`；
- 不包含真实数据的配置和测试代码。

不要提交：

- `.venv/`、`__pycache__/`；
- `data/`、SQLite 数据库和加密存储目录；
- 未经核验的额外模型文件；
- 授权测试视频、查询图片和恢复出的明文；
- `.env`、真实密钥、证书和临时文件。

首次协作准备可以使用：

```bash
git clone <仓库地址>
cd facecheck
git status
```

之后每位成员选择 uv 或 `venv + pip` 完成环境安装，再按照本 README 配置自己的本地模型、数据目录和运行时密钥。不要把本机绝对路径写入代码或提交。

## 开发历史

阶段性实现、性能基线、兼容性验证和历史测试记录见：

- [DEVELOPMENT_HISTORY.md](DEVELOPMENT_HISTORY.md)
