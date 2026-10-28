# FaceGuard

云端人像视频存储、传输与检索隐私保护系统课程实践项目。

FaceGuard 是一个使用 Python、FastAPI、SQLite 和 OpenCV 构建的本地 MVP。系统面向**获得授权的测试视频和查询图片**，提供：

- 视频上传和任务状态管理；
- 视频分片加密存储；
- SM4-CBC 数据加密；
- HMAC-SM3 分片完整性认证；
- 恢复后的 SHA-256 校验；
- YuNet 人脸检测；
- SFace 特征提取；
- 已加密视频恢复并进行内存相似度检索；
- 安全摘要、审计记录和性能实验展示；
- 中文 FaceGuard Web 控制台。

当前版本是适合课程演示的本机可信原型，不是生产云服务。默认只绑定 `127.0.0.1`，不要直接暴露到公网或不受信任的局域网。

## 目录结构

```text
facecheck/
├── frontend/                 # 中文控制台页面
├── models/                   # 本地 ONNX 模型，不纳入 Git
├── scripts/                  # 本地启动脚本
├── src/facecheck/
│   ├── api.py                # FastAPI 路由和响应模型
│   ├── crypto.py             # SM4、HMAC-SM3 和加密封装
│   ├── crypto_backend.py     # cryptography 加速后端
│   ├── database.py           # SQLite 任务和审计记录
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
- 可用的 YuNet 和 SFace ONNX 模型；
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

将以下模型放入项目的 `models/` 目录：

```text
models/face_detection_yunet_2023mar.onnx
models/face_recognition_sface_2021dec.onnx
```

模型不会由项目自动下载，也不应提交到 Git。模型来源、许可证和校验值应以实际下载页面或发布包为准；课程报告中请记录下载来源和文件 SHA-256。

如果模型在其他目录，启动前设置：

```text
FACECHECK_YUNET_MODEL=/path/to/face_detection_yunet_2023mar.onnx
FACECHECK_SFACE_MODEL=/path/to/face_recognition_sface_2021dec.onnx
```

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
9. 在“实验分析”查看已记录的性能和兼容性实验；
10. 在“审计日志”查看操作记录。

相似度是数学相似度，不是身份确认概率。不要使用未获授权的人脸图片、视频或身份信息。

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

默认测试使用合成数据和临时目录，不依赖外网、模型或真实人脸素材。真实模型集成测试需要显式提供本地模型和授权素材。

## 当前安全实现

- SM4-CBC 分块加密；
- 每个分片使用随机 IV；
- HMAC-SM3 认证数据绑定视频、分片序号和密钥版本；
- 认证通过后才进行分片解密；
- 恢复临时文件通过大小和 SHA-256 校验后才发布；
- 错误密钥、篡改密文、缺失分片和异常 manifest 会被拒绝；
- embedding 仅在内存中处理，不写入 API 响应；
- 默认绑定 `127.0.0.1`；
- 密钥来自运行时环境变量，不写入数据库、日志或 manifest。

## 待实现功能和已知限制

以下功能当前未实现，不应在课程报告中描述为已完成：

- 用户身份认证和细粒度权限控制；
- 多租户隔离；
- 生产级密钥托管；
- 密钥轮换、撤销和历史数据迁移；
- manifest 独立 HMAC/SM2 签名和防回滚；
- 密文域人脸检索；
- 可撤销人脸模板和持久化特征保护；
- 后台任务队列和分布式处理；
- 审计日志导出和正式审计报告；
- 面向公网的部署、TLS、限流和高可用。

当前检索是“恢复并验证明文视频后，在内存中执行人脸检测和相似度计算”，不是密文域检索。当前上传处理虽然有明确任务状态，但实际仍是同步请求处理。

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
- `models/*.onnx`；
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
