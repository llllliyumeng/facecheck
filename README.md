# Facecheck

基于 SM2/SM4 的云端人像视频存储、传输与检索隐私保护系统课程实践项目。

当前已完成一个可本地演示的 MVP：项目初始化、SM2/SM3/SM4 基础验证、视频读取/抽帧/分片/加密存储、SQLite 任务服务、YuNet 检测、SFace 特征提取，以及针对已加密视频的内存相似度检索。当前仍是本机可信测试原型，不是生产云服务。

## 最快启动（本地演示）

以下命令适用于 macOS，必须在项目根目录执行：

```bash
cd /Users/mac/Desktop/nuist/facecheck

# 第一次运行时创建虚拟环境并安装依赖
uv sync --python python3

# 配置运行时密钥。SM4 密钥必须是 16 个 UTF-8 字节；
# MAC 密钥应与 SM4 密钥不同，下面仅用于本机课程演示。
export FACECHECK_SM4_KEY='0123456789abcdef'
export FACECHECK_MAC_KEY='local-demo-mac-key'

# 配置本地模型路径（模型文件不能提交到 Git）
export FACECHECK_YUNET_MODEL="$PWD/models/face_detection_yunet_2023mar.onnx"
export FACECHECK_SFACE_MODEL="$PWD/models/face_recognition_sface_2021dec.onnx"

# 只绑定本机，启动后端和中文演示页面
uv run uvicorn facecheck.main:app --host 127.0.0.1 --port 8000
```

启动成功后，在浏览器打开：

- 中文演示页面：http://127.0.0.1:8000/
- 健康检查：http://127.0.0.1:8000/health
- Swagger API 文档：http://127.0.0.1:8000/docs

看到 `{"status":"ok"}` 表示后端已经启动。停止服务请回到运行 Uvicorn 的终端按 `Ctrl+C`。

### 页面演示步骤

1. 准备明确授权的测试视频。
2. 打开中文演示页面，在“上传授权视频”区域选择视频并点击“上传并处理”。
3. 等待任务显示为 `completed`。当前上传和加密是同步处理，视频较大时需要等待请求完成。
4. 在“授权图片检索”区域选择已完成的视频任务和授权查询图片。
5. 点击“开始检索”，查看候选编号、帧号、时间戳和相似度。
6. 相似度是向量之间的数学相似度，不是身份确认概率。

如果只想验证服务是否可用，可以另开一个终端执行：

```bash
curl http://127.0.0.1:8000/health
```

如果终端提示 `uv: command not found`，先执行：

```bash
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

退出当前终端后，若希望永久生效，再将这行加入 `~/.zshrc`。

### 使用自动启动脚本

项目也提供了 macOS/Linux Shell 启动脚本，可以自动检查虚拟环境和两个模型文件：

```bash
cd /Users/mac/Desktop/nuist/facecheck
chmod +x scripts/start_local.sh
./scripts/start_local.sh
```

脚本默认使用：

- 地址：`127.0.0.1`
- 端口：`8000`
- 数据目录：`data/`
- YuNet：`models/face_detection_yunet_2023mar.onnx`
- SFace：`models/face_recognition_sface_2021dec.onnx`

如果端口被占用，可以临时指定其他端口：

```bash
PORT=8001 ./scripts/start_local.sh
```

如果 8000 已经运行着 Facecheck，脚本会识别已有服务并提示访问地址，不会重复启动。若要重启服务，请在原运行窗口按 `Ctrl+C`；也可以直接使用其他端口。

如果修改了模型路径、密钥或启动脚本，必须先停止旧服务，再重新运行脚本；已经运行的 Uvicorn 进程不会自动继承新终端中的环境变量。

如果模型放在其他位置，可以指定模型路径：

```bash
FACECHECK_YUNET_MODEL="$HOME/models/face_detection_yunet_2023mar.onnx" \
FACECHECK_SFACE_MODEL="$HOME/models/face_recognition_sface_2021dec.onnx" \
./scripts/start_local.sh
```

脚本不会下载模型，也不会把密钥写入文件。未设置密钥时，它会使用仅用于课程本机演示的临时默认值，并在终端显示提示；生产环境必须通过受控运行时配置注入密钥。

## 环境要求

- macOS（Apple Silicon 可用）
- Python 3.9 或更高版本
- [uv](https://docs.astral.sh/uv/)

检查环境：

```bash
python3 --version
uv --version
```

如果终端提示 `zsh: command not found: uv`，说明 uv 的安装目录没有加入当前 zsh 的 `PATH`。执行下面的命令写入用户级 zsh 配置，然后重新加载配置：

```bash
printf '\nexport PATH="$HOME/.local/bin:$PATH"\n' >> ~/.zshrc
source ~/.zshrc
uv --version
```

也可以先用绝对路径确认 uv 已安装：

```bash
"$HOME/.local/bin/uv" --version
```

## 创建环境并安装依赖

在项目根目录执行：

```bash
uv sync --python python3
```

`uv sync` 会创建项目虚拟环境 `.venv`、解析依赖并生成 `uv.lock`。如果希望使用 uv 自动管理的其他 Python 版本，可以省略 `--python python3`。后续命令使用 `uv run`，无需手动激活虚拟环境。

## 启动开发服务器

```bash
uv run uvicorn facecheck.main:app --host 127.0.0.1 --port 8000
```

启动后访问：

- 健康检查：http://127.0.0.1:8000/health
- OpenAPI 文档：http://127.0.0.1:8000/docs
- 中文演示界面：http://127.0.0.1:8000/

阶段四服务默认只绑定本机开发地址，使用环境变量配置密钥和数据目录：

```bash
export FACECHECK_SM4_KEY='在本机安全注入的16字节测试密钥'
export FACECHECK_MAC_KEY='在本机安全注入的独立MAC密钥'
export FACECHECK_DATA_ROOT="$PWD/data"
export FACECHECK_YUNET_MODEL="$PWD/models/face_detection_yunet_2023mar.onnx"
export FACECHECK_SFACE_MODEL="$PWD/models/face_recognition_sface_2021dec.onnx"
uv run uvicorn facecheck.main:app --reload
```

`FACECHECK_SM4_KEY` 必须编码后为 16 字节；MAC 密钥不能为空。密钥不会写入 SQLite、清单、日志或 API 响应。未配置密钥时，上传任务会记录为 `failed`，不会通过硬编码密钥绕过配置。

也可以复制 `.env.example` 作为配置参考；项目不会自动读取 `.env`，请在 shell 或受控启动器中注入环境变量。

阶段四默认数据位置：

```text
data/facecheck.sqlite3
data/storage/<task_id>/
```

## 阶段四 API

上传授权测试视频：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/videos \
  -F "file=@authorized-sample.mp4"
```

响应只包含任务标识、状态、文件大小、非敏感元数据和受控 `storage_id`，不返回本机绝对路径或密钥：

```json
{
  "task_id": "generated-task-id",
  "original_filename": "authorized-sample.mp4",
  "source_size": 12345,
  "status": "completed",
  "storage_id": "generated-task-id",
  "error_message": null
}
```

查询任务：

```bash
curl http://127.0.0.1:8000/api/v1/tasks/<task_id>
curl "http://127.0.0.1:8000/api/v1/tasks?limit=20&offset=0"
```

上传成功表示请求已被接受并创建任务；只有 `status` 为 `completed` 才表示视频已完成本地加密存储。当前实现同步处理任务，状态仍显式经历 `pending`、`processing`、`completed` 或 `failed`，以便后续替换为后台任务队列。

健康检查预期返回：

```json
{"status":"ok"}
```

## MVP 演示流程

1. 从 OpenCV Zoo 获取 YuNet 和 SFace ONNX 模型，放入 `models/`。YuNet 目录文件标注 MIT License，SFace 目录文件标注 Apache 2.0 License；不要提交模型文件。
2. 准备明确授权的本地测试视频和查询图片。
3. 注入密钥和模型路径，启动后端：

```bash
export FACECHECK_SM4_KEY='本地运行时注入的16字节密钥'
export FACECHECK_MAC_KEY='本地运行时注入的独立MAC密钥'
export FACECHECK_YUNET_MODEL="$PWD/models/face_detection_yunet_2023mar.onnx"
export FACECHECK_SFACE_MODEL="$PWD/models/face_recognition_sface_2021dec.onnx"
uv run uvicorn facecheck.main:app --host 127.0.0.1 --port 8000
```

4. 打开 http://127.0.0.1:8000/，上传授权视频并等待状态为 `completed`。
5. 在“授权图片检索”区域选择已完成任务和授权查询图片，点击“开始检索”。
6. 页面显示候选编号、帧号、时间戳和数学相似度。相似度不是身份确认概率。

也可以使用 API：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/videos \
  -F "file=@authorized-sample.mp4"
curl http://127.0.0.1:8000/api/v1/videos
curl -X POST "http://127.0.0.1:8000/api/v1/videos/<task_id>/search?top_k=10" \
  -F "query_image=@authorized-query.png"
```

检索前会通过现有 `restore_encrypted_video` 完成分片集合检查、MAC 验证和最终 SHA-256 校验；验证失败不会进入 YuNet 或 SFace。查询图片只在请求内存中解码，不写入数据库、日志或响应。

上传和检索响应会返回非敏感处理统计，例如上传接收、视频读取、加密存储、恢复校验、视频解码、抽帧、人脸检测、SFace 特征提取和排序耗时。实际基准应以本机运行结果为准，不把示例耗时当作性能承诺。

当前实测授权视频对照（4.86 MB、960×720、30 FPS、19 个抽帧、12 个特征）：

- 明文检索约 0.94 秒；
- 加密视频检索约 56.40 秒，其中恢复和分片完整性校验约 55.22 秒；
- 上传加密存储约 57.38 秒。

这表明原始实现的主要瓶颈是 `gmssl` 纯 Python 实现执行 SM4-CBC 和 HMAC-SM3 的分片加密/恢复，不是 YuNet、SFace 或相似度排序。项目现在使用独立的 `crypto_backend.py` 适配层调用 `cryptography` 的 SM4、SM3 和 HMAC API；SM4-CBC、PKCS#7 填充、HMAC-SM3、MAC 校验顺序和现有 envelope 格式均保持不变。

原始 `gmssl` 实现和新后端均使用同一授权视频、1 MiB 分片、临时目录和临时测试密钥。新后端使用 `cryptography==42.0.8`，其 wheel 搭载的 OpenSSL 版本为 `3.2.2`。三次独立运行的结果如下；每次恢复后的 SHA-256 都与源文件一致：

| 实现与阶段 | 第一次 | 第二次 | 第三次 | 中位数 |
|---|---:|---:|---:|---:|
| 新后端加密总耗时 | 0.273 秒 | 0.265 秒 | 0.266 秒 | 0.266 秒 |
| 新后端加密 SM4-CBC | 0.170 秒 | 0.167 秒 | 0.168 秒 | 0.168 秒 |
| 新后端加密 HMAC-SM3 | 0.086 秒 | 0.086 秒 | 0.086 秒 | 0.086 秒 |
| 新后端恢复总耗时 | 0.266 秒 | 0.266 秒 | 0.265 秒 | 0.266 秒 |
| 新后端恢复 SM4-CBC | 0.168 秒 | 0.169 秒 | 0.168 秒 | 0.168 秒 |
| 新后端恢复 HMAC-SM3 | 0.085 秒 | 0.085 秒 | 0.086 秒 | 0.085 秒 |
| 新后端恢复 SHA-256 | 0.005 秒 | 0.005 秒 | 0.005 秒 | 0.005 秒 |

此前 `gmssl` 中位数约为加密 57.5 秒、恢复 55.3 秒；在相同输入和测试条件下，本次观测的新后端约快 200 倍以上。该结果是本机三次观测，不是吞吐量或延迟承诺。适配层测试了固定 IV 的密文互操作、SM3/HMAC 输出、错误密钥、篡改数据、旧 `gmssl` envelope 读取和新密文由 `gmssl` 解密；仍应在目标部署环境重新基准。

候选调研结论：

- [`cryptography` 对称加密文档](https://cryptography.io/en/latest/hazmat/primitives/symmetric-encryption/)和 API 提供 SM4-CBC、PKCS#7 所需接口，当前锁定版本实际暴露 `algorithms.SM4` 与 `hashes.SM3`。
- [PyCryptodome 官方 API](https://www.pycryptodome.org/src/api)当前未提供本项目所需的 SM4/SM3 组合，因此未接入。
- macOS 系统 LibreSSL 3.3.6 的本机命令行未列出 SM4/SM3；项目使用 `cryptography` wheel 自带的 OpenSSL，不依赖系统 LibreSSL。
- 这些实现仍只提供分组加密和哈希/MAC原语；项目继续保留独立 HMAC-SM3、认证先于解密和恢复后的 SHA-256 校验。

## 审计和安全状态

以下接口读取真实 SQLite 审计记录和任务元数据，不返回密钥、embedding、原始图片或绝对路径：

```bash
curl "http://127.0.0.1:8000/api/v1/audit?limit=50&offset=0"
curl http://127.0.0.1:8000/api/v1/security/summary
```

审计记录包括任务创建、处理中、完成/失败、检索成功和检索失败。视频列表和任务响应会展示 SM4-CBC、HMAC-SM3、分片数量、密钥版本、完整性状态和非敏感耗时。

安全状态如实说明：

- 已实现：SM4-CBC 加密、分片 HMAC-SM3、SHA-256 恢复一致性检查、错误密钥/篡改拒绝、SQLite 操作审计、审计查询。
- 部分实现：密钥版本字段和运行时密钥配置；尚未实现自动轮换、撤销和历史数据迁移。
- 尚未实现：身份认证、用户权限、多租户隔离、生产级密钥托管。
- manifest 有结构和一致性校验，但没有独立 HMAC/SM2 签名，因此不提供清单来源认证或防回滚。

## 运行测试

```bash
uv run pytest
```

阶段一预期健康检查测试通过。阶段二的密码学测试还可以单独运行：

```bash
uv run pytest tests/test_crypto.py
uv run pytest tests/test_video_pipeline.py
uv run python scripts/verify_crypto_dependencies.py
```

当前测试覆盖：

- SM3 `abc` 标准摘要向量
- SM4 标准单块向量
- SM4-CBC 加密解密、随机 IV、SM3 HMAC 风格的 Encrypt-then-MAC
- 错误加密密钥、错误 MAC 密钥和篡改密文拒绝
- SM2 密钥加密/解密
- SM2 签名、验签和篡改消息拒绝
- OpenCV 视频元数据读取和按时间间隔抽帧
- 文件分片边界、空文件和最后不足标准大小的分片
- 分片加密恢复后的 SHA-256 一致性
- 错误密钥、密文篡改、分片缺失/调换、额外分片和元数据篡改拒绝
- YuNet 人脸检测的独立模块测试（使用注入的假检测器，不伪造真实模型效果）

## 目录结构

```text
facecheck/
├── src/
│   └── facecheck/
│       ├── __init__.py
│       ├── chunks.py
│       ├── crypto.py
│       ├── api.py
│       ├── database.py
│       ├── face_detection.py
│       ├── face_embedding.py
│       ├── face_search.py
│       ├── main.py
│       ├── service.py
│       ├── storage.py
│       └── video.py
├── frontend/
│   └── index.html
├── tests/
│   ├── test_api.py
│   ├── test_crypto.py
│   ├── test_health.py
│   ├── test_video_pipeline.py
│   ├── test_face_detection.py
│   ├── test_face_embedding.py
│   ├── test_face_search.py
│   └── integration/
├── scripts/
│   └── verify_crypto_dependencies.py
├── .gitignore
├── pyproject.toml
├── README.md
└── uv.lock                 # uv sync 后生成
```

## 阶段二密码学说明

依赖为 `gmssl==3.2.2`。这是一个可在当前 Apple Silicon/Python 环境安装的纯 Python 国密算法库，提供：

- `CryptSM2`：SM2 公钥加密和签名/验签
- `sm3_hash`：SM3 摘要
- `CryptSM4`：SM4 ECB/CBC

先运行 `scripts/verify_crypto_dependencies.py` 可以验证当前环境中的 API 是否存在。

本阶段没有把 SM4-CBC 单独当作认证加密。由于该库没有提供适用于本项目的 SM4-GCM/SM4-GCM-SIV API，当前实验使用：

1. SM4-CBC 加密
2. 每次随机生成 16 字节 IV
3. 使用独立 MAC 密钥计算 HMAC-SM3
4. Encrypt-then-MAC，解密前先恒时验证认证标签
5. 将加密密钥的短标识绑定到认证数据，错误加密密钥会在解密前失败

这是一种经过验证的组合式替代方案，但不是 AEAD；后续若需要协议级生产部署，应优先选用经过审计并明确支持 AEAD 的实现。

阶段三通过 `associated_data` 扩展认证上下文。每个视频分片的上下文包含视频标识、分片序号和密钥版本，且会在 MAC 验证前参与认证。这样分片不能跨视频、跨序号或跨密钥版本复用。

## 阶段三本地视频链路

阶段三的代码分为三个模块：

- `video.py`：OpenCV 视频信息读取和按秒数间隔抽帧；抽帧结果包含帧序号、时间戳和图像对象，不负责加密。
- `chunks.py`：按字节大小迭代读取分片，保留分片序号和实际大小。
- `storage.py`：生成每个视频独立目录、写入密文分片和非敏感 JSON 清单，并按清单恢复文件。

示例：

```python
from pathlib import Path

from facecheck.storage import restore_encrypted_video, store_encrypted_video
from facecheck.video import extract_frames, read_video_info

source = Path("authorized-sample.mp4")
info = read_video_info(source)
frames = extract_frames(source, interval_seconds=1.0)

stored = store_encrypted_video(
    source,
    Path("storage"),
    b"0123456789abcdef",       # 仅为测试示例；生产环境不得硬编码
    b"independent-mac-key",   # 仅为测试示例；生产环境不得硬编码
    chunk_size=1024 * 1024,
    key_version="video-key-v1",
)
restore_encrypted_video(
    stored.directory,
    Path("restored.mp4"),
    b"0123456789abcdef",
    b"independent-mac-key",
)
```

清单只保存视频标识、密钥版本、分片大小/数量、源文件大小和 SHA-256，以及分片序号和文件名；不会保存加密密钥或 MAC 密钥。恢复失败时使用 `.partial` 临时文件，认证、分片集合或最终摘要检查失败会删除临时文件，不会把不完整文件作为成功结果。

## 阶段五第一部分：YuNet 人脸检测

人脸检测位于 `face_detection.py`，与 FastAPI、视频加密和数据库模块分离。当前检测输出仅包含：

- 帧编号和时间戳
- 视频宽高
- 人脸框坐标和置信度
- YuNet 可用的五个关键点

不会保存人脸裁剪图、原始帧像素或身份信息。特征提取和检索由独立模块负责，不由检测模块持久化。

OpenCV 4.x 的 `FaceDetectorYN` 已在当前环境中验证可用。获取模型时请从 [OpenCV Zoo YuNet 官方页面](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)下载对应 ONNX 文件，例如 `face_detection_yunet_2023mar.onnx`，并检查其许可和来源；该目录文件标注为 MIT License。本项目不自动从外网下载、不把模型文件提交到 Git，也不在测试中依赖模型文件。

示例配置：

```python
from pathlib import Path
from facecheck.face_detection import FaceDetectionConfig, detect_video

config = FaceDetectionConfig(
    model_path=Path("models/face_detection_yunet_2023mar.onnx"),
    score_threshold=0.9,
    frame_interval_seconds=1.0,
    max_duration_seconds=600.0,
    max_frames=5000,
    max_processing_seconds=60.0,
)
report = detect_video(Path("authorized-sample.mp4"), config)
```

模型不存在、模型无法加载、视频无法读取、抽帧间隔非法或超出大小/时间限制时会明确报错，不会生成假检测结果。当前默认测试使用注入的假检测器验证数据流和异常处理；真实 YuNet/SFace 集成测试需要用户自行准备已授权模型和素材，未包含在默认测试套件中。

## 阶段五第二部分：SFace 特征提取与内存检索

特征提取位于 `face_embedding.py`，独立于视频加密、任务 API 和数据库。它将现有 `FaceDetection` 的边框及五个关键点转换为 SFace 所需的 YuNet 记录格式，调用 OpenCV `FaceRecognizerSF` 完成人脸对齐和特征提取。

当前输出约定为：

- 默认 128 维；
- 一维 `float32` NumPy 数组；
- L2 归一化；
- 拒绝错误维度、NaN、无穷值和零向量。

`cosine_similarity` 和 `search_embeddings` 只在内存中工作。检索结果仅包含受控样本 ID 和相似度，不包含原始图像、文件路径、姓名或身份标签。当前不保存特征、不增加数据库字段或 API，也不实现身份识别、跨视频追踪或未经授权的人物搜索。

SFace 模型不会自动下载。请从 [OpenCV Zoo SFace 官方目录](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)获取模型，并确认其 Apache 2.0 许可；模型文件放入 `models/` 后仍被 `.gitignore` 忽略。真实模型测试必须使用已授权素材，并单独执行，不属于默认 pytest 套件。

特征向量属于敏感生物特征模板。本阶段不将其写入磁盘、日志、SQLite、清单或 API 响应；未来持久化前必须单独设计访问控制、密钥管理、审计、撤销和删除策略。

显式运行真实 SFace 集成测试时，需要准备官方 SFace ONNX 模型、YuNet 模型和一张有权处理的测试图片：

```bash
FACECHECK_RUN_FACE_INTEGRATION=1 \
FACECHECK_SFACE_MODEL=models/face_recognition_sface_2021dec.onnx \
FACECHECK_YUNET_MODEL=models/face_detection_yunet_2023mar.onnx \
FACECHECK_FACE_IMAGE=/absolute/path/to/authorized-test.png \
uv run pytest tests/integration/test_face_embedding_integration.py -vv
```

默认 `uv run pytest -vv` 不加载 SFace 模型；未设置显式开关时，集成测试会清楚地标记为 skipped。集成测试只验证 YuNet 能检测到人脸以及 SFace 输出为有限的 128 维 `float32`、L2 范数接近 1，不报告身份判断、准确率、召回率或相似度阈值。

### 单个授权视频的内存检索

`face_search.py` 将授权视频抽帧、YuNet 检测、SFace 特征提取和余弦相似度检索串联起来。结果只包含受控候选 ID、帧序号、时间戳和相似度，不返回特征向量、图像或文件路径。加密视频检索使用现有 `restore_encrypted_video` 完整性校验，并在唯一临时目录中恢复；成功和异常路径都会清理临时文件。

该服务仍然只适用于授权测试素材，不进行身份确认、跨视频追踪或未经授权的人物搜索。候选特征只在检索进程内存中存在；MVP 通过任务级 API 调用该流程，但不保存候选特征，也不报告准确率、召回率、吞吐量或已验证的识别阈值。

密钥用途严格区分：

- SM4 加密密钥：加密数据内容
- SM3 MAC 密钥：认证 SM4 密文的完整性和来源
- SM2 公钥加密：只用于加密小型数据密钥/密钥封装，不用于视频内容
- SM2 签名密钥：只用于数字签名和验签，不等同于密钥封装
- SM2 签名测试的随机数使用 Python `secrets`，不使用 `random` 或 `gmssl.func.random_hex`

验证依据：

- SM3 `abc` 摘要和 SM4 单块向量来自公开国密标准中广泛使用的标准向量
- SM2 API、固定测试密钥及操作方式参考 [py-gmssl 官方文档](https://github.com/py-gmssl/py-gmssl)
- SM 算法在 TLS 中的公开规范可参考 [RFC 8998](https://datatracker.ietf.org/doc/html/rfc8998)

## MVP 范围与限制

- 当前服务提供本地 HTTP API、SQLite 任务记录和操作审计；尚未实现身份认证、用户授权或多租户隔离。
- 当前仅适合本机可信测试环境，不能声称具备生产级权限控制、密钥托管或云端安全能力。启动时应明确使用 `--host 127.0.0.1`。
- 当前 MVP 提供本地授权视频的内存相似度检索，但不提供身份确认。
- 当前实现 YuNet 检测、SFace 特征和内存相似度检索，但不提供任何检测/识别准确率、性能指标或身份判断。
- 上传明文只保存在临时文件中，处理成功或失败后都会清理；SQLite 只保存任务元数据，不保存密钥。
- 当前密码学代码仅用于课程实践和测试向量验证，不应直接用于生产密钥管理。
- 当前 SM2 测试使用固定的公开示例密钥，不能用于真实数据。
- 仅使用已授权的测试数据；不要把密钥、令牌、个人数据或本地数据库提交到 Git。
- API 没有身份认证和多租户授权，必须只绑定 `127.0.0.1`，不要使用 `0.0.0.0` 暴露到网络。
- manifest 当前有结构、一致性和分片级认证关联校验，但没有独立 HMAC/SM2 签名，因此不能声称防止清单回滚或证明清单来源。
- 进程崩溃、断电或文件系统异常时可能遗留临时明文；普通删除不等于 SSD 安全擦除。
- 当前任务处理为同步执行，状态接口保留 `pending`、`processing`、`completed`、`failed`，尚未接入后台队列。
