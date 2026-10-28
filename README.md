# Facecheck

基于 SM2/SM4 的云端人像视频存储、传输与检索隐私保护系统课程实践项目。

当前已完成阶段一至阶段三：项目初始化、密码学基础能力验证，以及本地视频读取、抽帧、分片、加密存储和恢复验证。人脸处理、数据库和云端上传功能将在后续阶段单独实现和验收。

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
uv run uvicorn facecheck.main:app --reload
```

启动后访问：

- 健康检查：http://127.0.0.1:8000/health
- OpenAPI 文档：http://127.0.0.1:8000/docs

健康检查预期返回：

```json
{"status":"ok"}
```

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

## 目录结构

```text
facecheck/
├── src/
│   └── facecheck/
│       ├── __init__.py
│       ├── chunks.py
│       ├── crypto.py
│       ├── main.py
│       ├── storage.py
│       └── video.py
├── tests/
│   ├── test_crypto.py
│   ├── test_health.py
│   └── test_video_pipeline.py
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

## 阶段一范围与限制

- 当前只提供最小 HTTP 服务、健康检查和本地处理 API；尚未把视频链路接入 FastAPI。
- 当前不包含人脸识别、检索、权限控制、数据库或云端上传。
- 当前密码学代码仅用于课程实践和测试向量验证，不应直接用于生产密钥管理。
- 当前 SM2 测试使用固定的公开示例密钥，不能用于真实数据。
- 仅使用已授权的测试数据；不要把密钥、令牌、个人数据或本地数据库提交到 Git。
