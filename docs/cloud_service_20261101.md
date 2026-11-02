# FaceCheck 云端服务与在线检索改进（2026-11-01）

本轮已实现 HTTPS 云端启动、访问控制、持久化密钥与会话、受保护特征缓存及并发请求处理，并在当前 Mac 上启动真实 TLS 服务完成验证。没有重跑 YTF 正式实验，没有改写既有识别指标或国密 benchmark。已有课程报告 PDF 是本轮改进前的快照，本文件和新增验证 JSON 记录当前实现。

## 1. 已落地的功能

| 课题相关问题 | 本轮实现 | 验证边界 |
| --- | --- | --- |
| 云端传输与未授权访问 | 强制 HTTPS、组织级角色权限、Bearer 或 HttpOnly 会话、同源校验、凭据撤销 | 本机真实 TLS 验证通过；尚无远程部署结果 |
| 重启后密钥和会话不可用 | 二进制私有密钥文件、SQLite 持久化会话、独立运行目录 | 服务完整重启后会话及视频恢复通过 |
| 重复检索反复恢复和提取 Gallery 特征 | 加密持久化逐帧特征缓存、模型与源文件指纹、单视频并发合并 | 一个固定视频与 Query 冷/热缓存结果逐项一致 |
| 上传和识别阻塞异步服务 | 同步计算端点交给 FastAPI 工作线程，SQLite WAL、请求线程独立计时 | 事件循环响应定向测试通过，10 个真实 HTTPS 上传成功 |
| manifest 元数据替换 | 新上传 manifest 的 SHA-256 写入可信任务数据库；恢复时验证同一份字节快照 | metadata 篡改被拒绝；未改变 manifest 格式 |
| 特征泄露与篡改 | 复用现有 SM4-CBC + HMAC-SM3，缓存专用派生上下文，先认证后解密 | 错误密钥、缓存 MAC/密文及视频篡改均被拒绝 |

默认识别模型仍为 ArcFace，阈值仍来自完整 selection。GmSSL native 默认选择、Python 兼容后端、可选 OpenSSL SM4、SM2/SM3/SM4 原语及低复制优化保持原样。本轮没有调整采样、Query、Gallery 聚合、评分、正式实验划分或密文分块协议。

## 2. 实际测量

证据：[cloud_validation_20261101_v2.json](../results/cloud_validation_20261101_v2.json)。早期验证文件仍保留，最终代码结果以 v2 为准。

环境：Apple Silicon arm64、macOS 26.6.2、Python 3.9.6、GmSSL 3.2.0 native、SM4 实现 `gmssl`、ArcFace。脚本创建临时私有目录及 localhost 测试证书，客户端启用证书链和主机名验证，本轮协商 TLS 1.2。测试结束销毁临时凭据与数据。

| 验证项 | 实测 |
| --- | --- |
| 固定视频首次检索（含 HTTPS） | 4.940799 s |
| 同视频、同 Query 缓存命中检索（含 HTTPS） | 0.464568 s |
| 该请求加速比 | 10.635 倍 |
| 结果与分数 | 10 条返回逐项完全一致 |
| 首次视频处理 | 119 个采样帧，63 个有效特征；特征提取 3.570168 s，检测 0.526231 s |
| 命中缓存 | 查找与完整性绑定验证 0.158246 s；相似度计算 0.001228 s |
| 10 个同时 HTTPS 加密上传 | 10/10 成功；中位 0.063940 s，最小 0.057576 s，最大 0.099252 s |
| 恢复文件 SHA-256 | 与原上传一致 |
| manifest 元数据篡改 | 拒绝 |
| 视频密文与特征缓存篡改 | 拒绝 |
| 凭据撤销 | Bearer 和已登录会话同时失效 |
| 服务完整重启 | 会话与密钥继续有效，视频可恢复 |
| 原始测试素材 | SHA-256 未变 |
| 整个链路验证耗时 | 22.524638 s |

上传样本是四帧 32×24 合成视频。10/10 是一次功能并发验证，不能等同于长期 99.5% 可用性、10 路推理或日均 10 万视频。缓存比较只包含一段既有视频与一张 Query，未估计总体平均时延；冷启动仍超过 2 s。本轮未测 Top-1/TAR，不能将缓存加速作为识别准确率提升，也没有产生新的 SM4 吞吐成绩。

## 3. 在当前环境启动 HTTPS 服务

在项目根目录创建一个全新的私有实例目录：

```bash
.venv/bin/python scripts/init_cloud.py \
  --directory data/cloud-instance \
  --public-origin https://your-domain.example:8443
```

初始化器拒绝覆盖既有目录。目录权限为 0700，文件为 0600；生成随机 16 字节 SM4 主密钥、32 字节 MAC 主密钥、三类高熵访问令牌和凭据摘要。命令不输出任何密钥或令牌。`cloud.env` 含路径和配置，不含密钥本体，由 Uvicorn dotenv 解析，不需要 `source`。

准备与公开域名匹配的 TLS 证书，然后启动：

```bash
./scripts/start_cloud.sh data/cloud-instance/cloud.env \
  /path/to/fullchain.pem /path/to/privkey.pem
```

默认监听 `0.0.0.0:8443`，一个 worker，并发连接限制 64。可用 `PORT`、`FACECHECK_BIND_HOST` 修改监听地址；公开 origin 必须与浏览器地址一致。脚本强制 cloud 模式及直接 TLS，不信任代理头。若环境中残留原始 `FACECHECK_SM4_KEY` 或 `FACECHECK_MAC_KEY`，脚本拒绝启动，以免误用旧密钥。

实例使用新数据库和新存储，不自动迁移原演示数据库。保留私有实例目录并在重启时复用，视频和会话才可继续使用。备份必须同时保护密钥、数据库、密文及缓存；只备份密文不足以恢复。这里实现的是密钥持久化，没有新增主密钥轮换或 KMS 服务。

### 角色与访问

- `admin`：视频、检索、平台任务、审计及凭据撤销。
- `operator`：视频、检索、平台任务与审计；不能调用管理员凭据接口。
- `auditor`：审计、安全摘要、指标及 benchmark；不能读取或恢复视频、提交检索或启动任务。

凭据位于私有目录中的 `admin.token`、`operator.token`、`auditor.token`，由管理员通过受控方式分发。浏览器登录后使用 Secure、HttpOnly、SameSite=Strict Cookie，令牌不存入 localStorage/sessionStorage。API 客户端可以使用 Bearer。会话有效期 8 小时，Cookie 写请求要求准确匹配的同源 Origin。

管理员 API：`GET /api/v1/auth/admin/principals` 查看主体；`POST /api/v1/auth/admin/principals/{id}/revoke` 撤销凭据。撤销记录保存在数据库，立即拒绝该令牌及关联会话。私有凭据文件每次鉴权重新读取，删除或替换摘要也会使旧凭据及会话失效。不要撤销唯一管理员后丢弃私有配置；凭据恢复需由受信任的主机管理员处理。

未鉴权的本地演示兼容模式仍由 `scripts/start_local.sh` 启动，仅适合回环地址。`FACECHECK_DEPLOYMENT=cloud` 启动时检查 HTTPS origin、凭据、密钥可读性、独立密钥和实际 native 后端，检查失败拒绝启动。初始化器设置私有文件权限，运行主机管理员仍需维护这些权限。

## 4. Linux 单实例部署文件

`deploy/Dockerfile` 固定 GmSSL 3.2.0 源码提交 `ed20e56a016013ca663a0c202ee5f96edf4bc9ad`，多阶段构建，以非 root 用户运行。`deploy/compose.yaml` 提供 Caddy 自动 HTTPS、独立后端网络、只读模型与凭据挂载、持久化运行目录、只读根文件系统和健康检查。应用端口不对外发布；仅在该隔离代理网络中信任转发头。

以下命令供目标服务器使用，**本轮只验证了 Compose 配置，没有构建镜像或连接远程服务器**。首次构建会下载基础镜像、编译工具和 Python 依赖，通常涉及数百 MB 下载及额外构建空间，不能使用当前 Mac 实测代替 Linux 性能数据。

```bash
# 在有项目和三个模型文件的目标主机，使用实际部署用户执行。
.venv/bin/python scripts/init_cloud.py \
  --directory deploy/private --public-origin https://your-domain.example
mkdir -m 700 deploy/runtime
export FACECHECK_DOMAIN=your-domain.example
export FACECHECK_RUN_UID="$(id -u)"
export FACECHECK_RUN_GID="$(id -g)"
docker compose -f deploy/compose.yaml config --quiet
docker compose -f deploy/compose.yaml up --build -d
```

使用非 root 部署用户；UID/GID 必须与私有目录及运行目录所有者一致，避免以放宽密钥权限解决挂载访问问题。容器使用 Compose 指定的路径，不读取初始化器生成的主机路径 `cloud.env`。域名需指向服务器，开放 80/443 供证书签发和访问。本地 Docker 引擎未启动，镜像运行及远程公网 TLS 尚未验证。

## 5. 缓存及完整性边界

`FACECHECK_FEATURE_CACHE` 仅接受 `0` 或 `1`；cloud 默认开启，local 默认关闭。首次检索依次通过源数据完整性校验、恢复、原有检测与特征提取，随后将完整逐帧特征集合加密落盘。再次检索只在以下条件完全匹配时复用：

1. manifest 与所有源密文分片内容的 SHA-256。
2. 模型权重内容、检测与 embedding 配置、相关预处理与提取代码指纹及缓存 schema。
3. 缓存 MAC、解密结果结构、大小、维度、有限值及特征归一化校验。

缓存使用独立用途上下文派生密钥，先验 MAC 后解密，NPZ 加载禁止 pickle。原子发布、每视频 single-flight 锁避免半写文件和同时重复提取。损坏缓存明确拒绝请求，不静默重建掩盖篡改；授权管理员可在停用相关任务后清理损坏缓存。查询特征、评分、Top-K 和阈值仍实时计算，缓存没有改变识别算法。正式 YTF Runner 不经过该缓存。

视频 manifest 保持原格式。新上传将其摘要锚定在可信任务库，恢复函数验证后解析同一字节快照。cloud 拒绝无锚点历史对象，local 保留历史可读性。该机制依赖可信应用与任务库，不抵抗同时控制数据库、主密钥与存储的攻击者；它没有添加独立 manifest MAC。

当前方案在可信应用服务器内恢复数据或加载明文特征进行精确相似度检索。缓存属于加密静态存储和计算复用，尚不构成密文域 ANN、可撤销生物模板、多租户隔离或边缘设备推理。角色授权以同一组织为边界，没有逐视频所有者权限。视频 CBC+HMAC 的先认证后解密顺序保留，不提供对已获授权且已导出明文的追溯撤回。

## 6. 验证与文件

定向测试：**98 passed in 5.63 s**。没有运行完整实验或全量测试套件。

```bash
.venv/bin/python -m pytest -o addopts='' --tb=short \
  tests/test_cloud_access.py tests/test_protected_features.py \
  tests/test_api.py tests/test_health.py tests/test_face_search.py \
  tests/test_recognition_defaults.py tests/test_platform.py \
  tests/test_crypto.py tests/test_crypto_backends.py \
  tests/test_video_pipeline.py -q
```

真实 HTTPS 验证入口：

```bash
.venv/bin/python scripts/validate_cloud.py \
  --sample-video /absolute/path/to/authorized-video \
  --query-image /absolute/path/to/authorized-query \
  --output results/new-cloud-validation.json
```

输出必须为新路径，脚本拒绝覆盖。该脚本使用隔离测试密钥，不读取或写入正式实验结果。本轮固定样本是既有探索数据中的 Gallery `ytf_video_0009` 与 Query `ytf_video_0010`，未据此选择模型或阈值。

主要新增文件：`security.py`、`feature_store.py`、`frontend/auth.js`、三个 cloud 脚本、`deploy/`、两个定向测试文件与独立验证 JSON。主要修改：应用/路由/业务层接线、数据库 WAL、恢复函数可选 manifest 摘要参数、前端登录与缓存展示、README 和配置示例。没有修改密码原语或正式实验结果。
