# 国密后端和性能记录

本机核对环境为 Apple Silicon arm64、macOS 26.6.2、Python 3.9.6、gmssl Python 包 3.2.2、cryptography 42.0.8。原生后端加载 Homebrew GmSSL C 动态库 `/opt/homebrew/opt/gmssl/lib/libgmssl.dylib`，运行时版本为 GmSSL 3.2.0，通过 `ctypes` 调用其公开 C API。版本号分别指 Python 包和原生动态库，不应混为一个版本。

后端选择方式：

- 未设置 `FACECHECK_CRYPTO_BACKEND` 时默认 `auto`，先尝试 `GmSSLNativeBackend`，不可用时回退到 `PythonBackend`。
- `FACECHECK_CRYPTO_BACKEND=native` 强制加载 GmSSL C 后端；库不可用时启动失败。
- `FACECHECK_CRYPTO_BACKEND=python` 强制使用兼容后端。它使用 gmssl Python 实现 SM2，并使用 cryptography 实现 SM3、HMAC-SM3 和 SM4。
- Native 后端的 SM4 子实现由 `FACECHECK_SM4_NATIVE_IMPL` 选择，默认 `gmssl`；设为 `openssl` 使用 OpenSSL EVP SM4-CBC。该设置不替换 SM2/SM3 的 GmSSL native 路径。本机 OpenSSL 可选实现为 4.1.0；保存的基准没有可靠记录每次运行选择的 SM4 子实现，因此不将它单独作为性能结论。

已保留的 native 低复制实现包括：`ctypes` 直接指向不可变输入 bytes，缓存 SM4 round key，HMAC 分段更新而不先拼接整块认证数据，以及 OpenSSL 选项的线程本地 EVP context。所有 backend 仍走同一 envelope、PKCS#7、SM4-CBC、HMAC-SM3 和认证后解密路径；没有移除 MAC、完整性检查或错误密钥拒绝。

benchmark 使用 16 MiB（16,777,216 字节）内存数据，吞吐单位为十进制 MB/s；SM2 使用进程内随机测试密钥，不记录密钥值。每项 2 次预热、5 次计时，报告中位数、最小值、最大值和标准差。SM2 计入项目统一 API 的签名/验签调用、SM3-with-SM2 摘要和 ctypes/Python binding 调用；随机数生成不计入计时。SM4 分别报告纯 CBC 原语和保留完整密文格式、PKCS#7、随机 IV、SM3-HMAC、封装及内存复制的端到端路径；不计文件 I/O。1080p 项目路径使用 2 帧 MP4、1 MiB 分块，分别测 1 和 4 个加密工作线程。

## 结果

10/31 的多轮对照报告仍保存在 [`results/indicator_benchmark_20261031_native_repeat.json`](../results/indicator_benchmark_20261031_native_repeat.json) 和 [`results/indicator_benchmark_20261031_python_repeat.json`](../results/indicator_benchmark_20261031_python_repeat.json)，作为历史基线，不代表后续低复制优化结果。

| 指标 | native 中位数 | 目标 | 状态 |
|---|---:|---:|---|
| SM2 签名 | 6216 ops/s | >=1000 ops/s | 达标 |
| SM2 验签 | 1927 ops/s | >=500 ops/s | 达标 |
| SM4 CBC 加密原语 | 87.6 MB/s | >=500 MB/s | 未达标 |
| SM4 CBC 解密原语 | 97.4 MB/s | >=500 MB/s | 未达标 |
| 完整 SM4-CBC+HMAC 加密 | 74.9 MB/s | >=500 MB/s | 未达标 |
| 完整 SM4-CBC+HMAC 解密 | 81.5 MB/s | >=500 MB/s | 未达标 |
| 1080p、4 线程项目路径 | 22.11 ms | <=50 ms | 达标 |
| 5 种分辨率 | 5/5 成功 | 5/5 | 达标 |
| 10 路本地 API | 100% | >=99.5% | 达标 |

同一设备的 Python 对照报告为 [`results/indicator_benchmark_20261031_python_repeat.json`](../results/indicator_benchmark_20261031_python_repeat.json)：SM2 为 258.5/153.1 ops/s，SM4 原语为 62.5/62.4 MB/s，完整封装为 40.7/40.4 MB/s。

11/1 保存的低复制 native 结果见 [`results/indicator_benchmark_20261101_sm4_optimized.json`](../results/indicator_benchmark_20261101_sm4_optimized.json)：JSON 记录 `backend=gmssl-native`、7 轮、3 次预热和 16 MiB 输入，但没有记录 GmSSL 版本或 `FACECHECK_SM4_NATIVE_IMPL` 的实际值。以下数值按原记录保留，不据此推断当时 SM4 子实现。

| 指标 | 低复制结果 | 目标 | 状态 |
|---|---:|---:|---|
| SM2 签名 / 验签 | 6136 / 1899 ops/s | >=1000 / >=500 ops/s | 达标 |
| SM4 CBC 加密 / 解密原语 | 199.9 / 263.9 MB/s | >=500 MB/s | 未达标 |
| 完整 SM4-CBC+HMAC-SM3 加密 / 解密 | 146.3 / 175.6 MB/s | >=500 MB/s | 未达标 |
| 纯 HMAC-SM3 | 546.7 MB/s | - | 单项吞吐，不代表完整封装吞吐 |
| 两帧 1080p、4 线程项目路径 | 9.04 ms | <=50 ms | 达标 |

较后的 [`results/indicator_benchmark_20261101_sm4_final.json`](../results/indicator_benchmark_20261101_sm4_final.json) 也保留了独立 7 轮记录：SM4 原语为 94.0/123.9 MB/s、完整封装为 69.3/84.0 MB/s，两帧 1080p 4 线程为 18.02 ms。两份报告都记为 `gmssl-native` 且指向同一动态库路径，但没有记录足以判断 GmSSL 版本、源代码构建、SM4 子实现和运行差异的元数据；因此并列保留，不用后者覆盖低复制结果，也不宣称 199.9/263.9 MB/s 是当前机器每次运行都能稳定达到的数值。chunk matrix 的单项结果另见 [`results/indicator_benchmark_20261101_sm4_chunk_matrix.json`](../results/indicator_benchmark_20261101_sm4_chunk_matrix.json)。

## 正确性和安全边界

native 与 Python 的 SM4-CBC+PKCS#7 输出逐字节一致，空数据、短数据、整块、非整块和大数据均已互解。SM2 原生签名可由 Python 验签，Python 签名可由 native 验签；消息修改和错误公钥验签失败。错误密钥、篡改密文和篡改 MAC 仍在认证阶段拒绝。现有 SM4-CBC + HMAC-SM3 密文 envelope、分块协议和密钥长度未改变；该方案仍是 encrypt-then-MAC，不等同于 AEAD，调用方必须保持先验 MAC 后解密。

## 瓶颈和下一步

在当前机器上，SM2 native 指标已达标。低复制结果中的 SM4 原语加密仍差约 300.1 MB/s，完整封装加密仍差约 353.7 MB/s；解密分别差约 236.1 和 324.4 MB/s。1080p 两帧小样本时延达标，但不代表长视频或端到端上传检索时延。另一份后续记录明显较低，原因尚未由现存 benchmark 元数据确定。SM4 500 MB/s 因此仍是未达标指标；本次收尾不继续做性能优化或重跑基准。若后续需要验收，应先固定动态库构建和 `FACECHECK_SM4_NATIVE_IMPL`，再在目标 Linux ARM64/x86_64 或经验证硬件上按相同口径测量。纯 C 库参考吞吐不得替代完整项目路径指标。
