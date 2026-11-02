# 指标核对与验收边界

本表根据正式 ArcFace test、登记 manifest、国密 benchmark、HTTPS 云端验证和阶段二小型功能核验更新。原始机器核对 `results/facecheck_review_20261101_audit.json` 和阶段二初版 `results/facecheck_review_20261102_phase2_v2.json` 保留为历史快照；当前状态见 `results/facecheck_review_20261102_phase2_v8.json`，其中保存输入文件 SHA-256，生成时不运行识别实验或密码 benchmark。

状态定义：`PASS` 为当前测量达到目标；`FAIL` 为已知实际规模或分数低于目标；`PARTIAL` 为功能/单样本/小样本有证据但不符合目标验收口径；`NOT_IMPLEMENTED` 为无对应机制；`NOT_MEASURED` 为缺符合口径的测试。不能把 `encrypted_restore` 的认证解密恢复后检索称作密文域检索。

另有一项 20 身份 YuNet 阈值 smoke：既有 0.90 子集 Top-1 为 18/20，新跑 0.85 为 19/20，检测 Query 为 19/20 与 20/20。样本来自已有 selection，380 个 impostor pair 不足以估计 FAR=1e-3；此结果不进入下表正式指标、不改变默认阈值或正式 test 结论，详见 `data/ytf/experiments/ytf_arcface_yunet085_smoke20_20261101/`。

| 指标 | 当前证据 | 状态 | 差距/边界 |
|---|---|---|---|
| ≥1万人、10万段、100万帧综合集 | Manifest：1,004身份、1,004段Gallery、1,004张Query、190,191个Gallery源帧；正式test为502身份 | FAIL | 缺真实授权规模数据；合成/重复身份只能用于负载测试 |
| 明文1:N Top-1 ≥95% | ArcFace正式test 432/502 = 86.0558% | FAIL | 差8.9442个百分点；不得在该test上调参 |
| TAR@FAR≤1e-3 ≥90% | 435/502 = 86.6534%，FAR=0.000998004 | FAIL | FAR满足该点，TAR差3.3466个百分点；阈值来自独立selection |
| 密文域Top-1下降≤2个百分点 | 目前认证恢复后再明文检索 | NOT_IMPLEMENTED | 需先完成威胁模型、泄漏定义和安全索引 |
| 密文域结果与明文一致率≥99.9% | 恢复模式252,004对中有分数差最大值0 | PARTIAL | 只验证存储恢复一致，不代表密文域 ANN |
| SM4加/解密吞吐均≥500 MB/s | `final`完整CBC+HMAC-SM3加/解密中位数69.34/84.00 MB/s；`optimized`历史记录146.30/175.63 MB/s | FAIL | 所有已存完整封装记录均低于目标；optimized构建元数据不完整，不能作为稳定保证 |
| SM2签名/验签≥1,000/500次/s | `final`本机多轮中位数2,957.61/932.87次/s | PASS（本机基准） | 仍需在目标云端设备上复测 |
| 1080p单路加密≤50 ms | 两帧合成1080p文件，`final`4线程中位数18.02 ms | PARTIAL | 未测持续真实码流、长视频或端侧最坏时延 |
| 端到端平均响应≤2 s | HTTPS单样本冷启动4.9408 s；缓存命中0.4646 s | PARTIAL | 冷样本超时；冷/热各一次，未测总体均值、P95或并发检索 |
| ≥3类终端、5分辨率、10路并发 | 本机5种合成分辨率成功；10个四帧HTTPS上传均成功 | PARTIAL | 缺3类真实终端及持续10路视频推理 |
| 轮换后旧密钥失效率100% | v1 重加密、v2 rewrap、SQLite 持久化队列、数据库内引用/manifest 锚点事务切换、缓存清理及撤销门槛均有小样本原型 | PARTIAL | 单机 SQLite 不是生产分布式队列；真实 KMS/HSM、可信公钥目录部署、全部备份/灾备盘点、跨系统原子性和真实全局撤销未验收；100% 尚未证明 |
| 未授权解密成功率≤0.1% | 100次错误密钥解密，成功0次 | PARTIAL | 仅是有限密码拒绝样本，不覆盖API越权、多租户和凭据失陷 |
| 视频及特征篡改检出率≥99.9% | 100次随机视频分块篡改全部拒绝；缓存/视频定向篡改有功能验证 | PARTIAL | 未测特征索引总体检出率及大样本攻击集合 |
| API成功率≥99.5% | 10个本地并发API请求和10个HTTPS合成上传均成功 | PARTIAL | 不代表长期服务可用率、持续负载或推理并发 |
| 日均≥10万段视频或100万帧 | 无持续日吞吐压测 | NOT_MEASURED | 需部署目标服务器、对象存储、队列及资源监控 |
| 自动审计报告成功率≥99% | CSV/JSON/Markdown审计导出功能已实现 | PARTIAL | 未统计持续运行成功率 |
| 审计日志完整率≥99.9% | 本地 SHA-256 哈希链和校验接口已实现 | NOT_MEASURED | 无独立外部锚点，不能证明尾部截断/整库回滚；缺外部对账和丢失率测量 |

## 研究内容状态

| 研究内容 | 状态 | 说明 |
|---|---|---|
| SM2/SM4数字信封 | 部分实现 | opt-in v2：每视频随机 SM4/MAC 主密钥材料、SM2 接收方封装、key ID 和 HMAC-SM3 manifest；另有 SHA-256 pin 公钥目录和 provider adapter 接口，但没有可信生产部署/真实 KMS；Web/cloud 默认流程仍为 v1 |
| 视频分块与选择性加密 | 部分实现 | 全视频字节分块认证加密已实现；编解码器感知选择性加密未实现 |
| 密钥派生与轮换 | 部分实现 | v1 全量重加密、v2 rewrap、单机 SQLite 持久队列、DB 内引用和 manifest 锚点事务切换、缓存失效与撤销门槛已有原型；生产 KMS、分布式队列、外部备份清单及全局撤销未验收 |
| 人像特征保护与可撤销模板 | 部分实现 | 特征缓存静态加密和认证已实现；可撤销、跨版本不可关联模板未实现 |
| 密文安全索引与ANN | 未实现 | 当前服务端认证恢复后执行明文精确检索 |
| 边缘—云轻量化推理 | 未实现 | 没有边缘设备部署与协同协议验收 |
| 完整性与审计追溯 | 部分实现 | 分块MAC、恢复SHA-256、本地审计哈希链、快照校验和导出已实现；外部防删证明与完整率未测 |
| 平台集成 | 部分实现 | 课程平台和本机HTTPS原型已实现；多租户、远程持续负载和规模化部署未测 |

## 重建机器可读结果

下面命令仅读取已保存 JSON 和 manifest；输出路径必须不存在，脚本拒绝覆盖：

```bash
.venv/bin/python scripts/evaluate_indicators.py \
  --metrics data/ytf/experiments/ytf_arcface_test_20261101/metrics.json \
  --benchmark results/indicator_benchmark_20261101_sm4_final.json \
  --security-benchmark results/indicator_benchmark_20261031.json \
  --manifest data/ytf/experiment_full.json \
  --cloud-validation results/cloud_validation_20261101_v2.json \
  --output results/new_facecheck_review.json
```

细分阶段、威胁模型、待办和外部条件见 `docs/optimization_roadmap_20261101.md`、`docs/adr/0001-private-search-architecture.md`、`docs/adr/0002-video-key-envelope-and-rotation.md` 与 `docs/audit_integrity_20261101.md`。
