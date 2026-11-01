# ADR 0002：逐视频 SM2 密钥信封与轮换

- 状态：v2 存储和文件批量 rewrap 原型已实现；生产接入待实施
- 日期：2026-11-02
- 范围：阶段二的存储格式、密钥接口、旧格式兼容和轮换语义

## 背景

当前生产 v1 视频对象以进程配置中的 SM4/MAC 主密钥，结合 `key_version` 经 SM3 KDF 派生分块密钥。每个分块用 SM4-CBC 加密，并以 HMAC-SM3 认证视频 ID、块序号和密钥版本。云端另将 manifest 的 SHA-256 存在任务数据库中供恢复时比对；本地模式没有独立 manifest 锚点。新增 v2 opt-in 存储原型已用逐视频随机 key material、SM2 信封和 HMAC-SM3 manifest 认证补上格式路径；v1 仍是生产默认。

## 威胁模型与边界

### 保护目标

- 仅持有对象文件的存储侧不能恢复视频明文或 per-video 加密密钥。
- 对 manifest、密钥信封、块清单或块文件的修改，在恢复前失败关闭。
- 密钥轮换先生成并验证新副本，再允许调用方切换对象引用；失败时原对象可继续读取。

### 假设与不覆盖项

- SM2 公钥来自可信注册/证书来源，私钥由调用方保护。当前原型只提供内存中的 `KeyProvider` 实现，不是 KMS，也不提供生产密钥注入或私钥轮换托管。
- 拥有应用进程或私钥的攻击者可解密数据；本方案不保护运行时明文、内存转储、恶意服务端或终端被攻陷。
- 访问模式、对象 ID、时间、密文大小、分块数量及公开 manifest 中的源 SHA-256 仍可见。外部摘要锚点可检测云端 manifest 被替换，但不提供独立的本地防回滚。
- HMAC 可检测篡改，不证明 manifest 是最新版本。防回滚需要数据库中单调版本/可信锚点，并要覆盖备份恢复流程。

## v2 格式决策

1. v1 不改格式，继续由现有 `store_encrypted_video` 写入，旧对象仍可按既有密钥只读恢复。
2. 显式 opt-in 的 v2 为每段视频生成 48 字节随机 key material：前 16 字节作为 SM4 encryption master，后 32 字节作为独立 MAC master。沿用已有 SM3 KDF 派生块密钥，不改变现有分块格式和认证上下文。
3. 用接收方 SM2 公钥加密 key material。信封保存算法、外部提供的 `recipient_key_id` 和 base64 密文。加密明文内绑定格式标记、视频 ID、`key_version` 和 recipient key ID，阻止把另一对象的信封移入当前 manifest。
4. manifest `version=2`，加入 `key_envelope` 与 `manifest_authentication`。后者以独立 KDF context 派生 HMAC-SM3 密钥，对去掉认证字段后的 canonical manifest（含信封、块清单、大小和摘要）认证。
5. v2 恢复必须显式提供 `KeyProvider` 和数据库/调用方给出的预期视频 ID；先检查身份绑定、解包 key material 和 manifest HMAC，再验证块清单、认证并恢复。缺少 provider、key ID 错误、解包/认证失败均拒绝，不回退尝试 v1。
6. v2 先作为独立原型 API，不改现有 API/云端默认写路径。完成可信密钥提供、数据库迁移与回滚策略之前，不宣称生产 KMS 或全局撤销。

## 轮换语义

- **接收方密钥轮换 / rewrap**：旧私钥解包 per-video key material，新公钥重新封装；块密文不变。写入新的对象副本，更新 HMAC 和外部 manifest 锚点，完整恢复校验通过后由调用方原子切换数据库引用。只要旧 manifest、缓存或备份仍留存，旧私钥仍可能恢复它们；不能声称全局旧密钥失效。
- **per-video key 泄露 / 数据密钥轮换**：生成新的 key material 并重新加密全部分块；这与只更换接收方信封不同，需更高 I/O 和双份存储空间。
- 本地文件批量原型已写原子 JSON journal，记录批次指纹、对象 ID、源摘要、目标摘要、尝试次数和错误类型；重跑会验证已完成目标并续跑失败对象。journal 不保存 key material。
- 原型将目标对象写入新存储根并完整认证恢复后再发布，源对象保持可读；它不会切换应用数据库引用，也不处置缓存、备份/快照或旧副本。服务级队列、外部状态切换/审计和全局撤销仍未实现。

## 选择性加密

本阶段不改视频字节加密策略。继续加密全部视频块，避免暴露 codec frame type、帧边界、时间戳或依赖 GOP 结构的明文。若研究 codec-aware 选择性加密，另写泄漏分析和格式 ADR，并以全量加密为对照。

## 验收门槛

- v1 现有样本仍可恢复；v2 小样本能跨 SM2 provider 封装/解封、完成分块认证恢复及源摘要校验。
- 修改 v2 manifest、换 recipient key ID、错误预期视频 ID、错误私钥和改写块都必须失败，且不发布部分明文。
- 轮换验证源对象仍可读，目标对象由新 provider 恢复一致；未完成外部引用切换前不删除源对象。
- 批量轮换与旧密钥全局失效只有在真实 KMS/存储/缓存/备份清单、恢复演练及持久化批任务完成后才可验收。

## 当前原型实现记录

- `src/facecheck/key_management.py` 提供 `VideoKeyProvider` 协议和显式注入密钥的 `Sm2MemoryKeyProvider`；它不是 KMS，且不会把私钥持久化。
- `src/facecheck/storage.py` 的 `store_enveloped_video` / `restore_enveloped_video` 写读 manifest v2；旧版 `store_encrypted_video` / `restore_encrypted_video` 仍保持 v1 行为。
- `rewrap_enveloped_video` 与 `src/facecheck/key_rotation.py` 的 `rotate_enveloped_batch` 可生成独立目标副本并用目标 provider 完整恢复校验；数据库引用切换仍由后续服务集成负责。
- 新增的小型测试覆盖 v1 兼容、v2 往返、manifest/分块篡改、错误接收方、单视频 rewrap 和批次断点续跑。测试密钥只用于本地合成样本。
- SM2 加密互操作目前只在同一 gmssl Python 实现内完成小样本往返；尚未验证与 GmSSL C、厂商密码机或独立证书工具的密文编码互通，也没有测密钥信封时延。

## 需要外部决策

- 可信 SM2 公钥注册与 key ID 命名/撤销策略。
- 私钥所在 HSM/KMS、访问策略、备份和灾难恢复责任。
- 本地/云端数据库与对象存储的原子引用切换方式及备份保留期限。
- 旧密钥失效统计范围是否包含数据库快照、离线备份、临时文件、缓存和灾备副本。
