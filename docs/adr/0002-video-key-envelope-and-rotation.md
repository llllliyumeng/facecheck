# ADR 0002：逐视频 SM2 密钥信封与轮换

- 状态：v2 存储、单机 SQLite 轮换队列和引用切换原型已实现；生产接入待实施
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

- SM2 公钥必须来自可信注册/证书来源，私钥由调用方保护。代码提供独立 SHA-256 pin 校验的 `PinnedVideoKeyDirectory` 和受信 `module:function` provider factory 接口；目前没有已部署的可信目录或真实 KMS/HSM adapter。`Sm2MemoryKeyProvider` 仅供本地原型使用。
- 拥有应用进程或私钥的攻击者可解密数据；本方案不保护运行时明文、内存转储、恶意服务端或终端被攻陷。
- 访问模式、对象 ID、时间、密文大小、分块数量及公开 manifest 中的源 SHA-256 仍可见。外部摘要锚点可检测云端 manifest 被替换，但不提供独立的本地防回滚。
- HMAC 可检测篡改，不证明 manifest 是最新版本。防回滚需要数据库中单调版本/可信锚点，并要覆盖备份恢复流程。

## v2 格式决策

1. v1 不改格式，继续由现有 `store_encrypted_video` 写入，旧对象仍可按既有密钥只读恢复。
2. 显式 opt-in 的 v2 为每段视频生成 48 字节随机 key material：前 16 字节作为 SM4 encryption master，后 32 字节作为独立 MAC master。沿用已有 SM3 KDF 派生块密钥，不改变现有分块格式和认证上下文。
3. 用接收方 SM2 公钥加密 key material。信封保存算法、外部提供的 `recipient_key_id` 和 base64 密文。加密明文内绑定格式标记、视频 ID、`key_version` 和 recipient key ID，阻止把另一对象的信封移入当前 manifest。
4. manifest `version=2`，加入 `key_envelope` 与 `manifest_authentication`。后者以独立 KDF context 派生 HMAC-SM3 密钥，对去掉认证字段后的 canonical manifest（含信封、块清单、大小和摘要）认证。
5. v2 恢复必须显式提供 `KeyProvider` 和数据库/调用方给出的预期视频 ID；先检查身份绑定、解包 key material 和 manifest HMAC，再验证块清单、认证并恢复。缺少 provider、key ID 错误、解包/认证失败均拒绝，不回退尝试 v1。
6. v2 先作为显式原型 API，不改现有 API/云端默认写路径。完成可信密钥提供、数据库迁移与回滚策略之前，不宣称生产 KMS 或全局撤销。

## 轮换语义

- **接收方密钥轮换 / rewrap**：旧私钥解包 per-video key material，新公钥重新封装；块密文不变。写入新的对象副本，更新 HMAC 和外部 manifest 锚点，完整恢复校验通过后由调用方原子切换数据库引用。只要旧 manifest、缓存或备份仍留存，旧私钥仍可能恢复它们；不能声称全局旧密钥失效。
- **per-video key 泄露 / 数据密钥轮换**：生成新的 key material 并重新加密全部分块；这与只更换接收方信封不同，需更高 I/O 和双份存储空间。
- 本地文件批量原型保留原子 JSON journal 路径；另有 `rotation_control.py` 提供单 SQLite 数据库内的持久化任务/对象清单、重试和显式中断恢复。
- worker 先发布并认证恢复目标，再在一个数据库事务中 CAS 切换 `storage_id`、manifest SHA-256 锚点、key ID/version 和审计事件；源文件保留供回滚/处置。文件发布与 SQLite 提交不是跨存储原子事务，崩溃可能留下目标副本，由确定性目标路径支持恢复。
- 原型能盘点已引用对象、常见未引用副本、配置的特征缓存与配置备份目录；可使缓存失效，管理员 API 可在对象实际迁移/删除后登记处置证据。目录遗漏、外部备份、数据库快照和灾备介质仍依赖运维清单与独立核验。发现隐藏暂存项或未知存储项时，清单完整性门槛失败，需清理后重新建计划。
- 若 artifact 关联仍在使用旧对象的任务，数据库层拒绝登记其已迁移/删除；必须先成功切换任务引用，再处置旧源副本或备份。
- 撤销前门槛包括数据库引用、任务项、已登记缓存/副本、inventory digest、写入围栏和在途读取排空。写入围栏、读取排空、外部备份完整性目前是部署配置/运维确认，不是跨节点强制栅栏。KMS 回执接口只验证调用流程，不证明真实密钥已由生产 KMS 全局撤销。
- 当前队列和引用切换仅适用于单机 SQLite 原型；这不是分布式任务队列，也不是数据库与对象存储间的原子提交协议。旧密钥失效率 100% 尚未证明。

## 选择性加密

本阶段不改视频字节加密策略。继续加密全部视频块，避免暴露 codec frame type、帧边界、时间戳或依赖 GOP 结构的明文。若研究 codec-aware 选择性加密，另写泄漏分析和格式 ADR，并以全量加密为对照。

## 验收门槛

- v1 现有样本仍可恢复；v2 小样本能跨 SM2 provider 封装/解封、完成分块认证恢复及源摘要校验。
- 修改 v2 manifest、换 recipient key ID、错误预期视频 ID、错误私钥和改写块都必须失败，且不发布部分明文。
- 轮换验证源对象仍可读，目标对象由新 provider 恢复一致；未完成外部引用切换前不删除源对象。
- 批量轮换与旧密钥全局失效只有在真实 KMS/存储/缓存/备份清单、恢复演练、分布式持久化批任务及外部撤销核验完成后才可验收。

## 当前原型实现记录

- `src/facecheck/key_management.py` 提供 `VideoKeyProvider`、`PinnedVideoKeyDirectory`、provider registry/factory 接口及 `Sm2MemoryKeyProvider`。pin 必须通过独立受保护配置提供；实际生产信任根和 KMS/HSM 集成未部署。
- `src/facecheck/storage.py` 的 `store_enveloped_video` / `restore_enveloped_video` 写读 manifest v2；旧版 `store_encrypted_video` / `restore_encrypted_video` 仍保持 v1 行为。
- `rewrap_enveloped_video` 与 `src/facecheck/key_rotation.py` 的 `rotate_enveloped_batch` 保留文件 journal 路径；`src/facecheck/rotation_control.py` 另提供 SQLite 单机任务队列、数据库 CAS 切换、缓存失效和撤销门槛原型。
- 定向合成测试覆盖 v1/v2、篡改拒绝、rewrap、数据库指针切换/manifest 锚点、CAS 冲突、中断恢复、缓存失效、inventory 阻塞及撤销回调。测试密钥和回执仅用于本地合成样本。
- v1 转 v2 时会在受限临时目录短暂生成明文；正常退出会清理，强制终止可能留下 staging 文件。当前清单会将隐藏/未知存储项标记为不完整并阻止撤销，但不自动安全擦除残留数据。
- SM2 加密互操作目前只在同一 gmssl Python 实现内完成小样本往返；尚未验证与 GmSSL C、厂商密码机或独立证书工具的密文编码互通，也没有测密钥信封时延。

## 需要外部决策

- 可信 SM2 公钥注册与 key ID 命名/撤销策略。
- 私钥所在 HSM/KMS、访问策略、备份和灾难恢复责任。
- 生产数据库与对象存储的可恢复引用切换协议及备份保留期限；现有 SQLite 原子事务仅覆盖数据库内字段。
- 旧密钥失效统计范围是否包含数据库快照、离线备份、临时文件、缓存和灾备副本。
