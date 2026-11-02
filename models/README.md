# FaceGuard 模型

本目录在仓库中分发 YuNet 和 SFace；ArcFace 权重需单独取得并放入本地目录。本目录不包含视频、图片或个人数据。使用和分发权重须遵守各自许可；ArcFace 预训练权重限非商业研究用途，且因体积较大不纳入普通 Git。

## YuNet

- 文件：`face_detection_yunet_2023mar.onnx`
- 用途：YuNet 人脸检测
- 来源：[OpenCV Zoo - face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- 许可证：以 OpenCV Zoo 仓库和模型发布文件中的许可证声明为准
- SHA-256：`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`

## ArcFace（默认）

- 文件：`arcface_w600k_r50.onnx`
- 来源：InsightFace v0.7 官方 `buffalo_l` 模型包，`w600k_r50` ResNet-50
- 用途：112×112 五点对齐，512 维 L2 归一化人脸特征
- 许可：InsightFace 提供的预训练模型限非商业研究用途；代码许可不等同于权重许可
- SHA-256：`4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43`
- 获取：从 [InsightFace buffalo_l 模型包](https://github.com/deepinsight/insightface/tree/master/python-package#model-zoo) 中提取 `w600k_r50.onnx`，保存为上述文件名后核验 SHA-256
- 详细预处理与实验记录：见 `docs/ytf_arcface_w600k_r50.md`

## SFace（可选回退）

- 文件：`face_recognition_sface_2021dec.onnx`
- 用途：SFace 人脸特征提取
- 来源：[OpenCV Zoo - face_recognition_sface](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
- 许可证：以 OpenCV Zoo 仓库和模型发布文件中的许可证声明为准
- SHA-256：`0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`

## 校验模型

macOS/Linux：

```bash
shasum -a 256 models/*.onnx
```

Windows PowerShell：

```powershell
Get-FileHash .\models\*.onnx -Algorithm SHA256
```

模型文件不是人脸数据集，也不包含本项目的测试视频、查询图片或数据库。使用前仍应核对来源和许可证要求。
