# FaceGuard 模型

本目录包含本项目本地演示所需的两个 ONNX 模型。模型文件随课程项目仓库分发，方便队友克隆后直接启动；不包含视频、图片或个人数据。

## YuNet

- 文件：`face_detection_yunet_2023mar.onnx`
- 用途：YuNet 人脸检测
- 来源：[OpenCV Zoo - face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- 许可证：以 OpenCV Zoo 仓库和模型发布文件中的许可证声明为准
- SHA-256：`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`

## SFace

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
