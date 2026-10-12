# 人员检测资产

本项目使用 [OpenCV Zoo 的 YOLOX 人员检测模型](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox)，在本机 CPU 运行。模型文件由 `tools/prepare_person_detector.py` 从官方 OpenCV Hugging Face 镜像下载，不入 Git。

- 文件：`object_detection_yolox_2022nov.onnx`，35,858,002 bytes。
- SHA-256：`c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`。
- 上游许可证：Apache-2.0，完整原文见 `opencv-yolox-LICENSE.txt`，来自 OpenCV Zoo 对应模型目录。
- 本项目自行实现预处理、输出解码、匿名关联和时间账本；模型资产仍适用其上游许可证。
