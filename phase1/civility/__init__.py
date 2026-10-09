"""摄像头不文明行为识别 —— 第一阶段（本地开发 + 闭环验证）核心包。

模块划分：
    config      读取 settings.yaml / behaviors.yaml
    prompts     把"行为定义"编译成 Prompt（含 JSON Schema 与 few-shot）
    vlm         统一视觉推理后端：mock（假模型）/ cloud（OpenAI 兼容）/ cluster（A800）
    extract     从视频抽帧成窗口（ffmpeg + 缩放）
    aggregate   时序平滑与告警判定
    runner      编排：manifest → 推理 → 事件
    score       对着标注算 P/R/F1
"""

__version__ = "0.1.0"
