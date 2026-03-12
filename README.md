# macOS 智能语音助手 (Voice Assistant Mac)

这是一个基于 Python 开发的 macOS 语音助手，集成了唤醒词检测、语音识别 (ASR)、自然语言处理 (OpenClaw/LLM)、摄像头视觉分析（人脸及情绪识别）以及语音合成 (TTS) 功能。

## 核心功能
*   **唤醒词检测**：支持 "Picovoice" 唤醒（由 Porcupine 提供）。
*   **语音识别 (ASR)**：使用 OpenAI Whisper (Small 模型) 进行高质量中文识别。
*   **智能对话**：通过 OpenClaw Gateway 连接 LLM (如 豆包/DeepSeek)，提供智能回复。
*   **视觉分析**：自动调用摄像头进行人脸检测及情绪分析 (InsightFace + ONNX)。
*   **语音合成 (TTS)**：使用 macOS 原生语音或 `pyttsx3` 进行回复。

## 安装与运行

### 1. 环境准备
确保您的系统中已安装 Python 3.13+ 和 OpenClaw。

### 2. 自动配置 (推荐)
运行以下脚本进行环境初始化：
```bash
bash install_m4.sh
```

### 3. OpenClaw 配置
确保 OpenClaw Gateway 已启动并启用了 `chatCompletions` 接口：
```bash
openclaw gateway start
```

### 4. 运行助手
```bash
./venv/bin/python main.py
```

## 目录结构
*   `main.py`: 核心全功能入口。
*   `config.py`: 项目全局配置。
*   `models/`: 存放唤醒词 (.ppn) 及视觉分析模型 (.onnx)。
*   `project_analysis_zh.md`: 详细的项目技术分析报告。

## 开发者
*   **张德锋 (idefeng)**

## 版本记录
*   **v1.0.0**: 初始版本，修复了多项运行时 Bug，优化了中文识别准确度及 OpenClaw 接口对接。
