# macOS 智能语音助手 (Voice Assistant Mac)

这是一个基于 Python 开发的 macOS 语音助手，集成了唤醒词检测、语音识别 (ASR)、自然语言处理 (OpenClaw/LLM)、摄像头视觉分析以及长期记忆系统。

## 核心功能
*   **唤醒词检测**：支持中文唤醒词 "**你好，小德**"（由 Picovoice 提供）。
*   **长期记忆 (Deep Memory)**：集成 **ChromaDB** 向量数据库，支持跨会话记忆用户偏好和背景。
*   **语音识别 (ASR)**：使用 OpenAI Whisper (**Medium** 模型) 进行高精度中文识别。
*   **高级语音合成 (TTS)**：使用 **Edge-TTS** 提供的神经网络拟人音色，支持 Markdown/Emoji 自动过滤。
*   **丝滑交互 (Fluid UX)**：支持语音播报**随时打断**、分段加速播报及纯净终端日志输出。
*   **智能对话**：通过 OpenClaw Gateway 连接 LLM (如 DeepSeek)，支持情感理解与背景检索（RAG）。
*   **视觉分析**：自动调用摄像头进行人脸检测及情绪分析 (InsightFace + ONNX)。

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
*   `memory_manager.py`: 长期记忆管理引擎。
*   `models/`: 存放唤醒词 (.ppn) 及视觉分析模型 (.onnx)。

## 版本记录
*   **v6.0.1**: **感知增强修复版**。显著提升音频流读取稳定性，优化屏幕感知的鲁棒性与权限容错。
*   **v6.0.0**: **认主关怀版**。集成人脸识别识别身份、情绪分析调节语气，支持主动问候与屏幕洞察。
*   **v2.2.0**: **丝滑交互版**。
*   **v2.1.0**: **语音优化版**。实现 TTS 文本清理（过滤 Markdown 符号和表情）。
*   **v2.0.0**: **记忆觉醒版**。集成 ChromaDB 实现长期记忆存储与检索。
*   **v1.5.0**: **稳定性增强版**。增加 API 超时重试及“正在思考”语音反馈。
*   **v1.0.0**: 初始版本。基础 ASR/TTS/Vision 闭环。

## 开发者
*   **张德锋 (idefeng)**
