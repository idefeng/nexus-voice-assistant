# Voice Assistant Mac 项目分析报告

## 概述
这是一个基于 Python 的 macOS 语音助手项目。它集成了唤醒词检测、语音转文字 (ASR)、通过 OpenClaw 进行的大语言模型 (LLM) 交互以及文字转语音 (TTS)。部分版本还包含计算机视觉功能，如人脸检测和情绪识别。

## 技术栈
- **核心语言**: Python 3
- **唤醒词检测**: `pvporcupine` (Picovoice Porcupine)
- **语音转文字 (ASR)**:
  - `openai-whisper`: 高质量本地 ASR。
  - `speech_recognition` (Google): 简单版本中使用的云端 ASR。
- **文字转语音 (TTS)**:
  - `pyttsx3`: 跨平台 TTS 库。
  - MacOS `say` 命令: macOS 原生 TTS。
- **LLM 集成**: `OpenClaw` (用于 AI 交互的自定义 API/CLI)。
- **计算机视觉**:
  - `insightface`: 用于人脸检测和分析。
  - `onnxruntime`: 用于运行情绪识别模型 (`emotion.onnx`)。
  - `opencv-python`: 用于摄像头控制和图像处理。
- **音频处理**: `pyaudio`, `wave`, `struct`。

## 项目结构
- `main.py`: 功能最全的版本，包含 Whisper ASR、Porcupine 唤醒和计算机视觉（人脸+情绪）。
- `config.py`: 集中管理 API URL、会话密钥和硬件设置。
- `models/`: 包含预训练模型：
  - `wake_word.ppn`: Porcupine 唤醒词文件。
  - `emotion.onnx`: 情绪识别模型。
  - `insightface/`: 人脸识别模型资产。
- **实现变体**:
  - `final_assistant.py`: 使用 Google ASR 和原生 macOS `say` 命令的优化版本。
  - `simple_voice_assistant.py`: 使用 `config.py` 的基础实现。
  - `wake_word_assistant.py`: 专注于稳定唤醒检测的版本（"Hey Siri" 或 "Hey Pico"）。
  - `ultra_simple.py` / `easy_use.py`: 用于测试或低开销运行的 CLI 交互版本。
- **工具脚本**:
  - `test_microphone.py`: 验证音频输入的脚本。
  - `test_full.py`: 验证整个流水线的脚本。
  - `install_m4.sh`: 环境搭建辅助脚本。

## 核心逻辑流程 (`main.py`)
1. **初始化**: 加载 Whisper、Porcupine、InsightFace 和 TTS 引擎。
2. **监听**: 持续监控音频流以检测唤醒词（"Hey Pico"）。
3. **触法**: 一旦被唤醒：
   - 快速通过摄像头进行人脸和情绪分析。
   - 录制用户语音（时长可配置，默认 5 秒）。
   - 使用 Whisper 将语音转为文字。
4. **思考**: 将文字连同检测到的情绪发送至 OpenClaw API。
5. **响应**: 通过 `pyttsx3` 将 OpenClaw 的文字回复转换为语音。
6. **重置**: 重新回到监听唤醒词的状态。

## 关键配置 (`config.py`)
- `OPENCLAW_API_URL`: LLM 的后端接口地址。
- `ENABLE_FACE_RECOGNITION` / `ENABLE_EMOTION_ANALYSIS`: 功能开关。
- `RECORD_SECONDS`: 语音指令录制时长。
- `SAMPLERATE` / `CHANNELS`: 音频硬件参数。

## 观察与总结
- 项目采用模块化设计，允许不同复杂程度的实现（从纯 CLI 到带“视觉”的全功能 AI 助手）。
- 优先选择本地处理（Porcupine, Whisper, InsightFace），同时将“大脑”功能交给 OpenClaw。
- 在某些变体中使用 macOS 原生 `say` 命令，表明了对低延迟响应和环境特定优化的关注。
