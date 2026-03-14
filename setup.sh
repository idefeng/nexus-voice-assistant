#!/bin/bash

# 小德 (Xiaode) 语音助手 - 项目初始化脚本
# 支持 OS: macOS

set -e

echo "🐻 欢迎使用小德 (Xiaode) 语音助手初始化脚本"
echo "==========================================="

# 1. 检查操作系统
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ 抱歉，本脚本目前仅支持 macOS。"
    exit 1
fi

# 2. 检查并安装 Homebrew
if ! command -v brew &> /dev/null; then
    echo "📦 正在安装 Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✅ Homebrew 已安装"
fi

# 3. 安装系统依赖
echo "📦 正在安装系统依赖 (ffmpeg, portaudio)..."
brew install ffmpeg portaudio git

# 4. 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到 python3，请先安装 Python 3.9+"
    exit 1
fi

# 5. 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "🐍 正在创建 Python 虚拟环境..."
    python3 -m venv venv
else
    echo "✅ 虚拟环境 venv 已存在"
fi

# 6. 安装 Python 依赖
echo "🐍 正在安装 Python 依赖项..."
source venv/bin/activate
pip install --upgrade pip
# 针对 Mac (包括 M1/M2/M3/M4) 优化安装 PyTorch
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 7. 准备模型目录
echo "🤖 检查并准备模型..."
mkdir -p models

# 唤醒词参数文件检查
if [ ! -f "models/porcupine_params_zh.pv" ]; then
    echo "📥 正在获取唤醒词参数..."
    # 注意：这里假设用户已有此文件或从特定位置获取，此处仅作演示路径提醒
    echo "ℹ️ 请确保 models/porcupine_params_zh.pv 已就绪 (Porcupine 中文版需要此文件)"
fi

# 情绪分析模型检查
if [ ! -f "models/emotion.onnx" ]; then
    echo "📥 正在下载情绪分析模型..."
    curl -L -o models/emotion.onnx https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx
fi

# 8. 检查配置文件
if [ ! -f "config.py" ]; then
    echo "⚠️ 未找到 config.py，请参考项目仓库中的配置进行创建。"
else
    echo "✅ 配置文件 config.py 已就绪"
fi

echo "==========================================="
echo "✅ 初始化完成！"
echo "🚀 运行方法: source venv/bin/activate && python3 app.py"
