#!/bin/bash
echo "🚀 正在为M4 Mac安装语音助手依赖..."

# 检查Homebrew
if ! command -v brew &> /dev/null
then
    echo "📦 安装Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# 安装系统依赖
echo "📦 安装系统依赖..."
brew install portaudio ffmpeg git

# 创建Python虚拟环境
echo "🐍 创建Python虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 升级pip
echo "⬆️  升级pip..."
pip install --upgrade pip

# 安装Python依赖，使用M4优化源
echo "🐍 安装Python依赖..."
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 创建模型目录
mkdir -p models

# 下载预训练模型（M4优化版本）
echo "🤖 下载M4优化的预训练模型..."
cd models

# 唤醒词模型
curl -L -o wake_word.ppn https://github.com/Picovoice/porcupine/raw/master/resources/keyword_files/osx/hey%20pico_osx.ppn

# 人脸识别模型
curl -L -o buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip -q buffalo_l.zip -d insightface
rm buffalo_l.zip

# 情绪分析模型
curl -L -o emotion.onnx https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx

echo "✅ 安装完成！请配置config.py中的session key即可使用。"
