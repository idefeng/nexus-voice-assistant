import os
import sys
import subprocess
import asyncio
import edge_tts
import pvporcupine
import pyaudio
import struct
import whisper
import requests
import cv2
import numpy as np
import insightface
from onnxruntime import InferenceSession

from config import *

# 初始化Whisper模型
print(f"🔊 加载语音识别模型 ({WHISPER_MODEL})...")
whisper_model = whisper.load_model(WHISPER_MODEL)

# 初始化人脸识别模型
if ENABLE_FACE_RECOGNITION:
    print("👤 加载人脸识别模型...")
    face_analyzer = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# 初始化情绪分析模型
if ENABLE_EMOTION_ANALYSIS:
    print("😐 加载情绪分析模型...")
    emotion_session = InferenceSession("./models/emotion.onnx")
    emotion_labels = ['neutral', 'happy', 'surprise', 'sad', 'angry', 'disgust', 'fear', 'contempt']

# 初始化Porcupine唤醒引擎
print("🔔 初始化唤醒引擎...")
try:
    porcupine = pvporcupine.create(
        access_key=PORCUPINE_ACCESS_KEY,
        keyword_paths=[WAKE_WORD_PATH]
    )
except Exception as e:
    print(f"⚠️ 无法加载自定义唤醒词 {WAKE_WORD_PATH}: {e}")
    print("💡 尝试使用内置唤醒词 'picovoice' (请改喊 'Picovoice' 唤醒)")
    porcupine = pvporcupine.create(
        access_key=PORCUPINE_ACCESS_KEY,
        keywords=['picovoice']
    )

# 初始化音频流
pa = pyaudio.PyAudio()
audio_stream = pa.open(
    rate=porcupine.sample_rate,
    channels=CHANNELS,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length
)

def detect_face_and_emotion():
    """检测人脸和情绪"""
    cap = cv2.VideoCapture(CAMERA_ID)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None, None
    
    # 人脸检测
    faces = face_analyzer.get(frame)
    if len(faces) == 0:
        return None, None
    
    # 情绪分析
    face_img = cv2.resize(frame, (64, 64))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = face_img.astype(np.float32) / 255.0
    # ONNX model expects (batch, channels, height, width) -> (1, 1, 64, 64)
    face_img = np.expand_dims(face_img, axis=0) # (1, 64, 64)
    face_img = np.expand_dims(face_img, axis=0) # (1, 1, 64, 64)
    
    outputs = emotion_session.run(None, {'Input3': face_img})
    emotion_idx = np.argmax(outputs[0])
    emotion = emotion_labels[emotion_idx]
    
    return faces[0], emotion

def record_audio(max_duration):
    """录制音频，支持静音检测和长句子"""
    print(f"🎙️  开始录音 (最长 {max_duration}s)...")
    frames = []
    
    # 简单的静音检测参数
    CHUNK_SIZE = porcupine.frame_length
    SILENCE_THRESHOLD = 500  # 能量阈值
    SILENCE_DURATION = 1.5   # 连续静音秒数
    silence_frames_limit = int(porcupine.sample_rate / CHUNK_SIZE * SILENCE_DURATION)
    
    silence_counter = 0
    recorded_frames = 0
    max_frames = int(porcupine.sample_rate / CHUNK_SIZE * max_duration)
    
    while recorded_frames < max_frames:
        try:
            pcm_data = audio_stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(pcm_data)
            recorded_frames += 1
            
            # 计算能量用于静音停止（可选，这里先实现长句收录）
            pcm_unpacked = struct.unpack_from("h" * CHUNK_SIZE, pcm_data)
            energy = np.abs(pcm_unpacked).mean()
            
            if energy < SILENCE_THRESHOLD:
                silence_counter += 1
            else:
                silence_counter = 0
                
            # 如果连续录够了基准时长且持续静音，则提前结束
            if recorded_frames > int(porcupine.sample_rate / CHUNK_SIZE * 3) and silence_counter > silence_frames_limit:
                print("⏹️  检测到静音，停止录音")
                break
                
        except Exception as e:
            print(f"⚠️ 录音异常: {e}")
            break
            
    return b''.join(frames)

def audio_to_text(audio_data):
    """语音转文字"""
    print("✍️  识别中...")
    # 保存临时wav文件
    import wave
    wf = wave.open("/tmp/recording.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
    wf.setframerate(porcupine.sample_rate)
    wf.writeframes(audio_data)
    wf.close()
    
    # 识别
    result = whisper_model.transcribe("/tmp/recording.wav", language="zh")
    os.remove("/tmp/recording.wav")
    return result["text"]

def call_openclaw(text, emotion=None):
    """调用OpenClaw接口获取回复 (OpenAI兼容接口)"""
    try:
        # 根据情绪调整Prompt
        content = text
        if emotion:
            content = f"[用户当前情绪：{emotion}] {text}"
            
        headers = {
            "Authorization": f"Bearer {OPENCLAW_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": SESSION_KEY,  # 这里SESSION_KEY即Agent ID (e.g., 'scholar')
            "messages": [
                {"role": "user", "content": content}
            ]
        }
        
        response = requests.post(OPENCLAW_API_URL, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            return "抱歉，我听到了你的问题，但没有得到有效回复。"
        else:
            print(f"❌ OpenClaw返回错误: {response.status_code} - {response.text}")
            return "抱歉，我现在无法回答你的问题。"
    except Exception as e:
        print(f"❌ OpenClaw调用失败: {e}")
        return "网络连接出现问题，请稍后再试。"

async def _edge_speak(text):
    """内部异步语音合成函数"""
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    temp_file = "/tmp/reply.mp3"
    await communicate.save(temp_file)
    # 使用 afplay (macOS 自带) 播放 mp3
    subprocess.run(["afplay", temp_file])
    if os.path.exists(temp_file):
        os.remove(temp_file)

def speak(text):
    """语音合成，支持智能切换引擎"""
    if not text:
        return
    print(f"🤖 阿信：{text}")
    try:
        # 使用 edge-tts 获得高级自然语音
        asyncio.run(_edge_speak(text))
    except Exception as e:
        print(f"⚠️ Edge-TTS 失败，降级使用系统语音: {e}")
        try:
            subprocess.run(["say", "-v", "Mei-Jia", text], check=True)
        except:
            subprocess.run(["say", text])
 
def main():
    print("✅ 语音助手已启动，等待唤醒词...")
    print("💡 唤醒词：Hey Pico (可以在配置中修改)")
    
    try:
        while True:
            try:
                pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            except OSError as e:
                # 忽略特定平台的缓冲区溢出错误
                if e.errno == -9981:
                    continue
                raise
            
            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                print("\n🎉 唤醒成功！")
                speak("我在呢")
                
                # 检测人脸和情绪
                emotion = None
                if ENABLE_FACE_RECOGNITION and ENABLE_EMOTION_ANALYSIS:
                    face, emotion = detect_face_and_emotion()
                    if emotion:
                        print(f"😊 当前情绪：{emotion}")
                        speak(f"检测到你现在的状态是{emotion}")
                
                # 录制语音 (使用配置的长句录制时长)
                audio_data = record_audio(MAX_RECORD_SECONDS)
                
                # 语音转文字
                text = audio_to_text(audio_data)
                print(f"👤 你说：{text}")
                
                if not text.strip():
                    speak("抱歉，我没听清你说什么。")
                    continue
                
                # 调用OpenClaw
                response = call_openclaw(text, emotion)
                print(f"🤖 阿信：{response}")
                
                # 语音回复
                speak(response)
                
                print("\n✅ 回复完成，继续等待唤醒...")
                
    except KeyboardInterrupt:
        print("\n👋 退出程序")
    finally:
        audio_stream.close()
        pa.terminate()
        porcupine.delete()

if __name__ == "__main__":
    main()
