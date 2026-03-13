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
import threading
import rumps
import random
import datetime
import torch
import warnings
import logging
import base64
import json
from onnxruntime import InferenceSession

# 净化日志：屏蔽冗余输出、FutureWarning 和下载进度条
logging.getLogger("insightface").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from config import *
from memory_manager import memory_manager

# 状态表情常量
STATUS_IDLE = "💤"
STATUS_LISTENING = "🎙️"
STATUS_THINKING = "🤔"
STATUS_SPEAKING = "🗣️"
STATUS_ACTION = "⚙️"

# 自然语气词
NATURAL_FILLERS = ["嗯...", "我想想...", "好的，我明白了。", "原来是这样。", "让我想一下。"]

# 全局变量：用于追踪当前正在播放（实现打断）
current_speaker_process = [None]

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
        keyword_paths=[WAKE_WORD_PATH],
        model_path=PORCUPINE_MODEL_PATH
    )
except Exception as e:
    print(f"⚠️ 无法加载自定义唤醒词: {e}")
    porcupine = pvporcupine.create(access_key=PORCUPINE_ACCESS_KEY, keywords=['picovoice'])

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
    if not ret: return None, None
    faces = face_analyzer.get(frame)
    if len(faces) == 0: return None, None
    face_img = cv2.resize(frame, (64, 64))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = face_img.astype(np.float32) / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=0)
    outputs = emotion_session.run(None, {'Input3': face_img})
    emotion_idx = np.argmax(outputs[0])
    return faces[0], emotion_labels[emotion_idx]

def record_audio(max_duration):
    """录制音频，支持静音检测"""
    frames = []
    CHUNK_SIZE = porcupine.frame_length
    SILENCE_THRESHOLD = 500
    SILENCE_DURATION = 1.5
    silence_frames_limit = int(porcupine.sample_rate / CHUNK_SIZE * SILENCE_DURATION)
    silence_counter = 0
    recorded_frames = 0
    max_frames = int(porcupine.sample_rate / CHUNK_SIZE * max_duration)
    while recorded_frames < max_frames:
        try:
            pcm_data = audio_stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(pcm_data)
            recorded_frames += 1
            pcm_unpacked = struct.unpack_from("h" * CHUNK_SIZE, pcm_data)
            energy = np.abs(pcm_unpacked).mean()
            if energy < SILENCE_THRESHOLD: silence_counter += 1
            else: silence_counter = 0
            if recorded_frames > int(porcupine.sample_rate / CHUNK_SIZE * 3) and silence_counter > silence_frames_limit:
                break
        except: break
    return b''.join(frames)

def audio_to_text(audio_data):
    """语音转文字"""
    app.title = STATUS_THINKING
    print("✍️  识别中...")
    import wave
    wf = wave.open("/tmp/recording.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
    wf.setframerate(porcupine.sample_rate)
    wf.writeframes(audio_data)
    wf.close()
    try:
        with torch.no_grad():
            result = whisper_model.transcribe("/tmp/recording.wav", language="zh")
        os.remove("/tmp/recording.wav")
        return result["text"]
    except:
        if os.path.exists("/tmp/recording.wav"): os.remove("/tmp/recording.wav")
        return ""

def get_time_greeting():
    """获取时间相关的问候语"""
    hour = datetime.datetime.now().hour
    if 5 <= hour < 12: return "早安，我是小德 🐻，很高兴见到你！"
    else: return "你好，我是小德 🐻，今天过得好吗？"

def capture_screen():
    """执行 macOS 静默截图"""
    output_path = "/tmp/screen.png"
    print("📸  正在捕捉屏幕内容...")
    try:
        subprocess.run(["screencapture", "-x", output_path], check=True)
        return output_path
    except: return None

def encode_image_base64(image_path):
    """图片转 Base64"""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except: return None

def call_openclaw(text, emotion=None, image_path=None, history=[]):
    """调用OpenClaw接口获取回复"""
    app.title = STATUS_THINKING
    try:
        memory_context = ""
        if not history and ENABLE_MEMORY and memory_manager:
            memory_context = memory_manager.query_memory(text)
        
        system_prompt = "你是一个名叫小德 (Xiaode) 的助手，形象是 🐻。你可以使用工具来执行命令（如打开应用、搜索、管理文件）。优先尝试解决问题，如果涉及危险操作请先询问用户。"
        if memory_context: system_prompt += f"\n这是关于用户的一些背景记忆：\n{memory_context}"
        
        user_content = []
        if image_path:
            b64 = encode_image_base64(image_path)
            if b64:
                user_content.append({"type": "text", "text": text})
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        
        user_msg = {"role": "user", "content": user_content if user_content else text}
        routing_id = f"{OPENCLAW_CHANNEL}/{OPENCLAW_ACCOUNT}/{OPENCLAW_AGENT}"
        if not OPENCLAW_CHANNEL or not OPENCLAW_ACCOUNT: routing_id = SESSION_KEY
            
        payload = {
            "model": routing_id,
            "messages": [{"role": "system", "content": system_prompt}] + history + [user_msg],
            "tools": "auto"
        }
        headers = {"Authorization": f"Bearer {OPENCLAW_TOKEN}", "Content-Type": "application/json"}
        response = requests.post(OPENCLAW_API_URL, json=payload, headers=headers, timeout=120)
        if response.status_code == 200:
            choice = response.json()["choices"][0]["message"]
            if "tool_calls" in choice:
                return {"type": "tool_call", "calls": choice["tool_calls"], "raw_message": choice}
            return {"type": "text", "content": choice.get("content", "")}
        return {"type": "error", "content": "抱歉，无法回复。"}
    except Exception as e:
        print(f"❌ 调用失败: {e}")
        return {"type": "error", "content": "发生未知错误。"}

def clean_text_for_tts(text):
    """清理播报文本"""
    import re
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'[*_\-]{1,3}', '', text)
    text = re.sub(r'[`>]', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    text = re.sub(r'[^\u0000-\u05C0\u2100-\u214F\u4E00-\u9FFF\u3040-\u30FF\uff00-\uffef\s]', '', text) 
    return re.sub(r'\s+', ' ', text).strip()

async def _edge_speak(text):
    """播报音频"""
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    temp_file = f"/tmp/reply_{random.randint(1000, 9999)}.mp3"
    await communicate.save(temp_file)
    proc = subprocess.Popen(["afplay", temp_file])
    current_speaker_process[0] = proc
    proc.wait()
    current_speaker_process[0] = None
    if os.path.exists(temp_file): os.remove(temp_file)

def stop_speaking():
    """停止播报"""
    if current_speaker_process[0] and current_speaker_process[0].poll() is None:
        print("🛑 打断。")
        current_speaker_process[0].terminate()
        current_speaker_process[0] = None

def speak(text, with_filler=False):
    """流利播报"""
    if not text: return
    app.title = STATUS_SPEAKING
    stop_speaking()
    if with_filler and random.random() < 0.4:
        asyncio.run(_edge_speak(random.choice(NATURAL_FILLERS)))
    print(f"🤖 小德：{text}")
    import re
    segments = re.split(r'([。！？\n])', text)
    final_segments = []
    for i in range(0, len(segments)-1, 2):
        item = segments[i] + segments[i+1]
        if len(item.strip()) > 1: final_segments.append(item.strip())
    if len(segments) % 2 == 1:
        last = segments[-1].strip()
        if len(last) > 1: final_segments.append(last)
    for seg in final_segments or [text]:
        clean_seg = clean_text_for_tts(seg)
        if clean_seg: asyncio.run(_edge_speak(clean_seg))
    app.title = STATUS_IDLE

class VoiceAssistantApp(rumps.App):
    def __init__(self):
        super(VoiceAssistantApp, self).__init__("小德", title=STATUS_IDLE)
        self.menu = ["关于小德", "重启助手"]
    @rumps.clicked("关于小德")
    def about(self, _):
        rumps.alert("小德 v3.0.1\n您的温馨数字副官 🐻⚙️")
    @rumps.clicked("重启助手")
    def restart(self, _):
        os.execv(sys.executable, ['python'] + sys.argv)

def run_voice_assistant():
    """主逻辑"""
    history = []
    print("✅ 小德已就绪...")
    app.title = STATUS_IDLE
    try:
        while True:
            pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            if porcupine.process(pcm) >= 0:
                print("\n🎉 唤醒成功！")
                stop_speaking()
                e_res = [None]
                def b_vision():
                    try:
                        if ENABLE_FACE_RECOGNITION and ENABLE_EMOTION_ANALYSIS:
                            _, em = detect_face_and_emotion(); e_res[0] = em
                    except: pass
                threading.Thread(target=b_vision).start()
                speak("小德在呢 🐻")
                app.title = STATUS_LISTENING
                audio_data = record_audio(MAX_RECORD_SECONDS)
                text = audio_to_text(audio_data)
                if not text.strip(): continue
                print(f"👤 你说：{text}")
                i_path = capture_screen() if any(k in text for k in ["看下屏幕", "这是什么", "分析内容"]) else None
                if i_path: speak("好的，我这只小熊 🐻 帮你瞅瞅。")
                res = call_openclaw(text, e_res[0], i_path, history)
                if res["type"] == "text":
                    speak(res["content"], with_filler=True)
                    history.append({"role": "user", "content": text})
                    history.append({"role": "assistant", "content": res["content"]})
                elif res["type"] == "tool_call":
                    app.title = STATUS_ACTION
                    action_desc = f"我正准备帮您执行：{', '.join([c['function']['name'] for c in res['calls']])}，正在处理..."
                    speak(action_desc)
                if i_path and os.path.exists(i_path): os.remove(i_path)
                if ENABLE_MEMORY and memory_manager:
                    threading.Thread(target=memory_manager.extract_and_save_facts, args=(text, str(res)), daemon=True).start()
                app.title = STATUS_IDLE
                if len(history) > 10: history = history[-10:]
    except Exception as e:
        print(f"❌ 运行错误: {e}")
    finally:
        audio_stream.close()
        pa.terminate()
        porcupine.delete()

if __name__ == "__main__":
    app = VoiceAssistantApp()
    threading.Thread(target=run_voice_assistant, daemon=True).start()
    app.run()
