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
import time
import torch
import warnings
import logging
import base64
import json
from onnxruntime import InferenceSession

# 净化日志
logging.getLogger("insightface").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from config import *
from memory_manager import memory_manager

# 状态表情
STATUS_IDLE = "💤"
STATUS_LISTENING = "🎙️"
STATUS_THINKING = "🤔"
STATUS_SPEAKING = "🗣️"
STATUS_ACTION = "⚙️"

NATURAL_FILLERS = ["嗯...", "我想想...", "好的，我明白了。", "让我想一下。"]

# 全局状态
current_speaker_process = [None]
last_interaction_time = time.time()
proactive_cooldown = 120 # 用户自定义为2分钟测试
proactive_trigger_flag = threading.Event()

# 初始化模型
print(f"🔊 加载语音识别 ({WHISPER_MODEL})...")
whisper_model = whisper.load_model(WHISPER_MODEL)

if ENABLE_FACE_RECOGNITION:
    print("👤 加载人脸识别...")
    face_analyzer = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

if ENABLE_EMOTION_ANALYSIS:
    print("😐 加载情绪分析...")
    emotion_session = InferenceSession("./models/emotion.onnx")
    emotion_labels = ['neutral', 'happy', 'surprise', 'sad', 'angry', 'disgust', 'fear', 'contempt']

print("🔔 初始化唤醒引擎...")
try:
    porcupine = pvporcupine.create(
        access_key=PORCUPINE_ACCESS_KEY,
        keyword_paths=[WAKE_WORD_PATH],
        model_path=PORCUPINE_MODEL_PATH
    )
except:
    porcupine = pvporcupine.create(access_key=PORCUPINE_ACCESS_KEY, keywords=['picovoice'])

pa = pyaudio.PyAudio()
audio_stream = pa.open(
    rate=porcupine.sample_rate,
    channels=CHANNELS,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length
)

def detect_face_and_emotion():
    cap = cv2.VideoCapture(CAMERA_ID)
    ret, frame = cap.read()
    cap.release()
    if not ret: return None, None
    faces = face_analyzer.get(frame)
    if len(faces) == 0: return None, None
    face_img = cv2.resize(frame, (64, 64))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = face_img.astype(np.float32) / 255.0
    face_img = np.expand_dims(np.expand_dims(face_img, axis=0), axis=0)
    outputs = emotion_session.run(None, {'Input3': face_img})
    return faces[0], emotion_labels[np.argmax(outputs[0])]

def proactive_vision_loop():
    """(v5.0.0) 后台视觉轮询：主动发现主人"""
    global last_interaction_time
    while True:
        try:
            if time.time() - last_interaction_time > proactive_cooldown:
                face, _ = detect_face_and_emotion()
                if face:
                    print("👀 发现主人出现，准备主动问候...")
                    proactive_trigger_flag.set()
                    last_interaction_time = time.time()
            time.sleep(10)
        except: time.sleep(10)

def record_audio(max_duration):
    frames = []
    CHUNK_SIZE = porcupine.frame_length
    SILENCE_THRESHOLD, SILENCE_DURATION = 500, 1.0
    limit = int(porcupine.sample_rate / CHUNK_SIZE * SILENCE_DURATION)
    counter, recorded = 0, 0
    max_f = int(porcupine.sample_rate / CHUNK_SIZE * max_duration)
    while recorded < max_f:
        try:
            data = audio_stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data); recorded += 1
            energy = np.abs(struct.unpack_from("h" * CHUNK_SIZE, data)).mean()
            if energy < SILENCE_THRESHOLD: counter += 1
            else: counter = 0
            if recorded > int(porcupine.sample_rate / CHUNK_SIZE * 0.5) and counter > limit: break
        except: break
    return b''.join(frames)

def audio_to_text(audio_data):
    app.title = STATUS_THINKING
    wf = f"/tmp/r_{random.randint(0,99)}.wav"
    import wave
    with wave.open(wf, 'wb') as f:
        f.setnchannels(CHANNELS); f.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        f.setframerate(porcupine.sample_rate); f.writeframes(audio_data)
    try:
        with torch.no_grad(): res = whisper_model.transcribe(wf, language="zh")
        os.remove(wf); return res["text"]
    except:
        if os.path.exists(wf): os.remove(wf)
        return ""

def call_openclaw(text, emotion=None, image_path=None, history=[], is_proactive=False):
    app.title = STATUS_THINKING
    try:
        mem = ""
        if not history and ENABLE_MEMORY and memory_manager:
            mem = memory_manager.query_memory(text)
        
        sys_p = "你叫小德 (Xiaode) 🐻。你现在不仅能听命，还能主动感知用户并反馈。语气亲切、带点俏皮小熊感。"
        if is_proactive: sys_p += " 【这是一个主动问候场景】你刚通过摄像头看到主人出现在电脑前，请根据记忆和时间打个温馨招呼。"
        if mem: sys_p += f"\n背景记忆：{mem}"
        
        user_c = []
        if image_path:
            b64 = base64.b64encode(open(image_path, "rb").read()).decode('utf-8')
            user_c.append({"type": "text", "text": text})
            user_c.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        
        history_msgs = [{"role": "system", "content": sys_p}] + history
        history_msgs.append({"role": "user", "content": user_c if user_c else text})
        
        r = requests.post(OPENCLAW_API_URL, json={
            "model": f"{OPENCLAW_CHANNEL}/{OPENCLAW_ACCOUNT}/{OPENCLAW_AGENT}",
            "messages": history_msgs, "tools": "auto"
        }, headers={"Authorization": f"Bearer {OPENCLAW_TOKEN}"}, timeout=120)
        
        if r.status_code == 200:
            msg = r.json()["choices"][0]["message"]
            if "tool_calls" in msg: return {"type": "tool_call", "calls": msg["tool_calls"]}
            return {"type": "text", "content": msg.get("content", "")}
        return {"type": "text", "content": "抱歉，出了一点小状况。"}
    except Exception as e: return {"type": "error", "content": str(e)}

async def _stream_speak(text):
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    proc = subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0"], stdin=subprocess.PIPE)
    current_speaker_process[0] = proc
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            try: proc.stdin.write(chunk["data"])
            except: break
    if proc.stdin: proc.stdin.close()
    proc.wait(); current_speaker_process[0] = None

def speak(text, with_filler=False):
    if not text: return
    app.title = STATUS_SPEAKING
    if current_speaker_process[0] and current_speaker_process[0].poll() is None:
        current_speaker_process[0].terminate()
    if with_filler and random.random() < 0.3:
        asyncio.run(_stream_speak(random.choice(NATURAL_FILLERS)))
    import re
    segs = [s.strip() for s in re.split(r'([。！？\n])', text) if s.strip()]
    for seg in segs or [text]:
        clean = re.sub(r'#+\s*|[*_\-]{1,3}|[`>]|\[([^\]]+)\]\([^\)]+\)|[\U00010000-\U0010ffff]', '', seg)
        if clean.strip(): asyncio.run(_stream_speak(clean.strip()))
    app.title = STATUS_IDLE

class VoiceAssistantApp(rumps.App):
    def __init__(self):
        super(VoiceAssistantApp, self).__init__("小德", title=STATUS_IDLE)
        self.menu = ["关于小德", "重启助手"]
    @rumps.clicked("关于小德")
    def about(self, _): rumps.alert("小德 v5.0.1\n主动智能版 🐻✨")
    @rumps.clicked("重启助手")
    def restart(self, _): os.execv(sys.executable, ['python'] + sys.argv)

def run_voice_assistant():
    history = []
    print("✅ 小德 v5.0.1 已就绪...")
    threading.Thread(target=proactive_vision_loop, daemon=True).start()
    
    try:
        while True:
            if proactive_trigger_flag.is_set():
                proactive_trigger_flag.clear()
                res = call_openclaw("看到我了吗？打个招呼吧", is_proactive=True)
                if res["type"] == "text": speak(res["content"])
                continue

            pcm_data = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm_data)
            if porcupine.process(pcm) >= 0:
                global last_interaction_time
                last_interaction_time = time.time()
                speak("我在呢 🐻")
                app.title = STATUS_LISTENING
                audio = record_audio(MAX_RECORD_SECONDS)
                text = audio_to_text(audio)
                if not text.strip(): continue
                print(f"👤 你说：{text}")
                
                i_p = "/tmp/s.png" if any(k in text for k in ["看下屏幕", "内容", "分析"]) else None
                if i_p: subprocess.run(["screencapture", "-x", i_p])
                
                res = call_openclaw(text, None, i_p, history)
                if res["type"] == "text":
                    speak(res["content"], with_filler=True)
                    history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": res["content"]}])
                elif res["type"] == "tool_call":
                    app.title = STATUS_ACTION
                    speak(f"好的，我在帮您执行：{', '.join([c['function']['name'] for c in res['calls']])}")
                
                if i_p and os.path.exists(i_p): os.remove(i_p)
                if ENABLE_MEMORY: threading.Thread(target=memory_manager.extract_and_save_facts, args=(text, str(res)), daemon=True).start()
                if len(history) > 10: history = history[-10:]
    finally:
        audio_stream.close(); pa.terminate(); porcupine.delete()

if __name__ == "__main__":
    app = VoiceAssistantApp()
    threading.Thread(target=run_voice_assistant, daemon=True).start()
    app.run()
