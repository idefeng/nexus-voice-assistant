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
proactive_cooldown = 120 # 默认2分钟测试
proactive_trigger_flag = threading.Event()
proactive_type = ["greeting"] # 类型: greeting, screen_insight
scheduled_reminders = [
    {"hour": 10, "minute": 30, "msg": "德哥，该喝水休息一下啦 🐻", "done": False},
    {"hour": 15, "minute": 0, "msg": "下午茶时间到！要不要起来动一动？☕", "done": False},
    {"hour": 23, "minute": 0, "msg": "太晚了，记得早点休息哦 🌕", "done": False},
]

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
    print("📸 [感知] 正在开启摄像头...")
    cap = cv2.VideoCapture(CAMERA_ID)
    ret, frame = cap.read()
    cap.release()
    if not ret: 
        print("⚠️ [感知] 摄像头读取失败")
        return None, None
    faces = face_analyzer.get(frame)
    if len(faces) == 0: 
        print("⚠️ [感知] 未检测到人脸")
        return None, None
    print(f"👤 [感知] 检测到 {len(faces)} 张人脸")
    face_img = cv2.resize(frame, (64, 64))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = face_img.astype(np.float32) / 255.0
    face_img = np.expand_dims(np.expand_dims(face_img, axis=0), axis=0)
    outputs = emotion_session.run(None, {'Input3': face_img})
    emotion = emotion_labels[np.argmax(outputs[0])]
    print(f"😐 [感知] 情绪识别结果：{emotion}")
    return faces[0], emotion

def proactive_intelligence_loop():
    """(v5.0.3) 主动智能：视觉问候 + 屏幕洞察"""
    global last_interaction_time
    while True:
        try:
            now = datetime.datetime.now()
            
            # 1. 定时任务
            for r in scheduled_reminders:
                if r["hour"] == now.hour and r["minute"] == now.minute and not r["done"]:
                    speak(r["msg"]); r["done"] = True
                elif r["hour"] != now.hour: r["done"] = False
            
            # 2. 状态感知
            if time.time() - last_interaction_time > proactive_cooldown:
                face, _ = detect_face_and_emotion()
                if face:
                    # 随机选择一种方案：问候 或 屏幕洞察 (v5.0.3)
                    if random.random() < 0.3:
                        print("👁️ 触发主动屏幕洞察...")
                        proactive_type[0] = "screen_insight"
                    else:
                        print("👋 触发主动问候...")
                        proactive_type[0] = "greeting"
                    
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

def call_openclaw(text, emotion=None, image_path=None, history=[], is_proactive=False, p_mode="greeting"):
    app.title = STATUS_THINKING
    try:
        mem = ""
        if not history and ENABLE_MEMORY and memory_manager:
            mem = memory_manager.query_memory(text)
        
        vibe = "俏皮、温馨"
        if emotion in ['sad', 'angry']: vibe = "极其温柔、体贴"
            
        sys_p = f"你叫小德 (Xiaode) 🐻。你是一只温暖的小熊，语气：{vibe}。"
        if is_proactive:
            if p_mode == "screen_insight":
                sys_p += " 【场景：主动屏幕洞察】你看到主人正在处理屏幕上的内容，请基于截图给出一条有价值的建议或关怀（比如代码优化建议、内容摘要或温馨提醒）。"
            else:
                sys_p += " 【场景：主动问候】发现主人出现在电脑前，打个亲切招呼。"
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
        return {"type": "error", "content": f"服务响应异常 ({r.status_code})"}
    except Exception as e: return {"type": "error", "content": str(e)}

async def _stream_speak(text):
    if not text or not text.strip(): return
    try:
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        proc = subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0"], stdin=subprocess.PIPE)
        current_speaker_process[0] = proc
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                try: proc.stdin.write(chunk["data"])
                except: break
        if proc.stdin: proc.stdin.close()
        proc.wait(); current_speaker_process[0] = None
    except Exception as e:
        print(f"⚠️ TTS Streaming Error: {e}")

def speak(text, with_filler=False):
    if not text or not text.strip(): return
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
    def about(self, _): rumps.alert("小德 v5.0.4\n全能感知版 🐻✨🎯")
    @rumps.clicked("重启助手")
    def restart(self, _): os.execv(sys.executable, ['python'] + sys.argv)

def run_voice_assistant():
    history = []
    print("✅ 小德 v5.0.4 已就绪...")
    threading.Thread(target=proactive_intelligence_loop, daemon=True).start()
    
    try:
        while True:
            if proactive_trigger_flag.is_set():
                proactive_trigger_flag.clear()
                mode = proactive_type[0]
                i_p = None
                if mode == "screen_insight":
                    i_p = "/tmp/p_s.png"; subprocess.run(["screencapture", "-x", i_p])
                
                res = call_openclaw("打个招呼或给出屏幕见解吧", is_proactive=True, p_mode=mode, image_path=i_p)
                if res["type"] == "text": speak(res["content"])
                if i_p and os.path.exists(i_p): os.remove(i_p)
                continue

            pcm_data = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm_data)
            if porcupine.process(pcm) >= 0:
                global last_interaction_time
                last_interaction_time = time.time()
                
                # 并行人脸与情绪识别 (v5.0.4 恢复)
                face_data = {"face": None, "emotion": "neutral"}
                def b_vision():
                    f, e = detect_face_and_emotion()
                    if f: face_data.update({"face": f, "emotion": e})
                vision_thread = threading.Thread(target=b_vision)
                vision_thread.start()
                
                speak("我在呢 🐻")
                app.title = STATUS_LISTENING
                audio = record_audio(MAX_RECORD_SECONDS)
                text = audio_to_text(audio)
                if not text.strip(): continue
                
                vision_thread.join(timeout=1.5) # 稍微多给点时间
                print(f"👤 [最终状态] 情绪：{face_data['emotion']}")
                print(f"👤 你说：{text}")
                
                i_p = "/tmp/s.png" if any(k in text for k in ["看下屏幕", "内容", "分析"]) else None
                if i_p: subprocess.run(["screencapture", "-x", i_p])
                
                res = call_openclaw(text, face_data['emotion'], i_p, history)
                if res["type"] == "text":
                    speak(res["content"], with_filler=True)
                    history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": res["content"]}])
                elif res["type"] == "tool_call":
                    app.title = STATUS_ACTION
                    speak(f"好的，我在帮您执行：{', '.join([c['function']['name'] for c in res['calls']])}")
                elif res["type"] == "error":
                    speak("抱歉，我的大脑暂时开小差了。")
                
                if i_p and os.path.exists(i_p): os.remove(i_p)
                if ENABLE_MEMORY: threading.Thread(target=memory_manager.extract_and_save_facts, args=(text, str(res)), daemon=True).start()
                if len(history) > 10: history = history[-10:]
    finally:
        audio_stream.close(); pa.terminate(); porcupine.delete()

if __name__ == "__main__":
    app = VoiceAssistantApp()
    threading.Thread(target=run_voice_assistant, daemon=True).start()
    app.run()
