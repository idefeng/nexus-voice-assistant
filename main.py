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

# 日志净化 (v7.0.0)
def silence_stderr():
    f = open(os.devnull, 'w')
    os.dup2(f.fileno(), sys.stderr.fileno())

silence_stderr()

from config import *
from memory_manager import memory_manager
from scripts.context_helper import get_active_window_info, get_system_load
from ui_manager import init_ui_manager
from flet_ui import start_flet_ui

ui = init_ui_manager(UI_TYPE if ENABLE_UI else "none")

# 状态表情
STATUS_IDLE = "💤"
STATUS_LISTENING = "🎙️"
STATUS_THINKING = "🤔"
STATUS_SPEAKING = "🗣️"
STATUS_ACTION = "⚙️"
STATUS_LOCKED = "🔒"
STATUS_SLEEPING = "🌙"

NATURAL_FILLERS = ["嗯...", "我想想...", "好的，我明白了。", "让我想一下。"]

# 全局状态
is_sleeping = False
current_speaker_process = [None]
last_interaction_time = time.time()
proactive_cooldown = 120 
proactive_trigger_flag = threading.Event()
proactive_type = ["greeting"]
scheduled_reminders = [
    {"hour": 10, "minute": 30, "msg": "德哥，该喝水休息一下啦 🐻", "done": False},
    {"hour": 15, "minute": 0, "msg": "下午茶时间到！要不要起来动一动？☕", "done": False},
    {"hour": 23, "minute": 0, "msg": "太晚了，记得早点休息哦 🌕", "done": False},
]

# 加载主人特征
master_embedding = None
if os.path.exists(MASTER_FACE_EMBEDDING_PATH):
    print(f"🔐 已加载主人特征库: {MASTER_FACE_EMBEDDING_PATH}")
    master_embedding = np.load(MASTER_FACE_EMBEDDING_PATH)

# 相机锁 (v6.0.1)
camera_lock = threading.Lock()

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
    EMOTION_MAP = {
        'neutral': '平静', 'happy': '愉快', 'surprise': '惊讶', 
        'sad': '低落', 'angry': '愤怒', 'disgust': '厌恶', 
        'fear': '恐惧', 'contempt': '轻蔑'
    }

def calculate_ear(landmarks):
    """计算眼睛纵横比 (EAR)"""
    def dist(p1, p2): return np.linalg.norm(p1 - p2)
    # 左眼关键点 (36-41)
    l_v1 = dist(landmarks[37], landmarks[41])
    l_v2 = dist(landmarks[38], landmarks[40])
    l_h = dist(landmarks[36], landmarks[39])
    l_ear = (l_v1 + l_v2) / (2.0 * l_h)
    
    # 右眼关键点 (42-47)
    r_v1 = dist(landmarks[43], landmarks[47])
    r_v2 = dist(landmarks[44], landmarks[46])
    r_h = dist(landmarks[42], landmarks[45])
    r_ear = (r_v1 + r_v2) / (2.0 * r_h)
    
    return (l_ear + r_ear) / 2.0

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
try:
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=CHANNELS,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )
except Exception as e:
    print(f"❌ 无法启动音频流: {e}")
    sys.exit(1)

def detect_face_and_emotion():
    """检测人脸并分析情绪，同时返回 embedding (带锁保护)"""
    with camera_lock:
        print("📸 [感知] 正在开启摄像头...")
        cap = cv2.VideoCapture(CAMERA_ID)
        time.sleep(0.1)
        ret, frame = cap.read()
        cap.release()
        if not ret: 
            print("⚠️ [感知] 摄像头读取失败")
            return None, None, None, False
        faces = face_analyzer.get(frame)
        if len(faces) == 0: 
            print("⚠️ [感知] 未检测到人脸")
            return None, None, None, False
        
        face = faces[0]
        face_img = cv2.resize(frame, (64, 64))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = face_img.astype(np.float32) / 255.0
        face_img = np.expand_dims(np.expand_dims(face_img, axis=0), axis=0)
        outputs = emotion_session.run(None, {'Input3': face_img})
        raw_emotion = emotion_labels[np.argmax(outputs[0])]
        emotion = EMOTION_MAP.get(raw_emotion, "未知")
        
        # 疲劳计算 (EAR)
        ear = calculate_ear(face.landmark_3d_68)
        is_tired = ear < 0.22  # 经验阈值
        
        print(f"😐 [感知] 状态识别：{emotion} | EAR: {ear:.2f} {'(疲劳)' if is_tired else ''}")
        return face, emotion, face.normed_embedding, is_tired

def is_authorized(current_embedding):
    """对比当前人脸与主人特征"""
    if master_embedding is None: return True # 如果没注册，默认开放
    sim = np.dot(master_embedding, current_embedding)
    print(f"🔍 身份认证相似度: {sim:.2f}")
    return sim > FACE_SIMILARITY_THRESHOLD

def proactive_intelligence_loop():
    global last_interaction_time
    while True:
        if is_sleeping:
            time.sleep(5)
            continue
        try:
            now = datetime.datetime.now()
            for r in scheduled_reminders:
                if r["hour"] == now.hour and r["minute"] == now.minute and not r["done"]:
                    speak(r["msg"]); r["done"] = True
                elif r["hour"] != now.hour: r["done"] = False
            
            if time.time() - last_interaction_time > proactive_cooldown:
                face, emo, emb, tired = detect_face_and_emotion()
                if face and is_authorized(emb):
                    if tired:
                        proactive_type[0] = "fatigue_care"
                        print("🥱 触发疲劳关怀 (主)")
                    elif random.random() < 0.3:
                        proactive_type[0] = "screen_insight"
                        print("👁️ 触发主动屏幕洞察 (主)")
                    else:
                        proactive_type[0] = "greeting"
                        print("👋 触发主动问候 (主)")
                    proactive_trigger_flag.set()
                    last_interaction_time = time.time()
            time.sleep(10)
        except: time.sleep(10)

def record_audio(max_duration, is_follow_up=False):
    frames = []
    CHUNK_SIZE = porcupine.frame_length
    SILENCE_THRESHOLD = 500
    SILENCE_DURATION = 0.8 if is_follow_up else 1.2
    
    # 特殊逻辑：如果是连续对话，起始静音判定需要极短
    limit = int(porcupine.sample_rate / CHUNK_SIZE * SILENCE_DURATION)
    total_timeout = int(porcupine.sample_rate / CHUNK_SIZE * (FOLLOW_UP_TIMEOUT if is_follow_up else max_duration))
    
    counter, recorded = 0, 0
    while recorded < total_timeout:
        try:
            data = audio_stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data); recorded += 1
            energy = np.abs(struct.unpack_from("h" * CHUNK_SIZE, data)).mean()
            if energy < SILENCE_THRESHOLD: counter += 1
            else: counter = 0
            
            # 连续对话特殊逻辑：如果一开始就长时间静音，提前结束
            if is_follow_up and recorded > int(porcupine.sample_rate / CHUNK_SIZE * 3.0) and counter == recorded:
                return None # 表示没有追问

            if recorded > int(porcupine.sample_rate / CHUNK_SIZE * 0.5) and counter > limit: break
        except Exception as e:
            print(f"⚠️ 录音读取中断: {e}")
            break
    return b''.join(frames)

def audio_to_text(audio_data):
    ui.title = STATUS_THINKING
    wf = f"/tmp/r_{random.randint(0,99)}.wav"
    import wave
    with wave.open(wf, 'wb') as f:
        f.setnchannels(CHANNELS); f.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        f.setframerate(porcupine.sample_rate); f.writeframes(audio_data)
    try:
        with torch.no_grad(): res = whisper_model.transcribe(wf, language="zh")
        os.remove(wf)
        text = res["text"].strip()
        return text
    except:
        if os.path.exists(wf): os.remove(wf)
        return ""

# --- 工具集定义 (v8.0.0) ---
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_sports_data",
            "description": "获取用户的运动健身数据（如跑量、最近活动等）",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["latest", "monthly"], "description": "报表类型"},
                    "month": {"type": "string", "description": "目标月份 (YYYY-MM)"}
                },
                "required": ["type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_health_data",
            "description": "获取用户的健康/睡眠质量数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": ["sleep", "heart_rate"]}
                },
                "required": ["category"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_todo_tasks",
            "description": "获取用户的待办事项或重要安排",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

def execute_tool(name, args):
    """本地工具执行环境 (v8.0.0)"""
    print(f"🛠️ [工具执行] {name}({args})")
    try:
        if name == "get_sports_data":
            t, m = args.get("type"), args.get("month", "")
            base = "http://localhost:8000/api/v1/agent/"
            path = "latest_activity" if t == "latest" else "monthly_report"
            r = requests.get(f"{base}{path}", params={"target_month": m}, timeout=10)
            return r.json().get("report", "暂时没能获取到运动详情。")
        elif name == "get_health_data":
            return "根据最近监测，你昨晚深度睡眠达标，建议今天继续保持规律作息。"
        elif name == "get_todo_tasks":
            return "你明天有以下安排：1. 10:00 研发会议；2. 下午周报提交；3. 晚上 5 公里慢跑。"
        return f"错误：未定义的工具 {name}"
    except Exception as e:
        return f"API 访问失败: {str(e)}"

def call_openclaw(text, emotion=None, image_path=None, history=[], is_proactive=False, p_mode="greeting", is_tired=False):
    ui.title = STATUS_THINKING
    ui.update_state({"transcription": text, "response": ""})
    try:
        mem = ""
        if not history and ENABLE_MEMORY and memory_manager:
            mem = memory_manager.query_memory(text)
        
        vibe = "俏皮、温馨、充满智慧、深度体贴"
        sys_p = (
            f"你叫小德 (Xiaode) 🐻。你是一个全能的私人智能助理。你的风格是：{vibe}。"
            "你可以通过工具直接访问用户的[运动]、[睡眠]和[待办]数据。"
            "当用户问到相关话题时，请【必须】先执行工具查询再汇报结果。"
        )
        
        if emotion == 'angry': sys_p += "\n【反馈】用户愤怒，请多包容。"
        elif emotion == 'sad': sys_p += "\n【反馈】用户低落，给予鼓励。"
        if is_tired or p_mode == "fatigue_care":
            sys_p += "\n【生理状态】检测到用户疲劳。务必显式提及并关怀。"
            
        sys_p += f"\n【实时场景】用户正在使用: {get_active_window_info()}"
        if mem: sys_p += f"\n【历史背景】{mem}"
        
        user_msg = []
        if image_path:
            b64 = base64.b64encode(open(image_path, "rb").read()).decode('utf-8')
            user_msg.append({"type": "text", "text": text})
            user_msg.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        
        msgs = [{"role": "system", "content": sys_p}] + history
        msgs.append({"role": "user", "content": user_msg if user_msg else text})
        
        for _ in range(5):
            r = requests.post(OPENCLAW_API_URL, json={
                "model": f"{OPENCLAW_CHANNEL}/{OPENCLAW_ACCOUNT}/{OPENCLAW_AGENT}",
                "messages": msgs, "tools": TOOLS_SCHEMA
            }, headers={"Authorization": f"Bearer {OPENCLAW_TOKEN}"}, timeout=60)
            
            if r.status_code != 200: return {"type": "error", "content": f"API 响应异常 ({r.status_code})"}
            
            msg = r.json()["choices"][0]["message"]
            msgs.append(msg)
            
            if not msg.get("tool_calls"):
                content = msg.get("content", "")
                ui.update_state({"response": content})
                return {"type": "text", "content": content}
            
            for call in msg["tool_calls"]:
                name, tid = call["function"]["name"], call["id"]
                args = json.loads(call["function"]["arguments"])
                result = execute_tool(name, args)
                msgs.append({"role": "tool", "tool_call_id": tid, "name": name, "content": str(result)})
        
        return {"type": "text", "content": "我的大脑想得太多了，稍微休息一下吧。"}
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
    except: pass

def speak(text, with_filler=False):
    if not text or not text.strip(): return
    ui.title = STATUS_SPEAKING
    if current_speaker_process[0] and current_speaker_process[0].poll() is None:
        current_speaker_process[0].terminate()
    if with_filler and random.random() < 0.3:
        asyncio.run(_stream_speak(random.choice(NATURAL_FILLERS)))
    import re
    segs = [s.strip() for s in re.split(r'([。！？\n])', text) if s.strip()]
    for seg in segs or [text]:
        clean = re.sub(r'#+\s*|[*_\-]{1,3}|[`>]|\[([^\]]+)\]\([^\)]+\)|[\U00010000-\U0010ffff]', '', seg)
        if clean.strip(): asyncio.run(_stream_speak(clean.strip()))
    ui.title = STATUS_IDLE

class VoiceAssistantApp(rumps.App):
    def __init__(self):
        super(VoiceAssistantApp, self).__init__("小德", title=STATUS_IDLE)
        self.menu = ["关于小德", "鉴权注册", "重启助手"]
    @rumps.clicked("关于小德")
    def about(self, _): rumps.alert("小德 v6.0.0\n认主关怀版 🐻🛡️💖")
    @rumps.clicked("鉴权注册")
    def register(self, _):
        print("🛠️ 开始执行人脸注册...")
        rumps.notification("小德", "启动注册", "请正对摄像头，识别中...")
        time.sleep(1.5)
        face, _, emb, _ = detect_face_and_emotion()
        if face is not None:
            np.save(MASTER_FACE_EMBEDDING_PATH, emb)
            global master_embedding
            master_embedding = emb
            rumps.alert("✅ 认主成功！小德已记住您的面容。")
            print("🔐 主人特征已更新。")
        else:
            rumps.alert("❌ 失败：未探测到人脸。请确保光线充足并正对摄像头。")
            print("⚠️ 注册失败：未探测到有效人脸。")
    @rumps.clicked("重启助手")
    def restart(self, _): os.execv(sys.executable, ['python'] + sys.argv)

def run_voice_assistant():
    global is_sleeping
    history = []
    print("✅ 小德 v7.3.1 (Hotfix) 已就绪...")
    threading.Thread(target=proactive_intelligence_loop, daemon=True).start()
    
    follow_up = False 
    
    try:
        while True:
            if proactive_trigger_flag.is_set():
                follow_up = False
                proactive_trigger_flag.clear()
                mode = proactive_type[0]
                i_p = "/tmp/p_s.png" if mode == "screen_insight" else None
                if i_p: 
                    subprocess.run(["screencapture", "-x", i_p])
                res = call_openclaw("打个招呼吧", is_proactive=True, p_mode=mode, image_path=i_p)
                if res["type"] == "text":
                    print(f"🐻 小德 (主动): {res['content']}")
                    speak(res["content"])
                if i_p and os.path.exists(i_p): os.remove(i_p)
                continue

            is_triggered = False
            if not follow_up:
                # 检查唤醒词
                try:
                    pcm_data = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
                except OSError as e:
                    if e.errno == -9981 or "Unknown Error" in str(e):
                        time.sleep(0.1); continue
                    raise e
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm_data)
                is_triggered = porcupine.process(pcm) >= 0
                
                # 如果是唤醒触发，解除休眠 (v7.3.0)
                if is_triggered and is_sleeping:
                    is_sleeping = False
                    print("🌅 小德已被唤醒，正在解除休眠...")
                    ui.title = STATUS_IDLE
            else:
                is_triggered = True

            if is_triggered:
                global last_interaction_time
                last_interaction_time = time.time()
                
                face_res = {"face": None, "emotion": "平静", "emb": None, "tired": False}
                def b_vision():
                    f, e, m, t = detect_face_and_emotion()
                    if f: face_res.update({"face": f, "emotion": e, "emb": m, "tired": t})
                vt = threading.Thread(target=b_vision); vt.start()
                
                if not follow_up:
                    speak("我在呢 🐻")
                
                ui.title = STATUS_LISTENING
                ui.update_state({"transcription": "正在倾听...", "response": ""})
                audio = record_audio(MAX_RECORD_SECONDS, is_follow_up=follow_up)
                if audio is None: 
                    follow_up = False; ui.title = STATUS_IDLE; continue
                
                text = audio_to_text(audio)
                if not text or not text.strip(): 
                    follow_up = False; ui.title = STATUS_IDLE; continue
                
                vt.join(timeout=1.5)
                # 鉴权逻辑
                if master_embedding is not None:
                    if face_res["emb"] is None or not is_authorized(face_res["emb"]):
                        print("🚫 身份验证未通过，拒绝交互。")
                        ui.title = STATUS_LOCKED
                        follow_up = False; continue
                
                print(f"👤 你说：{text}")
                
                # 休眠指令识别 (v7.3.0)
                sleep_keywords = ["待机", "退出", "你先等等", "睡觉吧", "休眠", "退下"]
                if any(k in text for k in sleep_keywords):
                    is_sleeping = True
                    ui.title = STATUS_SLEEPING
                    follow_up = False
                    print("🌙 小德进入休眠状态。静默感知已开启。")
                    continue

                i_p = "/tmp/s.png" if any(k in text for k in ["看下屏幕", "内容", "分析"]) else None
                if i_p: 
                    print("📸 [动作] 正在获取屏幕快照...")
                    subprocess.run(["screencapture", "-x", i_p])
                
                res = call_openclaw(text, face_res['emotion'], i_p, history, is_tired=face_res['tired'])
                if res["type"] == "text":
                    print(f"🐻 小德：{res['content']}")
                    speak(res["content"], with_filler=True)
                    history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": res["content"]}])
                    follow_up = True
                    ui.title = "👂"
                else:
                    follow_up = False
                
                if i_p and os.path.exists(i_p): os.remove(i_p)
                if ENABLE_MEMORY and memory_manager:
                    memory_manager.save_emotion(face_res['emotion'])
                    threading.Thread(target=memory_manager.extract_and_save_facts, args=(text, str(res)), daemon=True).start()
                if len(history) > 10: history = history[-10:]
    finally:
        audio_stream.close(); pa.terminate(); porcupine.delete()

if __name__ == "__main__":
    if ENABLE_UI:
        if UI_TYPE == "flet":
            print("🚀 正在启动 Flet UI...")
            threading.Thread(target=run_voice_assistant, daemon=True).start()
            start_flet_ui(ui.state_queue)
            sys.exit(0)
        elif UI_TYPE == "rumps":
            print("🚀 正在启动 Rumps UI...")
            app = VoiceAssistantApp()
            ui.set_ui_instance(app)
            threading.Thread(target=run_voice_assistant, daemon=True).start()
            app.run()
            sys.exit(0)

    print("🚀 正在启动纯命令行模式...")
    run_voice_assistant()
