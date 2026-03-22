import os
import sys
import subprocess
import asyncio
import edge_tts
import pvporcupine
import pyaudio
import struct
import whisper
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
import httpx
from onnxruntime import InferenceSession
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# 加载根目录 .env
# main.py 在 apps/voice-assistant/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
import sys
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from packages.shared.models import LifeEvent
load_dotenv(os.path.join(ROOT_DIR, ".env"))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("Xiaode")

# 净化第三方库日志
logging.getLogger("insightface").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 日志净化 (v7.0.0) - 修复系统错误输出
# 已注释掉，避免干扰日志查看
# def silence_stderr():
#     try:
#         f = open(os.devnull, 'w')
#         os.dup2(f.fileno(), sys.stderr.fileno())
#     except Exception as e:
#         logger.warning(f"无法静默标准错误输出: {e}")
#
# silence_stderr()

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

# 全局状态管理
class AssistantState:
    def __init__(self):
        self.is_sleeping = False
        self.current_speaker_process = None
        self.last_interaction_time = time.time()
        self.proactive_cooldown = 120 
        self.proactive_trigger_flag = asyncio.Event()
        self.proactive_type = "greeting"
        self.scheduled_reminders = [
            {"hour": 10, "minute": 30, "msg": "德哥，该喝水休息一下啦 🐻", "done": False},
            {"hour": 15, "minute": 0, "msg": "下午茶时间到！要不要起来动一动？☕", "done": False},
            {"hour": 23, "minute": 0, "msg": "太晚了，记得早点休息哦 🌕", "done": False},
        ]
        self.master_embedding = None
        self.camera_frame = None
        self.camera_lock = threading.Lock()
        self.is_running = True

state = AssistantState()

# 加载主人特征
if os.path.exists(MASTER_FACE_EMBEDDING_PATH):
    try:
        state.master_embedding = np.load(MASTER_FACE_EMBEDDING_PATH)
        logger.info(f"🔐 已加载主人特征库: {MASTER_FACE_EMBEDDING_PATH}")
    except Exception as e:
        logger.error(f"加载主人特征失败: {e}")

# --- 资源加载 ---
logger.info(f"🔊 加载语音识别 ({WHISPER_MODEL})...")
whisper_model = whisper.load_model(WHISPER_MODEL)

face_analyzer = None
if ENABLE_FACE_RECOGNITION:
    logger.info("👤 加载人脸识别...")
    face_analyzer = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

emotion_session = None
emotion_labels = ['neutral', 'happy', 'surprise', 'sad', 'angry', 'disgust', 'fear', 'contempt']
EMOTION_MAP = {
    'neutral': '平静', 'happy': '愉快', 'surprise': '惊讶', 
    'sad': '低落', 'angry': '愤怒', 'disgust': '厌恶', 
    'fear': '恐惧', 'contempt': '轻蔑'
}
if ENABLE_EMOTION_ANALYSIS:
    logger.info("😐 加载情绪分析...")
    try:
        emotion_session = InferenceSession(EMOTION_MODEL_PATH)
    except Exception as e:
        logger.error(f"加载情绪分析模型失败: {e}")

# --- 背景相机类 ---
class BackgroundCamera:
    def __init__(self):
        self.cap = None
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        logger.info("📸 [系统] 启动后台相机线程...")
        while state.is_running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(CAMERA_ID)
                    time.sleep(1)
                
                ret, frame = self.cap.read()
                if ret:
                    with state.camera_lock:
                        state.camera_frame = frame.copy()
                else:
                    logger.warning("相机读取失败，尝试重新连接...")
                    self.cap.release()
                    self.cap = None
                    time.sleep(2)
            except Exception as e:
                logger.error(f"相机线程异常: {e}")
                time.sleep(5)
            time.sleep(0.1) # 10 FPS 足够了

bg_camera = BackgroundCamera()
if ENABLE_FACE_RECOGNITION:
    bg_camera.start()

# --- 视觉与感知函数 ---
def calculate_ear(landmarks):
    """计算眼睛纵横比 (EAR)"""
    def dist(p1, p2): return np.linalg.norm(p1 - p2)
    l_v1 = dist(landmarks[37], landmarks[41])
    l_v2 = dist(landmarks[38], landmarks[40])
    l_h = dist(landmarks[36], landmarks[39])
    l_ear = (l_v1 + l_v2) / (2.0 * l_h)
    
    r_v1 = dist(landmarks[43], landmarks[47])
    r_v2 = dist(landmarks[44], landmarks[46])
    r_h = dist(landmarks[42], landmarks[45])
    r_ear = (r_v1 + r_v2) / (2.0 * r_h)
    
    return (l_ear + r_ear) / 2.0

def detect_face_and_emotion():
    """从背景帧检测人脸并分析情绪"""
    with state.camera_lock:
        if state.camera_frame is None:
            return None, None, None, False
        frame = state.camera_frame.copy()

    try:
        faces = face_analyzer.get(frame)
        if len(faces) == 0: 
            ui.update_state({"emotion": ""})
            return None, None, None, False
        
        face = faces[0]
        emotion = "平静"
        if emotion_session:
            face_img = cv2.resize(frame, (64, 64))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_img = face_img.astype(np.float32) / 255.0
            face_img = np.expand_dims(np.expand_dims(face_img, axis=0), axis=0)
            outputs = emotion_session.run(None, {'Input3': face_img})
            raw_emotion = emotion_labels[np.argmax(outputs[0])]
            emotion = EMOTION_MAP.get(raw_emotion, "未知")
        
        ear = calculate_ear(face.landmark_3d_68)
        is_tired = ear < 0.22
        
        display_text = f"{emotion}" + (" (疲劳)" if is_tired else "")
        ui.update_state({"emotion": display_text})
        
        logger.info(f"😐 [感知] 状态识别：{emotion} | EAR: {ear:.2f} {'(疲劳)' if is_tired else ''}")
        return face, emotion, face.normed_embedding, is_tired
    except Exception as e:
        logger.error(f"人脸分析异常: {e}")
        return None, None, None, False

def is_authorized(current_embedding):
    """对比当前人脸与主人特征"""
    if state.master_embedding is None: return True
    sim = np.dot(state.master_embedding, current_embedding)
    logger.info(f"🔍 身份认证相似度: {sim:.2f}")
    return sim > FACE_SIMILARITY_THRESHOLD

# --- 音频与识别 ---
logger.info("🔔 初始化唤醒引擎...")
try:
    porcupine = pvporcupine.create(
        access_key=PORCUPINE_ACCESS_KEY,
        keyword_paths=[WAKE_WORD_PATH],
        model_path=PORCUPINE_MODEL_PATH
    )
except Exception as e:
    logger.warning(f"无法加载自定义唤醒词，尝试默认模式: {e}")
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
    logger.error(f"❌ 无法启动音频流: {e}")
    sys.exit(1)

async def record_audio(max_duration, is_follow_up=False):
    frames = []
    CHUNK_SIZE = porcupine.frame_length
    SILENCE_THRESHOLD = 500
    SILENCE_DURATION = 0.8 if is_follow_up else 1.2
    
    limit = int(porcupine.sample_rate / CHUNK_SIZE * SILENCE_DURATION)
    total_timeout = int(porcupine.sample_rate / CHUNK_SIZE * (FOLLOW_UP_TIMEOUT if is_follow_up else max_duration))
    
    counter, recorded = 0, 0
    while recorded < total_timeout:
        try:
            # 使用 to_thread 避免阻塞事件循环
            data = await asyncio.to_thread(audio_stream.read, CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data); recorded += 1
            energy = np.abs(struct.unpack_from("h" * CHUNK_SIZE, data)).mean()
            if energy < SILENCE_THRESHOLD: counter += 1
            else: counter = 0
            
            if is_follow_up and recorded > int(porcupine.sample_rate / CHUNK_SIZE * 3.0) and counter == recorded:
                return None 

            if recorded > int(porcupine.sample_rate / CHUNK_SIZE * 0.5) and counter > limit: break
        except Exception as e:
            logger.warning(f"⚠️ 录音读取中断: {e}")
            break
    return b''.join(frames)

async def audio_to_text(audio_data):
    ui.title = STATUS_THINKING
    wf = f"/tmp/r_{random.randint(0,99)}.wav"
    import wave
    try:
        def save_wav():
            with wave.open(wf, 'wb') as f:
                f.setnchannels(CHANNELS); f.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
                f.setframerate(porcupine.sample_rate); f.writeframes(audio_data)
        
        await asyncio.to_thread(save_wav)
        
        def transcribe():
            with torch.no_grad(): 
                return whisper_model.transcribe(wf, language="zh")
        
        res = await asyncio.to_thread(transcribe)
        return res["text"].strip()
    except Exception as e:
        logger.error(f"语音识别异常: {e}")
        return ""
    finally:
        if os.path.exists(wf): os.remove(wf)

# --- 工具与 AI 交互 ---
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_sports_data",
            "description": "获取用户的运动健身数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["latest", "monthly"]},
                    "month": {"type": "string"}
                },
                "required": ["type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_health_data",
            "description": "获取用户的健康/睡眠数据",
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
            "name": "get_daily_summary",
            "description": "获取用户人生系统的每日总结内容，作为背景知识",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

async def execute_tool_async(name, args):
    logger.info(f"🛠️ [工具执行] {name}({args})")
    ui.update_state({"status": STATUS_ACTION, "response": f"🛠️ 正在执行工具: {name}..."})
    try:
        async with httpx.AsyncClient() as client:
            if name == "get_sports_data":
                t, m = args.get("type"), args.get("month", "")
                base = "http://localhost:8000/api/v1/agent/"
                path = "latest_activity" if t == "latest" else "monthly_report"
                r = await client.get(f"{base}{path}", params={"target_month": m}, timeout=10)
                res = r.json().get("report", "暂时没能获取到运动详情。")
                ui.update_state({"status": STATUS_THINKING, "response": "✅ 成功获取数据，正在整理..."})
                return res
            elif name == "get_health_data":
                return "根据最近监测，你昨晚深度睡眠达标，建议今天继续保持规律作息。"
            elif name == "get_daily_summary":
                # 脑部组件路径: apps/voice-assistant -> apps -> Nexus-OS -> packages -> brain
                summary_path = os.path.join(ROOT_DIR, "packages/brain/storage/life/daily_summary.md")
                if os.path.exists(summary_path):
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        return f.read()
                return "未找到每日总结文件。"
            return f"错误：未定义的工具 {name}"
    except Exception as e:
        logger.error(f"工具执行失败: {e}")
        ui.update_state({"status": STATUS_THINKING, "response": f"❌ 工具执行失败: {str(e)}"})
        return f"API 访问失败: {str(e)}"

async def call_openclaw_async(text, emotion=None, image_path=None, history=[], is_proactive=False, p_mode="greeting", is_tired=False):
    ui.title = STATUS_THINKING
    ui.update_state({"transcription": text, "response": ""})
    try:
        mem = ""
        if not history and ENABLE_MEMORY and memory_manager:
            mem = await asyncio.to_thread(memory_manager.query_memory, text)
        
        sys_p = (
            f"你叫小德 (Xiaode) 🐻。你是一个智能助手。风格：俏皮、温馨、深度体贴。"
            f"\n【实时场景】用户正在使用: {get_active_window_info()}"
        )
        if mem: sys_p += f"\n【历史背景】{mem}"
        if emotion: sys_p += f"\n【当前情绪】{emotion}"
        if is_tired or p_mode == "fatigue_care": sys_p += "\n【生理状态】检测到用户疲劳，请给予关心。"
        
        user_msg = []
        if image_path:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
            user_msg = [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
        
        msgs = [{"role": "system", "content": sys_p}] + history
        msgs.append({"role": "user", "content": user_msg if user_msg else text})
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for _ in range(5):
                r = await client.post(OPENCLAW_API_URL, json={
                    "model": f"{OPENCLAW_CHANNEL}/{OPENCLAW_ACCOUNT}/{OPENCLAW_AGENT}",
                    "messages": msgs, "tools": TOOLS_SCHEMA
                }, headers={"Authorization": f"Bearer {OPENCLAW_TOKEN}"})
                
                if r.status_code != 200: 
                    return {"type": "error", "content": f"API 响应异常 ({r.status_code})"}
                
                msg = r.json()["choices"][0]["message"]
                msgs.append(msg)
                
                if not msg.get("tool_calls"):
                    content = msg.get("content", "")
                    ui.update_state({"response": content})
                    return {"type": "text", "content": content}
                
                for call in msg["tool_calls"]:
                    name = call["function"]["name"]
                    args = json.loads(call["function"]["arguments"])
                    result = await execute_tool_async(name, args)
                    msgs.append({"role": "tool", "tool_call_id": call["id"], "name": name, "content": str(result)})
        
        return {"type": "text", "content": "大脑过载了..."}
    except Exception as e:
        logger.error(f"AI 调用异常: {e}")
        return {"type": "error", "content": str(e)}

async def send_to_humansystems(text, fatigue_score=0.0):
    """
    将对话文本和疲劳值发送到 humansystems 后端 (已应用 LifeEvent 契约)
    """
    url = "http://localhost:8000/events/ingest"
    try:
        # 使用共享模型进行实例化，如有字段缺失 IDE 将在此处提示
        event = LifeEvent(
            content=text,
            fatigue_score=fatigue_score,
            source="mac-voice-assistant",
            sentiment=round(max(0.0, min(1.0, 1.0 - fatigue_score)), 2)
        )
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(url, json=event.model_dump())
            if r.status_code == 200:
                logger.info("✅ 数据已同步至 humansystems")
            else:
                logger.warning(f"⚠️ 数据同步失败: {r.status_code} - {r.text}")
    except Exception as e:
        logger.error(f"数据同步异常: {e}")

# --- TTS ---
async def _stream_speak_async(text):
    if not text or not text.strip(): return
    try:
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        proc = await asyncio.create_subprocess_exec(
            "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0",
            stdin=subprocess.PIPE
        )
        state.current_speaker_process = proc
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                try: 
                    proc.stdin.write(chunk["data"])
                    await proc.stdin.drain()
                except: break
        if proc.stdin: proc.stdin.close()
        await proc.wait()
        state.current_speaker_process = None
    except Exception as e:
        logger.warning(f"TTS 播放失败: {e}")

async def speak_async(text, with_filler=False):
    if not text or not text.strip(): return
    ui.title = STATUS_SPEAKING
    if state.current_speaker_process:
        try: state.current_speaker_process.terminate()
        except: pass
    
    if with_filler and random.random() < 0.3:
        await _stream_speak_async(random.choice(NATURAL_FILLERS))
    
    import re
    segs = [s.strip() for s in re.split(r'([。！？\n])', text) if s.strip()]
    for seg in segs or [text]:
        clean = re.sub(r'#+\s*|[*_\-]{1,3}|[`>]|\[([^\]]+)\]\([^\)]+\)|[\U00010000-\U0010ffff]', '', seg)
        if clean.strip(): await _stream_speak_async(clean.strip())
    ui.title = STATUS_IDLE

# --- 主逻辑循环 ---
async def proactive_loop():
    while state.is_running:
        if state.is_sleeping:
            await asyncio.sleep(5)
            continue
        try:
            now = datetime.datetime.now()
            for r in state.scheduled_reminders:
                if r["hour"] == now.hour and r["minute"] == now.minute and not r["done"]:
                    await speak_async(r["msg"])
                    r["done"] = True
                elif r["hour"] != now.hour: r["done"] = False
            
            if time.time() - state.last_interaction_time > state.proactive_cooldown:
                face, emo, emb, tired = detect_face_and_emotion()
                if face and is_authorized(emb):
                    if tired: state.proactive_type = "fatigue_care"
                    elif random.random() < 0.3: state.proactive_type = "screen_insight"
                    else: state.proactive_type = "greeting"
                    state.proactive_trigger_flag.set()
                    state.last_interaction_time = time.time()
        except Exception as e:
            logger.error(f"主动逻辑异常: {e}")
        await asyncio.sleep(10)

async def emotion_update_loop():
    """专门用于实时刷新 UI 上的情绪显示"""
    while state.is_running:
        if not state.is_sleeping:
            try:
                # 仅在非休眠状态下高频刷新情绪
                await asyncio.to_thread(detect_face_and_emotion)
            except Exception as e:
                logger.error(f"情绪更新循环异常: {e}")
        await asyncio.sleep(2.0)

async def main_loop():
    history = []
    follow_up = False 
    logger.info("✅ 小德 v9.1.1 (Dynamic-Emotion) 已就绪...")
    
    asyncio.create_task(proactive_loop())
    asyncio.create_task(emotion_update_loop())
    
    # 增加心跳任务 (每 2 分钟发送一次)，确保仪表盘显示 Online
    async def heartbeat_loop():
        while state.is_running:
            if not state.is_sleeping:
                try:
                    await send_to_humansystems(text="", fatigue_score=0.0)
                except: pass
            await asyncio.sleep(120) 
            
    asyncio.create_task(heartbeat_loop())

    while state.is_running:
        try:
            # 检查主动触发
            if state.proactive_trigger_flag.is_set():
                follow_up = False
                state.proactive_trigger_flag.clear()
                mode = state.proactive_type
                i_p = "/tmp/p_s.png" if mode == "screen_insight" else None
                if i_p: subprocess.run(["screencapture", "-x", i_p])
                res = await call_openclaw_async("小德想跟你打个招呼", is_proactive=True, p_mode=mode, image_path=i_p)
                if res["type"] == "text":
                    logger.info(f"🐻 小德 (主动): {res['content']}")
                    await speak_async(res["content"])
                if i_p and os.path.exists(i_p): os.remove(i_p)
                continue

            # 唤醒词检测
            is_triggered = False
            if not follow_up:
                pcm_data = await asyncio.to_thread(audio_stream.read, porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm_data)
                is_triggered = porcupine.process(pcm) >= 0
                if is_triggered and state.is_sleeping:
                    state.is_sleeping = False
                    logger.info("🌅 解除休眠...")
                    ui.title = STATUS_IDLE
            else:
                is_triggered = True

            if is_triggered:
                state.last_interaction_time = time.time()
                # 视觉感知并行
                face_info = await asyncio.to_thread(detect_face_and_emotion)
                face, emo, emb, tired = face_info
                
                # 鉴权
                if state.master_embedding is not None:
                    if emb is None or not is_authorized(emb):
                        logger.warning("🚫 鉴权未通过")
                        ui.title = STATUS_LOCKED
                        await asyncio.sleep(2)
                        follow_up = False; continue

                if not follow_up: await speak_async("我在呢 🐻")
                
                ui.title = STATUS_LISTENING
                ui.update_state({"transcription": "正在倾听...", "response": ""})
                audio = await record_audio(MAX_RECORD_SECONDS, is_follow_up=follow_up)
                if audio is None: 
                    follow_up = False; ui.title = STATUS_IDLE; continue
                
                text = await audio_to_text(audio)
                if not text or not text.strip(): 
                    follow_up = False; ui.title = STATUS_IDLE; continue
                
                # 休眠判定
                if any(k in text for k in ["待机", "睡觉", "退下"]):
                    state.is_sleeping = True
                    ui.title = STATUS_SLEEPING
                    follow_up = False; continue

                i_p = "/tmp/s.png" if any(k in text for k in ["看下屏幕", "分析"]) else None
                if i_p: subprocess.run(["screencapture", "-x", i_p])
                
                res = await call_openclaw_async(text, emo, i_p, history, is_tired=tired)
                if res["type"] == "text":
                    logger.info(f"🐻 小德：{res['content']}")
                    await speak_async(res["content"], with_filler=True)
                    history.append({"role": "user", "content": text})
                    history.append({"role": "assistant", "content": res["content"]})
                    # 同步到人生系统
                    asyncio.create_task(send_to_humansystems(text, fatigue_score=1.0 if tired else 0.0))
                    follow_up = True
                    ui.title = "👂"
                else:
                    follow_up = False
                
                if i_p and os.path.exists(i_p): os.remove(i_p)
                if ENABLE_MEMORY and memory_manager:
                    asyncio.create_task(asyncio.to_thread(memory_manager.extract_and_save_facts, text, res.get("content", "")))
                
                if len(history) > 10: history = history[-10:]
            
            await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"主循环异常: {e}")
            await asyncio.sleep(1)

# --- 启动器 ---
class VoiceAssistantApp(rumps.App):
    def __init__(self):
        super(VoiceAssistantApp, self).__init__("小德", title=STATUS_IDLE)
    @rumps.clicked("鉴权注册")
    def register(self, _):
        face, _, emb, _ = detect_face_and_emotion()
        if face:
            np.save(MASTER_FACE_EMBEDDING_PATH, emb)
            state.master_embedding = emb
            rumps.alert("✅ 认主成功！")
        else:
            rumps.alert("❌ 失败：未探测到人脸。")

def start_assistant():
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main_loop())
    except Exception as e:
        logger.critical(f"助手启动失败: {e}")

if __name__ == "__main__":
    try:
        if ENABLE_UI and UI_TYPE == "flet":
            threading.Thread(target=start_assistant, daemon=True).start()
            start_flet_ui(ui.state_queue)
        else:
            asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("👋 程序已被手动停止")
        sys.exit(0)
