import os
import sys
import asyncio
import threading
import logging
import cv2
import numpy as np
import time
import subprocess
import pvporcupine
import pyaudio
import struct
from config import *
from engine.perception import perception_engine
from engine.audio import audio_engine
from engine.brain import brain_engine
from engine.proactive import proactive_engine
from pynput import keyboard
from memory_manager import memory_manager
from ui_manager import init_ui_manager
from flet_ui import start_flet_ui

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("Xiaode")

class AssistantState:
    def __init__(self):
        self.is_sleeping = False
        self.is_running = True
        self.camera_frame = None
        self.camera_lock = threading.Lock()
        self.master_embedding = None
        self.history = []

    def load_master(self):
        if os.path.exists(MASTER_FACE_EMBEDDING_PATH):
            try:
                self.master_embedding = np.load(MASTER_FACE_EMBEDDING_PATH)
                logger.info(f"🔐 已加载主人特征库: {MASTER_FACE_EMBEDDING_PATH}")
            except Exception as e:
                logger.error(f"加载主人特征失败: {e}")

state = AssistantState()
ui = init_ui_manager(UI_TYPE if ENABLE_UI else "none")

# --- 后台任务 ---
async def camera_loop():
    logger.info("📸 [系统] 启动后台相机线程...")
    cap = None
    while state.is_running:
        try:
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(CAMERA_ID)
                await asyncio.sleep(1)
            
            ret, frame = cap.read()
            if ret:
                with state.camera_lock:
                    state.camera_frame = frame.copy()
            else:
                cap.release(); cap = None
                await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"相机循环异常: {e}")
            await asyncio.sleep(5)
        await asyncio.sleep(0.1) # 10 FPS

async def proactive_loop():
    while state.is_running:
        if state.is_sleeping:
            await asyncio.sleep(5); continue
            
        try:
            # 1. 检查提醒
            await proactive_engine.check_reminders(audio_engine.speak_async)
            
            # 2. 检查主动触发环境
            with state.camera_lock:
                frame = state.camera_frame.copy() if state.camera_frame is not None else None
            
            if frame is not None:
                face, emo, emb, tired = perception_engine.analyze_frame(frame)
                is_auth = True
                if state.master_embedding is not None and emb is not None:
                    sim = np.dot(state.master_embedding, emb)
                    is_auth = sim > FACE_SIMILARITY_THRESHOLD
                
                if proactive_engine.should_trigger_proactive(face is not None, is_auth, tired):
                    proactive_engine.trigger_flag.set()
                    
        except Exception as e:
            logger.error(f"主动逻辑循环异常: {e}")
        await asyncio.sleep(10)

async def main_loop():
    state.load_master()
    follow_up = False
    
    # 启动后台任务
    asyncio.create_task(camera_loop())
    asyncio.create_task(proactive_loop())
    
    # 快捷键监听
    def on_activate():
        logger.info("⌨️ [快捷键] 收到手动唤醒指令")
        proactive_engine.manual_trigger_flag.set()

    hotkey = keyboard.GlobalHotKeys({
        '<cmd>+<shift>+z': on_activate  # 自定义快捷键
    })
    hotkey.start()
    
    # 初始化唤醒词
    porcupine = pvporcupine.create(
        access_key=PORCUPINE_ACCESS_KEY,
        keyword_paths=[WAKE_WORD_PATH],
        model_path=PORCUPINE_MODEL_PATH
    )
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=CHANNELS,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )
    
    logger.info("✅ 小德 v10.0.0 (Modular-Pro) 已就绪...")
    
    while state.is_running:
        try:
            # 1. 处理触发 (主动 或 手动)
            is_manual = proactive_engine.manual_trigger_flag.is_set()
            if proactive_engine.trigger_flag.is_set() or is_manual:
                follow_up = False
                mode = "manual" if is_manual else proactive_engine.trigger_type
                proactive_engine.trigger_flag.clear()
                proactive_engine.manual_trigger_flag.clear()
                
                if not is_manual: proactive_engine.mark_triggered()
                
                i_p = "/tmp/p_s.png" if mode in ["screen_insight", "manual"] else None
                if i_p: subprocess.run(["screencapture", "-x", i_p])
                
                res = await brain_engine.call_llm_async("小德想跟你打个招呼", is_proactive=True, p_mode=mode, image_path=i_p, update_ui_cb=ui.update_state)
                if res["type"] == "text":
                    await audio_engine.speak_async(res["content"], update_ui_title_cb=lambda t: setattr(ui, 'title', t))
                if i_p and os.path.exists(i_p): os.remove(i_p)
                continue

            # 2. 唤醒检测
            is_triggered = False
            if not follow_up:
                pcm_data = await asyncio.to_thread(audio_stream.read, porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm_data)
                is_triggered = porcupine.process(pcm) >= 0
                if is_triggered and state.is_sleeping:
                    state.is_sleeping = False
                    ui.title = "💤"
            else:
                is_triggered = True

            if is_triggered:
                # 视觉快照鉴权
                with state.camera_lock:
                    frame = state.camera_frame.copy() if state.camera_frame is not None else None
                face, emo, emb, tired = perception_engine.analyze_frame(frame)
                
                if state.master_embedding is not None:
                    is_auth = False
                    if emb is not None:
                        sim = np.dot(state.master_embedding, emb)
                        is_auth = sim > FACE_SIMILARITY_THRESHOLD
                    
                    if not is_auth:
                        logger.warning("🚫 鉴权未通过")
                        ui.title = "🔒"
                        await asyncio.sleep(2)
                        follow_up = False; continue

                if not follow_up: await audio_engine.speak_async("我在呢 🐻")
                
                ui.title = "🎙️"
                ui.update_state({"transcription": "正在倾听...", "response": ""})
                
                audio_data = await audio_engine.record_audio(audio_stream, porcupine.frame_length, porcupine.sample_rate, MAX_RECORD_SECONDS, is_follow_up=follow_up)
                
                if audio_data is None: 
                    follow_up = False; ui.title = "💤"; continue
                
                text = await audio_engine.audio_to_text(audio_data, porcupine.sample_rate, CHANNELS)
                if not text or not text.strip(): 
                    follow_up = False; ui.title = "💤"; continue
                
                if any(k in text for k in ["待机", "睡觉", "退下"]):
                    state.is_sleeping = True
                    ui.title = "🌙"
                    follow_up = False; continue

                i_p = "/tmp/s.png" if any(k in text for k in ["看下屏幕", "分析"]) else None
                if i_p: subprocess.run(["screencapture", "-x", i_p])
                
                res = await brain_engine.call_llm_async(text, emo, i_p, state.history, is_tired=tired, update_ui_cb=ui.update_state)
                
                if res["type"] == "text":
                    await audio_engine.speak_async(res["content"], with_filler=True, update_ui_title_cb=lambda t: setattr(ui, 'title', t))
                    state.history.append({"role": "user", "content": text})
                    state.history.append({"role": "assistant", "content": res["content"]})
                    follow_up = True
                    ui.title = "👂"
                else:
                    follow_up = False
                
                if i_p and os.path.exists(i_p): os.remove(i_p)
                if ENABLE_MEMORY and memory_manager:
                    asyncio.create_task(asyncio.to_thread(memory_manager.extract_and_save_facts, text, res.get("content", "")))
                
                if len(state.history) > 10: state.history = state.history[-10:]

            await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"主循环异常: {e}")
            await asyncio.sleep(1)

def start_assistant():
    asyncio.run(main_loop())

if __name__ == "__main__":
    if ENABLE_UI and UI_TYPE == "flet":
        threading.Thread(target=start_assistant, daemon=True).start()
        start_flet_ui(ui.state_queue)
    else:
        asyncio.run(main_loop())
