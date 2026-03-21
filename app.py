import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["FUNASR_DISABLE_PROGRESS"] = "1"   # 禁用 funasr 推理进度条
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*trust_remote_code.*")
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

# --- 全局单实例锁 ---
PID_FILE = "/tmp/xiaode.pid"

# 先配置日志，以便锁逻辑使用
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("Xiaode")

def check_single_instance():
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, 'r') as f:
                content = f.read().strip()
                if not content: raise ValueError("PID file is empty")
                old_pid = int(content)
            # 检查进程是否仍然存活
            os.kill(old_pid, 0)
            print(f"⚠️  检测到程序已在运行 (PID: {old_pid})。请先关闭旧进程，或手动删除 {PID_FILE}。")
            sys.exit(1)
        except (ProcessLookupError, OSError, ValueError):
            # 进程不存在或文件内容有误，说明是残留锁，清理掉
            logger.info(f"♻️  清理残留的 PID 文件: {PID_FILE}")
            try:
                os.remove(PID_FILE)
            except: pass
    
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))

def cleanup_pid():
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)

import atexit
atexit.register(cleanup_pid)
check_single_instance()

# 已提前初始化

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

    try:
        # pynput 1.8+ GlobalHotKeys 存在回调签名兼容问题，使用 Listener 替代
        from pynput.keyboard import Key, Listener as KbListener
        _pressed_keys = set()
        def _on_press_hotkey(key):
            _pressed_keys.add(key)
            # 检测 Cmd+Shift+Z
            if (Key.cmd in _pressed_keys or Key.cmd_l in _pressed_keys or Key.cmd_r in _pressed_keys) and \
               (Key.shift in _pressed_keys or Key.shift_l in _pressed_keys or Key.shift_r in _pressed_keys):
                try:
                    if hasattr(key, 'char') and key.char == 'z':
                        on_activate()
                except AttributeError:
                    pass
        def _on_release_hotkey(key):
            _pressed_keys.discard(key)
        
        hotkey_listener = KbListener(on_press=_on_press_hotkey, on_release=_on_release_hotkey)
        hotkey_listener.daemon = True
        hotkey_listener.start()
        logger.info("⌨️ [系统] 手动唤醒快捷键已生效 (<cmd>+<shift>+z)")
    except Exception as e:
        logger.warning(f"⚠️ [系统] 快捷键监听启动失败 (可能由于 macOS 辅助功能权限未开启): {e}")
    
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
    
    print("\n" + "="*40)
    print("✅ 小德 v10.1.0 (感知革新版) 已就绪")
    print("👂 正在监听唤醒词 '小德'，请尝试呼唤他。")
    print("="*40 + "\n")
    logger.info("✅ 小德 v10.1.0 已就绪")
    
    last_heartbeat = time.time()
    
    while state.is_running:
        # 每隔 300 秒打印一个“正在监听”的心跳，让用户确认程序没挂
        if time.time() - last_heartbeat > 300:
            logger.info("💓 [系统] 小德正在守护中，随时准备被唤醒...")
            last_heartbeat = time.time()
            
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
                    logger.info(f"🤖 [小德 (主动)] {res['content']}")
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
                if not follow_up:
                    print("\n🔍 检测到唤醒词，正在进行人脸鉴权...")
                    logger.info("🔍 [系统] 检测到唤醒词，正在进行人脸鉴权...")
                
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
                        logger.warning("🚫 [鉴权失败] 未识别到主人人脸或人脸不匹配，已拒绝对话。")
                        ui.title = "🔒"
                        await asyncio.sleep(2)
                        follow_up = False; continue
                    else:
                        logger.info("✅ [鉴权通过] 已确认主人身份。")

                if not follow_up: await audio_engine.speak_async("我在呢 🐻")
                
                ui.title = "🎙️"
                ui.update_state({"transcription": "正在倾听...", "response": ""})
                
                print("🎤 正在倾听，请说话...")
                logger.info("🎤 [系统] 正在倾听，请说话...")
                audio_data = await audio_engine.record_audio(audio_stream, porcupine.frame_length, porcupine.sample_rate, MAX_RECORD_SECONDS, is_follow_up=follow_up)
                
                if audio_data is None: 
                    follow_up = False; ui.title = "💤"; continue
                
                text, voice_emotion = await audio_engine.audio_to_text(audio_data, porcupine.sample_rate, CHANNELS)
                if not text or not text.strip(): 
                    follow_up = False; ui.title = "💤"; continue
                
                # 语音情绪优先，视觉情绪作为辅助
                final_emotion = voice_emotion if voice_emotion != "平静" else emo
                
                print(f"\n{'='*20}\n🎤 [用户] {text} (情绪: {final_emotion})\n{'='*20}")
                logger.info(f"🎤 [用户] {text} (语音情绪: {voice_emotion}, 视觉情绪: {emo}, 最终: {final_emotion})")
                
                if any(k in text for k in ["待机", "睡觉", "退下"]):
                    state.is_sleeping = True
                    ui.title = "🌙"
                    follow_up = False; continue

                i_p = "/tmp/s.png" if any(k in text for k in ["看下屏幕", "分析"]) else None
                if i_p: subprocess.run(["screencapture", "-x", i_p])
                
                # 即时语音反馈 + LLM 调用并发执行（消除空白等待期）
                import random as _rnd
                thinking_phrases = [
                    "好的，让我想想。",
                    "收到，请稍等。",
                    "好的，我来帮你看看。",
                    "嗯，稍等一下哦。",
                    "我去查一查。",
                ]
                ui.title = "🤔"
                chosen_phrase = _rnd.choice(thinking_phrases)
                print(f"🤔 [思考中] {chosen_phrase}")
                thinking_task = asyncio.create_task(
                    audio_engine.speak_async(chosen_phrase)
                )
                llm_task = asyncio.create_task(
                    brain_engine.call_llm_async(text, final_emotion, i_p, state.history, is_tired=tired, update_ui_cb=ui.update_state)
                )
                # 等待两者都完成（语音反馈通常 <1秒，LLM 通常 2-10秒）
                await thinking_task
                res = await llm_task
                
                if res["type"] == "text":
                    print(f"\n🤖 [小德] {res['content']}")
                    print(f"{'-'*20}")
                    logger.info(f"🤖 [小德] {res['content']}")
                    was_interrupted = await audio_engine.speak_async(
                        res["content"], with_filler=True, 
                        update_ui_title_cb=lambda t: setattr(ui, 'title', t),
                        porcupine=porcupine, audio_stream=audio_stream
                    )
                    state.history.append({"role": "user", "content": text})
                    state.history.append({"role": "assistant", "content": res["content"]})
                    if was_interrupted:
                        # 播报被唤醒词中断 → 直接进入下一轮对话
                        logger.info("🛑 [系统] 播报被中断，直接进入倾听模式")
                        follow_up = True
                    else:
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
    try:
        if ENABLE_UI and UI_TYPE == "flet":
            threading.Thread(target=start_assistant, daemon=True).start()
            start_flet_ui(ui.state_queue)
        else:
            asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("👋 程序已被手动停止")
        sys.exit(0)
