import os
import asyncio
import edge_tts
import pyaudio
import struct
import torch
import wave
import random
import subprocess
import logging
import re
import numpy as np
import uuid
import json
import requests
from config import *

logger = logging.getLogger("Xiaode.Audio")

class AudioEngine:
    def __init__(self):
        # --- 加载 SenseVoice-Small (ASR + 情绪识别) ---
        logger.info(f"🔊 加载语音识别驱动 (SenseVoice-Small)...")
        from funasr import AutoModel
        self.asr_model = AutoModel(
            model=SENSEVOICE_MODEL,
            vad_model="fsmn-vad",       # FunASR 内置 VAD
            vad_kwargs={"max_single_segment_time": 30000},
            trust_remote_code=True,
            disable_update=True,        # 禁用版本检查，消除启动警告
        )
        logger.info("✅ SenseVoice-Small 加载完成")

        # --- 加载 Silero VAD (实时语音活动检测) ---
        logger.info("🎯 加载 Silero VAD...")
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True,
        )
        (self.get_speech_timestamps, _, self.read_audio, _, _) = self.vad_utils
        logger.info(f"✅ Silero VAD 加载完成 (threshold={VAD_THRESHOLD})")

        self.pa = pyaudio.PyAudio()
        self.current_speaker_process = None
        self.NATURAL_FILLERS = ["嗯...", "我想想...", "好的，我明白了。", "让我想一下。"]
        self.interrupt_event = asyncio.Event()  # 播报中断标志

        # SenseVoice 情绪映射
        self.EMOTION_MAP_SV = {
            "HAPPY": "愉快",
            "SAD": "低落",
            "ANGRY": "愤怒",
            "NEUTRAL": "平静",
        }

    def _silero_vad_check(self, audio_chunk_int16, sample_rate=16000):
        """使用 Silero VAD 检测音频块中是否包含语音
        
        Args:
            audio_chunk_int16: int16 格式的音频数据 (bytes 或 numpy array)
            sample_rate: 采样率
            
        Returns:
            float: 语音概率 (0-1)
        """
        try:
            if isinstance(audio_chunk_int16, bytes):
                audio_np = np.frombuffer(audio_chunk_int16, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_np = audio_chunk_int16.astype(np.float32) / 32768.0
            
            audio_tensor = torch.from_numpy(audio_np)
            
            # Silero VAD 需要 512 采样点 (16kHz 下约 32ms)
            if len(audio_tensor) < 512:
                return 0.0
            
            # 取前 512 个采样点
            chunk = audio_tensor[:512]
            speech_prob = self.vad_model(chunk, sample_rate).item()
            return speech_prob
        except Exception as e:
            logger.warning(f"⚠️ Silero VAD 检测异常: {e}")
            return 0.0

    async def record_audio(self, stream, frame_length, sample_rate, max_duration, is_follow_up=False):
        """录制音频数据 (使用 Silero VAD 进行语音检测)"""
        frames = []
        CHUNK_SIZE = frame_length
        
        # Silero VAD 配置
        vad_threshold = VAD_THRESHOLD
        silence_duration = VAD_MIN_SILENCE_MS / 1000.0 if is_follow_up else max(VAD_MIN_SILENCE_MS / 1000.0, 1.2)
        
        limit = int(sample_rate / CHUNK_SIZE * silence_duration)
        total_timeout = int(sample_rate / CHUNK_SIZE * (FOLLOW_UP_TIMEOUT if is_follow_up else max_duration))
        
        silence_counter, recorded = 0, 0
        has_speech = False  # 是否检测到过语音
        speech_frame_count = 0  # 语音帧计数
        
        # 重置 VAD 模型状态
        self.vad_model.reset_states()
        
        while recorded < total_timeout:
            try:
                data = await asyncio.to_thread(stream.read, CHUNK_SIZE, exception_on_overflow=False)
                frames.append(data)
                recorded += 1
                
                # 使用 Silero VAD 判断是否有语音
                speech_prob = self._silero_vad_check(data, sample_rate)
                
                if speech_prob >= vad_threshold:
                    silence_counter = 0
                    has_speech = True
                    speech_frame_count += 1
                else:
                    silence_counter += 1
                
                # 追问模式：如果长时间没有语音，退出
                if is_follow_up and recorded > int(sample_rate / CHUNK_SIZE * 3.0) and not has_speech:
                    return None 

                # 检测到语音后，如果静音超过阈值，结束录制
                if has_speech and silence_counter > limit:
                    break
                    
            except Exception as e:
                logger.warning(f"⚠️ 录音读取中断: {e}")
                break
        
        if not has_speech or speech_frame_count < 3:
            logger.info("🔇 未检测到有效语音")
            return None
            
        return b''.join(frames)

    def _parse_sensevoice_output(self, text):
        """解析 SenseVoice 输出，提取文字和情绪标签
        
        SenseVoice 输出格式: <|zh|><|NEUTRAL|><|Speech|><|woitn|>你好
        返回: (纯文字, 情绪中文名)
        """
        emotion = "平静"
        clean_text = text
        
        # 提取情绪标签
        import re
        emotion_match = re.search(r'<\|(HAPPY|SAD|ANGRY|NEUTRAL)\|>', text)
        if emotion_match:
            raw_emotion = emotion_match.group(1)
            emotion = self.EMOTION_MAP_SV.get(raw_emotion, "平静")
        
        # 清除所有特殊标签: <|...|>
        clean_text = re.sub(r'<\|[^|]*\|>', '', text).strip()
        
        return clean_text, emotion

    async def audio_to_text(self, audio_data, sample_rate, channels):
        """ASR: 语音转文字 + 情绪识别 (SenseVoice-Small)
        
        Returns:
            tuple: (识别文字, 情绪标签中文) 例如 ("你好", "愉快")
        """
        wf = f"/tmp/r_{random.randint(0,99)}.wav"
        try:
            def save_wav():
                with wave.open(wf, 'wb') as f:
                    f.setnchannels(channels)
                    f.setsampwidth(self.pa.get_sample_size(pyaudio.paInt16))
                    f.setframerate(sample_rate)
                    f.writeframes(audio_data)
            
            await asyncio.to_thread(save_wav)
            
            def transcribe():
                res = self.asr_model.generate(
                    input=wf,
                    cache={},
                    language="zh",
                    use_itn=True,
                    batch_size_s=0,
                )
                if res and len(res) > 0 and res[0].get("text"):
                    return res[0]["text"]
                return ""
            
            raw_text = await asyncio.to_thread(transcribe)
            if not raw_text:
                return "", "平静"
            
            text, emotion = self._parse_sensevoice_output(raw_text)
            logger.info(f"🎯 SenseVoice 识别: 文字=\"{text}\" 情绪={emotion} (原始: {raw_text})")
            return text, emotion
        except Exception as e:
            logger.error(f"语音识别异常: {e}")
            return "", "平静"
        finally:
            if os.path.exists(wf): os.remove(wf)

    async def _volcengine_speak_async(self, text):
        """Level 1: 火山引擎 (豆包) - 极致拟人 (V3 HTTP 异步流式)"""
        if not globals().get('VOLC_APPID') or not globals().get('VOLC_TOKEN'): 
            return False
            
        try:
            import base64
            import httpx
            
            url = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
            headers = {
                "X-Api-App-Key": globals().get('VOLC_APPID'),
                "X-Api-Access-Key": globals().get('VOLC_TOKEN'),
                "X-Api-Resource-Id": "seed-tts-2.0",
                "Content-Type": "application/json"
            }
            payload = {
                "user": {"uid": "xiaode_user", "device_id": "mac_client"},
                "req_params": {
                    "text": text,
                    "speaker": globals().get('VOLC_VOICE', 'BV700_V2_streaming'),
                    "audio_params": {
                        "format": "mp3", 
                        "sample_rate": 24000,
                        "speech_rate": 0,
                        "loudness_rate": 0
                    }
                }
            }
            
            proc = await asyncio.create_subprocess_exec(
                "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0",
                stdin=subprocess.PIPE
            )
            self.current_speaker_process = proc
            
            success = False
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    async with client.stream("POST", url, json=payload, headers=headers) as response:
                        if response.status_code != 200:
                            error_text = await response.aread()
                            logger.error(f"火山引擎 API 错误 (状态码 {response.status_code}): {error_text.decode('utf-8', errors='ignore')}")
                            return False
                        
                        async for line in response.aiter_lines():
                            if not line or not line.startswith("data:"): continue
                            try:
                                chunk = json.loads(line[5:])
                                if chunk.get("code") and chunk.get("code") != 0:
                                    logger.error(f"火山引擎 SSE 报错: {chunk}")
                                    break
                                
                                if "audio" in chunk and chunk["audio"]:
                                    audio_bytes = base64.b64decode(chunk["audio"])
                                    if audio_bytes:
                                        proc.stdin.write(audio_bytes)
                                        await proc.stdin.drain()
                                        success = True
                            except Exception as e:
                                logger.warning(f"解析火山引擎数据分片失败: {e}")
                                continue
            finally:
                if proc.stdin:
                    try:
                        proc.stdin.close()
                        await proc.stdin.wait_closed()
                    except: pass
                await proc.wait()
                self.current_speaker_process = None
                
            return success
        except Exception as e:
            logger.warning(f"火山引擎 TTS 调用失败: {e}")
            if self.current_speaker_process:
                try: self.current_speaker_process.terminate()
                except: pass
                self.current_speaker_process = None
            return False

    async def _edge_speak_async(self, text):
        """Level 2: Edge-TTS - 稳定高保真"""
        try:
            communicate = edge_tts.Communicate(text, TTS_VOICE)
            proc = await asyncio.create_subprocess_exec(
                "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0",
                stdin=subprocess.PIPE
            )
            self.current_speaker_process = proc
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    try:
                        proc.stdin.write(chunk["data"])
                        await proc.stdin.drain()
                    except: break
            if proc.stdin: proc.stdin.close()
            await proc.wait()
            self.current_speaker_process = None
            return True
        except Exception as e:
            logger.warning(f"Edge-TTS 播放失败: {e}")
            return False

    async def _local_speak_async(self, text):
        """Level 3: macOS say - 离线兜底"""
        try:
            logger.info("🔊 [系统] 正在尝试本地兜底播放...")
            proc = await asyncio.create_subprocess_exec("say", text)
            self.current_speaker_process = proc
            await proc.wait()
            self.current_speaker_process = None
            return True
        except Exception as e:
            logger.error(f"本地 TTS 彻底失败: {e}")
            return False

    async def _wake_word_monitor(self, porcupine, audio_stream):
        """在 TTS 播报期间并发监听唤醒词，检测到则设置中断标志"""
        try:
            while not self.interrupt_event.is_set():
                pcm_data = await asyncio.to_thread(
                    audio_stream.read, porcupine.frame_length, 
                    exception_on_overflow=False
                )
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm_data)
                if porcupine.process(pcm) >= 0:
                    logger.info("🛑 [中断] 播报期间检测到唤醒词，立即中断播报")
                    self.interrupt_event.set()
                    # 立即终止当前 ffplay 进程
                    if self.current_speaker_process:
                        try: self.current_speaker_process.terminate()
                        except: pass
                    return
                await asyncio.sleep(0.01)
        except Exception as e:
            logger.debug(f"唤醒词监听异常 (可忽略): {e}")

    async def speak_async(self, text, with_filler=False, update_ui_title_cb=None, 
                          porcupine=None, audio_stream=None):
        """TTS: 文字转语音 (支持唤醒词中断)
        
        Args:
            porcupine: Porcupine 唤醒词引擎实例 (传入则启用播报中断)
            audio_stream: PyAudio 输入流 (配合 porcupine 使用)
            
        Returns:
            bool: True=被唤醒词中断, False=正常播完
        """
        if not text or not text.strip(): return False
        
        if update_ui_title_cb: update_ui_title_cb("🗣️")
        
        if self.current_speaker_process:
            try: self.current_speaker_process.terminate()
            except: pass
        
        self.interrupt_event.clear()
        
        if with_filler and random.random() < 0.3:
            await self.speak_async(random.choice(self.NATURAL_FILLERS), with_filler=False)
        
        # 如果提供了 porcupine，启动并发唤醒词监听
        monitor_task = None
        if porcupine and audio_stream:
            monitor_task = asyncio.create_task(self._wake_word_monitor(porcupine, audio_stream))
        
        # 按句子分段，但将标点合并到前一个片段（避免单独标点导致 Edge-TTS 空音频）
        raw_segs = re.split(r'([。！？\n])', text)
        segs = []
        for i in range(0, len(raw_segs), 2):
            s = raw_segs[i]
            if i + 1 < len(raw_segs):
                s += raw_segs[i + 1]  # 将标点合并到前一句
            s = s.strip()
            if s:
                segs.append(s)
        
        for seg in segs or [text]:
            # 清理 Markdown 格式和所有 emoji（包括 BMP 平面的 ⏰⚡☀ 和补充平面的 😊🐻）
            clean = re.sub(
                r'#+\s*|[*_\-]{1,3}|[`>]|\[([^\]]+)\]\([^\)]+\)|'
                r'[\U00010000-\U0010ffff]|'           # 补充平面 emoji (😊🐻🎉等)
                r'[\u2600-\u27BF]|'                   # 杂项符号 (☀☁☂⚡等)
                r'[\u2300-\u23FF]|'                   # 技术符号 (⏰⏳⌚等)
                r'[\u2B50-\u2B55]|'                   # 星号等 (⭐⭕等)
                r'[\uFE00-\uFE0F]|'                   # 变体选择符
                r'[\u200D]|'                          # 零宽连接符
                r'[\u20E3]|'                          # 组合围绕符
                r'[\u2934-\u2935]|'                   # 箭头
                r'[\u25AA-\u25FE]|'                   # 几何图形
                r'[\u2702-\u27B0]|'                   # 装饰符号
                r'[\u00A9\u00AE]|'                    # ©®
                r'[\u203C\u2049]|'                    # ‼⁉
                r'[\u2139]|'                          # ℹ
                r'[\u2194-\u21AA]|'                   # 箭头
                r'[\u231A-\u231B]|'                   # ⌚⌛
                r'[\u25FB-\u25FE]|'                   # 方块
                r'[\u2614-\u2615]|'                   # ☔☕
                r'[\u2648-\u2653]|'                   # 星座符号
                r'[\u267F]|[\u2693]|[\u26A1]|'        # ♿⚓⚡
                r'[\u26AA-\u26AB]|'                   # ⚪⚫
                r'[\u26F0-\u26FA]|'                   # ⛰-⛺
                r'[\u2702]|[\u2708]|[\u270A-\u270D]', # ✂✈✊-✍
                '', seg)

            if not clean.strip() or len(clean.strip()) < 2: continue
            
            # 三级补位逻辑
            success = await self._volcengine_speak_async(clean)
            if not success:
                success = await self._edge_speak_async(clean)
            if not success:
                await self._local_speak_async(clean)
            
            # 检查是否被中断
            if self.interrupt_event.is_set():
                logger.info("🛑 [中断] 跳过剩余播报内容")
                break
        
        # 停止唤醒词监听
        if monitor_task:
            monitor_task.cancel()
            try: await monitor_task
            except asyncio.CancelledError: pass
        
        was_interrupted = self.interrupt_event.is_set()
        if update_ui_title_cb: update_ui_title_cb("💤")
        return was_interrupted

audio_engine = AudioEngine()
