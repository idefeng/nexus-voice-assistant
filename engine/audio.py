import os
import asyncio
import edge_tts
import pyaudio
import struct
import whisper
import torch
import wave
import random
import subprocess
import logging
import re
from config import *

logger = logging.getLogger("Xiaode.Audio")

class AudioEngine:
    def __init__(self):
        logger.info(f"🔊 加载语音识别驱动 ({WHISPER_MODEL})...")
        self.whisper_model = whisper.load_model(WHISPER_MODEL)
        self.pa = pyaudio.PyAudio()
        self.current_speaker_process = None
        self.NATURAL_FILLERS = ["嗯...", "我想想...", "好的，我明白了。", "让我想一下。"]

    async def record_audio(self, stream, frame_length, sample_rate, max_duration, is_follow_up=False):
        """录制音频数据"""
        frames = []
        CHUNK_SIZE = frame_length
        SILENCE_THRESHOLD = 500
        SILENCE_DURATION = 0.8 if is_follow_up else 1.2
        
        limit = int(sample_rate / CHUNK_SIZE * SILENCE_DURATION)
        total_timeout = int(sample_rate / CHUNK_SIZE * (FOLLOW_UP_TIMEOUT if is_follow_up else max_duration))
        
        counter, recorded = 0, 0
        while recorded < total_timeout:
            try:
                data = await asyncio.to_thread(stream.read, CHUNK_SIZE, exception_on_overflow=False)
                frames.append(data); recorded += 1
                energy = np.abs(struct.unpack_from("h" * CHUNK_SIZE, data)).mean()
                if energy < SILENCE_THRESHOLD: counter += 1
                else: counter = 0
                
                if is_follow_up and recorded > int(sample_rate / CHUNK_SIZE * 3.0) and counter == recorded:
                    return None 

                if recorded > int(sample_rate / CHUNK_SIZE * 0.5) and counter > limit: break
            except Exception as e:
                logger.warning(f"⚠️ 录音读取中断: {e}")
                break
        return b''.join(frames)

    async def audio_to_text(self, audio_data, sample_rate, channels):
        """ASR: 语音转文字"""
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
                with torch.no_grad(): 
                    return self.whisper_model.transcribe(wf, language="zh")
            
            res = await asyncio.to_thread(transcribe)
            return res["text"].strip()
        except Exception as e:
            logger.error(f"语音识别异常: {e}")
            return ""
        finally:
            if os.path.exists(wf): os.remove(wf)

    async def _stream_speak_async(self, text):
        if not text or not text.strip(): return
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
        except Exception as e:
            logger.warning(f"TTS 播放失败: {e}")

    async def speak_async(self, text, with_filler=False, update_ui_title_cb=None):
        """TTS: 文字转语音"""
        if not text or not text.strip(): return
        
        if update_ui_title_cb: update_ui_title_cb("🗣️")
        
        if self.current_speaker_process:
            try: self.current_speaker_process.terminate()
            except: pass
        
        if with_filler and random.random() < 0.3:
            await self._stream_speak_async(random.choice(self.NATURAL_FILLERS))
        
        segs = [s.strip() for s in re.split(r'([。！？\n])', text) if s.strip()]
        for seg in segs or [text]:
            clean = re.sub(r'#+\s*|[*_\-]{1,3}|[`>]|\[([^\]]+)\]\([^\)]+\)|[\U00010000-\U0010ffff]', '', seg)
            if clean.strip(): await self._stream_speak_async(clean.strip())
            
        if update_ui_title_cb: update_ui_title_cb("💤")

audio_engine = AudioEngine()
