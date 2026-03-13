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
from onnxruntime import InferenceSession

from config import *

# 状态表情常量
STATUS_IDLE = "💤"
STATUS_LISTENING = "🎙️"
STATUS_THINKING = "🤔"
STATUS_SPEAKING = "🗣️"

# 自然语气词
NATURAL_FILLERS = ["嗯...", "我想想...", "好的，我明白了。", "原来是这样。", "让我想一下。"]

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
    app.title = STATUS_THINKING
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

def get_time_greeting():
    """获取时间相关的问候语"""
    hour = datetime.datetime.now().hour
    if 5 <= hour < 11:
        return "早安！今天也是充满活力的一天。"
    elif 11 <= hour < 14:
        return "午安！记得休息一下按时吃饭哦。"
    elif 14 <= hour < 18:
        return "下午好！需要喝点咖啡提提神吗？"
    elif 18 <= hour < 23:
        return "晚上好！忙碌的一天辛苦了。"
    else:
        return "太晚了，注意休息哦，德哥。"

def call_openclaw(text, emotion=None):
    """调用OpenClaw接口获取回复 (增强情感理解)"""
    app.title = STATUS_THINKING
    try:
        # 情感引导指令
        system_prompt = "你是一个亲切、聪明的助手，名叫阿信。"
        if emotion:
            emotion_prompts = {
                'happy': "用户现在心情很好，你的回复应该保持轻快、活泼，并分享这份快乐。",
                'sad': "用户现在看起来有点难过，请展示出你的温柔和包容，多给予一些鼓励和支持。",
                'angry': "用户现在可能有情绪，请保持专业、冷静，并尝试用平和的语气引导，不要在这个时候开玩笑。",
                'surprise': "用户感到惊讶，你可以用好奇和探索的语气和他交流。",
                'neutral': "保持亲切自然的交流风格即可。"
            }
            system_prompt += f" {emotion_prompts.get(emotion, '根据用户的情感状态，给予人性化的反馈。')}"
            
        headers = {
            "Authorization": f"Bearer {OPENCLAW_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": SESSION_KEY,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
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

def speak(text, with_filler=False):
    """语音合成，支持智能切换引擎和自然语气词"""
    if not text:
        return
    
    app.title = STATUS_SPEAKING
    
    # 随机添加语气助词
    if with_filler and random.random() < 0.4:
        filler = random.choice(NATURAL_FILLERS)
        print(f"🤖 阿信 (Filler)：{filler}")
        asyncio.run(_edge_speak(filler))
        
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
    
    app.title = STATUS_IDLE
 
class VoiceAssistantApp(rumps.App):
    def __init__(self):
        super(VoiceAssistantApp, self).__init__("阿信", title=STATUS_IDLE)
        self.menu = ["关于阿信", "重启助手"]

    @rumps.clicked("关于阿信")
    def about(self, _):
        rumps.alert("阿信 v1.2.0\n您的智能数字副官\n\n状态说明：\n💤 待机\n🎙️ 倾听\n🤔 思考\n🗣️ 播报")

    @rumps.clicked("重启助手")
    def restart(self, _):
        os.execv(sys.executable, ['python'] + sys.argv)

def run_voice_assistant():
    """主逻辑循环，运行在后台线程"""
    global is_first_run
    is_first_run = True
    
    print("✅ 语音助手已启动，等待唤醒词...")
    app.title = STATUS_IDLE
    
    try:
        while True:
            try:
                pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            except OSError as e:
                if e.errno == -9981: continue
                raise
            
            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                print("\n🎉 唤醒成功！")
                
                # 1. 立即启动异步视觉分析（不阻塞主链路）
                emotion_result = [None] # 使用列表作为闭包容器获取线程结果
                def background_vision():
                    try:
                        if ENABLE_FACE_RECOGNITION and ENABLE_EMOTION_ANALYSIS:
                            _, emotion = detect_face_and_emotion()
                            emotion_result[0] = emotion
                            if emotion:
                                print(f"😊 [后台识别结果] 情绪：{emotion}")
                    except Exception as e:
                        print(f"⚠️ 视觉分析后台异常: {e}")

                vision_thread = threading.Thread(target=background_vision)
                vision_thread.start()

                # 2. 每日首次唤醒礼
                if is_first_run:
                    greeting = get_time_greeting()
                    speak(greeting)
                    is_first_run = False
                else:
                    speak("我在呢")
                
                # 3. 立即进入录音状态（录音时长受 MAX_RECORD_SECONDS 限制，并有静音检测）
                app.title = STATUS_LISTENING
                audio_data = record_audio(MAX_RECORD_SECONDS)
                
                # 4. 语音转文字
                text = audio_to_text(audio_data)
                print(f"👤 你说：{text}")
                
                if not text.strip():
                    speak("抱歉，我没听清你说什么。")
                    app.title = STATUS_IDLE
                    continue
                
                # 5. 等待视觉分析线程结束（如果还在运行）以获取最新的情绪
                vision_thread.join(timeout=2.0)
                emotion = emotion_result[0]
                
                # 6. 调用OpenClaw并带入情绪内容（不再单独语音播报情绪，由LLM在回复中体现）
                response = call_openclaw(text, emotion)
                speak(response, with_filler=True)
                
                print("\n✅ 回复完成，继续等待唤醒...")
                app.title = STATUS_IDLE
                
    except Exception as e:
        print(f"❌ 后台执行错误: {e}")
    finally:
        audio_stream.close()
        pa.terminate()
        porcupine.delete()

if __name__ == "__main__":
    app = VoiceAssistantApp()
    # 启动后台逻辑线程
    threading.Thread(target=run_voice_assistant, daemon=True).start()
    # 运行 UI 主循环
    app.run()
