#!/usr/bin/env python3
import os
import sys
import pvporcupine
import pyaudio
import struct
import speech_recognition as sr
import requests
import subprocess

from config import *

print("🚀 启动极简版语音助手...")

# 初始化语音识别器
r = sr.Recognizer()

# 初始化Porcupine唤醒引擎
try:
    porcupine = pvporcupine.create(
        access_key=PORCUPINE_ACCESS_KEY,
        keyword_paths=[WAKE_WORD_PATH]
    )
    print("✅ 唤醒引擎初始化成功")
except Exception as e:
    print(f"❌ 唤醒引擎初始化失败: {e}")
    print("💡 请检查Porcupine Access Key是否正确")
    sys.exit(1)

# 初始化音频流
pa = pyaudio.PyAudio()
audio_stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length
)

def speak(text):
    """调用系统say命令发音"""
    print(f"🤖 回复：{text}")
    subprocess.run(["say", "-v", "Mei-Jia", text])

def recognize_speech():
    """用macOS自带语音识别"""
    print("🎙️  正在听...")
    with sr.Microphone(sample_rate=16000) as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source, timeout=5, phrase_time_limit=10)
    
    try:
        text = r.recognize_google(audio, language="zh-CN")
        print(f"👤 你说：{text}")
        return text
    except sr.UnknownValueError:
        speak("抱歉，我没听清你说什么")
        return None
    except sr.RequestError:
        speak("语音识别服务暂时不可用")
        return None

def call_openclaw(text):
    """调用OpenClaw"""
    print("🤔 思考中...")
    payload = {
        "message": text,
        "session_key": SESSION_KEY
    }
    
    try:
        response = requests.post(OPENCLAW_API_URL, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return "抱歉，我现在无法回答你的问题。"
    except Exception as e:
        print(f"OpenClaw调用失败: {e}")
        return "连接失败，请检查OpenClaw服务是否运行。"

def main():
    print("✅ 语音助手已启动，等待唤醒词: Hey Pico")
    
    try:
        while True:
            pcm = audio_stream.read(porcupine.frame_length)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            
            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                print("\n🎉 唤醒成功！")
                speak("我在呢")
                
                # 识别语音
                text = recognize_speech()
                if not text:
                    continue
                
                # 调用OpenClaw
                response = call_openclaw(text)
                
                # 语音回复
                speak(response)
                
                print("\n✅ 回复完成，继续等待唤醒...")
                
    except KeyboardInterrupt:
        print("\n👋 退出程序")
    finally:
        audio_stream.close()
        pa.terminate()
        porcupine.delete()

if __name__ == "__main__":
    main()
