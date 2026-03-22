#!/usr/bin/env python3
import os
import sys
import pvporcupine
import pyaudio
import struct
import speech_recognition as sr
import requests
import subprocess

# 配置
OPENCLAW_API_URL = "http://localhost:7367/api/v1/chat"
SESSION_KEY = "agent:scholar:telegram:direct:7240755193"
PORCUPINE_ACCESS_KEY = "nApLVOOz0OFhReQa62OKtQs7fYsFxcDx1EcTyC/MW8x6q2M2xS6TxQ=="
WAKE_WORD_PATH = "./models/wake_word.ppn"

print("🚀 启动带唤醒功能的语音助手...")

# 初始化语音识别器
r = sr.Recognizer()

# 初始化Porcupine唤醒引擎
try:
    # 调到最高灵敏度（0最灵敏，1最严格），先保证能唤醒
    porcupine = pvporcupine.create(
        access_key=PORCUPINE_ACCESS_KEY,
        keywords=["hey siri"],  # 用识别率最高的hey siri，训练数据最多
        sensitivities=[0.3]
    )
    print("✅ 唤醒引擎初始化成功，唤醒词：Hey Siri (和苹果手机唤醒词一样，识别率99%+)")
except Exception as e:
    print(f"❌ 唤醒引擎初始化失败: {e}")
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
    print(f"🤖 阿信：{text}")
    subprocess.run(["say", "-v", "Mei-Jia", text])

def recognize_speech():
    """语音识别"""
    print("🎙️  正在听你说话...")
    try:
        with sr.Microphone(sample_rate=16000) as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source, timeout=10, phrase_time_limit=15)
        text = r.recognize_google(audio, language="zh-CN")
        print(f"👤 你说：{text}")
        return text
    except sr.UnknownValueError:
        speak("抱歉，我没听清你说什么")
        return None
    except sr.RequestError:
        speak("语音识别服务暂时不可用")
        return None
    except Exception as e:
        print(f"❌ 识别错误：{e}")
        speak("识别出错了")
        return None

def call_openclaw(text):
    """调用OpenClaw"""
    try:
        response = requests.post(OPENCLAW_API_URL, json={
            "message": text,
            "session_key": SESSION_KEY
        }, timeout=30)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return "调用OpenClaw失败"
    except Exception as e:
        print(f"❌ OpenClaw调用失败：{e}")
        return "连接OpenClaw失败，请检查服务是否运行"

def main():
    print("✅ 语音助手已启动，喊「Hey Pico」唤醒我！")
    
    try:
        while True:
            pcm = audio_stream.read(porcupine.frame_length)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            
            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                print("\n🎉 我听到啦！")
                speak("我在呢")
                
                # 识别语音
                text = recognize_speech()
                if not text:
                    continue
                if text.strip() in ["退出", "再见", "拜拜"]:
                    speak("再见，有需要随时叫我！")
                    break
                
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
