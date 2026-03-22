#!/usr/bin/env python3
import os
import requests
import subprocess
import speech_recognition as sr

# 配置
OPENCLAW_API_URL = "http://localhost:7367/api/v1/chat"
SESSION_KEY = "agent:scholar:telegram:direct:7240755193"

print("🚀 极简版语音助手启动成功！")
print("💡 按回车键开始录音，说你要问的问题，说完按Ctrl+C结束录音")

r = sr.Recognizer()

def speak(text):
    """用macOS自带TTS"""
    print(f"🤖 阿信：{text}")
    subprocess.run(["say", "-v", "Mei-Jia", text])

def recognize_speech():
    """macOS自带语音识别"""
    with sr.Microphone() as source:
        print("🎙️  正在录音，开始说话吧...（按Ctrl+C结束录音）")
        r.adjust_for_ambient_noise(source)
        try:
            audio = r.listen(source, timeout=30)
            text = r.recognize_google(audio, language="zh-CN")
            print(f"👤 你说：{text}")
            return text
        except KeyboardInterrupt:
            print("\n⏹️  录音结束")
            try:
                text = r.recognize_google(audio, language="zh-CN")
                print(f"👤 你说：{text}")
                return text
            except:
                return None
        except Exception as e:
            print(f"❌ 识别失败：{e}")
            speak("抱歉，我没听清")
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

if __name__ == "__main__":
    while True:
        input("\n👉 按回车键开始提问...")
        text = recognize_speech()
        if not text:
            continue
        if text in ["退出", "再见", "拜拜"]:
            speak("再见，有需要随时叫我！")
            break
        response = call_openclaw(text)
        speak(response)
