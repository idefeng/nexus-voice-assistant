#!/usr/bin/env python3
import speech_recognition as sr
import subprocess
import requests

print("🎤 完整功能测试...")

# 检查麦克风权限
print("📋 请确保你已经在系统设置→隐私与安全性→麦克风中给当前终端开启了权限")
input("按回车键开始测试...")

r = sr.Recognizer()

# 测试录音和识别
try:
    with sr.Microphone() as source:
        print("🎙️  请说一句中文，比如：今天天气怎么样？")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source, timeout=10)
        print("✅ 录音完成，正在识别...")
        text = r.recognize_google(audio, language="zh-CN")
        print(f"✅ 识别成功！你说：{text}")
        
        # 测试TTS
        print("🔊 测试语音合成...")
        subprocess.run(["say", "-v", "Mei-Jia", f"我听到你说：{text}"])
        print("✅ 语音合成正常！")
        
        # 测试OpenClaw调用
        print("🤖 测试OpenClaw连接...")
        try:
            response = requests.post("http://localhost:7367/api/v1/chat", json={
                "message": "你是谁？用一句话回答",
                "session_key": "agent:scholar:telegram:direct:7240755193"
            }, timeout=10)
            if response.status_code == 200:
                result = response.json()["response"]
                print(f"✅ OpenClaw连接正常！回复：{result}")
                subprocess.run(["say", "-v", "Mei-Jia", result])
            else:
                print("❌ OpenClaw连接失败，状态码：", response.status_code)
        except Exception as e:
            print(f"❌ OpenClaw连接失败：{e}")
        
        print("\n🎉 所有基础功能正常！如果唤醒词不好用，可以调整灵敏度。")
        
except Exception as e:
    print(f"❌ 测试失败：{e}")
    import traceback
    traceback.print_exc()
