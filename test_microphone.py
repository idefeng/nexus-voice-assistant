#!/usr/bin/env python3
import speech_recognition as sr
import subprocess

print("🎤 测试麦克风和语音识别...")

# 列出所有麦克风
r = sr.Recognizer()
mics = sr.Microphone.list_microphone_names()
print(f"可用麦克风：{mics}")

# 测试录音
try:
    with sr.Microphone() as source:
        print("🎙️  正在录音3秒，请说点什么...")
        r.adjust_for_ambient_noise(source)
        audio = r.record(source, duration=3)
        print("✅ 录音完成，正在识别...")
        text = r.recognize_google(audio, language="zh-CN")
        print(f"识别结果：{text}")
        # 测试TTS
        subprocess.run(["say", "-v", "Mei-Jia", f"我听到你说：{text}"])
        print("✅ 麦克风和语音识别工作正常！")
except Exception as e:
    print(f"❌ 测试失败：{e}")
    import traceback
    traceback.print_exc()
