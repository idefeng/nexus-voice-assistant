#!/usr/bin/env python3
import speech_recognition as sr
import subprocess

print("🎉 极简语音助手启动成功！")
print("💡 按回车键开始说话，说完按Ctrl+C结束录音\n")

r = sr.Recognizer()

def speak(text):
    print(f"🤖 回复：{text}")
    subprocess.run(["say", "-v", "Mei-Jia", text])

while True:
    input("👉 按回车键开始提问...")
    try:
        with sr.Microphone() as source:
            print("🎙️  正在录音，开始说话吧...")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source, timeout=30)
        
        print("✍️  识别中...")
        text = r.recognize_google(audio, language="zh-CN")
        print(f"👤 你说：{text}")
        
        if text in ["退出", "再见", "拜拜"]:
            speak("再见，下次聊！")
            break
        
        # 直接发送消息到当前会话
        cmd = f'openclaw message send --target 7240755193 --message "{text}"'
        subprocess.run(cmd, shell=True)
        speak("好的，我已经收到你的问题了，回复会出现在聊天窗口哦~")
        print("✅ 消息已发送，等待回复...\n")
        
    except KeyboardInterrupt:
        print("\n⏹️  录音结束")
        try:
            text = r.recognize_google(audio, language="zh-CN")
            print(f"👤 你说：{text}")
            cmd = f'openclaw message send --target 7240755193 --message "{text}"'
            subprocess.run(cmd, shell=True)
            speak("好的，我已经收到你的问题了，回复会出现在聊天窗口哦~")
        except:
            speak("抱歉，我没听清你说什么")
    except Exception as e:
        print(f"❌ 错误：{e}")
        speak("出错了，请重试")
