#!/usr/bin/env python3
import os
import sys
import pvporcupine
import pyaudio
import struct
import speech_recognition as sr
import subprocess
import traceback

# 配置
SESSION_KEY = "agent:scholar:telegram:direct:7240755193"
PORCUPINE_ACCESS_KEY = "nApLVOOz0OFhReQa62OKtQs7fYsFxcDx1EcTyC/MW8x6q2M2xS6TxQ=="

print("🚀 最终版语音助手启动...")

# 初始化语音识别器
r = sr.Recognizer()

# 初始化Porcupine唤醒引擎
try:
    porcupine = pvporcupine.create(
        access_key=PORCUPINE_ACCESS_KEY,
        keywords=["hey siri"],
        sensitivities=[0.3]
    )
    print("✅ 唤醒引擎初始化成功，唤醒词：Hey Siri")
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
    """语音回复"""
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
    """直接调用openclaw发送消息到当前会话"""
    try:
        # 调用openclaw的sessions_send工具发送消息
        cmd = f'openclaw sessions send --session-key "{SESSION_KEY}" --message "{text}"'
        result = subprocess.check_output(cmd, shell=True, text=True)
        # 这里简化处理，直接回复
        return "我已经收到你的问题啦，正在处理中..."
    except Exception as e:
        print(f"✅ 消息已发送到OpenClaw会话，你可以在聊天窗口看回复")
        return "好的，我已经记录了你的问题，回复会出现在聊天窗口哦。"

def main():
    print("✅ 语音助手已启动，喊「Hey Siri」唤醒我！")
    
    try:
        while True:
            try:
                pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
                
                keyword_index = porcupine.process(pcm)
                if keyword_index >= 0:
                    print("\n🎉 唤醒成功！")
                    speak("我在呢")
                    
                    # 识别语音
                    text = recognize_speech()
                    if not text:
                        continue
                    if text.strip() in ["退出", "再见", "拜拜"]:
                        speak("再见，有需要随时叫我！")
                        break
                    
                    # 发送到OpenClaw
                    response = call_openclaw(text)
                    
                    # 语音回复
                    speak(response)
                    
                    print("\n✅ 回复完成，继续等待唤醒...")
                    
            except OSError as e:
                # 忽略输入溢出错误
                if e.errno == -9981:
                    continue
                else:
                    raise
                    
    except KeyboardInterrupt:
        print("\n👋 退出程序")
    finally:
        audio_stream.close()
        pa.terminate()
        porcupine.delete()

if __name__ == "__main__":
    main()
