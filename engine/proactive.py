import time
import datetime
import random
import asyncio
import logging
from typing import List, Dict, Any

logger = logging.getLogger("Xiaode.Proactive")

class ProactiveEngine:
    def __init__(self):
        self.cooldown = 120 
        self.last_trigger_time = time.time()
        self.trigger_flag = asyncio.Event()
        self.manual_trigger_flag = asyncio.Event() # 手动快捷键触发
        self.trigger_type = "greeting"
        self.scheduled_reminders = [
            {"hour": 10, "minute": 30, "msg": "德哥，该喝水休息一下啦 🐻", "done": False},
            {"hour": 15, "minute": 0, "msg": "下午茶时间到！要不要起来动一动？☕", "done": False},
            {"hour": 23, "minute": 0, "msg": "太晚了，记得早点休息哦 🌕", "done": False},
        ]

    async def check_reminders(self, speak_cb):
        """检查并触发定时提醒"""
        now = datetime.datetime.now()
        for r in self.scheduled_reminders:
            if r["hour"] == now.hour and r["minute"] == now.minute and not r["done"]:
                await speak_cb(r["msg"])
                r["done"] = True
            elif r["hour"] != now.hour:
                r["done"] = False

    def should_trigger_proactive(self, face_detected, is_authorized, is_tired):
        """判断是否应该触发主动交互"""
        if time.time() - self.last_trigger_time < self.cooldown:
            return False
            
        if not face_detected or not is_authorized:
            return False
            
        # 决策逻辑
        if is_tired:
            self.trigger_type = "fatigue_care"
            return True
        elif random.random() < 0.2: # 降低频率
            self.trigger_type = "screen_insight"
            return True
        elif random.random() < 0.1:
            self.trigger_type = "greeting"
            return True
            
        return False

    def mark_triggered(self):
        self.last_trigger_time = time.time()
        self.trigger_flag.clear()

proactive_engine = ProactiveEngine()
