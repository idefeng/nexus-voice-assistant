import httpx
import json
import base64
import logging
import asyncio
from typing import Optional, Dict, Any, List
from config import *
from memory_manager import memory_manager
from scripts.context_helper import get_active_window_info

logger = logging.getLogger("Xiaode.Brain")

class BrainEngine:
    def __init__(self):
        self.TOOLS_SCHEMA = [
            {
                "type": "function",
                "function": {
                    "name": "get_sports_data",
                    "description": "获取用户的运动健身数据",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["latest", "monthly"]},
                            "month": {"type": "string"}
                        },
                        "required": ["type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_health_data",
                    "description": "获取用户的健康/睡眠数据",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string", "enum": ["sleep", "heart_rate"]}
                        },
                        "required": ["category"]
                    }
                }
            }
        ]

    async def execute_tool_async(self, name, args, update_ui_cb=None):
        logger.info(f"🛠️ [工具执行] {name}({args})")
        if update_ui_cb:
            update_ui_cb({"status": "⚙️", "response": f"🛠️ 正在执行工具: {name}..."})
        
        try:
            async with httpx.AsyncClient() as client:
                if name == "get_sports_data":
                    t, m = args.get("type"), args.get("month", "")
                    base = "http://localhost:8000/api/v1/agent/"
                    path = "latest_activity" if t == "latest" else "monthly_report"
                    r = await client.get(f"{base}{path}", params={"target_month": m}, timeout=10)
                    res = r.json().get("report", "暂时没能获取到运动详情。")
                    if update_ui_cb:
                        update_ui_cb({"status": "🤔", "response": "✅ 成功获取数据，正在整理..."})
                    return res
                elif name == "get_health_data":
                    return "根据最近监测，你昨晚深度睡眠达标，建议今天继续保持规律作息。"
                return f"错误：未定义的工具 {name}"
        except Exception as e:
            logger.error(f"工具执行失败: {e}")
            if update_ui_cb:
                update_ui_cb({"status": "🤔", "response": f"❌ 工具执行失败: {str(e)}"})
            return f"API 访问失败: {str(e)}"

    async def call_llm_async(self, text, emotion=None, image_path=None, history=[], is_proactive=False, p_mode="greeting", is_tired=False, update_ui_cb=None):
        if update_ui_cb:
            update_ui_cb({"status": "🤔", "transcription": text, "response": ""})
            
        try:
            mem = ""
            if not history and ENABLE_MEMORY and memory_manager:
                mem = await asyncio.to_thread(memory_manager.query_memory, text)
            
            sys_p = (
                f"你叫小德 (Xiaode) 🐻。你是一个智能助手。风格：俏皮、温馨、深度体贴。"
                f"\n【实时场景】用户正在使用: {get_active_window_info()}"
            )
            if mem: sys_p += f"\n【历史背景】{mem}"
            if emotion: sys_p += f"\n【当前情绪】{emotion}"
            if is_tired or p_mode == "fatigue_care": sys_p += "\n【生理状态】检测到用户疲劳，请给予关心。"
            
            user_msg = []
            if image_path:
                with open(image_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
                user_msg = [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ]
            
            msgs = [{"role": "system", "content": sys_p}] + history
            msgs.append({"role": "user", "content": user_msg if user_msg else text})
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                for _ in range(5):
                    r = await client.post(OPENCLAW_API_URL, json={
                        "model": f"{OPENCLAW_CHANNEL}/{OPENCLAW_ACCOUNT}/{OPENCLAW_AGENT}",
                        "messages": msgs, "tools": self.TOOLS_SCHEMA
                    }, headers={"Authorization": f"Bearer {OPENCLAW_TOKEN}"})
                    
                    if r.status_code != 200: 
                        return {"type": "error", "content": f"API 响应异常 ({r.status_code})"}
                    
                    msg = r.json()["choices"][0]["message"]
                    msgs.append(msg)
                    
                    if not msg.get("tool_calls"):
                        content = msg.get("content", "")
                        if update_ui_cb:
                            update_ui_cb({"response": content})
                        return {"type": "text", "content": content}
                    
                    for call in msg["tool_calls"]:
                        name = call["function"]["name"]
                        args = json.loads(call["function"]["arguments"])
                        result = await self.execute_tool_async(name, args, update_ui_cb)
                        msgs.append({"role": "tool", "tool_call_id": call["id"], "name": name, "content": str(result)})
            
            return {"type": "text", "content": "大脑过载了..."}
        except Exception as e:
            logger.error(f"AI 调用异常: {e}")
            return {"type": "error", "content": str(e)}

brain_engine = BrainEngine()
