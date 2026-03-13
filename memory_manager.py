import chromadb
from chromadb.config import Settings
import os
import json
import datetime
import requests
from config import MEMORY_DB_PATH, MEMORY_COLLECTION_NAME, OPENCLAW_API_URL, OPENCLAW_TOKEN, SESSION_KEY

class MemoryManager:
    def __init__(self):
        # 初始化 ChromaDB
        if not os.path.exists(MEMORY_DB_PATH):
            os.makedirs(MEMORY_DB_PATH)
        
        self.client = chromadb.PersistentClient(path=MEMORY_DB_PATH)
        self.collection = self.client.get_or_create_collection(name=MEMORY_COLLECTION_NAME)
        self.emotion_collection = self.client.get_or_create_collection(name="emotion_history")

    def save_memory(self, text, metadata=None):
        """保存一段记忆"""
        timestamp = datetime.datetime.now().isoformat()
        metadata = metadata or {}
        metadata["timestamp"] = timestamp
        
        # 使用哈希或其他 ID 生成方式
        import hashlib
        mem_id = hashlib.md5(f"{timestamp}_{text[:50]}".encode()).hexdigest()
        
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[mem_id]
        )
        print(f"📦 记忆已存入: {text[:50]}...")

    def query_memory(self, text, n_results=3):
        """检索相关记忆"""
        try:
            results = self.collection.query(
                query_texts=[text],
                n_results=n_results
            )
            if results["documents"] and len(results["documents"][0]) > 0:
                return "\n".join(results["documents"][0])
            return ""
        except Exception as e:
            print(f"⚠️ 记忆检索失败: {e}")
            return ""

    def save_emotion(self, emotion):
        """保存单次情绪记录"""
        timestamp = datetime.datetime.now().isoformat()
        import hashlib
        emo_id = hashlib.md5(f"emo_{timestamp}".encode()).hexdigest()
        self.emotion_collection.add(
            documents=[emotion],
            metadatas=[{"timestamp": timestamp}],
            ids=[emo_id]
        )

    def get_recent_emotions(self, limit=5):
        """获取最近的情绪轨迹"""
        try:
            results = self.emotion_collection.get(limit=limit)
            if results["documents"]:
                return results["documents"]
            return []
        except:
            return []

    def extract_and_save_facts(self, user_text, ai_response):
        """
        利用 LLM 从对话中提取事实并保存。
        这是一个辅助方法，通常在后台运行。
        """
        prompt = f"""
        从以下对话中提取关于用户的持久性事实或偏好（如性格、工作、喜好、技术栈等）。
        如果没有任何值得记住的信息，请只回复“NONE”。
        如果有多个事实，请分别列出。
        
        用户：{user_text}
        助手：{ai_response}
        
        请直接输出提取的事实，不要有额外解释。
        """
        
        try:
            headers = {
                "Authorization": f"Bearer {OPENCLAW_TOKEN}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": SESSION_KEY,
                "messages": [
                    {"role": "system", "content": "你是一个信息提取专家，专注于记录用户的偏好。"},
                    {"role": "user", "content": prompt}
                ]
            }
            response = requests.post(OPENCLAW_API_URL, json=payload, headers=headers, timeout=30)
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"].strip()
                if content != "NONE" and len(content) > 2:
                    facts = content.split("\n")
                    for fact in facts:
                        if fact.strip():
                            self.save_memory(fact.strip(), {"type": "user_fact"})
            else:
                print(f"⚠️ 事实提取失败: {response.status_code}")
        except Exception as e:
            print(f"⚠️ 事实提取异常: {e}")

# 全局单例
if os.environ.get("ENABLE_MEMORY", "True") == "True":
    memory_manager = MemoryManager()
else:
    memory_manager = None
