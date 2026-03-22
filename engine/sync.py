import asyncio
import httpx
import logging
from config import SYNC_INGEST_URL

logger = logging.getLogger("Xiaode.Sync")

async def sync_event(text: str, fatigue_score: float):
    """
    异步同步语音识别结果和疲劳分数到后端。
    采用“发射后不管”(Fire-and-forget) 模式。
    """
    if not SYNC_INGEST_URL:
        return

    payload = {
        "final_text": text,
        "fatigue_score": fatigue_score
    }

    try:
        # 使用较短的超时时间，确保由于网络问题不会导致长时间挂起
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.post(SYNC_INGEST_URL, json=payload)
            if response.status_code == 200:
                # 仅在同步成功时打印一条隐蔽的 Debug 信息
                logger.debug(f"🤫 [同步外挂] 数据成功推送到后端: {text[:20]}...")
            else:
                # 失败时不抛出异常，保持静默，以免干扰主流程日志
                pass
    except Exception:
        # 忽略所有错误，确保 decoupling
        pass

def sync_event_detached(text: str, fatigue_score: float):
    """用于在主流程中非阻塞触发同步任务"""
    asyncio.create_task(sync_event(text, fatigue_score))
