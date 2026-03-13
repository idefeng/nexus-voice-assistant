# OpenClaw 基础配置
OPENCLAW_API_URL = "http://localhost:18789/v1/chat/completions"
OPENCLAW_TOKEN = "b5993a1fab396eb87cf92a6ab8c6e4962d198d2409b3714a"

# Advanced Routing (v2.4.0)
# 格式: channel/accountId/agentId
OPENCLAW_CHANNEL = "telegram"
OPENCLAW_ACCOUNT = "main"
OPENCLAW_AGENT = "main"

# 默认降级模型/Agent ID
SESSION_KEY = "main" 

# 唤醒词配置
WAKE_WORD_PATH = "./models/小德_zh_mac_v4_0_0.ppn"
PORCUPINE_MODEL_PATH = "./models/porcupine_params_zh.pv"
PORCUPINE_ACCESS_KEY = "nApLVOOz0OFhReQa62OKtQs7fYsFxcDx1EcTyC/MW8x6q2M2xS6TxQ=="

# 功能开关
ENABLE_VOICE_RECOGNITION = True
ENABLE_FACE_RECOGNITION = True
ENABLE_EMOTION_ANALYSIS = True
ENABLE_MEMORY = True

# 音频配置
SAMPLERATE = 16000
CHANNELS = 1
MAX_RECORD_SECONDS = 30

# 模型配置
WHISPER_MODEL = "medium"

# 数据库路径
MEMORY_DB_PATH = "./memory_db"
MEMORY_COLLECTION_NAME = "asxin_memories"

# 摄像头配置
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# TTS配置
TTS_VOICE = "zh-CN-XiaoxiaoNeural"
