# OpenClaw配置
OPENCLAW_API_URL = "http://localhost:18789/v1/chat/completions"
OPENCLAW_TOKEN = "b5993a1fab396eb87cf92a6ab8c6e4962d198d2409b3714a"
SESSION_KEY = "scholar"  # 使用Agent ID作为模型名

# 唤醒词配置
WAKE_WORD_PATH = "./models/小德_zh_mac_v4_0_0.ppn"
PORCUPINE_MODEL_PATH = "./models/porcupine_params_zh.pv"
PORCUPINE_ACCESS_KEY = "nApLVOOz0OFhReQa62OKtQs7fYsFxcDx1EcTyC/MW8x6q2M2xS6TxQ=="  # 已自动填充

# 功能开关
ENABLE_VOICE_RECOGNITION = True
ENABLE_FACE_RECOGNITION = True
ENABLE_EMOTION_ANALYSIS = True
ENABLE_MEMORY = True  # v2.0.0 长期记忆开关

# 音频配置
SAMPLERATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5
MAX_RECORD_SECONDS = 30  # 支持长句子收录

# 模型配置
WHISPER_MODEL = "medium"  # 升级到 medium 以获得更好识别精度

# 数据库路径
MEMORY_DB_PATH = "./memory_db"
MEMORY_COLLECTION_NAME = "asxin_memories"

# 摄像头配置
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# TTS配置
TTS_RATE = 200
TTS_VOLUME = 0.9
TTS_VOICE = "zh-CN-XiaoxiaoNeural"  # 使用 Edge-TTS 的优秀音色
