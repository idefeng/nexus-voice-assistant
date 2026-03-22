import cv2
import numpy as np
import insightface
import logging
from onnxruntime import InferenceSession
from config import *

logger = logging.getLogger("Xiaode.Perception")

class PerceptionEngine:
    def __init__(self):
        self.face_analyzer = None
        self.emotion_session = None
        self.emotion_labels = ['neutral', 'happy', 'surprise', 'sad', 'angry', 'disgust', 'fear', 'contempt']
        self.EMOTION_MAP = {
            'neutral': '平静', 'happy': '愉快', 'surprise': '惊讶', 
            'sad': '低落', 'angry': '愤怒', 'disgust': '厌恶', 
            'fear': '恐惧', 'contempt': '轻蔑'
        }
        self._init_models()

    def _init_models(self):
        if ENABLE_FACE_RECOGNITION:
            logger.info("👤 加载人脸识别引擎...")
            self.face_analyzer = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

        if ENABLE_EMOTION_ANALYSIS:
            logger.info("😐 加载情绪分析模型...")
            try:
                self.emotion_session = InferenceSession(EMOTION_MODEL_PATH)
            except Exception as e:
                logger.error(f"加载情绪分析模型失败: {e}")

    def calculate_ear(self, landmarks):
        """计算眼睛纵横比 (EAR)"""
        def dist(p1, p2): return np.linalg.norm(p1 - p2)
        l_v1 = dist(landmarks[37], landmarks[41])
        l_v2 = dist(landmarks[38], landmarks[40])
        l_h = dist(landmarks[36], landmarks[39])
        l_ear = (l_v1 + l_v2) / (2.0 * l_h)
        
        r_v1 = dist(landmarks[43], landmarks[47])
        r_v2 = dist(landmarks[44], landmarks[46])
        r_h = dist(landmarks[42], landmarks[45])
        r_ear = (r_v1 + r_v2) / (2.0 * r_h)
        
        return (l_ear + r_ear) / 2.0

    def analyze_frame(self, frame):
        """分析单帧图像，提取人脸、情绪和疲劳状态"""
        if frame is None or self.face_analyzer is None:
            return None, None, None, False

        try:
            faces = self.face_analyzer.get(frame)
            if len(faces) == 0: 
                return None, None, None, False
            
            face = faces[0]
            emotion = "平静"
            if self.emotion_session:
                face_img = cv2.resize(frame, (64, 64))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                face_img = face_img.astype(np.float32) / 255.0
                face_img = np.expand_dims(np.expand_dims(face_img, axis=0), axis=0)
                outputs = self.emotion_session.run(None, {'Input3': face_img})
                raw_emotion = self.emotion_labels[np.argmax(outputs[0])]
                emotion = self.EMOTION_MAP.get(raw_emotion, "未知")
            
            ear = self.calculate_ear(face.landmark_3d_68)
            is_tired = ear < 0.22
            
            return face, emotion, face.normed_embedding, is_tired
        except Exception as e:
            logger.error(f"人脸分析异常: {e}")
            return None, None, None, False

perception_engine = PerceptionEngine()
