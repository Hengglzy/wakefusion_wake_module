"""
唇动检测独立模块 (Lip-Sync VAD)
基于 MediaPipe Face Mesh，通过计算嘴唇内部纵横比(MAR)的动态方差，判断用户是否在说话。
"""
import mediapipe as mp
import numpy as np
from collections import deque
import logging

logger = logging.getLogger("lip_sync")
logger.setLevel(logging.INFO)

class LipSyncDetector:
    def __init__(self, history_len=5, variance_threshold=0.0003, mar_closed_threshold=0.10):
        """
        初始化唇动检测器
        Args:
            history_len: 追踪历史帧数（默认5帧，约0.15秒）
            variance_threshold: 判定为说话的 MAR 方差阈值（越小越灵敏，默认0.0003）
            mar_closed_threshold: 嘴巴闭合的绝对 MAR 阈值（低于此值直接判定为不说话）
        """
        self.history_len = history_len
        self.variance_threshold = variance_threshold
        self.mar_closed_threshold = mar_closed_threshold
        self.mar_history = deque(maxlen=history_len)
        
        # 状态稳定性：要求连续多帧都是 talking 才输出 True（减少抖动误报）
        self._talking_confirm_count = 0
        self._talking_confirm_threshold = 2  # 🌟 2帧防抖
        
        # 初始化 MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        logger.info(f"✅ 唇动检测模块已初始化 (history={history_len}, variance_threshold={variance_threshold}, mar_closed_threshold={mar_closed_threshold})")

    def start_sync(self):
        """启动口型同步检测（已废弃，保留接口兼容性）"""
        pass
    
    def stop_sync(self):
        """停止口型同步检测（已废弃，保留接口兼容性）"""
        pass
    
    def process_frame(self, frame_rgb: np.ndarray) -> bool:
        """
        处理单帧图像，判断是否在说话
        
        优化策略：
        1. 绝对门限：如果 MAR 太小（嘴巴几乎闭合），直接判定为不说话
        2. 方差判断：只有明显的连续开合动作（方差足够大）才判定为说话
        3. 状态稳定性：要求连续多帧确认，减少抖动误报
        
        Args:
            frame_rgb: RGB格式的图像 numpy 数组
        Returns:
            bool: 是否在说话 (is_talking)
        """
        is_talking = False
        try:
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # 获取嘴唇关键点坐标（Face Mesh 的固定索引）
                # 13: 上唇内缘中心, 14: 下唇内缘中心
                # 78: 左嘴角, 308: 右嘴角
                top_lip = landmarks[13]
                bottom_lip = landmarks[14]
                left_lip = landmarks[78]
                right_lip = landmarks[308]
                
                # 计算垂直距离和水平距离
                v_dist = np.sqrt((top_lip.x - bottom_lip.x)**2 + (top_lip.y - bottom_lip.y)**2)
                h_dist = np.sqrt((left_lip.x - right_lip.x)**2 + (left_lip.y - right_lip.y)**2)
                
                if h_dist > 0:
                    mar = v_dist / h_dist
                    
                    # 🌟 优化1：绝对门限 - 如果嘴巴几乎闭合，直接判定为不说话
                    if mar < self.mar_closed_threshold:
                        # 嘴巴闭合，清空历史，避免历史数据干扰
                        self.mar_history.clear()
                        self._talking_confirm_count = 0
                        is_talking = False
                    else:
                        # 嘴巴张开，记录 MAR 历史
                        self.mar_history.append(mar)
                        
                        # 只有历史数据积累够了，才进行方差判断
                        if len(self.mar_history) == self.history_len:
                            mar_variance = float(np.var(self.mar_history))
                            
                            # 🌟 优化2：提高方差阈值，只有明显的连续开合动作才算说话
                            # 使用初始化时的 variance_threshold（默认 0.0005），而不是硬编码的 0.0001
                            if mar_variance > self.variance_threshold:
                                # 本帧判定为 talking
                                self._talking_confirm_count += 1
                            else:
                                # 本帧判定为 not talking，重置计数器
                                self._talking_confirm_count = 0
                            
                            # 🌟 优化3：状态稳定性 - 要求连续多帧确认才输出 True
                            if self._talking_confirm_count >= self._talking_confirm_threshold:
                                is_talking = True
                            else:
                                is_talking = False
                        else:
                            # 历史数据还不够，保持当前状态或默认 False
                            is_talking = False
            else:
                # 没检测到人脸，清空历史状态
                self.mar_history.clear()
                self._talking_confirm_count = 0
                
        except Exception as e:
            logger.error(f"唇动检测异常: {e}")
            
        return is_talking

    def close(self):
        """释放资源"""
        self.face_mesh.close()
