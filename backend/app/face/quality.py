"""
人脸质量评估：模糊度、尺寸、姿态角度
用于过滤低质量样本，提升识别稳定性
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .detector import FaceInfo
from ..config import settings


@dataclass
class QualityResult:
    """质量评估结果"""

    passed: bool  # 是否通过质量门控
    blur_score: float  # 模糊分数（Laplacian 方差，越高越清晰）
    size_ok: bool  # 尺寸是否满足
    pose_ok: bool  # 姿态是否满足
    yaw: float = 0.0  # 水平转头角度（度）
    pitch: float = 0.0  # 俯仰角度（度）
    reason: Optional[str] = None  # 未通过原因

    @property
    def quality_score(self) -> float:
        """综合质量分（用于 track 内排序选最优样本）"""
        # 模糊分数归一化到 0~1，加上尺寸和姿态惩罚
        blur_norm = min(self.blur_score / 500.0, 1.0)
        pose_penalty = 1.0 - (abs(self.yaw) + abs(self.pitch)) / 180.0
        return blur_norm * 0.6 + pose_penalty * 0.4


def compute_blur_score(image: np.ndarray) -> float:
    """
    计算图像模糊度（Laplacian 方差）
    值越高表示越清晰
    """
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def estimate_pose_from_landmarks(landmarks: np.ndarray) -> tuple:
    """
    根据 5 点关键点粗略估计 yaw/pitch
    landmarks: shape (5, 2)，顺序: 左眼、右眼、鼻尖、左嘴角、右嘴角

    Returns:
        (yaw, pitch) 单位为度
    """
    if landmarks is None or landmarks.shape[0] < 5:
        return 0.0, 0.0

    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]
    left_mouth = landmarks[3]
    right_mouth = landmarks[4]

    # 双眼中心
    eye_center = (left_eye + right_eye) / 2
    # 嘴巴中心
    mouth_center = (left_mouth + right_mouth) / 2

    # 粗略 yaw：鼻尖相对于双眼中心的水平偏移
    eye_dist = np.linalg.norm(right_eye - left_eye)
    if eye_dist < 1e-6:
        return 0.0, 0.0

    # 鼻尖到双眼中心线的水平偏移（归一化）
    yaw_ratio = (nose[0] - eye_center[0]) / eye_dist
    yaw = float(np.clip(yaw_ratio * 90, -90, 90))  # 粗略映射到 -90~90 度

    # 粗略 pitch：鼻尖到眼-嘴中线的垂直位置
    face_height = np.linalg.norm(mouth_center - eye_center)
    if face_height < 1e-6:
        return yaw, 0.0

    # 鼻尖应该在眼和嘴之间，偏上=抬头，偏下=低头
    expected_nose_y = eye_center[1] + face_height * 0.4
    pitch_ratio = (nose[1] - expected_nose_y) / face_height
    pitch = float(np.clip(pitch_ratio * 60, -45, 45))

    return yaw, pitch


def assess_quality(
    image: np.ndarray,
    face: FaceInfo,
    min_size: int = None,
    blur_threshold: float = None,
    max_pose_angle: float = None,
) -> QualityResult:
    """
    综合评估人脸质量

    Args:
        image: 原始帧图像 (BGR)
        face: 检测到的人脸信息
        min_size: 人脸框短边最小像素（默认用配置）
        blur_threshold: 模糊阈值（默认用配置）
        max_pose_angle: 最大姿态角度（默认用配置）

    Returns:
        QualityResult
    """
    min_size = min_size or settings.MIN_FACE_SIZE
    blur_threshold = blur_threshold or settings.BLUR_THRESHOLD
    max_pose_angle = max_pose_angle or settings.MAX_POSE_ANGLE

    # 1. 尺寸检查
    size_ok = face.min_side >= min_size

    # 2. 裁剪人脸区域用于模糊检测
    x1, y1, x2, y2 = map(int, face.bbox)
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    face_crop = image[y1:y2, x1:x2]

    blur_score = compute_blur_score(face_crop)
    blur_ok = blur_score >= blur_threshold

    # 3. 姿态估计
    yaw, pitch = estimate_pose_from_landmarks(face.landmarks)
    pose_ok = abs(yaw) <= max_pose_angle and abs(pitch) <= max_pose_angle

    # 综合判断
    passed = size_ok and blur_ok and pose_ok

    reason = None
    if not passed:
        reasons = []
        if not size_ok:
            reasons.append(f"尺寸过小({face.min_side:.0f}<{min_size})")
        if not blur_ok:
            reasons.append(f"模糊({blur_score:.1f}<{blur_threshold})")
        if not pose_ok:
            reasons.append(f"姿态过大(yaw={yaw:.1f}, pitch={pitch:.1f})")
        reason = "; ".join(reasons)

    return QualityResult(
        passed=passed,
        blur_score=blur_score,
        size_ok=size_ok,
        pose_ok=pose_ok,
        yaw=yaw,
        pitch=pitch,
        reason=reason,
    )
