"""
活体检测模块 - 基于Silent-Face-Anti-Spoofing的ONNX版本
使用双模型融合判断是否为真实人脸，无需PyTorch依赖

标签说明:
    0: 纸质照片攻击
    1: 真实人脸
    2: 屏幕翻拍攻击
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import Dict, Tuple, List, Optional
from pathlib import Path


class AntiSpoofDetector:
    """
    活体检测器，使用ONNX Runtime进行推理
    融合两个MiniFASNet模型的预测结果
    """

    # 类标签
    LABELS = {0: "纸质照片", 1: "真实人脸", 2: "屏幕翻拍"}

    def __init__(
        self,
        model_dir: str = "models/anti_spoof",
        threshold: float = 0.8,
        use_gpu: bool = True,
    ):
        """
        初始化活体检测器

        Args:
            model_dir: ONNX模型目录
            threshold: 真实人脸判断阈值
            use_gpu: 是否使用GPU加速
        """
        self.model_dir = Path(model_dir)
        self.threshold = threshold
        self.input_size = (80, 80)
        self.sessions: Dict[str, ort.InferenceSession] = {}

        # 配置执行提供者
        if use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        # 加载所有ONNX模型
        self._load_models(providers)

        print(f"[AntiSpoof] 已加载 {len(self.sessions)} 个活体检测模型")

    def _load_models(self, providers: List[str]) -> None:
        """加载模型目录下的所有ONNX模型"""
        if not self.model_dir.exists():
            raise FileNotFoundError(f"模型目录不存在: {self.model_dir}")

        onnx_files = list(self.model_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"未找到ONNX模型: {self.model_dir}")

        for model_path in onnx_files:
            model_name = model_path.name
            session = ort.InferenceSession(str(model_path), providers=providers)
            self.sessions[model_name] = session
            print(f"  - 已加载: {model_name}")

    def _parse_model_scale(self, model_name: str) -> float:
        """
        从模型名称解析scale参数

        Examples:
            "2.7_80x80_MiniFASNetV2.onnx" -> 2.7
            "4_0_0_80x80_MiniFASNetV1SE.onnx" -> 4.0
        """
        parts = model_name.split("_")
        if parts[0] == "4":
            return 4.0
        return float(parts[0])

    def _get_crop_box(
        self, img_w: int, img_h: int, bbox: Tuple[int, int, int, int], scale: float
    ) -> Tuple[int, int, int, int]:
        """
        根据scale参数扩展人脸框

        Args:
            img_w, img_h: 图像尺寸
            bbox: 人脸框 (x, y, w, h)
            scale: 扩展比例 (2.7 或 4.0)

        Returns:
            扩展后的裁剪框 (x1, y1, x2, y2)
        """
        x, y, w, h = bbox

        # 计算中心点
        center_x = x + w / 2
        center_y = y + h / 2

        # 扩展尺寸
        new_size = int(max(w, h) * scale)
        half_size = new_size / 2

        # 计算新边界
        x1 = int(center_x - half_size)
        y1 = int(center_y - half_size)
        x2 = int(center_x + half_size)
        y2 = int(center_y + half_size)

        # 裁剪到图像边界内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w - 1, x2)
        y2 = min(img_h - 1, y2)

        return x1, y1, x2, y2

    def _preprocess(
        self, img_bgr: np.ndarray, bbox: Tuple[int, int, int, int], scale: float
    ) -> np.ndarray:
        """
        预处理人脸图像

        Args:
            img_bgr: BGR格式图像
            bbox: 人脸框 (x, y, w, h)
            scale: 裁剪扩展比例

        Returns:
            模型输入张量 (1, 3, 80, 80)
        """
        h, w = img_bgr.shape[:2]

        # 获取扩展裁剪框
        x1, y1, x2, y2 = self._get_crop_box(w, h, bbox, scale)

        # 裁剪人脸
        face_crop = img_bgr[y1 : y2 + 1, x1 : x2 + 1]

        # 检查裁剪是否有效
        if face_crop.size == 0:
            # 返回零张量
            return np.zeros((1, 3, 80, 80), dtype=np.float32)

        # 调整大小
        face_resized = cv2.resize(face_crop, self.input_size)

        # 转换为float32，保持BGR格式（模型训练时使用BGR）
        face_float = face_resized.astype(np.float32)

        # HWC -> CHW
        face_chw = np.transpose(face_float, (2, 0, 1))

        # 添加batch维度
        return np.expand_dims(face_chw, axis=0)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """计算softmax"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def predict(self, img_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
        """
        执行活体检测

        Args:
            img_bgr: BGR格式图像
            bbox: 人脸框 (x, y, w, h)

        Returns:
            检测结果字典:
                - is_real: 是否为真实人脸
                - label: 类别标签 (0=纸质照片, 1=真实, 2=屏幕翻拍)
                - label_text: 类别文本
                - confidence: 预测置信度
                - scores: 各类别得分
        """
        # 累积所有模型的预测
        predictions = np.zeros((1, 3))

        for model_name, session in self.sessions.items():
            # 获取模型对应的scale参数
            scale = self._parse_model_scale(model_name)

            # 预处理
            input_data = self._preprocess(img_bgr, bbox, scale)

            # 推理
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            output = session.run([output_name], {input_name: input_data})[0]

            # Softmax
            probs = self._softmax(output)
            predictions += probs

        # 平均融合
        predictions = predictions / len(self.sessions)
        scores = predictions[0]

        # 获取预测标签
        label = int(np.argmax(scores))
        confidence = float(scores[label])

        # 判断是否为真实人脸
        is_real = (label == 1) and (confidence >= self.threshold)

        return {
            "is_real": is_real,
            "label": label,
            "label_text": self.LABELS[label],
            "confidence": confidence,
            "scores": {
                "paper": float(scores[0]),
                "real": float(scores[1]),
                "screen": float(scores[2]),
            },
        }

    def predict_batch(
        self, img_bgr: np.ndarray, bboxes: List[Tuple[int, int, int, int]]
    ) -> List[Dict]:
        """
        批量活体检测

        Args:
            img_bgr: BGR格式图像
            bboxes: 人脸框列表

        Returns:
            检测结果列表
        """
        return [self.predict(img_bgr, bbox) for bbox in bboxes]


# 单例实例
_detector: Optional[AntiSpoofDetector] = None


def get_anti_spoof_detector(
    model_dir: str = "models/anti_spoof", threshold: float = 0.8, use_gpu: bool = True
) -> AntiSpoofDetector:
    """获取活体检测器单例"""
    global _detector
    if _detector is None:
        _detector = AntiSpoofDetector(model_dir, threshold, use_gpu)
    return _detector


def check_liveness(img_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
    """
    便捷函数：检测单个人脸的活体状态

    Args:
        img_bgr: BGR格式图像
        bbox: 人脸框 (x, y, w, h)

    Returns:
        检测结果字典
    """
    detector = get_anti_spoof_detector()
    return detector.predict(img_bgr, bbox)
