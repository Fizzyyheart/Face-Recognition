"""
人脸检测 + 对齐 + 特征提取（embedding）
封装 InsightFace，对外提供统一接口
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import cv2

# InsightFace 延迟导入（首次调用时加载，节省启动时间）
_face_analyzer = None


@dataclass
class FaceInfo:
    """单张人脸的检测结果"""

    bbox: np.ndarray  # [x1, y1, x2, y2]
    landmarks: np.ndarray  # 5 点关键点 (5, 2)
    det_score: float  # 检测置信度
    embedding: Optional[np.ndarray] = None  # 512-d 特征向量（可选）
    age: Optional[int] = None
    gender: Optional[str] = None

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def min_side(self) -> float:
        return min(self.width, self.height)

    @property
    def center(self) -> tuple:
        cx = (self.bbox[0] + self.bbox[2]) / 2
        cy = (self.bbox[1] + self.bbox[3]) / 2
        return (cx, cy)

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox.tolist(),
            "det_score": self.det_score,
            "width": self.width,
            "height": self.height,
        }


def get_face_analyzer():
    """
    获取 InsightFace FaceAnalysis 单例（延迟加载）
    """
    global _face_analyzer
    if _face_analyzer is None:
        from insightface.app import FaceAnalysis
        from ..config import settings

        # InsightFace 会在 root 目录下寻找 models/buffalo_l
        # 所以 root 应该是 models 的父目录（即项目根目录）
        # 只加载检测和识别模型，禁用 landmark 和 genderage 以提升速度
        _face_analyzer = FaceAnalysis(
            name=settings.INSIGHTFACE_MODEL_NAME,
            root=str(settings.MODEL_DIR.parent),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],  # 只加载必要模型
        )
        # det_size: (640,640) 精度高但慢；(320,320) 速度快适合实时
        # CPU 模式下建议使用较小分辨率
        _face_analyzer.prepare(ctx_id=settings.INSIGHTFACE_CTX_ID, det_size=(320, 320))
    return _face_analyzer


def detect_faces(
    image: np.ndarray,
    extract_embedding: bool = True,
    max_faces: int = 50,
) -> List[FaceInfo]:
    """
    检测图像中的人脸，并可选提取 embedding

    Args:
        image: BGR 格式图像 (H, W, 3)
        extract_embedding: 是否提取 512-d embedding
        max_faces: 最多返回人脸数

    Returns:
        FaceInfo 列表（按检测置信度降序）
    """
    analyzer = get_face_analyzer()
    # InsightFace 返回的 faces 是 Face 对象列表
    faces = analyzer.get(image, max_num=max_faces)

    results: List[FaceInfo] = []
    for face in faces:
        info = FaceInfo(
            bbox=face.bbox.astype(np.float32),
            landmarks=face.kps.astype(np.float32)
            if face.kps is not None
            else np.zeros((5, 2), dtype=np.float32),
            det_score=float(face.det_score),
            embedding=face.embedding
            if extract_embedding and face.embedding is not None
            else None,
            age=int(face.age)
            if hasattr(face, "age") and face.age is not None
            else None,
            gender="M"
            if hasattr(face, "gender") and face.gender == 1
            else ("F" if hasattr(face, "gender") and face.gender == 0 else None),
        )
        results.append(info)

    # 按置信度降序
    results.sort(key=lambda x: x.det_score, reverse=True)
    return results


def extract_embedding(image: np.ndarray, face: FaceInfo) -> np.ndarray:
    """
    对已检测的人脸单独提取 embedding（用于 embedding 为 None 时补提）
    """
    analyzer = get_face_analyzer()
    # 用 bbox 裁剪后重新过模型
    x1, y1, x2, y2 = map(int, face.bbox)
    crop = image[max(0, y1) : y2, max(0, x1) : x2]
    if crop.size == 0:
        return np.zeros(512, dtype=np.float32)
    faces = analyzer.get(crop, max_num=1)
    if faces and faces[0].embedding is not None:
        return faces[0].embedding
    return np.zeros(512, dtype=np.float32)


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    计算两个 embedding 的余弦相似度（范围 -1 ~ 1）
    """
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    return float(np.dot(emb1, emb2) / (norm1 * norm2))


class FaceRecognizer:
    """
    人脸识别器：基于数据库中的人员特征进行识别
    """

    def __init__(self, threshold: float = None):
        """
        初始化识别器，从数据库加载人员特征

        Args:
            threshold: 识别阈值（余弦相似度），默认使用配置值
        """
        from ..config import settings

        self.threshold = threshold or settings.RECOGNITION_THRESHOLD
        self.persons = {}  # {person_id: {"name": str, "embeddings": List[np.ndarray]}}
        self.person_centers = {}  # {person_id: np.ndarray} 每个人的中心向量
        self._load_from_db()

    def _load_from_db(self):
        """从数据库加载人员特征"""
        from ..db.database import SessionLocal
        from ..db.models import Person, FaceSample

        session = SessionLocal()
        try:
            persons = session.query(Person).all()

            for person in persons:
                samples = (
                    session.query(FaceSample)
                    .filter(
                        FaceSample.person_id == person.id,
                        FaceSample.embedding.isnot(None),
                    )
                    .order_by(FaceSample.quality_score.desc())
                    .limit(50)
                    .all()
                )

                if not samples:
                    continue

                embeddings = []
                for s in samples:
                    emb = np.frombuffer(s.embedding, dtype=np.float32)
                    if len(emb) == 512:
                        embeddings.append(emb / np.linalg.norm(emb))

                if embeddings:
                    self.persons[person.id] = {
                        "name": person.name,
                        "embeddings": embeddings,
                    }
                    # 计算中心向量
                    center = np.mean(embeddings, axis=0)
                    self.person_centers[person.id] = center / np.linalg.norm(center)

            print(f"识别器已加载 {len(self.persons)} 人的特征")
        finally:
            session.close()

    def recognize(self, embedding: np.ndarray, top_k: int = 1) -> List[tuple]:
        """
        识别单个人脸

        Args:
            embedding: 512维人脸特征
            top_k: 返回前k个匹配结果

        Returns:
            List of (person_id, name, similarity) 或 空列表（未识别）
        """
        if embedding is None or len(embedding) != 512:
            return []

        emb_norm = embedding / np.linalg.norm(embedding)

        scores = []
        for pid, center in self.person_centers.items():
            sim = float(np.dot(emb_norm, center))
            if sim >= self.threshold:
                scores.append((pid, self.persons[pid]["name"], sim))

        # 按相似度降序
        scores.sort(key=lambda x: -x[2])
        return scores[:top_k]

    def recognize_face(self, face: FaceInfo) -> Optional[tuple]:
        """
        识别 FaceInfo 对象

        Returns:
            (person_id, name, similarity) 或 None
        """
        if face.embedding is None:
            return None
        results = self.recognize(face.embedding, top_k=1)
        return results[0] if results else None
