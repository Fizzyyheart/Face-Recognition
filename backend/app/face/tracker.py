"""
人脸跟踪器：基于 IoU + Embedding 相似度的帧间关联
目的：把同一个人跨帧关联成 track，减少重复样本
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from .detector import FaceInfo, compute_similarity
from .quality import QualityResult
from ..config import settings


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """计算两个 bbox 的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area < 1e-6:
        return 0.0
    return inter_area / union_area


@dataclass
class TrackSample:
    """track 中的一个样本"""

    frame_idx: int
    face: FaceInfo
    quality: QualityResult
    image_crop: Optional[np.ndarray] = None  # 裁剪后的人脸图（可选）


@dataclass
class Track:
    """一个人脸轨迹"""

    track_id: int
    samples: List[TrackSample] = field(default_factory=list)
    last_bbox: Optional[np.ndarray] = None
    age: int = 0  # 连续丢失帧数
    is_active: bool = True

    def add_sample(self, sample: TrackSample):
        self.samples.append(sample)
        self.last_bbox = sample.face.bbox.copy()
        self.age = 0

    def get_best_samples(self, k: int = 10) -> List[TrackSample]:
        """获取质量最高的 k 个样本"""
        # 按 quality_score 降序
        sorted_samples = sorted(
            self.samples, key=lambda s: s.quality.quality_score, reverse=True
        )
        return sorted_samples[:k]

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def best_embedding(self) -> Optional[np.ndarray]:
        """返回质量最高样本的 embedding"""
        best = self.get_best_samples(1)
        if best and best[0].face.embedding is not None:
            return best[0].face.embedding
        return None


class SimpleTracker:
    """
    基于 IoU + Embedding 相似度的跟踪器
    - 首先尝试 IoU 匹配（位置接近）
    - IoU 失败时，使用 embedding 相似度匹配（同一人脸）
    """

    def __init__(
        self,
        iou_threshold: float = None,
        embedding_threshold: float = None,
        max_age: int = None,
        max_samples_per_track: int = None,
    ):
        self.iou_threshold = iou_threshold or settings.TRACK_IOU_THRESHOLD
        self.embedding_threshold = embedding_threshold or settings.RECOGNITION_THRESHOLD
        self.max_age = max_age or settings.TRACK_MAX_AGE
        self.max_samples = max_samples_per_track or settings.TRACK_MAX_SAMPLES

        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1

    def _compute_embedding_similarity(
        self, emb1: Optional[np.ndarray], emb2: Optional[np.ndarray]
    ) -> float:
        """计算两个 embedding 的余弦相似度"""
        if emb1 is None or emb2 is None:
            return 0.0
        return float(compute_similarity(emb1, emb2))

    def _crop_face(
        self, frame: Optional[np.ndarray], face: FaceInfo, margin: float = 0.3
    ) -> Optional[np.ndarray]:
        """
        裁剪人脸区域，带边距扩展
        
        Args:
            frame: 原始帧
            face: 人脸信息
            margin: 边距比例（相对于人脸宽高）
            
        Returns:
            裁剪后的图片，如果裁剪不完整（被边界截断）则返回 None
        """
        if frame is None:
            return None
        
        x1, y1, x2, y2 = face.bbox
        face_w = x2 - x1
        face_h = y2 - y1
        
        # 扩展边距
        margin_w = face_w * margin
        margin_h = face_h * margin
        
        # 期望的裁剪区域
        crop_x1 = x1 - margin_w
        crop_y1 = y1 - margin_h
        crop_x2 = x2 + margin_w
        crop_y2 = y2 + margin_h
        
        expected_w = crop_x2 - crop_x1
        expected_h = crop_y2 - crop_y1
        
        # 边界检查
        h, w = frame.shape[:2]
        crop_x1 = int(max(0, crop_x1))
        crop_y1 = int(max(0, crop_y1))
        crop_x2 = int(min(w, crop_x2))
        crop_y2 = int(min(h, crop_y2))
        
        actual_w = crop_x2 - crop_x1
        actual_h = crop_y2 - crop_y1
        
        # 检查是否被边界截断太多（保留至少 80% 的期望区域）
        if actual_w < expected_w * 0.8 or actual_h < expected_h * 0.8:
            return None
        
        # 检查宽高比是否正常（人脸通常接近正方形，允许 0.6 ~ 1.6）
        aspect_ratio = actual_w / max(actual_h, 1)
        if aspect_ratio < 0.6 or aspect_ratio > 1.6:
            return None
        
        return frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    def _add_face_to_track(
        self,
        track: Track,
        face: FaceInfo,
        quality: QualityResult,
        frame_idx: int,
        frame: Optional[np.ndarray],
        results: List[tuple],
    ):
        """将人脸添加到 track"""
        crop = self._crop_face(frame, face)

        if quality.passed and track.sample_count < self.max_samples:
            sample = TrackSample(
                frame_idx=frame_idx,
                face=face,
                quality=quality,
                image_crop=crop,
            )
            track.add_sample(sample)
        else:
            # 即使不保存样本，也要更新 track 位置
            track.last_bbox = face.bbox.copy()
            track.age = 0

        results.append((track.track_id, face, quality))

    def update(
        self,
        frame_idx: int,
        faces: List[FaceInfo],
        qualities: List[QualityResult],
        frame: Optional[np.ndarray] = None,
    ) -> List[tuple]:
        """
        更新跟踪器

        Args:
            frame_idx: 当前帧索引
            faces: 检测到的人脸列表
            qualities: 对应的质量评估结果
            frame: 原始帧（用于裁剪保存，可选）

        Returns:
            List of (track_id, face, quality) 元组
        """
        results = []

        # 获取活跃 tracks
        active_tracks = [t for t in self.tracks.values() if t.is_active]

        # 第一轮：IoU 匹配（位置接近）
        matched_faces = set()
        matched_tracks = set()

        if active_tracks and faces:
            # 计算 IoU 矩阵
            iou_matrix = np.zeros((len(active_tracks), len(faces)))
            for i, track in enumerate(active_tracks):
                for j, face in enumerate(faces):
                    iou_matrix[i, j] = compute_iou(track.last_bbox, face.bbox)

            # 贪心匹配
            while True:
                max_iou = iou_matrix.max()
                if max_iou < self.iou_threshold:
                    break
                i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                track = active_tracks[i]
                face = faces[j]
                quality = qualities[j]

                self._add_face_to_track(track, face, quality, frame_idx, frame, results)
                matched_faces.add(j)
                matched_tracks.add(track.track_id)

                # 标记已匹配，防止重复
                iou_matrix[i, :] = -1
                iou_matrix[:, j] = -1

        # 第二轮：Embedding 匹配（IoU 失败但是同一人脸）
        unmatched_faces = [j for j in range(len(faces)) if j not in matched_faces]
        unmatched_tracks = [t for t in active_tracks if t.track_id not in matched_tracks]

        if unmatched_tracks and unmatched_faces:
            # 计算 embedding 相似度矩阵
            emb_matrix = np.zeros((len(unmatched_tracks), len(unmatched_faces)))
            for i, track in enumerate(unmatched_tracks):
                track_emb = track.best_embedding
                for j_idx, j in enumerate(unmatched_faces):
                    face_emb = faces[j].embedding
                    emb_matrix[i, j_idx] = self._compute_embedding_similarity(track_emb, face_emb)

            # 贪心匹配
            while True:
                max_sim = emb_matrix.max()
                if max_sim < self.embedding_threshold:
                    break
                i, j_idx = np.unravel_index(emb_matrix.argmax(), emb_matrix.shape)
                track = unmatched_tracks[i]
                j = unmatched_faces[j_idx]
                face = faces[j]
                quality = qualities[j]

                self._add_face_to_track(track, face, quality, frame_idx, frame, results)
                matched_faces.add(j)
                matched_tracks.add(track.track_id)

                # 标记已匹配
                emb_matrix[i, :] = -1
                emb_matrix[:, j_idx] = -1

        # 未匹配的人脸：创建新 track
        for j, (face, quality) in enumerate(zip(faces, qualities)):
            if j in matched_faces:
                continue

            # 在创建新 track 前，检查是否与任何已有 track（包括不活跃的）相似
            best_match_track = None
            best_sim = 0.0
            if face.embedding is not None:
                for track in self.tracks.values():
                    track_emb = track.best_embedding
                    sim = self._compute_embedding_similarity(track_emb, face.embedding)
                    if sim > best_sim and sim >= self.embedding_threshold:
                        best_sim = sim
                        best_match_track = track

            if best_match_track is not None:
                # 重新激活匹配到的 track
                best_match_track.is_active = True
                self._add_face_to_track(best_match_track, face, quality, frame_idx, frame, results)
            else:
                # 创建新 track
                track = Track(track_id=self.next_track_id)
                self.next_track_id += 1

                crop = self._crop_face(frame, face)
                if quality.passed:
                    sample = TrackSample(
                        frame_idx=frame_idx,
                        face=face,
                        quality=quality,
                        image_crop=crop,
                    )
                    track.add_sample(sample)

                track.last_bbox = face.bbox.copy()
                self.tracks[track.track_id] = track
                results.append((track.track_id, face, quality))

        # 更新未匹配的 tracks 的 age
        for track in active_tracks:
            if track.track_id not in matched_tracks:
                track.age += 1
                if track.age > self.max_age:
                    track.is_active = False

        return results

    def get_all_tracks(self) -> List[Track]:
        """获取所有 tracks（包括已结束的）"""
        return list(self.tracks.values())

    def get_active_tracks(self) -> List[Track]:
        """获取活跃 tracks"""
        return [t for t in self.tracks.values() if t.is_active]

    def get_finished_tracks(self, min_samples: int = 1) -> List[Track]:
        """获取已结束且样本数 >= min_samples 的 tracks"""
        return [
            t
            for t in self.tracks.values()
            if not t.is_active and t.sample_count >= min_samples
        ]
