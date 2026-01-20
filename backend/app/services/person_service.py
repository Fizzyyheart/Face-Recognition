"""
人员管理服务
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
from sqlalchemy.orm import Session

from backend.app.db.models import Person, FaceSample, Cluster, ClusterStatus
from backend.app.face.detector import detect_faces
from backend.app.config import settings


class PersonService:
    """人员管理服务"""

    def __init__(self, db_session: Session):
        self.db = db_session
        self.seeds_dir = settings.SEEDS_DIR
        self.seeds_dir.mkdir(parents=True, exist_ok=True)

    def get_all_persons(self) -> List[Person]:
        """获取所有人员"""
        return self.db.query(Person).order_by(Person.name).all()

    def get_person_by_id(self, person_id: int) -> Optional[Person]:
        """根据ID获取人员"""
        return self.db.query(Person).filter(Person.id == person_id).first()

    def get_person_by_name(self, name: str) -> Optional[Person]:
        """根据姓名获取人员"""
        return self.db.query(Person).filter(Person.name == name).first()

    def add_person(
        self,
        name: str,
        student_id: Optional[str] = None,
        face_image: Optional[np.ndarray] = None,
        face_image_path: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[Person]]:
        """
        添加新人员

        Args:
            name: 姓名
            student_id: 学号（可选）
            face_image: 人脸图像 (BGR numpy array)
            face_image_path: 人脸图像路径

        Returns:
            (success, message, person)
        """
        # 检查姓名是否已存在
        existing = self.get_person_by_name(name)
        if existing:
            return False, f"人员 '{name}' 已存在", None

        # 检查学号是否已存在
        if student_id:
            existing_sid = (
                self.db.query(Person).filter(Person.student_id == student_id).first()
            )
            if existing_sid:
                return False, f"学号 '{student_id}' 已被使用", None

        # 加载图像
        if face_image is None and face_image_path:
            # 支持中文路径
            img_data = np.fromfile(face_image_path, dtype=np.uint8)
            face_image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if face_image is None:
                return False, f"无法读取图像: {face_image_path}", None

        # 检测人脸并提取特征
        embedding = None
        if face_image is not None:
            faces = detect_faces(face_image, extract_embedding=True)
            if not faces:
                return False, "图像中未检测到人脸", None
            if len(faces) > 1:
                return (
                    False,
                    f"图像中检测到多张人脸({len(faces)}), 请提供单人照片",
                    None,
                )

            embedding = faces[0].embedding

            # 保存种子图像
            seed_path = self.seeds_dir / f"{name}.jpg"
            cv2.imencode(".jpg", face_image)[1].tofile(str(seed_path))

        # 创建人员记录
        person = Person(name=name, student_id=student_id)
        self.db.add(person)
        self.db.flush()

        # 如果有特征向量，创建 FaceSample
        if embedding is not None:
            sample = FaceSample(
                person_id=person.id,
                embedding=embedding.tobytes(),
                quality_score=1.0,
                image_path=str(self.seeds_dir / f"{name}.jpg"),
            )
            self.db.add(sample)

        self.db.commit()
        return True, f"成功添加人员: {name}", person

    def delete_person(self, person_id: int) -> Tuple[bool, str]:
        """
        删除人员及其所有关联数据

        Args:
            person_id: 人员ID

        Returns:
            (success, message)
        """
        person = self.get_person_by_id(person_id)
        if not person:
            return False, f"未找到ID为 {person_id} 的人员"

        name = person.name

        # 删除种子图像
        seed_path = self.seeds_dir / f"{name}.jpg"
        if seed_path.exists():
            seed_path.unlink()

        # 删除关联的 FaceSample（但不删除图像文件，因为可能被其他地方引用）
        self.db.query(FaceSample).filter(FaceSample.person_id == person_id).delete()

        # 更新关联的 Cluster
        self.db.query(Cluster).filter(Cluster.person_id == person_id).update(
            {"person_id": None, "status": ClusterStatus.PENDING.value}
        )

        # 删除人员
        self.db.delete(person)
        self.db.commit()

        return True, f"成功删除人员: {name}"

    def update_person(
        self,
        person_id: int,
        name: Optional[str] = None,
        student_id: Optional[str] = None,
        face_image: Optional[np.ndarray] = None,
    ) -> Tuple[bool, str]:
        """
        更新人员信息

        Args:
            person_id: 人员ID
            name: 新姓名
            student_id: 新学号
            face_image: 新人脸图像

        Returns:
            (success, message)
        """
        person = self.get_person_by_id(person_id)
        if not person:
            return False, f"未找到ID为 {person_id} 的人员"

        old_name = person.name

        if name and name != person.name:
            # 检查新姓名是否已存在
            existing = self.get_person_by_name(name)
            if existing:
                return False, f"人员 '{name}' 已存在"

            # 重命名种子图像
            old_seed = self.seeds_dir / f"{old_name}.jpg"
            new_seed = self.seeds_dir / f"{name}.jpg"
            if old_seed.exists():
                old_seed.rename(new_seed)

            person.name = name

        if student_id is not None:
            person.student_id = student_id

        # 更新人脸特征
        if face_image is not None:
            faces = detect_faces(face_image, extract_embedding=True)
            if not faces:
                return False, "图像中未检测到人脸"
            if len(faces) > 1:
                return False, f"图像中检测到多张人脸({len(faces)})"

            # 保存新种子图像
            current_name = name or person.name
            seed_path = self.seeds_dir / f"{current_name}.jpg"
            cv2.imencode(".jpg", face_image)[1].tofile(str(seed_path))

            # 更新或添加 FaceSample
            sample = FaceSample(
                person_id=person.id,
                embedding=faces[0].embedding.tobytes(),
                quality_score=1.0,
                image_path=str(seed_path),
            )
            self.db.add(sample)

        self.db.commit()
        return True, f"成功更新人员信息"

    def get_person_stats(self, person_id: int) -> dict:
        """获取人员统计信息"""
        person = self.get_person_by_id(person_id)
        if not person:
            return {}

        sample_count = (
            self.db.query(FaceSample).filter(FaceSample.person_id == person_id).count()
        )

        seed_path = self.seeds_dir / f"{person.name}.jpg"

        return {
            "id": person.id,
            "name": person.name,
            "student_id": person.student_id,
            "sample_count": sample_count,
            "has_seed": seed_path.exists(),
            "seed_path": str(seed_path) if seed_path.exists() else None,
            "created_at": person.created_at.isoformat() if person.created_at else None,
        }

    def register_from_camera(
        self, name: str, frame: np.ndarray, student_id: Optional[str] = None
    ) -> Tuple[bool, str, Optional[Person]]:
        """
        从摄像头帧注册新人员

        Args:
            name: 姓名
            frame: 摄像头帧
            student_id: 学号

        Returns:
            (success, message, person)
        """
        # 检测人脸
        faces = detect_faces(frame, extract_embedding=True)
        if not faces:
            return False, "未检测到人脸，请正对摄像头", None
        if len(faces) > 1:
            return False, f"检测到多张人脸({len(faces)})，请确保只有一人", None

        # 获取最佳人脸区域
        face = faces[0]
        x1, y1, x2, y2 = map(int, face.bbox)

        # 扩大裁剪区域
        h, w = frame.shape[:2]
        pad = 30
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        face_crop = frame[y1:y2, x1:x2]

        return self.add_person(name, student_id, face_crop)
