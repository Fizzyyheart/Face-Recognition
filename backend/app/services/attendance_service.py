"""
考勤签到服务
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict

import numpy as np
from sqlalchemy.orm import Session as DBSession
from sqlalchemy import and_

from backend.app.db.models import (
    Person,
    Session,
    Attendance,
    AttendanceStatus,
    RecognitionEvent,
)
from backend.app.face.detector import FaceRecognizer, detect_faces
from backend.app.face.anti_spoof import AntiSpoofDetector
from backend.app.config import settings


class AttendanceService:
    """考勤签到服务"""

    def __init__(self, db_session: DBSession):
        self.db = db_session
        self.recognizer: Optional[FaceRecognizer] = None
        self.anti_spoof: Optional[AntiSpoofDetector] = None
        self.current_session: Optional[Session] = None
        self._recognized_persons: Dict[int, datetime] = {}  # 防止重复签到

        # 活体检测配置
        self.enable_anti_spoof: bool = True  # 是否启用活体检测
        self.anti_spoof_threshold: float = 0.8  # 活体检测阈值

    def load_recognizer(self, threshold: Optional[float] = None) -> None:
        """加载识别器"""
        if self.recognizer is None:
            self.recognizer = FaceRecognizer(threshold=threshold)

    def load_anti_spoof(
        self,
        model_dir: str = "models/anti_spoof",
        threshold: float = 0.8,
        use_gpu: bool = True,
    ) -> None:
        """加载活体检测器"""
        if self.anti_spoof is None:
            self.anti_spoof = AntiSpoofDetector(
                model_dir=model_dir, threshold=threshold, use_gpu=use_gpu
            )
            self.anti_spoof_threshold = threshold

    def set_anti_spoof_enabled(self, enabled: bool) -> None:
        """设置是否启用活体检测"""
        self.enable_anti_spoof = enabled

    # ==================== 会话管理 ====================

    def create_session(
        self,
        name: str,
        late_after_sec: int = 600,
        start_time: Optional[datetime] = None,
    ) -> Session:
        """
        创建新的考勤会话

        Args:
            name: 会话名称（如"机器视觉第3周"）
            late_after_sec: 迟到阈值（秒），默认10分钟
            start_time: 开始时间，默认为当前时间
        """
        # 结束所有活跃会话
        self.db.query(Session).filter(Session.is_active == True).update(
            {"is_active": False, "end_time": datetime.now()}
        )

        session = Session(
            name=name,
            start_time=start_time or datetime.now(),
            late_after_sec=late_after_sec,
            is_active=True,
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)  # 刷新对象，确保它与数据库同步

        self.current_session = None  # 不缓存，强制重新查询
        self._recognized_persons.clear()
        return session

    def end_session(self, session_id: Optional[int] = None) -> Tuple[bool, str]:
        """结束考勤会话"""
        if session_id:
            session = self.db.query(Session).filter(Session.id == session_id).first()
        else:
            # 不使用缓存的current_session，而是直接查询活跃会话
            session = self.db.query(Session).filter(Session.is_active == True).first()

        if not session:
            return False, "没有活跃的会话"

        session_name = session.name  # 先保存名称
        session.is_active = False
        session.end_time = datetime.now()
        self.db.commit()

        self.current_session = None
        self._recognized_persons.clear()

        return True, f"会话 '{session_name}' 已结束"

    def get_active_session(self) -> Optional[Session]:
        """获取当前活跃会话"""
        if self.current_session and self.current_session.is_active:
            return self.current_session

        self.current_session = (
            self.db.query(Session).filter(Session.is_active == True).first()
        )
        return self.current_session

    def get_all_sessions(self) -> List[Session]:
        """获取所有会话"""
        return self.db.query(Session).order_by(Session.created_at.desc()).all()

    def delete_session(self, session_id: int) -> Tuple[bool, str]:
        """删除会话及其相关的所有考勤记录"""
        session = self.db.query(Session).filter(Session.id == session_id).first()

        if not session:
            return False, "会话不存在"

        if session.is_active:
            return False, "无法删除活跃会话，请先结束会话"

        session_name = session.name

        # 删除相关的考勤记录
        self.db.query(Attendance).filter(Attendance.session_id == session_id).delete()

        # 删除相关的识别事件
        self.db.query(RecognitionEvent).filter(
            RecognitionEvent.session_id == session_id
        ).delete()

        # 删除会话
        self.db.delete(session)
        self.db.commit()

        return True, f"会话 '{session_name}' 已删除"

    # ==================== 签到处理 ====================

    def process_frame(
        self, frame: np.ndarray, record_event: bool = False
    ) -> List[Dict]:
        """
        处理摄像头帧，执行识别和签到

        Args:
            frame: 摄像头帧 (BGR)
            record_event: 是否记录识别事件

        Returns:
            识别结果列表 [{"person_id", "name", "similarity", "bbox", "is_new_checkin", "is_real", "spoof_type"}, ...]
        """
        if self.recognizer is None:
            self.load_recognizer()

        # 获取活跃会话（使用缓存，避免频繁查询数据库）
        session = self.get_active_session()
        session_id = session.id if session else None
        results = []

        # 检测人脸
        faces = detect_faces(frame, extract_embedding=True)

        for face in faces:
            bbox = face.bbox.tolist()

            # 先执行人脸识别
            result = self.recognizer.recognize_face(face)

            if result:
                person_id, name, similarity = result

                # 检查是否需要签到（使用缓存快速判断）
                cache_key = (session_id, person_id) if session_id else None
                needs_checkin = cache_key and cache_key not in self._recognized_persons

                # 只有需要签到时才执行活体检测
                is_real = True
                spoof_type = None

                if needs_checkin and self.enable_anti_spoof:
                    # 懒加载活体检测器
                    if self.anti_spoof is None:
                        try:
                            self.load_anti_spoof()
                        except Exception as e:
                            print(f"[警告] 活体检测器加载失败: {e}")
                            self.enable_anti_spoof = False

                    if self.anti_spoof is not None:
                        bbox_xywh = (
                            int(bbox[0]),
                            int(bbox[1]),
                            int(bbox[2] - bbox[0]),
                            int(bbox[3] - bbox[1]),
                        )
                        spoof_result = self.anti_spoof.predict(frame, bbox_xywh)
                        is_real = spoof_result["is_real"]
                        spoof_type = spoof_result["label_text"] if not is_real else None

                # 活体检测通过才能签到
                if is_real:
                    is_new = (
                        self._handle_checkin(person_id, session) if session else False
                    )
                    results.append(
                        {
                            "person_id": person_id,
                            "name": name,
                            "similarity": similarity,
                            "bbox": bbox,
                            "is_new_checkin": is_new,
                            "is_real": True,
                            "spoof_type": None,
                        }
                    )
                else:
                    # 活体检测失败 - 拒绝签到
                    results.append(
                        {
                            "person_id": person_id,
                            "name": f"{name} [假脸]",
                            "similarity": similarity,
                            "bbox": bbox,
                            "is_new_checkin": False,
                            "is_real": False,
                            "spoof_type": spoof_type,
                        }
                    )
            else:
                # 未识别的人脸
                results.append(
                    {
                        "person_id": None,
                        "name": "Unknown",
                        "similarity": 0.0,
                        "bbox": bbox,
                        "is_new_checkin": False,
                        "is_real": True,
                        "spoof_type": None,
                    }
                )

        return results

    def _handle_checkin(self, person_id: int, session: Optional[Session]) -> bool:
        """
        处理签到逻辑
        注意：每个会话中，每人只能打卡一次，不会重复计算
        使用内存缓存快速去重，避免频繁数据库查询

        Returns:
            是否为新签到（首次识别）
        """
        if not session:
            return False

        # 使用session.id而不是直接访问session对象的其他属性，减少detached问题
        session_id = session.id

        # 快速检查：如果已在内存缓存中，直接返回False（不是新签到）
        cache_key = (session_id, person_id)
        if cache_key in self._recognized_persons:
            return False

        now = datetime.now()

        # 查找是否已有签到记录（仅在缓存未命中时查询数据库）
        attendance = (
            self.db.query(Attendance)
            .filter(
                and_(
                    Attendance.session_id == session_id,
                    Attendance.person_id == person_id,
                )
            )
            .first()
        )

        if attendance:
            # 数据库中已有记录，添加到缓存并返回False
            self._recognized_persons[cache_key] = now
            return False
        else:
            # 创建新签到记录（仅第一次）
            # 先获取session的所有需要属性，避免在commit后访问detached对象
            start_time = session.start_time
            late_after_sec = session.late_after_sec

            # 计算是否迟到
            if start_time:
                elapsed = (now - start_time).total_seconds()
                if elapsed > late_after_sec:
                    status = AttendanceStatus.LATE.value
                else:
                    status = AttendanceStatus.PRESENT.value
            else:
                status = AttendanceStatus.PRESENT.value

            attendance = Attendance(
                session_id=session_id,
                person_id=person_id,
                first_seen=now,
                last_seen=now,
                status=status,
                seen_count=1,
            )
            self.db.add(attendance)
            self.db.commit()

            # 记录到缓存（使用session_id + person_id作为key）
            self._recognized_persons[cache_key] = now

            return True

    def _record_event(
        self, session_id: int, person_id: int, similarity: float, bbox: list
    ) -> None:
        """记录识别事件"""
        import json

        event = RecognitionEvent(
            session_id=session_id,
            person_id=person_id,
            similarity_score=similarity,
            bbox_json=json.dumps(bbox),
        )
        self.db.add(event)
        self.db.commit()

    def manual_checkin(
        self, person_id: int, session_id: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        手动签到

        Args:
            person_id: 人员ID
            session_id: 会话ID，默认为当前会话
        """
        session = (
            self.db.query(Session).filter(Session.id == session_id).first()
            if session_id
            else self.get_active_session()
        )

        if not session:
            return False, "没有活跃的会话"

        person = self.db.query(Person).filter(Person.id == person_id).first()
        if not person:
            return False, f"未找到ID为 {person_id} 的人员"

        # 检查是否已签到
        existing = (
            self.db.query(Attendance)
            .filter(
                and_(
                    Attendance.session_id == session.id,
                    Attendance.person_id == person_id,
                )
            )
            .first()
        )

        if existing:
            return False, f"{person.name} 已签到"

        attendance = Attendance(
            session_id=session.id,
            person_id=person_id,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            status=AttendanceStatus.PRESENT.value,
            seen_count=1,
        )
        self.db.add(attendance)
        self.db.commit()

        return True, f"{person.name} 手动签到成功"

    # ==================== 统计查询 ====================

    def get_session_attendance(self, session_id: int) -> Dict:
        """
        获取会话的考勤统计

        Returns:
            {
                "session": Session,
                "total_persons": int,
                "checked_in": int,
                "present": int,
                "late": int,
                "absent": int,
                "attendance_list": [...]
            }
        """
        session = self.db.query(Session).filter(Session.id == session_id).first()
        if not session:
            return {}

        total_persons = self.db.query(Person).count()
        attendances = (
            self.db.query(Attendance).filter(Attendance.session_id == session_id).all()
        )

        present = sum(
            1 for a in attendances if a.status == AttendanceStatus.PRESENT.value
        )
        late = sum(1 for a in attendances if a.status == AttendanceStatus.LATE.value)
        checked_in = len(attendances)
        absent = total_persons - checked_in

        # 获取详细列表
        attendance_list = []
        for att in attendances:
            person = self.db.query(Person).filter(Person.id == att.person_id).first()
            attendance_list.append(
                {
                    "person_id": att.person_id,
                    "name": person.name if person else "Unknown",
                    "student_id": person.student_id if person else None,
                    "status": att.status,
                    "first_seen": att.first_seen.isoformat()
                    if att.first_seen
                    else None,
                    "last_seen": att.last_seen.isoformat() if att.last_seen else None,
                    "seen_count": att.seen_count,
                }
            )

        # 添加未签到人员
        checked_ids = {a.person_id for a in attendances}
        all_persons = self.db.query(Person).all()
        for person in all_persons:
            if person.id not in checked_ids:
                attendance_list.append(
                    {
                        "person_id": person.id,
                        "name": person.name,
                        "student_id": person.student_id,
                        "status": AttendanceStatus.ABSENT.value,
                        "first_seen": None,
                        "last_seen": None,
                        "seen_count": 0,
                    }
                )

        return {
            "session": {
                "id": session.id,
                "name": session.name,
                "start_time": session.start_time.isoformat()
                if session.start_time
                else None,
                "end_time": session.end_time.isoformat() if session.end_time else None,
                "is_active": session.is_active,
            },
            "total_persons": total_persons,
            "checked_in": checked_in,
            "present": present,
            "late": late,
            "absent": absent,
            "attendance_list": attendance_list,
        }

    def get_person_attendance_history(
        self, person_id: int, limit: int = 30
    ) -> List[Dict]:
        """获取个人考勤历史"""
        attendances = (
            self.db.query(Attendance)
            .filter(Attendance.person_id == person_id)
            .order_by(Attendance.created_at.desc())
            .limit(limit)
            .all()
        )

        history = []
        for att in attendances:
            session = (
                self.db.query(Session).filter(Session.id == att.session_id).first()
            )
            history.append(
                {
                    "session_id": att.session_id,
                    "session_name": session.name if session else "Unknown",
                    "status": att.status,
                    "first_seen": att.first_seen.isoformat()
                    if att.first_seen
                    else None,
                    "date": att.created_at.strftime("%Y-%m-%d")
                    if att.created_at
                    else None,
                }
            )

        return history
