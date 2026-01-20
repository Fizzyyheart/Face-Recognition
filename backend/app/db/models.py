"""
SQLite 数据库模型定义
"""

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    ForeignKey,
    LargeBinary,
    Boolean,
    Enum,
)
from sqlalchemy.orm import declarative_base, relationship
import enum

Base = declarative_base()


class ClusterStatus(enum.Enum):
    """簇状态"""

    PENDING = "pending"  # 待命名
    LABELED = "labeled"  # 已命名
    MERGED = "merged"  # 已合并到其他簇
    NOISE = "noise"  # 噪声（丢弃）


class AttendanceStatus(enum.Enum):
    """考勤状态"""

    PRESENT = "present"  # 正常
    LATE = "late"  # 迟到
    EARLY_LEAVE = "early"  # 早退
    ABSENT = "absent"  # 缺勤


class Person(Base):
    """人员表"""

    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(50), unique=True, nullable=True, index=True)  # 学号
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    # 关系
    face_samples = relationship("FaceSample", back_populates="person")
    attendances = relationship("Attendance", back_populates="person")


class Cluster(Base):
    """聚类簇表"""

    __tablename__ = "clusters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    status = Column(String(20), default=ClusterStatus.PENDING.value)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=True)  # 命名后关联
    preview_image_path = Column(String(500), nullable=True)  # 簇预览图路径
    sample_count = Column(Integer, default=0)
    center_embedding = Column(LargeBinary, nullable=True)  # 簇中心向量
    created_at = Column(DateTime, default=datetime.now)

    # 关系
    face_samples = relationship("FaceSample", back_populates="cluster")
    person = relationship("Person")


class FaceSample(Base):
    """人脸样本表（从视频抽取）"""

    __tablename__ = "face_samples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=True)
    cluster_id = Column(Integer, ForeignKey("clusters.id"), nullable=True)

    # 来源信息
    source_video = Column(String(500), nullable=True)  # 来源视频路径
    frame_index = Column(Integer, nullable=True)  # 帧索引
    track_id = Column(Integer, nullable=True)  # 跟踪 ID

    # 人脸信息
    image_path = Column(String(500), nullable=True)  # 裁剪人脸图路径
    bbox_json = Column(Text, nullable=True)  # bbox JSON
    embedding = Column(LargeBinary, nullable=False)  # 512-d 特征向量

    # 质量信息
    quality_score = Column(Float, default=0.0)
    blur_score = Column(Float, default=0.0)
    det_score = Column(Float, default=0.0)

    created_at = Column(DateTime, default=datetime.now)

    # 关系
    person = relationship("Person", back_populates="face_samples")
    cluster = relationship("Cluster", back_populates="face_samples")


class Session(Base):
    """考勤会话表（一节课 = 一个 session）"""

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=True)  # 课程名/会话名
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    late_after_sec = Column(Integer, default=600)  # 迟到阈值（秒）
    is_active = Column(Boolean, default=False)  # 是否正在进行
    created_at = Column(DateTime, default=datetime.now)

    # 关系
    attendances = relationship("Attendance", back_populates="session")
    recognition_events = relationship("RecognitionEvent", back_populates="session")


class Attendance(Base):
    """考勤记录表"""

    __tablename__ = "attendances"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=False)

    first_seen = Column(DateTime, nullable=True)  # 首次识别时间
    last_seen = Column(DateTime, nullable=True)  # 最后识别时间
    status = Column(String(20), default=AttendanceStatus.PRESENT.value)
    best_score = Column(Float, default=0.0)  # 最高相似度
    seen_count = Column(Integer, default=0)  # 识别次数

    created_at = Column(DateTime, default=datetime.now)

    # 关系
    session = relationship("Session", back_populates="attendances")
    person = relationship("Person", back_populates="attendances")


class RecognitionEvent(Base):
    """识别事件表（用于调试/审计，可选）"""

    __tablename__ = "recognition_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    track_id = Column(Integer, nullable=True)

    person_id = Column(
        Integer, ForeignKey("persons.id"), nullable=True
    )  # 识别结果，None=unknown
    similarity_score = Column(Float, default=0.0)
    bbox_json = Column(Text, nullable=True)
    image_path = Column(String(500), nullable=True)  # 截图路径（可选）

    created_at = Column(DateTime, default=datetime.now)

    # 关系
    session = relationship("Session", back_populates="recognition_events")
