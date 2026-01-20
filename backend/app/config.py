"""
全局配置：路径、模型参数、阈值等
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ============ 路径 ============
    # backend/ 的父目录就是 Face Recognition/
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]  # Face Recognition/
    DATA_DIR: Path = PROJECT_ROOT / "data"
    VIDEO_DIR: Path = PROJECT_ROOT / "VIdeo"
    MODEL_DIR: Path = PROJECT_ROOT / "models"
    DB_PATH: Path = DATA_DIR / "face_attendance.db"
    SEEDS_DIR: Path = DATA_DIR / "seeds"
    DATABASE_URL: str = f"sqlite:///{DATA_DIR / 'face_attendance.db'}"

    # ============ InsightFace 模型 ============
    # 使用 buffalo_l（精度高）或 buffalo_s（速度快）
    INSIGHTFACE_MODEL_NAME: str = "buffalo_l"
    # 执行设备：0=GPU, -1=CPU
    INSIGHTFACE_CTX_ID: int = 0

    # ============ 视频抽帧 ============
    # 每隔 N 帧做一次检测（降低计算量）
    FRAME_SAMPLE_INTERVAL: int = 3
    # 视频缩放比例（1.0=原始尺寸，0.5=减半）
    FRAME_SCALE: float = 1.0

    # ============ 质量门控 ============
    # 人脸框短边最小像素
    MIN_FACE_SIZE: int = 80
    # Laplacian 方差阈值（低于此值认为模糊）
    BLUR_THRESHOLD: float = 100.0
    # 最大允许 yaw/pitch 角度（度）
    MAX_POSE_ANGLE: float = 35.0

    # ============ 跟踪 ============
    # IoU 阈值（用于帧间关联）
    TRACK_IOU_THRESHOLD: float = 0.3
    # track 最大丢失帧数（超过则结束 track）
    TRACK_MAX_AGE: int = 30
    # 每个 track 最多保留的高质量样本数
    TRACK_MAX_SAMPLES: int = 20

    # ============ 识别 ============
    # 余弦相似度阈值（低于此值视为 unknown）
    RECOGNITION_THRESHOLD: float = 0.6
    # 连续确认帧数
    CONFIRM_FRAMES: int = 5

    # ============ 聚类 ============
    # HDBSCAN 最小簇大小
    CLUSTER_MIN_SIZE: int = 5
    # HDBSCAN 最小样本数
    CLUSTER_MIN_SAMPLES: int = 3

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# 确保关键目录存在
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
(settings.DATA_DIR / "faces").mkdir(exist_ok=True)
(settings.DATA_DIR / "gallery").mkdir(exist_ok=True)
