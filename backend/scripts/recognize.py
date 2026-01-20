"""
人脸识别脚本：支持实时摄像头和上传图片/视频识别
用法：
    conda activate face-rec
    # 实时摄像头识别
    python -m backend.scripts.recognize --camera
    # 识别图片
    python -m backend.scripts.recognize --image path/to/image.jpg
    # 识别视频
    python -m backend.scripts.recognize --video path/to/video.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# 中文字体路径 (Windows)
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑
try:
    CHINESE_FONT = ImageFont.truetype(FONT_PATH, 24)
except Exception:
    CHINESE_FONT = None
    print("警告: 无法加载中文字体，将使用默认字体")

from backend.app.config import settings
from backend.app.face.detector import detect_faces, FaceRecognizer


def put_chinese_text(
    img: np.ndarray, text: str, position: tuple, color: tuple = (255, 255, 255)
):
    """在图像上绘制中文文本（优化版：只转换小区域）"""
    if CHINESE_FONT is None:
        # 回退到 OpenCV（不支持中文）
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img

    x, y = position
    h, w = img.shape[:2]

    # 估算文本尺寸
    text_w = len(text) * 15 + 10
    text_h = 30

    # 确保区域在图像范围内
    x1 = max(0, x - 5)
    y1 = max(0, y - 5)
    x2 = min(w, x + text_w + 5)
    y2 = min(h, y + text_h + 5)

    if x2 <= x1 or y2 <= y1:
        return img

    # 只转换需要绘制文字的小区域
    roi = img[y1:y2, x1:x2].copy()
    roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(roi_pil)

    # BGR -> RGB
    color_rgb = (color[2], color[1], color[0])
    draw.text((x - x1, y - y1), text, font=CHINESE_FONT, fill=color_rgb)

    # 转换回 OpenCV 格式并写回原图
    roi_result = cv2.cvtColor(np.array(roi_pil), cv2.COLOR_RGB2BGR)
    img[y1:y2, x1:x2] = roi_result
    return img


def draw_face_box(
    frame: np.ndarray,
    bbox: np.ndarray,
    name: str,
    similarity: float,
    color: tuple = (0, 255, 0),
):
    """在帧上绘制人脸框和标签"""
    x1, y1, x2, y2 = map(int, bbox)

    # 绘制人脸框
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # 准备标签文本
    if name:
        label = f"{name} ({similarity:.2f})"
    else:
        label = "Unknown"
        color = (0, 0, 255)  # 红色表示未知

    # 绘制标签背景
    text_height = 30
    text_width = len(label) * 15 + 20  # 估算宽度

    cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
    put_chinese_text(frame, label, (x1 + 5, y1 - text_height), (255, 255, 255))


def recognize_camera(
    camera_id: int = 0,
    recognizer: FaceRecognizer = None,
    frame_skip: int = 2,
    show_fps: bool = True,
    resolution: str = "480p",
):
    """
    实时摄像头人脸识别

    Args:
        camera_id: 摄像头ID
        recognizer: 识别器实例
        frame_skip: 每隔多少帧进行一次识别
        show_fps: 是否显示FPS
        resolution: 分辨率 "480p"(30fps) 或 "720p"(15fps)
    """
    if recognizer is None:
        recognizer = FaceRecognizer()

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头 {camera_id}")
        return

    # 设置摄像头分辨率
    if resolution == "720p":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:  # 默认 480p，帧率更高
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=" * 50)
    print("实时人脸识别")
    print(f"分辨率: {actual_w}x{actual_h} @ {actual_fps}fps")
    print("按 'q' 退出, 按 's' 截图")
    print("=" * 50)

    frame_count = 0
    fps_start_time = time.time()
    fps = 0

    # 缓存最近的识别结果（用于平滑显示）
    cached_results: Dict[
        int, tuple
    ] = {}  # {track_id: (bbox, name, similarity, last_seen)}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = time.time()

        # 每隔 N 帧进行识别
        if frame_count % frame_skip == 0:
            # 检测人脸
            faces = detect_faces(frame, extract_embedding=True)

            # 识别每张人脸
            new_results = {}
            for i, face in enumerate(faces):
                result = recognizer.recognize_face(face)
                if result:
                    pid, name, sim = result
                    new_results[i] = (face.bbox, name, sim, current_time)
                else:
                    new_results[i] = (face.bbox, None, 0.0, current_time)

            cached_results = new_results

        # 绘制结果
        for track_id, (bbox, name, sim, last_seen) in cached_results.items():
            # 只显示最近 0.5 秒内的结果
            if current_time - last_seen < 0.5:
                if name:
                    draw_face_box(frame, bbox, name, sim, (0, 255, 0))
                else:
                    draw_face_box(frame, bbox, "Unknown", 0, (0, 0, 255))

        # 计算 FPS
        if show_fps:
            elapsed = current_time - fps_start_time
            if elapsed > 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = current_time

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Persons: {len(recognizer.persons)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

        # 显示画面
        cv2.imshow("Face Recognition", frame)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            # 截图
            screenshot_path = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(screenshot_path, frame)
            print(f"截图已保存: {screenshot_path}")

    cap.release()
    cv2.destroyAllWindows()


def recognize_image(
    image_path: str,
    recognizer: FaceRecognizer = None,
    output_path: str = None,
):
    """
    识别图片中的人脸

    Args:
        image_path: 图片路径
        recognizer: 识别器实例
        output_path: 输出图片路径（可选）
    """
    if recognizer is None:
        recognizer = FaceRecognizer()

    # 读取图片（支持中文路径）
    img_array = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if image is None:
        print(f"错误: 无法读取图片 {image_path}")
        return

    print(f"识别图片: {image_path}")

    # 检测人脸
    faces = detect_faces(image, extract_embedding=True)
    print(f"检测到 {len(faces)} 张人脸")

    # 识别每张人脸
    results = []
    for face in faces:
        result = recognizer.recognize_face(face)
        if result:
            pid, name, sim = result
            results.append((face, name, sim))
            draw_face_box(image, face.bbox, name, sim, (0, 255, 0))
            print(f"  ✓ {name} (相似度: {sim:.3f})")
        else:
            results.append((face, None, 0.0))
            draw_face_box(image, face.bbox, "Unknown", 0, (0, 0, 255))
            print(f"  ✗ Unknown")

    # 保存或显示结果
    if output_path:
        cv2.imencode(".jpg", image)[1].tofile(output_path)
        print(f"结果已保存: {output_path}")
    else:
        # 显示图片
        cv2.imshow("Recognition Result", image)
        print("按任意键关闭...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return results


def recognize_video(
    video_path: str,
    recognizer: FaceRecognizer = None,
    output_path: str = None,
    frame_skip: int = 3,
):
    """
    识别视频中的人脸

    Args:
        video_path: 视频路径
        recognizer: 识别器实例
        output_path: 输出视频路径（可选）
        frame_skip: 每隔多少帧进行一次识别
    """
    if recognizer is None:
        recognizer = FaceRecognizer()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return

    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"视频: {video_path}")
    print(f"帧数: {total_frames}, FPS: {fps:.1f}, 分辨率: {width}x{height}")

    # 输出视频写入器
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    cached_results = {}
    recognized_persons = set()

    from tqdm import tqdm

    pbar = tqdm(total=total_frames, desc="识别视频")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔 N 帧进行识别
        if frame_count % frame_skip == 0:
            faces = detect_faces(frame, extract_embedding=True)

            cached_results = {}
            for i, face in enumerate(faces):
                result = recognizer.recognize_face(face)
                if result:
                    pid, name, sim = result
                    cached_results[i] = (face.bbox, name, sim)
                    recognized_persons.add(name)
                else:
                    cached_results[i] = (face.bbox, None, 0.0)

        # 绘制结果
        for track_id, (bbox, name, sim) in cached_results.items():
            if name:
                draw_face_box(frame, bbox, name, sim, (0, 255, 0))
            else:
                draw_face_box(frame, bbox, "Unknown", 0, (0, 0, 255))

        # 写入或显示
        if writer:
            writer.write(frame)
        else:
            cv2.imshow("Video Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    if writer:
        writer.release()
        print(f"结果已保存: {output_path}")
    else:
        cv2.destroyAllWindows()

    print(f"\n识别到的人员: {', '.join(sorted(recognized_persons))}")


def main():
    parser = argparse.ArgumentParser(description="人脸识别")
    parser.add_argument("--camera", action="store_true", help="使用摄像头实时识别")
    parser.add_argument("--camera-id", type=int, default=0, help="摄像头ID")
    parser.add_argument(
        "--resolution",
        type=str,
        default="480p",
        choices=["480p", "720p"],
        help="分辨率: 480p(30fps) 或 720p(15fps)",
    )
    parser.add_argument("--image", type=str, help="识别图片")
    parser.add_argument("--video", type=str, help="识别视频")
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument("--threshold", type=float, default=None, help="识别阈值")
    parser.add_argument("--frame-skip", type=int, default=2, help="帧跳过数")

    args = parser.parse_args()

    # 初始化识别器
    print("正在加载识别器...")
    recognizer = FaceRecognizer(threshold=args.threshold)

    if args.camera:
        recognize_camera(
            camera_id=args.camera_id,
            recognizer=recognizer,
            frame_skip=args.frame_skip,
            resolution=args.resolution,
        )
    elif args.image:
        recognize_image(
            image_path=args.image,
            recognizer=recognizer,
            output_path=args.output,
        )
    elif args.video:
        recognize_video(
            video_path=args.video,
            recognizer=recognizer,
            output_path=args.output,
            frame_skip=args.frame_skip,
        )
    else:
        parser.print_help()
        print("\n请指定 --camera, --image 或 --video")


if __name__ == "__main__":
    main()
