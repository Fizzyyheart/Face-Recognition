"""
视频考勤处理脚本
用法:
    conda activate face-rec
    python -m backend.scripts.process_video_attendance --video "VIdeo/最终.mp4" --name "机器视觉第X周"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Set
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import settings
from backend.app.face.detector import detect_faces, FaceRecognizer
from backend.app.db.database import SessionLocal
from backend.app.db.models import Person


def process_video_for_attendance(
    video_path: str,
    session_name: str,
    output_excel: str = None,
    frame_skip: int = 5,
    threshold: float = None,
    late_threshold_sec: int = 600,  # 10分钟
):
    """
    处理视频并生成考勤表

    Args:
        video_path: 视频文件路径
        session_name: 会话名称
        output_excel: 输出Excel路径（可选，默认自动生成）
        frame_skip: 每隔多少帧进行一次识别
        threshold: 识别阈值
        late_threshold_sec: 迟到阈值（秒）
    """
    print("=" * 60)
    print("视频考勤处理")
    print("=" * 60)

    # 初始化识别器
    print("\n[1/4] 加载人脸识别模型...")
    recognizer = FaceRecognizer(threshold=threshold)
    print(f"  ✓ 已加载 {len(recognizer.persons)} 个已注册人员")

    # 打开视频
    print(f"\n[2/4] 读取视频: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ✗ 错误: 无法打开视频文件")
        return

    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_sec = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  帧数: {total_frames}")
    print(f"  FPS: {fps:.1f}")
    print(f"  时长: {duration_sec / 60:.1f} 分钟")
    print(f"  分辨率: {width}x{height}")

    # 考勤记录
    attendance_records: Dict[
        int, Dict
    ] = {}  # {person_id: {"name", "first_frame", "last_frame", "count"}}
    recognized_persons: Set[str] = set()

    print(f"\n[3/4] 处理视频帧 (每{frame_skip}帧检测一次)...")
    frame_count = 0

    pbar = tqdm(total=total_frames, desc="识别进度", unit="帧")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔 N 帧进行识别
        if frame_count % frame_skip == 0:
            faces = detect_faces(frame, extract_embedding=True)

            for face in faces:
                result = recognizer.recognize_face(face)
                if result:
                    person_id, name, similarity = result
                    recognized_persons.add(name)

                    if person_id not in attendance_records:
                        attendance_records[person_id] = {
                            "name": name,
                            "first_frame": frame_count,
                            "last_frame": frame_count,
                            "count": 1,
                            "max_similarity": similarity,
                        }
                    else:
                        attendance_records[person_id]["last_frame"] = frame_count
                        attendance_records[person_id]["count"] += 1
                        attendance_records[person_id]["max_similarity"] = max(
                            attendance_records[person_id]["max_similarity"], similarity
                        )

        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    print(f"\n  ✓ 识别到 {len(recognized_persons)} 个人员")

    # 获取所有注册人员
    db = SessionLocal()
    all_persons = db.query(Person).all()
    db.close()

    # 生成考勤表
    print(f"\n[4/4] 生成考勤表...")

    attendance_list = []
    for person in all_persons:
        record = attendance_records.get(person.id)

        if record:
            # 计算首次出现时间（视频中的时间）
            first_time_sec = record["first_frame"] / fps if fps > 0 else 0

            # 判断是否迟到
            if first_time_sec > late_threshold_sec:
                status = "迟到"
            else:
                status = "正常"

            attendance_list.append(
                {
                    "姓名": person.name,
                    "学号": person.student_id or "-",
                    "状态": status,
                    "首次出现": f"{first_time_sec / 60:.1f} 分钟",
                    "识别次数": record["count"],
                    "相似度": f"{record['max_similarity']:.2%}",
                }
            )
        else:
            attendance_list.append(
                {
                    "姓名": person.name,
                    "学号": person.student_id or "-",
                    "状态": "缺勤",
                    "首次出现": "-",
                    "识别次数": 0,
                    "相似度": "-",
                }
            )

    # 统计
    total = len(all_persons)
    present_count = sum(1 for a in attendance_list if a["状态"] == "正常")
    late_count = sum(1 for a in attendance_list if a["状态"] == "迟到")
    absent_count = sum(1 for a in attendance_list if a["状态"] == "缺勤")

    print(f"\n{'=' * 60}")
    print(f"考勤统计: {session_name}")
    print(f"{'=' * 60}")
    print(f"  总人数: {total}")
    print(f"  正常:   {present_count} ({present_count / total * 100:.1f}%)")
    print(f"  迟到:   {late_count} ({late_count / total * 100:.1f}%)")
    print(f"  缺勤:   {absent_count} ({absent_count / total * 100:.1f}%)")
    print(f"{'=' * 60}")

    # 输出 Excel
    if output_excel is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_excel = f"考勤表_{session_name}_{timestamp}.xlsx"

    try:
        import pandas as pd

        df = pd.DataFrame(attendance_list)

        # 按状态排序：正常 -> 迟到 -> 缺勤
        status_order = {"正常": 0, "迟到": 1, "缺勤": 2}
        df["排序"] = df["状态"].map(status_order)
        df = df.sort_values("排序").drop(columns=["排序"])

        # 保存Excel
        with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="考勤表", index=False)

            # 添加统计信息sheet
            stats_df = pd.DataFrame(
                [
                    {"项目": "会话名称", "值": session_name},
                    {"项目": "视频文件", "值": Path(video_path).name},
                    {
                        "项目": "处理时间",
                        "值": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    {"项目": "迟到阈值", "值": f"{late_threshold_sec // 60} 分钟"},
                    {"项目": "", "值": ""},
                    {"项目": "总人数", "值": total},
                    {"项目": "正常", "值": present_count},
                    {"项目": "迟到", "值": late_count},
                    {"项目": "缺勤", "值": absent_count},
                ]
            )
            stats_df.to_excel(writer, sheet_name="统计信息", index=False)

        print(f"\n✓ 考勤表已保存: {output_excel}")

    except ImportError:
        print("\n[提示] 需要安装 pandas 和 openpyxl 才能导出Excel")
        print("  pip install pandas openpyxl")

        # 输出CSV作为备选
        output_csv = output_excel.replace(".xlsx", ".csv")
        with open(output_csv, "w", encoding="utf-8-sig") as f:
            f.write("姓名,学号,状态,首次出现,识别次数,相似度\n")
            for row in attendance_list:
                f.write(
                    f"{row['姓名']},{row['学号']},{row['状态']},{row['首次出现']},{row['识别次数']},{row['相似度']}\n"
                )
        print(f"✓ 考勤表已保存 (CSV): {output_csv}")

    # 打印详细名单
    print(f"\n{'=' * 60}")
    print("详细考勤名单")
    print(f"{'=' * 60}")
    print(f"{'姓名':<10} {'学号':<12} {'状态':<6} {'首次出现':<12} {'识别次数'}")
    print("-" * 60)
    for row in attendance_list:
        status_icon = (
            "✓" if row["状态"] == "正常" else ("⏰" if row["状态"] == "迟到" else "✗")
        )
        print(
            f"{row['姓名']:<10} {row['学号']:<12} {status_icon} {row['状态']:<4} {row['首次出现']:<12} {row['识别次数']}"
        )

    return attendance_list


def main():
    parser = argparse.ArgumentParser(description="视频考勤处理")
    parser.add_argument("--video", type=str, required=True, help="视频文件路径")
    parser.add_argument("--name", type=str, default=None, help="会话名称")
    parser.add_argument("--output", type=str, default=None, help="输出Excel路径")
    parser.add_argument("--frame-skip", type=int, default=5, help="帧跳过数（默认5）")
    parser.add_argument("--threshold", type=float, default=None, help="识别阈值")
    parser.add_argument("--late", type=int, default=10, help="迟到阈值（分钟，默认10）")

    args = parser.parse_args()

    # 默认会话名称
    if args.name is None:
        args.name = f"考勤_{datetime.now().strftime('%Y%m%d')}"

    process_video_for_attendance(
        video_path=args.video,
        session_name=args.name,
        output_excel=args.output,
        frame_skip=args.frame_skip,
        threshold=args.threshold,
        late_threshold_sec=args.late * 60,
    )


if __name__ == "__main__":
    main()
